"""DEM 读取、栅格穿越和航段最高高程提取。"""

import math
from pathlib import Path

from PIL import Image


class DemGrid:
    """读取带地理标签的单波段 GeoTIFF。"""

    def __init__(self, path: Path) -> None:
        self.image = Image.open(path)
        self.pixels = self.image.load()
        self.width, self.height = self.image.size
        scale = self.image.tag_v2.get(33550)
        tie = self.image.tag_v2.get(33922)
        if not scale or not tie:
            raise ValueError("DEM 缺少 ModelPixelScale 或 ModelTiepoint 标签")
        self.dx, self.dy = float(scale[0]), float(scale[1])
        self.lon0, self.lat0 = float(tie[3]), float(tie[4])

    def fractional_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        """把经纬度转换为浮点像元列、行坐标。"""

        return (lon - self.lon0) / self.dx, (self.lat0 - lat) / self.dy

    def contains(self, lon: float, lat: float) -> bool:
        """判断一个经纬度点是否落在 DEM 栅格范围内。"""

        col, row = self.fractional_pixel(lon, lat)
        return 0 <= col < self.width and 0 <= row < self.height

    def elevation_at(self, lon: float, lat: float) -> float:
        """读取最接近目标点的有效 DEM 像元高程。"""

        if not self.contains(lon, lat):
            raise ValueError("节点坐标超出 DEM 范围")
        col, row = self.fractional_pixel(lon, lat)
        # 边界内但接近最外层中心时，round 可能超过最大索引，故在此夹紧。
        column_index = min(self.width - 1, max(0, round(col)))
        row_index = min(self.height - 1, max(0, round(row)))
        value = float(self.pixels[column_index, row_index])
        if value <= -30_000:
            raise ValueError("节点对应 DEM 像元为 NoData")
        return value

    def audit_metadata(self) -> dict[str, float | int]:
        """统计 DEM 范围、高程范围和 NoData 数，用于预处理审计。"""

        valid: list[float] = []
        nodata_count = 0
        for value in self.image.getdata():
            height = float(value)
            if height <= -30_000:
                nodata_count += 1
            else:
                valid.append(height)
        if not valid:
            raise ValueError("DEM 不含有效高程像元")
        return {
            "width": self.width,
            "height": self.height,
            "resolution_lon_deg": self.dx,
            "resolution_lat_deg": self.dy,
            "lon_min": self.lon0,
            "lon_max": self.lon0 + self.width * self.dx,
            "lat_min": self.lat0 - self.height * self.dy,
            "lat_max": self.lat0,
            "elevation_min_m": min(valid),
            "elevation_max_m": max(valid),
            "nodata_cell_count": nodata_count,
        }

    def traversed_cells(self, lon1: float, lat1: float, lon2: float, lat2: float) -> list[tuple[int, int]]:
        """使用栅格穿越算法提取线段经过的全部像元。"""

        x0, y0 = self.fractional_pixel(lon1, lat1)
        x1, y1 = self.fractional_pixel(lon2, lat2)
        col, row = math.floor(x0), math.floor(y0)
        end_col, end_row = math.floor(x1), math.floor(y1)
        if not (0 <= col < self.width and 0 <= row < self.height):
            raise ValueError("航段起点超出 DEM 范围")
        if not (0 <= end_col < self.width and 0 <= end_row < self.height):
            raise ValueError("航段终点超出 DEM 范围")

        dx, dy = x1 - x0, y1 - y0
        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
        delta_x = abs(1 / dx) if dx else math.inf
        delta_y = abs(1 / dy) if dy else math.inf
        max_x = ((col + 1 if step_x > 0 else col) - x0) / dx if dx else math.inf
        max_y = ((row + 1 if step_y > 0 else row) - y0) / dy if dy else math.inf
        cells: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        def add(r: int, c: int) -> None:
            if 0 <= r < self.height and 0 <= c < self.width and (r, c) not in seen:
                cells.append((r, c))
                seen.add((r, c))

        add(row, col)
        while col != end_col or row != end_row:
            if abs(max_x - max_y) <= 1e-12:
                add(row, col + step_x)
                add(row + step_y, col)
                col, row = col + step_x, row + step_y
                max_x, max_y = max_x + delta_x, max_y + delta_y
            elif max_x < max_y:
                col, max_x = col + step_x, max_x + delta_x
            else:
                row, max_y = row + step_y, max_y + delta_y
            add(row, col)
        return cells

    def elevations(self, cells: list[tuple[int, int]]) -> list[float]:
        """返回有效像元高程，并排除约定的极小 NoData 值。"""

        values = [float(self.pixels[col, row]) for row, col in cells]
        valid = [value for value in values if value > -30_000]
        if not valid:
            raise ValueError("航段经过的 DEM 像元全部为 NoData")
        return valid

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        """返回 DEM 像元中心的经度、纬度，供缓存和地图复用。"""

        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError("DEM 像元索引超出范围")
        return self.lon0 + (col + 0.5) * self.dx, self.lat0 - (row + 0.5) * self.dy

    def terrain_profile(self, cells: list[tuple[int, int]]) -> list[tuple[int, int, float, float, float]]:
        """将航段穿越格网转为有序的 ``行、列、中心经纬度、高程`` 地形剖面。"""

        profile: list[tuple[int, int, float, float, float]] = []
        for row, col in cells:
            height = float(self.pixels[col, row])
            if height <= -30_000:
                raise ValueError("航段地形剖面包含 NoData 像元")
            lon, lat = self.cell_center(row, col)
            profile.append((row, col, lon, lat, height))
        return profile
