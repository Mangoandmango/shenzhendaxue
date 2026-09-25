"""DEM 读取、栅格穿越和航段最高高程提取。"""

from dataclasses import dataclass
import math
from pathlib import Path

from PIL import Image

from uav_rescue.geo.coordinates import LocalEnu


@dataclass(frozen=True)
class DemIntersection:
    """一条 ENU 平面线段与一个 RasterPixelIsPoint 半像元区域的交集。"""

    row: int
    col: int
    elevation_m: float
    s_in_m: float
    s_out_m: float
    intersection_length_m: float
    contact_type: str

    @property
    def has_positive_length(self) -> bool:
        return self.intersection_length_m > 1e-7


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
        geo_keys = self.image.tag_v2.get(34735)
        raster_type = None
        if geo_keys and len(geo_keys) >= 4:
            for offset in range(4, len(geo_keys), 4):
                if int(geo_keys[offset]) == 1025:
                    raster_type = int(geo_keys[offset + 3])
                    break
        if raster_type != 2:
            raise ValueError("DEM 必须声明 RasterPixelIsPoint，才能使用半像元中心边界")

    def fractional_pixel_center(self, lon: float, lat: float) -> tuple[float, float]:
        """把经纬度转换为相对第 0 像元中心的浮点列、行坐标。"""

        return (lon - self.lon0) / self.dx, (self.lat0 - lat) / self.dy

    def contains(self, lon: float, lat: float) -> bool:
        """判断一个经纬度点是否落在 DEM 栅格范围内。"""

        col, row = self.fractional_pixel_center(lon, lat)
        return -0.5 <= col < self.width - 0.5 and -0.5 <= row < self.height - 0.5

    def elevation_at(self, lon: float, lat: float) -> float:
        """读取最接近目标点的有效 DEM 像元高程。"""

        if not self.contains(lon, lat):
            raise ValueError("节点坐标超出 DEM 范围")
        col, row = self.fractional_pixel_center(lon, lat)
        column_index = math.floor(col + 0.5)
        row_index = math.floor(row + 0.5)
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

    @staticmethod
    def _touching_indices(value: float, tolerance: float = 1e-8) -> tuple[int, ...]:
        boundary_index = round(value - 0.5)
        if abs(value - (boundary_index + 0.5)) <= tolerance:
            return boundary_index, boundary_index + 1
        return (math.floor(value + 0.5),)

    @staticmethod
    def _boundaries_between(first: float, second: float) -> list[float]:
        lower, upper = sorted((first, second))
        start = math.ceil(lower - 0.5 - 1e-12)
        end = math.floor(upper - 0.5 + 1e-12)
        return [index + 0.5 for index in range(start, end + 1)]

    def _cell_elevation(self, row: int, col: int) -> float:
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError("ENU 航段穿越像元超出 DEM 范围")
        elevation = float(self.pixels[col, row])
        if elevation <= -30_000:
            raise ValueError("ENU 航段穿越 NoData 像元")
        return elevation

    def trace_enu_segment(
        self,
        enu: LocalEnu,
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
        boundary_search_step_m: float = 5.0,
    ) -> tuple[float, tuple[DemIntersection, ...]]:
        """沿同一条 ENU 直线求半像元边界交点、真实路径区间和接触类型。

        反算到经纬度后的轨迹在栅格坐标中并非严格直线，因此先用短 ENU
        区间包围边界，再对真实映射函数二分求根，而不是连接两个栅格端点。
        """

        if boundary_search_step_m <= 0:
            raise ValueError("边界搜索步长必须为正")
        east1, north1, _ = enu.from_geodetic_m(lon1, lat1, 0.0)
        east2, north2, _ = enu.from_geodetic_m(lon2, lat2, 0.0)
        distance = math.hypot(east2 - east1, north2 - north1)

        def grid_at(s_m: float) -> tuple[float, float]:
            ratio = 0.0 if distance <= 0 else min(1.0, max(0.0, s_m / distance))
            lon, lat, _ = enu.to_geodetic_m(
                east1 + (east2 - east1) * ratio,
                north1 + (north2 - north1) * ratio,
            )
            return self.fractional_pixel_center(lon, lat)

        if distance <= 1e-9:
            col_f, row_f = grid_at(0.0)
            col = math.floor(col_f + 0.5)
            row = math.floor(row_f + 0.5)
            record = DemIntersection(row, col, self._cell_elevation(row, col), 0.0, 0.0, 0.0, "endpoint")
            return 0.0, (record,)

        coarse_steps = max(1, math.ceil(distance / boundary_search_step_m))
        coarse_s = [distance * index / coarse_steps for index in range(coarse_steps + 1)]
        coarse_grid = [grid_at(value) for value in coarse_s]
        breakpoints = [0.0, distance]

        def solve_root(left_s: float, right_s: float, axis: int, boundary: float) -> float:
            left_value = grid_at(left_s)[axis] - boundary
            right_value = grid_at(right_s)[axis] - boundary
            if abs(left_value) <= 1e-12:
                return left_s
            if abs(right_value) <= 1e-12:
                return right_s
            for _ in range(45):
                middle_s = (left_s + right_s) / 2.0
                middle_value = grid_at(middle_s)[axis] - boundary
                if abs(middle_value) <= 1e-12 or right_s - left_s <= 1e-7:
                    return middle_s
                if (left_value < 0) == (middle_value < 0):
                    left_s, left_value = middle_s, middle_value
                else:
                    right_s, right_value = middle_s, middle_value
            return (left_s + right_s) / 2.0

        for index in range(coarse_steps):
            left_s, right_s = coarse_s[index], coarse_s[index + 1]
            for axis in (0, 1):
                first, second = coarse_grid[index][axis], coarse_grid[index + 1][axis]
                for boundary in self._boundaries_between(first, second):
                    if abs(first - boundary) <= 1e-12 and abs(second - boundary) <= 1e-12:
                        # 轨迹沿边界延伸，由区间中点分类为 boundary；不人为按搜索步长切碎。
                        continue
                    if (first - boundary) * (second - boundary) <= 1e-15:
                        breakpoints.append(solve_root(left_s, right_s, axis, boundary))

        breakpoints.sort()
        unique_breakpoints: list[float] = []
        for value in breakpoints:
            if not unique_breakpoints or value - unique_breakpoints[-1] > 1e-6:
                unique_breakpoints.append(value)
            else:
                unique_breakpoints[-1] = (unique_breakpoints[-1] + value) / 2.0

        records: list[DemIntersection] = []
        for s_in, s_out in zip(unique_breakpoints, unique_breakpoints[1:]):
            if s_out - s_in <= 1e-7:
                continue
            col_f, row_f = grid_at((s_in + s_out) / 2.0)
            cols = self._touching_indices(col_f)
            rows = self._touching_indices(row_f)
            contact_type = "interior" if len(cols) == 1 and len(rows) == 1 else "boundary"
            for row in rows:
                for col in cols:
                    records.append(DemIntersection(
                        row, col, self._cell_elevation(row, col), s_in, s_out,
                        s_out - s_in, contact_type,
                    ))

        positive_cells = {
            (record.row, record.col)
            for record in records
            if record.has_positive_length
        }
        zero_contacts: set[tuple[int, int, int, str]] = set()
        for s_m in unique_breakpoints:
            col_f, row_f = grid_at(s_m)
            cols = self._touching_indices(col_f)
            rows = self._touching_indices(row_f)
            if s_m <= 1e-6 or distance - s_m <= 1e-6:
                contact_type = "endpoint"
            elif len(cols) == 2 and len(rows) == 2:
                contact_type = "corner"
            else:
                contact_type = "boundary"
            for row in rows:
                for col in cols:
                    if not (0 <= row < self.height and 0 <= col < self.width):
                        continue
                    if (row, col) in positive_cells:
                        continue
                    key = (row, col, round(s_m * 1_000_000), contact_type)
                    if key in zero_contacts:
                        continue
                    zero_contacts.add(key)
                    records.append(DemIntersection(
                        row, col, self._cell_elevation(row, col), s_m, s_m, 0.0, contact_type,
                    ))

        priority = {"interior": 0, "boundary": 1, "corner": 2, "endpoint": 3}
        records.sort(key=lambda item: (item.s_in_m, item.s_out_m, priority[item.contact_type], item.row, item.col))
        return distance, tuple(records)

    def elevations(self, cells: list[tuple[int, int]]) -> list[float]:
        """返回有效像元高程，并排除约定的极小 NoData 值。"""

        values = [float(self.pixels[col, row]) for row, col in cells]
        valid = [value for value in values if value > -30_000]
        if not valid:
            raise ValueError("航段经过的 DEM 像元全部为 NoData")
        return valid

    @staticmethod
    def positive_intersections(records: tuple[DemIntersection, ...]) -> tuple[DemIntersection, ...]:
        """正式穿越记录：内部或沿边界具有非零 ENU 路径长度的像元。"""

        return tuple(record for record in records if record.has_positive_length)

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        """返回 DEM 像元中心的经度、纬度，供缓存和地图复用。"""

        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError("DEM 像元索引超出范围")
        return self.lon0 + col * self.dx, self.lat0 - row * self.dy

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
