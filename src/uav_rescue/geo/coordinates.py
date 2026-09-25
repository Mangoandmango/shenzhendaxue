"""以 O01 为原点的 WGS84 局部 ENU 航段几何。"""

import math

# WGS84 椭球参数；所有影响模型的水平路径均在此局部 ENU 平面定义。
WGS84_SEMI_MAJOR_M = 6_378_137.0
WGS84_FLATTENING = 1 / 298.257_223_563
WGS84_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2 - WGS84_FLATTENING)


def geodetic_to_ecef_m(lon_deg: float, lat_deg: float, height_m: float) -> tuple[float, float, float]:
    """把 WGS84 经度、纬度和椭球近似高转换为 ECEF 三维直角坐标，单位 m。"""

    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    radius = WGS84_SEMI_MAJOR_M / math.sqrt(1 - WGS84_ECCENTRICITY_SQUARED * math.sin(lat) ** 2)
    x = (radius + height_m) * math.cos(lat) * math.cos(lon)
    y = (radius + height_m) * math.cos(lat) * math.sin(lon)
    z = (radius * (1 - WGS84_ECCENTRICITY_SQUARED) + height_m) * math.sin(lat)
    return x, y, z


def ecef_to_geodetic_m(x: float, y: float, z: float) -> tuple[float, float, float]:
    """把 WGS84 ECEF 坐标反算为经度、纬度和椭球高。"""

    lon = math.atan2(y, x)
    horizontal = math.hypot(x, y)
    lat = math.atan2(z, horizontal * (1 - WGS84_ECCENTRICITY_SQUARED))
    height = 0.0
    for _ in range(10):
        radius = WGS84_SEMI_MAJOR_M / math.sqrt(1 - WGS84_ECCENTRICITY_SQUARED * math.sin(lat) ** 2)
        height = horizontal / math.cos(lat) - radius
        lat = math.atan2(z, horizontal * (1 - WGS84_ECCENTRICITY_SQUARED * radius / (radius + height)))
    return math.degrees(lon), math.degrees(lat), height


class LocalEnu:
    """以指定原点建立东—北—天（ENU）局部米制坐标系。"""

    def __init__(self, origin_lon_deg: float, origin_lat_deg: float, origin_height_m: float) -> None:
        self.origin_lon_deg = origin_lon_deg
        self.origin_lat_deg = origin_lat_deg
        self.origin_height_m = origin_height_m
        self._origin_ecef = geodetic_to_ecef_m(origin_lon_deg, origin_lat_deg, origin_height_m)
        self._lon = math.radians(origin_lon_deg)
        self._lat = math.radians(origin_lat_deg)

    def from_geodetic_m(self, lon_deg: float, lat_deg: float, height_m: float) -> tuple[float, float, float]:
        """返回相对原点的 east、north、up，单位 m。"""

        x, y, z = geodetic_to_ecef_m(lon_deg, lat_deg, height_m)
        dx, dy, dz = (x - self._origin_ecef[0], y - self._origin_ecef[1], z - self._origin_ecef[2])
        east = -math.sin(self._lon) * dx + math.cos(self._lon) * dy
        north = (
            -math.sin(self._lat) * math.cos(self._lon) * dx
            - math.sin(self._lat) * math.sin(self._lon) * dy
            + math.cos(self._lat) * dz
        )
        up = (
            math.cos(self._lat) * math.cos(self._lon) * dx
            + math.cos(self._lat) * math.sin(self._lon) * dy
            + math.sin(self._lat) * dz
        )
        return east, north, up

    def to_geodetic_m(self, east_m: float, north_m: float, up_m: float = 0.0) -> tuple[float, float, float]:
        """将 ENU 坐标反算到 WGS84；用于把同一条 ENU 直线映射回 DEM。"""

        dx = (-math.sin(self._lon) * east_m - math.sin(self._lat) * math.cos(self._lon) * north_m
              + math.cos(self._lat) * math.cos(self._lon) * up_m)
        dy = (math.cos(self._lon) * east_m - math.sin(self._lat) * math.sin(self._lon) * north_m
              + math.cos(self._lat) * math.sin(self._lon) * up_m)
        dz = math.cos(self._lat) * north_m + math.sin(self._lat) * up_m
        return ecef_to_geodetic_m(self._origin_ecef[0] + dx, self._origin_ecef[1] + dy, self._origin_ecef[2] + dz)

    def horizontal_distance_m(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """唯一的水平距离口径：两点在本地 ENU 平面上的欧氏距离。"""

        east1, north1, _ = self.from_geodetic_m(lon1, lat1, 0.0)
        east2, north2, _ = self.from_geodetic_m(lon2, lat2, 0.0)
        return math.hypot(east2 - east1, north2 - north1)

    def straight_path_lonlat(self, lon1: float, lat1: float, lon2: float, lat2: float,
                             max_step_m: float = 0.25) -> tuple[float, list[tuple[float, float]]]:
        """在 ENU 平面定义直线，并返回长度及反算至经纬度的密集 DEM 采样点。"""

        if max_step_m <= 0:
            raise ValueError("DEM 采样步长必须为正")
        east1, north1, _ = self.from_geodetic_m(lon1, lat1, 0.0)
        east2, north2, _ = self.from_geodetic_m(lon2, lat2, 0.0)
        distance = math.hypot(east2 - east1, north2 - north1)
        steps = max(1, math.ceil(distance / max_step_m))
        points = [self.to_geodetic_m(east1 + (east2 - east1) * index / steps,
                                     north1 + (north2 - north1) * index / steps)[:2]
                  for index in range(steps + 1)]
        return distance, points

    def interpolate_lonlat(self, lon1: float, lat1: float, lon2: float, lat2: float,
                           fraction: float) -> tuple[float, float]:
        """返回 ENU 直线上指定比例位置，而非经纬度空间的线性插值。"""

        ratio = min(1.0, max(0.0, fraction))
        east1, north1, _ = self.from_geodetic_m(lon1, lat1, 0.0)
        east2, north2, _ = self.from_geodetic_m(lon2, lat2, 0.0)
        lon, lat, _ = self.to_geodetic_m(east1 + (east2 - east1) * ratio,
                                         north1 + (north2 - north1) * ratio)
        return lon, lat
