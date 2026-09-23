"""WGS84 经纬度计算与本地 ENU 米制坐标转换。"""

import math

from uav_rescue.domain import Node


# WGS84 椭球参数；采用 ENU 而非随意投影，避免改变原始 DEM 的经纬度配准。
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


def haversine_m(a: Node, b: Node, earth_radius_m: float = 6_371_008.8) -> float:
    """计算两个节点间的大圆水平距离，单位为 m。"""

    phi1, phi2 = math.radians(a.lat), math.radians(b.lat)
    dphi = phi2 - phi1
    dlambda = math.radians(b.lon - a.lon)
    value = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * earth_radius_m * math.asin(math.sqrt(value))
