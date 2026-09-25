"""问题三可复用的三维视线遮挡判断。"""

from dataclasses import dataclass
import math

from uav_rescue.geo.dem import DemGrid, DemIntersection
from uav_rescue.geo.coordinates import LocalEnu


@dataclass(frozen=True)
class LineOfSightResult:
    clear: bool
    minimum_clearance_m: float
    obstructing_cells: tuple[tuple[int, int], ...]
    boundary_contact_would_obstruct: bool
    corner_contact_would_obstruct: bool


def evaluate_line_of_sight(
    dem: DemGrid,
    enu: LocalEnu,
    lon1: float,
    lat1: float,
    altitude1_m: float,
    lon2: float,
    lat2: float,
    altitude2_m: float,
    exclude_endpoint_cells: bool = True,
    numerical_tolerance_m: float = 1e-7,
) -> LineOfSightResult:
    """按真实 ENU 路径区间比较三维视线与分片常数 DEM 地形。"""

    distance, records = dem.trace_enu_segment(enu, lon1, lat1, lon2, lat2)
    start_col_f, start_row_f = dem.fractional_pixel_center(lon1, lat1)
    end_col_f, end_row_f = dem.fractional_pixel_center(lon2, lat2)
    endpoint_cells = {
        (math.floor(start_row_f + 0.5), math.floor(start_col_f + 0.5)),
        (math.floor(end_row_f + 0.5), math.floor(end_col_f + 0.5)),
    }

    def ray_altitude(s_m: float) -> float:
        ratio = 0.0 if distance <= 0 else s_m / distance
        return altitude1_m + ratio * (altitude2_m - altitude1_m)

    minimum_clearance = float("inf")
    obstructing: list[tuple[int, int]] = []
    for record in dem.positive_intersections(records):
        if exclude_endpoint_cells and (record.row, record.col) in endpoint_cells:
            continue
        lower_ray_altitude = min(ray_altitude(record.s_in_m), ray_altitude(record.s_out_m))
        clearance = lower_ray_altitude - record.elevation_m
        if record.contact_type != "interior":
            continue
        minimum_clearance = min(minimum_clearance, clearance)
        if clearance <= numerical_tolerance_m:
            obstructing.append((record.row, record.col))

    boundary_sensitive = False
    corner_sensitive = False
    for record in records:
        if record.contact_type not in {"boundary", "corner"}:
            continue
        if exclude_endpoint_cells and (record.row, record.col) in endpoint_cells:
            continue
        lower_ray_altitude = min(ray_altitude(record.s_in_m), ray_altitude(record.s_out_m))
        if lower_ray_altitude - record.elevation_m <= numerical_tolerance_m:
            if record.contact_type == "corner":
                corner_sensitive = True
            else:
                boundary_sensitive = True

    if minimum_clearance == float("inf"):
        minimum_clearance = float("nan")
    return LineOfSightResult(
        clear=not obstructing,
        minimum_clearance_m=minimum_clearance,
        obstructing_cells=tuple(dict.fromkeys(obstructing)),
        boundary_contact_would_obstruct=boundary_sensitive,
        corner_contact_would_obstruct=corner_sensitive,
    )


def is_line_of_sight_clear(
    dem: DemGrid,
    enu: LocalEnu,
    lon1: float,
    lat1: float,
    altitude1_m: float,
    lon2: float,
    lat2: float,
    altitude2_m: float,
) -> bool:
    """兼容布尔接口；遮挡仅返回几何状态，链路可用性仍由预算决定。"""

    return evaluate_line_of_sight(
        dem, enu, lon1, lat1, altitude1_m, lon2, lat2, altitude2_m,
    ).clear
