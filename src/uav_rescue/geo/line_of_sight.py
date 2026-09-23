"""问题三可复用的三维视线遮挡判断。"""

from uav_rescue.geo.dem import DemGrid


def is_line_of_sight_clear(
    dem: DemGrid,
    lon1: float,
    lat1: float,
    altitude1_m: float,
    lon2: float,
    lat2: float,
    altitude2_m: float,
) -> bool:
    """逐像元比较视线高度和地形高程；端点海拔为绝对海拔。"""

    cells = dem.traversed_cells(lon1, lat1, lon2, lat2)
    count = max(len(cells) - 1, 1)
    for index, (row, col) in enumerate(cells):
        terrain = float(dem.pixels[col, row])
        if terrain <= -30_000:
            continue
        ratio = index / count
        line_altitude = altitude1_m + ratio * (altitude2_m - altitude1_m)
        if terrain > line_altitude:
            return False
    return True

