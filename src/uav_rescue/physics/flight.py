"""航段几何、飞行时间与统一航段物理接口。"""

from uav_rescue.domain import LegGeometry, LegSimulation, Node, TransportDrone
from uav_rescue.geo.coordinates import LocalEnu
from uav_rescue.geo.dem import DemGrid
from uav_rescue.physics.energy import equivalent_range_m, leg_energy_kwh


def build_leg(
    dem: DemGrid,
    enu: LocalEnu,
    origin: Node,
    destination: Node,
    origin_operation_altitude_m: float,
    destination_operation_altitude_m: float,
    clearance_m: float = 50.0,
) -> LegGeometry:
    """按沿线最高 DEM 高程加净空构造一个航段。"""

    distance, records = dem.trace_enu_segment(enu, origin.lon, origin.lat, destination.lon, destination.lat)
    traversed = dem.positive_intersections(records)
    cruise_altitude = max(record.elevation_m for record in traversed) + clearance_m
    return LegGeometry(
        origin.node_id,
        destination.node_id,
        distance,
        cruise_altitude,
        max(0.0, cruise_altitude - origin_operation_altitude_m),
        max(0.0, cruise_altitude - destination_operation_altitude_m),
        len({(record.row, record.col) for record in traversed}),
    )


def leg_time_s(drone: TransportDrone, leg: LegGeometry) -> float:
    """计算爬升、巡航和下降时间之和。"""

    return (
        leg.climb_m / drone.climb_speed_mps
        + leg.distance_m / drone.cruise_speed_mps
        + leg.descent_m / drone.descent_speed_mps
    )


def simulate_leg_from_geometry(
    drone: TransportDrone,
    leg: LegGeometry,
    payload_kg: float,
) -> LegSimulation:
    """在已缓存航段几何量上计算 ``(i, j, k, q) -> [T(q), E(q)]``。

    问题一可直接传入单架次总载荷；问题二在每次投送后传入更新的剩余载荷；
    问题三可在该结果外叠加通信链路约束。几何量和载荷状态因此被明确分离。
    """

    if not 0.0 <= payload_kg <= drone.max_payload_kg:
        raise ValueError(f"载荷 {payload_kg} kg 超出 {drone.model} 的允许范围")
    return LegSimulation(
        geometry=leg,
        model=drone.model,
        payload_kg=payload_kg,
        equivalent_range_m=equivalent_range_m(drone, payload_kg),
        flight_time_s=leg_time_s(drone, leg),
        energy_kwh=leg_energy_kwh(drone, leg, payload_kg),
    )


def simulate_leg(
    dem: DemGrid,
    enu: LocalEnu,
    drone: TransportDrone,
    origin: Node,
    destination: Node,
    payload_kg: float,
    origin_operation_altitude_m: float,
    destination_operation_altitude_m: float,
    clearance_m: float = 50.0,
) -> LegSimulation:
    """从节点与 DEM 构造航段后调用统一接口，适合没有预建缓存的场景。"""

    leg = build_leg(
        dem, enu, origin, destination, origin_operation_altitude_m,
        destination_operation_altitude_m, clearance_m,
    )
    return simulate_leg_from_geometry(drone, leg, payload_kg)
