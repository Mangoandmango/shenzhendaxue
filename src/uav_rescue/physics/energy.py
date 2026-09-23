"""载荷—航程与运输能耗公式。"""

from uav_rescue.domain import LegGeometry, TransportDrone


def equivalent_range_m(drone: TransportDrone, payload_kg: float) -> float:
    """实现题面给出的 3/2 次载荷—等效航程关系。"""

    ratio = min(max(payload_kg / drone.max_payload_kg, 0.0), 1.0)
    return drone.empty_range_m - (drone.empty_range_m - drone.full_range_m) * ratio ** 1.5


def leg_energy_kwh(
    drone: TransportDrone,
    leg: LegGeometry,
    payload_kg: float,
    gravity_mps2: float = 9.81,
    joules_per_kwh: float = 3_600_000.0,
) -> float:
    """计算水平航程能耗和爬升附加能耗。"""

    horizontal = drone.usable_energy_kwh * leg.distance_m / equivalent_range_m(drone, payload_kg)
    climb = (
        (drone.empty_mass_kg + payload_kg)
        * gravity_mps2
        * leg.climb_m
        / (drone.climb_efficiency * joules_per_kwh)
    )
    return horizontal + climb


def round_trip_energy_kwh(
    drone: TransportDrone,
    outbound: LegGeometry,
    inbound: LegGeometry,
    outbound_payload_kg: float,
) -> float:
    """单点任务去程带货、返程空载。"""

    return leg_energy_kwh(drone, outbound, outbound_payload_kg) + leg_energy_kwh(drone, inbound, 0.0)


def maximum_safe_payload_kg(
    drone: TransportDrone,
    outbound: LegGeometry,
    inbound: LegGeometry,
) -> float | None:
    """用二分法求满足返航安全余量的最大载荷。"""

    energy_limit = (1.0 - drone.reserve_ratio) * drone.usable_energy_kwh
    if round_trip_energy_kwh(drone, outbound, inbound, 0.0) > energy_limit:
        return None
    if round_trip_energy_kwh(drone, outbound, inbound, drone.max_payload_kg) <= energy_limit:
        return drone.max_payload_kg
    lower, upper = 0.0, drone.max_payload_kg
    for _ in range(80):
        middle = (lower + upper) / 2
        if round_trip_energy_kwh(drone, outbound, inbound, middle) <= energy_limit:
            lower = middle
        else:
            upper = middle
    return lower

