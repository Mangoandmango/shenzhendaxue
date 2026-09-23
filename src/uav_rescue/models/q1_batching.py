"""问题一候选架次生成与精确集合划分。"""

from dataclasses import dataclass
from functools import lru_cache

from uav_rescue.domain import Box, LegGeometry, TransportDrone
from uav_rescue.physics.flight import simulate_leg_from_geometry


@dataclass(frozen=True)
class CandidateTrip:
    service_id: str
    model: str
    mask: int
    mass_kg: float
    volume_m3: float
    energy_kwh: float
    duration_s: float


def generate_candidates(
    service_id: str,
    boxes: list[Box],
    drones: dict[str, TransportDrone],
    outbound: LegGeometry,
    inbound: LegGeometry,
) -> list[CandidateTrip]:
    """枚举该服务区全部满足质量、体积和能量条件的候选架次。"""

    count = len(boxes)
    mass = [0.0] * (1 << count)
    volume = [0.0] * (1 << count)
    box_count = [0] * (1 << count)
    for mask in range(1, 1 << count):
        bit = mask & -mask
        index = bit.bit_length() - 1
        previous = mask ^ bit
        mass[mask] = mass[previous] + boxes[index].mass_kg
        volume[mask] = volume[previous] + boxes[index].volume_m3
        box_count[mask] = box_count[previous] + 1

    result: list[CandidateTrip] = []
    for mask in range(1, 1 << count):
        for model, drone in sorted(drones.items()):
            if mass[mask] > drone.max_payload_kg + 1e-10 or volume[mask] > drone.volume_m3 + 1e-12:
                continue
            outbound_simulation = simulate_leg_from_geometry(drone, outbound, mass[mask])
            inbound_simulation = simulate_leg_from_geometry(drone, inbound, 0.0)
            energy = outbound_simulation.energy_kwh + inbound_simulation.energy_kwh
            if energy > (1.0 - drone.reserve_ratio) * drone.usable_energy_kwh + 1e-10:
                continue
            duration = (
                drone.prep_s
                + box_count[mask] * drone.load_per_box_s
                + outbound_simulation.flight_time_s
                + inbound_simulation.flight_time_s
                + drone.handoff_base_s
                + box_count[mask] * drone.handoff_per_box_s
            )
            result.append(CandidateTrip(service_id, model, mask, mass[mask], volume[mask], energy, duration))
    return result


def solve_exact_partition(boxes: list[Box], candidates: list[CandidateTrip]) -> tuple[tuple[int, float, float], list[CandidateTrip]]:
    """依次最小化架次数、总能耗和累计作业时间。"""

    full_mask = (1 << len(boxes)) - 1
    by_box: list[list[CandidateTrip]] = [[] for _ in boxes]
    for candidate in candidates:
        for index in range(len(boxes)):
            if candidate.mask & (1 << index):
                by_box[index].append(candidate)

    @lru_cache(maxsize=None)
    def solve(covered: int):
        if covered == full_mask:
            return (0, 0.0, 0.0), ()
        uncovered = (~covered) & full_mask
        first = (uncovered & -uncovered).bit_length() - 1
        best = None
        for candidate in by_box[first]:
            if candidate.mask & covered:
                continue
            tail = solve(covered | candidate.mask)
            if tail is None:
                continue
            score = (1 + tail[0][0], candidate.energy_kwh + tail[0][1], candidate.duration_s + tail[0][2])
            proposal = score, (candidate,) + tail[1]
            if best is None or score < best[0]:
                best = proposal
        return best

    answer = solve(0)
    if answer is None:
        raise RuntimeError(f"{boxes[0].service_id} 不存在覆盖全部货箱的可行组批")
    return answer[0], list(answer[1])
