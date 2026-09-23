"""问题一逐箱与逐架次校验。"""

from collections import Counter
from dataclasses import dataclass

from uav_rescue.domain import Box, LegGeometry, TransportDrone
from uav_rescue.models.q1_batching import CandidateTrip
from uav_rescue.physics.flight import simulate_leg_from_geometry


@dataclass(frozen=True)
class Q1ValidationRecord:
    """从货箱明细重新计算得到的逐架次复核记录。"""

    service_id: str
    trip_index: int
    model: str
    box_ids: str
    recomputed_mass_kg: float
    recomputed_volume_m3: float
    recomputed_energy_kwh: float
    recomputed_duration_s: float
    return_soc: float
    result: str


def validate_q1_solution(
    boxes_by_service: dict[str, list[Box]],
    drones: dict[str, TransportDrone],
    solutions: dict[str, list[CandidateTrip]],
    legs: dict[str, tuple[LegGeometry, LegGeometry]],
) -> list[Q1ValidationRecord]:
    """脱离优化器汇总量，按原始货箱和公共物理公式逐架次重新复核。"""

    expected = {box.box_id for boxes in boxes_by_service.values() for box in boxes}
    observed: list[str] = []
    records: list[Q1ValidationRecord] = []
    for service_id, trips in solutions.items():
        boxes = boxes_by_service[service_id]
        outbound, inbound = legs[service_id]
        for trip_index, trip in enumerate(trips, start=1):
            if trip.service_id != service_id:
                raise AssertionError("架次跨服务区或服务区编号不一致")
            drone = drones[trip.model]
            selected = [boxes[index] for index in range(len(boxes)) if trip.mask & (1 << index)]
            if not selected:
                raise AssertionError("出现未装载货箱的空架次")
            mass = sum(box.mass_kg for box in selected)
            volume = sum(box.volume_m3 for box in selected)
            outbound_simulation = simulate_leg_from_geometry(drone, outbound, mass)
            inbound_simulation = simulate_leg_from_geometry(drone, inbound, 0.0)
            energy = outbound_simulation.energy_kwh + inbound_simulation.energy_kwh
            duration = (
                drone.prep_s
                + len(selected) * drone.load_per_box_s
                + outbound_simulation.flight_time_s
                + inbound_simulation.flight_time_s
                + drone.handoff_base_s
                + len(selected) * drone.handoff_per_box_s
            )
            return_soc = 1.0 - energy / drone.usable_energy_kwh
            if mass > drone.max_payload_kg + 1e-9:
                raise AssertionError("架次超过额定载荷")
            if volume > drone.volume_m3 + 1e-12:
                raise AssertionError("架次超过装载体积")
            if energy > (1.0 - drone.reserve_ratio) * drone.usable_energy_kwh + 1e-9:
                raise AssertionError("架次不满足返航安全余量")
            if abs(mass - trip.mass_kg) > 1e-9 or abs(volume - trip.volume_m3) > 1e-12:
                raise AssertionError("优化器记录的质量或体积与逐箱重算结果不一致")
            if abs(energy - trip.energy_kwh) > 1e-9 or abs(duration - trip.duration_s) > 1e-8:
                raise AssertionError("优化器记录的能耗或时间与独立重算结果不一致")
            observed.extend(box.box_id for box in selected)
            records.append(Q1ValidationRecord(
                service_id, trip_index, trip.model,
                ";".join(box.box_id for box in selected),
                mass, volume, energy, duration, return_soc, "通过",
            ))
    counts = Counter(observed)
    if set(counts) != expected or any(value != 1 for value in counts.values()):
        raise AssertionError("货箱未被唯一覆盖")
    return records
