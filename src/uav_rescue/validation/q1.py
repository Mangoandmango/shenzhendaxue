"""问题一逐箱与逐架次校验。"""

from collections import Counter

from uav_rescue.domain import Box, TransportDrone
from uav_rescue.models.q1_batching import CandidateTrip


def validate_q1_solution(
    boxes_by_service: dict[str, list[Box]],
    drones: dict[str, TransportDrone],
    solutions: dict[str, list[CandidateTrip]],
) -> None:
    """检查唯一覆盖、服务区、质量、体积和允许能耗。"""

    expected = {box.box_id for boxes in boxes_by_service.values() for box in boxes}
    observed: list[str] = []
    for service_id, trips in solutions.items():
        boxes = boxes_by_service[service_id]
        for trip in trips:
            drone = drones[trip.model]
            if trip.mass_kg > drone.max_payload_kg + 1e-9:
                raise AssertionError("架次超过额定载荷")
            if trip.volume_m3 > drone.volume_m3 + 1e-12:
                raise AssertionError("架次超过装载体积")
            if trip.energy_kwh > (1.0 - drone.reserve_ratio) * drone.usable_energy_kwh + 1e-9:
                raise AssertionError("架次不满足返航安全余量")
            observed.extend(boxes[index].box_id for index in range(len(boxes)) if trip.mask & (1 << index))
    counts = Counter(observed)
    if set(counts) != expected or any(value != 1 for value in counts.values()):
        raise AssertionError("货箱未被唯一覆盖")

