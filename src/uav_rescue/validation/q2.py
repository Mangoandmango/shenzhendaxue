"""问题二最终排程的独立复算与约束审计。"""

from __future__ import annotations

from dataclasses import dataclass

from uav_rescue.domain import Box, TransportBattery, TransportDrone, TransportUnit
from uav_rescue.models.q2_transport import (
    RouteEvaluator,
    ScheduleResult,
    hard_deadline_s,
)
from uav_rescue.physics.battery import recharge_time_s


@dataclass(frozen=True)
class CheckRow:
    check: str
    object_id: str
    calculated: object
    allowed: object
    passed: bool
    detail: str = ""


def validate_q2_solution(
    schedule: ScheduleResult,
    boxes: dict[str, Box],
    drones: dict[str, TransportDrone],
    units: dict[str, TransportUnit],
    batteries: dict[str, TransportBattery],
    evaluator: RouteEvaluator,
    tolerance: float = 1e-6,
) -> list[CheckRow]:
    """不依赖优化器目标缓存，从路线、原始参数和资源区间重新检查。"""

    rows: list[CheckRow] = []
    occurrences = {box_id: 0 for box_id in boxes}
    for trip in schedule.trips:
        for box_id in trip.route.box_ids:
            occurrences[box_id] = occurrences.get(box_id, 0) + 1
    for box_id in sorted(boxes):
        rows.append(CheckRow(
            "货箱唯一配送", box_id, occurrences.get(box_id, 0), 1,
            occurrences.get(box_id, 0) == 1,
        ))

    for trip in schedule.trips:
        matches = {
            item.model: item
            for item in evaluator.evaluate(trip.route.box_ids, trip.route.service_sequence)
        }
        recomputed = matches.get(trip.evaluation.model)
        rows.append(CheckRow(
            "机型物理可行", trip.trip_id, trip.evaluation.model,
            "存在可行复算", recomputed is not None,
        ))
        if recomputed is None:
            continue
        drone = drones[recomputed.model]
        rows.extend([
            CheckRow("载质量", trip.trip_id, recomputed.mass_kg, drone.max_payload_kg,
                     recomputed.mass_kg <= drone.max_payload_kg + tolerance),
            CheckRow("装载体积", trip.trip_id, recomputed.volume_m3, drone.volume_m3,
                     recomputed.volume_m3 <= drone.volume_m3 + tolerance),
            CheckRow("返航SOC", trip.trip_id, recomputed.end_soc, drone.reserve_ratio,
                     recomputed.end_soc + tolerance >= drone.reserve_ratio),
            CheckRow("架次能耗复算", trip.trip_id, trip.evaluation.energy_kwh, recomputed.energy_kwh,
                     abs(trip.evaluation.energy_kwh - recomputed.energy_kwh) <= tolerance),
            CheckRow("返航时刻复算", trip.trip_id, trip.return_s,
                     trip.start_s + recomputed.duration_s,
                     abs(trip.return_s - trip.start_s - recomputed.duration_s) <= tolerance),
        ])
        payloads = [leg.payload_before_kg for leg in recomputed.legs]
        decreasing = all(
            payloads[index + 1] <= payloads[index] + tolerance
            for index in range(len(payloads) - 1)
        ) and abs(payloads[-1]) <= tolerance
        rows.append(CheckRow(
            "逐航段载荷递减", trip.trip_id,
            "|".join(f"{value:.6f}" for value in payloads),
            "非增且返程为0", decreasing,
        ))
        rows.append(CheckRow(
            "实体机型兼容", trip.trip_id, units[trip.unit_id].model, recomputed.model,
            units[trip.unit_id].model == recomputed.model,
        ))
        rows.append(CheckRow(
            "电池机型兼容", trip.trip_id, batteries[trip.battery_id].model, recomputed.model,
            batteries[trip.battery_id].model == recomputed.model,
        ))
        expected_charge = recharge_time_s(
            recomputed.end_soc, batteries[trip.battery_id].full_charge_time_s
        )
        rows.append(CheckRow(
            "充电时间复算", trip.trip_id, trip.recharge_s, expected_charge,
            abs(trip.recharge_s - expected_charge) <= tolerance,
        ))
        rows.append(CheckRow(
            "电池释放时刻", trip.trip_id, trip.battery_release_s,
            trip.return_s + expected_charge,
            abs(trip.battery_release_s - trip.return_s - expected_charge) <= tolerance,
        ))
        for box_id, offset in recomputed.delivery_offsets_s:
            delivered = trip.delivery_time(box_id)
            rows.append(CheckRow(
                "逐箱送达时刻复算", box_id, delivered, trip.start_s + offset,
                abs(delivered - trip.start_s - offset) <= tolerance,
                trip.trip_id,
            ))
            deadline = hard_deadline_s(boxes[box_id])
            if deadline is not None:
                rows.append(CheckRow(
                    "硬时限", box_id, delivered, deadline,
                    delivered <= deadline + tolerance, trip.trip_id,
                ))

    for unit_id in sorted(units):
        intervals = sorted(
            (trip.start_s, trip.return_s, trip.trip_id)
            for trip in schedule.trips if trip.unit_id == unit_id
        )
        for first, second in zip(intervals, intervals[1:]):
            rows.append(CheckRow(
                "无人机时间冲突", f"{unit_id}:{first[2]}->{second[2]}",
                second[0], first[1], second[0] + tolerance >= first[1],
                "下一架次开始不得早于前一架次返航",
            ))
    for battery_id in sorted(batteries):
        intervals = sorted(
            (trip.start_s, trip.battery_release_s, trip.trip_id)
            for trip in schedule.trips if trip.battery_id == battery_id
        )
        for first, second in zip(intervals, intervals[1:]):
            rows.append(CheckRow(
                "电池任务充电冲突", f"{battery_id}:{first[2]}->{second[2]}",
                second[0], first[1], second[0] + tolerance >= first[1],
                "电池从架次开始占用至返航后充至100%",
            ))
    return rows


def assert_q2_valid(rows: list[CheckRow]) -> None:
    failures = [row for row in rows if not row.passed]
    if failures:
        summary = "; ".join(f"{row.check}:{row.object_id}" for row in failures[:10])
        raise AssertionError(f"问题二独立复核失败，共 {len(failures)} 项：{summary}")
