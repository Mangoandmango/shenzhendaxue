"""问题二的路线、架次物理量、资源排程与目标定义。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from uav_rescue.domain import (
    Box,
    LegGeometry,
    TransportBattery,
    TransportDrone,
    TransportUnit,
)
from uav_rescue.physics.battery import recharge_time_s, remaining_soc
from uav_rescue.physics.flight import simulate_leg_from_geometry


TOLERANCE = 1e-8


def hard_deadline_s(box: Box) -> float | None:
    """返回货箱的有效硬截止时间。

    用户统一口径为：所有物资的期望送达时间均是硬时间窗；
    首批保障货箱还同时受首批截止时间限制，两者取较早值。
    """

    deadlines: list[float] = []
    if box.expected_deadline_s is not None:
        deadlines.append(box.expected_deadline_s)
    if box.is_first_batch and box.first_deadline_s is not None:
        deadlines.append(box.first_deadline_s)
    return min(deadlines) if deadlines else None


@dataclass(frozen=True)
class RoutePlan:
    """尚未分配机型和时刻的一条候选路线。"""

    box_ids: tuple[str, ...]
    service_sequence: tuple[str, ...]


@dataclass(frozen=True)
class TypedRoutePlan:
    """ALNS 搜索层的架次编码：路线与指定机型。"""

    route: RoutePlan
    model: str


@dataclass(frozen=True)
class LegDetail:
    origin: str
    destination: str
    payload_before_kg: float
    flight_time_s: float
    energy_kwh: float


@dataclass(frozen=True)
class TripEvaluation:
    """某机型从时刻0执行路线时的相对时间与能耗。"""

    model: str
    mass_kg: float
    volume_m3: float
    duration_s: float
    energy_kwh: float
    end_soc: float
    delivery_offsets_s: tuple[tuple[str, float], ...]
    legs: tuple[LegDetail, ...]

    def delivery_offset(self, box_id: str) -> float:
        return dict(self.delivery_offsets_s)[box_id]


@dataclass(frozen=True)
class ScheduledTrip:
    trip_id: str
    route: RoutePlan
    evaluation: TripEvaluation
    unit_id: str
    battery_id: str
    start_s: float
    return_s: float
    recharge_s: float
    battery_release_s: float

    def delivery_time(self, box_id: str) -> float:
        return self.start_s + self.evaluation.delivery_offset(box_id)


@dataclass(frozen=True, order=True)
class Objective:
    """排程的硬违约与四个软目标原始值；多目标选择在求解器中完成。"""

    hard_violation_count: int
    hard_tardiness_s: float
    weighted_soft_tardiness: float
    makespan_s: float
    energy_kwh: float
    trip_count: int


@dataclass(frozen=True)
class ScheduleResult:
    trips: tuple[ScheduledTrip, ...]
    objective: Objective


class RouteEvaluator:
    """复用统一物理模块，并缓存同一路线的三机型计算结果。"""

    def __init__(
        self,
        boxes: dict[str, Box],
        drones: dict[str, TransportDrone],
        legs: dict[tuple[str, str], LegGeometry],
    ) -> None:
        self.boxes = boxes
        self.drones = drones
        self.legs = legs

    @lru_cache(maxsize=None)
    def evaluate(self, box_ids: tuple[str, ...], service_sequence: tuple[str, ...]) -> tuple[TripEvaluation, ...]:
        route = RoutePlan(box_ids, service_sequence)
        if not route.box_ids or not route.service_sequence:
            return ()
        if len(set(route.service_sequence)) != len(route.service_sequence):
            return ()
        if {self.boxes[box_id].service_id for box_id in route.box_ids} != set(route.service_sequence):
            return ()
        result: list[TripEvaluation] = []
        for _, drone in sorted(self.drones.items()):
            evaluation = self._evaluate_for_drone(route, drone)
            if evaluation is not None:
                result.append(evaluation)
        return tuple(result)

    def _evaluate_for_drone(self, route: RoutePlan, drone: TransportDrone) -> TripEvaluation | None:
        selected = [self.boxes[box_id] for box_id in route.box_ids]
        mass = sum(box.mass_kg for box in selected)
        volume = sum(box.volume_m3 for box in selected)
        if mass > drone.max_payload_kg + TOLERANCE or volume > drone.volume_m3 + TOLERANCE:
            return None

        boxes_by_service: dict[str, list[Box]] = {service: [] for service in route.service_sequence}
        for box in selected:
            boxes_by_service[box.service_id].append(box)

        elapsed = drone.prep_s + len(selected) * drone.load_per_box_s
        payload = mass
        energy = 0.0
        current = "O01"
        deliveries: list[tuple[str, float]] = []
        leg_details: list[LegDetail] = []
        for service in route.service_sequence:
            simulation = simulate_leg_from_geometry(drone, self.legs[(current, service)], payload)
            elapsed += simulation.flight_time_s
            energy += simulation.energy_kwh
            leg_details.append(LegDetail(
                current, service, payload, simulation.flight_time_s, simulation.energy_kwh
            ))
            service_boxes = boxes_by_service[service]
            elapsed += drone.handoff_base_s + len(service_boxes) * drone.handoff_per_box_s
            deliveries.extend((box.box_id, elapsed) for box in service_boxes)
            payload -= sum(box.mass_kg for box in service_boxes)
            current = service

        return_simulation = simulate_leg_from_geometry(drone, self.legs[(current, "O01")], 0.0)
        elapsed += return_simulation.flight_time_s
        energy += return_simulation.energy_kwh
        leg_details.append(LegDetail(
            current, "O01", 0.0, return_simulation.flight_time_s, return_simulation.energy_kwh
        ))
        end_soc = remaining_soc(drone.usable_energy_kwh, energy)
        if end_soc + TOLERANCE < drone.reserve_ratio:
            return None
        return TripEvaluation(
            drone.model, mass, volume, elapsed, energy, end_soc,
            tuple(sorted(deliveries)), tuple(leg_details),
        )


def route_from_boxes(box_ids: tuple[str, ...], boxes: dict[str, Box]) -> RoutePlan:
    """按货箱首次出现顺序压缩为不重复的服务区访问序列。"""

    services: list[str] = []
    for box_id in box_ids:
        service = boxes[box_id].service_id
        if service not in services:
            services.append(service)
    return RoutePlan(box_ids, tuple(services))


def objective_from_schedule(trips: list[ScheduledTrip], boxes: dict[str, Box]) -> Objective:
    deliveries = {
        box_id: trip.delivery_time(box_id)
        for trip in trips
        for box_id in trip.route.box_ids
    }
    hard_count = 0
    hard_tardiness = 0.0
    weighted_soft = 0.0
    for box_id, box in boxes.items():
        delivered = deliveries.get(box_id, math.inf)
        deadline = hard_deadline_s(box)
        if deadline is not None:
            late = max(0.0, delivered - deadline)
            if late > TOLERANCE:
                hard_count += 1
                hard_tardiness += late
        elif box.expected_deadline_s is not None:
            weighted_soft += (box.priority or 1.0) * max(0.0, delivered - box.expected_deadline_s)
    return Objective(
        hard_count,
        hard_tardiness,
        weighted_soft,
        max((trip.return_s for trip in trips), default=0.0),
        sum(trip.evaluation.energy_kwh for trip in trips),
        len(trips),
    )


def schedule_routes(
    routes: list[RoutePlan],
    evaluator: RouteEvaluator,
    boxes: dict[str, Box],
    units: dict[str, TransportUnit],
    batteries: dict[str, TransportBattery],
) -> ScheduleResult:
    """按硬截止时间和松弛时间列表调度，同时选择机型、实体机和电池。"""

    unit_available = {unit_id: 0.0 for unit_id in units}
    battery_available = {battery_id: 0.0 for battery_id in batteries}

    def route_priority(route: RoutePlan) -> tuple:
        selected = [boxes[box_id] for box_id in route.box_ids]
        hard = [hard_deadline_s(box) for box in selected if hard_deadline_s(box) is not None]
        expected = [box.expected_deadline_s for box in selected if box.expected_deadline_s is not None]
        medical_count = sum(box.category == "医疗物资" for box in selected)
        return (
            min(hard, default=math.inf),
            -medical_count,
            min(expected, default=math.inf),
            -sum(box.priority or 0.0 for box in selected),
            route.service_sequence,
            route.box_ids,
        )

    scheduled: list[ScheduledTrip] = []
    for trip_number, route in enumerate(sorted(routes, key=route_priority), start=1):
        evaluations = evaluator.evaluate(route.box_ids, route.service_sequence)
        if not evaluations:
            raise RuntimeError(f"候选路线不存在可行机型：{route.box_ids}")
        best = None
        for evaluation in evaluations:
            compatible_units = sorted(
                unit_id for unit_id, unit in units.items() if unit.model == evaluation.model
            )
            compatible_batteries = sorted(
                battery_id for battery_id, battery in batteries.items()
                if battery.model == evaluation.model
            )
            for unit_id in compatible_units:
                for battery_id in compatible_batteries:
                    start = max(unit_available[unit_id], battery_available[battery_id])
                    hard_count = 0
                    hard_tardiness = 0.0
                    soft = 0.0
                    for box_id, offset in evaluation.delivery_offsets_s:
                        box = boxes[box_id]
                        delivered = start + offset
                        deadline = hard_deadline_s(box)
                        if deadline is not None:
                            late = max(0.0, delivered - deadline)
                            hard_count += int(late > TOLERANCE)
                            hard_tardiness += late
                        elif box.expected_deadline_s is not None:
                            soft += (box.priority or 1.0) * max(
                                0.0, delivered - box.expected_deadline_s
                            )
                    return_s = start + evaluation.duration_s
                    key = (
                        hard_count, hard_tardiness, soft, return_s,
                        evaluation.energy_kwh, start,
                        unit_available[unit_id], battery_available[battery_id],
                        evaluation.model, unit_id, battery_id,
                    )
                    if best is None or key < best[0]:
                        best = key, evaluation, unit_id, battery_id, start, return_s
        if best is None:
            raise RuntimeError(f"候选路线无法分配实体机或电池：{route.box_ids}")
        _, evaluation, unit_id, battery_id, start, return_s = best
        battery = batteries[battery_id]
        recharge = recharge_time_s(evaluation.end_soc, battery.full_charge_time_s)
        battery_release = return_s + recharge
        scheduled.append(ScheduledTrip(
            f"T{trip_number:03d}", route, evaluation, unit_id, battery_id,
            start, return_s, recharge, battery_release,
        ))
        unit_available[unit_id] = return_s
        battery_available[battery_id] = battery_release

    return ScheduleResult(tuple(scheduled), objective_from_schedule(scheduled, boxes))


def schedule_typed_routes_event_driven(
    plans: tuple[TypedRoutePlan, ...] | list[TypedRoutePlan],
    evaluator: RouteEvaluator,
    boxes: dict[str, Box],
    units: dict[str, TransportUnit],
    batteries: dict[str, TransportBattery],
) -> ScheduleResult:
    """用事件时钟联合分配指定机型的架次、实体机与共享电池。

    同一事件时刻只在当前已空闲的资源中派发；若没有可派发组合，
    时钟直接跳到下一个兼容无人机与满电电池同时可用的时刻。
    """

    indexed: list[tuple[int, TypedRoutePlan, TripEvaluation]] = []
    for index, plan in enumerate(plans):
        matches = {
            item.model: item
            for item in evaluator.evaluate(
                plan.route.box_ids, plan.route.service_sequence
            )
        }
        evaluation = matches.get(plan.model)
        if evaluation is None:
            raise RuntimeError(
                f"架次在指定机型下物理不可行：{plan.model}:{plan.route.box_ids}"
            )
        if not any(unit.model == plan.model for unit in units.values()):
            raise RuntimeError(f"指定机型没有实体无人机：{plan.model}")
        if not any(battery.model == plan.model for battery in batteries.values()):
            raise RuntimeError(f"指定机型没有共享电池：{plan.model}")
        indexed.append((index, plan, evaluation))

    remaining = indexed.copy()
    unit_available = {unit_id: 0.0 for unit_id in units}
    battery_available = {battery_id: 0.0 for battery_id in batteries}
    current_time = 0.0
    scheduled: list[ScheduledTrip] = []

    while remaining:
        available_units: dict[str, list[str]] = {}
        available_batteries: dict[str, list[str]] = {}
        for model in {plan.model for _, plan, _ in remaining}:
            available_units[model] = sorted(
                unit_id for unit_id, unit in units.items()
                if unit.model == model and unit_available[unit_id] <= current_time + TOLERANCE
            )
            available_batteries[model] = sorted(
                battery_id for battery_id, battery in batteries.items()
                if battery.model == model
                and battery_available[battery_id] <= current_time + TOLERANCE
            )

        candidates: list[tuple[tuple[object, ...], int, TypedRoutePlan, TripEvaluation]] = []
        for original_index, plan, evaluation in remaining:
            if not available_units[plan.model] or not available_batteries[plan.model]:
                continue
            violation_count = 0
            tardiness = 0.0
            minimum_slack = math.inf
            earliest_deadline = math.inf
            medical_count = 0
            priority_sum = 0.0
            for box_id, offset in evaluation.delivery_offsets_s:
                box = boxes[box_id]
                deadline = hard_deadline_s(box)
                if deadline is None:
                    continue
                delivery = current_time + offset
                late = max(0.0, delivery - deadline)
                violation_count += int(late > TOLERANCE)
                tardiness += late
                minimum_slack = min(minimum_slack, deadline - delivery)
                earliest_deadline = min(earliest_deadline, deadline)
                medical_count += int(box.category == "医疗物资")
                priority_sum += box.priority or 0.0
            # 有序架次列表本身就是搜索层给出的派发优先级。在新增硬违约
            # 与迟到量相同时，先保留该顺序，再用截止时间和松弛量打破平局。
            key: tuple[object, ...] = (
                violation_count,
                tardiness,
                original_index,
                earliest_deadline,
                -medical_count,
                -priority_sum,
                minimum_slack,
                current_time + evaluation.duration_s,
                evaluation.energy_kwh,
                plan.model,
                plan.route.service_sequence,
                plan.route.box_ids,
            )
            candidates.append((key, original_index, plan, evaluation))

        if not candidates:
            next_times: list[float] = []
            for _, plan, _ in remaining:
                next_unit = min(
                    unit_available[unit_id]
                    for unit_id, unit in units.items() if unit.model == plan.model
                )
                next_battery = min(
                    battery_available[battery_id]
                    for battery_id, battery in batteries.items()
                    if battery.model == plan.model
                )
                next_times.append(max(next_unit, next_battery))
            next_time = min(next_times)
            if next_time <= current_time + TOLERANCE:
                raise RuntimeError("事件驱动解码器无法推进时钟")
            current_time = next_time
            continue

        _, original_index, plan, evaluation = min(candidates, key=lambda item: item[0])
        unit_id = available_units[plan.model][0]
        battery_id = available_batteries[plan.model][0]
        return_s = current_time + evaluation.duration_s
        recharge_s = recharge_time_s(
            evaluation.end_soc, batteries[battery_id].full_charge_time_s
        )
        battery_release_s = return_s + recharge_s
        trip_id = f"T{len(scheduled) + 1:03d}"
        scheduled.append(ScheduledTrip(
            trip_id,
            plan.route,
            evaluation,
            unit_id,
            battery_id,
            current_time,
            return_s,
            recharge_s,
            battery_release_s,
        ))
        unit_available[unit_id] = return_s
        battery_available[battery_id] = battery_release_s
        remaining = [item for item in remaining if item[0] != original_index]

    return ScheduleResult(tuple(scheduled), objective_from_schedule(scheduled, boxes))
