"""问题二多起点启发式：巨型服务区序列、DP切分、列表调度和反馈修复。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

from uav_rescue.domain import Box, LegGeometry, Node, TransportBattery, TransportDrone, TransportUnit
from uav_rescue.models.q2_transport import (
    Objective,
    RouteEvaluator,
    RoutePlan,
    ScheduleResult,
    hard_deadline_s,
    route_from_boxes,
    schedule_routes,
)


@dataclass(frozen=True)
class SearchRecord:
    label: str
    seed: int | None
    preference: str
    iterations: int
    elapsed_s: float
    objective: Objective


@dataclass(frozen=True)
class Q2SearchResult:
    # best 为 Pareto 前沿上经 min-max 归一化后距理想点最近的综合折中解。
    best: ScheduleResult
    baseline: ScheduleResult
    records: tuple[SearchRecord, ...]
    pareto_front: tuple[ScheduleResult, ...]
    representatives: dict[str, ScheduleResult]


PREFERENCE_PROFILES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("完工时间优先", (1, 0, 2, 3)),
    ("能耗优先", (2, 0, 1, 3)),
    ("最少架次优先", (3, 0, 1, 2)),
)


def pareto_vector(schedule: ScheduleResult) -> tuple[float, float, float, float]:
    objective = schedule.objective
    return (
        objective.weighted_soft_tardiness,
        objective.makespan_s,
        objective.energy_kwh,
        float(objective.trip_count),
    )


def _hard_key(schedule: ScheduleResult) -> tuple[int, float]:
    return schedule.objective.hard_violation_count, schedule.objective.hard_tardiness_s


def _preference_key(schedule: ScheduleResult, order: tuple[int, ...]) -> tuple[float, ...]:
    vector = pareto_vector(schedule)
    return (*_hard_key(schedule), *(vector[index] for index in order))


def _dominates(left: ScheduleResult, right: ScheduleResult, tolerance: float = 1e-8) -> bool:
    left_values = pareto_vector(left)
    right_values = pareto_vector(right)
    return (
        all(a <= b + tolerance for a, b in zip(left_values, right_values))
        and any(a < b - tolerance for a, b in zip(left_values, right_values))
    )


def pareto_filter_schedules(candidates: list[ScheduleResult]) -> tuple[ScheduleResult, ...]:
    """在最优硬约束层级内提取四目标 Pareto 前沿。"""
    if not candidates:
        return ()
    best_hard = min(_hard_key(candidate) for candidate in candidates)
    eligible = [candidate for candidate in candidates if _hard_key(candidate) == best_hard]
    unique: dict[tuple[float, ...], ScheduleResult] = {}
    for candidate in eligible:
        key = tuple(round(value, 8) for value in pareto_vector(candidate))
        unique.setdefault(key, candidate)
    front = [
        candidate for candidate in unique.values()
        if not any(_dominates(other, candidate) for other in unique.values() if other is not candidate)
    ]
    return tuple(sorted(front, key=pareto_vector))


def normalized_ideal_distance(
    schedule: ScheduleResult,
    front: tuple[ScheduleResult, ...],
) -> tuple[float, tuple[float, float, float, float]]:
    vectors = [pareto_vector(item) for item in front]
    minima = tuple(min(vector[index] for vector in vectors) for index in range(4))
    maxima = tuple(max(vector[index] for vector in vectors) for index in range(4))
    vector = pareto_vector(schedule)
    normalized = tuple(
        0.0 if maxima[index] - minima[index] <= 1e-12
        else (vector[index] - minima[index]) / (maxima[index] - minima[index])
        for index in range(4)
    )
    distance = math.sqrt(sum(value * value for value in normalized) / 4.0)
    return distance, normalized


def select_pareto_representatives(
    front: tuple[ScheduleResult, ...],
) -> dict[str, ScheduleResult]:
    if not front:
        raise ValueError("Pareto 前沿不能为空")
    result = {
        name: min(front, key=lambda item, order=order: _preference_key(item, order))
        for name, order in PREFERENCE_PROFILES
    }
    # 所有期望送达时间都是硬时间窗后，可行解的迟到指标均为0。
    # 因此“及时性优先”不再单独搜索，以零违约解中完工最早者作为其代表。
    result["及时性优先"] = result["完工时间优先"]
    result["综合折中"] = min(
        front,
        key=lambda item: (normalized_ideal_distance(item, front)[0], pareto_vector(item)),
    )
    return result


def _service_box_order(service_id: str, boxes_by_service: dict[str, list[Box]]) -> list[Box]:
    return sorted(
        boxes_by_service[service_id],
        key=lambda box: (
            hard_deadline_s(box) if hard_deadline_s(box) is not None else math.inf,
            box.expected_deadline_s if box.expected_deadline_s is not None else math.inf,
            -(box.priority or 0.0),
            box.box_id,
        ),
    )


def boxes_for_service_order(order: tuple[str, ...], boxes_by_service: dict[str, list[Box]]) -> tuple[str, ...]:
    return tuple(
        box.box_id
        for service_id in order
        for box in _service_box_order(service_id, boxes_by_service)
    )


def _interval_cost(
    route: RoutePlan,
    evaluator: RouteEvaluator,
    boxes: dict[str, Box],
) -> tuple[tuple[float, ...], str] | None:
    evaluations = evaluator.evaluate(route.box_ids, route.service_sequence)
    if not evaluations:
        return None
    best = None
    for evaluation in evaluations:
        hard_count = 0
        hard_tardiness = 0.0
        soft = 0.0
        for box_id, offset in evaluation.delivery_offsets_s:
            box = boxes[box_id]
            deadline = hard_deadline_s(box)
            if deadline is not None:
                late = max(0.0, offset - deadline)
                hard_count += int(late > 1e-8)
                hard_tardiness += late
            elif box.expected_deadline_s is not None:
                soft += (box.priority or 1.0) * max(0.0, offset - box.expected_deadline_s)
        cost = (
            float(hard_count), hard_tardiness, soft,
            evaluation.duration_s, evaluation.energy_kwh, 1.0,
        )
        candidate = cost, evaluation.model
        if best is None or candidate < best:
            best = candidate
    return best


def split_giant_sequence(
    sequence: tuple[str, ...],
    evaluator: RouteEvaluator,
    boxes: dict[str, Box],
    maximum_mass_kg: float,
    maximum_volume_m3: float,
    single_service_only: bool = False,
) -> list[RoutePlan]:
    """用字典序DP把货箱序列切为物理可行架次。"""

    count = len(sequence)
    best: list[tuple[tuple[float, ...], list[RoutePlan]] | None] = [None] * (count + 1)
    best[0] = ((0.0,) * 6, [])
    for start in range(count):
        if best[start] is None:
            continue
        mass = 0.0
        volume = 0.0
        first_service = boxes[sequence[start]].service_id
        for end in range(start + 1, count + 1):
            box = boxes[sequence[end - 1]]
            if single_service_only and box.service_id != first_service:
                break
            mass += box.mass_kg
            volume += box.volume_m3
            if mass > maximum_mass_kg + 1e-8 or volume > maximum_volume_m3 + 1e-10:
                break
            route = route_from_boxes(sequence[start:end], boxes)
            interval = _interval_cost(route, evaluator, boxes)
            if interval is None:
                continue
            cost = tuple(a + b for a, b in zip(best[start][0], interval[0]))
            proposal = cost, best[start][1] + [route]
            if best[end] is None or proposal[0] < best[end][0]:
                best[end] = proposal
    if best[count] is None:
        raise RuntimeError("巨型序列不存在覆盖全部货箱的物理可行切分")
    return best[count][1]


def _split_offending_route(
    schedule: ScheduleResult,
    boxes: dict[str, Box],
) -> list[RoutePlan] | None:
    """按最早迟到硬货箱拆分其架次；普通货箱优先移出紧急架次。"""

    violations: list[tuple[float, float, str, str]] = []
    for trip in schedule.trips:
        for box_id in trip.route.box_ids:
            deadline = hard_deadline_s(boxes[box_id])
            if deadline is None:
                continue
            delivered = trip.delivery_time(box_id)
            if delivered > deadline + 1e-8:
                violations.append((deadline, delivered - deadline, trip.trip_id, box_id))
    if not violations:
        return None
    _, _, trip_id, late_box_id = min(violations)
    target = next(trip for trip in schedule.trips if trip.trip_id == trip_id)
    if len(target.route.box_ids) <= 1:
        return None
    target_ids = list(target.route.box_ids)
    hard_ids = [box_id for box_id in target_ids if hard_deadline_s(boxes[box_id]) is not None]
    ordinary_ids = [box_id for box_id in target_ids if hard_deadline_s(boxes[box_id]) is None]
    if hard_ids and ordinary_ids:
        parts = [tuple(hard_ids), tuple(ordinary_ids)]
    else:
        late_service = boxes[late_box_id].service_id
        urgent_part = tuple(box_id for box_id in target_ids if boxes[box_id].service_id == late_service)
        remainder = tuple(box_id for box_id in target_ids if boxes[box_id].service_id != late_service)
        if not remainder:
            urgent_part = (late_box_id,)
            remainder = tuple(box_id for box_id in target_ids if box_id != late_box_id)
        parts = [urgent_part, remainder]
    replacement = [route_from_boxes(part, boxes) for part in parts if part]
    routes: list[RoutePlan] = []
    for trip in schedule.trips:
        routes.extend(replacement if trip.trip_id == trip_id else [trip.route])
    return routes


def repair_schedule(
    routes: list[RoutePlan],
    evaluator: RouteEvaluator,
    boxes: dict[str, Box],
    units: dict[str, TransportUnit],
    batteries: dict[str, TransportBattery],
    repair_limit: int,
) -> ScheduleResult:
    current_routes = routes
    schedule = schedule_routes(current_routes, evaluator, boxes, units, batteries)
    for _ in range(repair_limit):
        if schedule.objective.hard_violation_count == 0:
            return schedule
        repaired = _split_offending_route(schedule, boxes)
        if repaired is None:
            break
        current_routes = repaired
        schedule = schedule_routes(current_routes, evaluator, boxes, units, batteries)
    return schedule


def _deterministic_orders(
    service_ids: list[str],
    boxes_by_service: dict[str, list[Box]],
    nodes: dict[str, Node],
    legs: dict[tuple[str, str], LegGeometry],
) -> list[tuple[str, tuple[str, ...]]]:
    def min_hard(service: str) -> float:
        values = [hard_deadline_s(box) for box in boxes_by_service[service] if hard_deadline_s(box) is not None]
        return min(values, default=math.inf)

    def min_expected(service: str) -> float:
        return min(
            (box.expected_deadline_s for box in boxes_by_service[service] if box.expected_deadline_s is not None),
            default=math.inf,
        )

    edd = tuple(sorted(service_ids, key=lambda s: (min_hard(s), min_expected(s), s)))
    urgency = tuple(sorted(
        service_ids,
        key=lambda s: (
            min_hard(s),
            -sum(box.priority or 0.0 for box in boxes_by_service[s]),
            s,
        ),
    ))
    polar = tuple(sorted(
        service_ids,
        key=lambda s: (
            math.atan2(nodes[s].lat - nodes["O01"].lat, nodes[s].lon - nodes["O01"].lon),
            legs[("O01", s)].distance_m,
        ),
    ))
    distance = tuple(sorted(service_ids, key=lambda s: (legs[("O01", s)].distance_m, s)))
    remaining = set(service_ids)
    nearest: list[str] = []
    current = "O01"
    while remaining:
        nxt = min(remaining, key=lambda s: (legs[(current, s)].distance_m, min_hard(s), s))
        nearest.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return [
        ("最早硬截止", edd),
        ("紧迫度", urgency),
        ("地理极角", polar),
        ("调度中心距离", distance),
        ("最近邻", tuple(nearest)),
    ]


def _random_neighbor(order: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
    values = list(order)
    operation = rng.randrange(3)
    left, right = sorted(rng.sample(range(len(values)), 2))
    if operation == 0:
        values[left], values[right] = values[right], values[left]
    elif operation == 1:
        values[left:right + 1] = reversed(values[left:right + 1])
    else:
        item = values.pop(right)
        values.insert(left, item)
    return tuple(values)


def solve_q2(
    boxes_by_service: dict[str, list[Box]],
    nodes: dict[str, Node],
    drones: dict[str, TransportDrone],
    units: dict[str, TransportUnit],
    batteries: dict[str, TransportBattery],
    legs: dict[tuple[str, str], LegGeometry],
    seed_base: int = 20260901,
    random_seed_count: int = 20,
    max_iterations: int = 5000,
    stall_iterations: int = 800,
    time_limit_per_seed_s: float = 60.0,
    repair_limit: int = 30,
) -> Q2SearchResult:
    boxes = {box.box_id: box for items in boxes_by_service.values() for box in items}
    evaluator = RouteEvaluator(boxes, drones, legs)
    service_ids = sorted(boxes_by_service)
    maximum_mass = max(drone.max_payload_kg for drone in drones.values())
    maximum_volume = max(drone.volume_m3 for drone in drones.values())
    solution_cache: dict[tuple[str, ...], ScheduleResult] = {}

    def evaluate_order(order: tuple[str, ...]) -> ScheduleResult:
        if order not in solution_cache:
            sequence = boxes_for_service_order(order, boxes_by_service)
            routes = split_giant_sequence(
                sequence, evaluator, boxes, maximum_mass, maximum_volume
            )
            solution_cache[order] = repair_schedule(
                routes, evaluator, boxes, units, batteries, repair_limit
            )
        return solution_cache[order]

    records: list[SearchRecord] = []
    deterministic = _deterministic_orders(service_ids, boxes_by_service, nodes, legs)
    starts: list[tuple[str, int | None, tuple[str, ...]]] = [
        (label, None, order) for label, order in deterministic
    ]
    for offset in range(random_seed_count):
        seed = seed_base + offset
        order = service_ids.copy()
        random.Random(seed).shuffle(order)
        starts.append((f"随机种子{seed}", seed, tuple(order)))

    for start_index, (label, seed, initial_order) in enumerate(starts):
        for preference_index, (preference, preference_order) in enumerate(PREFERENCE_PROFILES):
            base_rng_seed = seed if seed is not None else seed_base - start_index - 1
            rng = random.Random(base_rng_seed + preference_index * 1_000_003)
            start_clock = time.perf_counter()
            current_order = initial_order
            current = evaluate_order(current_order)
            run_best = current
            no_improvement = 0
            iterations = 0
            while iterations < max_iterations and no_improvement < stall_iterations:
                if time.perf_counter() - start_clock >= time_limit_per_seed_s:
                    break
                iterations += 1
                neighbor_order = _random_neighbor(current_order, rng)
                neighbor = evaluate_order(neighbor_order)
                if _preference_key(neighbor, preference_order) < _preference_key(current, preference_order):
                    current_order, current = neighbor_order, neighbor
                if _preference_key(neighbor, preference_order) < _preference_key(run_best, preference_order):
                    run_best = neighbor
                    no_improvement = 0
                else:
                    no_improvement += 1
            elapsed = time.perf_counter() - start_clock
            records.append(SearchRecord(
                label, seed, preference, iterations, elapsed, run_best.objective
            ))

    if not solution_cache:
        raise RuntimeError("问题二搜索未生成任何方案")

    baseline_routes: list[RoutePlan] = []
    for service_id in service_ids:
        sequence = tuple(box.box_id for box in _service_box_order(service_id, boxes_by_service))
        baseline_routes.extend(split_giant_sequence(
            sequence, evaluator, boxes, maximum_mass, maximum_volume,
            single_service_only=True,
        ))
    baseline = repair_schedule(
        baseline_routes, evaluator, boxes, units, batteries, repair_limit
    )
    front = pareto_filter_schedules([*solution_cache.values(), baseline])
    representatives = select_pareto_representatives(front)
    return Q2SearchResult(
        representatives["综合折中"], baseline, tuple(records), front, representatives
    )


__all__ = [
    "PREFERENCE_PROFILES", "Q2SearchResult", "SearchRecord", "boxes_for_service_order",
    "normalized_ideal_distance", "pareto_filter_schedules", "pareto_vector",
    "repair_schedule", "select_pareto_representatives", "solve_q2",
    "split_giant_sequence",
]
