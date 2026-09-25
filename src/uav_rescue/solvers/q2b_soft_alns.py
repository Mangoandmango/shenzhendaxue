"""问题二强化版B的统一、时间窗口径可配置ALNS候选生成器。

软、硬两版共用初解、破坏、修复、接受准则和停止规则；唯一变化是
``DeadlinePolicy`` 决定哪些期望送达时间进入硬约束或软迟到目标。
ALNS中的事件驱动排程仅用于廉价候选评价，最终资源排程由MILP完成。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

from uav_rescue.domain import Box, LegGeometry, Node, TransportBattery, TransportDrone, TransportUnit
from uav_rescue.models.q2_joint import (
    DeadlinePolicy,
    hard_deadline_s,
    is_soft_expected_window,
)
from uav_rescue.models.q2_transport import (
    Objective, RouteEvaluator, RoutePlan, ScheduleResult, TypedRoutePlan,
    route_from_boxes,
)
from uav_rescue.models.q2_transport import (
    schedule_typed_routes_event_driven as schedule_hard_event_driven,
)
from uav_rescue.models.q2_transport_soft import (
    schedule_typed_routes_event_driven as schedule_soft_event_driven,
)
from uav_rescue.solvers.q2_solver import repair_schedule as repair_hard_schedule
from uav_rescue.solvers.q2_soft_solver import (
    normalized_ideal_distance, pareto_filter_schedules, select_pareto_representatives,
    repair_schedule as repair_soft_schedule, split_giant_sequence,
)


PROFILE_ORDERS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("及时性优先", (0, 1, 2, 3)),
    ("完工时间优先", (1, 0, 2, 3)),
    ("能耗优先", (2, 0, 1, 3)),
    ("最少架次优先", (3, 0, 1, 2)),
)
DESTROY_OPERATORS = ("随机货箱", "整架次", "关键末架次", "关键资源链")
REPAIR_OPERATORS = ("贪婪插入", "后悔值插入", "机型重选")


@dataclass(frozen=True)
class ALNSSolution:
    trips: tuple[TypedRoutePlan, ...]


@dataclass(frozen=True)
class RunRecord:
    profile: str
    seed: int
    iterations: int
    elapsed_s: float
    stop_reason: str
    best_objective: Objective


@dataclass(frozen=True)
class Q2BSoftResult:
    best: ScheduleResult
    baseline: ScheduleResult
    records: tuple[RunRecord, ...]
    pareto_front: tuple[ScheduleResult, ...]
    representatives: dict[str, ScheduleResult]


def _solution_key(solution: ALNSSolution) -> tuple[tuple[object, ...], ...]:
    return tuple((trip.model, trip.route.service_sequence, tuple(sorted(trip.route.box_ids))) for trip in solution.trips)


def _vector(schedule: ScheduleResult) -> tuple[float, float, float, float]:
    objective = schedule.objective
    return (
        objective.weighted_soft_tardiness, objective.makespan_s,
        objective.energy_kwh, float(objective.trip_count),
    )


def _key(schedule: ScheduleResult, order: tuple[int, ...]) -> tuple[float, ...]:
    objective = schedule.objective
    vector = _vector(schedule)
    return (
        float(objective.hard_violation_count), objective.hard_tardiness_s,
        *(vector[index] for index in order),
    )


def _choose_model(route: RoutePlan, evaluator: RouteEvaluator, mode: str) -> TypedRoutePlan:
    evaluations = evaluator.evaluate(route.box_ids, route.service_sequence)
    if not evaluations:
        raise RuntimeError(f"路线无可行机型：{route.box_ids}")
    if mode == "energy":
        item = min(evaluations, key=lambda value: (value.energy_kwh, value.duration_s, value.model))
    elif mode == "time":
        item = min(evaluations, key=lambda value: (value.duration_s, value.energy_kwh, value.model))
    else:
        item = min(evaluations, key=lambda value: (value.duration_s / 3600 + value.energy_kwh / 4, value.model))
    return TypedRoutePlan(route, item.model)


def _initial_solutions(
    boxes_by_service: dict[str, list[Box]], boxes: dict[str, Box], evaluator: RouteEvaluator,
    legs: dict[tuple[str, str], LegGeometry], drones: dict[str, TransportDrone],
    units: dict[str, TransportUnit], batteries: dict[str, TransportBattery],
    deadline_fn, repair_schedule_fn,
) -> list[ALNSSolution]:
    services = sorted(boxes_by_service)
    hard = lambda service: min((deadline_fn(box) for box in boxes_by_service[service] if deadline_fn(box) is not None), default=math.inf)
    expected = lambda service: min((box.expected_deadline_s for box in boxes_by_service[service] if box.expected_deadline_s is not None), default=math.inf)
    orders = [
        tuple(sorted(services, key=lambda service: (hard(service), expected(service), service))),
        tuple(sorted(services, key=lambda service: (-sum(box.priority or 0.0 for box in boxes_by_service[service]), expected(service), service))),
        tuple(sorted(services, key=lambda service: (legs[("O01", service)].distance_m, service))),
    ]
    maximum_mass = max(drone.max_payload_kg for drone in drones.values())
    maximum_volume = max(drone.volume_m3 for drone in drones.values())
    result: list[ALNSSolution] = []
    for order in orders:
        sequence = tuple(
            box.box_id for service in order for box in sorted(
                boxes_by_service[service],
                key=lambda box: (deadline_fn(box) or math.inf, box.expected_deadline_s or math.inf, -(box.priority or 0.0), box.box_id),
            )
        )
        routes = split_giant_sequence(sequence, evaluator, boxes, maximum_mass, maximum_volume)
        repaired = repair_schedule_fn(routes, evaluator, boxes, units, batteries, repair_limit=30)
        result.append(ALNSSolution(tuple(
            TypedRoutePlan(trip.route, trip.evaluation.model)
            for trip in sorted(repaired.trips, key=lambda trip: (trip.start_s, trip.trip_id))
        )))
        for mode in ("balanced", "time", "energy"):
            result.append(ALNSSolution(tuple(_choose_model(route, evaluator, mode) for route in routes)))
    return result


def _match(solution: ALNSSolution, schedule: ScheduleResult) -> dict[int, object]:
    unused = set(range(len(solution.trips)))
    mapped: dict[int, object] = {}
    for scheduled in schedule.trips:
        index = next(index for index in sorted(unused) if solution.trips[index].route == scheduled.route and solution.trips[index].model == scheduled.evaluation.model)
        mapped[index] = scheduled
        unused.remove(index)
    return mapped


def _remove(solution: ALNSSolution, remove_ids: set[str], boxes: dict[str, Box]) -> tuple[ALNSSolution, tuple[str, ...]]:
    trips: list[TypedRoutePlan] = []
    actual: list[str] = []
    for trip in solution.trips:
        kept = tuple(box_id for box_id in trip.route.box_ids if box_id not in remove_ids)
        actual.extend(box_id for box_id in trip.route.box_ids if box_id in remove_ids)
        if not kept:
            continue
        services = {boxes[box_id].service_id for box_id in kept}
        sequence = tuple(service for service in trip.route.service_sequence if service in services)
        trips.append(TypedRoutePlan(RoutePlan(tuple(sorted(kept)), sequence), trip.model))
    return ALNSSolution(tuple(trips)), tuple(sorted(actual))


def _destroy(name: str, solution: ALNSSolution, schedule: ScheduleResult, boxes: dict[str, Box], rng: random.Random, count: int) -> tuple[ALNSSolution, tuple[str, ...]]:
    all_ids = [box_id for trip in solution.trips for box_id in trip.route.box_ids]
    count = min(max(1, count), len(all_ids))
    mapping = _match(solution, schedule)
    if name == "随机货箱":
        removed = set(rng.sample(all_ids, count))
    elif name == "整架次":
        indexes = list(range(len(solution.trips)))
        rng.shuffle(indexes)
        removed = set()
        for index in indexes:
            removed.update(solution.trips[index].route.box_ids)
            if len(removed) >= count:
                break
    else:
        last = max(mapping, key=lambda index: (mapping[index].return_s, index))
        if name == "关键末架次":
            indexes = [last]
        else:
            unit = mapping[last].unit_id
            indexes = sorted((index for index in mapping if mapping[index].unit_id == unit), key=lambda index: -mapping[index].return_s)
        removed = set()
        for index in indexes:
            removed.update(solution.trips[index].route.box_ids)
            if len(removed) >= count:
                break
    return _remove(solution, removed, boxes)


@dataclass(frozen=True)
class _Option:
    proxy: tuple[float, ...]
    index: int | None
    trip: TypedRoutePlan


def _options(
    box_id: str, solution: ALNSSolution, boxes: dict[str, Box],
    evaluator: RouteEvaluator, repair: str, policy: DeadlinePolicy,
) -> list[_Option]:
    box = boxes[box_id]
    values: list[_Option] = []
    for index, old in enumerate(solution.trips[:10]):
        sequences = (old.route.service_sequence,) if box.service_id in old.route.service_sequence else tuple(old.route.service_sequence[:pos] + (box.service_id,) + old.route.service_sequence[pos:] for pos in range(len(old.route.service_sequence) + 1))
        models = tuple(evaluator.drones) if repair == "机型重选" else (old.model,)
        for sequence in sequences:
            route = RoutePlan(tuple(sorted((*old.route.box_ids, box_id))), sequence)
            for item in evaluator.evaluate(route.box_ids, route.service_sequence):
                if item.model not in models:
                    continue
                hard_late = sum(
                    int(
                        (deadline := hard_deadline_s(boxes[value], policy)) is not None
                        and item.delivery_offset(value) > deadline
                    )
                    for value in route.box_ids
                )
                soft_late = sum(
                    (boxes[value].priority or 1.0)
                    * max(
                        0.0,
                        item.delivery_offset(value)
                        - (boxes[value].expected_deadline_s or math.inf),
                    )
                    for value in route.box_ids
                    if is_soft_expected_window(boxes[value], policy)
                )
                values.append(_Option((hard_late, soft_late, item.duration_s, item.energy_kwh), index, TypedRoutePlan(route, item.model)))
    route = route_from_boxes((box_id,), boxes)
    for item in evaluator.evaluate(route.box_ids, route.service_sequence):
        values.append(_Option((0.0, 0.0, item.duration_s, item.energy_kwh + 1.0), None, TypedRoutePlan(route, item.model)))
    return sorted(values, key=lambda value: (*value.proxy, value.trip.model, value.trip.route.service_sequence, value.trip.route.box_ids))


def _apply(solution: ALNSSolution, option: _Option) -> ALNSSolution:
    trips = list(solution.trips)
    if option.index is None:
        trips.append(option.trip)
    else:
        trips[option.index] = option.trip
    return ALNSSolution(tuple(trips))


def _partial_key(
    schedule: ScheduleResult, boxes: dict[str, Box], order: tuple[int, ...],
    policy: DeadlinePolicy,
) -> tuple[float, ...]:
    hard_count = hard_tardiness = soft = 0.0
    for trip in schedule.trips:
        for box_id in trip.route.box_ids:
            box = boxes[box_id]
            deadline = hard_deadline_s(box, policy)
            late = max(
                0.0,
                trip.delivery_time(box_id)
                - (deadline if deadline is not None else (box.expected_deadline_s or math.inf)),
            )
            if deadline is not None:
                hard_count += int(late > 1e-8)
                hard_tardiness += late
            elif is_soft_expected_window(box, policy):
                soft += (box.priority or 1.0) * late
    vector = (soft, schedule.objective.makespan_s, schedule.objective.energy_kwh, float(schedule.objective.trip_count))
    return (hard_count, hard_tardiness, *(vector[index] for index in order))


def _repair(
    solution: ALNSSolution, removed: tuple[str, ...], boxes: dict[str, Box],
    evaluator: RouteEvaluator, decode, order: tuple[int, ...], repair: str,
    top_k: int, policy: DeadlinePolicy,
) -> ALNSSolution:
    pending = sorted(
        removed,
        key=lambda box_id: (
            hard_deadline_s(boxes[box_id], policy) or math.inf,
            boxes[box_id].expected_deadline_s or math.inf,
            -(boxes[box_id].priority or 0.0),
            box_id,
        ),
    )
    while pending:
        if repair == "后悔值插入":
            choices = []
            for box_id in pending:
                options = _options(box_id, solution, boxes, evaluator, repair, policy)
                regret = sum(
                    later - first
                    for later, first in zip(
                        options[min(1, len(options) - 1)].proxy,
                        options[0].proxy,
                    )
                )
                choices.append((regret, box_id, options))
            _, box_id, candidates = max(choices)
        else:
            box_id = pending[0]
            candidates = _options(box_id, solution, boxes, evaluator, repair, policy)
        unique: list[_Option] = []
        seen: set[tuple[object, ...]] = set()
        for option in candidates:
            marker = (option.index, option.trip.model, option.trip.route.service_sequence, option.trip.route.box_ids)
            if marker not in seen:
                seen.add(marker)
                unique.append(option)
            if len(unique) >= top_k:
                break
        selected = min(
            unique,
            key=lambda option: (
                _partial_key(
                    decode(_apply(solution, option)), boxes, order, policy
                ),
                option.proxy,
            ),
        )
        solution = _apply(solution, selected)
        pending.remove(box_id)
    return solution


def solve_q2_joint_outer(
    boxes_by_service: dict[str, list[Box]], nodes: dict[str, Node], drones: dict[str, TransportDrone],
    units: dict[str, TransportUnit], batteries: dict[str, TransportBattery], legs: dict[tuple[str, str], LegGeometry],
    seed_base: int = 20260924, seeds_per_profile: int = 8, max_iterations: int = 1200,
    stall_iterations: int = 350, time_limit_per_run_s: float = 90.0, top_k: int = 3,
    policy: DeadlinePolicy = DeadlinePolicy.ORDINARY_SOFT,
) -> Q2BSoftResult:
    del nodes
    boxes = {box.box_id: box for group in boxes_by_service.values() for box in group}
    evaluator = RouteEvaluator(boxes, drones, legs)
    deadline_fn = lambda box: hard_deadline_s(box, policy)
    repair_schedule_fn = (
        repair_soft_schedule
        if policy == DeadlinePolicy.ORDINARY_SOFT
        else repair_hard_schedule
    )
    event_decoder = (
        schedule_soft_event_driven
        if policy == DeadlinePolicy.ORDINARY_SOFT
        else schedule_hard_event_driven
    )
    initial = _initial_solutions(
        boxes_by_service, boxes, evaluator, legs, drones, units, batteries,
        deadline_fn, repair_schedule_fn,
    )
    cache: dict[tuple[tuple[object, ...], ...], ScheduleResult] = {}
    def decode(solution: ALNSSolution) -> ScheduleResult:
        key = _solution_key(solution)
        if key not in cache:
            cache[key] = event_decoder(
                solution.trips, evaluator, boxes, units, batteries
            )
        return cache[key]
    schedules = [(solution, decode(solution)) for solution in initial]
    baseline = min((schedule for _, schedule in schedules), key=lambda schedule: _key(schedule, PROFILE_ORDERS[0][1]))
    archive: dict[tuple[float, ...], ScheduleResult] = {}
    def archive_add(schedule: ScheduleResult) -> None:
        if schedule.objective.hard_violation_count == 0:
            objective = schedule.objective
            archive.setdefault(tuple(round(value, 8) for value in (objective.weighted_soft_tardiness, objective.makespan_s, objective.energy_kwh, objective.trip_count)), schedule)
    for _, schedule in schedules:
        archive_add(schedule)
    records: list[RunRecord] = []
    for profile_index, (profile, order) in enumerate(PROFILE_ORDERS):
        for offset in range(seeds_per_profile):
            seed = seed_base + profile_index * 100_003 + offset
            rng = random.Random(seed)
            started = time.perf_counter()
            current_solution, current = min(schedules, key=lambda item: _key(item[1], order))
            best = current
            no_improvement = iteration = 0
            temperature = 0.08
            destroy_weights = {name: 1.0 for name in DESTROY_OPERATORS}
            repair_weights = {name: 1.0 for name in REPAIR_OPERATORS}
            stop_reason = "最大迭代数"
            while iteration < max_iterations:
                if time.perf_counter() - started >= time_limit_per_run_s:
                    stop_reason = "运行时间上限"; break
                if no_improvement >= stall_iterations:
                    stop_reason = "连续无改进上限"; break
                iteration += 1
                destroy = rng.choices(tuple(destroy_weights), weights=tuple(destroy_weights.values()), k=1)[0]
                repair = rng.choices(tuple(repair_weights), weights=tuple(repair_weights.values()), k=1)[0]
                partial, removed = _destroy(destroy, current_solution, current, boxes, rng, rng.randint(2, 8))
                candidate_solution = _repair(
                    partial, removed, boxes, evaluator, decode, order,
                    repair, top_k, policy,
                )
                candidate = decode(candidate_solution)
                current_key, candidate_key = _key(current, order), _key(candidate, order)
                accepted = candidate_key < current_key
                if not accepted and candidate_key[:2] == current_key[:2]:
                    delta = sum((candidate_key[index] - current_key[index]) / max(abs(current_key[index]), 1.0) for index in range(2, len(current_key)))
                    accepted = delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-9))
                reward = 0.2
                if accepted:
                    current_solution, current = candidate_solution, candidate
                    archive_add(current)
                    reward = 1.0
                if candidate_key < _key(best, order):
                    best = candidate; no_improvement = 0; reward = 6.0
                else:
                    no_improvement += 1
                destroy_weights[destroy] = 0.9 * destroy_weights[destroy] + 0.1 * reward
                repair_weights[repair] = 0.9 * repair_weights[repair] + 0.1 * reward
                temperature *= 0.997
            archive_add(best)
            records.append(RunRecord(profile, seed, iteration, time.perf_counter() - started, stop_reason, best.objective))
    if not archive:
        raise RuntimeError("未获得满足当前硬时间窗口径的可行方案")
    front = pareto_filter_schedules(list(archive.values()))
    representatives = select_pareto_representatives(front)
    best = representatives["综合折中"]
    return Q2BSoftResult(best, baseline, tuple(records), front, representatives)


def solve_q2b_soft(
    boxes_by_service: dict[str, list[Box]], nodes: dict[str, Node],
    drones: dict[str, TransportDrone], units: dict[str, TransportUnit],
    batteries: dict[str, TransportBattery],
    legs: dict[tuple[str, str], LegGeometry], seed_base: int = 20260924,
    seeds_per_profile: int = 8, max_iterations: int = 1200,
    stall_iterations: int = 350, time_limit_per_run_s: float = 90.0,
    top_k: int = 3,
) -> Q2BSoftResult:
    """兼容旧入口；等价于普通物资软时间窗口径。"""

    return solve_q2_joint_outer(
        boxes_by_service, nodes, drones, units, batteries, legs,
        seed_base=seed_base,
        seeds_per_profile=seeds_per_profile,
        max_iterations=max_iterations,
        stall_iterations=stall_iterations,
        time_limit_per_run_s=time_limit_per_run_s,
        top_k=top_k,
        policy=DeadlinePolicy.ORDINARY_SOFT,
    )
