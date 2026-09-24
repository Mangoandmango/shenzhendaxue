"""问题二方案B：ALNS与事件驱动资源排程的一体化求解器。"""

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
    TypedRoutePlan,
    hard_deadline_s,
    route_from_boxes,
    schedule_typed_routes_event_driven,
)
from uav_rescue.solvers.q2_solver import (
    normalized_ideal_distance,
    pareto_filter_schedules,
    repair_schedule,
    select_pareto_representatives,
    split_giant_sequence,
)


PROFILE_WEIGHTS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("完工时间优先", (0.60, 0.20, 0.20)),
    ("均衡", (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
    ("资源节约", (0.20, 0.45, 0.35)),
)

DESTROY_OPERATORS = (
    "random_box",
    "whole_trip",
    "critical_deadline",
    "related",
    "high_energy",
    "resource_bottleneck",
    "route_segment",
)

REPAIR_OPERATORS = (
    "greedy",
    "regret2",
    "regret3",
    "deadline_first",
    "energy_margin",
    "model_reselect",
)


@dataclass(frozen=True)
class ALNSSolution:
    trips: tuple[TypedRoutePlan, ...]


@dataclass(frozen=True)
class ALNSRunRecord:
    profile: str
    seed: int
    iterations: int
    elapsed_s: float
    feasible_iteration: int | None
    accepted_moves: int
    improving_moves: int
    best_objective: Objective
    stop_reason: str


@dataclass(frozen=True)
class ConvergenceRecord:
    profile: str
    seed: int
    iteration: int
    elapsed_s: float
    temperature: float
    destroy_operator: str
    repair_operator: str
    accepted: bool
    current_objective: Objective
    best_objective: Objective


@dataclass(frozen=True)
class OperatorRecord:
    profile: str
    seed: int
    kind: str
    operator: str
    final_weight: float
    uses: int
    score: float


@dataclass(frozen=True)
class Q2ALNSResult:
    best: ScheduleResult
    baseline: ScheduleResult
    greedy_initial: ScheduleResult
    records: tuple[ALNSRunRecord, ...]
    convergence: tuple[ConvergenceRecord, ...]
    operator_records: tuple[OperatorRecord, ...]
    pareto_front: tuple[ScheduleResult, ...]
    representatives: dict[str, ScheduleResult]


def _solution_key(solution: ALNSSolution) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (trip.model, trip.route.service_sequence, tuple(sorted(trip.route.box_ids)))
        for trip in solution.trips
    )


def _objective_tuple(objective: Objective) -> tuple[float, ...]:
    return (
        float(objective.hard_violation_count),
        objective.hard_tardiness_s,
        objective.makespan_s,
        objective.energy_kwh,
        float(objective.trip_count),
    )


def _soft_score(
    schedule: ScheduleResult,
    weights: tuple[float, float, float],
    scales: tuple[float, float, float],
) -> float:
    objective = schedule.objective
    values = (objective.makespan_s, objective.energy_kwh, float(objective.trip_count))
    return sum(
        weight * value / max(scale, 1e-9)
        for weight, value, scale in zip(weights, values, scales)
    )


def _quality_key(
    schedule: ScheduleResult,
    weights: tuple[float, float, float],
    scales: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        float(schedule.objective.hard_violation_count),
        schedule.objective.hard_tardiness_s,
        _soft_score(schedule, weights, scales),
    )


def _roulette(weights: dict[str, float], rng: random.Random) -> str:
    total = sum(max(value, 1e-12) for value in weights.values())
    draw = rng.random() * total
    cumulative = 0.0
    for name in sorted(weights):
        cumulative += max(weights[name], 1e-12)
        if draw <= cumulative:
            return name
    return sorted(weights)[-1]


def _route_evaluation(
    trip: TypedRoutePlan,
    evaluator: RouteEvaluator,
):
    return next(
        (
            item for item in evaluator.evaluate(
                trip.route.box_ids, trip.route.service_sequence
            ) if item.model == trip.model
        ),
        None,
    )


def _choose_model(
    route: RoutePlan,
    evaluator: RouteEvaluator,
    mode: str,
) -> TypedRoutePlan:
    evaluations = evaluator.evaluate(route.box_ids, route.service_sequence)
    if not evaluations:
        raise RuntimeError(f"初始路线无可行机型：{route.box_ids}")
    if mode == "fast":
        chosen = min(evaluations, key=lambda item: (item.duration_s, item.energy_kwh, item.model))
    elif mode == "energy":
        chosen = min(evaluations, key=lambda item: (item.energy_kwh, item.duration_s, item.model))
    else:
        chosen = min(
            evaluations,
            key=lambda item: (
                item.duration_s / 3600.0 + item.energy_kwh / 4.0,
                item.duration_s,
                item.model,
            ),
        )
    return TypedRoutePlan(route, chosen.model)


def _service_orders(
    boxes_by_service: dict[str, list[Box]],
    legs: dict[tuple[str, str], LegGeometry],
) -> tuple[tuple[str, ...], ...]:
    services = sorted(boxes_by_service)

    def deadline(service: str) -> float:
        return min(
            hard_deadline_s(box) or math.inf for box in boxes_by_service[service]
        )

    earliest = tuple(sorted(services, key=lambda item: (deadline(item), item)))
    distance = tuple(sorted(services, key=lambda item: (legs[("O01", item)].distance_m, item)))
    remaining = set(services)
    nearest: list[str] = []
    current = "O01"
    while remaining:
        nxt = min(
            remaining,
            key=lambda item: (legs[(current, item)].distance_m, deadline(item), item),
        )
        nearest.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return earliest, distance, tuple(nearest)


def _ordered_boxes(
    order: tuple[str, ...], boxes_by_service: dict[str, list[Box]]
) -> tuple[str, ...]:
    return tuple(
        box.box_id
        for service in order
        for box in sorted(
            boxes_by_service[service],
            key=lambda item: (
                hard_deadline_s(item) or math.inf,
                -(item.priority or 0.0),
                item.box_id,
            ),
        )
    )


def _initial_solutions(
    boxes_by_service: dict[str, list[Box]],
    boxes: dict[str, Box],
    drones: dict[str, TransportDrone],
    evaluator: RouteEvaluator,
    legs: dict[tuple[str, str], LegGeometry],
    units: dict[str, TransportUnit],
    batteries: dict[str, TransportBattery],
) -> tuple[list[ALNSSolution], ALNSSolution]:
    max_mass = max(item.max_payload_kg for item in drones.values())
    max_volume = max(item.volume_m3 for item in drones.values())
    result: list[ALNSSolution] = []
    for order in _service_orders(boxes_by_service, legs):
        routes = split_giant_sequence(
            _ordered_boxes(order, boxes_by_service),
            evaluator,
            boxes,
            max_mass,
            max_volume,
        )
        repaired = repair_schedule(
            routes, evaluator, boxes, units, batteries, repair_limit=30
        )
        result.append(ALNSSolution(tuple(
            TypedRoutePlan(trip.route, trip.evaluation.model)
            for trip in sorted(repaired.trips, key=lambda item: (item.start_s, item.trip_id))
        )))
        for mode in ("fast", "balanced", "energy"):
            result.append(ALNSSolution(tuple(
                _choose_model(route, evaluator, mode) for route in routes
            )))

    direct_routes: list[RoutePlan] = []
    for service in sorted(boxes_by_service):
        sequence = _ordered_boxes((service,), boxes_by_service)
        direct_routes.extend(split_giant_sequence(
            sequence,
            evaluator,
            boxes,
            max_mass,
            max_volume,
            single_service_only=True,
        ))
    direct_schedule = repair_schedule(
        direct_routes, evaluator, boxes, units, batteries, repair_limit=30
    )
    direct = ALNSSolution(tuple(
        TypedRoutePlan(trip.route, trip.evaluation.model)
        for trip in sorted(direct_schedule.trips, key=lambda item: (item.start_s, item.trip_id))
    ))
    return result, direct


def _match_scheduled_trips(
    solution: ALNSSolution, schedule: ScheduleResult
) -> dict[int, object]:
    unused = set(range(len(solution.trips)))
    result: dict[int, object] = {}
    for scheduled in schedule.trips:
        match = next(
            index for index in sorted(unused)
            if solution.trips[index].model == scheduled.evaluation.model
            and solution.trips[index].route == scheduled.route
        )
        result[match] = scheduled
        unused.remove(match)
    return result


def _remove_boxes(
    solution: ALNSSolution,
    removed_box_ids: set[str],
    boxes: dict[str, Box],
) -> tuple[ALNSSolution, tuple[str, ...]]:
    trips: list[TypedRoutePlan] = []
    actual_removed: list[str] = []
    for trip in solution.trips:
        kept = tuple(
            box_id for box_id in trip.route.box_ids if box_id not in removed_box_ids
        )
        actual_removed.extend(
            box_id for box_id in trip.route.box_ids if box_id in removed_box_ids
        )
        if not kept:
            continue
        kept_services = {boxes[box_id].service_id for box_id in kept}
        sequence = tuple(
            service for service in trip.route.service_sequence if service in kept_services
        )
        trips.append(TypedRoutePlan(RoutePlan(tuple(sorted(kept)), sequence), trip.model))
    return ALNSSolution(tuple(trips)), tuple(sorted(actual_removed))


def _destroy(
    operator: str,
    solution: ALNSSolution,
    schedule: ScheduleResult,
    boxes: dict[str, Box],
    rng: random.Random,
    removal_count: int,
) -> tuple[ALNSSolution, tuple[str, ...]]:
    all_ids = [box_id for trip in solution.trips for box_id in trip.route.box_ids]
    removal_count = min(max(1, removal_count), len(all_ids))
    scheduled_by_index = _match_scheduled_trips(solution, schedule)

    if operator == "random_box":
        removed = set(rng.sample(all_ids, removal_count))
    elif operator == "whole_trip":
        indexes = list(range(len(solution.trips)))
        rng.shuffle(indexes)
        removed = set()
        for index in indexes:
            removed.update(solution.trips[index].route.box_ids)
            if len(removed) >= removal_count:
                break
    elif operator == "critical_deadline":
        slack: list[tuple[float, str]] = []
        for index, trip in enumerate(solution.trips):
            scheduled = scheduled_by_index[index]
            for box_id in trip.route.box_ids:
                deadline = hard_deadline_s(boxes[box_id]) or math.inf
                slack.append((deadline - scheduled.delivery_time(box_id), box_id))
        removed = {box_id for _, box_id in sorted(slack)[:removal_count]}
    elif operator == "related":
        seed_id = rng.choice(all_ids)
        seed_box = boxes[seed_id]
        seed_deadline = hard_deadline_s(seed_box) or 0.0
        ranked = sorted(
            all_ids,
            key=lambda box_id: (
                boxes[box_id].service_id != seed_box.service_id,
                boxes[box_id].category != seed_box.category,
                abs((hard_deadline_s(boxes[box_id]) or 0.0) - seed_deadline),
                box_id,
            ),
        )
        removed = set(ranked[:removal_count])
    elif operator == "high_energy":
        ranked_indexes = sorted(
            range(len(solution.trips)),
            key=lambda index: (
                -scheduled_by_index[index].evaluation.energy_kwh
                / len(solution.trips[index].route.box_ids),
                index,
            ),
        )
        removed = set()
        for index in ranked_indexes:
            removed.update(solution.trips[index].route.box_ids)
            if len(removed) >= removal_count:
                break
    elif operator == "resource_bottleneck":
        ranked_indexes = sorted(
            range(len(solution.trips)),
            key=lambda index: (
                -scheduled_by_index[index].start_s,
                -scheduled_by_index[index].recharge_s,
                -scheduled_by_index[index].return_s,
            ),
        )
        removed = set()
        for index in ranked_indexes:
            removed.update(solution.trips[index].route.box_ids)
            if len(removed) >= removal_count:
                break
    elif operator == "route_segment":
        trip = rng.choice(solution.trips)
        services = list(trip.route.service_sequence)
        if len(services) == 1:
            candidates = list(trip.route.box_ids)
            removed = set(rng.sample(candidates, min(removal_count, len(candidates))))
        else:
            start = rng.randrange(len(services))
            length = rng.randint(1, len(services) - start)
            selected = set(services[start:start + length])
            removed = {
                box_id for box_id in trip.route.box_ids
                if boxes[box_id].service_id in selected
            }
    else:
        raise ValueError(f"未知破坏算子：{operator}")

    if len(removed) > removal_count and operator not in {"whole_trip", "high_energy", "resource_bottleneck"}:
        removed = set(rng.sample(sorted(removed), removal_count))
    return _remove_boxes(solution, removed, boxes)


@dataclass(frozen=True)
class _InsertionOption:
    score: float
    trip_index: int | None
    trip: TypedRoutePlan


def _insertion_options(
    box_id: str,
    solution: ALNSSolution,
    boxes: dict[str, Box],
    evaluator: RouteEvaluator,
    weights: tuple[float, float, float],
    scales: tuple[float, float, float],
    operator: str,
    max_candidate_trips: int = 8,
) -> list[_InsertionOption]:
    box = boxes[box_id]
    reselect = operator == "model_reselect"
    ranked_indexes = sorted(
        range(len(solution.trips)),
        key=lambda index: (
            box.service_id not in solution.trips[index].route.service_sequence,
            min(
                evaluator.legs[(service, box.service_id)].distance_m
                if service != box.service_id else 0.0
                for service in solution.trips[index].route.service_sequence
            ),
            len(solution.trips[index].route.box_ids),
            index,
        ),
    )[:max_candidate_trips]
    options: list[_InsertionOption] = []

    for index in ranked_indexes:
        old_trip = solution.trips[index]
        old_evaluation = _route_evaluation(old_trip, evaluator)
        if old_evaluation is None:
            continue
        if box.service_id in old_trip.route.service_sequence:
            sequences = (old_trip.route.service_sequence,)
        else:
            sequences = tuple(
                old_trip.route.service_sequence[:position]
                + (box.service_id,)
                + old_trip.route.service_sequence[position:]
                for position in range(len(old_trip.route.service_sequence) + 1)
            )
        models = sorted(evaluator.drones) if reselect else [old_trip.model]
        for sequence in sequences:
            route = RoutePlan(
                tuple(sorted((*old_trip.route.box_ids, box_id))),
                sequence,
            )
            evaluations = {
                item.model: item
                for item in evaluator.evaluate(route.box_ids, route.service_sequence)
            }
            for model in models:
                evaluation = evaluations.get(model)
                if evaluation is None:
                    continue
                late_count = 0
                tardiness = 0.0
                for candidate_id, offset in evaluation.delivery_offsets_s:
                    deadline = hard_deadline_s(boxes[candidate_id])
                    if deadline is not None and offset > deadline:
                        late_count += 1
                        tardiness += offset - deadline
                delta_duration = max(0.0, evaluation.duration_s - old_evaluation.duration_s)
                delta_energy = max(0.0, evaluation.energy_kwh - old_evaluation.energy_kwh)
                score = (
                    1e6 * late_count
                    + 100.0 * tardiness
                    + weights[0] * delta_duration / max(scales[0], 1e-9)
                    + weights[1] * delta_energy / max(scales[1], 1e-9)
                )
                if operator == "deadline_first":
                    score += 0.01 * evaluation.delivery_offset(box_id)
                elif operator == "energy_margin":
                    score += 10.0 * (1.0 - evaluation.end_soc)
                options.append(_InsertionOption(
                    score,
                    index,
                    TypedRoutePlan(route, model),
                ))

    route = route_from_boxes((box_id,), boxes)
    for evaluation in evaluator.evaluate(route.box_ids, route.service_sequence):
        score = (
            weights[0] * evaluation.duration_s / max(scales[0], 1e-9)
            + weights[1] * evaluation.energy_kwh / max(scales[1], 1e-9)
            + weights[2] / max(scales[2], 1e-9)
        )
        if operator == "energy_margin":
            score += 10.0 * (1.0 - evaluation.end_soc)
        options.append(_InsertionOption(
            score,
            None,
            TypedRoutePlan(route, evaluation.model),
        ))
    return sorted(
        options,
        key=lambda item: (
            item.score,
            item.trip.model,
            item.trip.route.service_sequence,
            item.trip.route.box_ids,
            -1 if item.trip_index is None else item.trip_index,
        ),
    )


def _apply_insertion(solution: ALNSSolution, option: _InsertionOption) -> ALNSSolution:
    trips = list(solution.trips)
    if option.trip_index is None:
        trips.append(option.trip)
    else:
        trips[option.trip_index] = option.trip
    return ALNSSolution(tuple(trips))


def _repair(
    operator: str,
    partial: ALNSSolution,
    removed: tuple[str, ...],
    boxes: dict[str, Box],
    evaluator: RouteEvaluator,
    weights: tuple[float, float, float],
    scales: tuple[float, float, float],
    rng: random.Random,
) -> ALNSSolution:
    solution = partial
    pending = list(removed)
    if operator == "deadline_first":
        pending.sort(key=lambda box_id: (hard_deadline_s(boxes[box_id]) or math.inf, box_id))
    elif operator == "energy_margin":
        pending.sort(key=lambda box_id: (-boxes[box_id].mass_kg, box_id))
    elif operator in {"greedy", "model_reselect"}:
        rng.shuffle(pending)

    while pending:
        if operator in {"regret2", "regret3"}:
            regret_k = 2 if operator == "regret2" else 3
            choices: list[tuple[float, float, str, _InsertionOption]] = []
            for box_id in pending:
                options = _insertion_options(
                    box_id, solution, boxes, evaluator, weights, scales, operator
                )
                if not options:
                    raise RuntimeError(f"货箱无可行插入位置：{box_id}")
                kth = options[min(regret_k - 1, len(options) - 1)].score
                choices.append((kth - options[0].score, -options[0].score, box_id, options[0]))
            _, _, chosen_id, chosen_option = max(choices)
        else:
            chosen_id = pending[0]
            options = _insertion_options(
                chosen_id, solution, boxes, evaluator, weights, scales, operator
            )
            if not options:
                raise RuntimeError(f"货箱无可行插入位置：{chosen_id}")
            chosen_option = options[0]
        solution = _apply_insertion(solution, chosen_option)
        pending.remove(chosen_id)

    if operator == "model_reselect":
        solution = ALNSSolution(tuple(
            _choose_model(trip.route, evaluator, "balanced") for trip in solution.trips
        ))
    return solution


def _archive_add(
    archive: dict[tuple[float, ...], ScheduleResult], schedule: ScheduleResult
) -> None:
    if schedule.objective.hard_violation_count != 0:
        return
    key = tuple(round(value, 8) for value in _objective_tuple(schedule.objective))
    archive.setdefault(key, schedule)


def solve_q2_alns(
    boxes_by_service: dict[str, list[Box]],
    nodes: dict[str, Node],
    drones: dict[str, TransportDrone],
    units: dict[str, TransportUnit],
    batteries: dict[str, TransportBattery],
    legs: dict[tuple[str, str], LegGeometry],
    seed_base: int = 20260901,
    seeds_per_profile: int = 8,
    max_iterations: int = 1200,
    stall_iterations: int = 350,
    time_limit_per_run_s: float = 90.0,
    min_removal_count: int = 2,
    max_removal_count: int = 8,
    reaction_factor: float = 0.20,
    update_period: int = 50,
    initial_temperature: float = 0.05,
    cooling_rate: float = 0.997,
    log_period: int = 20,
    warm_start: tuple[TypedRoutePlan, ...] | None = None,
    enabled_destroy_operators: tuple[str, ...] | None = None,
    enabled_repair_operators: tuple[str, ...] | None = None,
) -> Q2ALNSResult:
    del nodes  # 节点几何已经统一封装在有向航段缓存中。
    boxes = {box.box_id: box for values in boxes_by_service.values() for box in values}
    destroy_names = enabled_destroy_operators or DESTROY_OPERATORS
    repair_names = enabled_repair_operators or REPAIR_OPERATORS
    if not destroy_names or not repair_names:
        raise ValueError("破坏算子集和修复算子集均不能为空")
    if any(name not in DESTROY_OPERATORS for name in destroy_names):
        raise ValueError("启用了未知破坏算子")
    if any(name not in REPAIR_OPERATORS for name in repair_names):
        raise ValueError("启用了未知修复算子")
    evaluator = RouteEvaluator(boxes, drones, legs)
    initial_solutions, direct_solution = _initial_solutions(
        boxes_by_service, boxes, drones, evaluator, legs, units, batteries
    )
    if warm_start:
        initial_solutions.insert(0, ALNSSolution(warm_start))
    schedule_cache: dict[tuple[tuple[object, ...], ...], ScheduleResult] = {}

    def decode(solution: ALNSSolution) -> ScheduleResult:
        key = _solution_key(solution)
        if key not in schedule_cache:
            schedule_cache[key] = schedule_typed_routes_event_driven(
                solution.trips, evaluator, boxes, units, batteries
            )
        return schedule_cache[key]

    baseline = decode(direct_solution)
    initial_schedules = [(solution, decode(solution)) for solution in initial_solutions]
    greedy_initial = min(
        (schedule for _, schedule in initial_schedules),
        key=lambda item: _objective_tuple(item.objective),
    )
    scales = (
        max(baseline.objective.makespan_s, 1.0),
        max(baseline.objective.energy_kwh, 1.0),
        max(float(baseline.objective.trip_count), 1.0),
    )
    archive: dict[tuple[float, ...], ScheduleResult] = {}
    _archive_add(archive, baseline)
    for _, schedule in initial_schedules:
        _archive_add(archive, schedule)

    run_records: list[ALNSRunRecord] = []
    convergence: list[ConvergenceRecord] = []
    operator_records: list[OperatorRecord] = []

    for profile_index, (profile, weights) in enumerate(PROFILE_WEIGHTS):
        for seed_offset in range(seeds_per_profile):
            seed = seed_base + profile_index * 100_003 + seed_offset
            rng = random.Random(seed)
            start_clock = time.perf_counter()
            current_solution, current = min(
                initial_schedules,
                key=lambda item: _quality_key(item[1], weights, scales),
            )
            best_solution, best = current_solution, current
            feasible_iteration = 0 if current.objective.hard_violation_count == 0 else None
            temperature = initial_temperature
            destroy_weights = {name: 1.0 for name in destroy_names}
            repair_weights = {name: 1.0 for name in repair_names}
            destroy_score = {name: 0.0 for name in destroy_names}
            repair_score = {name: 0.0 for name in repair_names}
            destroy_uses = {name: 0 for name in destroy_names}
            repair_uses = {name: 0 for name in repair_names}
            destroy_total_uses = {name: 0 for name in destroy_names}
            repair_total_uses = {name: 0 for name in repair_names}
            destroy_total_score = {name: 0.0 for name in destroy_names}
            repair_total_score = {name: 0.0 for name in repair_names}
            accepted_moves = 0
            improving_moves = 0
            no_improvement = 0
            stop_reason = "最大迭代数"
            iteration = 0

            while iteration < max_iterations:
                elapsed = time.perf_counter() - start_clock
                if elapsed >= time_limit_per_run_s:
                    stop_reason = "运行时间上限"
                    break
                if no_improvement >= stall_iterations:
                    stop_reason = "连续无改进上限"
                    break
                iteration += 1
                destroy_name = _roulette(destroy_weights, rng)
                repair_name = _roulette(repair_weights, rng)
                removal_count = rng.randint(min_removal_count, max_removal_count)
                partial, removed = _destroy(
                    destroy_name,
                    current_solution,
                    current,
                    boxes,
                    rng,
                    removal_count,
                )
                candidate_solution = _repair(
                    repair_name,
                    partial,
                    removed,
                    boxes,
                    evaluator,
                    weights,
                    scales,
                    rng,
                )
                candidate = decode(candidate_solution)
                current_key = _quality_key(current, weights, scales)
                candidate_key = _quality_key(candidate, weights, scales)
                accepted = False
                reward = 0.2
                if candidate_key < current_key:
                    accepted = True
                    reward = 4.0
                elif candidate_key[0] == current_key[0]:
                    delta = (
                        (candidate_key[1] - current_key[1]) / max(scales[0], 1.0)
                        + candidate_key[2] - current_key[2]
                    )
                    if delta <= 0.0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
                        accepted = True
                        reward = 1.0
                if accepted:
                    current_solution, current = candidate_solution, candidate
                    accepted_moves += 1
                    _archive_add(archive, current)
                if _quality_key(candidate, weights, scales) < _quality_key(best, weights, scales):
                    best_solution, best = candidate_solution, candidate
                    improving_moves += 1
                    no_improvement = 0
                    reward = 8.0
                else:
                    no_improvement += 1
                if feasible_iteration is None and candidate.objective.hard_violation_count == 0:
                    feasible_iteration = iteration

                destroy_uses[destroy_name] += 1
                repair_uses[repair_name] += 1
                destroy_score[destroy_name] += reward
                repair_score[repair_name] += reward
                destroy_total_uses[destroy_name] += 1
                repair_total_uses[repair_name] += 1
                destroy_total_score[destroy_name] += reward
                repair_total_score[repair_name] += reward
                if iteration % update_period == 0:
                    for name in destroy_names:
                        if destroy_uses[name]:
                            average = destroy_score[name] / destroy_uses[name]
                            destroy_weights[name] = (
                                (1.0 - reaction_factor) * destroy_weights[name]
                                + reaction_factor * average
                            )
                        destroy_score[name] = 0.0
                        destroy_uses[name] = 0
                    for name in repair_names:
                        if repair_uses[name]:
                            average = repair_score[name] / repair_uses[name]
                            repair_weights[name] = (
                                (1.0 - reaction_factor) * repair_weights[name]
                                + reaction_factor * average
                            )
                        repair_score[name] = 0.0
                        repair_uses[name] = 0
                if iteration == 1 or iteration % log_period == 0 or reward == 8.0:
                    convergence.append(ConvergenceRecord(
                        profile,
                        seed,
                        iteration,
                        time.perf_counter() - start_clock,
                        temperature,
                        destroy_name,
                        repair_name,
                        accepted,
                        current.objective,
                        best.objective,
                    ))
                temperature *= cooling_rate

            elapsed = time.perf_counter() - start_clock
            _archive_add(archive, best)
            run_records.append(ALNSRunRecord(
                profile,
                seed,
                iteration,
                elapsed,
                feasible_iteration,
                accepted_moves,
                improving_moves,
                best.objective,
                stop_reason,
            ))
            for name, value in sorted(destroy_weights.items()):
                operator_records.append(OperatorRecord(
                    profile, seed, "破坏", name, value,
                    destroy_total_uses[name], destroy_total_score[name],
                ))
            for name, value in sorted(repair_weights.items()):
                operator_records.append(OperatorRecord(
                    profile, seed, "修复", name, value,
                    repair_total_uses[name], repair_total_score[name],
                ))

    if not archive:
        raise RuntimeError("ALNS未获得任何零硬违约方案")
    front = pareto_filter_schedules(list(archive.values()))
    representatives = select_pareto_representatives(front)
    best = min(
        front,
        key=lambda item: (normalized_ideal_distance(item, front)[0], _objective_tuple(item.objective)),
    )
    return Q2ALNSResult(
        best,
        baseline,
        greedy_initial,
        tuple(run_records),
        tuple(convergence),
        tuple(operator_records),
        front,
        representatives,
    )


__all__ = [
    "ALNSRunRecord",
    "ConvergenceRecord",
    "DESTROY_OPERATORS",
    "OperatorRecord",
    "PROFILE_WEIGHTS",
    "Q2ALNSResult",
    "REPAIR_OPERATORS",
    "solve_q2_alns",
]
