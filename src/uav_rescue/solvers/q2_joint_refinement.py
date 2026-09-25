"""ALNS候选运输结构与MILP资源排程的双向协调改进。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from uav_rescue.domain import Box, TransportBattery, TransportUnit
from uav_rescue.models.q2_joint import DeadlinePolicy
from uav_rescue.models.q2_transport import (
    RouteEvaluator,
    RoutePlan,
    ScheduleResult,
)
from uav_rescue.solvers.q2_milp_scheduler import MILPScheduleResult, schedule_routes_milp


@dataclass(frozen=True)
class JointRefinementRecord:
    round_index: int
    candidate_label: str
    route_count: int
    accepted: bool
    objective: object
    all_milp_stages_optimal: bool
    variable_count: int
    constraint_count: int


@dataclass(frozen=True)
class JointRefinementResult:
    best: MILPScheduleResult
    records: tuple[JointRefinementRecord, ...]
    evaluated_route_sets: int


def _route_key(routes: tuple[RoutePlan, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(sorted(
        (
            tuple(sorted(route.box_ids)),
            route.service_sequence,
        )
        for route in routes
    ))


def _canonical_routes(schedule: ScheduleResult) -> tuple[RoutePlan, ...]:
    return tuple(
        RoutePlan(tuple(sorted(trip.route.box_ids)), tuple(trip.route.service_sequence))
        for trip in schedule.trips
    )


def _objective_key(schedule: ScheduleResult, policy: DeadlinePolicy) -> tuple[float, ...]:
    objective = schedule.objective
    if policy == DeadlinePolicy.ORDINARY_SOFT:
        return (
            float(objective.hard_violation_count),
            objective.hard_tardiness_s,
            objective.weighted_soft_tardiness,
            objective.makespan_s,
            objective.energy_kwh,
            float(objective.trip_count),
        )
    return (
        float(objective.hard_violation_count),
        objective.hard_tardiness_s,
        objective.makespan_s,
        objective.energy_kwh,
        float(objective.trip_count),
    )


def _valid_cover(routes: tuple[RoutePlan, ...], boxes: dict[str, Box]) -> bool:
    ids = [box_id for route in routes for box_id in route.box_ids]
    return len(ids) == len(boxes) and len(set(ids)) == len(ids) and set(ids) == set(boxes)


def _physically_feasible(routes: tuple[RoutePlan, ...], evaluator: RouteEvaluator) -> bool:
    return all(
        evaluator.evaluate(route.box_ids, route.service_sequence)
        for route in routes
    )


def _cheap_score(routes: tuple[RoutePlan, ...], evaluator: RouteEvaluator) -> tuple[float, ...]:
    evaluations = [
        min(
            evaluator.evaluate(route.box_ids, route.service_sequence),
            key=lambda item: (item.duration_s, item.energy_kwh),
        )
        for route in routes
    ]
    durations = [item.duration_s for item in evaluations]
    return (
        max(durations, default=0.0),
        sum(durations),
        sum(item.energy_kwh for item in evaluations),
        float(len(routes)),
    )


def _replace_route(
    routes: list[RoutePlan], source_index: int, replacement: list[RoutePlan],
) -> tuple[RoutePlan, ...]:
    return tuple(routes[:source_index] + replacement + routes[source_index + 1:])


def critical_resource_neighbors(
    schedule: ScheduleResult,
    boxes: dict[str, Box],
    evaluator: RouteEvaluator,
    maximum_candidates: int = 12,
) -> tuple[tuple[str, tuple[RoutePlan, ...]], ...]:
    """根据MILP排程的完工/电池关键链生成运输结构邻域。

    邻域只改变组批、架次边界与服务区组合；机型、实体机、电池和开始时刻
    继续交由下一次MILP联合决定。
    """

    routes = list(_canonical_routes(schedule))
    route_index_by_key = {
        (route.box_ids, route.service_sequence): index
        for index, route in enumerate(routes)
    }
    critical = sorted(
        schedule.trips,
        key=lambda trip: (trip.return_s, trip.battery_release_s),
        reverse=True,
    )[: min(3, len(schedule.trips))]
    proposals: dict[tuple[tuple[object, ...], ...], tuple[str, tuple[RoutePlan, ...]]] = {}

    for critical_trip in critical:
        source_key = (
            tuple(sorted(critical_trip.route.box_ids)),
            tuple(critical_trip.route.service_sequence),
        )
        source_index = route_index_by_key.get(source_key)
        if source_index is None:
            continue
        source = routes[source_index]
        source_services = tuple(source.service_sequence)

        # 多服务区长架次按访问序列切分，以释放关键无人机/电池链。
        for cut in range(1, len(source_services)):
            left_services = source_services[:cut]
            right_services = source_services[cut:]
            left = RoutePlan(
                tuple(sorted(
                    box_id for box_id in source.box_ids
                    if boxes[box_id].service_id in set(left_services)
                )),
                left_services,
            )
            right = RoutePlan(
                tuple(sorted(
                    box_id for box_id in source.box_ids
                    if boxes[box_id].service_id in set(right_services)
                )),
                right_services,
            )
            candidate = _replace_route(routes, source_index, [left, right])
            if _valid_cover(candidate, boxes) and _physically_feasible(candidate, evaluator):
                proposals.setdefault(_route_key(candidate), ("关键架次切分", candidate))

        # 将关键架次中的一个完整服务区移到其他架次，改变组批与资源占用长度。
        for service in source_services:
            moved = tuple(sorted(
                box_id for box_id in source.box_ids
                if boxes[box_id].service_id == service
            ))
            kept = tuple(sorted(box_id for box_id in source.box_ids if box_id not in moved))
            kept_services = tuple(item for item in source_services if item != service)
            for target_index, target in enumerate(routes):
                if target_index == source_index:
                    continue
                if service in target.service_sequence:
                    sequences = (target.service_sequence,)
                else:
                    sequences = tuple(
                        target.service_sequence[:position]
                        + (service,)
                        + target.service_sequence[position:]
                        for position in range(len(target.service_sequence) + 1)
                    )
                for sequence in sequences:
                    target_new = RoutePlan(
                        tuple(sorted((*target.box_ids, *moved))),
                        tuple(sequence),
                    )
                    candidate_list = routes.copy()
                    candidate_list[target_index] = target_new
                    if kept:
                        candidate_list[source_index] = RoutePlan(kept, kept_services)
                    else:
                        candidate_list.pop(source_index)
                    candidate = tuple(candidate_list)
                    if _valid_cover(candidate, boxes) and _physically_feasible(candidate, evaluator):
                        proposals.setdefault(
                            _route_key(candidate),
                            ("关键服务区跨架次移动", candidate),
                        )

    ranked = sorted(
        proposals.values(),
        key=lambda item: (_cheap_score(item[1], evaluator), item[0], _route_key(item[1])),
    )
    return tuple(ranked[:maximum_candidates])


def refine_with_milp_feedback(
    initial_schedules: tuple[ScheduleResult, ...] | list[ScheduleResult],
    evaluator: RouteEvaluator,
    boxes: dict[str, Box],
    units: dict[str, TransportUnit],
    batteries: dict[str, TransportBattery],
    policy: DeadlinePolicy,
    *,
    rounds: int = 1,
    candidates_per_round: int = 6,
    time_limit_per_stage_s: float = 5.0,
    mip_rel_gap: float = 1e-4,
    additional_route_sets: tuple[
        tuple[str, tuple[RoutePlan, ...], ScheduleResult | None], ...
    ] = (),
) -> JointRefinementResult:
    """MILP评价ALNS候选，并把资源关键链反馈为下一轮运输邻域。"""

    if not initial_schedules and not additional_route_sets:
        raise ValueError("联合改进至少需要一个ALNS候选或外部运输结构")
    cache: dict[tuple[tuple[object, ...], ...], MILPScheduleResult] = {}
    records: list[JointRefinementRecord] = []
    record_route_keys: list[tuple[tuple[object, ...], ...]] = []

    def evaluate(
        label: str,
        routes: tuple[RoutePlan, ...],
        round_index: int,
        incumbent: ScheduleResult | None = None,
    ) -> MILPScheduleResult | None:
        key = _route_key(routes)
        if key in cache:
            return cache[key]
        try:
            result = schedule_routes_milp(
                routes,
                evaluator,
                boxes,
                units,
                batteries,
                policy,
                time_limit_per_stage_s=time_limit_per_stage_s,
                mip_rel_gap=mip_rel_gap,
                allow_hard_violations=False,
                incumbent_schedule=incumbent,
            )
        except RuntimeError:
            return None
        cache[key] = result
        record_route_keys.append(key)
        records.append(JointRefinementRecord(
            round_index,
            label,
            len(routes),
            False,
            result.schedule.objective,
            result.all_stages_optimal,
            result.variable_count,
            result.constraint_count,
        ))
        return result

    evaluated = [
        result
        for index, schedule in enumerate(initial_schedules)
        if (result := evaluate(
            f"ALNS候选{index + 1}", _canonical_routes(schedule), 0, schedule
        )) is not None
    ]
    evaluated.extend(
        result
        for label, routes, incumbent in additional_route_sets
        if (result := evaluate(label, routes, 0, incumbent)) is not None
    )
    if not evaluated:
        raise RuntimeError("所有ALNS候选在联合MILP中均不可行")
    best = min(evaluated, key=lambda item: _objective_key(item.schedule, policy))

    for round_index in range(1, rounds + 1):
        neighbors = critical_resource_neighbors(
            best.schedule,
            boxes,
            evaluator,
            maximum_candidates=candidates_per_round,
        )
        improved = False
        for label, routes in neighbors:
            candidate = evaluate(label, routes, round_index)
            if candidate is None:
                continue
            if _objective_key(candidate.schedule, policy) < _objective_key(best.schedule, policy):
                best = candidate
                improved = True
        if not improved:
            break

    accepted_route_key = _route_key(_canonical_routes(best.schedule))
    records = [
        JointRefinementRecord(
            row.round_index,
            row.candidate_label,
            row.route_count,
            route_key == accepted_route_key,
            row.objective,
            row.all_milp_stages_optimal,
            row.variable_count,
            row.constraint_count,
        )
        for row, route_key in zip(records, record_route_keys)
    ]
    return JointRefinementResult(best, tuple(records), len(cache))


__all__ = [
    "JointRefinementRecord",
    "JointRefinementResult",
    "critical_resource_neighbors",
    "refine_with_milp_feedback",
]
