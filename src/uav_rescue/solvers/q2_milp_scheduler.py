"""固定运输架次结构下的机型—实体机—电池—时刻联合MILP。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from uav_rescue.domain import Box, TransportBattery, TransportUnit
from uav_rescue.models.q2_joint import (
    DeadlinePolicy,
    hard_deadline_s,
    is_soft_expected_window,
)
from uav_rescue.models.q2_transport import (
    Objective,
    RouteEvaluator,
    RoutePlan,
    ScheduleResult,
    ScheduledTrip,
    TripEvaluation,
)
from uav_rescue.physics.battery import recharge_time_s


TOLERANCE = 1e-7


@dataclass(frozen=True)
class MILPStageRecord:
    objective: str
    value: float
    dual_bound: float | None
    mip_gap: float | None
    status: str
    wall_time_s: float


@dataclass(frozen=True)
class MILPScheduleResult:
    schedule: ScheduleResult
    stages: tuple[MILPStageRecord, ...]
    all_stages_optimal: bool
    variable_count: int
    constraint_count: int
    backend: str


class _Builder:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integrality: list[int] = []
        self.rows: list[dict[int, float]] = []
        self.row_lower: list[float] = []
        self.row_upper: list[float] = []

    def variable(
        self, name: str, lower: float = 0.0, upper: float = math.inf,
        integer: bool = False,
    ) -> int:
        index = len(self.names)
        self.names.append(name)
        self.lower.append(lower)
        self.upper.append(upper)
        self.integrality.append(1 if integer else 0)
        return index

    def constraint(
        self, coefficients: dict[int, float],
        lower: float = -math.inf, upper: float = math.inf,
    ) -> None:
        cleaned = {index: value for index, value in coefficients.items() if abs(value) > 1e-14}
        self.rows.append(cleaned)
        self.row_lower.append(lower)
        self.row_upper.append(upper)

    def linear_constraint(self) -> LinearConstraint:
        row_indexes: list[int] = []
        column_indexes: list[int] = []
        values: list[float] = []
        for row_index, row in enumerate(self.rows):
            for column_index, value in row.items():
                row_indexes.append(row_index)
                column_indexes.append(column_index)
                values.append(value)
        matrix = coo_matrix(
            (values, (row_indexes, column_indexes)),
            shape=(len(self.rows), len(self.names)),
        ).tocsr()
        return LinearConstraint(
            matrix,
            np.asarray(self.row_lower, dtype=float),
            np.asarray(self.row_upper, dtype=float),
        )


def _objective_from_schedule(
    trips: list[ScheduledTrip], boxes: dict[str, Box], policy: DeadlinePolicy,
) -> Objective:
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
        deadline = hard_deadline_s(box, policy)
        if deadline is not None:
            late = max(0.0, delivered - deadline)
            hard_count += int(late > 1e-6)
            hard_tardiness += late
        if is_soft_expected_window(box, policy):
            weighted_soft += (box.priority or 1.0) * max(
                0.0, delivered - float(box.expected_deadline_s)
            )
    return Objective(
        hard_count,
        hard_tardiness,
        weighted_soft,
        max((trip.return_s for trip in trips), default=0.0),
        sum(trip.evaluation.energy_kwh for trip in trips),
        len(trips),
    )


def _sum_coefficients(variables: list[int], coefficient: float = 1.0) -> dict[int, float]:
    return {index: coefficient for index in variables}


def _add(coefficients: dict[int, float], index: int, value: float) -> None:
    coefficients[index] = coefficients.get(index, 0.0) + value


def schedule_routes_milp(
    routes: tuple[RoutePlan, ...] | list[RoutePlan],
    evaluator: RouteEvaluator,
    boxes: dict[str, Box],
    units: dict[str, TransportUnit],
    batteries: dict[str, TransportBattery],
    policy: DeadlinePolicy,
    *,
    time_limit_per_stage_s: float = 10.0,
    mip_rel_gap: float = 1e-6,
    allow_hard_violations: bool = False,
    incumbent_schedule: ScheduleResult | None = None,
    backend: str = "auto",
) -> MILPScheduleResult:
    """联合优化机型、实体无人机、电池、资源先后关系和开始时刻。

    目标采用逐层 ε 约束：硬违约数、硬迟到、普通物资加权迟到（软口径）、
    完工时间、能耗。每层把当前最好值作为后一层上界，避免任意大权重。
    """

    routes = tuple(routes)
    if not routes:
        empty = ScheduleResult((), Objective(0, 0.0, 0.0, 0.0, 0.0, 0))
        return MILPScheduleResult(empty, (), True, 0, 0, "none")
    if time_limit_per_stage_s <= 0:
        raise ValueError("MILP每阶段时间上限必须为正")

    unit_ids = sorted(units)
    battery_ids = sorted(batteries)
    evaluations: dict[tuple[int, str], TripEvaluation] = {}
    charge_times: dict[tuple[int, str], float] = {}
    models_by_route: dict[int, tuple[str, ...]] = {}
    for route_index, route in enumerate(routes):
        feasible_models: list[str] = []
        for evaluation in evaluator.evaluate(route.box_ids, route.service_sequence):
            model_name = evaluation.model
            matching_units = [unit for unit in units.values() if unit.model == model_name]
            matching_batteries = [
                battery for battery in batteries.values() if battery.model == model_name
            ]
            if not matching_units or not matching_batteries:
                continue
            charge = recharge_time_s(
                evaluation.end_soc, matching_batteries[0].full_charge_time_s
            )
            evaluations[(route_index, model_name)] = evaluation
            charge_times[(route_index, model_name)] = charge
            feasible_models.append(model_name)
        if not feasible_models:
            raise RuntimeError(f"架次不存在资源兼容的可行机型：{route.box_ids}")
        models_by_route[route_index] = tuple(sorted(feasible_models))

    horizon = sum(
        max(
            evaluations[(route_index, model_name)].duration_s
            + charge_times[(route_index, model_name)]
            for model_name in models_by_route[route_index]
        )
        for route_index in range(len(routes))
    ) + 1.0
    big_m = horizon
    builder = _Builder()
    starts = {
        route_index: builder.variable(f"start[{route_index}]", 0.0, horizon)
        for route_index in range(len(routes))
    }
    makespan = builder.variable("makespan", 0.0, horizon)
    unit_assignment: dict[tuple[int, str], int] = {}
    battery_assignment: dict[tuple[int, str], int] = {}

    for route_index in range(len(routes)):
        for unit_id in unit_ids:
            if units[unit_id].model in models_by_route[route_index]:
                unit_assignment[(route_index, unit_id)] = builder.variable(
                    f"unit[{route_index},{unit_id}]", 0.0, 1.0, True
                )
        for battery_id in battery_ids:
            if batteries[battery_id].model in models_by_route[route_index]:
                battery_assignment[(route_index, battery_id)] = builder.variable(
                    f"battery[{route_index},{battery_id}]", 0.0, 1.0, True
                )
        builder.constraint(_sum_coefficients([
            variable for (index, _), variable in unit_assignment.items()
            if index == route_index
        ]), 1.0, 1.0)
        builder.constraint(_sum_coefficients([
            variable for (index, _), variable in battery_assignment.items()
            if index == route_index
        ]), 1.0, 1.0)
        for model_name in models_by_route[route_index]:
            coefficients: dict[int, float] = {}
            for unit_id in unit_ids:
                if units[unit_id].model == model_name and (route_index, unit_id) in unit_assignment:
                    _add(coefficients, unit_assignment[(route_index, unit_id)], 1.0)
            for battery_id in battery_ids:
                if batteries[battery_id].model == model_name and (route_index, battery_id) in battery_assignment:
                    _add(coefficients, battery_assignment[(route_index, battery_id)], -1.0)
            builder.constraint(coefficients, 0.0, 0.0)

    # 同质资源对称性削减：编号较大的资源承担任务前，较小编号资源的任务数不少于它。
    for model_name in sorted({unit.model for unit in units.values()}):
        compatible = [unit_id for unit_id in unit_ids if units[unit_id].model == model_name]
        for left, right in zip(compatible, compatible[1:]):
            coefficients: dict[int, float] = {}
            for route_index in range(len(routes)):
                if (route_index, left) in unit_assignment:
                    _add(coefficients, unit_assignment[(route_index, left)], 1.0)
                if (route_index, right) in unit_assignment:
                    _add(coefficients, unit_assignment[(route_index, right)], -1.0)
            builder.constraint(coefficients, 0.0, math.inf)
    for model_name in sorted({battery.model for battery in batteries.values()}):
        compatible = [
            battery_id for battery_id in battery_ids
            if batteries[battery_id].model == model_name
        ]
        for left, right in zip(compatible, compatible[1:]):
            coefficients = {}
            for route_index in range(len(routes)):
                if (route_index, left) in battery_assignment:
                    _add(coefficients, battery_assignment[(route_index, left)], 1.0)
                if (route_index, right) in battery_assignment:
                    _add(coefficients, battery_assignment[(route_index, right)], -1.0)
            builder.constraint(coefficients, 0.0, math.inf)

    # 同一实体无人机上的架次不能重叠。
    for first in range(len(routes)):
        for second in range(first + 1, len(routes)):
            common_units = [
                unit_id for unit_id in unit_ids
                if (first, unit_id) in unit_assignment
                and (second, unit_id) in unit_assignment
            ]
            for unit_id in common_units:
                model_name = units[unit_id].model
                before = builder.variable(
                    f"unit_order[{first},{second},{unit_id}]", 0.0, 1.0, True
                )
                first_duration = evaluations[(first, model_name)].duration_s
                second_duration = evaluations[(second, model_name)].duration_s
                # before=1 表示 first 在 second 之前。
                coefficients = {
                    starts[first]: 1.0,
                    starts[second]: -1.0,
                    before: big_m,
                    unit_assignment[(first, unit_id)]: big_m,
                    unit_assignment[(second, unit_id)]: big_m,
                }
                builder.constraint(coefficients, -math.inf, 3.0 * big_m - first_duration)
                coefficients = {
                    starts[second]: 1.0,
                    starts[first]: -1.0,
                    before: -big_m,
                    unit_assignment[(first, unit_id)]: big_m,
                    unit_assignment[(second, unit_id)]: big_m,
                }
                builder.constraint(coefficients, -math.inf, 2.0 * big_m - second_duration)

    # 同一电池从架次开始占用至返航后充至100%，期间不能重用。
    for first in range(len(routes)):
        for second in range(first + 1, len(routes)):
            common_batteries = [
                battery_id for battery_id in battery_ids
                if (first, battery_id) in battery_assignment
                and (second, battery_id) in battery_assignment
            ]
            for battery_id in common_batteries:
                model_name = batteries[battery_id].model
                before = builder.variable(
                    f"battery_order[{first},{second},{battery_id}]", 0.0, 1.0, True
                )
                first_occupancy = (
                    evaluations[(first, model_name)].duration_s
                    + charge_times[(first, model_name)]
                )
                second_occupancy = (
                    evaluations[(second, model_name)].duration_s
                    + charge_times[(second, model_name)]
                )
                coefficients = {
                    starts[first]: 1.0,
                    starts[second]: -1.0,
                    before: big_m,
                    battery_assignment[(first, battery_id)]: big_m,
                    battery_assignment[(second, battery_id)]: big_m,
                }
                builder.constraint(coefficients, -math.inf, 3.0 * big_m - first_occupancy)
                coefficients = {
                    starts[second]: 1.0,
                    starts[first]: -1.0,
                    before: -big_m,
                    battery_assignment[(first, battery_id)]: big_m,
                    battery_assignment[(second, battery_id)]: big_m,
                }
                builder.constraint(coefficients, -math.inf, 2.0 * big_m - second_occupancy)

    delivery_vars: dict[str, int] = {}
    hard_violation_vars: list[int] = []
    hard_tardiness_vars: list[int] = []
    soft_tardiness_terms: dict[int, float] = {}
    for route_index, route in enumerate(routes):
        # 任务和充电均限制在安全上界内，进一步收紧开始时刻。
        coefficients = {starts[route_index]: 1.0}
        for battery_id in battery_ids:
            variable = battery_assignment.get((route_index, battery_id))
            if variable is None:
                continue
            model_name = batteries[battery_id].model
            occupancy = (
                evaluations[(route_index, model_name)].duration_s
                + charge_times[(route_index, model_name)]
            )
            _add(coefficients, variable, occupancy)
        builder.constraint(coefficients, -math.inf, horizon)

        completion = {starts[route_index]: 1.0, makespan: -1.0}
        for unit_id in unit_ids:
            variable = unit_assignment.get((route_index, unit_id))
            if variable is not None:
                _add(
                    completion,
                    variable,
                    evaluations[(route_index, units[unit_id].model)].duration_s,
                )
        builder.constraint(completion, -math.inf, 0.0)

        for box_id in route.box_ids:
            delivered = builder.variable(f"delivery[{box_id}]", 0.0, horizon)
            delivery_vars[box_id] = delivered
            equality = {delivered: 1.0, starts[route_index]: -1.0}
            for unit_id in unit_ids:
                variable = unit_assignment.get((route_index, unit_id))
                if variable is not None:
                    offset = evaluations[
                        (route_index, units[unit_id].model)
                    ].delivery_offset(box_id)
                    _add(equality, variable, -offset)
            builder.constraint(equality, 0.0, 0.0)

            box = boxes[box_id]
            deadline = hard_deadline_s(box, policy)
            if deadline is not None:
                if not allow_hard_violations:
                    builder.constraint({delivered: 1.0}, -math.inf, deadline)
                else:
                    # 迟到等式的松弛量必须同时覆盖排程上界和截止时刻；若直接用
                    # horizon，当 deadline > horizon 时，未违约分支会被错误截断。
                    lateness_m = horizon + deadline
                    violation = builder.variable(
                        f"hard_violation[{box_id}]", 0.0, 1.0, True
                    )
                    tardiness = builder.variable(
                        f"hard_tardiness[{box_id}]", 0.0, horizon
                    )
                    hard_violation_vars.append(violation)
                    hard_tardiness_vars.append(tardiness)
                    builder.constraint(
                        {delivered: 1.0, violation: -big_m},
                        -math.inf, deadline,
                    )
                    builder.constraint(
                        {delivered: 1.0, tardiness: -1.0},
                        -math.inf, deadline,
                    )
                    builder.constraint(
                        {tardiness: 1.0, delivered: -1.0, violation: lateness_m},
                        -math.inf, lateness_m - deadline,
                    )
                    builder.constraint(
                        {tardiness: 1.0, violation: -horizon},
                        -math.inf, 0.0,
                    )
            if is_soft_expected_window(box, policy):
                tardiness = builder.variable(
                    f"soft_tardiness[{box_id}]", 0.0, horizon
                )
                builder.constraint(
                    {delivered: 1.0, tardiness: -1.0},
                    -math.inf, float(box.expected_deadline_s),
                )
                soft_tardiness_terms[tardiness] = box.priority or 1.0

    energy_terms: dict[int, float] = {}
    for (route_index, unit_id), variable in unit_assignment.items():
        energy_terms[variable] = evaluations[
            (route_index, units[unit_id].model)
        ].energy_kwh

    stages: list[tuple[str, dict[int, float]]] = [
        ("hard_violation_count", _sum_coefficients(hard_violation_vars)),
        ("hard_tardiness_s", _sum_coefficients(hard_tardiness_vars)),
    ]
    if policy == DeadlinePolicy.ORDINARY_SOFT:
        stages.append(("weighted_soft_tardiness", soft_tardiness_terms))
    stages.extend([
        ("makespan_s", {makespan: 1.0}),
        ("energy_kwh", energy_terms),
    ])

    final_x: np.ndarray | None = None
    start_vector = np.full(len(builder.names), np.nan, dtype=float)
    if incumbent_schedule is not None:
        unused = set(range(len(incumbent_schedule.trips)))
        for route_index, route in enumerate(routes):
            match = next((
                trip_index for trip_index in sorted(unused)
                if tuple(sorted(incumbent_schedule.trips[trip_index].route.box_ids))
                == tuple(sorted(route.box_ids))
                and tuple(incumbent_schedule.trips[trip_index].route.service_sequence)
                == tuple(route.service_sequence)
            ), None)
            if match is None:
                continue
            trip = incumbent_schedule.trips[match]
            unused.remove(match)
            start_vector[starts[route_index]] = trip.start_s
            for unit_id in unit_ids:
                variable = unit_assignment.get((route_index, unit_id))
                if variable is not None:
                    start_vector[variable] = float(unit_id == trip.unit_id)
            for battery_id in battery_ids:
                variable = battery_assignment.get((route_index, battery_id))
                if variable is not None:
                    start_vector[variable] = float(battery_id == trip.battery_id)
        start_vector[makespan] = incumbent_schedule.objective.makespan_s

    selected_backend = backend
    if backend == "auto":
        try:
            import gurobipy  # noqa: F401
        except ImportError:
            selected_backend = "scipy"
        else:
            selected_backend = "gurobi"
    if selected_backend not in {"gurobi", "scipy"}:
        raise ValueError(f"未知MILP后端：{selected_backend}")

    def solve_stage(objective: np.ndarray):
        if selected_backend == "scipy":
            return milp(
                objective,
                integrality=np.asarray(builder.integrality, dtype=int),
                bounds=Bounds(
                    np.asarray(builder.lower, dtype=float),
                    np.asarray(builder.upper, dtype=float),
                ),
                constraints=builder.linear_constraint(),
                options={
                    "disp": False,
                    "presolve": True,
                    "time_limit": time_limit_per_stage_s,
                    "mip_rel_gap": mip_rel_gap,
                },
            )

        import gurobipy as gp
        from gurobipy import GRB

        model = gp.Model("q2_joint_schedule")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit_per_stage_s
        model.Params.MIPGap = mip_rel_gap
        variables = [
            model.addVar(
                lb=builder.lower[index],
                ub=(GRB.INFINITY if math.isinf(builder.upper[index]) else builder.upper[index]),
                vtype=GRB.INTEGER if builder.integrality[index] else GRB.CONTINUOUS,
                name=builder.names[index],
            )
            for index in range(len(builder.names))
        ]
        model.update()
        for index, value in enumerate(start_vector):
            if math.isfinite(value):
                variables[index].Start = float(value)
        for row, lower, upper in zip(builder.rows, builder.row_lower, builder.row_upper):
            expression = gp.quicksum(value * variables[index] for index, value in row.items())
            if math.isfinite(lower) and math.isfinite(upper) and abs(lower - upper) <= 1e-10:
                model.addConstr(expression == lower)
            else:
                if math.isfinite(lower):
                    model.addConstr(expression >= lower)
                if math.isfinite(upper):
                    model.addConstr(expression <= upper)
        model.setObjective(
            gp.quicksum(float(value) * variables[index] for index, value in enumerate(objective) if value),
            GRB.MINIMIZE,
        )
        model.optimize()

        class Result:
            pass

        result = Result()
        result.x = (
            np.asarray([variable.X for variable in variables], dtype=float)
            if model.SolCount else None
        )
        result.status = 0 if model.Status == GRB.OPTIMAL else 1
        result.message = f"Gurobi status {model.Status}"
        result.mip_dual_bound = model.ObjBound if model.SolCount or model.Status != GRB.LOADED else None
        result.mip_gap = model.MIPGap if model.SolCount else None
        return result

    records: list[MILPStageRecord] = []
    all_optimal = True
    for stage_name, objective_terms in stages:
        if not objective_terms:
            records.append(MILPStageRecord(stage_name, 0.0, 0.0, 0.0, "CONSTANT", 0.0))
            continue
        objective = np.zeros(len(builder.names), dtype=float)
        for variable, coefficient in objective_terms.items():
            objective[variable] = coefficient
        started = time.perf_counter()
        result = solve_stage(objective)
        elapsed = time.perf_counter() - started
        if result.x is None:
            if final_x is None:
                raise RuntimeError(f"MILP资源调度失败：{stage_name}: {result.message}")
            # SciPy/HiGHS暂不提供MIP warm start。高层目标被锁定后，次级阶段
            # 可能在很短时限内尚未重新找到 incumbent；此时保留上一阶段已验证
            # 的可行解并如实标记，不能把无 incumbent 误报成模型不可行。
            records.append(MILPStageRecord(
                stage_name,
                float("nan"),
                getattr(result, "mip_dual_bound", None),
                getattr(result, "mip_gap", None),
                "NO_INCUMBENT_LIMIT",
                elapsed,
            ))
            all_optimal = False
            break
        value = float(objective @ result.x)
        status = "OPTIMAL" if result.status == 0 else "FEASIBLE_LIMIT"
        all_optimal = all_optimal and result.status == 0
        records.append(MILPStageRecord(
            stage_name,
            value,
            getattr(result, "mip_dual_bound", None),
            getattr(result, "mip_gap", None),
            status,
            elapsed,
        ))
        # 后续阶段不得恶化已经取得的上层目标；若前层限时，后层仍允许进一步改善。
        tolerance = max(1e-4, 1e-6 * max(1.0, abs(value)))
        builder.constraint(objective_terms, -math.inf, value + tolerance)
        final_x = result.x
        start_vector = result.x.copy()

    if final_x is None:
        raise RuntimeError("MILP没有执行任何优化阶段")

    scheduled: list[tuple[int, ScheduledTrip]] = []
    for route_index, route in enumerate(routes):
        unit_id = next(
            unit_id for unit_id in unit_ids
            if (route_index, unit_id) in unit_assignment
            and final_x[unit_assignment[(route_index, unit_id)]] > 0.5
        )
        battery_id = next(
            battery_id for battery_id in battery_ids
            if (route_index, battery_id) in battery_assignment
            and final_x[battery_assignment[(route_index, battery_id)]] > 0.5
        )
        model_name = units[unit_id].model
        if batteries[battery_id].model != model_name:
            raise RuntimeError("MILP返回了机型不相容的无人机—电池组合")
        evaluation = evaluations[(route_index, model_name)]
        start_s = max(0.0, float(final_x[starts[route_index]]))
        return_s = start_s + evaluation.duration_s
        charge_s = charge_times[(route_index, model_name)]
        scheduled.append((route_index, ScheduledTrip(
            "",
            route,
            evaluation,
            unit_id,
            battery_id,
            start_s,
            return_s,
            charge_s,
            return_s + charge_s,
        )))
    scheduled.sort(key=lambda item: (item[1].start_s, item[0]))
    trips = [
        ScheduledTrip(
            f"T{trip_number:03d}",
            trip.route,
            trip.evaluation,
            trip.unit_id,
            trip.battery_id,
            trip.start_s,
            trip.return_s,
            trip.recharge_s,
            trip.battery_release_s,
        )
        for trip_number, (_, trip) in enumerate(scheduled, start=1)
    ]
    schedule = ScheduleResult(tuple(trips), _objective_from_schedule(trips, boxes, policy))
    return MILPScheduleResult(
        schedule,
        tuple(records),
        all_optimal,
        len(builder.names),
        len(builder.rows),
        selected_backend,
    )


__all__ = ["MILPScheduleResult", "MILPStageRecord", "schedule_routes_milp"]
