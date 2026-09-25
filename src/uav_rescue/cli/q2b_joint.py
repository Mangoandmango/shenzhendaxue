"""问题二强化版B：ALNS运输候选与MILP资源排程双向协调。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from pathlib import Path

from uav_rescue.cli.q2b import (
    _load_scheme_a_warm_start,
    _write_schedule as write_hard_schedule,
)
from uav_rescue.cli.q2b_soft import _write_schedule as write_soft_schedule
from uav_rescue.cli.q2b_time import TIME_PROFILE
from uav_rescue.io.readers import (
    load_boxes,
    load_leg_geometry_cache,
    load_nodes,
    load_transport_drones,
    load_transport_resources,
)
from uav_rescue.io.writers import write_csv, write_json
from uav_rescue.models.q2_joint import DeadlinePolicy
from uav_rescue.models.q2_transport import RouteEvaluator, RoutePlan
from uav_rescue.models.q2_transport import (
    Objective, ScheduleResult, ScheduledTrip,
)
from uav_rescue.physics.battery import recharge_time_s
from uav_rescue.settings import load_toml, project_path
from uav_rescue.solvers.q2_alns import solve_q2_alns
from uav_rescue.solvers.q2_joint_refinement import refine_with_milp_feedback
from uav_rescue.validation.q2 import (
    assert_q2_valid as assert_hard_valid,
    validate_q2_solution as validate_hard,
)
from uav_rescue.validation.q2_soft import (
    assert_q2_valid as assert_soft_valid,
    validate_q2_solution as validate_soft,
)


def _objective_key(schedule, policy: DeadlinePolicy) -> tuple[float, ...]:
    objective = schedule.objective
    if policy == DeadlinePolicy.ORDINARY_SOFT:
        return (
            objective.hard_violation_count,
            objective.hard_tardiness_s,
            objective.weighted_soft_tardiness,
            objective.makespan_s,
            objective.energy_kwh,
            objective.trip_count,
        )
    return (
        objective.hard_violation_count,
        objective.hard_tardiness_s,
        objective.makespan_s,
        objective.energy_kwh,
        objective.trip_count,
    )


def _load_schedule(
    output_root: Path,
    evaluator: RouteEvaluator,
    batteries,
) -> ScheduleResult:
    """读取已复核排程，作为跨口径运输结构和MILP热启动。"""

    table_root = output_root / "tables"
    final_root = next(
        (
            candidate for candidate in (
                table_root / "最终方案",
                table_root / "B3最终方案",
            )
            if candidate.exists()
        ),
        None,
    )
    if final_root is None:
        raise FileNotFoundError(f"无法定位已复核运输结构：{output_root}")
    trip_path = final_root / "运输架次.csv"
    delivery_paths = sorted(final_root.glob("逐箱交付*.csv"))
    if not trip_path.exists() or len(delivery_paths) != 1:
        raise FileNotFoundError(f"无法定位已复核运输结构：{output_root}")
    with trip_path.open(encoding="utf-8-sig", newline="") as handle:
        trip_rows = list(csv.DictReader(handle))
    with delivery_paths[0].open(encoding="utf-8-sig", newline="") as handle:
        delivery_rows = list(csv.DictReader(handle))
    boxes_by_trip: dict[str, list[str]] = {}
    for row in delivery_rows:
        boxes_by_trip.setdefault(row["架次编号"], []).append(row["货箱编号"])
    trips: list[ScheduledTrip] = []
    for row in sorted(trip_rows, key=lambda item: item["架次编号"]):
        trip_id = row["架次编号"]
        box_ids = tuple(sorted(boxes_by_trip.get(trip_id, ())))
        sequence = tuple(
            service for service in row["访问服务区顺序"].split("->") if service
        )
        if not box_ids or not sequence:
            raise ValueError(f"运输结构记录不完整：{output_root}:{trip_id}")
        route = RoutePlan(box_ids, sequence)
        model = row["机型"]
        evaluation = next(
            (
                item for item in evaluator.evaluate(route.box_ids, route.service_sequence)
                if item.model == model
            ),
            None,
        )
        if evaluation is None:
            raise ValueError(f"已复核架次在当前物理口径下不可行：{output_root}:{trip_id}")
        battery_id = row["电池编号"]
        start_s = float(row["开始时刻（s）"])
        return_s = start_s + evaluation.duration_s
        recharge_s = recharge_time_s(
            evaluation.end_soc, batteries[battery_id].full_charge_time_s
        )
        trips.append(ScheduledTrip(
            trip_id,
            route,
            evaluation,
            row["无人机编号"],
            battery_id,
            start_s,
            return_s,
            recharge_s,
            return_s + recharge_s,
        ))
    objective = Objective(
        0,
        0.0,
        0.0,
        max((trip.return_s for trip in trips), default=0.0),
        sum(trip.evaluation.energy_kwh for trip in trips),
        len(trips),
    )
    return ScheduleResult(tuple(trips), objective)


def run(
    policy: DeadlinePolicy,
    output_directory: Path | None = None,
    *,
    outer_seeds: int | None = None,
    outer_iterations: int | None = None,
    initial_candidate_limit: int | None = None,
    feedback_rounds: int | None = None,
    candidates_per_round: int | None = None,
    milp_time_limit_per_stage_s: float | None = None,
    additional_route_sources: tuple[Path, ...] = (),
) -> dict[str, object]:
    base = load_toml("configs/base.toml")
    config_name = (
        "configs/q2b_joint_soft.toml"
        if policy == DeadlinePolicy.ORDINARY_SOFT
        else "configs/q2b_joint_hard.toml"
    )
    config = load_toml(config_name)
    optimization = config["optimization"]
    effective_outer_seeds = (
        int(optimization["seeds_per_profile"])
        if outer_seeds is None else outer_seeds
    )
    effective_outer_iterations = (
        int(optimization["max_iterations"])
        if outer_iterations is None else outer_iterations
    )
    effective_candidate_limit = (
        int(optimization["initial_candidate_limit"])
        if initial_candidate_limit is None else initial_candidate_limit
    )
    effective_feedback_rounds = (
        int(optimization["feedback_rounds"])
        if feedback_rounds is None else feedback_rounds
    )
    effective_candidates_per_round = (
        int(optimization["candidates_per_round"])
        if candidates_per_round is None else candidates_per_round
    )
    effective_milp_time_limit = (
        float(optimization["milp_time_limit_per_stage_s"])
        if milp_time_limit_per_stage_s is None else milp_time_limit_per_stage_s
    )
    paths = base["paths"]
    nodes = load_nodes(project_path(paths["node_workbook"]))
    boxes_by_service = load_boxes(project_path(paths["cargo_workbook"]))
    boxes = {box.box_id: box for values in boxes_by_service.values() for box in values}
    drones = load_transport_drones(project_path(paths["transport_workbook"]))
    units, batteries = load_transport_resources(project_path(paths["transport_workbook"]))
    legs = load_leg_geometry_cache(
        project_path(config["cache"]["leg_geometry"]),
        nodes,
        float(base["physics"]["service_operation_height_m"]),
    )
    evaluator = RouteEvaluator(boxes, drones, legs)

    # B+-S 与 B+-H 共享旧B3的强ALNS候选生成器。外层固定按全硬口径
    # 产生保守、零违约的运输结构；policy 只在MILP联合排程层生效。
    # 这样两版具有完全一致的强搜索能力，比较仅反映时间窗口径。
    outer = solve_q2_alns(
        boxes_by_service,
        nodes,
        drones,
        units,
        batteries,
        legs,
        seed_base=int(optimization["seed_base"]),
        seeds_per_profile=effective_outer_seeds,
        max_iterations=effective_outer_iterations,
        stall_iterations=int(optimization["stall_iterations"]),
        time_limit_per_run_s=float(optimization["time_limit_per_run_s"]),
        min_removal_count=int(optimization["min_removal_count"]),
        max_removal_count=int(optimization["max_removal_count"]),
        reaction_factor=float(optimization["reaction_factor"]),
        update_period=int(optimization["update_period"]),
        initial_temperature=float(optimization["initial_temperature"]),
        cooling_rate=float(optimization["cooling_rate"]),
        log_period=int(optimization["log_period"]),
        warm_start=_load_scheme_a_warm_start(),
        profiles=TIME_PROFILE,
        objective_mode="time_lex",
        enable_critical_operators=True,
        critical_operator_period=3,
        critical_exact_top_k=int(optimization["exact_top_k"]),
        exact_top_k=int(optimization["exact_top_k"]),
    )

    initial_candidates = tuple(sorted(
        outer.pareto_front,
        key=lambda schedule: _objective_key(schedule, policy),
    )[:effective_candidate_limit])
    legacy_b3_source = project_path(config["warm_start"]["legacy_b3_output"])
    if not legacy_b3_source.exists():
        raise FileNotFoundError(
            f"B+要求旧B3路线作为必备候选，但未找到：{legacy_b3_source}"
        )
    mandatory_sources = (legacy_b3_source, *additional_route_sources)
    additional_schedules = tuple(
        (
            f"跨口径候选:{source.name}",
            _load_schedule(source, evaluator, batteries),
        )
        for source in mandatory_sources
    )
    additional_route_sets = tuple(
        (
            label,
            tuple(trip.route for trip in schedule.trips),
            schedule,
        )
        for label, schedule in additional_schedules
    )
    joint = refine_with_milp_feedback(
        initial_candidates,
        evaluator,
        boxes,
        units,
        batteries,
        policy,
        rounds=effective_feedback_rounds,
        candidates_per_round=effective_candidates_per_round,
        time_limit_per_stage_s=effective_milp_time_limit,
        mip_rel_gap=float(optimization["milp_rel_gap"]),
        additional_route_sets=additional_route_sets,
    )
    best = joint.best.schedule
    output_root = output_directory or project_path(config["output"]["directory"])
    table_root = output_root / "tables"
    if policy == DeadlinePolicy.ORDINARY_SOFT:
        checks = validate_soft(best, boxes, drones, units, batteries, evaluator)
        assert_soft_valid(checks)
        write_soft_schedule(table_root / "最终方案", best, boxes, checks)
    else:
        checks = validate_hard(best, boxes, drones, units, batteries, evaluator)
        assert_hard_valid(checks)
        write_hard_schedule(table_root / "最终方案", best, boxes, checks)

    write_csv(table_root / "MILP逐层求解记录.csv", [
        "目标层", "目标值", "对偶界", "MIP_gap", "状态", "运行时间_s",
    ], [[
        stage.objective,
        stage.value,
        stage.dual_bound,
        stage.mip_gap,
        stage.status,
        stage.wall_time_s,
    ] for stage in joint.best.stages])
    write_csv(table_root / "联合改进候选记录.csv", [
        "反馈轮次", "候选来源", "架次数", "是否最终采用", "硬违约数",
        "硬迟到_s", "普通物资加权迟到", "完工时间_s", "能耗_kWh",
        "MILP各层均最优", "变量数", "约束数",
    ], [[
        row.round_index,
        row.candidate_label,
        row.route_count,
        "是" if row.accepted else "否",
        row.objective.hard_violation_count,
        row.objective.hard_tardiness_s,
        row.objective.weighted_soft_tardiness,
        row.objective.makespan_s,
        row.objective.energy_kwh,
        "是" if row.all_milp_stages_optimal else "否",
        row.variable_count,
        row.constraint_count,
    ] for row in joint.records])
    check_counts = Counter("通过" if row.passed else "失败" for row in checks)
    metadata = {
        "algorithm": "B3 ALNS transport search + Top-K event-driven screening + continuous-time MILP resource scheduling + critical-resource feedback",
        "outer_search_implementation": "solve_q2_alns (B3 strong skeleton)",
        "outer_search_deadline_policy": "all_hard shared candidate pool",
        "deadline_policy": policy.value,
        "outer_best_objective": asdict(outer.best.objective),
        "joint_best_objective": asdict(best.objective),
        "evaluated_route_sets": joint.evaluated_route_sets,
        "additional_route_sources": [str(path) for path in mandatory_sources],
        "milp_all_stages_optimal": joint.best.all_stages_optimal,
        "milp_backend": joint.best.backend,
        "milp_stages": [asdict(stage) for stage in joint.best.stages],
        "checks": dict(check_counts),
        "configuration": dict(optimization),
        "effective_configuration": {
            "seed_base": int(optimization["seed_base"]),
            "seeds_per_profile": effective_outer_seeds,
            "max_iterations": effective_outer_iterations,
            "stall_iterations": int(optimization["stall_iterations"]),
            "time_limit_per_run_s": float(optimization["time_limit_per_run_s"]),
            "min_removal_count": int(optimization["min_removal_count"]),
            "max_removal_count": int(optimization["max_removal_count"]),
            "reaction_factor": float(optimization["reaction_factor"]),
            "update_period": int(optimization["update_period"]),
            "initial_temperature": float(optimization["initial_temperature"]),
            "cooling_rate": float(optimization["cooling_rate"]),
            "log_period": int(optimization["log_period"]),
            "exact_top_k": int(optimization["exact_top_k"]),
            "initial_candidate_limit": effective_candidate_limit,
            "feedback_rounds": effective_feedback_rounds,
            "candidates_per_round": effective_candidates_per_round,
            "milp_time_limit_per_stage_s": effective_milp_time_limit,
            "milp_rel_gap": float(optimization["milp_rel_gap"]),
        },
    }
    write_json(output_root / "run_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="运行问题二强化版B的ALNS-MILP联合优化")
    parser.add_argument("--policy", choices=[item.value for item in DeadlinePolicy], required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--outer-seeds", type=int, default=None)
    parser.add_argument("--outer-iterations", type=int, default=None)
    parser.add_argument("--initial-candidate-limit", type=int, default=None)
    parser.add_argument("--feedback-rounds", type=int, default=None)
    parser.add_argument("--candidates-per-round", type=int, default=None)
    parser.add_argument("--milp-time-limit-per-stage", type=float, default=None)
    parser.add_argument(
        "--additional-route-source",
        action="append",
        type=Path,
        default=[],
        help="加入另一已复核结果的运输结构并用当前时间窗口径重新排程；可重复指定",
    )
    args = parser.parse_args()
    metadata = run(
        DeadlinePolicy(args.policy),
        args.output_dir,
        outer_seeds=args.outer_seeds,
        outer_iterations=args.outer_iterations,
        initial_candidate_limit=args.initial_candidate_limit,
        feedback_rounds=args.feedback_rounds,
        candidates_per_round=args.candidates_per_round,
        milp_time_limit_per_stage_s=args.milp_time_limit_per_stage,
        additional_route_sources=tuple(args.additional_route_source),
    )
    objective = metadata["joint_best_objective"]
    print(
        f"联合优化完成：硬违约{objective['hard_violation_count']}，"
        f"软迟到{objective['weighted_soft_tardiness']:.3f}，"
        f"完工{objective['makespan_s']:.3f}s，"
        f"能耗{objective['energy_kwh']:.6f}kWh，"
        f"{objective['trip_count']}架次"
    )


if __name__ == "__main__":
    main()
