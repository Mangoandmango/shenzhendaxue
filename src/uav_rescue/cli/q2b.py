"""问题二方案B的完整运行、输出与独立复核入口。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import statistics

from uav_rescue.io.readers import (
    load_boxes,
    load_leg_geometry_cache,
    load_nodes,
    load_transport_drones,
    load_transport_resources,
)
from uav_rescue.io.writers import write_csv, write_json
from uav_rescue.models.q2_transport import (
    RouteEvaluator,
    RoutePlan,
    TypedRoutePlan,
    hard_deadline_s,
)
from uav_rescue.settings import load_toml, project_path
from uav_rescue.solvers.q2_alns import (
    DESTROY_OPERATORS,
    PROFILE_WEIGHTS,
    Q2ALNSResult,
    REPAIR_OPERATORS,
    solve_q2_alns,
)
from uav_rescue.solvers.q2_solver import normalized_ideal_distance, pareto_vector
from uav_rescue.validation.q2 import assert_q2_valid, validate_q2_solution
from uav_rescue.visualization.q2b_plots import plot_alns_convergence, plot_pareto_front


def _objective_row(label: str, objective) -> list[object]:
    return [
        label,
        objective.hard_violation_count,
        objective.hard_tardiness_s,
        objective.makespan_s,
        objective.energy_kwh,
        objective.trip_count,
    ]


def _write_schedule(directory: Path, schedule, boxes: dict, checks) -> None:
    write_csv(directory / "运输架次.csv", [
        "架次编号", "无人机编号", "机型", "电池编号", "开始时刻（s）",
        "访问服务区顺序", "返回O01时刻（s）", "架次能耗（kWh）", "返航SOC",
    ], [[
        trip.trip_id,
        trip.unit_id,
        trip.evaluation.model,
        trip.battery_id,
        trip.start_s,
        "->".join(trip.route.service_sequence),
        trip.return_s,
        trip.evaluation.energy_kwh,
        trip.evaluation.end_soc,
    ] for trip in schedule.trips])
    write_csv(directory / "逐箱交付.csv", [
        "货箱编号", "架次编号", "服务区", "物资类型", "要求时刻（s）",
        "交付完成时刻（s）", "松弛量（s）", "是否按时",
    ], [[
        box_id,
        trip.trip_id,
        boxes[box_id].service_id,
        boxes[box_id].category,
        hard_deadline_s(boxes[box_id]),
        trip.delivery_time(box_id),
        hard_deadline_s(boxes[box_id]) - trip.delivery_time(box_id),
        "是" if trip.delivery_time(box_id) <= hard_deadline_s(boxes[box_id]) + 1e-8 else "否",
    ] for trip in schedule.trips for box_id in sorted(trip.route.box_ids)])
    write_csv(directory / "无人机资源占用.csv", [
        "无人机编号", "机型", "架次编号", "开始时刻（s）", "返航时刻（s）",
    ], sorted([
        [trip.unit_id, trip.evaluation.model, trip.trip_id, trip.start_s, trip.return_s]
        for trip in schedule.trips
    ]))
    write_csv(directory / "电池任务与充电周转.csv", [
        "电池编号", "机型", "架次编号", "任务开始（s）", "返航时刻（s）",
        "返航SOC", "充电时长（s）", "充至100%时刻（s）",
    ], sorted([[
        trip.battery_id,
        trip.evaluation.model,
        trip.trip_id,
        trip.start_s,
        trip.return_s,
        trip.evaluation.end_soc,
        trip.recharge_s,
        trip.battery_release_s,
    ] for trip in schedule.trips]))
    write_csv(directory / "逐航段明细.csv", [
        "架次编号", "航段序号", "起点", "终点", "航段前载荷（kg）",
        "飞行时间（s）", "能耗（kWh）",
    ], [[
        trip.trip_id,
        index,
        leg.origin,
        leg.destination,
        leg.payload_before_kg,
        leg.flight_time_s,
        leg.energy_kwh,
    ] for trip in schedule.trips for index, leg in enumerate(trip.evaluation.legs, start=1)])
    write_csv(directory / "独立复核.csv", [
        "检查项", "对象编号", "计算值", "允许值", "是否通过", "说明",
    ], [[
        row.check,
        row.object_id,
        row.calculated,
        row.allowed,
        "通过" if row.passed else "失败",
        row.detail,
    ] for row in checks])


def _existing_scheme_a_objective() -> dict[str, object] | None:
    path = project_path("outputs/q2/run_metadata.json")
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle).get("best_objective")


def _load_scheme_a_warm_start() -> tuple[TypedRoutePlan, ...] | None:
    """从已经独立复核的增强型方案A结果构造ALNS热启动解。"""

    trip_path = project_path("outputs/q2/tables/Q2_运输架次.csv")
    box_path = project_path("outputs/q2/tables/Q2_逐箱交付.csv")
    if not trip_path.exists() or not box_path.exists():
        return None
    rows: dict[str, dict[str, object]] = {}
    with trip_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["架次编号"]] = {
                "model": row["机型编号"],
                "services": tuple(row["访问服务区顺序"].split("->")),
                "boxes": [],
            }
    with box_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["架次编号"]]["boxes"].append(row["货箱编号"])
    return tuple(
        TypedRoutePlan(
            RoutePlan(tuple(value["boxes"]), tuple(value["services"])),
            str(value["model"]),
        )
        for _, value in sorted(rows.items())
    )


def _improvement(new_value: float, old_value: float) -> float:
    return 100.0 * (old_value - new_value) / old_value if old_value else 0.0


def _write_report(
    output_root: Path,
    result: Q2ALNSResult,
    scheme_a: dict[str, object] | None,
    checks,
    ablation_results: list[tuple[str, Q2ALNSResult]],
) -> None:
    best = result.best.objective
    baseline = result.baseline.objective
    initial = result.greedy_initial.objective
    values = [record.best_objective.makespan_s for record in result.records]
    runtimes = [record.elapsed_s for record in result.records]
    scheme_a_line = ""
    if scheme_a is not None:
        scheme_a_line = (
            f"\n- 相对增强型方案A：完工时间改变 "
            f"{_improvement(best.makespan_s, float(scheme_a['makespan_s'])):.3f}%，"
            f"能耗改变 {_improvement(best.energy_kwh, float(scheme_a['energy_kwh'])):.3f}%，"
            f"架次数由 {scheme_a['trip_count']} 变为 {best.trip_count}。"
        )
    ablation_text = "\n".join(
        f"- {label}：{variant.best.objective.trip_count} 架次，"
        f"完工 {variant.best.objective.makespan_s:.6f} s，"
        f"能耗 {variant.best.objective.energy_kwh:.9f} kWh。"
        for label, variant in ablation_results
    )
    report = f"""# 问题二方案B计算结果与分析

## 计算口径

方案B使用带机型的有序架次编码，ALNS直接修改货箱组批、服务区顺序、架次边界和机型。每个候选解都由事件驱动解码器联合分配实体无人机、共享电池和开始时刻。所有80个货箱的期望送达时间均作为硬时间窗；首批货箱另受首批截止时间约束，取两者较早值。

## 主要结果

- 硬时限违反：{best.hard_violation_count} 箱，总迟到 {best.hard_tardiness_s:.6f} s。
- 综合折中方案：{best.trip_count} 架次，完工时间 {best.makespan_s:.6f} s，总能耗 {best.energy_kwh:.9f} kWh。
- 相对ALNS热启动初始解：完工时间改善 {_improvement(best.makespan_s, initial.makespan_s):.3f}%，能耗改善 {_improvement(best.energy_kwh, initial.energy_kwh):.3f}%，架次数由 {initial.trip_count} 变为 {best.trip_count}。
- 相对单点直投基线：完工时间改善 {_improvement(best.makespan_s, baseline.makespan_s):.3f}%，能耗改善 {_improvement(best.energy_kwh, baseline.energy_kwh):.3f}%。{scheme_a_line}

## 随机搜索与复核

- 独立运行数：{len(result.records)}，三类权重分别为 {', '.join(name for name, _ in PROFILE_WEIGHTS)}。
- 各运行最佳完工时间：均值 {statistics.fmean(values):.6f} s，标准差 {statistics.pstdev(values):.6f} s。
- 单次运行时间：均值 {statistics.fmean(runtimes):.3f} s，最大 {max(runtimes):.3f} s。
- Pareto前沿点数：{len(result.pareto_front)}。所有可行解迟到指标均为0，因此前沿实际由完工时间、能耗和架次数构成。
- 独立复核：{sum(row.passed for row in checks)}/{len(checks)} 项通过。

## 消融实验

{ablation_text}

消融实验使用较少的固定随机种子和迭代预算，用于判断组件的方向性贡献，不与完整方案做等计算预算的显著性结论。

## 结果解释边界

ALNS结果是固定算法参数、搜索算子和随机种子集下得到的最优可行解，不声称全局最优。运输架次、逐箱送达、无人机与电池时序、收敛记录、算子权重和逐项复核均已输出至 `outputs/q2b/tables/`。
"""
    (output_root / "问题二方案B计算结果与分析.md").write_text(report, encoding="utf-8")


def run(
    output_directory: Path | None = None,
    seeds_per_profile: int | None = None,
    max_iterations: int | None = None,
    stall_iterations: int | None = None,
    time_limit_per_run_s: float | None = None,
) -> dict[str, object]:
    base = load_toml("configs/base.toml")
    q2b = load_toml("configs/q2b.toml")
    paths = base["paths"]
    physics = base["physics"]
    optimization = q2b["optimization"]
    output_root = output_directory or project_path(q2b["output"]["directory"])
    table_root = output_root / "tables"

    nodes = load_nodes(project_path(paths["node_workbook"]))
    boxes_by_service = load_boxes(project_path(paths["cargo_workbook"]))
    boxes = {box.box_id: box for values in boxes_by_service.values() for box in values}
    drones = load_transport_drones(project_path(paths["transport_workbook"]))
    units, batteries = load_transport_resources(project_path(paths["transport_workbook"]))
    legs = load_leg_geometry_cache(
        project_path(q2b["cache"]["leg_geometry"]),
        nodes,
        float(physics["service_operation_height_m"]),
    )
    warm_start = _load_scheme_a_warm_start()
    result = solve_q2_alns(
        boxes_by_service,
        nodes,
        drones,
        units,
        batteries,
        legs,
        seed_base=int(optimization["seed_base"]),
        seeds_per_profile=int(optimization["seeds_per_profile"])
        if seeds_per_profile is None else seeds_per_profile,
        max_iterations=int(optimization["max_iterations"])
        if max_iterations is None else max_iterations,
        stall_iterations=int(optimization["stall_iterations"])
        if stall_iterations is None else stall_iterations,
        time_limit_per_run_s=float(optimization["time_limit_per_run_s"])
        if time_limit_per_run_s is None else time_limit_per_run_s,
        min_removal_count=int(optimization["min_removal_count"]),
        max_removal_count=int(optimization["max_removal_count"]),
        reaction_factor=float(optimization["reaction_factor"]),
        update_period=int(optimization["update_period"]),
        initial_temperature=float(optimization["initial_temperature"]),
        cooling_rate=float(optimization["cooling_rate"]),
        log_period=int(optimization["log_period"]),
        warm_start=warm_start,
    )
    ablation_results: list[tuple[str, Q2ALNSResult]] = []
    ablation = q2b.get("ablation", {})
    if bool(ablation.get("enabled", False)):
        common_ablation = {
            "seed_base": int(optimization["seed_base"]),
            "seeds_per_profile": int(ablation["seeds_per_profile"]),
            "max_iterations": int(ablation["max_iterations"]),
            "stall_iterations": int(ablation["stall_iterations"]),
            "time_limit_per_run_s": float(ablation["time_limit_per_run_s"]),
            "min_removal_count": int(optimization["min_removal_count"]),
            "max_removal_count": int(optimization["max_removal_count"]),
            "reaction_factor": float(optimization["reaction_factor"]),
            "update_period": int(optimization["update_period"]),
            "initial_temperature": float(optimization["initial_temperature"]),
            "cooling_rate": float(optimization["cooling_rate"]),
            "log_period": int(optimization["log_period"]),
            "warm_start": warm_start,
        }
        variants = (
            ("关闭自适应权重", {"reaction_factor": 0.0}),
            ("关闭资源瓶颈破坏", {
                "enabled_destroy_operators": tuple(
                    name for name in DESTROY_OPERATORS if name != "resource_bottleneck"
                )
            }),
            ("关闭regret插入", {
                "enabled_repair_operators": tuple(
                    name for name in REPAIR_OPERATORS if name not in {"regret2", "regret3"}
                )
            }),
        )
        for label, overrides in variants:
            parameters = common_ablation | overrides
            ablation_results.append((label, solve_q2_alns(
                boxes_by_service, nodes, drones, units, batteries, legs, **parameters
            )))
    evaluator = RouteEvaluator(boxes, drones, legs)
    checks = validate_q2_solution(result.best, boxes, drones, units, batteries, evaluator)
    assert_q2_valid(checks)
    representative_checks = {
        name: validate_q2_solution(schedule, boxes, drones, units, batteries, evaluator)
        for name, schedule in result.representatives.items()
    }
    for rows in representative_checks.values():
        assert_q2_valid(rows)

    _write_schedule(table_root, result.best, boxes, checks)
    for name, schedule in result.representatives.items():
        _write_schedule(
            output_root / "pareto" / name,
            schedule,
            boxes,
            representative_checks[name],
        )
    write_csv(table_root / "ALNS独立运行统计.csv", [
        "权重方案", "随机种子", "迭代次数", "运行时间（s）", "首次可行迭代",
        "接受移动数", "改进移动数", "停止原因", "硬时限违反数",
        "完工时间（s）", "总能耗（kWh）", "架次数",
    ], [[
        record.profile,
        record.seed,
        record.iterations,
        record.elapsed_s,
        record.feasible_iteration,
        record.accepted_moves,
        record.improving_moves,
        record.stop_reason,
        record.best_objective.hard_violation_count,
        record.best_objective.makespan_s,
        record.best_objective.energy_kwh,
        record.best_objective.trip_count,
    ] for record in result.records])
    write_csv(table_root / "ALNS收敛记录.csv", [
        "权重方案", "随机种子", "迭代", "累计运行时间（s）", "温度", "破坏算子",
        "修复算子", "是否接受", "当前硬违约", "当前完工时间", "当前能耗",
        "当前架次", "最佳硬违约", "最佳完工时间", "最佳能耗", "最佳架次",
    ], [[
        row.profile,
        row.seed,
        row.iteration,
        row.elapsed_s,
        row.temperature,
        row.destroy_operator,
        row.repair_operator,
        "是" if row.accepted else "否",
        row.current_objective.hard_violation_count,
        row.current_objective.makespan_s,
        row.current_objective.energy_kwh,
        row.current_objective.trip_count,
        row.best_objective.hard_violation_count,
        row.best_objective.makespan_s,
        row.best_objective.energy_kwh,
        row.best_objective.trip_count,
    ] for row in result.convergence])
    write_csv(table_root / "ALNS算子权重.csv", [
        "权重方案", "随机种子", "算子类型", "算子", "最终权重", "总使用次数", "总得分",
    ], [[
        row.profile,
        row.seed,
        row.kind,
        row.operator,
        row.final_weight,
        row.uses,
        row.score,
    ] for row in result.operator_records])
    write_csv(table_root / "ALNS消融实验.csv", [
        "实验组", "硬时限违反数", "完工时间（s）", "总能耗（kWh）", "架次数",
        "相对完整方案完工变化（%）", "能耗变化（%）", "架次变化",
    ], [[
        label,
        variant.best.objective.hard_violation_count,
        variant.best.objective.makespan_s,
        variant.best.objective.energy_kwh,
        variant.best.objective.trip_count,
        -_improvement(variant.best.objective.makespan_s, result.best.objective.makespan_s),
        -_improvement(variant.best.objective.energy_kwh, result.best.objective.energy_kwh),
        variant.best.objective.trip_count - result.best.objective.trip_count,
    ] for label, variant in [(
        "完整方案B", result
    ), *ablation_results]])
    scheme_a = _existing_scheme_a_objective()
    comparison = [
        _objective_row("单点直投基线", result.baseline.objective),
        _objective_row("ALNS初始解（增强型方案A热启动）", result.greedy_initial.objective),
    ]
    if scheme_a is not None:
        comparison.append([
            "增强型方案A",
            scheme_a["hard_violation_count"],
            scheme_a["hard_tardiness_s"],
            scheme_a["makespan_s"],
            scheme_a["energy_kwh"],
            scheme_a["trip_count"],
        ])
    comparison.append(_objective_row("方案B ALNS综合折中", result.best.objective))
    write_csv(table_root / "方案对比.csv", [
        "方案", "硬时限违反数", "硬时限总迟到（s）", "完工时间（s）",
        "总能耗（kWh）", "架次数",
    ], comparison)
    write_csv(table_root / "Pareto前沿.csv", [
        "Pareto点", "完工时间（s）", "总能耗（kWh）", "架次数",
        "归一化完工时间", "归一化能耗", "归一化架次", "理想点距离",
    ], [[
        index,
        schedule.objective.makespan_s,
        schedule.objective.energy_kwh,
        schedule.objective.trip_count,
        normalized[1],
        normalized[2],
        normalized[3],
        distance,
    ] for index, schedule in enumerate(result.pareto_front, start=1)
       for distance, normalized in [normalized_ideal_distance(schedule, result.pareto_front)]])
    write_csv(table_root / "Pareto代表方案.csv", [
        "方案", "完工时间（s）", "总能耗（kWh）", "架次数", "理想点距离",
    ], [[
        name,
        schedule.objective.makespan_s,
        schedule.objective.energy_kwh,
        schedule.objective.trip_count,
        normalized_ideal_distance(schedule, result.pareto_front)[0],
    ] for name, schedule in result.representatives.items()])

    counts = Counter("通过" if row.passed else "失败" for row in checks)
    metadata = {
        "algorithm": "ALNS + event-driven joint drone-battery scheduling",
        "deadline_policy": "所有货箱期望送达时间均为硬时间窗；首批取期望与首批截止的较早值",
        "configuration": {
            key: optimization[key] for key in optimization
        } | {
            "seeds_per_profile": int(optimization["seeds_per_profile"])
            if seeds_per_profile is None else seeds_per_profile,
            "max_iterations": int(optimization["max_iterations"])
            if max_iterations is None else max_iterations,
            "stall_iterations": int(optimization["stall_iterations"])
            if stall_iterations is None else stall_iterations,
            "time_limit_per_run_s": float(optimization["time_limit_per_run_s"])
            if time_limit_per_run_s is None else time_limit_per_run_s,
        },
        "profile_weights": {name: weights for name, weights in PROFILE_WEIGHTS},
        "best_objective": asdict(result.best.objective),
        "greedy_initial_objective": asdict(result.greedy_initial.objective),
        "baseline_objective": asdict(result.baseline.objective),
        "scheme_a_objective": scheme_a,
        "run_count": len(result.records),
        "pareto_count": len(result.pareto_front),
        "checks": dict(counts),
        "all_representatives_passed": all(
            all(row.passed for row in rows) for rows in representative_checks.values()
        ),
        "ablation": {
            label: asdict(variant.best.objective)
            for label, variant in ablation_results
        },
    }
    write_json(output_root / "run_metadata.json", metadata)
    plot_alns_convergence(result, output_root / "figures" / "ALNS收敛曲线.png")
    plot_pareto_front(result, output_root / "figures" / "Pareto前沿.png")
    _write_report(output_root, result, scheme_a, checks, ablation_results)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="求解问题二方案B ALNS与事件驱动调度")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds-per-profile", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--stall-iterations", type=int, default=None)
    parser.add_argument("--time-limit-per-run", type=float, default=None)
    args = parser.parse_args()
    metadata = run(
        args.output_dir,
        args.seeds_per_profile,
        args.max_iterations,
        args.stall_iterations,
        args.time_limit_per_run,
    )
    objective = metadata["best_objective"]
    print(
        "问题二方案B完成："
        f"硬时限违反 {objective['hard_violation_count']} 箱，"
        f"{objective['trip_count']} 架次，"
        f"完工 {objective['makespan_s']:.2f} s，"
        f"能耗 {objective['energy_kwh']:.6f} kWh"
    )


if __name__ == "__main__":
    main()
