"""运行问题二方案B的题意混合时间窗版本。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from uav_rescue.io.readers import (
    load_boxes, load_leg_geometry_cache, load_nodes, load_transport_drones,
    load_transport_resources,
)
from uav_rescue.io.writers import write_csv, write_json
from uav_rescue.models.q2_transport_soft import RouteEvaluator, hard_deadline_s
from uav_rescue.settings import load_toml, project_path
from uav_rescue.solvers.q2_soft_solver import normalized_ideal_distance, pareto_vector
from uav_rescue.solvers.q2b_soft_alns import Q2BSoftResult, solve_q2b_soft
from uav_rescue.validation.q2_soft import assert_q2_valid, validate_q2_solution


def _write_schedule(directory: Path, schedule, boxes: dict, checks) -> None:
    write_csv(directory / "运输架次.csv", [
        "架次编号", "无人机编号", "机型", "电池编号", "开始时刻（s）", "访问服务区顺序",
        "返回O01时刻（s）", "架次能耗（kWh）", "返航SOC",
    ], [[
        trip.trip_id, trip.unit_id, trip.evaluation.model, trip.battery_id, trip.start_s,
        "->".join(trip.route.service_sequence), trip.return_s, trip.evaluation.energy_kwh,
        trip.evaluation.end_soc,
    ] for trip in schedule.trips])
    write_csv(directory / "逐箱交付与时间窗.csv", [
        "货箱编号", "架次编号", "服务区", "物资类型", "时间窗类型", "要求时刻（s）",
        "交付完成时刻（s）", "迟到（s）", "应急优先系数", "加权迟到",
    ], [[
        box_id, trip.trip_id, boxes[box_id].service_id, boxes[box_id].category,
        "硬" if hard_deadline_s(boxes[box_id]) is not None else "软",
        hard_deadline_s(boxes[box_id]) if hard_deadline_s(boxes[box_id]) is not None else boxes[box_id].expected_deadline_s,
        trip.delivery_time(box_id),
        max(0.0, trip.delivery_time(box_id) - (hard_deadline_s(boxes[box_id]) if hard_deadline_s(boxes[box_id]) is not None else boxes[box_id].expected_deadline_s)),
        boxes[box_id].priority,
        0.0 if hard_deadline_s(boxes[box_id]) is not None else (boxes[box_id].priority or 1.0) * max(0.0, trip.delivery_time(box_id) - (boxes[box_id].expected_deadline_s or float("inf"))),
    ] for trip in schedule.trips for box_id in sorted(trip.route.box_ids)])
    write_csv(directory / "无人机资源占用.csv", ["无人机编号", "机型", "架次编号", "开始时刻（s）", "返航时刻（s)"], [
        [trip.unit_id, trip.evaluation.model, trip.trip_id, trip.start_s, trip.return_s]
        for trip in sorted(schedule.trips, key=lambda item: (item.unit_id, item.start_s))
    ])
    write_csv(directory / "电池任务与充电周转.csv", [
        "电池编号", "机型", "架次编号", "任务开始（s）", "返航时刻（s）", "返航SOC",
        "充电时长（s）", "充至100%时刻（s）",
    ], [[
        trip.battery_id, trip.evaluation.model, trip.trip_id, trip.start_s, trip.return_s,
        trip.evaluation.end_soc, trip.recharge_s, trip.battery_release_s,
    ] for trip in sorted(schedule.trips, key=lambda item: (item.battery_id, item.start_s))])
    write_csv(directory / "逐航段明细.csv", ["架次编号", "航段序号", "起点", "终点", "航段前载荷（kg）", "飞行时间（s）", "能耗（kWh）"], [[
        trip.trip_id, index, leg.origin, leg.destination, leg.payload_before_kg,
        leg.flight_time_s, leg.energy_kwh,
    ] for trip in schedule.trips for index, leg in enumerate(trip.evaluation.legs, 1)])
    write_csv(directory / "独立复核.csv", ["检查项", "对象编号", "计算值", "允许值", "是否通过", "说明"], [[
        row.check, row.object_id, row.calculated, row.allowed, "通过" if row.passed else "失败", row.detail,
    ] for row in checks])


def run(output_directory: Path | None = None, seeds_per_profile: int | None = None, max_iterations: int | None = None) -> dict[str, object]:
    base, config = load_toml("configs/base.toml"), load_toml("configs/q2b_soft.toml")
    paths, physics, optimization = base["paths"], base["physics"], config["optimization"]
    root = output_directory or project_path(config["output"]["directory"])
    nodes = load_nodes(project_path(paths["node_workbook"]))
    boxes_by_service = load_boxes(project_path(paths["cargo_workbook"]))
    boxes = {box.box_id: box for values in boxes_by_service.values() for box in values}
    drones = load_transport_drones(project_path(paths["transport_workbook"]))
    units, batteries = load_transport_resources(project_path(paths["transport_workbook"]))
    legs = load_leg_geometry_cache(project_path(config["cache"]["leg_geometry"]), nodes, float(physics["service_operation_height_m"]))
    result = solve_q2b_soft(
        boxes_by_service, nodes, drones, units, batteries, legs,
        seed_base=int(optimization["seed_base"]),
        seeds_per_profile=int(optimization["seeds_per_profile"]) if seeds_per_profile is None else seeds_per_profile,
        max_iterations=int(optimization["max_iterations"]) if max_iterations is None else max_iterations,
        stall_iterations=int(optimization["stall_iterations"]),
        time_limit_per_run_s=float(optimization["time_limit_per_run_s"]),
        top_k=int(optimization["top_k"]),
    )
    evaluator = RouteEvaluator(boxes, drones, legs)
    checks = validate_q2_solution(result.best, boxes, drones, units, batteries, evaluator)
    assert_q2_valid(checks)
    _write_schedule(root / "tables" / "综合折中", result.best, boxes, checks)
    write_csv(root / "tables" / "独立运行统计.csv", [
        "偏好", "随机种子", "迭代次数", "运行时间（s）", "停止原因", "硬违约数", "硬总迟到（s）",
        "普通物资加权迟到", "完工时间（s）", "能耗（kWh）", "架次数",
    ], [[
        record.profile, record.seed, record.iterations, record.elapsed_s, record.stop_reason,
        record.best_objective.hard_violation_count, record.best_objective.hard_tardiness_s,
        record.best_objective.weighted_soft_tardiness, record.best_objective.makespan_s,
        record.best_objective.energy_kwh, record.best_objective.trip_count,
    ] for record in result.records])
    write_csv(root / "tables" / "Pareto前沿.csv", [
        "编号", "普通物资加权迟到", "完工时间（s）", "能耗（kWh）", "架次数", "理想点距离",
    ], [[
        index, *pareto_vector(schedule), normalized_ideal_distance(schedule, result.pareto_front)[0],
    ] for index, schedule in enumerate(result.pareto_front, 1)])
    for name, schedule in result.representatives.items():
        representative_checks = validate_q2_solution(schedule, boxes, drones, units, batteries, evaluator)
        assert_q2_valid(representative_checks)
        _write_schedule(root / "pareto" / name, schedule, boxes, representative_checks)
    check_counts = Counter("通过" if row.passed else "失败" for row in checks)
    metadata = {
        "algorithm": "ALNS + event-driven joint drone-battery scheduling",
        "deadline_policy": "医疗物资期望送达时间、首批保障截止时间为硬约束；其他货箱期望送达时间为按应急优先系数加权的软约束",
        "configuration": dict(optimization),
        "best_objective": asdict(result.best.objective),
        "baseline_objective": asdict(result.baseline.objective),
        "pareto_count": len(result.pareto_front), "checks": dict(check_counts),
    }
    write_json(root / "run_metadata.json", metadata)
    (root / "问题二方案B混合时间窗说明.md").write_text(
        "# 问题二方案B混合时间窗版\n\n"
        "医疗物资的期望送达时间、首批保障货箱的首批截止时间为硬约束。"
        "其他货箱的期望送达时间为软时间窗，以应急优先系数乘正迟到时间计入配送及时性目标。\n\n"
        f"综合折中方案：硬违约 {result.best.objective.hard_violation_count} 箱；"
        f"普通物资加权迟到 {result.best.objective.weighted_soft_tardiness:.6f}；"
        f"完工 {result.best.objective.makespan_s:.3f} s；能耗 {result.best.objective.energy_kwh:.6f} kWh；"
        f"架次 {result.best.objective.trip_count}。\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="运行问题二方案B混合时间窗版")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds-per-profile", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()
    metadata = run(args.output_dir, args.seeds_per_profile, args.max_iterations)
    objective = metadata["best_objective"]
    print(f"方案B混合时间窗版完成：硬违约{objective['hard_violation_count']}，加权迟到{objective['weighted_soft_tardiness']:.3f}，完工{objective['makespan_s']:.3f}s")


if __name__ == "__main__":
    main()
