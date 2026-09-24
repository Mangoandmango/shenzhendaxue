"""问题二完整计算、输出与独立复核入口。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from uav_rescue.io.readers import (
    load_boxes,
    load_leg_geometry_cache,
    load_nodes,
    load_transport_drones,
    load_transport_resources,
)
from uav_rescue.io.writers import write_csv, write_json
from uav_rescue.models.q2_transport import RouteEvaluator, hard_deadline_s
from uav_rescue.settings import load_toml, project_path
from uav_rescue.solvers.q2_solver import (
    Q2SearchResult,
    normalized_ideal_distance,
    pareto_vector,
    solve_q2,
)
from uav_rescue.validation.q2 import assert_q2_valid, validate_q2_solution


def _objective_row(label: str, objective) -> list[object]:
    return [
        label,
        objective.hard_violation_count,
        objective.hard_tardiness_s,
        objective.weighted_soft_tardiness,
        objective.makespan_s,
        objective.energy_kwh,
        objective.trip_count,
    ]


def _write_representative_tables(
    directory: Path, schedule, boxes: dict, checks
) -> None:
    write_csv(directory / "运输架次.csv", [
        "架次编号", "无人机编号", "机型编号", "电池编号", "开始时刻（s）",
        "访问服务区顺序", "返回O01时刻（s）", "架次能耗（kWh）",
    ], [[
        trip.trip_id, trip.unit_id, trip.evaluation.model, trip.battery_id,
        trip.start_s, "->".join(trip.route.service_sequence), trip.return_s,
        trip.evaluation.energy_kwh,
    ] for trip in schedule.trips])
    write_csv(directory / "逐箱交付.csv", [
        "货箱编号", "架次编号", "服务区编号", "交付完成时刻（s）",
    ], [[
        box_id, trip.trip_id, boxes[box_id].service_id, trip.delivery_time(box_id)
    ] for trip in schedule.trips for box_id in sorted(trip.route.box_ids)])
    write_csv(directory / "独立复核.csv", [
        "检查项", "对象编号", "计算值", "允许值", "是否通过", "说明",
    ], [[
        row.check, row.object_id, row.calculated, row.allowed,
        "通过" if row.passed else "失败", row.detail,
    ] for row in checks])


def _write_schedule_tables(
    output_root: Path, result: Q2SearchResult, boxes: dict, checks, representative_checks
) -> None:
    table_dir = output_root / "tables"
    schedule = result.best
    write_csv(table_dir / "Q2_运输架次.csv", [
        "架次编号", "无人机编号", "机型编号", "电池编号", "开始时刻（s）",
        "访问服务区顺序", "返回O01时刻（s）", "架次能耗（kWh）",
    ], [
        [
            trip.trip_id, trip.unit_id, trip.evaluation.model, trip.battery_id,
            trip.start_s, "->".join(trip.route.service_sequence), trip.return_s,
            trip.evaluation.energy_kwh,
        ]
        for trip in schedule.trips
    ])
    write_csv(table_dir / "Q2_逐箱交付.csv", [
        "货箱编号", "架次编号", "服务区编号", "交付完成时刻（s）",
    ], [
        [box_id, trip.trip_id, boxes[box_id].service_id, trip.delivery_time(box_id)]
        for trip in schedule.trips
        for box_id in sorted(trip.route.box_ids)
    ])
    write_csv(table_dir / "逐箱时限明细.csv", [
        "货箱编号", "架次编号", "服务区编号", "物资类型", "是否硬时限",
        "要求时刻（s）", "送达时刻（s）", "迟到时间（s）", "应急优先系数",
    ], [
        [
            box_id, trip.trip_id, boxes[box_id].service_id, boxes[box_id].category,
            "是" if hard_deadline_s(boxes[box_id]) is not None else "否",
            hard_deadline_s(boxes[box_id]) if hard_deadline_s(boxes[box_id]) is not None
            else boxes[box_id].expected_deadline_s,
            trip.delivery_time(box_id),
            max(0.0, trip.delivery_time(box_id) - (
                hard_deadline_s(boxes[box_id]) if hard_deadline_s(boxes[box_id]) is not None
                else boxes[box_id].expected_deadline_s
            )),
            boxes[box_id].priority,
        ]
        for trip in schedule.trips
        for box_id in sorted(trip.route.box_ids)
    ])
    write_csv(table_dir / "无人机资源占用.csv", [
        "无人机编号", "机型", "架次编号", "开始时刻（s）", "返航时刻（s）",
    ], sorted([
        [trip.unit_id, trip.evaluation.model, trip.trip_id, trip.start_s, trip.return_s]
        for trip in schedule.trips
    ]))
    write_csv(table_dir / "电池任务与充电周转.csv", [
        "电池编号", "机型", "架次编号", "任务开始（s）", "返航时刻（s）",
        "返航SOC", "充电时长（s）", "充至100%时刻（s）",
    ], sorted([
        [
            trip.battery_id, trip.evaluation.model, trip.trip_id, trip.start_s,
            trip.return_s, trip.evaluation.end_soc, trip.recharge_s,
            trip.battery_release_s,
        ]
        for trip in schedule.trips
    ]))
    write_csv(table_dir / "逐航段明细.csv", [
        "架次编号", "航段序号", "起点", "终点", "航段前载荷（kg）",
        "飞行时间（s）", "能耗（kWh）",
    ], [
        [
            trip.trip_id, index, leg.origin, leg.destination, leg.payload_before_kg,
            leg.flight_time_s, leg.energy_kwh,
        ]
        for trip in schedule.trips
        for index, leg in enumerate(trip.evaluation.legs, start=1)
    ])
    write_csv(table_dir / "独立复核.csv", [
        "检查项", "对象编号", "计算值", "允许值", "是否通过", "说明",
    ], [
        [row.check, row.object_id, row.calculated, row.allowed, "通过" if row.passed else "失败", row.detail]
        for row in checks
    ])
    write_csv(table_dir / "多起点搜索稳定性.csv", [
        "初始解", "随机种子", "搜索偏好", "迭代次数", "运行时间（s）", "硬时限违反数",
        "硬时限总迟到（s）", "加权迟到（硬窗，可行解为0）", "完工时间（s）",
        "总能耗（kWh）", "架次数",
    ], [
        [
            record.label, record.seed, record.preference, record.iterations, record.elapsed_s,
            record.objective.hard_violation_count, record.objective.hard_tardiness_s,
            record.objective.weighted_soft_tardiness, record.objective.makespan_s,
            record.objective.energy_kwh, record.objective.trip_count,
        ]
        for record in result.records
    ])
    write_csv(table_dir / "单点与多点方案比较.csv", [
        "方案", "硬时限违反数", "硬时限总迟到（s）", "加权迟到（硬窗，可行解为0）",
        "完工时间（s）", "总能耗（kWh）", "架次数",
    ], [
        _objective_row("单点直投基线", result.baseline.objective),
        _objective_row("Pareto综合折中方案", result.best.objective),
    ])
    representative_names: dict[tuple[float, ...], list[str]] = {}
    for name, schedule in result.representatives.items():
        representative_names.setdefault(pareto_vector(schedule), []).append(name)
    write_csv(table_dir / "Pareto前沿.csv", [
        "Pareto点", "加权迟到（硬窗，可行解为0）", "完工时间（s）", "总能耗（kWh）",
        "架次数", "归一化迟到（恒为0）", "归一化完工时间", "归一化能耗",
        "归一化架次", "理想点距离", "代表方案",
    ], [
        [
            index, *pareto_vector(schedule), *normalized,
            distance, "|".join(representative_names.get(pareto_vector(schedule), [])),
        ]
        for index, schedule in enumerate(result.pareto_front, start=1)
        for distance, normalized in [normalized_ideal_distance(schedule, result.pareto_front)]
    ])
    write_csv(table_dir / "Pareto代表方案.csv", [
        "方案", "硬时限违反数", "硬时限总迟到（s）", "加权迟到（硬窗，可行解为0）",
        "完工时间（s）", "总能耗（kWh）", "架次数", "理想点距离",
    ], [
        [*_objective_row(name, schedule.objective),
         normalized_ideal_distance(schedule, result.pareto_front)[0]]
        for name, schedule in result.representatives.items()
    ])
    for name, schedule in result.representatives.items():
        _write_representative_tables(
            output_root / "pareto" / name, schedule, boxes, representative_checks[name]
        )


def run(
    output_directory: Path | None = None,
    random_seed_count: int | None = None,
    max_iterations: int | None = None,
    stall_iterations: int | None = None,
    time_limit_per_seed_s: float | None = None,
) -> dict[str, object]:
    base = load_toml("configs/base.toml")
    q2 = load_toml("configs/q2.toml")
    paths = base["paths"]
    physics = base["physics"]
    optimization = q2["optimization"]
    output_root = output_directory or project_path(q2["output"]["directory"])

    nodes = load_nodes(project_path(paths["node_workbook"]))
    boxes_by_service = load_boxes(project_path(paths["cargo_workbook"]))
    boxes = {box.box_id: box for values in boxes_by_service.values() for box in values}
    drones = load_transport_drones(project_path(paths["transport_workbook"]))
    units, batteries = load_transport_resources(project_path(paths["transport_workbook"]))
    legs = load_leg_geometry_cache(
        project_path(q2["cache"]["leg_geometry"]),
        nodes,
        float(physics["service_operation_height_m"]),
    )
    result = solve_q2(
        boxes_by_service,
        nodes,
        drones,
        units,
        batteries,
        legs,
        seed_base=int(optimization["seed_base"]),
        random_seed_count=(
            int(optimization["random_seed_count"])
            if random_seed_count is None else random_seed_count
        ),
        max_iterations=(
            int(optimization["max_iterations"])
            if max_iterations is None else max_iterations
        ),
        stall_iterations=(
            int(optimization["stall_iterations"])
            if stall_iterations is None else stall_iterations
        ),
        time_limit_per_seed_s=(
            float(optimization["time_limit_per_seed_s"])
            if time_limit_per_seed_s is None else time_limit_per_seed_s
        ),
        repair_limit=int(optimization["repair_limit"]),
    )
    evaluator = RouteEvaluator(boxes, drones, legs)
    checks = validate_q2_solution(
        result.best, boxes, drones, units, batteries, evaluator
    )
    representative_checks = {
        name: validate_q2_solution(schedule, boxes, drones, units, batteries, evaluator)
        for name, schedule in result.representatives.items()
    }
    _write_schedule_tables(output_root, result, boxes, checks, representative_checks)
    check_counts = Counter("通过" if row.passed else "失败" for row in checks)
    metadata = {
        "configuration": {
            "seed_base": int(optimization["seed_base"]),
            "random_seed_count": int(optimization["random_seed_count"])
            if random_seed_count is None else random_seed_count,
            "max_iterations": int(optimization["max_iterations"])
            if max_iterations is None else max_iterations,
            "stall_iterations": int(optimization["stall_iterations"])
            if stall_iterations is None else stall_iterations,
            "time_limit_per_seed_s": float(optimization["time_limit_per_seed_s"])
            if time_limit_per_seed_s is None else time_limit_per_seed_s,
            "repair_limit": int(optimization["repair_limit"]),
            "battery_occupancy": "架次开始至返航后充至100%",
            "drone_occupancy": "架次开始至返航完成",
        },
        "best_objective": asdict(result.best.objective),
        "selection_rule": "所有期望送达时间为硬约束；零违约Pareto前沿上完工时间、能耗和架次的min-max归一化等权理想点距离最小",
        "pareto_count": len(result.pareto_front),
        "pareto_representatives": {
            name: asdict(schedule.objective)
            for name, schedule in result.representatives.items()
        },
        "baseline_objective": asdict(result.baseline.objective),
        "checks": dict(check_counts),
        "all_representatives_passed": all(
            all(row.passed for row in rows) for rows in representative_checks.values()
        ),
    }
    write_json(output_root / "run_metadata.json", metadata)
    assert_q2_valid(checks)
    for rows in representative_checks.values():
        assert_q2_valid(rows)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="求解问题二并生成排程、基线与独立复核结果")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-seeds", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--stall-iterations", type=int, default=None)
    parser.add_argument("--time-limit-per-seed", type=float, default=None)
    args = parser.parse_args()
    metadata = run(
        args.output_dir,
        args.random_seeds,
        args.max_iterations,
        args.stall_iterations,
        args.time_limit_per_seed,
    )
    objective = metadata["best_objective"]
    print(
        "问题二完成："
        f"硬时限违反 {objective['hard_violation_count']} 箱，"
        f"{objective['trip_count']} 架次，"
        f"完工 {objective['makespan_s']:.2f} s，"
        f"能耗 {objective['energy_kwh']:.6f} kWh"
    )


if __name__ == "__main__":
    main()
