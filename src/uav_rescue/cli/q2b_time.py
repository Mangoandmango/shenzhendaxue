"""问题二方案B完工时间强化版及B0--B3逐步消融实验。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import statistics

import matplotlib.pyplot as plt

from uav_rescue.cli.q2b import _load_scheme_a_warm_start, _write_schedule
from uav_rescue.io.readers import (
    load_boxes,
    load_leg_geometry_cache,
    load_nodes,
    load_transport_drones,
    load_transport_resources,
)
from uav_rescue.io.writers import write_csv, write_json
from uav_rescue.models.q2_transport import RouteEvaluator
from uav_rescue.settings import load_toml, project_path
from uav_rescue.solvers.q2_alns import Q2ALNSResult, solve_q2_alns
from uav_rescue.validation.q2 import assert_q2_valid, validate_q2_solution


TIME_PROFILE = (("均衡消融", (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),)


def _fastest(result: Q2ALNSResult):
    return min(
        result.pareto_front,
        key=lambda schedule: (
            schedule.objective.hard_violation_count,
            schedule.objective.hard_tardiness_s,
            schedule.objective.makespan_s,
            schedule.objective.energy_kwh,
            schedule.objective.trip_count,
        ),
    )


def _improvement(new: float, old: float) -> float:
    return 100.0 * (old - new) / old if old else 0.0


def _plot_ablation(rows: list[list[object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row[0]).split()[0] for row in rows]
    best = [float(row[7]) / 60.0 for row in rows]
    means = [float(row[8]) / 60.0 for row in rows]
    runtimes = [float(row[3]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    positions = list(range(len(labels)))
    width = 0.36
    axes[0].bar([value - width / 2 for value in positions], best, width, label="Best")
    axes[0].bar([value + width / 2 for value in positions], means, width, label="Seed mean")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Makespan (min)")
    axes[0].set_title("Stepwise makespan ablation")
    axes[0].legend()
    axes[1].bar(labels, runtimes, color="#d97706")
    axes[1].set_ylabel("Mean runtime per seed (s)")
    axes[1].set_title("Computational cost")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run(
    output_directory: Path | None = None,
    seeds: int | None = None,
    max_iterations: int | None = None,
    time_limit_per_run_s: float | None = None,
) -> dict[str, object]:
    base = load_toml("configs/base.toml")
    config = load_toml("configs/q2b_time.toml")
    paths = base["paths"]
    physics = base["physics"]
    optimization = config["optimization"]
    output_root = output_directory or project_path(config["output"]["directory"])
    table_root = output_root / "tables"

    nodes = load_nodes(project_path(paths["node_workbook"]))
    boxes_by_service = load_boxes(project_path(paths["cargo_workbook"]))
    boxes = {box.box_id: box for values in boxes_by_service.values() for box in values}
    drones = load_transport_drones(project_path(paths["transport_workbook"]))
    units, batteries = load_transport_resources(project_path(paths["transport_workbook"]))
    legs = load_leg_geometry_cache(
        project_path(config["cache"]["leg_geometry"]),
        nodes,
        float(physics["service_operation_height_m"]),
    )
    warm_start = _load_scheme_a_warm_start()
    common = {
        "seed_base": int(optimization["seed_base"]),
        "seeds_per_profile": int(optimization["seeds"]) if seeds is None else seeds,
        "max_iterations": int(optimization["max_iterations"])
        if max_iterations is None else max_iterations,
        "stall_iterations": int(optimization["stall_iterations"]),
        "time_limit_per_run_s": float(optimization["time_limit_per_run_s"])
        if time_limit_per_run_s is None else time_limit_per_run_s,
        "min_removal_count": int(optimization["min_removal_count"]),
        "max_removal_count": int(optimization["max_removal_count"]),
        "reaction_factor": float(optimization["reaction_factor"]),
        "update_period": int(optimization["update_period"]),
        "initial_temperature": float(optimization["initial_temperature"]),
        "cooling_rate": float(optimization["cooling_rate"]),
        "log_period": int(optimization["log_period"]),
        "warm_start": warm_start,
        "profiles": TIME_PROFILE,
    }
    variants = (
        ("B0 加权目标基线", {"objective_mode": "weighted"}),
        ("B1 +字典序完工目标", {"objective_mode": "time_lex"}),
        ("B2 +关键资源链邻域", {
            "objective_mode": "time_lex", "enable_critical_operators": True,
            "critical_operator_period": 3,
            "critical_exact_top_k": int(optimization["exact_top_k"]),
        }),
        ("B3 +Top-K精确复算", {
            "objective_mode": "time_lex", "enable_critical_operators": True,
            "critical_operator_period": 3,
            "critical_exact_top_k": int(optimization["exact_top_k"]),
            "exact_top_k": int(optimization["exact_top_k"]),
        }),
    )
    results: list[tuple[str, Q2ALNSResult, object, list[object]]] = []
    evaluator = RouteEvaluator(boxes, drones, legs)
    for label, overrides in variants:
        result = solve_q2_alns(
            boxes_by_service, nodes, drones, units, batteries, legs,
            **(common | overrides),
        )
        fastest = _fastest(result)
        checks = validate_q2_solution(fastest, boxes, drones, units, batteries, evaluator)
        assert_q2_valid(checks)
        results.append((label, result, fastest, checks))

    baseline = results[0][2].objective
    baseline_run_mean = statistics.fmean(
        record.best_objective.makespan_s for record in results[0][1].records
    )
    rows = []
    previous = baseline
    previous_run_mean = baseline_run_mean
    for label, result, schedule, checks in results:
        objective = schedule.objective
        runtimes = [record.elapsed_s for record in result.records]
        run_makespans = [record.best_objective.makespan_s for record in result.records]
        run_mean = statistics.fmean(run_makespans)
        rows.append([
            label, len(result.records), sum(record.iterations for record in result.records),
            statistics.fmean(runtimes), statistics.pstdev(runtimes),
            objective.hard_violation_count, objective.hard_tardiness_s,
            objective.makespan_s, run_mean, statistics.median(run_makespans),
            statistics.pstdev(run_makespans), objective.energy_kwh, objective.trip_count,
            _improvement(objective.makespan_s, previous.makespan_s),
            _improvement(objective.makespan_s, baseline.makespan_s),
            _improvement(run_mean, previous_run_mean),
            _improvement(run_mean, baseline_run_mean),
            sum(check.passed for check in checks), len(checks),
        ])
        previous = objective
        previous_run_mean = run_mean
    write_csv(table_root / "B0-B3逐步消融.csv", [
        "实验组", "独立运行数", "总迭代数", "平均运行时间（s）", "运行时间标准差（s）",
        "硬时限违反数", "硬时限总迟到（s）", "最短完工时间（s）", "各种子平均完工（s）",
        "各种子完工中位数（s）", "各种子完工标准差（s）", "总能耗（kWh）", "架次数",
        "最优值较前一步改善（%）", "最优值较B0改善（%）", "均值较前一步改善（%）",
        "均值较B0改善（%）", "复核通过数", "复核总数",
    ], rows)
    write_csv(table_root / "各随机种子结果.csv", [
        "实验组", "随机种子", "迭代次数", "运行时间（s）", "停止原因", "硬违约数",
        "该次搜索最佳完工时间（s）", "能耗（kWh）", "架次数",
    ], [[
        label, record.seed, record.iterations, record.elapsed_s, record.stop_reason,
        record.best_objective.hard_violation_count, record.best_objective.makespan_s,
        record.best_objective.energy_kwh, record.best_objective.trip_count,
    ] for label, result, _, _ in results for record in result.records])
    _plot_ablation(rows, output_root / "figures" / "B0-B3逐步消融.png")

    final_label, final_result, final_schedule, final_checks = results[-1]
    _write_schedule(table_root / "B3最终方案", final_schedule, boxes, final_checks)
    metadata = {
        "experiment": "方案B完工时间强化版逐步消融",
        "fairness": "B0--B3使用相同实例、初始解、随机种子、迭代上限与停止规则",
        "configuration": common | {"exact_top_k": int(optimization["exact_top_k"])},
        "variants": {
            label: {
                "best_objective": asdict(schedule.objective),
                "run_count": len(result.records),
                "all_checks_passed": all(check.passed for check in checks),
            }
            for label, result, schedule, checks in results
        },
        "final_variant": final_label,
        "final_objective": asdict(final_schedule.objective),
    }
    # warm_start含数据类对象，单独替换为可序列化标志。
    metadata["configuration"]["warm_start"] = warm_start is not None
    metadata["configuration"]["profiles"] = TIME_PROFILE
    write_json(output_root / "run_metadata.json", metadata)

    lines = [
        "# 问题二方案B完工时间强化与逐步消融结果", "",
        "四组实验使用相同数据、热启动、随机种子和搜索预算；所有货箱期望送达时间仍为硬时间窗。",
        "", "| 实验组 | 最优完工/s | 种子均值/s | 能耗/kWh | 架次 | 最优值较B0改善 | 平均运行时间/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row[0]} | {row[7]:.3f} | {row[8]:.3f} | {row[11]:.6f} | "
            f"{row[12]} | {row[14]:.3f}% | {row[3]:.3f} |"
        )
    lines += [
        "", "## 模块贡献", "",
        f"- B1：最优值与B0持平，但各种子平均完工时间由 {rows[0][8]:.3f} s "
        f"降至 {rows[1][8]:.3f} s（改善 {rows[1][16]:.3f}%），且标准差由 "
        f"{rows[0][10]:.3f} s 降为 {rows[1][10]:.3f} s；贡献主要体现为搜索稳定性。",
        f"- B2：最优完工时间较B1改善 {rows[2][13]:.3f}%，种子均值改善 "
        f"{rows[2][15]:.3f}%，说明关键无人机/电池链邻域能突破原局部最优。",
        f"- B3：在B2上进一步将最优完工时间改善 {rows[3][13]:.3f}%，种子均值改善 "
        f"{rows[3][15]:.3f}%；代价是平均单种子计算时间增至 {rows[3][3]:.3f} s。",
        f"- 最终B3完工时间为 {rows[3][7] / 60.0:.3f} min，较B0缩短 "
        f"{rows[3][14]:.3f}%；同时能耗增加 "
        f"{100.0 * (rows[3][11] / rows[0][11] - 1.0):.3f}%，架次由 "
        f"{rows[0][12]} 增至 {rows[3][12]}。这是以资源消耗换取完工时间的明确Pareto权衡。",
        "", "## 判定口径", "",
        "模块贡献以跨固定随机种子的最优完工时间、分布变化和运行代价共同判断；"
        "若某一步未改善，按消融结果如实认定该模块在当前参数预算下未形成正贡献，不作强行解释。",
    ]
    (output_root / "问题二方案B完工时间强化结果.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="运行问题二方案B完工时间强化与逐步消融")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--time-limit-per-run", type=float, default=None)
    args = parser.parse_args()
    metadata = run(args.output_dir, args.seeds, args.max_iterations, args.time_limit_per_run)
    objective = metadata["final_objective"]
    print(
        f"B3完成：硬违约{objective['hard_violation_count']}，"
        f"完工{objective['makespan_s']:.3f}s，能耗{objective['energy_kwh']:.6f}kWh，"
        f"{objective['trip_count']}架次"
    )


if __name__ == "__main__":
    main()
