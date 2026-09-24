"""问题二方案B的收敛与Pareto图。"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from uav_rescue.solvers.q2_alns import PROFILE_WEIGHTS, Q2ALNSResult
from uav_rescue.visualization.common import configure_matplotlib


def plot_alns_convergence(result: Q2ALNSResult, output_path: Path, dpi: int = 180) -> None:
    configure_matplotlib()
    grouped = defaultdict(list)
    for row in result.convergence:
        grouped[(row.profile, row.seed)].append(row)
    colors = {
        "完工时间优先": "#4472C4",
        "均衡": "#70AD47",
        "资源节约": "#ED7D31",
    }
    weights = dict(PROFILE_WEIGHTS)
    scales = (
        result.baseline.objective.makespan_s,
        result.baseline.objective.energy_kwh,
        float(result.baseline.objective.trip_count),
    )
    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    labelled: set[str] = set()
    for (profile, _), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: item.iteration)
        axis.step(
            [item.iteration for item in rows],
            [
                sum(
                    weight * value / scale
                    for weight, value, scale in zip(
                        weights[profile],
                        (
                            item.best_objective.makespan_s,
                            item.best_objective.energy_kwh,
                            float(item.best_objective.trip_count),
                        ),
                        scales,
                    )
                )
                for item in rows
            ],
            where="post",
            color=colors.get(profile, "#777777"),
            alpha=0.38,
            linewidth=1.0,
            label=profile if profile not in labelled else None,
        )
        labelled.add(profile)
    axis.set_xlabel("迭代次数")
    axis.set_ylabel("当前最佳归一化目标 F")
    axis.set_title("ALNS多种子收敛过程")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_front(result: Q2ALNSResult, output_path: Path, dpi: int = 180) -> None:
    configure_matplotlib()
    fig, axis = plt.subplots(figsize=(8.4, 5.8))
    trips = [schedule.objective.trip_count for schedule in result.pareto_front]
    scatter = axis.scatter(
        [schedule.objective.makespan_s for schedule in result.pareto_front],
        [schedule.objective.energy_kwh for schedule in result.pareto_front],
        c=trips,
        cmap="viridis_r",
        s=90,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )
    for index, schedule in enumerate(result.pareto_front, start=1):
        axis.annotate(
            f"P{index} / {schedule.objective.trip_count}架",
            (schedule.objective.makespan_s, schedule.objective.energy_kwh),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("全部任务完成时间（s）")
    axis.set_ylabel("总能耗（kWh）")
    axis.set_title("零硬违约方案的Pareto前沿")
    axis.grid(alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("架次数")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
