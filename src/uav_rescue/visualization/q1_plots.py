"""问题一安全载荷与组批结果图。"""

from pathlib import Path

import matplotlib.pyplot as plt

from uav_rescue.visualization.common import configure_matplotlib


def plot_safe_payloads(rows: list[dict], output_path: Path, dpi: int = 180) -> None:
    """绘制三种机型在各服务区的最大安全载荷分组柱状图。"""

    configure_matplotlib()
    services = sorted({row["service_id"] for row in rows})
    models = ["A", "B", "C"]
    lookup = {(row["service_id"], row["model"]): row["safe_payload_kg"] for row in rows}
    x = list(range(len(services)))
    width = 0.25
    fig, axis = plt.subplots(figsize=(12, 5.5))
    for index, model in enumerate(models):
        values = [lookup[(service, model)] for service in services]
        axis.bar([value + (index - 1) * width for value in x], values, width, label=f"{model} 型")
    axis.set_xticks(x, services, rotation=45)
    axis.set_ylabel("最大安全载荷（kg）")
    axis.set_title("20% 返航安全余量下的最大安全载荷")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_trip_counts(summary_rows: list[list[object]], output_path: Path, dpi: int = 180) -> None:
    """绘制各服务区最优组批的架次数。"""

    configure_matplotlib()
    services = [str(row[0]) for row in summary_rows]
    trips = [int(row[2]) for row in summary_rows]
    fig, axis = plt.subplots(figsize=(10, 4.8))
    bars = axis.bar(services, trips, color="#4472C4")
    axis.bar_label(bars, padding=3)
    axis.set_ylabel("往返架次数")
    axis.set_title("各服务区最优货箱组批架次数")
    axis.set_ylim(0, max(trips) + 0.7)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

