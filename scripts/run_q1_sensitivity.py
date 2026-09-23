#!/usr/bin/env python3
"""问题一返航安全余量敏感性分析。"""

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_rescue.cli.q1 import run  # noqa: E402
from uav_rescue.io.writers import write_csv  # noqa: E402


DEFAULT_RATIOS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def read_rows(path: Path) -> list[dict[str, str]]:
    """读取问题一写出的 UTF-8-SIG CSV 结果。"""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ratio_label(ratio: float) -> str:
    """把安全余量转为稳定的目录名和表格标签。"""

    return f"{ratio:.0%}"


def plot_summary(rows: list[list[object]], output_path: Path, dpi: int) -> None:
    """绘制安全余量变化下的架次数、总能耗和累计作业时间。"""

    import matplotlib.pyplot as plt

    from uav_rescue.visualization.common import configure_matplotlib

    configure_matplotlib()
    feasible = [row for row in rows if row[1] == "可行"]
    labels = [str(row[0]) for row in feasible]
    trips = [int(row[2]) for row in feasible]
    energies = [float(row[3]) for row in feasible]
    durations = [float(row[4]) / 3600.0 for row in feasible]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(labels, trips, marker="o", color="#4472C4")
    axes[0].set_title("总往返架次数")
    axes[0].set_ylabel("架次")
    axes[1].plot(labels, energies, marker="o", color="#ED7D31")
    axes[1].set_title("总运输能耗")
    axes[1].set_ylabel("kWh")
    axes[2].plot(labels, durations, marker="o", color="#70AD47")
    axes[2].set_title("累计作业时间")
    axes[2].set_ylabel("h")
    for axis in axes:
        axis.set_xlabel("返航安全余量")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_payload_means(rows: list[list[object]], output_path: Path, dpi: int) -> None:
    """绘制各机型跨服务区平均最大安全载荷的变化。"""

    import matplotlib.pyplot as plt

    from uav_rescue.visualization.common import configure_matplotlib

    configure_matplotlib()
    by_model: dict[str, list[tuple[str, float]]] = {"A": [], "B": [], "C": []}
    for ratio, model, mean_payload, _feasible_services in rows:
        if mean_payload != "":
            by_model[str(model)].append((str(ratio), float(mean_payload)))
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    for model, color in zip(("A", "B", "C"), ("#4472C4", "#ED7D31", "#70AD47")):
        values = by_model[model]
        axis.plot([item[0] for item in values], [item[1] for item in values], marker="o", label=f"{model} 型", color=color)
    axis.set_xlabel("返航安全余量")
    axis.set_ylabel("平均最大安全载荷（kg）")
    axis.set_title("返航安全余量对各机型安全载荷的影响")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def run_sensitivity(
    ratios: tuple[float, ...],
    output_root: Path,
    dpi: int = 180,
    generate_plots: bool = True,
) -> None:
    """对多个返航安全余量独立求解，并汇总供论文分析的表格与图形。"""

    summary_rows: list[list[object]] = []
    payload_rows: list[list[object]] = []
    payload_mean_rows: list[list[object]] = []
    service_rows: list[list[object]] = []
    for ratio in ratios:
        label = ratio_label(ratio)
        scenario_dir = output_root / "scenarios" / f"reserve_{label}"
        try:
            result = run(scenario_dir, generate_plots=False, reserve_ratio_override=ratio)
        except RuntimeError as error:
            summary_rows.append([label, "不可行", "", "", "", str(error)])
            continue
        summary_rows.append([
            label, "可行", int(result["trips"]), result["energy_kwh"], result["duration_s"], ""
        ])
        safe_rows = read_rows(scenario_dir / "tables" / "最大安全载荷.csv")
        summary_by_service = read_rows(scenario_dir / "tables" / "组批汇总.csv")
        grouping_rows = read_rows(scenario_dir / "tables" / "货箱组批方案.csv")
        for row in safe_rows:
            payload_rows.append([
                label, row["服务区"], row["机型"], row["最大安全载荷_kg"], row["额定载荷_kg"],
                row["空载往返可行性"],
            ])
        for model in ("A", "B", "C"):
            values = [
                float(row["最大安全载荷_kg"])
                for row in safe_rows
                if row["机型"] == model and row["空载往返可行性"] == "可行"
            ]
            payload_mean_rows.append([label, model, sum(values) / len(values) if values else "", len(values)])
        models_by_service: dict[str, set[str]] = {}
        for row in grouping_rows:
            models_by_service.setdefault(row["服务区"], set()).add(row["机型"])
        for row in summary_by_service:
            if row["服务区"] == "全部":
                continue
            service_rows.append([
                label, row["服务区"], row["架次数"], row["总能耗_kWh"], row["累计作业时间_s"],
                "+".join(sorted(models_by_service[row["服务区"]])),
            ])

    write_csv(output_root / "返航余量情景汇总.csv", [
        "返航安全余量", "可行性", "总往返架次数", "总运输能耗_kWh", "累计作业时间_s", "不可行原因"
    ], summary_rows)
    write_csv(output_root / "最大安全载荷敏感性.csv", [
        "返航安全余量", "服务区", "机型", "最大安全载荷_kg", "额定载荷_kg", "空载往返可行性"
    ], payload_rows)
    write_csv(output_root / "各机型平均安全载荷.csv", [
        "返航安全余量", "机型", "平均最大安全载荷_kg", "可行服务区数"
    ], payload_mean_rows)
    write_csv(output_root / "服务区组批敏感性.csv", [
        "返航安全余量", "服务区", "架次数", "总能耗_kWh", "累计作业时间_s", "采用机型"
    ], service_rows)
    if generate_plots:
        plot_summary(summary_rows, output_root / "figures" / "返航余量对总体指标的影响.png", dpi)
        plot_payload_means(payload_mean_rows, output_root / "figures" / "返航余量对平均安全载荷的影响.png", dpi)


def render_existing_plots(output_root: Path, dpi: int) -> None:
    """只依据已完成的情景汇总表重绘图形，避免重复执行全部优化。"""

    summary_rows = [
        [
            row["返航安全余量"], row["可行性"], row["总往返架次数"],
            row["总运输能耗_kWh"], row["累计作业时间_s"], row["不可行原因"],
        ]
        for row in read_rows(output_root / "返航余量情景汇总.csv")
    ]
    payload_mean_rows = [
        [row["返航安全余量"], row["机型"], row["平均最大安全载荷_kg"], row["可行服务区数"]]
        for row in read_rows(output_root / "各机型平均安全载荷.csv")
    ]
    plot_summary(summary_rows, output_root / "figures" / "返航余量对总体指标的影响.png", dpi)
    plot_payload_means(payload_mean_rows, output_root / "figures" / "返航余量对平均安全载荷的影响.png", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行问题一返航安全余量敏感性分析")
    parser.add_argument("--ratios", nargs="*", type=float, default=list(DEFAULT_RATIOS), help="余量比例，例如 0.1 0.2 0.3")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "q1" / "sensitivity")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-plots", action="store_true", help="仅输出情景表，不绘制敏感性图")
    parser.add_argument("--plots-only", action="store_true", help="只根据已有情景表绘制敏感性图")
    args = parser.parse_args()
    ratios = tuple(sorted(set(args.ratios)))
    if not ratios or any(ratio < 0 or ratio >= 1 for ratio in ratios):
        raise ValueError("返航安全余量必须位于 [0, 1) 内")
    if args.plots_only:
        render_existing_plots(args.output_dir, args.dpi)
        print(f"问题一敏感性图已重绘：{args.output_dir / 'figures'}")
        return
    run_sensitivity(ratios, args.output_dir, args.dpi, generate_plots=not args.no_plots)
    print(f"问题一敏感性分析完成：{len(ratios)} 个余量情景，结果目录：{args.output_dir}")


if __name__ == "__main__":
    main()
