"""问题一完整计算入口。"""

import argparse
from pathlib import Path

from uav_rescue.geo.dem import DemGrid
from uav_rescue.io.readers import load_boxes, load_nodes, load_transport_drones
from uav_rescue.io.writers import write_csv
from uav_rescue.models.q1_batching import generate_candidates, solve_exact_partition
from uav_rescue.physics.energy import maximum_safe_payload_kg
from uav_rescue.physics.flight import build_leg, simulate_leg_from_geometry
from uav_rescue.settings import PROJECT_ROOT, load_toml, project_path
from uav_rescue.validation.q1 import validate_q1_solution


def run(
    output_directory: Path | None = None,
    generate_plots: bool = True,
    reserve_ratio_override: float | None = None,
) -> dict[str, float]:
    """读取附件、求解问题一、校验并输出表格和图片。"""

    base = load_toml("configs/base.toml")
    q1 = load_toml("configs/q1.toml")
    paths = base["paths"]
    physics = base["physics"]
    override = float(q1["optimization"]["reserve_ratio_override"])
    if reserve_ratio_override is not None:
        override = reserve_ratio_override
    reserve_override = None if override < 0 else override

    nodes = load_nodes(project_path(paths["node_workbook"]))
    boxes_by_service = load_boxes(project_path(paths["cargo_workbook"]))
    drones = load_transport_drones(project_path(paths["transport_workbook"]), reserve_override)
    dem = DemGrid(project_path(paths["dem"]))
    output_root = output_directory or project_path(q1["output"]["directory"])
    table_dir, figure_dir = output_root / "tables", output_root / "figures"
    origin = nodes["O01"]

    geometry_rows: list[list[object]] = []
    safe_rows_csv: list[list[object]] = []
    safe_rows_plot: list[dict] = []
    legs = {}
    for service_id in sorted(boxes_by_service):
        service = nodes[service_id]
        outbound = build_leg(
            dem, origin, service, origin.ground_m,
            service.ground_m + physics["service_operation_height_m"],
            physics["terrain_clearance_m"],
        )
        inbound = build_leg(
            dem, service, origin,
            service.ground_m + physics["service_operation_height_m"], origin.ground_m,
            physics["terrain_clearance_m"],
        )
        legs[service_id] = outbound, inbound
        geometry_rows.append([
            service_id, outbound.distance_m, outbound.cruise_altitude_m,
            outbound.climb_m, outbound.descent_m, inbound.climb_m,
            inbound.descent_m, outbound.traversed_cells,
        ])
        for model, drone in sorted(drones.items()):
            safe = maximum_safe_payload_kg(drone, outbound, inbound)
            limit = (1.0 - drone.reserve_ratio) * drone.usable_energy_kwh
            empty_outbound = simulate_leg_from_geometry(drone, outbound, 0.0)
            empty_inbound = simulate_leg_from_geometry(drone, inbound, 0.0)
            if safe is None:
                # 某机型不可行不应阻止其他机型完成该服务区的组批；显式保留该信息供敏感性分析使用。
                safe_rows_csv.append([
                    service_id, model, "", drone.max_payload_kg, drone.volume_m3,
                    empty_outbound.energy_kwh + empty_inbound.energy_kwh, "", limit, "不可行",
                ])
                continue
            boundary_outbound = simulate_leg_from_geometry(drone, outbound, safe)
            safe_rows_csv.append([
                service_id, model, safe, drone.max_payload_kg, drone.volume_m3,
                empty_outbound.energy_kwh + empty_inbound.energy_kwh,
                boundary_outbound.energy_kwh + empty_inbound.energy_kwh, limit, "可行",
            ])
            safe_rows_plot.append({"service_id": service_id, "model": model, "safe_payload_kg": safe})

    solutions = {}
    grouping_rows: list[list[object]] = []
    summary_rows: list[list[object]] = []
    total_trips, total_energy, total_duration = 0, 0.0, 0.0
    for service_id in sorted(boxes_by_service):
        boxes = boxes_by_service[service_id]
        outbound, inbound = legs[service_id]
        candidates = generate_candidates(service_id, boxes, drones, outbound, inbound)
        score, trips = solve_exact_partition(boxes, candidates)
        solutions[service_id] = trips
        total_trips += score[0]
        total_energy += score[1]
        total_duration += score[2]
        summary_rows.append([service_id, len(boxes), score[0], score[1], score[2]])
        for trip_index, trip in enumerate(sorted(trips, key=lambda item: (item.model, -item.mass_kg)), start=1):
            selected = [boxes[index] for index in range(len(boxes)) if trip.mask & (1 << index)]
            drone = drones[trip.model]
            grouping_rows.append([
                service_id, trip_index, trip.model,
                ";".join(box.box_id for box in selected),
                ";".join(box.category for box in selected),
                len(selected), trip.mass_kg, trip.volume_m3, trip.energy_kwh,
                trip.duration_s, 1.0 - trip.energy_kwh / drone.usable_energy_kwh,
                (1.0 - drone.reserve_ratio) * drone.usable_energy_kwh - trip.energy_kwh,
            ])

    validate_q1_solution(boxes_by_service, drones, solutions)
    write_csv(table_dir / "航段几何.csv", [
        "服务区", "单程水平距离_m", "巡航海拔_m", "去程爬升_m", "去程下降_m",
        "返程爬升_m", "返程下降_m", "穿越DEM像元数"], geometry_rows)
    write_csv(table_dir / "最大安全载荷.csv", [
        "服务区", "机型", "最大安全载荷_kg", "额定载荷_kg", "可用体积_m3",
        "空载往返能耗_kWh", "边界载荷往返能耗_kWh", "允许能耗上限_kWh", "空载往返可行性"], safe_rows_csv)
    write_csv(table_dir / "货箱组批方案.csv", [
        "服务区", "架次序号", "机型", "货箱编号", "物资类型", "箱数", "总质量_kg",
        "总体积_m3", "往返能耗_kWh", "作业时间_s", "返航SOC", "相对允许能耗上限余量_kWh"], grouping_rows)
    write_csv(table_dir / "组批汇总.csv", [
        "服务区", "货箱数", "架次数", "总能耗_kWh", "累计作业时间_s"],
        summary_rows + [["全部", sum(len(items) for items in boxes_by_service.values()), total_trips, total_energy, total_duration]])

    if generate_plots:
        # 延迟导入绘图库，使缺少 matplotlib 时仍可单独检查计算模块。
        from uav_rescue.visualization.q1_plots import plot_safe_payloads, plot_trip_counts

        dpi = int(q1["output"]["figure_dpi"])
        plot_safe_payloads(safe_rows_plot, figure_dir / "最大安全载荷.png", dpi)
        plot_trip_counts(summary_rows, figure_dir / "各服务区架次数.png", dpi)
    return {"boxes": 80, "trips": total_trips, "energy_kwh": total_energy, "duration_s": total_duration}


def main() -> None:
    parser = argparse.ArgumentParser(description="求解问题一并输出结果表和图片")
    parser.add_argument("--output-dir", type=Path, default=None, help="可选的结果输出目录")
    parser.add_argument("--no-plots", action="store_true", help="跳过 matplotlib 绘图，仅计算表格")
    args = parser.parse_args()
    result = run(args.output_dir, generate_plots=not args.no_plots)
    print(f"问题一完成：{result['boxes']} 箱，{result['trips']} 架次，{result['energy_kwh']:.6f} kWh，{result['duration_s']:.2f} s")
    print(f"项目根目录：{PROJECT_ROOT}")


if __name__ == "__main__":
    main()
