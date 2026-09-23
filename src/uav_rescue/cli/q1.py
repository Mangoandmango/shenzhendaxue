"""问题一完整计算入口。"""

import argparse
from dataclasses import replace
from pathlib import Path

from uav_rescue.geo.dem import DemGrid
from uav_rescue.io.readers import load_boxes, load_nodes, load_transport_drones
from uav_rescue.io.writers import write_csv
from uav_rescue.models.q1_batching import (
    combine_pareto_fronts_by_trip_count,
    epsilon_representatives,
    find_reserve_breakpoints,
    generate_candidates,
    prune_dominated_candidates,
    simple_trip_lower_bound,
    solve_exact_partition,
    solve_pareto_partitions,
)
from uav_rescue.physics.energy import maximum_safe_payload_kg
from uav_rescue.physics.flight import build_leg, simulate_leg_from_geometry
from uav_rescue.settings import PROJECT_ROOT, load_toml, project_path
from uav_rescue.validation.q1 import validate_q1_solution


def run(
    output_directory: Path | None = None,
    generate_plots: bool = True,
    reserve_ratio_override: float | None = None,
    compute_pareto: bool = True,
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
    safe_payloads: dict[str, dict[str, float | None]] = {}
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
            service_id, outbound.distance_m,
            outbound.cruise_altitude_m - physics["terrain_clearance_m"], outbound.cruise_altitude_m,
            outbound.climb_m, outbound.descent_m, inbound.climb_m,
            inbound.descent_m, outbound.traversed_cells,
        ])
        safe_payloads[service_id] = {}
        for model, drone in sorted(drones.items()):
            safe = maximum_safe_payload_kg(drone, outbound, inbound)
            safe_payloads[service_id][model] = safe
            limit = (1.0 - drone.reserve_ratio) * drone.usable_energy_kwh
            empty_outbound = simulate_leg_from_geometry(drone, outbound, 0.0)
            empty_inbound = simulate_leg_from_geometry(drone, inbound, 0.0)
            full_outbound = simulate_leg_from_geometry(drone, outbound, drone.max_payload_kg)
            empty_energy = empty_outbound.energy_kwh + empty_inbound.energy_kwh
            full_energy = full_outbound.energy_kwh + empty_inbound.energy_kwh
            full_payload_critical_reserve = 1.0 - full_energy / drone.usable_energy_kwh
            empty_trip_critical_reserve = 1.0 - empty_energy / drone.usable_energy_kwh
            if safe is None:
                # 某机型不可行不应阻止其他机型完成该服务区的组批；显式保留该信息供敏感性分析使用。
                safe_rows_csv.append([
                    service_id, model, "", drone.max_payload_kg, drone.volume_m3,
                    empty_energy, "", limit, "空载往返不可行",
                    full_payload_critical_reserve, empty_trip_critical_reserve, "不可行",
                ])
                continue
            boundary_outbound = simulate_leg_from_geometry(drone, outbound, safe)
            limiting_reason = "额定载重" if abs(safe - drone.max_payload_kg) <= 1e-7 else "能源安全余量"
            safe_rows_csv.append([
                service_id, model, safe, drone.max_payload_kg, drone.volume_m3,
                empty_energy, boundary_outbound.energy_kwh + empty_inbound.energy_kwh, limit,
                limiting_reason, full_payload_critical_reserve, empty_trip_critical_reserve, "可行",
            ])
            safe_rows_plot.append({"service_id": service_id, "model": model, "safe_payload_kg": safe})

    solutions = {}
    grouping_rows: list[list[object]] = []
    summary_rows: list[list[object]] = []
    lower_bound_rows: list[list[object]] = []
    candidate_stat_rows: list[list[object]] = []
    total_trips, total_energy, total_duration = 0, 0.0, 0.0
    local_pareto_fronts_by_count = []
    extra_trip_levels = int(q1["optimization"].get("tradeoff_extra_trips", 2))
    for service_id in sorted(boxes_by_service):
        boxes = boxes_by_service[service_id]
        outbound, inbound = legs[service_id]
        raw_candidates = generate_candidates(service_id, boxes, drones, outbound, inbound)
        candidates = prune_dominated_candidates(raw_candidates)
        score, trips = solve_exact_partition(boxes, candidates)
        if compute_pareto:
            local_pareto_fronts_by_count.append(
                solve_pareto_partitions(boxes, candidates, score[0] + extra_trip_levels)
            )
        solutions[service_id] = trips
        total_trips += score[0]
        total_energy += score[1]
        total_duration += score[2]
        summary_rows.append([service_id, len(boxes), score[0], score[1], score[2]])
        mass_bound, volume_bound, simple_bound = simple_trip_lower_bound(
            boxes, safe_payloads[service_id], drones
        )
        lower_bound_rows.append([
            service_id, sum(box.mass_kg for box in boxes), sum(box.volume_m3 for box in boxes),
            mass_bound, volume_bound, simple_bound, score[0], score[0] - simple_bound,
            "达到简单下界" if score[0] == simple_bound else "不可拆货箱及机型联合容量使简单下界不紧",
        ])
        candidate_stat_rows.append([
            service_id, len(raw_candidates), len(candidates), len(raw_candidates) - len(candidates)
        ])
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

    validation_records = validate_q1_solution(boxes_by_service, drones, solutions, legs)
    write_csv(table_dir / "航段几何.csv", [
        "服务区", "单程水平距离_m", "沿线最高地形_m", "巡航海拔_m", "去程爬升_m", "去程下降_m",
        "返程爬升_m", "返程下降_m", "穿越DEM像元数"], geometry_rows)
    write_csv(table_dir / "最大安全载荷.csv", [
        "服务区", "机型", "最大安全载荷_kg", "额定载荷_kg", "可用体积_m3",
        "空载往返能耗_kWh", "边界载荷往返能耗_kWh", "允许能耗上限_kWh", "安全载荷受限原因",
        "额定载荷临界安全余量", "空载可行临界安全余量", "空载往返可行性"], safe_rows_csv)
    write_csv(table_dir / "货箱组批方案.csv", [
        "服务区", "架次序号", "机型", "货箱编号", "物资类型", "箱数", "总质量_kg",
        "总体积_m3", "往返能耗_kWh", "作业时间_s", "返航SOC", "相对允许能耗上限余量_kWh"], grouping_rows)
    write_csv(table_dir / "组批汇总.csv", [
        "服务区", "货箱数", "架次数", "总能耗_kWh", "累计作业时间_s"],
        summary_rows + [["全部", sum(len(items) for items in boxes_by_service.values()), total_trips, total_energy, total_duration]])
    write_csv(table_dir / "架次数下界验证.csv", [
        "服务区", "总质量_kg", "总体积_m3", "质量下界_架次", "体积下界_架次",
        "简单综合下界_架次", "精确最优架次", "下界差距_架次", "解释",
    ], lower_bound_rows)
    write_csv(table_dir / "候选架次筛选统计.csv", [
        "服务区", "筛选前候选数", "筛选后候选数", "删除支配候选数",
    ], candidate_stat_rows)
    write_csv(table_dir / "独立复核.csv", [
        "服务区", "架次序号", "机型", "货箱编号", "重算总质量_kg", "重算总体积_m3",
        "重算往返能耗_kWh", "重算累计作业时间_s", "重算返航SOC", "复核结果",
    ], [
        [record.service_id, record.trip_index, record.model, record.box_ids,
         record.recomputed_mass_kg, record.recomputed_volume_m3, record.recomputed_energy_kwh,
         record.recomputed_duration_s, record.return_soc, record.result]
        for record in validation_records
    ])

    if compute_pareto:
        global_fronts_by_count = combine_pareto_fronts_by_trip_count(
            local_pareto_fronts_by_count, total_trips + extra_trip_levels
        )
        global_frontier = global_fronts_by_count[total_trips]
        frontier_rows = [
            [index + 1, item.trip_count, item.energy_kwh, item.duration_s]
            for index, item in enumerate(sorted(global_frontier, key=lambda item: (item.energy_kwh, item.duration_s)))
        ]
        epsilon_rows = [
            [label, item.trip_count, item.energy_kwh, item.duration_s, item.duration_s]
            for label, item in epsilon_representatives(global_frontier)
        ]
        write_csv(table_dir / "固定最少架次数Pareto前沿.csv", [
            "Pareto序号", "总往返架次数", "总运输能耗_kWh", "累计作业时间_s"
        ], frontier_rows)
        write_csv(table_dir / "ε约束代表解.csv", [
            "代表方案", "总往返架次数", "总运输能耗_kWh", "累计作业时间_s", "ε_累计作业时间上限_s"
        ], epsilon_rows)
        tradeoff_rows = []
        for trip_count in range(total_trips, total_trips + extra_trip_levels + 1):
            frontier = global_fronts_by_count.get(trip_count, [])
            if not frontier:
                continue
            minimum_energy = min(frontier, key=lambda item: (item.energy_kwh, item.duration_s))
            minimum_duration = min(frontier, key=lambda item: (item.duration_s, item.energy_kwh))
            tradeoff_rows.append([
                trip_count, len(frontier),
                minimum_energy.energy_kwh, minimum_energy.duration_s,
                minimum_duration.duration_s, minimum_duration.energy_kwh,
                100.0 * (minimum_energy.energy_kwh / total_energy - 1.0),
                100.0 * (minimum_duration.duration_s / total_duration - 1.0),
            ])
        write_csv(table_dir / "跨架次数权衡.csv", [
            "总往返架次数", "Pareto解数量", "该架次数最低能耗_kWh", "最低能耗方案累计作业时间_s",
            "该架次数最短累计作业时间_s", "最短时间方案总能耗_kWh",
            "最低能耗较最少架次方案变化_pct", "最短时间较最少架次方案变化_pct",
        ], tradeoff_rows)

        # 在零余量候选全集上定位离散架次数变化点；阈值处仍可行，超过阈值后发生表中变化。
        zero_reserve_drones = {
            model: replace(drone, reserve_ratio=0.0) for model, drone in drones.items()
        }
        breakpoint_events: dict[float, list[tuple[str, object]]] = {}
        for service_id in sorted(boxes_by_service):
            outbound, inbound = legs[service_id]
            zero_candidates = prune_dominated_candidates(generate_candidates(
                service_id, boxes_by_service[service_id], zero_reserve_drones, outbound, inbound
            ))
            for breakpoint in find_reserve_breakpoints(
                boxes_by_service[service_id], zero_candidates, zero_reserve_drones
            ):
                breakpoint_events.setdefault(breakpoint.reserve_ratio, []).append((service_id, breakpoint))
        running_counts: dict[str, int | None] = {}
        for events in breakpoint_events.values():
            for service_id, breakpoint in events:
                running_counts.setdefault(service_id, breakpoint.before_trip_count)
        breakpoint_rows = []
        for reserve_ratio, events in sorted(breakpoint_events.items()):
            for service_id, breakpoint in events:
                running_counts[service_id] = breakpoint.after_trip_count
            global_count = (
                "不可行" if any(value is None for value in running_counts.values())
                else sum(int(value) for value in running_counts.values())
            )
            for service_id, breakpoint in events:
                breakpoint_rows.append([
                    reserve_ratio, service_id,
                    breakpoint.before_trip_count if breakpoint.before_trip_count is not None else "不可行",
                    breakpoint.after_trip_count if breakpoint.after_trip_count is not None else "不可行",
                    global_count,
                ])
        write_csv(table_dir / "返航余量架次数临界点.csv", [
            "临界返航安全余量", "发生变化的服务区", "变化前最少架次数",
            "超过临界值后最少架次数", "变化后全局最少架次数",
        ], breakpoint_rows)

    if generate_plots:
        # 延迟导入绘图库，使缺少 matplotlib 时仍可单独检查计算模块。
        from uav_rescue.visualization.q1_plots import plot_safe_payloads, plot_trip_counts

        dpi = int(q1["output"]["figure_dpi"])
        plot_safe_payloads(safe_rows_plot, figure_dir / "最大安全载荷.png", dpi)
        plot_trip_counts(summary_rows, figure_dir / "各服务区架次数.png", dpi)
    return {
        # 从清洗后的实际数据动态统计，避免将当前附件的 80 箱写死在程序中。
        "boxes": sum(len(items) for items in boxes_by_service.values()),
        "trips": total_trips,
        "energy_kwh": total_energy,
        "duration_s": total_duration,
        "pareto_points": len(global_frontier) if compute_pareto else 0,
    }


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
