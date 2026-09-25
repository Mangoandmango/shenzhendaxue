"""运行问题三方案 A 固定运输基线。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

from uav_rescue.geo.dem import DemGrid
from uav_rescue.geo.coordinates import LocalEnu
from uav_rescue.models.q3_joint import split_blind_intervals
from uav_rescue.settings import PROJECT_ROOT, load_toml
from uav_rescue.solvers.q3_solver import (
    build_relay_columns, find_blind_intervals, generate_candidate_sites,
    load_communication_parameters, reconstruct_transport_samples, select_and_assign_columns,
    apply_forced_trip_delays, stagger_transport_schedule,
)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    started = time.perf_counter()
    config = load_toml("configs/q3.toml")
    communication = config["communication"]
    baseline = config["baseline"]
    dem = DemGrid(PROJECT_ROOT / "data/raw/geo/镇龙乡及周边地理数据/数字高程模型数据（DEM）/镇龙乡及周边30米DEM.tif")
    params = load_communication_parameters(PROJECT_ROOT / "data/processed/communication_parameters.csv")
    step = float(communication["trajectory_time_step_s"])
    samples = reconstruct_transport_samples(
        PROJECT_ROOT, step,
        PROJECT_ROOT / baseline["transport_schedule"],
        PROJECT_ROOT / baseline["transport_deliveries"],
    )
    intervals = find_blind_intervals(PROJECT_ROOT, dem, params, samples, step)
    intervals = split_blind_intervals(intervals, float(communication["blind_service_chunk_s"]))
    node_rows = []
    with (PROJECT_ROOT / "data/processed/nodes.csv").open(encoding="utf-8-sig", newline="") as handle:
        node_rows = list(csv.DictReader(handle))
    origin = next(row for row in node_rows if row["node_id"] == "O01")
    enu = LocalEnu(float(origin["longitude_deg"]), float(origin["latitude_deg"]), 0.0)
    sites = generate_candidate_sites(
        dem, enu, intervals,
        float(baseline["candidate_grid_step_m"]), list(baseline["candidate_height_levels_m"]),
        list(baseline["candidate_ray_fractions"]), int(baseline["candidate_neighbor_radius"]),
    )
    columns = build_relay_columns(PROJECT_ROOT, dem, params, intervals, sites,
                                  int(baseline["max_candidates_per_blind_interval"]))
    output = PROJECT_ROOT / "outputs/q3_baseline"
    assigned, conflicts = select_and_assign_columns(intervals, columns)
    feedback_applied = False
    feedback_error = ""
    original_conflicts = list(conflicts)
    if conflicts and bool(baseline.get("enable_feedback", True)):
        try:
            forced = config.get("feedback", {}).get("forced_trip_delays_s", {})
            if forced:
                repaired_schedule, repaired_deliveries = apply_forced_trip_delays(
                    PROJECT_ROOT, PROJECT_ROOT / baseline["transport_schedule"],
                    PROJECT_ROOT / baseline["transport_deliveries"], forced, output / "feedback",
                )
            else:
                repaired_schedule, repaired_deliveries = stagger_transport_schedule(
                    PROJECT_ROOT, PROJECT_ROOT / baseline["transport_schedule"],
                    PROJECT_ROOT / baseline["transport_deliveries"], intervals, output / "feedback",
                )
            samples = reconstruct_transport_samples(PROJECT_ROOT, step, repaired_schedule, repaired_deliveries)
            intervals = find_blind_intervals(PROJECT_ROOT, dem, params, samples, step)
            intervals = split_blind_intervals(intervals, float(communication["blind_service_chunk_s"]))
            sites = generate_candidate_sites(
                dem, enu, intervals,
                float(baseline["candidate_grid_step_m"]), list(baseline["candidate_height_levels_m"]),
                list(baseline["candidate_ray_fractions"]), int(baseline["candidate_neighbor_radius"]),
            )
            columns = build_relay_columns(PROJECT_ROOT, dem, params, intervals, sites,
                                          int(baseline["max_candidates_per_blind_interval"]))
            assigned, conflicts = select_and_assign_columns(intervals, columns)
            feedback_applied = True
        except RuntimeError as error:
            feedback_error = str(error)
    _write_csv(output / "tables/运输轨迹采样.csv",
               ["架次编号", "时刻_s", "经度", "纬度", "飞行海拔_m", "阶段"],
               [[s.trip_id, s.time_s, s.lon, s.lat, s.altitude_m, s.phase] for s in samples])
    _write_csv(output / "tables/直连黑区间.csv",
               ["黑区编号", "架次编号", "开始时刻_s", "结束时刻_s", "采样点数"],
               [[b.interval_id, b.trip_id, b.start_s, b.end_s, len(b.samples)] for b in intervals])
    _write_csv(output / "tables/中继候选列.csv",
               ["候选列", "黑区编号", "悬停点", "经度", "纬度", "离地高度_m", "服务开始_s", "服务结束_s",
                "任务开始_s", "返航_s", "能耗_kWh", "返航SOC", "最小裕量_dB"],
               [[c.column_id, "|".join(c.covered_interval_ids), c.site.site_id, c.site.lon, c.site.lat,
                 c.site.height_agl_m, c.service_start_s, c.service_end_s, c.mission_start_s, c.return_s,
                 c.energy_kwh, c.end_soc, c.minimum_margin_db] for c in columns])
    _write_csv(output / "tables/中继调度.csv",
               ["候选列", "中继机", "能源组件", "黑区编号", "悬停点", "经度", "纬度", "离地高度_m",
                "任务开始_s", "服务开始_s", "服务结束_s", "返航_s", "机体释放_s", "组件释放_s",
                "能耗_kWh", "返航SOC", "最小裕量_dB"],
               [[c.column_id, f"R{unit:02d}", f"R-ENERGY-{component:02d}", "|".join(c.covered_interval_ids),
                 c.site.site_id, c.site.lon, c.site.lat, c.site.height_agl_m, c.mission_start_s,
                 c.service_start_s, c.service_end_s, c.return_s, c.unit_release_s,
                 c.component_release_s, c.energy_kwh, c.end_soc, c.minimum_margin_db]
                for c, unit, component in assigned])
    summary = {
        "transport_baseline": baseline["transport_schedule"],
        "trajectory_sample_count": len(samples),
        "blind_interval_count": len(intervals),
        "candidate_site_count": len(sites),
        "feasible_column_count": len(columns),
        "selected_relay_sorties": len(assigned),
        "feedback_applied": feedback_applied,
        "feedback_error": feedback_error,
        "original_candidate_set_conflicts": original_conflicts,
        "uncovered_or_resource_conflict_intervals": conflicts,
        "coarse_time_step_s": step,
        "status": "feasible_coarse" if not conflicts else "no_feasible_schedule_in_candidate_set",
        "interpretation": (
            "已在当前离散候选集内得到粗粒度可行解"
            if not conflicts else
            "当前固定运输排程在离散候选集内未找到满足两架中继资源约束的排程，应触发方案A的运输时刻反馈修复"
        ),
        "runtime_s": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
