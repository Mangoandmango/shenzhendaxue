"""针对缺少持续驻留列的架次，局部扩充共同悬停候选点并严格DEM核验。"""
from __future__ import annotations

import csv
import json
import argparse
from collections import Counter, defaultdict

from uav_rescue.geo.coordinates import LocalEnu
from uav_rescue.geo.dem import DemGrid
from uav_rescue.models.q3_joint import BlindInterval
from uav_rescue.settings import PROJECT_ROOT, load_toml
from uav_rescue.solvers.q3_solver import (build_relay_columns, generate_candidate_sites,
                                          load_communication_parameters, reconstruct_transport_samples)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, header, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(header); writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip-ids", default="", help="逗号分隔；默认处理所有缺持续驻留列的架次")
    args = parser.parse_args()
    config = load_toml("configs/q3.toml")
    base = config["baseline"]
    root = PROJECT_ROOT
    q3 = root / "outputs/q3/tables"
    black_rows = read_rows(q3 / "直连黑区间.csv")
    old_columns = read_rows(q3 / "中继候选列.csv")
    persistent_count = Counter()
    for row in old_columns:
        if "-P" in row["候选列"]:
            persistent_count.update(row["黑区编号"].split("|"))
    target_trip_ids = sorted({row["架次编号"] for row in black_rows if persistent_count[row["黑区编号"]] == 0})
    if args.trip_ids:
        requested = {item.strip() for item in args.trip_ids.split(",") if item.strip()}
        target_trip_ids = [item for item in target_trip_ids if item in requested]
        if not target_trip_ids:
            raise ValueError("指定架次不在缺持续驻留列的目标集合中")
    samples = reconstruct_transport_samples(root, float(config["communication"]["trajectory_time_step_s"]),
                                            root / base["transport_schedule"], root / base["transport_deliveries"])
    sample_map = defaultdict(list)
    for sample in samples:
        sample_map[sample.trip_id].append(sample)
    intervals = []
    for row in black_rows:
        start, end = float(row["开始时刻_s"]), float(row["结束时刻_s"])
        members = tuple(s for s in sample_map[row["架次编号"]] if start - 1e-7 <= s.time_s <= end + 1e-7)
        intervals.append(BlindInterval(row["黑区编号"], row["架次编号"], start, end, members))
    all_intervals = tuple(intervals)
    node_rows = read_rows(root / "data/processed/nodes.csv")
    origin = next(row for row in node_rows if row["node_id"] == "O01")
    enu = LocalEnu(float(origin["longitude_deg"]), float(origin["latitude_deg"]), 0.0)
    dem = DemGrid(root / "data/raw/geo/镇龙乡及周边地理数据/数字高程模型数据（DEM）/镇龙乡及周边30米DEM.tif")
    params = load_communication_parameters(root / "data/processed/communication_parameters.csv")
    by_trip = defaultdict(list)
    for interval in all_intervals:
        if interval.trip_id in target_trip_ids:
            by_trip[interval.trip_id].append(interval)
    sites_by_interval = {}
    for trip_id, chain in by_trip.items():
        # 相邻黑区共用一组局部网格，才可能产生同点连续驻留列。
        sites = generate_candidate_sites(dem, enu, tuple(chain), 500.0, [100.0, 200.0, 300.0],
                                         [0.25, 0.5, 0.75], 1)
        for interval in chain:
            sites_by_interval[interval.interval_id] = sites
    sites = tuple({site.site_id: site for values in sites_by_interval.values() for site in values}.values())
    columns = build_relay_columns(root, dem, params, all_intervals, sites, 200, 1, 0, sites_by_interval)
    suffix = "_" + "_".join(target_trip_ids)
    out = root / "outputs/q3/scheme2/candidate_expansion" / suffix
    header = ["候选列", "黑区编号", "悬停点", "经度", "纬度", "离地高度_m", "服务开始_s", "服务结束_s",
              "任务开始_s", "返航_s", "能耗_kWh", "返航SOC", "最小裕量_dB", "最大并发运输机数", "覆盖采样点数"]
    write_rows(out / "持续驻留扩充服务列.csv", header,
               [[c.column_id, "|".join(c.covered_interval_ids), c.site.site_id, c.site.lon, c.site.lat,
                 c.site.height_agl_m, c.service_start_s, c.service_end_s, c.mission_start_s, c.return_s,
                 c.energy_kwh, c.end_soc, c.minimum_margin_db, c.max_concurrent_transports, c.covered_sample_count]
                for c in columns])
    meta = {"target_trip_ids": target_trip_ids, "target_trip_count": len(target_trip_ids),
            "generated_site_count": len(sites), "strict_feasible_column_count": len(columns),
            "continuous_residence_column_count": sum("-P" in c.column_id for c in columns),
            "method": "targeted_shared_local_grid_500m_full_dem_validation"}
    (out / "扩充诊断.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
