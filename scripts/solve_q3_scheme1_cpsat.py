"""基于已核验的服务列，快速重求问题三方案1的精确资源调度。

此脚本不重新计算DEM链路；它仅读取 ``outputs/q3/tables/中继候选列.csv``，
在两架中继机、六个能源组件的时间不重叠约束下选择服务列并写出可复核排程。
"""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

from uav_rescue.models.q3_joint import BlindInterval, RelayColumn, RelaySite
from uav_rescue.physics.battery import recharge_time_s
from uav_rescue.settings import PROJECT_ROOT, load_toml
from uav_rescue.solvers.q3_solver import select_and_assign_columns_cpsat


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay-count", type=int, default=2)
    parser.add_argument("--component-count", type=int, default=6)
    parser.add_argument("--max-time-s", type=float, default=120.0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    config = load_toml("configs/q3.toml")
    baseline = config["baseline"]
    output = PROJECT_ROOT / "outputs/q3"
    interval_rows = read_rows(output / "tables/直连黑区间.csv")
    intervals = tuple(
        BlindInterval(row["黑区编号"], row["架次编号"], float(row["开始时刻_s"]),
                      float(row["结束时刻_s"]), ())
        for row in interval_rows
    )
    relay = read_rows(PROJECT_ROOT / "data/processed/relay_drones.csv")[0]
    component = read_rows(PROJECT_ROOT / "data/processed/relay_energy_components.csv")[0]
    turnaround_s = float(relay["turnaround_s"])
    full_charge_s = float(component["full_charge_time_s"])
    columns: list[RelayColumn] = []
    for row in read_rows(output / "tables/中继候选列.csv"):
        end_soc = float(row["返航SOC"])
        return_s = float(row["返航_s"])
        height = float(row["离地高度_m"])
        columns.append(RelayColumn(
            row["候选列"], RelaySite(row["悬停点"], float(row["经度"]), float(row["纬度"]), 0.0, height),
            tuple(row["黑区编号"].split("|")), int(row["最大并发运输机数"]),
            int(row["覆盖采样点数"]), float(row["服务开始_s"]), float(row["服务结束_s"]),
            float(row["任务开始_s"]), return_s, return_s + turnaround_s,
            float(row["能耗_kWh"]), end_soc,
            return_s + recharge_time_s(end_soc, full_charge_s), float(row["最小裕量_dB"]),
        ))
    assigned, conflicts, diagnostics = select_and_assign_columns_cpsat(
        intervals, tuple(columns),
        relay_count=args.relay_count, component_count=args.component_count,
        concurrent_transport_limit=int(baseline["relay_concurrent_transport_limit"]),
        max_time_s=args.max_time_s,
    )
    suffix = f"_{args.tag}" if args.tag else ""
    table = output / "tables" / f"中继调度_CP_SAT{suffix}.csv"
    if assigned:
        write_csv(table,
                  ["候选列", "中继机", "能源组件", "黑区编号", "悬停点", "经度", "纬度", "离地高度_m",
                   "任务开始_s", "服务开始_s", "服务结束_s", "返航_s", "机体释放_s", "组件释放_s",
                   "能耗_kWh", "返航SOC", "最小裕量_dB", "最大并发运输机数", "覆盖采样点数"],
                  [[c.column_id, f"R{unit:02d}", f"R-ENERGY-{component_id:02d}", "|".join(c.covered_interval_ids),
                    c.site.site_id, c.site.lon, c.site.lat, c.site.height_agl_m, c.mission_start_s,
                    c.service_start_s, c.service_end_s, c.return_s, c.unit_release_s, c.component_release_s,
                    c.energy_kwh, c.end_soc, c.minimum_margin_db, c.max_concurrent_transports,
                    c.covered_sample_count]
                   for c, unit, component_id in assigned])
    covered = {interval_id for c, _, _ in assigned for interval_id in c.covered_interval_ids}
    report = {
        "scheme": "方案1 固定运输中继基线",
        "solver": "CP-SAT",
        "relay_count": args.relay_count,
        "component_count": args.component_count,
        "candidate_source": "outputs/q3/tables/中继候选列.csv",
        "required_interval_count": len(intervals),
        "covered_interval_count": len(covered),
        "uncovered_intervals": sorted({item.interval_id for item in intervals} - covered),
        "conflicts": conflicts,
        "diagnostics": diagnostics,
        "schedule_table": str(table.relative_to(PROJECT_ROOT)) if assigned else None,
    }
    path = output / "tables" / f"方案1_CP_SAT_求解诊断{suffix}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
