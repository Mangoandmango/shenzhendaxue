"""问题三方案2：固定运输结构下的时序—通信联合 CP-SAT 优化。

固定问题二 B+-S 的货箱组批、服务顺序、机型、运输无人机与电池编号；
只允许各运输架次延后（5 秒粒度），并与中继服务列、两架中继机和六个
能源组件的任务/充电占用同步排程。所有输入服务列均已完成 DEM 核验。
"""

from __future__ import annotations

import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path

from ortools.sat.python import cp_model

from uav_rescue.physics.battery import recharge_time_s
from uav_rescue.settings import PROJECT_ROOT


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, header: list[str], data: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(data)


def deadline_of(box: dict[str, str]) -> float:
    deadline = float(box["expected_deadline_s"])
    if box["is_first_batch"] in {"1", "true", "True", "是"} and box["first_deadline_s"]:
        deadline = min(deadline, float(box["first_deadline_s"]))
    return deadline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay-count", type=int, default=2)
    parser.add_argument("--component-count", type=int, default=6)
    parser.add_argument("--ignore-delivery-deadlines", action="store_true")
    parser.add_argument("--diagnostic-delay-cap-s", type=int, default=18_000)
    parser.add_argument("--only-trip", default="")
    parser.add_argument("--stop-after-first-solution", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    output = PROJECT_ROOT / "outputs/q3"
    q2_dir = PROJECT_ROOT / "outputs/q2b_joint_soft/tables/最终方案"
    trips = rows(q2_dir / "运输架次.csv")
    deliveries = rows(q2_dir / "逐箱交付与时间窗.csv")
    interval_rows = rows(output / "tables/直连黑区间.csv")
    raw_columns = rows(output / "tables/中继候选列.csv")
    boxes = {row["box_id"]: row for row in rows(PROJECT_ROOT / "data/processed/boxes.csv")}
    relay = rows(PROJECT_ROOT / "data/processed/relay_drones.csv")[0]
    component = rows(PROJECT_ROOT / "data/processed/relay_energy_components.csv")[0]

    if args.only_trip:
        trips = [row for row in trips if row["架次编号"] == args.only_trip]
        deliveries = [row for row in deliveries if row["架次编号"] == args.only_trip]
        interval_rows = [row for row in interval_rows if row["架次编号"] == args.only_trip]
        if not trips:
            raise ValueError(f"未知架次：{args.only_trip}")

    intervals = {
        row["黑区编号"]: {"trip": row["架次编号"], "start": float(row["开始时刻_s"]), "end": float(row["结束时刻_s"])}
        for row in interval_rows
    }
    trip_by_id = {row["架次编号"]: row for row in trips}
    deliveries_by_trip: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in deliveries:
        deliveries_by_trip[row["架次编号"]].append(row)

    # 方案2的保守口径：不让任何箱子的到达时刻晚于问题二 B+-S 已满足的目标时刻。
    # 这比“软时窗允许迟到”更严格，却避免把通信可行性建立在额外延误上。
    max_delay: dict[str, int] = {}
    for trip_id, trip_deliveries in deliveries_by_trip.items():
        slack = min(deadline_of(boxes[row["货箱编号"]]) - float(row["交付完成时刻（s）"])
                    for row in trip_deliveries)
        max_delay[trip_id] = args.diagnostic_delay_cap_s if args.ignore_delivery_deadlines else max(0, int(slack // 5) * 5)

    # 跨架次的共享覆盖在各架次不同平移后不再天然有效；但同一架次内
    # 可一次服务多个黑区的列必须保留。故仅按“运输架次”拆分服务模板。
    # 旅行/建链常数由原列反推，服务能耗按悬停时长线性修正。
    service_power = float(relay["hover_power_kw"]) + float(relay["communication_power_kw"])
    usable_energy = float(relay["usable_energy_kwh"])
    turnaround = float(relay["turnaround_s"])
    charge_full = float(component["full_charge_time_s"])
    columns: list[dict[str, object]] = []
    seen: set[tuple[str, tuple[str, ...], int, int]] = set()
    for raw in raw_columns:
        covered = [item for item in raw["黑区编号"].split("|") if item in intervals]
        if not covered:
            continue
        old_service_start, old_service_end = float(raw["服务开始_s"]), float(raw["服务结束_s"])
        old_duration = old_service_end - old_service_start
        old_energy = float(raw["能耗_kWh"])
        old_mission, old_return = float(raw["任务开始_s"]), float(raw["返航_s"])
        covered_by_trip: dict[str, list[str]] = defaultdict(list)
        for interval_id in covered:
            covered_by_trip[str(intervals[interval_id]["trip"])].append(interval_id)
        for trip_id, interval_ids in covered_by_trip.items():
            interval_ids = sorted(interval_ids, key=lambda item: float(intervals[item]["start"]))
            service_start = min(float(intervals[item]["start"]) for item in interval_ids)
            service_end = max(float(intervals[item]["end"]) for item in interval_ids)
            key = (raw["悬停点"], tuple(interval_ids), int(round(old_mission + (service_start - old_service_start))),
                   int(round(old_return + (service_end - old_service_end))))
            if key in seen:
                continue
            seen.add(key)
            duration = service_end - service_start
            energy = old_energy + service_power * (duration - old_duration) / 3600.0
            end_soc = 1.0 - energy / usable_energy
            return_s = old_return + (service_end - old_service_end)
            columns.append({
                "id": f"{raw['候选列']}@{trip_id}", "intervals": tuple(interval_ids), "trip": trip_id,
                "linked_trips": (trip_id,),
                "site": raw["悬停点"], "lon": float(raw["经度"]), "lat": float(raw["纬度"]),
                "height": float(raw["离地高度_m"]), "mission": old_mission + (service_start - old_service_start),
                "service_start": service_start, "service_end": service_end, "return": return_s,
                "unit_release": return_s + turnaround, "component_release": return_s + recharge_time_s(end_soc, charge_full),
                "energy": energy, "soc": end_soc, "margin": float(raw["最小裕量_dB"]),
            })
        # 保留跨架次的原始共享列：只有被选中时才强制这些架次采用同一平移量。
        # 因而共享悬停服务的物理含义不会在方案2中被错误丢弃。
        linked_trips = tuple(sorted(covered_by_trip))
        if len(linked_trips) > 1:
            key = (raw["悬停点"], tuple(sorted(covered)), int(round(old_mission)), int(round(old_return)))
            if key not in seen:
                seen.add(key)
                end_soc = float(raw["返航SOC"])
                return_s = old_return
                columns.append({
                    "id": f"{raw['候选列']}@SHARED", "intervals": tuple(sorted(covered)), "trip": linked_trips[0],
                    "linked_trips": linked_trips, "site": raw["悬停点"], "lon": float(raw["经度"]),
                    "lat": float(raw["纬度"]), "height": float(raw["离地高度_m"]), "mission": old_mission,
                    "service_start": old_service_start, "service_end": old_service_end, "return": return_s,
                    "unit_release": return_s + turnaround,
                    "component_release": return_s + recharge_time_s(end_soc, charge_full),
                    "energy": old_energy, "soc": end_soc, "margin": float(raw["最小裕量_dB"]),
                })

    # 同一黑区中，若候选 A 不晚于 B 到位、且更早释放机体/组件、能耗不高，
    # 则 B 对任何其他任务的可行性和目标都无贡献，可安全剔除。
    # 这是真正的支配剪枝，而不是任意截断候选数。
    grouped_columns: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for column in columns:
        grouped_columns[tuple(column["intervals"])].append(column)
    pruned: list[dict[str, object]] = []
    for options in grouped_columns.values():
        for candidate in options:
            dominated = any(
                other is not candidate and
                float(other["mission"]) >= float(candidate["mission"]) - 1e-7 and
                float(other["unit_release"]) <= float(candidate["unit_release"]) + 1e-7 and
                float(other["component_release"]) <= float(candidate["component_release"]) + 1e-7 and
                float(other["energy"]) <= float(candidate["energy"]) + 1e-9 and
                (float(other["mission"]) > float(candidate["mission"]) + 1e-7 or
                 float(other["unit_release"]) < float(candidate["unit_release"]) - 1e-7 or
                 float(other["component_release"]) < float(candidate["component_release"]) - 1e-7 or
                 float(other["energy"]) < float(candidate["energy"]) - 1e-9)
                for other in options
            )
            if not dominated:
                pruned.append(candidate)
    raw_template_count = len(columns)
    columns = pruned

    by_interval: dict[str, list[int]] = defaultdict(list)
    for index, column in enumerate(columns):
        for interval_id in column["intervals"]:
            by_interval[str(interval_id)].append(index)
    missing = sorted(set(intervals) - set(by_interval))
    if missing:
        raise RuntimeError(f"单黑区服务模板缺失：{missing}")

    model = cp_model.CpModel()
    delay_ticks = {
        trip_id: model.NewIntVar(0, max_delay[trip_id] // 5, f"delay_tick_{trip_id}")
        for trip_id in trip_by_id
    }
    delay_s = {trip_id: 5 * variable for trip_id, variable in delay_ticks.items()}

    # 保留问题二中相同运输机和相同电池的架次先后关系，延迟仅向后传播。
    for field, include_charge in (("无人机编号", False), ("电池编号", True)):
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for trip in trips:
            grouped[trip[field]].append(trip)
        for group in grouped.values():
            ordered = sorted(group, key=lambda row: (float(row["开始时刻（s）"]), row["架次编号"]))
            for previous, following in zip(ordered, ordered[1:]):
                previous_id, following_id = previous["架次编号"], following["架次编号"]
                release = float(previous["返回O01时刻（s）"])
                if include_charge:
                    # 电池释放含问题二已有能耗对应的充电时间，实体编号不变。
                    drone = next(item for item in rows(PROJECT_ROOT / "data/processed/transport_drones.csv")
                                 if item["model"] == previous["机型"])
                    battery = next(item for item in rows(PROJECT_ROOT / "data/processed/transport_batteries.csv")
                                   if item["model"] == previous["机型"])
                    end_soc = 1.0 - float(previous["架次能耗（kWh）"]) / float(drone["usable_energy_kwh"])
                    release += recharge_time_s(end_soc, float(battery["full_charge_time_s"]))
                # 使用秒级整数时间轴，避免浮点约束。
                model.Add(int(round(float(following["开始时刻（s）"]))) + delay_s[following_id] >=
                          int(round(release)) + delay_s[previous_id])

    selected = [model.NewBoolVar(f"select_{index}") for index in range(len(columns))]
    unit_presence: dict[tuple[int, int], object] = {}
    component_presence: dict[tuple[int, int], object] = {}
    unit_intervals = [[] for _ in range(args.relay_count)]
    component_intervals = [[] for _ in range(args.component_count)]
    for index, column in enumerate(columns):
        trip_id = str(column["trip"])
        mission = int(round(float(column["mission"])))
        unit_release = max(mission + 1, int(round(float(column["unit_release"]))))
        component_release = max(mission + 1, int(round(float(column["component_release"]))))
        for unit in range(args.relay_count):
            present = model.NewBoolVar(f"unit_{index}_{unit}")
            unit_presence[(index, unit)] = present
            unit_intervals[unit].append(model.NewOptionalIntervalVar(
                mission + delay_s[trip_id], unit_release - mission, unit_release + delay_s[trip_id], present,
                f"relay_{index}_{unit}",
            ))
        for component_id in range(args.component_count):
            present = model.NewBoolVar(f"component_{index}_{component_id}")
            component_presence[(index, component_id)] = present
            component_intervals[component_id].append(model.NewOptionalIntervalVar(
                mission + delay_s[trip_id], component_release - mission,
                component_release + delay_s[trip_id], present, f"energy_{index}_{component_id}",
            ))
        model.Add(sum(unit_presence[(index, unit)] for unit in range(args.relay_count)) == selected[index])
        model.Add(sum(component_presence[(index, item)] for item in range(args.component_count)) == selected[index])
        for linked_trip in column["linked_trips"]:
            if linked_trip != trip_id:
                model.Add(delay_ticks[str(linked_trip)] == delay_ticks[trip_id]).OnlyEnforceIf(selected[index])
    for resources in unit_intervals + component_intervals:
        model.AddNoOverlap(resources)
    for interval_id, options in by_interval.items():
        model.Add(sum(selected[index] for index in options) >= 1)

    energy_cost = [int(round(float(column["energy"]) * 10_000)) for column in columns]
    # 字典序：先尽量少延迟，其次少中继架次，最后低能耗。
    energy_span = sum(energy_cost) + 1
    sortie_weight = energy_span
    delay_weight = (len(columns) + 1) * sortie_weight + energy_span
    model.Minimize(delay_weight * sum(delay_ticks.values()) +
                   sortie_weight * sum(selected) +
                   sum(energy_cost[index] * selected[index] for index in range(len(columns))))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180.0
    solver.parameters.num_search_workers = 8
    solver.parameters.stop_after_first_solution = args.stop_after_first_solution
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    report: dict[str, object] = {
        "scheme": "方案2 时序通信联合优化主模型", "solver": "CP-SAT",
        "time_granularity_s": 5, "candidate_template_count": len(columns),
        "candidate_template_count_before_dominance_pruning": raw_template_count,
        "relay_count": args.relay_count, "component_count": args.component_count,
        "required_interval_count": len(intervals), "status": status_name,
        "wall_time_s": solver.WallTime(), "objective": solver.ObjectiveValue(),
        "best_bound": solver.BestObjectiveBound(),
        "deadline_policy": "忽略交付时限的诊断运行" if args.ignore_delivery_deadlines else "所有货箱不晚于问题二B+-S既有目标时刻",
    }
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        report["result"] = "no_feasible_schedule_under_current_conservative_deadline_policy"
    else:
        delays = {trip_id: int(solver.Value(variable)) * 5 for trip_id, variable in delay_ticks.items()}
        chosen = [index for index, variable in enumerate(selected) if solver.Value(variable)]
        report.update({"result": "feasible", "total_delay_s": sum(delays.values()),
                       "selected_relay_sorties": len(chosen), "trip_delays_s": delays})
        name_suffix = f"_{args.tag}" if args.tag else ""
        write_csv(output / "scheme2" / f"运输架次_调整后{name_suffix}.csv",
                  list(trips[0]),
                  [[row[key] if key not in {"开始时刻（s）", "返回O01时刻（s）"} else
                    (float(row[key]) + delays[row["架次编号"]] if key == "开始时刻（s）" else
                     float(row[key]) + delays[row["架次编号"]])
                    for key in trips[0]] for row in trips])
        relay_rows = []
        for index in chosen:
            column = columns[index]; trip_id = str(column["trip"]); delay = delays[trip_id]
            unit = next(item + 1 for item in range(args.relay_count) if solver.Value(unit_presence[(index, item)]))
            component_id = next(item + 1 for item in range(args.component_count) if solver.Value(component_presence[(index, item)]))
            relay_rows.append([column["id"], f"R{unit:02d}", f"R-ENERGY-{component_id:02d}", "|".join(column["intervals"]), trip_id,
                               column["site"], column["lon"], column["lat"], column["height"],
                               float(column["mission"]) + delay, float(column["service_start"]) + delay,
                               float(column["service_end"]) + delay, float(column["return"]) + delay,
                               float(column["unit_release"]) + delay, float(column["component_release"]) + delay,
                               column["energy"], column["soc"], column["margin"]])
        write_csv(output / "scheme2" / f"中继调度{name_suffix}.csv",
                  ["服务列", "中继机", "能源组件", "黑区编号", "运输架次", "悬停点", "经度", "纬度", "离地高度_m",
                   "任务开始_s", "服务开始_s", "服务结束_s", "返航_s", "机体释放_s", "组件释放_s", "能耗_kWh", "返航SOC", "最小裕量_dB"], relay_rows)
    name_suffix = f"_{args.tag}" if args.tag else ""
    (output / "scheme2/求解诊断.json").parent.mkdir(parents=True, exist_ok=True)
    (output / "scheme2" / f"求解诊断{name_suffix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
