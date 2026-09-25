"""问题三方案2正式主模型：固定运输结构下的时序—资源—中继通信联合优化。

固定货箱组批、航线、访问顺序、机型和各架次的相对轨迹；优化起飞时刻、
同机型运输机/电池分配及其执行顺序，以及中继服务列、机体和能源组件分配。
题面要求中继服务后返回 O01，故本模型只使用“单次出发—连续驻留—返回”列，
不允许候选悬停点之间直接转场。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ortools.sat.python import cp_model

from uav_rescue.physics.battery import recharge_time_s
from uav_rescue.settings import PROJECT_ROOT


GRID = 30  # s；最终结果仍须用 0.25 s 通信回放核验


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, header: list[str], values: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(values)


def tick(seconds: float) -> int:
    """时间参数使用向上取整，避免离散化把资源释放时间提前。"""
    return max(0, int(-(-seconds // GRID)))


def floor_tick(seconds: float) -> int:
    """硬期限向下取整，与交付时刻的向上取整配合，保持约束保守。"""
    return max(0, int(seconds // GRID))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-library", type=Path, default=None,
                        help="合并后的候选列；默认使用方案1经DEM验证的基础任务库")
    parser.add_argument("--time-limit-s", type=float, default=300.0)
    parser.add_argument("--relay-count", type=int, default=2,
                        help="诊断时可放宽；正式口径为 2")
    parser.add_argument("--component-count", type=int, default=6,
                        help="诊断时可放宽；正式口径为 6")
    parser.add_argument("--max-columns-per-cover", type=int, default=8,
                        help="同一覆盖集合在Pareto剪枝后保留的代表列数；0 表示不截断")
    parser.add_argument("--ignore-transport-resources", action="store_true",
                        help="仅用于不可行来源诊断；保留交付硬窗但不施加运输机/电池NoOverlap")
    parser.add_argument("--ignore-delivery-windows", action="store_true",
                        help="仅用于不可行来源诊断；不施加任何交付硬/软窗约束与目标")
    parser.add_argument("--horizon-s", type=int, default=21600,
                        help="允许的计划时域上界；不是交付期限")
    parser.add_argument("--tag", default="正式主模型")
    args = parser.parse_args()

    root = PROJECT_ROOT
    q2 = root / "outputs/q2b_joint_soft/tables/最终方案"
    q3 = root / "outputs/q3/tables"
    out = root / "outputs/q3/scheme2/formal"
    trips = read_rows(q2 / "运输架次.csv")
    delivery_rows = read_rows(q2 / "逐箱交付与时间窗.csv")
    black_rows = read_rows(q3 / "直连黑区间.csv")
    candidate_path = args.candidate_library or (q3 / "中继候选列.csv")
    raw_columns = read_rows(candidate_path)
    transport_drones = {r["model"]: r for r in read_rows(root / "data/processed/transport_drones.csv")}
    transport_batteries = {r["model"]: r for r in read_rows(root / "data/processed/transport_batteries.csv")}
    relay = read_rows(root / "data/processed/relay_drones.csv")[0]
    component = read_rows(root / "data/processed/relay_energy_components.csv")[0]

    trip = {r["架次编号"]: r for r in trips}
    base_start = {k: float(v["开始时刻（s）"]) for k, v in trip.items()}
    base_start_tick = {k: int(round(v / GRID)) for k, v in base_start.items()}
    black = {r["黑区编号"]: r for r in black_rows}
    # 同机型的实体集合来自问题二 B+-S 的完整资源池，而不是“原架次绑定”。
    units: dict[str, list[str]] = defaultdict(list)
    batteries: dict[str, list[str]] = defaultdict(list)
    for r in trips:
        if r["无人机编号"] not in units[r["机型"]]:
            units[r["机型"]].append(r["无人机编号"])
        if r["电池编号"] not in batteries[r["机型"]]:
            batteries[r["机型"]].append(r["电池编号"])

    # 服务列保留完整列覆盖。若基础列同时覆盖多架运输机，不能把它直接
    # 分配给一架中继机；但其中每一个黑区已经通过该悬停点的全采样链路核验，
    # 可以安全拆成“单黑区、单运输机”的独立出发—服务—返回列。
    service_power = float(relay["hover_power_kw"]) + float(relay["communication_power_kw"])
    usable_relay_energy = float(relay["usable_energy_kwh"])
    normalized_rows: list[dict[str, str]] = []
    derived_capacity_rows = 0
    for row in raw_columns:
        covers = [x for x in row["黑区编号"].split("|") if x in black]
        if int(float(row.get("最大并发运输机数", "1"))) <= 1:
            normalized_rows.append(row)
            continue
        old_start, old_end = float(row["服务开始_s"]), float(row["服务结束_s"])
        old_duration = old_end - old_start
        for interval_id in covers:
            interval = black[interval_id]
            new_start, new_end = float(interval["开始时刻_s"]), float(interval["结束时刻_s"])
            derived = dict(row)
            derived["候选列"] = f"{row['候选列']}@{interval_id}"
            derived["黑区编号"] = interval_id
            derived["服务开始_s"], derived["服务结束_s"] = str(new_start), str(new_end)
            derived["任务开始_s"] = str(float(row["任务开始_s"]) + new_start - old_start)
            derived["返航_s"] = str(float(row["返航_s"]) + new_end - old_end)
            derived["能耗_kWh"] = str(float(row["能耗_kWh"]) + service_power * (new_end - new_start - old_duration) / 3600.0)
            derived["最大并发运输机数"] = "1"
            normalized_rows.append(derived)
            derived_capacity_rows += 1

    columns: list[dict[str, object]] = []
    by_black: dict[str, list[int]] = defaultdict(list)
    for row in normalized_rows:
        covers = tuple(x for x in row["黑区编号"].split("|") if x in black)
        if not covers:
            continue
        covered_trips = tuple(sorted({black[x]["架次编号"] for x in covers}))
        anchor = covered_trips[0]
        # 共享列的各覆盖航次必须相同平移；否则它的原始连续服务几何被破坏。
        col = {
            "id": row["候选列"], "covers": covers, "trips": covered_trips, "anchor": anchor,
            "mission": float(row["任务开始_s"]), "return": float(row["返航_s"]),
            "energy": float(row["能耗_kWh"]), "site": row["悬停点"],
            "service_start": float(row["服务开始_s"]), "service_end": float(row["服务结束_s"]),
            "margin": float(row["最小裕量_dB"]),
        }
        idx = len(columns); columns.append(col)
        for interval_id in covers:
            by_black[interval_id].append(idx)
    # 同一黑区集合内，若一条列到位不早于另一条、返航/组件释放不晚、
    # 能耗不高且链路裕量不低，则另一条对任何资源排程都不占优，可安全删除。
    # 这不是按经验任意删列；随后的小上限仅用于首次可行解搜索的工作集。
    raw_column_count = len(columns)
    by_cover: dict[tuple[tuple[str, ...], tuple[str, ...]], list[dict[str, object]]] = defaultdict(list)
    for col in columns:
        by_cover[(tuple(col["covers"]), tuple(col["trips"]))].append(col)
    columns = []
    for options in by_cover.values():
        survivors = []
        for candidate in options:
            anchor = str(candidate["anchor"])
            c_mission = float(candidate["mission"]) - base_start[anchor]
            c_return = float(candidate["return"]) - base_start[anchor]
            dominated = any(
                other is not candidate and
                float(other["mission"]) - base_start[str(other["anchor"])] >= c_mission - 1e-7 and
                float(other["return"]) - base_start[str(other["anchor"])] <= c_return + 1e-7 and
                float(other["energy"]) <= float(candidate["energy"]) + 1e-9 and
                float(other["margin"]) >= float(candidate["margin"]) - 1e-9 and
                (float(other["mission"]) - base_start[str(other["anchor"])] > c_mission + 1e-7 or
                 float(other["return"]) - base_start[str(other["anchor"])] < c_return - 1e-7 or
                 float(other["energy"]) < float(candidate["energy"]) - 1e-9 or
                 float(other["margin"]) > float(candidate["margin"]) + 1e-9)
                for other in options
            )
            if not dominated:
                survivors.append(candidate)
        survivors.sort(key=lambda x: (float(x["return"]) - base_start[str(x["anchor"])],
                                      float(x["energy"]), -float(x["margin"]), str(x["id"])))
        columns.extend(survivors if args.max_columns_per_cover <= 0 else survivors[:args.max_columns_per_cover])
    by_black = defaultdict(list)
    for idx, col in enumerate(columns):
        for interval_id in col["covers"]:
            by_black[str(interval_id)].append(idx)
    missing = sorted(set(black) - set(by_black))
    if missing:
        raise RuntimeError(f"候选任务库缺少黑区覆盖，不能进入正式求解：{missing}")

    model = cp_model.CpModel()
    horizon = tick(args.horizon_s)
    starts = {k: model.NewIntVar(0, horizon, f"start_{k}") for k in trip}

    # 每个交付的相对完成时刻固定；硬窗必须满足，软窗以优先系数加入目标。
    weighted_lateness: list[cp_model.LinearExpr] = []
    for row in delivery_rows:
        if args.ignore_delivery_windows:
            continue
        trip_id = row["架次编号"]
        relative_done = tick(float(row["交付完成时刻（s）"]) - base_start[trip_id])
        due = floor_tick(float(row["要求时刻（s）"]))
        if row["时间窗类型"] == "硬":
            model.Add(starts[trip_id] + relative_done <= due)
        else:
            late = model.NewIntVar(0, horizon, f"late_{row['货箱编号']}")
            model.Add(late >= starts[trip_id] + relative_done - due)
            weighted_lateness.append(int(round(float(row["应急优先系数"]))) * late)

    # 运输机和运输电池均可在同机型内重新匹配；NoOverlap 允许模型自行决定执行顺序。
    transport_assign: dict[tuple[str, str], cp_model.IntVar] = {}
    battery_assign: dict[tuple[str, str], cp_model.IntVar] = {}
    for model_name in units:
        trip_ids = [r["架次编号"] for r in trips if r["机型"] == model_name]
        drone_intervals = {u: [] for u in units[model_name]}
        battery_intervals = {b: [] for b in batteries[model_name]}
        for trip_id in trip_ids:
            r = trip[trip_id]
            flight_dur = tick(float(r["返回O01时刻（s）"]) - base_start[trip_id])
            end_soc = 1.0 - float(r["架次能耗（kWh）"]) / float(transport_drones[model_name]["usable_energy_kwh"])
            charge_dur = tick(recharge_time_s(end_soc, float(transport_batteries[model_name]["full_charge_time_s"])))
            for unit in units[model_name]:
                x = model.NewBoolVar(f"transport_{trip_id}_{unit}")
                transport_assign[(trip_id, unit)] = x
                drone_intervals[unit].append(model.NewOptionalIntervalVar(
                    starts[trip_id], flight_dur, starts[trip_id] + flight_dur, x, f"flight_{trip_id}_{unit}"))
            for battery in batteries[model_name]:
                x = model.NewBoolVar(f"battery_{trip_id}_{battery}")
                battery_assign[(trip_id, battery)] = x
                battery_intervals[battery].append(model.NewOptionalIntervalVar(
                    starts[trip_id], flight_dur + charge_dur, starts[trip_id] + flight_dur + charge_dur,
                    x, f"battery_{trip_id}_{battery}"))
            model.Add(sum(transport_assign[(trip_id, u)] for u in units[model_name]) == 1)
            model.Add(sum(battery_assign[(trip_id, b)] for b in batteries[model_name]) == 1)
        if not args.ignore_transport_resources:
            for values in list(drone_intervals.values()) + list(battery_intervals.values()):
                model.AddNoOverlap(values)

    selected = [model.NewBoolVar(f"relay_col_{i}") for i in range(len(columns))]
    relay_assign: dict[tuple[int, int, int], cp_model.IntVar] = {}
    relay_intervals = [[] for _ in range(args.relay_count)]
    component_intervals = [[] for _ in range(args.component_count)]
    for i, col in enumerate(columns):
        anchor = str(col["anchor"])
        mission_offset = tick(float(col["mission"]) - base_start[anchor])
        return_offset = tick(float(col["return"]) - base_start[anchor])
        mission = model.NewIntVar(0, horizon, f"relay_start_{i}")
        model.Add(mission == starts[anchor] + mission_offset)
        # 保留共享列时强制其覆盖航次使用同一平移量；不满足时该列不能被选择。
        for linked_trip in col["trips"]:
            if linked_trip != anchor:
                model.Add(starts[str(linked_trip)] - base_start_tick[str(linked_trip)] ==
                          starts[anchor] - base_start_tick[anchor]).OnlyEnforceIf(selected[i])
        relay_dur = max(1, return_offset - mission_offset + tick(float(relay["turnaround_s"])))
        end_soc = 1.0 - float(col["energy"]) / usable_relay_energy
        component_dur = max(relay_dur, return_offset - mission_offset + tick(
            recharge_time_s(end_soc, float(component["full_charge_time_s"]))))
        all_assignments = []
        for u in range(args.relay_count):
            per_unit = []
            for c in range(args.component_count):
                z = model.NewBoolVar(f"relay_assign_{i}_{u}_{c}")
                relay_assign[(i, u, c)] = z; all_assignments.append(z); per_unit.append(z)
            u_present = model.NewBoolVar(f"relay_unit_{i}_{u}")
            model.Add(sum(per_unit) == u_present)
            relay_intervals[u].append(model.NewOptionalIntervalVar(
                mission, relay_dur, mission + relay_dur, u_present, f"relay_task_{i}_{u}"))
        for c in range(args.component_count):
            c_present = model.NewBoolVar(f"relay_component_{i}_{c}")
            model.Add(sum(relay_assign[(i, u, c)] for u in range(args.relay_count)) == c_present)
            component_intervals[c].append(model.NewOptionalIntervalVar(
                mission, component_dur, mission + component_dur, c_present, f"component_task_{i}_{c}"))
        model.Add(sum(all_assignments) == selected[i])
    for values in relay_intervals + component_intervals:
        model.AddNoOverlap(values)
    for interval_id, choices in by_black.items():
        model.Add(sum(selected[i] for i in choices) >= 1)

    # 字典序近似：先最小化软窗加权迟到，再最少中继架次，再低中继能耗。
    energy = [int(round(float(c["energy"]) * 10_000)) for c in columns]
    energy_span = sum(energy) + 1
    sortie_weight = energy_span
    lateness_weight = (len(columns) + 1) * sortie_weight + energy_span
    model.Minimize(lateness_weight * sum(weighted_lateness) +
                   sortie_weight * sum(selected) + sum(energy[i] * selected[i] for i in range(len(columns))))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.time_limit_s
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    report: dict[str, object] = {
        "scheme": "方案2 正式主模型", "model_scope": "时序、同型运输机/电池资源、中继任务/机体/组件联合优化",
        "candidate_library": str(candidate_path), "candidate_column_count": len(columns),
        "candidate_column_count_before_pareto_pruning": raw_column_count,
        "max_columns_per_cover": args.max_columns_per_cover,
        "derived_single_transport_columns_from_overloaded_base_columns": derived_capacity_rows,
        "black_interval_count": len(black), "time_grid_s": GRID,
        "relay_count": args.relay_count, "component_count": args.component_count,
        "transport_resource_constraints": not args.ignore_transport_resources,
        "delivery_window_constraints": not args.ignore_delivery_windows,
        "status": solver.StatusName(status), "wall_time_s": solver.WallTime(),
        "objective": solver.ObjectiveValue(), "best_bound": solver.BestObjectiveBound(),
        "relay_direct_relocation": False,
    }
    suffix = args.tag.replace("/", "_")
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        chosen = [i for i, x in enumerate(selected) if solver.Value(x)]
        report.update({"result": "feasible", "selected_relay_sorties": len(chosen),
                       "total_weighted_lateness_tick": sum(solver.Value(x) for x in weighted_lateness)})
        write_rows(out / f"运输资源与时序_{suffix}.csv",
                   ["架次编号", "机型", "起飞_s", "返回O01_s", "运输机", "电池"],
                   [[tid, r["机型"], solver.Value(starts[tid]) * GRID,
                     (solver.Value(starts[tid]) + tick(float(r["返回O01时刻（s）"]) - base_start[tid])) * GRID,
                     next(u for u in units[r["机型"]] if solver.Value(transport_assign[(tid, u)])),
                     next(b for b in batteries[r["机型"]] if solver.Value(battery_assign[(tid, b)]))]
                    for tid, r in trip.items()])
        relay_output = []
        for i in chosen:
            c = columns[i]
            u, e = next((u, e) for u in range(args.relay_count) for e in range(args.component_count) if solver.Value(relay_assign[(i, u, e)]))
            start_tick = solver.Value(starts[str(c["anchor"])])
            relay_output.append([c["id"], "|".join(c["covers"]), f"R{u+1:02d}", f"R-ENERGY-{e+1:02d}", c["site"],
                                 (start_tick + tick(float(c["mission"]) - base_start[str(c["anchor"])])) * GRID,
                                 (start_tick + tick(float(c["service_start"]) - base_start[str(c["anchor"])])) * GRID,
                                 (start_tick + tick(float(c["service_end"]) - base_start[str(c["anchor"])])) * GRID,
                                 (start_tick + tick(float(c["return"]) - base_start[str(c["anchor"])])) * GRID,
                                 c["energy"], c["margin"]])
        write_rows(out / f"中继调度_{suffix}.csv",
                   ["服务列", "黑区", "中继机", "组件", "悬停点", "任务开始_s", "服务开始_s", "服务结束_s", "返航_s", "能耗_kWh", "最小裕量_dB"], relay_output)
    else:
        report["result"] = "no_solution_within_time_limit" if status == cp_model.UNKNOWN else "infeasible_under_current_candidate_library"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"求解诊断_{suffix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
