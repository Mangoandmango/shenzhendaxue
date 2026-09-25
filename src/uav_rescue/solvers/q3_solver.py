"""问题三方案 A 的确定性基线求解器。"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import math
from pathlib import Path

from uav_rescue.geo.coordinates import LocalEnu
from uav_rescue.geo.dem import DemGrid
from uav_rescue.io.readers import assert_enu_geometry_cache
from uav_rescue.models.q3_joint import (
    BlindInterval, CommunicationParameters, PositionSample, RelayColumn, RelaySite,
    check_link, extract_blind_intervals,
)
from uav_rescue.physics.battery import recharge_time_s


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _operation_altitude(node: dict[str, str]) -> float:
    return float(node["ground_elevation_m"]) + (0.0 if node["node_id"] == "O01" else 30.0)


def load_communication_parameters(path: Path) -> CommunicationParameters:
    values = {(row["category"], row["parameter"]): float(row["value"]) for row in _rows(path)}
    def value(category: str, parameter: str) -> float:
        return values[(category, parameter)]
    return CommunicationParameters(
        value("传播参数", "载波频率（MHz）"), value("传播参数", "系统损耗（dB）"),
        value("传播参数", "地形遮挡附加损耗（dB）"), value("接收参数", "接收灵敏度（dBm）"),
        value("接收参数", "衰落裕量（dB）"), value("运输无人机", "发射功率（dBm）"),
        value("运输无人机", "天线增益（dBi）"), value("中继接入端", "发射功率（dBm）"),
        value("中继接入端", "天线增益（dBi）"), value("中继回传端", "发射功率（dBm）"),
        value("中继回传端", "天线增益（dBi）"), value("固定网关 G01", "发射功率（dBm）"),
        value("固定网关 G01", "天线增益（dBi）"), value("固定网关 G01", "天线离地高度（m）"),
    )


@dataclass(frozen=True)
class _Segment:
    start_s: float
    end_s: float
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    phase: str

    def position(self, time_s: float, enu: LocalEnu) -> tuple[float, float, float]:
        ratio = 0.0 if self.end_s <= self.start_s else (time_s - self.start_s) / (self.end_s - self.start_s)
        ratio = min(1.0, max(0.0, ratio))
        lon, lat = enu.interpolate_lonlat(self.a[0], self.a[1], self.b[0], self.b[1], ratio)
        return lon, lat, self.a[2] + ratio * (self.b[2] - self.a[2])


def reconstruct_transport_samples(project: Path, time_step_s: float,
                                  schedule_path: Path | None = None,
                                  deliveries_path: Path | None = None) -> list[PositionSample]:
    assert_enu_geometry_cache(project / "data/cache")
    nodes = {row["node_id"]: row for row in _rows(project / "data/processed/nodes.csv")}
    origin_row = nodes["O01"]
    enu = LocalEnu(float(origin_row["longitude_deg"]), float(origin_row["latitude_deg"]), 0.0)
    drones = {row["model"]: row for row in _rows(project / "data/processed/transport_drones.csv")}
    geometry = {(row["origin_id"], row["destination_id"]): row
                for row in _rows(project / "data/cache/leg_geometry.csv")}
    deliveries = _rows(deliveries_path or project / "outputs/q2/tables/Q2_逐箱交付.csv")
    trip_service_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in deliveries:
        service_id = row.get("服务区编号") or row.get("服务区")
        if not service_id:
            raise ValueError("逐箱交付表缺少服务区编号字段")
        trip_service_counts[row["架次编号"]][service_id] += 1
    trips = _rows(schedule_path or project / "outputs/q2/tables/Q2_运输架次.csv")
    samples: list[PositionSample] = []
    for trip in trips:
        trip_id = trip["架次编号"]
        model = trip.get("机型编号") or trip.get("机型")
        if not model:
            raise ValueError("运输架次表缺少机型字段")
        drone = drones[model]
        sequence = trip["访问服务区顺序"].split("->")
        box_count = sum(trip_service_counts[trip_id].values())
        current_time = float(trip["开始时刻（s）"]) + float(drone["prep_s"]) + box_count * float(drone["load_per_box_s"])
        current = "O01"
        segments: list[_Segment] = []
        for destination in sequence + ["O01"]:
            leg = geometry[(current, destination)]
            origin_node, destination_node = nodes[current], nodes[destination]
            origin_alt, destination_alt = _operation_altitude(origin_node), _operation_altitude(destination_node)
            cruise_alt = float(leg["cruise_altitude_m"])
            climb = max(0.0, cruise_alt - origin_alt) / float(drone["climb_speed_mps"])
            cruise = float(leg["horizontal_distance_m"]) / float(drone["cruise_speed_mps"])
            descent = max(0.0, cruise_alt - destination_alt) / float(drone["descent_speed_mps"])
            origin = (float(origin_node["longitude_deg"]), float(origin_node["latitude_deg"]), origin_alt)
            destination_point = (float(destination_node["longitude_deg"]), float(destination_node["latitude_deg"]), destination_alt)
            top_origin = (origin[0], origin[1], cruise_alt)
            top_destination = (destination_point[0], destination_point[1], cruise_alt)
            segments.append(_Segment(current_time, current_time + climb, origin, top_origin, "爬升"))
            current_time += climb
            segments.append(_Segment(current_time, current_time + cruise, top_origin, top_destination, "巡航"))
            current_time += cruise
            segments.append(_Segment(current_time, current_time + descent, top_destination, destination_point, "下降"))
            current_time += descent
            if destination != "O01":
                handoff = float(drone["handoff_base_s"]) + trip_service_counts[trip_id][destination] * float(drone["handoff_per_box_s"])
                segments.append(_Segment(current_time, current_time + handoff, destination_point, destination_point, "投送"))
                current_time += handoff
            current = destination
        expected_return = float(trip["返回O01时刻（s）"])
        if abs(current_time - expected_return) > 1e-4:
            raise ValueError(f"{trip_id} 轨迹重建结束时刻不一致：{current_time} != {expected_return}")
        start = segments[0].start_s
        time_s = math.ceil(start / time_step_s) * time_step_s
        segment_index = 0
        while time_s <= expected_return + 1e-7:
            while segment_index + 1 < len(segments) and time_s > segments[segment_index].end_s + 1e-8:
                segment_index += 1
            segment = segments[segment_index]
            lon, lat, altitude = segment.position(time_s, enu)
            samples.append(PositionSample(trip_id, float(time_s), lon, lat, altitude, segment.phase))
            time_s += time_step_s
    return samples


def find_blind_intervals(project: Path, dem: DemGrid, params: CommunicationParameters,
                         samples: list[PositionSample], time_step_s: float) -> tuple[BlindInterval, ...]:
    gateway_row = next(row for row in _rows(project / "data/processed/nodes.csv") if row["node_id"] == "O01")
    enu = LocalEnu(float(gateway_row["longitude_deg"]), float(gateway_row["latitude_deg"]), 0.0)
    gateway = (float(gateway_row["longitude_deg"]), float(gateway_row["latitude_deg"]),
               float(gateway_row["ground_elevation_m"]) + params.gateway_height_agl_m)
    available: dict[tuple[str, float], bool] = {}
    for sample in samples:
        result = check_link(dem, enu, params, (sample.lon, sample.lat, sample.altitude_m), gateway,
                            params.transport_gateway_limit_db)
        available[(sample.trip_id, sample.time_s)] = result.available
    return extract_blind_intervals(samples, available, time_step_s)


def generate_candidate_sites(dem: DemGrid, enu: LocalEnu,
                             intervals: tuple[BlindInterval, ...], grid_step_m: float,
                             heights_m: list[float], ray_fractions: list[float],
                             neighbor_radius: int) -> tuple[RelaySite, ...]:
    points: set[tuple[float, float]] = set()
    for interval in intervals:
        probes = (interval.samples[0], interval.samples[len(interval.samples) // 2], interval.samples[-1])
        for sample in probes:
            east, north, _ = enu.from_geodetic_m(sample.lon, sample.lat, 0.0)
            anchors = [(east, north)] + [(f * east, f * north) for f in ray_fractions]
            for east, north in anchors:
                base_east = round(east / grid_step_m) * grid_step_m
                base_north = round(north / grid_step_m) * grid_step_m
                for east in range(-neighbor_radius, neighbor_radius + 1):
                    for north in range(-neighbor_radius, neighbor_radius + 1):
                        lon, lat, _ = enu.to_geodetic_m(base_east + east * grid_step_m,
                                                         base_north + north * grid_step_m)
                        candidate = (lon, lat)
                        if dem.contains(*candidate):
                            points.add((round(candidate[0], 9), round(candidate[1], 9)))
    sites: list[RelaySite] = []
    for point_index, (lon, lat) in enumerate(sorted(points), start=1):
        ground = dem.elevation_at(lon, lat)
        for height in heights_m:
            sites.append(RelaySite(f"P{point_index:04d}-H{int(height):03d}", lon, lat, ground, height))
    return tuple(sites)


def build_relay_columns(project: Path, dem: DemGrid, params: CommunicationParameters,
                        intervals: tuple[BlindInterval, ...], sites: tuple[RelaySite, ...],
                        max_per_interval: int) -> tuple[RelayColumn, ...]:
    relay = _rows(project / "data/processed/relay_drones.csv")[0]
    component = _rows(project / "data/processed/relay_energy_components.csv")[0]
    origin = next(row for row in _rows(project / "data/processed/nodes.csv") if row["node_id"] == "O01")
    enu = LocalEnu(float(origin["longitude_deg"]), float(origin["latitude_deg"]), 0.0)
    gateway = (float(origin["longitude_deg"]), float(origin["latitude_deg"]),
               float(origin["ground_elevation_m"]) + params.gateway_height_agl_m)
    origin_alt = float(origin["ground_elevation_m"])
    usable = float(relay["usable_energy_kwh"])
    reserve = float(relay["reserve_ratio"]) / 100.0 if float(relay["reserve_ratio"]) > 1 else float(relay["reserve_ratio"])
    columns: list[RelayColumn] = []
    for interval in intervals:
        feasible: list[RelayColumn] = []
        for site in sites:
            backhaul = check_link(dem, enu, params, (site.lon, site.lat, site.altitude_m), gateway,
                                  params.relay_gateway_limit_db)
            if not backhaul.available:
                continue
            margins = [backhaul.margin_db]
            ok = True
            for sample in interval.samples:
                access = check_link(dem, enu, params, (sample.lon, sample.lat, sample.altitude_m),
                                    (site.lon, site.lat, site.altitude_m), params.transport_relay_limit_db)
                margins.append(access.margin_db)
                if not access.available:
                    ok = False
                    break
            if not ok:
                continue
            distance_m, records = dem.trace_enu_segment(
                enu, float(origin["longitude_deg"]), float(origin["latitude_deg"]), site.lon, site.lat,
            )
            traversed = dem.positive_intersections(records)
            if traversed:
                cruise_alt = max(max(record.elevation_m for record in traversed) + 50.0, site.altitude_m)
            else:
                # 候选点与 O01 水平位置相同：仅执行垂直起降，不存在水平航段净空像元。
                cruise_alt = max(origin_alt, site.altitude_m)
            outbound = ((cruise_alt - origin_alt) / float(relay["climb_speed_mps"]) +
                        distance_m / float(relay["cruise_speed_mps"]) +
                        (cruise_alt - site.altitude_m) / float(relay["descent_speed_mps"]))
            inbound = ((cruise_alt - site.altitude_m) / float(relay["climb_speed_mps"]) +
                       distance_m / float(relay["cruise_speed_mps"]) +
                       (cruise_alt - origin_alt) / float(relay["descent_speed_mps"]))
            setup = float(relay["link_setup_s"])
            prep = float(relay["prep_s"])
            mission_start = interval.start_s - prep - outbound - setup
            if mission_start < -1e-7:
                continue
            service_duration = interval.end_s - interval.start_s
            horizontal_energy = float(relay["cruise_power_kw"]) * 2 * distance_m / float(relay["cruise_speed_mps"]) / 3600.0
            climb_energy = (float(relay["takeoff_mass_kg"]) * 9.81 *
                            ((cruise_alt - origin_alt) + (cruise_alt - site.altitude_m)) /
                            float(relay["climb_efficiency"]) / 3_600_000.0)
            service_energy = (float(relay["hover_power_kw"]) + float(relay["communication_power_kw"])) * (setup + service_duration) / 3600.0
            energy = horizontal_energy + climb_energy + service_energy
            end_soc = 1.0 - energy / usable
            if end_soc < reserve - 1e-9:
                continue
            return_s = interval.end_s + inbound
            unit_release = return_s + float(relay["turnaround_s"])
            component_release = return_s + recharge_time_s(end_soc, float(component["full_charge_time_s"]))
            covered = [interval.interval_id]
            for other in intervals:
                if other.interval_id == interval.interval_id:
                    continue
                if other.start_s + 1e-7 < interval.start_s or other.end_s > interval.end_s + 1e-7:
                    continue
                if all(check_link(
                    dem, enu, params, (sample.lon, sample.lat, sample.altitude_m),
                    (site.lon, site.lat, site.altitude_m), params.transport_relay_limit_db,
                ).available for sample in other.samples):
                    covered.append(other.interval_id)
            feasible.append(RelayColumn("", site, tuple(sorted(covered)), interval.start_s, interval.end_s,
                                        mission_start, return_s, unit_release, energy, end_soc,
                                        component_release, min(margins)))
        feasible.sort(key=lambda col: (-col.minimum_margin_db, col.energy_kwh, col.site.site_id))
        for rank, column in enumerate(feasible[:max_per_interval], start=1):
            columns.append(RelayColumn(f"{interval.interval_id}-C{rank:02d}", column.site,
                                       column.covered_interval_ids, column.service_start_s,
                                       column.service_end_s, column.mission_start_s, column.return_s,
                                       column.unit_release_s, column.energy_kwh, column.end_soc,
                                       column.component_release_s, column.minimum_margin_db))
    # 构造“持续驻留列”：同一悬停点可跨过短暂无需求时段继续悬停，
    # 避免上一架次返航周转尚未结束、下一批黑区已经出现的假性无解。
    interval_by_id = {item.interval_id: item for item in intervals}
    by_site: dict[str, dict[str, RelayColumn]] = defaultdict(dict)
    for column in columns:
        interval_id = column.covered_interval_ids[0]
        old = by_site[column.site.site_id].get(interval_id)
        if old is None or column.energy_kwh < old.energy_kwh:
            by_site[column.site.site_id][interval_id] = column
    persistent: list[RelayColumn] = []
    service_power = float(relay["hover_power_kw"]) + float(relay["communication_power_kw"])
    setup = float(relay["link_setup_s"])
    full_charge = float(component["full_charge_time_s"])
    for site_id, available in by_site.items():
        ordered = sorted(available, key=lambda item: (interval_by_id[item].start_s, interval_by_id[item].end_s))
        for left in range(len(ordered)):
            group = [ordered[left]]
            for right in range(left + 1, len(ordered)):
                previous = interval_by_id[group[-1]]
                current = interval_by_id[ordered[right]]
                if current.start_s - previous.end_s > 1_800.0:
                    break
                group.append(ordered[right])
                first = interval_by_id[group[0]]
                last = interval_by_id[group[-1]]
                base = available[group[0]]
                base_duration = base.service_end_s - base.service_start_s
                travel_energy = base.energy_kwh - service_power * (setup + base_duration) / 3600.0
                energy = travel_energy + service_power * (setup + last.end_s - first.start_s) / 3600.0
                end_soc = 1.0 - energy / usable
                if end_soc < reserve - 1e-9:
                    break
                return_delta = base.return_s - base.service_end_s
                turnaround_delta = base.unit_release_s - base.return_s
                return_s = last.end_s + return_delta
                persistent.append(RelayColumn(
                    f"{site_id}-P{left + 1:02d}-{right + 1:02d}", base.site,
                    tuple(sorted(set(group))), first.start_s, last.end_s,
                    base.mission_start_s, return_s, return_s + turnaround_delta,
                    energy, end_soc, return_s + recharge_time_s(end_soc, full_charge),
                    min(available[item].minimum_margin_db for item in group),
                ))
    columns.extend(persistent)
    return tuple(columns)


def select_and_assign_columns(intervals: tuple[BlindInterval, ...], columns: tuple[RelayColumn, ...],
                              relay_count: int = 2, component_count: int = 6) -> tuple[list[tuple[RelayColumn, int, int]], list[str]]:
    by_interval: dict[str, list[RelayColumn]] = defaultdict(list)
    for column in columns:
        for interval_id in column.covered_interval_ids:
            by_interval[interval_id].append(column)
    missing = [item.interval_id for item in intervals if not by_interval[item.interval_id]]
    if missing:
        return [], missing
    uncovered = {item.interval_id for item in intervals}
    order = {item.interval_id: (item.start_s, item.end_s) for item in intervals}
    if missing:
        return [], missing

    visited = 0
    node_limit = 250_000

    def search(remaining: frozenset[str], unit_free: tuple[float, ...],
               component_free: tuple[float, ...],
               chosen: list[tuple[RelayColumn, int, int]]) -> list[tuple[RelayColumn, int, int]] | None:
        nonlocal visited
        visited += 1
        if visited > node_limit:
            return None
        if not remaining:
            return chosen
        target = min(remaining, key=lambda item: order[item])
        options = sorted(
            by_interval[target],
            key=lambda column: (
                -len(remaining.intersection(column.covered_interval_ids)),
                column.energy_kwh, -column.minimum_margin_db, column.column_id,
            ),
        )
        # 同一覆盖集合只保留能耗最低的少量代表，避免候选位置造成组合爆炸。
        signatures: dict[tuple[str, ...], int] = defaultdict(int)
        filtered: list[RelayColumn] = []
        for column in options:
            signature = tuple(sorted(remaining.intersection(column.covered_interval_ids)))
            if signatures[signature] >= 4:
                continue
            signatures[signature] += 1
            filtered.append(column)
        for column in filtered:
            units = [index for index, free in enumerate(unit_free) if free <= column.mission_start_s + 1e-7]
            components = [index for index, free in enumerate(component_free) if free <= column.mission_start_s + 1e-7]
            for unit in units:
                for component in components:
                    next_units = list(unit_free)
                    next_components = list(component_free)
                    next_units[unit] = column.unit_release_s
                    next_components[component] = column.component_release_s
                    result = search(
                        remaining.difference(column.covered_interval_ids),
                        tuple(next_units), tuple(next_components),
                        chosen + [(column, unit + 1, component + 1)],
                    )
                    if result is not None:
                        return result
        return None

    result = search(frozenset(uncovered), tuple(0.0 for _ in range(relay_count)),
                    tuple(0.0 for _ in range(component_count)), [])
    if result is None:
        return [], [min(uncovered, key=lambda item: order[item])]
    return result, []


def stagger_transport_schedule(project: Path, schedule_path: Path, deliveries_path: Path,
                               intervals: tuple[BlindInterval, ...], output_dir: Path,
                               relay_lead_s: float = 0.0, relay_tail_s: float = 0.0
                               ) -> tuple[Path, Path]:
    """在硬时限和既有实体资源顺序内，仅平移架次开始时刻。

    两条保守“中继通道”分别预留出发准备/建链提前量与返航周转尾量。
    这一步不改变组批、路线、机型、实体机和电池编号。
    """

    trips = _rows(schedule_path)
    deliveries = _rows(deliveries_path)
    boxes = {row["box_id"]: row for row in _rows(project / "data/processed/boxes.csv")}
    drones = {row["model"]: row for row in _rows(project / "data/processed/transport_drones.csv")}
    batteries = {row["model"]: row for row in _rows(project / "data/processed/transport_batteries.csv")}
    old_start = {row["架次编号"]: float(row["开始时刻（s）"]) for row in trips}
    relative_blind: dict[str, tuple[float, float]] = {}
    for item in intervals:
        start, end = item.start_s - old_start[item.trip_id], item.end_s - old_start[item.trip_id]
        if item.trip_id in relative_blind:
            previous = relative_blind[item.trip_id]
            relative_blind[item.trip_id] = (min(previous[0], start), max(previous[1], end))
        else:
            relative_blind[item.trip_id] = (start, end)
    delivery_rows_by_trip: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in deliveries:
        delivery_rows_by_trip[row["架次编号"]].append(row)
    slack: dict[str, float] = {}
    for trip_id, rows in delivery_rows_by_trip.items():
        values: list[float] = []
        for row in rows:
            box = boxes[row["货箱编号"]]
            deadline = float(box["expected_deadline_s"])
            if box["is_first_batch"] in {"1", "true", "True", "是"} and box["first_deadline_s"]:
                deadline = min(deadline, float(box["first_deadline_s"]))
            values.append(deadline - float(row["交付完成时刻（s）"]))
        slack[trip_id] = min(values)

    trip_by_id = {row["架次编号"]: row for row in trips}
    new_start = dict(old_start)
    relay_free = [0.0, 0.0]
    # 最迟可行黑区开始时刻优先，使时限紧的后续架次不会被松弛任务挤占。
    jobs = sorted(relative_blind, key=lambda trip_id: (
        old_start[trip_id] + relative_blind[trip_id][0] + slack[trip_id], trip_id
    ))
    for trip_id in jobs:
        rel_start, rel_end = relative_blind[trip_id]
        alternatives = [(max(old_start[trip_id], free + relay_lead_s - rel_start), index)
                        for index, free in enumerate(relay_free)]
        start, lane = min(alternatives)
        if start - old_start[trip_id] > slack[trip_id] + 1e-7:
            raise RuntimeError(f"两通道错峰仍无法满足 {trip_id} 的硬时限")
        new_start[trip_id] = start
        relay_free[lane] = start + rel_end + relay_tail_s

    # 保持原方案中同一实体机、同一电池的先后顺序；延迟只向后传播。
    resource_groups: list[list[list[dict[str, str]]]] = []
    for field in ("无人机编号", "电池编号"):
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for trip in trips:
            grouped[trip[field]].append(trip)
        resource_groups.append([sorted(group, key=lambda row: (old_start[row["架次编号"]], row["架次编号"]))
                                for group in grouped.values()])
    for _ in range(10):
        changed = False
        for group_set_index, groups in enumerate(resource_groups):
            for group in groups:
                free = 0.0
                for trip in group:
                    trip_id = trip["架次编号"]
                    if new_start[trip_id] < free - 1e-7:
                        new_start[trip_id] = free
                        changed = True
                    duration = float(trip["返回O01时刻（s）"]) - old_start[trip_id]
                    free = new_start[trip_id] + duration
                    if group_set_index == 1:
                        drone = drones[trip["机型编号"]]
                        end_soc = 1.0 - float(trip["架次能耗（kWh）"]) / float(drone["usable_energy_kwh"])
                        free += recharge_time_s(end_soc, float(batteries[trip["机型编号"]]["full_charge_time_s"]))
        if not changed:
            break
    for trip_id, start in new_start.items():
        if start - old_start[trip_id] > slack[trip_id] + 1e-7:
            raise RuntimeError(
                f"资源延迟传播后 {trip_id} 平移 {start-old_start[trip_id]:.1f}s，"
                f"超过剩余时限 {slack[trip_id]:.1f}s"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    shifted_schedule = output_dir / "运输架次.csv"
    with shifted_schedule.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trips[0]))
        writer.writeheader()
        for row in trips:
            delta = new_start[row["架次编号"]] - old_start[row["架次编号"]]
            updated = dict(row)
            updated["开始时刻（s）"] = new_start[row["架次编号"]]
            updated["返回O01时刻（s）"] = float(row["返回O01时刻（s）"]) + delta
            writer.writerow(updated)
    shifted_deliveries = output_dir / "逐箱交付.csv"
    with shifted_deliveries.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deliveries[0]))
        writer.writeheader()
        for row in deliveries:
            updated = dict(row)
            delta = new_start[row["架次编号"]] - old_start[row["架次编号"]]
            updated["交付完成时刻（s）"] = float(row["交付完成时刻（s）"]) + delta
            writer.writerow(updated)
    return shifted_schedule, shifted_deliveries


def apply_forced_trip_delays(project: Path, schedule_path: Path, deliveries_path: Path,
                             forced_delays_s: dict[str, float], output_dir: Path) -> tuple[Path, Path]:
    """施加少量指定架次平移，并按原实体机/电池顺序传播延迟。"""
    trips = _rows(schedule_path)
    deliveries = _rows(deliveries_path)
    boxes = {row["box_id"]: row for row in _rows(project / "data/processed/boxes.csv")}
    drones = {row["model"]: row for row in _rows(project / "data/processed/transport_drones.csv")}
    batteries = {row["model"]: row for row in _rows(project / "data/processed/transport_batteries.csv")}
    old_start = {row["架次编号"]: float(row["开始时刻（s）"]) for row in trips}
    new_start = {trip_id: start + float(forced_delays_s.get(trip_id, 0.0))
                 for trip_id, start in old_start.items()}
    by_trip: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in deliveries:
        by_trip[row["架次编号"]].append(row)
    slack: dict[str, float] = {}
    for trip_id, rows in by_trip.items():
        values = []
        for row in rows:
            box = boxes[row["货箱编号"]]
            deadline = float(box["expected_deadline_s"])
            if box["is_first_batch"] in {"1", "true", "True", "是"} and box["first_deadline_s"]:
                deadline = min(deadline, float(box["first_deadline_s"]))
            values.append(deadline - float(row["交付完成时刻（s）"]))
        slack[trip_id] = min(values)
    for _ in range(len(trips) + 1):
        changed = False
        for field in ("无人机编号", "电池编号"):
            groups: dict[str, list[dict[str, str]]] = defaultdict(list)
            for trip in trips:
                groups[trip[field]].append(trip)
            for group in groups.values():
                free = 0.0
                for trip in sorted(group, key=lambda row: (old_start[row["架次编号"]], row["架次编号"])):
                    trip_id = trip["架次编号"]
                    if new_start[trip_id] < free - 1e-7:
                        new_start[trip_id] = free
                        changed = True
                    duration = float(trip["返回O01时刻（s）"]) - old_start[trip_id]
                    free = new_start[trip_id] + duration
                    if field == "电池编号":
                        drone = drones[trip["机型编号"]]
                        end_soc = 1.0 - float(trip["架次能耗（kWh）"]) / float(drone["usable_energy_kwh"])
                        free += recharge_time_s(end_soc, float(batteries[trip["机型编号"]]["full_charge_time_s"]))
        if not changed:
            break
    for trip_id, start in new_start.items():
        if start - old_start[trip_id] > slack[trip_id] + 1e-7:
            raise RuntimeError(f"指定错峰传播后 {trip_id} 超出硬时限")
    output_dir.mkdir(parents=True, exist_ok=True)
    shifted_schedule = output_dir / "运输架次.csv"
    with shifted_schedule.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trips[0])); writer.writeheader()
        for row in trips:
            updated = dict(row); trip_id = row["架次编号"]
            delta = new_start[trip_id] - old_start[trip_id]
            updated["开始时刻（s）"] = new_start[trip_id]
            updated["返回O01时刻（s）"] = float(row["返回O01时刻（s）"]) + delta
            writer.writerow(updated)
    shifted_deliveries = output_dir / "逐箱交付.csv"
    with shifted_deliveries.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deliveries[0])); writer.writeheader()
        for row in deliveries:
            updated = dict(row)
            updated["交付完成时刻（s）"] = float(row["交付完成时刻（s）"]) + new_start[row["架次编号"]] - old_start[row["架次编号"]]
            writer.writerow(updated)
    return shifted_schedule, shifted_deliveries
