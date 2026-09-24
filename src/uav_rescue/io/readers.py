"""读取赛题 Excel 与预留 CSV 接口。"""

from collections import defaultdict
import csv
from pathlib import Path

import openpyxl

from uav_rescue.domain import (
    Box,
    LegGeometry,
    Node,
    TransportBattery,
    TransportDrone,
    TransportUnit,
)


def read_nonempty_excel_rows(path: Path, sheet: str = "数据") -> list[tuple]:
    """读取工作表中至少含一个非空单元格的行。"""

    worksheet = openpyxl.load_workbook(path, read_only=True, data_only=True)[sheet]
    return [row for row in worksheet.iter_rows(values_only=True) if any(value is not None for value in row)]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """预留的标准 CSV 读取接口，统一使用 UTF-8-SIG。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_nodes(path: Path) -> dict[str, Node]:
    """读取 O01 与全部服务区节点。"""

    result: dict[str, Node] = {}
    for row in read_nonempty_excel_rows(path):
        node_id = row[0]
        if isinstance(node_id, str) and (node_id == "O01" or node_id.startswith("S")):
            population = None if len(row) < 6 or row[5] is None else int(row[5])
            result[node_id] = Node(
                node_id, str(row[1]), float(row[2]), float(row[3]), float(row[4]), population
            )
    return result


def load_boxes(path: Path) -> dict[str, list[Box]]:
    """读取逐箱货箱清单，并按服务区分组。"""

    grouped: dict[str, list[Box]] = defaultdict(list)
    rows = read_nonempty_excel_rows(path, "逐箱货箱清单")
    for row in rows[1:]:
        box = Box(
            box_id=str(row[0]), service_id=str(row[1]), category=str(row[2]),
            mass_kg=float(row[3]), volume_m3=float(row[4]),
            is_first_batch=str(row[5]).strip() == "是",
            first_deadline_s=None if row[6] is None else float(row[6]),
            expected_deadline_s=None if row[7] is None else float(row[7]),
            priority=None if row[8] is None else float(row[8]),
        )
        grouped[box.service_id].append(box)
    return dict(grouped)


def load_transport_drones(path: Path, reserve_override: float | None = None) -> dict[str, TransportDrone]:
    """读取三类运输机型；可由实验配置覆盖返航余量。"""

    result: dict[str, TransportDrone] = {}
    for row in read_nonempty_excel_rows(path):
        if row[0] not in {"A", "B", "C"} or len(row) < 18 or row[3] is None:
            continue
        reserve = float(row[9]) / 100.0 if reserve_override is None else reserve_override
        result[str(row[0])] = TransportDrone(
            model=str(row[0]), name=str(row[1]), empty_mass_kg=float(row[2]),
            max_payload_kg=float(row[3]), volume_m3=float(row[4]),
            cruise_speed_mps=float(row[5]), empty_range_m=float(row[6]),
            full_range_m=float(row[7]), usable_energy_kwh=float(row[8]),
            reserve_ratio=reserve, prep_s=float(row[10]), load_per_box_s=float(row[11]),
            handoff_base_s=float(row[12]), handoff_per_box_s=float(row[13]),
            climb_speed_mps=float(row[14]), descent_speed_mps=float(row[15]),
            climb_efficiency=float(row[16]),
        )
    return result


def load_transport_resources(path: Path) -> tuple[dict[str, TransportUnit], dict[str, TransportBattery]]:
    """读取实体运输无人机与共享电池，并给电池生成稳定编号。"""

    rows = read_nonempty_excel_rows(path)
    units: dict[str, TransportUnit] = {}
    batteries: dict[str, TransportBattery] = {}
    in_units = False
    for row in rows:
        first = row[0]
        if first == "无人机编号":
            in_units = True
            continue
        if in_units:
            if isinstance(first, str) and first.startswith("U"):
                units[str(first)] = TransportUnit(str(first), str(row[1]), str(row[2]))
                continue
            in_units = False
        if first in {"A", "B", "C"} and len(row) >= 3 and row[1] is not None and row[2] is not None:
            if isinstance(row[1], (int, float)) and isinstance(row[2], (int, float)):
                model = str(first)
                count = int(row[1])
                full_charge_time_s = float(row[2])
                for index in range(1, count + 1):
                    battery_id = f"{model}-BAT-{index:02d}"
                    batteries[battery_id] = TransportBattery(
                        battery_id, model, full_charge_time_s
                    )
    if not units or not batteries:
        raise ValueError("运输无人机附件中未能完整读取实体机或共享电池库存")
    return units, batteries


def load_leg_geometry_cache(
    path: Path,
    nodes: dict[str, Node],
    service_operation_height_m: float = 30.0,
) -> dict[tuple[str, str], LegGeometry]:
    """从公共缓存恢复有向航段几何及相对起降高度。"""

    result: dict[tuple[str, str], LegGeometry] = {}
    for row in read_csv_rows(path):
        origin = row["origin_id"]
        destination = row["destination_id"]
        cruise_altitude = float(row["cruise_altitude_m"])
        origin_altitude = nodes[origin].ground_m + (
            0.0 if origin == "O01" else service_operation_height_m
        )
        destination_altitude = nodes[destination].ground_m + (
            0.0 if destination == "O01" else service_operation_height_m
        )
        result[(origin, destination)] = LegGeometry(
            origin,
            destination,
            float(row["horizontal_distance_m"]),
            cruise_altitude,
            max(0.0, cruise_altitude - origin_altitude),
            max(0.0, cruise_altitude - destination_altitude),
            int(row["traversed_dem_cell_count"]),
        )
    expected = len(nodes) * (len(nodes) - 1)
    if len(result) != expected:
        raise ValueError(f"航段缓存应含 {expected} 条有向航段，实际为 {len(result)}")
    return result
