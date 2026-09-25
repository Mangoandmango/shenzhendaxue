#!/usr/bin/env python3
"""将原始附件标准化为 CSV，并生成公共 DEM 航段缓存和数据审计报告。"""

from collections import Counter
from pathlib import Path
import sys


# 从脚本运行时显式加入 src，避免要求用户先安装项目。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_rescue.geo.coordinates import LocalEnu  # noqa: E402
from uav_rescue.geo.dem import DemGrid  # noqa: E402
from uav_rescue.io.readers import (  # noqa: E402
    ENU_GEOMETRY_DEFINITION,
    load_boxes,
    load_nodes,
    load_transport_drones,
    read_nonempty_excel_rows,
)
from uav_rescue.io.writers import write_csv, write_json  # noqa: E402
from uav_rescue.settings import load_toml, project_path  # noqa: E402


def rows_after_header(rows: list[tuple], header_name: str, id_prefix: str) -> list[tuple]:
    """定位首列标题后，返回直到下一个空/分节标题前的连续数据行。"""

    for index, row in enumerate(rows):
        if row[0] == header_name:
            result: list[tuple] = []
            for item in rows[index + 1:]:
                if not isinstance(item[0], str) or not item[0].startswith(id_prefix):
                    break
                result.append(item)
            return result
    raise ValueError(f"未找到表头：{header_name}")


def extract_transport_resources(path: Path) -> tuple[list[tuple], list[tuple]]:
    """从运输机附件提取实体无人机和共享电池库存。"""

    rows = read_nonempty_excel_rows(path)
    units = rows_after_header(rows, "无人机编号", "U")
    batteries: list[tuple] = []
    for index, row in enumerate(rows):
        if row[0] == "机型编号" and len(row) > 1 and row[1] == "共享电池组总数（组）":
            for item in rows[index + 1:]:
                if item[0] is None:
                    break
                batteries.append(item)
            break
    return units, batteries


def extract_relay_resources(path: Path) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """提取中继机型、实体中继机与共享能源组件库存。"""

    rows = read_nonempty_excel_rows(path)
    model_rows: list[tuple] = []
    for index, row in enumerate(rows):
        if row[0] == "机型编号" and len(row) > 1 and row[1] == "机型名称":
            model_rows.append(rows[index + 1])
            break
    units = rows_after_header(rows, "中继无人机编号", "R")
    components: list[tuple] = []
    for index, row in enumerate(rows):
        if row[0] == "机型编号" and len(row) > 1 and row[1] == "共享能源组件总数（组）":
            for item in rows[index + 1:]:
                if item[0] is None:
                    break
                components.append(item)
            break
    return model_rows, units, components


def extract_communication_parameters(path: Path) -> list[tuple]:
    """读取通信参数表，保留类别、名称、符号和数值。"""

    rows = read_nonempty_excel_rows(path)
    for index, row in enumerate(rows):
        if row[0] == "参数类别":
            return [item for item in rows[index + 1:] if item[0] is not None]
    raise ValueError("通信参数表缺少表头")


def build_leg_cache(nodes: dict, dem: DemGrid, enu: LocalEnu) -> tuple[list[list[object]], list[list[object]]]:
    """预计算航段几何摘要及按穿越顺序保存的完整 DEM 地形剖面。"""

    geometry_rows: list[list[object]] = []
    profile_rows: list[list[object]] = []
    for origin_id, origin in sorted(nodes.items()):
        for destination_id, destination in sorted(nodes.items()):
            if origin_id == destination_id:
                continue
            horizontal_distance, records = dem.trace_enu_segment(
                enu, origin.lon, origin.lat, destination.lon, destination.lat,
            )
            traversed = dem.positive_intersections(records)
            max_terrain = max(record.elevation_m for record in traversed)
            geometry_rows.append([
                origin_id, destination_id, horizontal_distance, max_terrain,
                max_terrain + 50.0, len({(record.row, record.col) for record in traversed}),
            ])
            for sequence, record in enumerate(records):
                profile_rows.append([
                    origin_id, destination_id, sequence,
                    record.row, record.col, record.elevation_m,
                    record.s_in_m, record.s_out_m, record.intersection_length_m,
                    record.contact_type,
                ])
    return geometry_rows, profile_rows


def node_enu_coordinates(enu: LocalEnu, node: object) -> tuple[float, float, float]:
    """水平坐标与航段同样以 h=0 的 ENU 平面定义；Up 保留节点表高程信息。"""

    east, north, _ = enu.from_geodetic_m(node.lon, node.lat, 0.0)
    _, _, up = enu.from_geodetic_m(node.lon, node.lat, node.ground_m)
    return east, north, up


def main() -> None:
    """执行预处理、审计与缓存构建。"""

    config = load_toml("configs/base.toml")
    paths = config["paths"]
    processed_dir = project_path("data/processed")
    cache_dir = project_path("data/cache")

    # 读取原始数据；所有派生文件均写入 processed 或 cache，绝不写回 raw。
    nodes = load_nodes(project_path(paths["node_workbook"]))
    boxes_by_service = load_boxes(project_path(paths["cargo_workbook"]))
    drones = load_transport_drones(project_path(paths["transport_workbook"]))
    dem = DemGrid(project_path(paths["dem"]))
    boxes = [box for service_boxes in boxes_by_service.values() for box in service_boxes]
    origin = nodes["O01"]
    enu = LocalEnu(origin.lon, origin.lat, 0.0)

    transport_units, transport_batteries = extract_transport_resources(project_path(paths["transport_workbook"]))
    relay_models, relay_units, relay_components = extract_relay_resources(project_path(paths["relay_workbook"]))
    communication = extract_communication_parameters(project_path(paths["communication_workbook"]))
    demand_rows = read_nonempty_excel_rows(project_path(paths["cargo_workbook"]))[1:]

    # 写出供四问共享的扁平化节点和逐箱货箱表。
    write_csv(processed_dir / "nodes.csv", [
        "node_id", "node_type", "name", "longitude_deg", "latitude_deg", "ground_elevation_m",
        "local_east_m", "local_north_m", "local_up_m", "population"
    ], [
        [node.node_id, "dispatch_center" if node.node_id == "O01" else "service_area", node.name, node.lon,
         node.lat, node.ground_m, *node_enu_coordinates(enu, node), node.population]
        for node in sorted(nodes.values(), key=lambda item: item.node_id)
    ])
    write_json(processed_dir / "coordinate_reference.json", {
        "geographic_crs": "WGS84 longitude/latitude (EPSG:4326)",
        "local_metric_system": "ENU (east, north, up), metres",
        "origin_node_id": origin.node_id,
        "origin_longitude_deg": origin.lon,
        "origin_latitude_deg": origin.lat,
        "origin_ground_elevation_m": origin.ground_m,
        "horizontal_distance_note": "所有航段、通信链路和 DEM 采样路径均按以 O01 为原点的 ENU 平面直线定义。",
    })
    write_csv(processed_dir / "boxes.csv", [
        "box_id", "service_id", "category", "mass_kg", "volume_m3", "is_first_batch",
        "first_deadline_s", "expected_deadline_s", "priority"
    ], [
        [box.box_id, box.service_id, box.category, box.mass_kg, box.volume_m3, int(box.is_first_batch),
         box.first_deadline_s, box.expected_deadline_s, box.priority]
        for box in sorted(boxes, key=lambda item: item.box_id)
    ])
    write_csv(processed_dir / "transport_drones.csv", [
        "model", "name", "empty_mass_kg", "max_payload_kg", "volume_m3", "cruise_speed_mps",
        "empty_range_m", "full_range_m", "usable_energy_kwh", "reserve_ratio", "prep_s",
        "load_per_box_s", "handoff_base_s", "handoff_per_box_s", "climb_speed_mps",
        "descent_speed_mps", "climb_efficiency"
    ], [
        [drone.model, drone.name, drone.empty_mass_kg, drone.max_payload_kg, drone.volume_m3,
         drone.cruise_speed_mps, drone.empty_range_m, drone.full_range_m, drone.usable_energy_kwh,
         drone.reserve_ratio, drone.prep_s, drone.load_per_box_s, drone.handoff_base_s,
         drone.handoff_per_box_s, drone.climb_speed_mps, drone.descent_speed_mps, drone.climb_efficiency]
        for drone in sorted(drones.values(), key=lambda item: item.model)
    ])
    write_csv(processed_dir / "transport_units.csv", ["unit_id", "model", "initial_location"], [
        [row[0], row[1], row[2]] for row in transport_units
    ])
    write_csv(processed_dir / "transport_batteries.csv", ["model", "battery_count", "full_charge_time_s"], [
        [row[0], row[1], row[2]] for row in transport_batteries
    ])
    write_csv(processed_dir / "relay_drones.csv", [
        "model", "name", "empty_mass_with_energy_kg", "communication_module_mass_kg", "takeoff_mass_kg",
        "cruise_speed_mps", "cruise_power_kw", "usable_energy_kwh", "reserve_ratio", "prep_s",
        "link_setup_s", "turnaround_s", "climb_speed_mps", "descent_speed_mps", "climb_efficiency",
        "descent_efficiency", "hover_power_kw", "communication_power_kw", "max_hover_height_agl_m"
    ], relay_models)
    write_csv(processed_dir / "relay_units.csv", ["unit_id", "model", "initial_location"], [
        [row[0], row[1], row[2]] for row in relay_units
    ])
    write_csv(processed_dir / "relay_energy_components.csv", ["model", "component_count", "full_charge_time_s"], [
        [row[0], row[1], row[2]] for row in relay_components
    ])
    write_csv(processed_dir / "communication_parameters.csv", ["category", "parameter", "symbol", "value"], [
        [row[0], row[1], row[3], row[4]] for row in communication
    ])
    write_csv(processed_dir / "demand_summary.csv", [
        "service_id", "category", "total_box_count", "first_batch_count", "box_mass_kg", "box_volume_m3",
        "priority", "first_deadline_s", "expected_deadline_s"
    ], demand_rows)

    # 该缓存只含地形与距离信息，问题二、三可在此基础上叠加载荷、时间和通信约束。
    leg_rows, profile_rows = build_leg_cache(nodes, dem, enu)
    write_csv(cache_dir / "leg_geometry.csv", [
        "origin_id", "destination_id", "horizontal_distance_m", "max_terrain_elevation_m",
        "cruise_altitude_m", "traversed_dem_cell_count"
    ], leg_rows)
    write_csv(cache_dir / "leg_terrain_profile.csv", [
        "origin_id", "destination_id", "record_index", "dem_row", "dem_col", "terrain_elevation_m",
        "s_in_m", "s_out_m", "intersection_length_m", "contact_type"
    ], profile_rows)
    write_json(cache_dir / "geometry_metadata.json", {
        "geometry_definition": ENU_GEOMETRY_DEFINITION,
        "horizontal_distance": "WGS84 local ENU Euclidean distance, origin O01",
        "dem_path": "the same ENU straight line inverse-mapped to WGS84; half-pixel boundary roots solved on the mapped path",
        "dem_pixel_reference": "RasterPixelIsPoint; pixel-center lattice with boundaries at index +/- 0.5",
        "contact_policy": "cruise uses positive-length interior/boundary records; LOS uses interior records nominally and reports boundary/corner sensitivity; endpoint cells are excluded from LOS",
    })

    # 按“服务区、物资类型”核对需求汇总表与逐箱清单。
    demand_counts = Counter((str(row[0]), str(row[1])) for row in demand_rows for _ in range(int(row[2])))
    box_counts = Counter((box.service_id, box.category) for box in boxes)
    node_dem_differences = {
        node_id: abs(node.ground_m - dem.elevation_at(node.lon, node.lat))
        for node_id, node in nodes.items()
    }
    dem_metadata = dem.audit_metadata()
    service_ids = {node_id for node_id in nodes if node_id.startswith("S")}
    box_ids = [box.box_id for box in boxes]
    audit = {
        "counts": {
            "nodes": len(nodes), "service_areas": len(service_ids), "boxes": len(boxes),
            "transport_models": len(drones), "transport_units": len(transport_units),
            "transport_battery_groups": sum(int(row[1]) for row in transport_batteries),
            "relay_models": len(relay_models), "relay_units": len(relay_units),
            "relay_energy_components": sum(int(row[1]) for row in relay_components),
            "communication_parameters": len(communication), "directed_leg_cache_rows": len(leg_rows),
            "terrain_profile_samples": len(profile_rows),
        },
        "checks": {
            "node_ids_unique": len(nodes) == len(set(nodes)),
            "box_ids_unique": len(box_ids) == len(set(box_ids)),
            "box_service_ids_defined": set(boxes_by_service).issubset(service_ids),
            "each_service_has_box": service_ids == set(boxes_by_service),
            "demand_summary_matches_boxes": demand_counts == box_counts,
            "all_box_mass_positive": all(box.mass_kg > 0 for box in boxes),
            "all_box_volume_positive": all(box.volume_m3 > 0 for box in boxes),
            "all_nodes_within_dem": all(dem.contains(node.lon, node.lat) for node in nodes.values()),
            "first_batch_boxes_have_deadline": all(
                (not box.is_first_batch) or box.first_deadline_s is not None for box in boxes
            ),
            "all_boxes_have_expected_deadline": all(box.expected_deadline_s is not None for box in boxes),
            "terrain_profile_has_one_or_more_sample_per_leg": len(profile_rows) >= len(leg_rows),
        },
        "dem": dem_metadata,
        "node_dem_elevation_difference_m": {
            "max": max(node_dem_differences.values()),
            "mean": sum(node_dem_differences.values()) / len(node_dem_differences),
        },
        "warnings": [
            "节点表海拔与最近 DEM 像元存在分辨率与定位导致的差异；起降作业高度使用节点表，沿线净空使用 DEM。",
            "leg_terrain_profile.csv 保存 ENU-supercover 的真实路径区间与接触类型；巡航净空和通信遮挡分别使用各自上层判据。"
        ],
    }
    write_json(processed_dir / "data_audit.json", audit)

    failed = [name for name, passed in audit["checks"].items() if not passed]
    if failed:
        raise AssertionError(f"数据预处理检查失败：{', '.join(failed)}")
    print(
        "数据预处理完成："
        f"{audit['counts']['service_areas']} 个服务区、{audit['counts']['boxes']} 个货箱、"
        f"{audit['counts']['directed_leg_cache_rows']} 条有向航段缓存"
    )


if __name__ == "__main__":
    main()
