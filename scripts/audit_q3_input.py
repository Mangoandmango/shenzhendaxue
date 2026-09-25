#!/usr/bin/env python3
"""审计问题三固定运输输入与问题二 B+-S 交付时刻的一致性。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_rescue.solvers.q3_solver import _operation_altitude, _rows  # noqa: E402
from uav_rescue.settings import load_toml  # noqa: E402


def main() -> None:
    config = load_toml("configs/q3.toml")
    baseline = config["baseline"]
    nodes = {row["node_id"]: row for row in _rows(PROJECT_ROOT / "data/processed/nodes.csv")}
    drones = {row["model"]: row for row in _rows(PROJECT_ROOT / "data/processed/transport_drones.csv")}
    geometry = {(row["origin_id"], row["destination_id"]): row
                for row in _rows(PROJECT_ROOT / "data/cache/leg_geometry.csv")}
    trips = _rows(PROJECT_ROOT / baseline["transport_schedule"])
    deliveries = _rows(PROJECT_ROOT / baseline["transport_deliveries"])
    by_trip_service: dict[tuple[str, str], list[float]] = {}
    for row in deliveries:
        service = row.get("服务区编号") or row.get("服务区")
        key = (row["架次编号"], service)
        by_trip_service.setdefault(key, []).append(float(row["交付完成时刻（s）"]))

    errors: list[str] = []
    checked_services = 0
    for trip in trips:
        trip_id = trip["架次编号"]
        model = trip.get("机型编号") or trip.get("机型")
        drone = drones[model]
        sequence = trip["访问服务区顺序"].split("->")
        box_count = sum(len(values) for (candidate_trip, _), values in by_trip_service.items()
                        if candidate_trip == trip_id)
        current = float(trip["开始时刻（s）"]) + float(drone["prep_s"])
        current += box_count * float(drone["load_per_box_s"])
        origin = "O01"
        for destination in sequence + ["O01"]:
            leg = geometry[(origin, destination)]
            cruise_altitude = float(leg["cruise_altitude_m"])
            climb = max(0.0, cruise_altitude - _operation_altitude(nodes[origin])) / float(drone["climb_speed_mps"])
            cruise = float(leg["horizontal_distance_m"]) / float(drone["cruise_speed_mps"])
            descent = max(0.0, cruise_altitude - _operation_altitude(nodes[destination])) / float(drone["descent_speed_mps"])
            current += climb + cruise + descent
            if destination != "O01":
                current += float(drone["handoff_base_s"]) + len(by_trip_service[(trip_id, destination)]) * float(drone["handoff_per_box_s"])
                checked_services += 1
                for recorded in by_trip_service[(trip_id, destination)]:
                    if abs(recorded - current) > 1e-4:
                        errors.append(f"{trip_id}/{destination}: 记录={recorded}, 重算={current}")
            origin = destination
        returned = float(trip["返回O01时刻（s）"])
        if abs(returned - current) > 1e-4:
            errors.append(f"{trip_id}/返航: 记录={returned}, 重算={current}")

    referenced_trips = {row["架次编号"] for row in deliveries}
    scheduled_trips = {row["架次编号"] for row in trips}
    orphan_trips = sorted(referenced_trips - scheduled_trips)
    if orphan_trips:
        errors.append("逐箱交付表含未排程架次: " + ",".join(orphan_trips))
    result = {
        "transport_schedule": baseline["transport_schedule"],
        "deliveries": baseline["transport_deliveries"],
        "trip_count": len(trips),
        "delivery_count": len(deliveries),
        "checked_service_visits": checked_services,
        "error_count": len(errors),
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
