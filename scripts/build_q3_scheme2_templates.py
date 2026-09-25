"""将问题二 B+-S 与问题三基线转换为方案2的相对时间模板。"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from uav_rescue.settings import PROJECT_ROOT


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, header: list[str], values: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(values)


def main() -> None:
    q2 = PROJECT_ROOT / "outputs/q2b_joint_soft/tables/最终方案"
    q3 = PROJECT_ROOT / "outputs/q3/tables"
    out = PROJECT_ROOT / "outputs/q3/scheme2/templates"
    trips = read_rows(q2 / "运输架次.csv")
    deliveries = read_rows(q2 / "逐箱交付与时间窗.csv")
    black = read_rows(q3 / "直连黑区间.csv")
    columns = read_rows(q3 / "中继候选列.csv")
    boxes = {row["box_id"]: row for row in read_rows(PROJECT_ROOT / "data/processed/boxes.csv")}
    starts = {row["架次编号"]: float(row["开始时刻（s）"]) for row in trips}
    black_by_id = {row["黑区编号"]: row for row in black}

    write_rows(out / "运输架次相对模板.csv",
               ["架次编号", "运输机型", "问题二运输机实体", "问题二电池", "相对返航_s", "架次能耗_kWh", "返航SOC", "访问服务区顺序"],
               [[row["架次编号"], row["机型"], row["无人机编号"], row["电池编号"],
                 float(row["返回O01时刻（s）"]) - starts[row["架次编号"]], row["架次能耗（kWh）"],
                 row["返航SOC"], row["访问服务区顺序"]] for row in trips])
    delivery_values = []
    for row in deliveries:
        box = boxes[row["货箱编号"]]
        trip_id = row["架次编号"]
        delivery_values.append([row["货箱编号"], trip_id, row["服务区"], row["物资类型"], row["时间窗类型"],
                                float(row["交付完成时刻（s）"]) - starts[trip_id],
                                box["first_deadline_s"], box["expected_deadline_s"], row["应急优先系数"]])
    write_rows(out / "逐箱交付相对模板.csv",
               ["货箱编号", "架次编号", "服务区", "物资类型", "时间窗类型", "相对交付_s", "首批硬期限_s", "期望时刻_s", "优先系数"], delivery_values)
    write_rows(out / "相对通信黑区.csv",
               ["黑区编号", "架次编号", "相对开始_s", "相对结束_s", "采样点数"],
               [[row["黑区编号"], row["架次编号"], float(row["开始时刻_s"]) - starts[row["架次编号"]],
                 float(row["结束时刻_s"]) - starts[row["架次编号"]], row["采样点数"]] for row in black])

    template_values, coverage_values = [], []
    for row in columns:
        covered = row["黑区编号"].split("|")
        trips_covered = sorted({black_by_id[item]["架次编号"] for item in covered})
        anchor = trips_covered[0]
        anchor_start = starts[anchor]
        template_values.append([row["候选列"], anchor, "|".join(trips_covered), int(len(trips_covered) > 1),
                                row["悬停点"], row["经度"], row["纬度"], row["离地高度_m"],
                                float(row["任务开始_s"]) - anchor_start, float(row["服务开始_s"]) - anchor_start,
                                float(row["服务结束_s"]) - anchor_start, float(row["返航_s"]) - anchor_start,
                                row["能耗_kWh"], row["返航SOC"], row["最小裕量_dB"]])
        for interval_id in covered:
            interval = black_by_id[interval_id]
            trip_id = interval["架次编号"]
            coverage_values.append([row["候选列"], interval_id, trip_id,
                                    float(interval["开始时刻_s"]) - starts[trip_id],
                                    float(interval["结束时刻_s"]) - starts[trip_id]])
    write_rows(out / "中继服务列相对模板.csv",
               ["服务列", "锚定架次", "覆盖架次", "跨架次共享", "悬停点", "经度", "纬度", "离地高度_m",
                "相对任务开始_s", "相对服务开始_s", "相对服务结束_s", "相对返航_s", "能耗_kWh", "返航SOC", "最小裕量_dB"], template_values)
    write_rows(out / "服务列覆盖相对模板.csv",
               ["服务列", "黑区编号", "架次编号", "黑区相对开始_s", "黑区相对结束_s"], coverage_values)
    manifest = {
        "scheme": "方案2 固定运输结构下的运输时序资源中继通信联合优化",
        "transport_trip_count": len(trips), "black_interval_count": len(black),
        "raw_service_column_count": len(columns),
        "continuous_residence_column_count": sum("-P" in row["候选列"] for row in columns),
        "cross_trip_shared_column_count": sum("|" in row["黑区编号"] for row in columns),
        "time_reference": "每个运输架次起飞时刻为零点；跨架次共享列被标记并须在主模型中施加相同平移量或重建为独立任务。",
        "return_rule": "题面规定中继无人机服务结束后返回O01；主模型允许持续驻留合并黑区，不默认允许悬停点间转场。",
    }
    (out / "模板说明.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
