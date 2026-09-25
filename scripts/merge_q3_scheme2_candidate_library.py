"""合并方案2基础候选列与定向严格DEM核验得到的持续驻留列。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from uav_rescue.settings import PROJECT_ROOT


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    root = PROJECT_ROOT
    header, base = read_rows(root / "outputs/q3/tables/中继候选列.csv")
    _, expansion = read_rows(root / "outputs/q3/scheme2/candidate_expansion/_T004_T006/持续驻留扩充服务列.csv")
    # 只追加连续驻留列：单区列在基础库已存在；这使任务库小且可解释，
    # 同时不把局部网格的同类单区候选无谓放大到主模型中。
    appended = [r for r in expansion if "-P" in r["候选列"]]
    seen = {(r["黑区编号"], r["悬停点"], r["服务开始_s"], r["服务结束_s"], r["任务开始_s"], r["返航_s"])
            for r in base}
    kept: list[dict[str, str]] = []
    for r in appended:
        key = (r["黑区编号"], r["悬停点"], r["服务开始_s"], r["服务结束_s"], r["任务开始_s"], r["返航_s"])
        if key not in seen:
            kept.append(r); seen.add(key)
    output = root / "outputs/q3/scheme2/candidate_library/中继候选列_正式主模型.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(base + kept)
    meta = {
        "base_column_count": len(base), "added_strict_dem_continuous_residence_count": len(kept),
        "total_column_count": len(base) + len(kept),
        "expansion_source": "candidate_expansion/_T004_T006",
        "selection_rule": "仅追加持续驻留列；每列已通过全采样DEM链路核验；不允许悬停点间直接转场。",
    }
    (output.parent / "任务库说明.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
