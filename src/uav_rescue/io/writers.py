"""结果表输出。"""

import csv
import json
from pathlib import Path
from typing import Iterable


def write_csv(path: Path, header: list[str], rows: Iterable[Iterable[object]]) -> None:
    """使用 Excel 兼容的 UTF-8-SIG 编码写出 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    """以 UTF-8 和固定缩进写出可人工阅读的审计结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
