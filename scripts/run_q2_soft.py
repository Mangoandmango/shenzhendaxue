#!/usr/bin/env python3
"""运行问题二增强型方案A的普通物资软约束对照版本。"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_rescue.cli.q2_soft import main  # noqa: E402


if __name__ == "__main__":
    main()
