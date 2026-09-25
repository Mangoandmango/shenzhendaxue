#!/usr/bin/env python3
"""无需安装包即可运行问题二强化版B的ALNS-MILP联合优化。"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_rescue.cli.q2b_joint import main  # noqa: E402


if __name__ == "__main__":
    main()
