#!/usr/bin/env python3
"""无需安装包即可从项目根目录运行问题三方案1与方案2a。"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_rescue.cli.q3 import main  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("demand", "full"), default="full")
    args = parser.parse_args()
    main(stage=args.stage)
