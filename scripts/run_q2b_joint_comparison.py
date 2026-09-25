#!/usr/bin/env python3
"""在两次完整独立搜索之后，用共享运输结构池公平复评 B+-S/B+-H。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_rescue.cli.q2b_joint import run  # noqa: E402
from uav_rescue.models.q2_joint import DeadlinePolicy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="运行问题二强化版B共享候选池公平比较")
    parser.add_argument(
        "--soft-source", type=Path,
        default=ROOT / "outputs" / "q2b_joint_soft",
    )
    parser.add_argument(
        "--hard-source", type=Path,
        default=ROOT / "outputs" / "q2b_joint_hard",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "outputs" / "q2b_joint_comparison",
    )
    parser.add_argument("--feedback-rounds", type=int, default=1)
    parser.add_argument("--candidates-per-round", type=int, default=6)
    parser.add_argument("--milp-time-limit-per-stage", type=float, default=10.0)
    args = parser.parse_args()
    sources = (args.hard_source, args.soft_source)
    results = {}
    for policy, directory_name in (
        (DeadlinePolicy.ORDINARY_SOFT, "ordinary_soft"),
        (DeadlinePolicy.ALL_HARD, "all_hard"),
    ):
        results[policy.value] = run(
            policy,
            args.output_root / directory_name,
            outer_seeds=0,
            outer_iterations=0,
            initial_candidate_limit=0,
            feedback_rounds=args.feedback_rounds,
            candidates_per_round=args.candidates_per_round,
            milp_time_limit_per_stage_s=args.milp_time_limit_per_stage,
            additional_route_sources=sources,
        )
    for policy, metadata in results.items():
        objective = metadata["joint_best_objective"]
        print(
            f"{policy}: 硬违约={objective['hard_violation_count']}，"
            f"软迟到={objective['weighted_soft_tardiness']:.3f}，"
            f"完工={objective['makespan_s']:.3f}s，"
            f"能耗={objective['energy_kwh']:.6f}kWh，"
            f"架次={objective['trip_count']}"
        )


if __name__ == "__main__":
    main()
