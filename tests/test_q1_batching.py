"""问题一集合划分与多目标剪枝的单元测试。"""

import unittest
from pathlib import Path
import sys


# unittest discover 不会读取 pyproject.toml 中的 src 布局，因此显式加入项目源码目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_rescue.domain import Box, TransportDrone
from uav_rescue.models.q1_batching import (
    CandidateTrip,
    find_reserve_breakpoints,
    prune_dominated_candidates,
    simple_trip_lower_bound,
    solve_pareto_partition,
)


class QuestionOneBatchingTests(unittest.TestCase):
    @staticmethod
    def sample_drone(model: str, energy: float = 10.0) -> TransportDrone:
        return TransportDrone(model, "测试", 10, 10, 1, 10, 100, 60, energy, 0.2, 0, 0, 0, 0, 2, 2, 0.8)

    def test_fixed_trip_count_pareto_frontier(self) -> None:
        """同为两架次时，应同时保留低能耗慢方案与高能耗快方案。"""

        boxes = [
            Box("B1", "S001", "测试", 1.0, 0.01, None),
            Box("B2", "S001", "测试", 1.0, 0.01, None),
        ]
        candidates = [
            # 低能耗、长时间的两架次方案：总指标为 (2 kWh, 6 s)。
            CandidateTrip("S001", "A", 0b01, 1.0, 0.01, 1.0, 3.0),
            CandidateTrip("S001", "A", 0b10, 1.0, 0.01, 1.0, 3.0),
            # 高能耗、短时间的两架次方案：总指标为 (6 kWh, 2 s)。
            CandidateTrip("S001", "B", 0b01, 1.0, 0.01, 3.0, 1.0),
            CandidateTrip("S001", "B", 0b10, 1.0, 0.01, 3.0, 1.0),
        ]

        frontier = solve_pareto_partition(boxes, candidates, fixed_trip_count=2)
        metrics = {(item.energy_kwh, item.duration_s) for item in frontier}
        # 两种机型允许混搭，故还会产生 (4 kWh, 4 s) 的真实折中方案。
        self.assertEqual(metrics, {(2.0, 6.0), (4.0, 4.0), (6.0, 2.0)})

    def test_dominance_pruning_only_compares_same_box_subset(self) -> None:
        candidates = [
            CandidateTrip("S001", "A", 0b01, 1, 0.1, 1.0, 4.0),
            CandidateTrip("S001", "B", 0b01, 1, 0.1, 2.0, 5.0),  # 同集合且两项均差
            CandidateTrip("S001", "C", 0b10, 1, 0.1, 3.0, 6.0),  # 不同集合，不能比较
        ]
        retained = prune_dominated_candidates(candidates)
        self.assertEqual({(item.model, item.mask) for item in retained}, {("A", 0b01), ("C", 0b10)})

    def test_simple_lower_bound_uses_safe_mass_and_volume(self) -> None:
        boxes = [
            Box("B1", "S001", "测试", 6.0, 0.6, None),
            Box("B2", "S001", "测试", 6.0, 0.6, None),
        ]
        drones = {"A": self.sample_drone("A")}
        self.assertEqual(simple_trip_lower_bound(boxes, {"A": 10.0}, drones), (2, 2, 2))

    def test_reserve_breakpoint_detects_split_and_infeasibility(self) -> None:
        boxes = [
            Box("B1", "S001", "测试", 1.0, 0.1, None),
            Box("B2", "S001", "测试", 1.0, 0.1, None),
        ]
        drones = {"A": self.sample_drone("A")}
        candidates = [
            CandidateTrip("S001", "A", 0b11, 2, 0.2, 8.0, 1.0),  # 超过20%余量后失效
            CandidateTrip("S001", "A", 0b01, 1, 0.1, 6.0, 1.0),
            CandidateTrip("S001", "A", 0b10, 1, 0.1, 6.0, 1.0),  # 超过40%余量后失效
        ]
        breakpoints = find_reserve_breakpoints(boxes, candidates, drones)
        self.assertEqual(
            [(round(item.reserve_ratio, 6), item.before_trip_count, item.after_trip_count) for item in breakpoints],
            [(0.2, 1, 2), (0.4, 2, None)],
        )


if __name__ == "__main__":
    unittest.main()
