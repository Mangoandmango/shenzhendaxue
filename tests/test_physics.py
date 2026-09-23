"""公共公式的边界检查。"""

import unittest
from pathlib import Path
import sys


# unittest discover 不会读取 pyproject.toml 中的 src 布局，因此显式加入项目源码目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_rescue.domain import LegGeometry, TransportDrone
from uav_rescue.physics.battery import recharge_time_s
from uav_rescue.physics.energy import equivalent_range_m, leg_energy_kwh
from uav_rescue.physics.flight import leg_time_s, simulate_leg_from_geometry


def sample_drone() -> TransportDrone:
    return TransportDrone("X", "测试", 10, 20, 1, 10, 100, 60, 5, 0.2, 0, 0, 0, 0, 2, 2, 0.8)


class PhysicsFormulaTests(unittest.TestCase):
    def test_equivalent_range_endpoints(self) -> None:
        drone = sample_drone()
        self.assertEqual(equivalent_range_m(drone, 0), 100)
        self.assertEqual(equivalent_range_m(drone, 20), 60)

    def test_two_stage_recharge_boundaries(self) -> None:
        self.assertAlmostEqual(recharge_time_s(0.0, 1000), 1000)
        self.assertAlmostEqual(recharge_time_s(0.9, 1000), 350)
        self.assertAlmostEqual(recharge_time_s(1.0, 1000), 0)

    def test_unified_leg_simulation_matches_existing_formulas(self) -> None:
        """统一接口只封装既有物理公式，不应改变问题一的数值口径。"""

        drone = sample_drone()
        leg = LegGeometry("A", "B", 50.0, 150.0, 20.0, 10.0, 3)
        result = simulate_leg_from_geometry(drone, leg, 10.0)
        self.assertEqual(result.model, "X")
        self.assertEqual(result.payload_kg, 10.0)
        self.assertAlmostEqual(result.equivalent_range_m, equivalent_range_m(drone, 10.0))
        self.assertAlmostEqual(result.flight_time_s, leg_time_s(drone, leg))
        self.assertAlmostEqual(result.energy_kwh, leg_energy_kwh(drone, leg, 10.0))


if __name__ == "__main__":
    unittest.main()
