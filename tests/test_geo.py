"""公共地理预处理的基本回归测试。"""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_rescue.geo.coordinates import LocalEnu  # noqa: E402


class CoordinateTests(unittest.TestCase):
    def test_enu_origin_is_zero(self) -> None:
        """原点自身必须映射到零东、零北、零天坐标。"""

        enu = LocalEnu(109.2308517, 23.0085095, 127.7)
        east, north, up = enu.from_geodetic_m(109.2308517, 23.0085095, 127.7)
        self.assertAlmostEqual(east, 0.0, places=6)
        self.assertAlmostEqual(north, 0.0, places=6)
        self.assertAlmostEqual(up, 0.0, places=6)

    def test_enu_east_and_north_directions(self) -> None:
        """小范围内增大经度应向东、增大纬度应向北。"""

        enu = LocalEnu(109.0, 23.0, 100.0)
        east, _, _ = enu.from_geodetic_m(109.001, 23.0, 100.0)
        _, north, _ = enu.from_geodetic_m(109.0, 23.001, 100.0)
        self.assertGreater(east, 0.0)
        self.assertGreater(north, 0.0)

    def test_enu_straight_path_round_trips_to_endpoints(self) -> None:
        enu = LocalEnu(109.0, 23.0, 0.0)
        distance, points = enu.straight_path_lonlat(109.0, 23.0, 109.001, 23.001, max_step_m=5.0)
        self.assertGreater(distance, 0.0)
        self.assertAlmostEqual(points[0][0], 109.0, places=8)
        self.assertAlmostEqual(points[0][1], 23.0, places=8)
        self.assertAlmostEqual(points[-1][0], 109.001, places=8)
        self.assertAlmostEqual(points[-1][1], 23.001, places=8)


if __name__ == "__main__":
    unittest.main()
