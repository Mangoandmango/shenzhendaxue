"""ENU-supercover、半像元边界与视线高度插值的回归测试。"""

import math
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_rescue.geo.dem import DemGrid  # noqa: E402
from uav_rescue.geo.line_of_sight import evaluate_line_of_sight  # noqa: E402


class IdentityEnu:
    """测试专用恒等映射，使 ENU 米与栅格坐标可直接核验。"""

    @staticmethod
    def from_geodetic_m(lon: float, lat: float, height: float) -> tuple[float, float, float]:
        return lon, lat, height

    @staticmethod
    def to_geodetic_m(east: float, north: float, up: float = 0.0) -> tuple[float, float, float]:
        return east, north, up

    @staticmethod
    def horizontal_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        return math.hypot(lon2 - lon1, lat2 - lat1)


def synthetic_dem(default_elevation: float = 0.0) -> DemGrid:
    dem = DemGrid.__new__(DemGrid)
    dem.width = 6
    dem.height = 6
    dem.dx = 1.0
    dem.dy = 1.0
    dem.lon0 = 0.0
    dem.lat0 = 0.0
    dem.pixels = {(col, row): default_elevation for col in range(dem.width) for row in range(dem.height)}
    return dem


class DemSupercoverTests(unittest.TestCase):
    def test_half_pixel_boundary_creates_true_intervals(self) -> None:
        dem = synthetic_dem()
        distance, records = dem.trace_enu_segment(IdentityEnu(), 0.0, 0.0, 2.0, 0.0, 0.2)
        positive = dem.positive_intersections(records)
        self.assertAlmostEqual(distance, 2.0)
        self.assertEqual([(item.row, item.col) for item in positive], [(0, 0), (0, 1), (0, 2)])
        self.assertAlmostEqual(positive[0].s_out_m, 0.5, places=6)
        self.assertAlmostEqual(positive[1].s_in_m, 0.5, places=6)
        self.assertAlmostEqual(positive[1].s_out_m, 1.5, places=6)

    def test_corner_only_cells_are_retained_for_sensitivity(self) -> None:
        dem = synthetic_dem()
        _, records = dem.trace_enu_segment(IdentityEnu(), 0.0, 0.0, 1.0, -1.0, 0.2)
        positive_cells = {(item.row, item.col) for item in dem.positive_intersections(records)}
        corner_cells = {(item.row, item.col) for item in records if item.contact_type == "corner"}
        self.assertEqual(positive_cells, {(0, 0), (1, 1)})
        self.assertEqual(corner_cells, {(0, 1), (1, 0)})

    def test_path_along_half_pixel_boundary_is_classified_for_sensitivity(self) -> None:
        dem = synthetic_dem()
        dem.pixels[(0, 1)] = 200.0
        dem.pixels[(1, 1)] = 200.0
        _, records = dem.trace_enu_segment(IdentityEnu(), 0.5, 0.0, 0.5, -2.0, 0.2)
        positive = dem.positive_intersections(records)
        self.assertTrue(positive)
        self.assertTrue(all(item.contact_type == "boundary" for item in positive))
        result = evaluate_line_of_sight(
            dem, IdentityEnu(), 0.5, 0.0, 100.0, 0.5, -2.0, 100.0,
        )
        self.assertTrue(result.clear)
        self.assertTrue(result.boundary_contact_would_obstruct)

    def test_los_uses_true_path_interval_not_cell_sequence_index(self) -> None:
        dem = synthetic_dem()
        dem.pixels[(1, 0)] = 110.0
        result = evaluate_line_of_sight(
            dem, IdentityEnu(), 0.0, 0.0, 100.0, 2.0, 0.0, 120.0,
        )
        self.assertFalse(result.clear)
        self.assertIn((0, 1), result.obstructing_cells)
        # 中间像元区间为 [0.5, 1.5]，视线最低海拔应为 105 m，净空 -5 m。
        self.assertAlmostEqual(result.minimum_clearance_m, -5.0, places=6)


if __name__ == "__main__":
    unittest.main()
