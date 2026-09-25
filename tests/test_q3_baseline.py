import unittest
from unittest.mock import patch

from uav_rescue.geo.coordinates import LocalEnu
from uav_rescue.models.q3_joint import (
    BlindInterval, CommunicationParameters, PositionSample, RelayColumn, RelaySite,
    check_link, three_dimensional_distance_km,
)
from uav_rescue.solvers.q3_solver import _free_space_possible, select_and_assign_columns


class Q3BaselineTests(unittest.TestCase):
    def test_link_limits_match_problem_parameters(self) -> None:
        params = CommunicationParameters(2400, 3, 10, -98, 8, 20, 3, 20, 6, 19, 8, 27, 12, 20)
        self.assertAlmostEqual(params.transport_gateway_limit_db, 122.0)
        self.assertAlmostEqual(params.transport_relay_limit_db, 116.0)
        self.assertAlmostEqual(params.relay_gateway_limit_db, 126.0)

    def test_three_dimensional_distance_is_positive(self) -> None:
        enu = LocalEnu(109.2, 23.0, 0.0)
        self.assertGreater(three_dimensional_distance_km(enu, 109.2, 23.0, 100, 109.2, 23.0, 200), 0)

    def test_free_space_prefilter_only_rejects_impossible_links(self) -> None:
        params = CommunicationParameters(2400, 3, 10, -98, 8, 20, 3, 20, 6, 19, 8, 27, 12, 20)
        enu = LocalEnu(109.2, 23.0, 0.0)
        self.assertTrue(_free_space_possible(
            enu, params, (109.2, 23.0, 100.0), (109.201, 23.0, 200.0),
            params.transport_relay_limit_db,
        ))
        self.assertFalse(_free_space_possible(
            enu, params, (109.2, 23.0, 100.0), (110.2, 23.0, 200.0),
            params.transport_relay_limit_db,
        ))

    def test_obstruction_adds_loss_but_does_not_force_unavailable(self) -> None:
        params = CommunicationParameters(2400, 3, 10, -98, 8, 20, 3, 20, 6, 19, 8, 27, 12, 20)
        enu = LocalEnu(109.2, 23.0, 0.0)
        with patch("uav_rescue.models.q3_joint.evaluate_line_of_sight") as sight:
            sight.return_value.clear = False
            sight.return_value.minimum_clearance_m = -1.0
            sight.return_value.boundary_contact_would_obstruct = False
            sight.return_value.corner_contact_would_obstruct = False
            result = check_link(None, enu, params, (109.2, 23.0, 100.0), (109.2, 23.0, 200.0), 200.0)
        self.assertFalse(result.line_of_sight)
        self.assertTrue(result.available)
        self.assertAlmostEqual(result.margin_db, result.limit_db - result.total_loss_db)

    def test_relay_concurrency_limit_filters_columns(self) -> None:
        intervals = (
            BlindInterval("B1", "T1", 10.0, 20.0, (
                PositionSample("T1", 10.0, 0.0, 0.0, 100.0, "巡航"),
                PositionSample("T1", 20.0, 0.0, 0.0, 100.0, "巡航"),
            )),
            BlindInterval("B2", "T2", 10.0, 20.0, (
                PositionSample("T2", 10.0, 0.0, 0.0, 100.0, "巡航"),
                PositionSample("T2", 20.0, 0.0, 0.0, 100.0, "巡航"),
            )),
        )
        site = RelaySite("P1", 0.0, 0.0, 0.0, 100.0)
        column = RelayColumn(
            "C1", site, ("B1", "B2"), 2, 4, 10.0, 20.0, 0.0, 25.0, 30.0,
            1.0, 0.8, 40.0, 10.0,
        )

        assigned, conflicts = select_and_assign_columns(
            intervals, (column,), relay_count=1, component_count=1, concurrent_transport_limit=1
        )
        self.assertEqual(assigned, [])
        self.assertEqual(conflicts, ["B1", "B2"])

        assigned, conflicts = select_and_assign_columns(
            intervals, (column,), relay_count=1, component_count=1, concurrent_transport_limit=0
        )
        self.assertEqual(conflicts, [])
        self.assertEqual(len(assigned), 1)


if __name__ == "__main__":
    unittest.main()
