import unittest
from unittest.mock import patch

from uav_rescue.geo.coordinates import LocalEnu
from uav_rescue.models.q3_joint import CommunicationParameters, check_link, three_dimensional_distance_km


class Q3BaselineTests(unittest.TestCase):
    def test_link_limits_match_problem_parameters(self) -> None:
        params = CommunicationParameters(2400, 3, 10, -98, 8, 20, 3, 20, 6, 19, 8, 27, 12, 20)
        self.assertAlmostEqual(params.transport_gateway_limit_db, 122.0)
        self.assertAlmostEqual(params.transport_relay_limit_db, 116.0)
        self.assertAlmostEqual(params.relay_gateway_limit_db, 126.0)

    def test_three_dimensional_distance_is_positive(self) -> None:
        enu = LocalEnu(109.2, 23.0, 0.0)
        self.assertGreater(three_dimensional_distance_km(enu, 109.2, 23.0, 100, 109.2, 23.0, 200), 0)

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


if __name__ == "__main__":
    unittest.main()
