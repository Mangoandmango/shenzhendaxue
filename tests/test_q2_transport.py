"""问题二路线物理量、资源时序与独立复核测试。"""

import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_rescue.domain import (  # noqa: E402
    Box,
    LegGeometry,
    TransportBattery,
    TransportDrone,
    TransportUnit,
)
from uav_rescue.models.q2_transport import (  # noqa: E402
    Objective,
    RouteEvaluator,
    RoutePlan,
    ScheduleResult,
    TypedRoutePlan,
    hard_deadline_s,
    schedule_routes,
    schedule_typed_routes_event_driven,
)
from uav_rescue.solvers.q2_solver import (  # noqa: E402
    pareto_filter_schedules,
    select_pareto_representatives,
)
from uav_rescue.validation.q2 import validate_q2_solution  # noqa: E402


def sample_drone() -> TransportDrone:
    return TransportDrone(
        "A", "测试", 10, 20, 1, 10, 1000, 800, 10, 0.2,
        10, 1, 5, 1, 2, 2, 0.8,
    )


class QuestionTwoTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.boxes = {
            "B1": Box("B1", "S001", "医疗物资", 4, 0.1, True, 1000, 1000, 10),
            "B2": Box("B2", "S002", "普通", 3, 0.1, False, None, 2000, 2),
        }
        nodes = ["O01", "S001", "S002"]
        self.legs = {
            (origin, destination): LegGeometry(origin, destination, 10, 100, 0, 0, 1)
            for origin in nodes for destination in nodes if origin != destination
        }
        self.drones = {"A": sample_drone()}
        self.evaluator = RouteEvaluator(self.boxes, self.drones, self.legs)

    def test_multistop_route_decreases_payload_and_returns_empty(self) -> None:
        evaluations = self.evaluator.evaluate(("B1", "B2"), ("S001", "S002"))
        self.assertEqual(len(evaluations), 1)
        payloads = [leg.payload_before_kg for leg in evaluations[0].legs]
        self.assertEqual(payloads, [7, 3, 0])

    def test_all_expected_delivery_times_are_hard_windows(self) -> None:
        self.assertEqual(hard_deadline_s(self.boxes["B2"]), 2000)
        first_batch = Box("B3", "S001", "饮用水", 1, 0.01, True, 900, 1200, 1)
        self.assertEqual(hard_deadline_s(first_batch), 900)

    def test_drone_can_reuse_after_return_with_second_full_battery(self) -> None:
        units = {"U01": TransportUnit("U01", "A", "O01")}
        batteries = {
            "A-BAT-01": TransportBattery("A-BAT-01", "A", 1000),
            "A-BAT-02": TransportBattery("A-BAT-02", "A", 1000),
        }
        routes = [
            RoutePlan(("B1",), ("S001",)),
            RoutePlan(("B2",), ("S002",)),
        ]
        schedule = schedule_routes(
            routes, self.evaluator, self.boxes, units, batteries
        )
        first, second = schedule.trips
        self.assertEqual(second.start_s, first.return_s)
        self.assertNotEqual(first.battery_id, second.battery_id)
        self.assertGreater(first.battery_release_s, first.return_s)
        checks = validate_q2_solution(
            schedule, self.boxes, self.drones, units, batteries, self.evaluator
        )
        self.assertTrue(all(row.passed for row in checks))

    def test_event_decoder_respects_typed_routes_and_resource_events(self) -> None:
        units = {"U01": TransportUnit("U01", "A", "O01")}
        batteries = {
            "A-BAT-01": TransportBattery("A-BAT-01", "A", 1000),
            "A-BAT-02": TransportBattery("A-BAT-02", "A", 1000),
        }
        plans = (
            TypedRoutePlan(RoutePlan(("B1",), ("S001",)), "A"),
            TypedRoutePlan(RoutePlan(("B2",), ("S002",)), "A"),
        )
        schedule = schedule_typed_routes_event_driven(
            plans, self.evaluator, self.boxes, units, batteries
        )
        first, second = schedule.trips
        self.assertEqual(first.evaluation.model, "A")
        self.assertEqual(second.start_s, first.return_s)
        self.assertNotEqual(first.battery_id, second.battery_id)
        checks = validate_q2_solution(
            schedule, self.boxes, self.drones, units, batteries, self.evaluator
        )
        self.assertTrue(all(row.passed for row in checks))

    def test_pareto_filter_and_representative_selection(self) -> None:
        def result(soft: float, makespan: float, energy: float, trips: int) -> ScheduleResult:
            return ScheduleResult((), Objective(0, 0.0, soft, makespan, energy, trips))

        timely = result(10, 100, 10, 3)
        fast_efficient = result(20, 90, 8, 2)
        middle = result(15, 95, 9, 3)
        dominated = result(12, 110, 11, 4)
        infeasible = ScheduleResult((), Objective(1, 1.0, 0, 1, 1, 1))
        front = pareto_filter_schedules([
            timely, fast_efficient, middle, dominated, infeasible,
        ])
        self.assertEqual(set(front), {timely, fast_efficient, middle})
        representatives = select_pareto_representatives(front)
        self.assertEqual(representatives["完工时间优先"], fast_efficient)
        self.assertEqual(representatives["及时性优先"], fast_efficient)
        self.assertEqual(representatives["能耗优先"], fast_efficient)
        self.assertEqual(representatives["最少架次优先"], fast_efficient)
        self.assertIn(representatives["综合折中"], front)


if __name__ == "__main__":
    unittest.main()
