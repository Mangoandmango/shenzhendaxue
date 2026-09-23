"""问题一候选架次生成与精确集合划分。"""

from dataclasses import dataclass
from functools import lru_cache
import math

from uav_rescue.domain import Box, LegGeometry, TransportDrone
from uav_rescue.physics.flight import simulate_leg_from_geometry


@dataclass(frozen=True)
class CandidateTrip:
    service_id: str
    model: str
    mask: int
    mass_kg: float
    volume_m3: float
    energy_kwh: float
    duration_s: float


@dataclass(frozen=True)
class ParetoSolution:
    """固定架次数下一个能耗—累计作业时间非支配解。"""

    trip_count: int
    energy_kwh: float
    duration_s: float
    trips: tuple[CandidateTrip, ...]


@dataclass(frozen=True)
class ReserveBreakpoint:
    """返航安全余量刚超过某阈值后的最少架次数变化。"""

    reserve_ratio: float
    before_trip_count: int | None
    after_trip_count: int | None


def pareto_filter(solutions: list[ParetoSolution], tolerance: float = 1e-9) -> list[ParetoSolution]:
    """保留能耗和时间均不存在更优解的方案，并去除相同指标的重复方案。"""

    ordered = sorted(solutions, key=lambda item: (item.energy_kwh, item.duration_s, item.trip_count))
    retained: list[ParetoSolution] = []
    best_duration = float("inf")
    for solution in ordered:
        if solution.duration_s < best_duration - tolerance:
            retained.append(solution)
            best_duration = solution.duration_s
    return retained


def generate_candidates(
    service_id: str,
    boxes: list[Box],
    drones: dict[str, TransportDrone],
    outbound: LegGeometry,
    inbound: LegGeometry,
) -> list[CandidateTrip]:
    """枚举该服务区全部满足质量、体积和能量条件的候选架次。"""

    count = len(boxes)
    mass = [0.0] * (1 << count)
    volume = [0.0] * (1 << count)
    box_count = [0] * (1 << count)
    for mask in range(1, 1 << count):
        bit = mask & -mask
        index = bit.bit_length() - 1
        previous = mask ^ bit
        mass[mask] = mass[previous] + boxes[index].mass_kg
        volume[mask] = volume[previous] + boxes[index].volume_m3
        box_count[mask] = box_count[previous] + 1

    result: list[CandidateTrip] = []
    for mask in range(1, 1 << count):
        for model, drone in sorted(drones.items()):
            if mass[mask] > drone.max_payload_kg + 1e-10 or volume[mask] > drone.volume_m3 + 1e-12:
                continue
            outbound_simulation = simulate_leg_from_geometry(drone, outbound, mass[mask])
            inbound_simulation = simulate_leg_from_geometry(drone, inbound, 0.0)
            energy = outbound_simulation.energy_kwh + inbound_simulation.energy_kwh
            if energy > (1.0 - drone.reserve_ratio) * drone.usable_energy_kwh + 1e-10:
                continue
            duration = (
                drone.prep_s
                + box_count[mask] * drone.load_per_box_s
                + outbound_simulation.flight_time_s
                + inbound_simulation.flight_time_s
                + drone.handoff_base_s
                + box_count[mask] * drone.handoff_per_box_s
            )
            result.append(CandidateTrip(service_id, model, mask, mass[mask], volume[mask], energy, duration))
    return result


def prune_dominated_candidates(candidates: list[CandidateTrip], tolerance: float = 1e-9) -> list[CandidateTrip]:
    """删除覆盖相同货箱、但能耗和时间均不占优的机型方案。

    问题一不调度实体无人机，因此同一货箱集合只需保留能耗—时间非支配机型。
    若两个候选指标完全相同，则按机型编号保留一个确定性代表。
    """

    by_mask: dict[int, list[CandidateTrip]] = {}
    for candidate in candidates:
        by_mask.setdefault(candidate.mask, []).append(candidate)

    retained: list[CandidateTrip] = []
    for mask_candidates in by_mask.values():
        ordered = sorted(mask_candidates, key=lambda item: (item.energy_kwh, item.duration_s, item.model))
        best_duration = float("inf")
        for candidate in ordered:
            if candidate.duration_s < best_duration - tolerance:
                retained.append(candidate)
                best_duration = candidate.duration_s
    return retained


def simple_trip_lower_bound(
    boxes: list[Box],
    safe_payload_by_model: dict[str, float | None],
    drones: dict[str, TransportDrone],
) -> tuple[int, int, int]:
    """按总质量和总体积给出不考虑货箱不可拆的简单架次数下界。"""

    feasible_models = [model for model, payload in safe_payload_by_model.items() if payload is not None]
    if not feasible_models:
        raise RuntimeError(f"{boxes[0].service_id} 不存在可完成空载往返的机型")
    maximum_mass = max(float(safe_payload_by_model[model]) for model in feasible_models)
    maximum_volume = max(drones[model].volume_m3 for model in feasible_models)
    mass_bound = math.ceil((sum(box.mass_kg for box in boxes) - 1e-10) / maximum_mass)
    volume_bound = math.ceil((sum(box.volume_m3 for box in boxes) - 1e-12) / maximum_volume)
    return mass_bound, volume_bound, max(mass_bound, volume_bound)


def find_reserve_breakpoints(
    boxes: list[Box],
    candidates_at_zero_reserve: list[CandidateTrip],
    drones: dict[str, TransportDrone],
) -> list[ReserveBreakpoint]:
    """利用有限候选架次的能量阈值，精确定位最少架次数的阶跃点。

    候选架次在余量 ``rho`` 不超过 ``1-E/E_use`` 时可行，因此最少架次数只可能在
    这些有限阈值处发生变化。对有序阈值进行二分定位，避免密集扫描余量网格。
    """

    candidates = [
        candidate for candidate in candidates_at_zero_reserve
        if 1.0 - candidate.energy_kwh / drones[candidate.model].usable_energy_kwh >= 0.0
    ]
    thresholds = sorted({
        1.0 - candidate.energy_kwh / drones[candidate.model].usable_energy_kwh
        for candidate in candidates
    })
    if not thresholds:
        return []

    def minimum_trip_count(reserve_ratio: float) -> int | None:
        feasible = [
            candidate for candidate in candidates
            if 1.0 - candidate.energy_kwh / drones[candidate.model].usable_energy_kwh >= reserve_ratio
        ]
        try:
            score, _ = solve_exact_partition(boxes, feasible)
        except RuntimeError:
            return None
        return score[0]

    state_cache: dict[int, int | None] = {}

    def state_after(index: int) -> int | None:
        if index not in state_cache:
            ratio_after = math.nextafter(thresholds[index], math.inf)
            state_cache[index] = minimum_trip_count(ratio_after)
        return state_cache[index]

    current_state = minimum_trip_count(0.0)
    search_after = -1
    final_index = len(thresholds) - 1
    result: list[ReserveBreakpoint] = []
    while search_after < final_index and state_after(final_index) != current_state:
        left, right = search_after + 1, final_index
        while left < right:
            middle = (left + right) // 2
            if state_after(middle) == current_state:
                left = middle + 1
            else:
                right = middle
        next_state = state_after(left)
        result.append(ReserveBreakpoint(thresholds[left], current_state, next_state))
        current_state = next_state
        search_after = left
    return result


def solve_exact_partition(boxes: list[Box], candidates: list[CandidateTrip]) -> tuple[tuple[int, float, float], list[CandidateTrip]]:
    """依次最小化架次数、总能耗和累计作业时间。"""

    full_mask = (1 << len(boxes)) - 1
    by_box: list[list[CandidateTrip]] = [[] for _ in boxes]
    for candidate in candidates:
        for index in range(len(boxes)):
            if candidate.mask & (1 << index):
                by_box[index].append(candidate)

    @lru_cache(maxsize=None)
    def solve(covered: int):
        if covered == full_mask:
            return (0, 0.0, 0.0), ()
        uncovered = (~covered) & full_mask
        first = (uncovered & -uncovered).bit_length() - 1
        best = None
        for candidate in by_box[first]:
            if candidate.mask & covered:
                continue
            tail = solve(covered | candidate.mask)
            if tail is None:
                continue
            score = (1 + tail[0][0], candidate.energy_kwh + tail[0][1], candidate.duration_s + tail[0][2])
            proposal = score, (candidate,) + tail[1]
            if best is None or score < best[0]:
                best = proposal
        return best

    answer = solve(0)
    if answer is None:
        raise RuntimeError(f"{boxes[0].service_id} 不存在覆盖全部货箱的可行组批")
    return answer[0], list(answer[1])


def solve_pareto_partition(
    boxes: list[Box],
    candidates: list[CandidateTrip],
    fixed_trip_count: int,
) -> list[ParetoSolution]:
    """在固定最少架次数条件下精确求解能耗—累计作业时间 Pareto 前沿。

    对每个已覆盖货箱集合和已使用架次数，只保留能耗、时间均非支配的部分解。
    因后续附加架次对两个指标都只会增加，该剪枝不会丢失最终 Pareto 解。
    """

    full_mask = (1 << len(boxes)) - 1
    by_box: list[list[CandidateTrip]] = [[] for _ in boxes]
    for candidate in candidates:
        for index in range(len(boxes)):
            if candidate.mask & (1 << index):
                by_box[index].append(candidate)

    @lru_cache(maxsize=None)
    def solve(covered: int) -> dict[int, tuple[ParetoSolution, ...]]:
        if covered == full_mask:
            return {0: (ParetoSolution(0, 0.0, 0.0, ()),)}
        uncovered = (~covered) & full_mask
        first = (uncovered & -uncovered).bit_length() - 1
        result: dict[int, list[ParetoSolution]] = {}
        for candidate in by_box[first]:
            if candidate.mask & covered:
                continue
            for tail_count, tail_solutions in solve(covered | candidate.mask).items():
                count = 1 + tail_count
                if count > fixed_trip_count:
                    continue
                current = result.setdefault(count, [])
                current.extend(
                    ParetoSolution(
                        count,
                        candidate.energy_kwh + tail.energy_kwh,
                        candidate.duration_s + tail.duration_s,
                        (candidate,) + tail.trips,
                    )
                    for tail in tail_solutions
                )
                result[count] = pareto_filter(current)
        return {count: tuple(solutions) for count, solutions in result.items()}

    frontier = list(solve(0).get(fixed_trip_count, ()))
    if not frontier:
        raise RuntimeError(f"{boxes[0].service_id} 在 {fixed_trip_count} 架次下不存在 Pareto 可行解")
    return frontier


def solve_pareto_partitions(
    boxes: list[Box],
    candidates: list[CandidateTrip],
    maximum_trip_count: int,
) -> dict[int, list[ParetoSolution]]:
    """一次求出不超过给定架次数的逐架次数精确 Pareto 前沿。"""

    full_mask = (1 << len(boxes)) - 1
    by_box: list[list[CandidateTrip]] = [[] for _ in boxes]
    for candidate in candidates:
        for index in range(len(boxes)):
            if candidate.mask & (1 << index):
                by_box[index].append(candidate)

    @lru_cache(maxsize=None)
    def solve(covered: int) -> dict[int, tuple[ParetoSolution, ...]]:
        if covered == full_mask:
            return {0: (ParetoSolution(0, 0.0, 0.0, ()),)}
        uncovered = (~covered) & full_mask
        first = (uncovered & -uncovered).bit_length() - 1
        result: dict[int, list[ParetoSolution]] = {}
        for candidate in by_box[first]:
            if candidate.mask & covered:
                continue
            for tail_count, tail_solutions in solve(covered | candidate.mask).items():
                count = 1 + tail_count
                if count > maximum_trip_count:
                    continue
                current = result.setdefault(count, [])
                current.extend(
                    ParetoSolution(
                        count,
                        candidate.energy_kwh + tail.energy_kwh,
                        candidate.duration_s + tail.duration_s,
                        (candidate,) + tail.trips,
                    )
                    for tail in tail_solutions
                )
                result[count] = pareto_filter(current)
        return {count: tuple(frontier) for count, frontier in result.items()}

    return {count: list(frontier) for count, frontier in solve(0).items() if count > 0}


def combine_pareto_fronts(fronts: list[list[ParetoSolution]]) -> list[ParetoSolution]:
    """利用服务区之间相互独立的特性，合并各服务区 Pareto 前沿。"""

    combined = [ParetoSolution(0, 0.0, 0.0, ())]
    for frontier in fronts:
        proposals = [
            ParetoSolution(
                previous.trip_count + current.trip_count,
                previous.energy_kwh + current.energy_kwh,
                previous.duration_s + current.duration_s,
                previous.trips + current.trips,
            )
            for previous in combined
            for current in frontier
        ]
        combined = pareto_filter(proposals)
    return combined


def combine_pareto_fronts_by_trip_count(
    fronts: list[dict[int, list[ParetoSolution]]],
    maximum_total_trip_count: int,
) -> dict[int, list[ParetoSolution]]:
    """合并各服务区前沿，并在每个总架次数内分别进行非支配筛选。"""

    combined: dict[int, list[ParetoSolution]] = {0: [ParetoSolution(0, 0.0, 0.0, ())]}
    for service_fronts in fronts:
        proposals: dict[int, list[ParetoSolution]] = {}
        for previous_count, previous_frontier in combined.items():
            for current_count, current_frontier in service_fronts.items():
                total_count = previous_count + current_count
                if total_count > maximum_total_trip_count:
                    continue
                bucket = proposals.setdefault(total_count, [])
                bucket.extend(
                    ParetoSolution(
                        total_count,
                        previous.energy_kwh + current.energy_kwh,
                        previous.duration_s + current.duration_s,
                        previous.trips + current.trips,
                    )
                    for previous in previous_frontier
                    for current in current_frontier
                )
        combined = {count: pareto_filter(frontier) for count, frontier in proposals.items()}
    return combined


def epsilon_representatives(frontier: list[ParetoSolution], maximum: int = 5) -> list[tuple[str, ParetoSolution]]:
    """从完整 Pareto 前沿选取可按 ε 约束解释的代表方案。"""

    if not frontier:
        return []
    ordered = sorted(frontier, key=lambda item: (item.energy_kwh, item.duration_s))
    if len(ordered) <= maximum:
        return [(f"Pareto_{index + 1}", item) for index, item in enumerate(ordered)]
    indices = sorted({round(index * (len(ordered) - 1) / (maximum - 1)) for index in range(maximum)})
    labels = ["最低能耗", "折中方案1", "折中方案2", "折中方案3", "最低时间"]
    return [(labels[position], ordered[index]) for position, index in enumerate(indices)]
