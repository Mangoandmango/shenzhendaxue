"""问题三方案 A：固定问题二运输排程下的通信需求模型。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from uav_rescue.communication.link_budget import allowed_path_loss_db, free_space_path_loss_db
from uav_rescue.geo.coordinates import LocalEnu
from uav_rescue.geo.dem import DemGrid
from uav_rescue.geo.line_of_sight import evaluate_line_of_sight


@dataclass(frozen=True)
class PositionSample:
    trip_id: str
    time_s: float
    lon: float
    lat: float
    altitude_m: float
    phase: str


@dataclass(frozen=True)
class CommunicationParameters:
    frequency_mhz: float
    system_loss_db: float
    obstruction_loss_db: float
    sensitivity_dbm: float
    fade_margin_db: float
    transport_power_dbm: float
    transport_gain_dbi: float
    relay_access_power_dbm: float
    relay_access_gain_dbi: float
    relay_backhaul_power_dbm: float
    relay_backhaul_gain_dbi: float
    gateway_power_dbm: float
    gateway_gain_dbi: float
    gateway_height_agl_m: float

    def _limit(self, power_a: float, gain_a: float, power_b: float, gain_b: float) -> float:
        a_to_b = allowed_path_loss_db(power_a, gain_a, gain_b, self.system_loss_db,
                                      self.sensitivity_dbm, self.fade_margin_db)
        b_to_a = allowed_path_loss_db(power_b, gain_b, gain_a, self.system_loss_db,
                                      self.sensitivity_dbm, self.fade_margin_db)
        return min(a_to_b, b_to_a)

    @property
    def transport_gateway_limit_db(self) -> float:
        return self._limit(self.transport_power_dbm, self.transport_gain_dbi,
                           self.gateway_power_dbm, self.gateway_gain_dbi)

    @property
    def transport_relay_limit_db(self) -> float:
        return self._limit(self.transport_power_dbm, self.transport_gain_dbi,
                           self.relay_access_power_dbm, self.relay_access_gain_dbi)

    @property
    def relay_gateway_limit_db(self) -> float:
        return self._limit(self.relay_backhaul_power_dbm, self.relay_backhaul_gain_dbi,
                           self.gateway_power_dbm, self.gateway_gain_dbi)


@dataclass(frozen=True)
class LinkCheck:
    available: bool
    total_loss_db: float
    limit_db: float
    margin_db: float
    line_of_sight: bool
    minimum_clearance_m: float
    boundary_contact_would_obstruct: bool
    corner_contact_would_obstruct: bool


@dataclass(frozen=True)
class BlindInterval:
    interval_id: str
    trip_id: str
    start_s: float
    end_s: float
    samples: tuple[PositionSample, ...]


@dataclass(frozen=True)
class RelaySite:
    site_id: str
    lon: float
    lat: float
    ground_m: float
    height_agl_m: float

    @property
    def altitude_m(self) -> float:
        return self.ground_m + self.height_agl_m


@dataclass(frozen=True)
class RelayColumn:
    column_id: str
    site: RelaySite
    covered_interval_ids: tuple[str, ...]
    max_concurrent_transports: int
    covered_sample_count: int
    service_start_s: float
    service_end_s: float
    mission_start_s: float
    return_s: float
    unit_release_s: float
    energy_kwh: float
    end_soc: float
    component_release_s: float
    minimum_margin_db: float


def three_dimensional_distance_km(enu: LocalEnu, lon1: float, lat1: float, altitude1_m: float,
                                  lon2: float, lat2: float, altitude2_m: float) -> float:
    horizontal_m = enu.horizontal_distance_m(lon1, lat1, lon2, lat2)
    return max(math.hypot(horizontal_m, altitude2_m - altitude1_m) / 1000.0, 1e-9)


def check_link(dem: DemGrid, enu: LocalEnu, params: CommunicationParameters,
               endpoint_a: tuple[float, float, float],
               endpoint_b: tuple[float, float, float], limit_db: float) -> LinkCheck:
    lon1, lat1, alt1 = endpoint_a
    lon2, lat2, alt2 = endpoint_b
    sight = evaluate_line_of_sight(dem, enu, lon1, lat1, alt1, lon2, lat2, alt2)
    loss = free_space_path_loss_db(
        params.frequency_mhz,
        three_dimensional_distance_km(enu, lon1, lat1, alt1, lon2, lat2, alt2),
    ) + (0.0 if sight.clear else params.obstruction_loss_db)
    margin = limit_db - loss
    return LinkCheck(
        margin >= -1e-9, loss, limit_db, margin, sight.clear,
        sight.minimum_clearance_m, sight.boundary_contact_would_obstruct,
        sight.corner_contact_would_obstruct,
    )


def extract_blind_intervals(samples: list[PositionSample],
                            direct_available: dict[tuple[str, float], bool],
                            time_step_s: float, max_gap_steps: int = 1) -> tuple[BlindInterval, ...]:
    by_trip: dict[str, list[PositionSample]] = {}
    for sample in samples:
        if not direct_available[(sample.trip_id, sample.time_s)]:
            by_trip.setdefault(sample.trip_id, []).append(sample)
    intervals: list[BlindInterval] = []
    for trip_id, rows in sorted(by_trip.items()):
        rows.sort(key=lambda item: item.time_s)
        groups: list[list[PositionSample]] = []
        for row in rows:
            if not groups or row.time_s - groups[-1][-1].time_s > time_step_s * max_gap_steps + 1e-6:
                groups.append([row])
            else:
                groups[-1].append(row)
        for index, group in enumerate(groups, start=1):
            intervals.append(BlindInterval(f"{trip_id}-B{index:02d}", trip_id,
                                           group[0].time_s, group[-1].time_s, tuple(group)))
    return tuple(intervals)


def split_blind_intervals(intervals: tuple[BlindInterval, ...],
                          maximum_duration_s: float) -> tuple[BlindInterval, ...]:
    """把长黑区切成可切换中继的服务片，同时保留全部采样点。"""
    if maximum_duration_s <= 0:
        return intervals
    result: list[BlindInterval] = []
    for interval in intervals:
        chunks: list[list[PositionSample]] = []
        for sample in interval.samples:
            if not chunks or sample.time_s - chunks[-1][0].time_s > maximum_duration_s + 1e-7:
                chunks.append([sample])
            else:
                chunks[-1].append(sample)
        if len(chunks) == 1:
            result.append(interval)
            continue
        for index, chunk in enumerate(chunks, start=1):
            result.append(BlindInterval(
                f"{interval.interval_id}-S{index:02d}", interval.trip_id,
                chunk[0].time_s, chunk[-1].time_s, tuple(chunk),
            ))
    return tuple(result)
