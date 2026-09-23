"""跨问题共享的数据对象。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    """调度中心或服务区。"""

    node_id: str
    name: str
    lon: float
    lat: float
    ground_m: float
    population: int | None = None


@dataclass(frozen=True)
class Box:
    """不可拆分货箱。"""

    box_id: str
    service_id: str
    category: str
    mass_kg: float
    volume_m3: float
    is_first_batch: bool = False
    first_deadline_s: float | None = None
    expected_deadline_s: float | None = None
    priority: float | None = None


@dataclass(frozen=True)
class TransportDrone:
    """运输无人机机型参数。"""

    model: str
    name: str
    empty_mass_kg: float
    max_payload_kg: float
    volume_m3: float
    cruise_speed_mps: float
    empty_range_m: float
    full_range_m: float
    usable_energy_kwh: float
    reserve_ratio: float
    prep_s: float
    load_per_box_s: float
    handoff_base_s: float
    handoff_per_box_s: float
    climb_speed_mps: float
    descent_speed_mps: float
    climb_efficiency: float


@dataclass(frozen=True)
class LegGeometry:
    """一个有向飞行航段的几何量。"""

    origin: str
    destination: str
    distance_m: float
    cruise_altitude_m: float
    climb_m: float
    descent_m: float
    traversed_cells: int


@dataclass(frozen=True)
class LegSimulation:
    """统一航段物理接口的输出：给定机型、航段和当前载荷后的状态量。"""

    geometry: LegGeometry
    model: str
    payload_kg: float
    equivalent_range_m: float
    flight_time_s: float
    energy_kwh: float
