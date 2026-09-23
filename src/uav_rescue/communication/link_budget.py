"""问题三可复用的自由空间损耗和双向链路预算。"""

import math


def free_space_path_loss_db(frequency_mhz: float, distance_km: float) -> float:
    """按题面公式计算自由空间传播损耗。"""

    if frequency_mhz <= 0 or distance_km <= 0:
        raise ValueError("频率和距离必须为正")
    return 32.45 + 20 * math.log10(frequency_mhz) + 20 * math.log10(distance_km)


def allowed_path_loss_db(
    transmit_power_dbm: float,
    transmit_gain_dbi: float,
    receive_gain_dbi: float,
    system_loss_db: float,
    sensitivity_dbm: float,
    fade_margin_db: float,
) -> float:
    """计算单向链路允许的最大总传播损耗。"""

    effective_threshold = sensitivity_dbm + fade_margin_db
    return transmit_power_dbm + transmit_gain_dbi + receive_gain_dbi - system_loss_db - effective_threshold


def link_available(total_path_loss_db: float, forward_limit_db: float, reverse_limit_db: float) -> bool:
    """双向链路取两个方向门限的较小值。"""

    return total_path_loss_db <= min(forward_limit_db, reverse_limit_db)

