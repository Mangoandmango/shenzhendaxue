"""问题二、三可复用的 SOC 与两阶段充电公式。"""


def remaining_soc(usable_energy_kwh: float, consumed_energy_kwh: float) -> float:
    """由任务能耗计算任务结束时 SOC。"""

    if usable_energy_kwh <= 0:
        raise ValueError("可用电量必须为正")
    return 1.0 - consumed_energy_kwh / usable_energy_kwh


def recharge_time_s(soc: float, full_charge_time_s: float) -> float:
    """实现题目规定的 0%～90% 快充、90%～100% 慢充模型。"""

    if not 0.0 <= soc <= 1.0:
        raise ValueError("SOC 必须位于 [0, 1]")
    if soc < 0.90:
        return full_charge_time_s * (0.65 * (0.90 - soc) / 0.90 + 0.35)
    return full_charge_time_s * 0.35 * (1.0 - soc) / 0.10

