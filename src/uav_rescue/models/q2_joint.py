"""问题二联合运输—资源排程的时间窗口径。"""

from __future__ import annotations

from enum import StrEnum

from uav_rescue.domain import Box


class DeadlinePolicy(StrEnum):
    """问题二保留的两种时间窗口径。"""

    ORDINARY_SOFT = "ordinary_soft"
    ALL_HARD = "all_hard"


def hard_deadline_s(box: Box, policy: DeadlinePolicy) -> float | None:
    """返回货箱的有效硬截止时间。

    普通物资软时间窗口径下，医疗物资期望时间与首批保障截止为硬约束；
    全硬口径下，所有货箱期望时间均为硬约束。多个硬截止同时存在时取较早值。
    """

    deadlines: list[float] = []
    if policy == DeadlinePolicy.ALL_HARD or box.category == "医疗物资":
        if box.expected_deadline_s is not None:
            deadlines.append(box.expected_deadline_s)
    if box.is_first_batch and box.first_deadline_s is not None:
        deadlines.append(box.first_deadline_s)
    return min(deadlines) if deadlines else None


def is_soft_expected_window(box: Box, policy: DeadlinePolicy) -> bool:
    """普通物资期望送达时间是否作为软时间窗计罚。"""

    return (
        policy == DeadlinePolicy.ORDINARY_SOFT
        and box.category != "医疗物资"
        and box.expected_deadline_s is not None
    )


__all__ = ["DeadlinePolicy", "hard_deadline_s", "is_soft_expected_window"]
