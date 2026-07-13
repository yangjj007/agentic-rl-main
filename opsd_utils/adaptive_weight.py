"""Adaptive OPD loss weighting helpers."""
from __future__ import annotations

import math
from typing import Any


def _finite_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def opsd_adaptive_multiplier(
    reward_std_mean: Any,
    *,
    enabled: bool,
    std_target: float = 0.25,
    max_mult: float = 2.0,
) -> float:
    """Return variance-adaptive OPD multiplier.

    Formula:
        1 + (max_mult - 1) * clamp(1 - reward_std_mean / std_target, 0, 1)
    """
    if not enabled:
        return 1.0
    target = _finite_float(std_target, 0.25)
    max_multiplier = max(1.0, _finite_float(max_mult, 2.0))
    if target <= 0:
        return 1.0
    reward_std = max(0.0, _finite_float(reward_std_mean, target))
    low_variance_ratio = max(0.0, min(1.0, 1.0 - reward_std / target))
    return 1.0 + (max_multiplier - 1.0) * low_variance_ratio


def effective_opsd_weight(
    base_opsd_weight: Any,
    reward_std_mean: Any,
    *,
    enabled: bool,
    std_target: float = 0.25,
    max_mult: float = 2.0,
) -> tuple[float, float]:
    """Return (effective_weight, adaptive_multiplier)."""
    base_weight = _finite_float(base_opsd_weight, 1.0)
    multiplier = opsd_adaptive_multiplier(
        reward_std_mean,
        enabled=enabled,
        std_target=std_target,
        max_mult=max_mult,
    )
    return base_weight * multiplier, multiplier
