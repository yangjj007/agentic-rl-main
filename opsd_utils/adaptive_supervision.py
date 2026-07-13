from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveSupervisionConfig:
    ema_alpha: float = 0.10
    target_readiness: float = 0.20
    opsd_initial_weight: float = 1.50
    opsd_final_weight: float = 0.50
    teacher_initial_weight: float = 0.50
    teacher_final_weight: float = 0.0
    opd_initial_cap: int = 8
    opd_final_cap: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.ema_alpha) <= 1.0:
            raise ValueError("ema_alpha must be between 0 and 1")
        if not math.isfinite(float(self.target_readiness)) or self.target_readiness <= 0:
            raise ValueError("target_readiness must be finite and positive")
        if self.opd_initial_cap < self.opd_final_cap:
            raise ValueError("opd_initial_cap must be >= opd_final_cap")


@dataclass(frozen=True)
class AdaptiveSupervisionState:
    step: int
    update_count: int
    mixed_rate: float
    zero_loss_rate: float
    mixed_ema: float
    zero_loss_ema: float
    readiness: float
    mastery: float
    supervision: float
    opsd_weight: float
    teacher_traj_weight: float
    opd_max_per_prompt: int


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _finite_rate(value: float, fallback: float) -> float:
    value = float(value)
    return _clamp01(value) if math.isfinite(value) else fallback


def _smoothstep(value: float) -> float:
    value = _clamp01(value)
    return value * value * (3.0 - 2.0 * value)


def _interpolate(initial: float, final: float, supervision: float) -> float:
    return float(final) + (float(initial) - float(final)) * supervision


class AdaptiveSupervisionController:
    def __init__(self, config: AdaptiveSupervisionConfig):
        self.config = config
        self._last_step: int | None = None
        self._state = self._make_state(
            step=-1,
            update_count=0,
            mixed_rate=0.0,
            zero_loss_rate=1.0,
            mixed_ema=0.0,
            zero_loss_ema=1.0,
            mastery=0.0,
        )

    @property
    def state(self) -> AdaptiveSupervisionState:
        return self._state

    def _make_state(
        self,
        *,
        step: int,
        update_count: int,
        mixed_rate: float,
        zero_loss_rate: float,
        mixed_ema: float,
        zero_loss_ema: float,
        mastery: float,
    ) -> AdaptiveSupervisionState:
        readiness = _clamp01(mixed_ema * (1.0 - zero_loss_ema))
        mastery = max(float(mastery), readiness)
        transition = _smoothstep(mastery / float(self.config.target_readiness))
        supervision = 1.0 - transition
        cap_value = _interpolate(
            self.config.opd_initial_cap,
            self.config.opd_final_cap,
            supervision,
        )
        return AdaptiveSupervisionState(
            step=int(step),
            update_count=int(update_count),
            mixed_rate=float(mixed_rate),
            zero_loss_rate=float(zero_loss_rate),
            mixed_ema=float(mixed_ema),
            zero_loss_ema=float(zero_loss_ema),
            readiness=readiness,
            mastery=mastery,
            supervision=supervision,
            opsd_weight=_interpolate(
                self.config.opsd_initial_weight,
                self.config.opsd_final_weight,
                supervision,
            ),
            teacher_traj_weight=_interpolate(
                self.config.teacher_initial_weight,
                self.config.teacher_final_weight,
                supervision,
            ),
            opd_max_per_prompt=max(
                self.config.opd_final_cap,
                min(self.config.opd_initial_cap, int(math.ceil(cap_value))),
            ),
        )

    def update(
        self,
        *,
        step: int,
        mixed_rate: float,
        zero_loss_rate: float,
    ) -> AdaptiveSupervisionState:
        step = int(step)
        if self._last_step == step:
            return self._state

        mixed_rate = _finite_rate(mixed_rate, 0.0)
        zero_loss_rate = _finite_rate(zero_loss_rate, 1.0)
        alpha = float(self.config.ema_alpha)
        mixed_ema = alpha * mixed_rate + (1.0 - alpha) * self._state.mixed_ema
        zero_loss_ema = alpha * zero_loss_rate + (1.0 - alpha) * self._state.zero_loss_ema
        self._state = self._make_state(
            step=step,
            update_count=self._state.update_count + 1,
            mixed_rate=mixed_rate,
            zero_loss_rate=zero_loss_rate,
            mixed_ema=mixed_ema,
            zero_loss_ema=zero_loss_ema,
            mastery=self._state.mastery,
        )
        self._last_step = step
        return self._state

    def update_signal(
        self,
        *,
        step: int,
        signal_rate: float,
    ) -> AdaptiveSupervisionState:
        step = int(step)
        if self._last_step == step:
            return self._state

        signal_rate = _finite_rate(signal_rate, 0.0)
        alpha = float(self.config.ema_alpha)
        signal_ema = alpha * signal_rate + (1.0 - alpha) * self._state.mixed_ema
        self._state = self._make_state(
            step=step,
            update_count=self._state.update_count + 1,
            mixed_rate=signal_rate,
            zero_loss_rate=0.0,
            mixed_ema=signal_ema,
            zero_loss_ema=0.0,
            mastery=self._state.mastery,
        )
        self._last_step = step
        return self._state
