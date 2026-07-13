from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicTriggerConfig:
    ema_alpha: float = 0.10
    min_progress: float = 0.20
    patience_steps: int = 20
    sampling_mixed_max: float = 0.20
    sampling_zero_loss_min: float = 0.70
    rl_mixed_min: float = 0.30
    rl_zero_loss_max: float = 0.30


class DynamicTriggerMonitor:
    """Observe candidate signal-driven phase triggers without controlling training."""

    def __init__(self, config: DynamicTriggerConfig):
        self.config = config
        self._mixed_ema: float | None = None
        self._zero_loss_ema: float | None = None
        self._sampling_streak = 0
        self._rl_streak = 0
        self._sampling_trigger_progress: float | None = None
        self._rl_trigger_progress: float | None = None

    def _update_ema(self, previous: float | None, value: float) -> float:
        if previous is None:
            return float(value)
        alpha = max(0.0, min(float(self.config.ema_alpha), 1.0))
        return alpha * float(value) + (1.0 - alpha) * previous

    def update(self, *, mixed_rate: float, zero_loss_rate: float, progress: float) -> dict[str, float]:
        self._mixed_ema = self._update_ema(self._mixed_ema, mixed_rate)
        self._zero_loss_ema = self._update_ema(self._zero_loss_ema, zero_loss_rate)

        eligible = float(progress) >= float(self.config.min_progress)
        sampling_now = eligible and (
            self._mixed_ema <= float(self.config.sampling_mixed_max)
            and self._zero_loss_ema >= float(self.config.sampling_zero_loss_min)
        )
        rl_now = eligible and (
            self._mixed_ema >= float(self.config.rl_mixed_min)
            and self._zero_loss_ema <= float(self.config.rl_zero_loss_max)
        )

        self._sampling_streak = self._sampling_streak + 1 if sampling_now else 0
        self._rl_streak = self._rl_streak + 1 if rl_now else 0
        patience = max(1, int(self.config.patience_steps))
        if self._sampling_trigger_progress is None and self._sampling_streak >= patience:
            self._sampling_trigger_progress = float(progress)
        if self._rl_trigger_progress is None and self._rl_streak >= patience:
            self._rl_trigger_progress = float(progress)

        return {
            "dynamic_mixed_rate_ema": float(self._mixed_ema),
            "dynamic_zero_loss_rate_ema": float(self._zero_loss_ema),
            "dynamic_sampling_needed_now": float(sampling_now),
            "dynamic_sampling_needed_streak": float(self._sampling_streak),
            "dynamic_sampling_would_trigger": float(self._sampling_trigger_progress is not None),
            "dynamic_sampling_trigger_progress": (
                float(self._sampling_trigger_progress)
                if self._sampling_trigger_progress is not None
                else -1.0
            ),
            "dynamic_rl_ready_now": float(rl_now),
            "dynamic_rl_ready_streak": float(self._rl_streak),
            "dynamic_rl_would_trigger": float(self._rl_trigger_progress is not None),
            "dynamic_rl_trigger_progress": (
                float(self._rl_trigger_progress) if self._rl_trigger_progress is not None else -1.0
            ),
        }
