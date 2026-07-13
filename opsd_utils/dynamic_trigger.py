from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicTriggerSnapshot:
    mixed_rate_ema: float
    zero_loss_rate_ema: float
    mixed_ready: bool
    zero_loss_ready: bool
    joint_ready: bool
    ready_streak: int
    would_trigger: bool


class DynamicTriggerMonitor:
    """Observe a future signal-driven phase gate without controlling training."""

    def __init__(
        self,
        *,
        ema_alpha: float = 0.1,
        min_progress: float = 0.25,
        mixed_threshold: float = 0.30,
        zero_loss_threshold: float = 0.25,
        patience_steps: int = 20,
    ) -> None:
        self.ema_alpha = max(0.0, min(float(ema_alpha), 1.0))
        self.min_progress = max(0.0, min(float(min_progress), 1.0))
        self.mixed_threshold = float(mixed_threshold)
        self.zero_loss_threshold = float(zero_loss_threshold)
        self.patience_steps = max(1, int(patience_steps))
        self._mixed_ema: float | None = None
        self._zero_loss_ema: float | None = None
        self._ready_streak = 0

    @staticmethod
    def _ema(previous: float | None, value: float, alpha: float) -> float:
        if previous is None:
            return float(value)
        return alpha * float(value) + (1.0 - alpha) * previous

    def update(
        self,
        *,
        progress: float,
        mixed_rate: float,
        zero_loss_rate: float,
    ) -> DynamicTriggerSnapshot:
        self._mixed_ema = self._ema(self._mixed_ema, mixed_rate, self.ema_alpha)
        self._zero_loss_ema = self._ema(self._zero_loss_ema, zero_loss_rate, self.ema_alpha)
        progress_ready = float(progress) >= self.min_progress
        mixed_ready = self._mixed_ema >= self.mixed_threshold
        zero_loss_ready = self._zero_loss_ema <= self.zero_loss_threshold
        joint_ready = progress_ready and mixed_ready and zero_loss_ready
        self._ready_streak = self._ready_streak + 1 if joint_ready else 0
        return DynamicTriggerSnapshot(
            mixed_rate_ema=self._mixed_ema,
            zero_loss_rate_ema=self._zero_loss_ema,
            mixed_ready=mixed_ready,
            zero_loss_ready=zero_loss_ready,
            joint_ready=joint_ready,
            ready_streak=self._ready_streak,
            would_trigger=self._ready_streak >= self.patience_steps,
        )

