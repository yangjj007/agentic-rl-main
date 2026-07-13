from __future__ import annotations

from opsd_utils.dynamic_trigger import DynamicTriggerMonitor


def test_dynamic_trigger_observes_ema_and_patience_without_early_activation() -> None:
    monitor = DynamicTriggerMonitor(
        ema_alpha=0.5,
        min_progress=0.25,
        mixed_threshold=0.30,
        zero_loss_threshold=0.25,
        patience_steps=2,
    )

    early = monitor.update(progress=0.20, mixed_rate=0.40, zero_loss_rate=0.10)
    assert early.joint_ready is False
    assert early.ready_streak == 0
    assert early.would_trigger is False

    first = monitor.update(progress=0.30, mixed_rate=0.40, zero_loss_rate=0.10)
    assert first.mixed_ready is True
    assert first.zero_loss_ready is True
    assert first.joint_ready is True
    assert first.ready_streak == 1
    assert first.would_trigger is False

    second = monitor.update(progress=0.40, mixed_rate=0.40, zero_loss_rate=0.10)
    assert second.ready_streak == 2
    assert second.would_trigger is True


def test_dynamic_trigger_resets_streak_when_signal_regresses() -> None:
    monitor = DynamicTriggerMonitor(
        ema_alpha=1.0,
        min_progress=0.0,
        mixed_threshold=0.30,
        zero_loss_threshold=0.25,
        patience_steps=2,
    )
    monitor.update(progress=0.1, mixed_rate=0.5, zero_loss_rate=0.1)

    regressed = monitor.update(progress=0.2, mixed_rate=0.1, zero_loss_rate=0.9)

    assert regressed.mixed_ready is False
    assert regressed.zero_loss_ready is False
    assert regressed.ready_streak == 0
    assert regressed.would_trigger is False

