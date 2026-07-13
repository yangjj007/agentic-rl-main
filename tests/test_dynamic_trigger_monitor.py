from __future__ import annotations

from opsd_utils.dynamic_trigger_monitor import DynamicTriggerConfig, DynamicTriggerMonitor


def _config(**overrides: float | int) -> DynamicTriggerConfig:
    values = {
        "ema_alpha": 1.0,
        "min_progress": 0.2,
        "patience_steps": 2,
        "sampling_mixed_max": 0.2,
        "sampling_zero_loss_min": 0.7,
        "rl_mixed_min": 0.3,
        "rl_zero_loss_max": 0.3,
    }
    values.update(overrides)
    return DynamicTriggerConfig(**values)


def test_sampling_needed_latches_after_patience() -> None:
    monitor = DynamicTriggerMonitor(_config())

    first = monitor.update(mixed_rate=0.1, zero_loss_rate=0.8, progress=0.25)
    second = monitor.update(mixed_rate=0.1, zero_loss_rate=0.8, progress=0.30)

    assert first["dynamic_sampling_needed_now"] == 1.0
    assert first["dynamic_sampling_needed_streak"] == 1.0
    assert first["dynamic_sampling_would_trigger"] == 0.0
    assert second["dynamic_sampling_would_trigger"] == 1.0
    assert second["dynamic_sampling_trigger_progress"] == 0.30
    assert second["dynamic_rl_would_trigger"] == 0.0


def test_rl_ready_latches_independently() -> None:
    monitor = DynamicTriggerMonitor(_config())

    monitor.update(mixed_rate=0.4, zero_loss_rate=0.2, progress=0.40)
    metrics = monitor.update(mixed_rate=0.4, zero_loss_rate=0.2, progress=0.45)

    assert metrics["dynamic_rl_ready_now"] == 1.0
    assert metrics["dynamic_rl_ready_streak"] == 2.0
    assert metrics["dynamic_rl_would_trigger"] == 1.0
    assert metrics["dynamic_rl_trigger_progress"] == 0.45
    assert metrics["dynamic_sampling_would_trigger"] == 0.0


def test_conditions_do_not_accumulate_before_min_progress() -> None:
    monitor = DynamicTriggerMonitor(_config())

    metrics = monitor.update(mixed_rate=0.1, zero_loss_rate=0.9, progress=0.10)

    assert metrics["dynamic_sampling_needed_now"] == 0.0
    assert metrics["dynamic_sampling_needed_streak"] == 0.0


def test_ema_is_updated_and_trigger_progress_stays_latched() -> None:
    monitor = DynamicTriggerMonitor(_config(ema_alpha=0.5, patience_steps=1))

    first = monitor.update(mixed_rate=0.1, zero_loss_rate=0.9, progress=0.25)
    second = monitor.update(mixed_rate=0.5, zero_loss_rate=0.1, progress=0.50)

    assert first["dynamic_mixed_rate_ema"] == 0.1
    assert first["dynamic_zero_loss_rate_ema"] == 0.9
    assert second["dynamic_mixed_rate_ema"] == 0.3
    assert second["dynamic_zero_loss_rate_ema"] == 0.5
    assert second["dynamic_sampling_would_trigger"] == 1.0
    assert second["dynamic_sampling_trigger_progress"] == 0.25
