from __future__ import annotations

from opsd_utils.phase_schedule import (
    resolve_schedule_step,
    schedule_active,
    training_progress,
)


def test_progress_boundaries_scale_with_training_horizon() -> None:
    assert resolve_schedule_step(
        mode="progress", absolute_step=999, progress=0.25, max_steps=10
    ) == 3
    assert resolve_schedule_step(
        mode="progress", absolute_step=999, progress=0.50, max_steps=10
    ) == 5
    assert resolve_schedule_step(
        mode="progress", absolute_step=999, progress=0.75, max_steps=10
    ) == 8

    assert resolve_schedule_step(
        mode="progress", absolute_step=999, progress=0.25, max_steps=588
    ) == 147
    assert resolve_schedule_step(
        mode="progress", absolute_step=999, progress=0.50, max_steps=588
    ) == 294
    assert resolve_schedule_step(
        mode="progress", absolute_step=999, progress=0.75, max_steps=588
    ) == 441


def test_step_mode_preserves_absolute_boundary() -> None:
    assert resolve_schedule_step(
        mode="step", absolute_step=294, progress=0.10, max_steps=1000
    ) == 294


def test_schedule_active_uses_resolved_progress_boundary() -> None:
    assert not schedule_active(
        global_step=4,
        mode="progress",
        absolute_step=294,
        progress=0.50,
        max_steps=10,
    )
    assert schedule_active(
        global_step=5,
        mode="progress",
        absolute_step=294,
        progress=0.50,
        max_steps=10,
    )


def test_training_progress_is_clamped() -> None:
    assert training_progress(global_step=-1, max_steps=10) == 0.0
    assert training_progress(global_step=5, max_steps=10) == 0.5
    assert training_progress(global_step=11, max_steps=10) == 1.0
    assert training_progress(global_step=5, max_steps=None) == 0.0
