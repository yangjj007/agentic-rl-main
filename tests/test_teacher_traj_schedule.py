from __future__ import annotations

from opsd_utils.teacher_traj_schedule import (
    effective_linear_weight,
    effective_teacher_traj_exposure_weight,
    effective_teacher_traj_weight,
)


def test_effective_linear_weight_decays_to_final_weight() -> None:
    assert effective_linear_weight(
        base_weight=1.5,
        global_step=441,
        decay_enabled=True,
        decay_start_step=294,
        decay_end_step=441,
        final_weight=0.5,
    ) == 0.5


def test_teacher_traj_weight_decay_disabled_returns_base_weight() -> None:
    assert effective_teacher_traj_weight(
        base_weight=0.5,
        global_step=400,
        decay_enabled=False,
        decay_start_step=294,
        decay_end_step=441,
        final_weight=0.0,
    ) == 0.5


def test_teacher_traj_weight_decay_keeps_base_before_start() -> None:
    assert effective_teacher_traj_weight(
        base_weight=0.5,
        global_step=293,
        decay_enabled=True,
        decay_start_step=294,
        decay_end_step=441,
        final_weight=0.0,
    ) == 0.5


def test_teacher_traj_weight_decay_reaches_final_after_end() -> None:
    assert effective_teacher_traj_weight(
        base_weight=0.5,
        global_step=441,
        decay_enabled=True,
        decay_start_step=294,
        decay_end_step=441,
        final_weight=0.0,
    ) == 0.0


def test_teacher_traj_weight_decay_interpolates_linearly() -> None:
    assert effective_teacher_traj_weight(
        base_weight=0.5,
        global_step=50,
        decay_enabled=True,
        decay_start_step=0,
        decay_end_step=100,
        final_weight=0.1,
    ) == 0.3


def test_teacher_traj_weight_decay_uses_normalized_progress() -> None:
    assert effective_teacher_traj_weight(
        base_weight=0.5,
        global_step=3,
        decay_enabled=True,
        decay_start_step=294,
        decay_end_step=441,
        final_weight=0.0,
        schedule_mode="progress",
        max_steps=8,
        decay_start_progress=0.25,
        decay_end_progress=0.50,
    ) == 0.25


def test_teacher_traj_exposure_weight_is_zero_when_disabled() -> None:
    assert effective_teacher_traj_exposure_weight(
        scheduled_weight=0.5,
        enabled=False,
        global_traj_count=8,
    ) == 0.0


def test_teacher_traj_exposure_weight_is_zero_without_global_trajectories() -> None:
    assert effective_teacher_traj_exposure_weight(
        scheduled_weight=0.5,
        enabled=True,
        global_traj_count=0,
    ) == 0.0


def test_teacher_traj_exposure_weight_uses_schedule_when_globally_active() -> None:
    assert effective_teacher_traj_exposure_weight(
        scheduled_weight=0.5,
        enabled=True,
        global_traj_count=8,
    ) == 0.5
