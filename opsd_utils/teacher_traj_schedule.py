from __future__ import annotations

from typing import Optional

from opsd_utils.phase_schedule import training_progress


def effective_linear_weight(
    *,
    base_weight: float,
    global_step: int,
    decay_enabled: bool,
    decay_start_step: int,
    decay_end_step: int,
    final_weight: float,
    max_steps: Optional[int] = None,
    schedule_mode: str = "step",
    decay_start_progress: float = 0.0,
    decay_end_progress: float = 1.0,
) -> float:
    """Linearly decay a loss weight after a configured step."""
    base = float(base_weight)
    if not decay_enabled:
        return base
    use_progress = str(schedule_mode or "step").lower() == "progress" and int(max_steps or 0) > 0
    if use_progress:
        start = float(decay_start_progress)
        end = float(decay_end_progress)
        step = training_progress(global_step, max_steps)
    else:
        start = int(decay_start_step)
        end = int(decay_end_step)
        step = int(global_step)
    final = float(final_weight)
    if end <= start:
        return final if step >= start else base
    if step < start:
        return base
    if step >= end:
        return final
    progress = (step - start) / float(end - start)
    return base + (final - base) * progress


def effective_teacher_traj_weight(
    *,
    base_weight: float,
    global_step: int,
    decay_enabled: bool,
    decay_start_step: int,
    decay_end_step: int,
    final_weight: float,
    max_steps: Optional[int] = None,
    schedule_mode: str = "step",
    decay_start_progress: float = 0.0,
    decay_end_progress: float = 1.0,
) -> float:
    """Linearly decay teacher-trajectory FKL weight after a configured step."""
    return effective_linear_weight(
        base_weight=base_weight,
        global_step=global_step,
        decay_enabled=decay_enabled,
        decay_start_step=decay_start_step,
        decay_end_step=decay_end_step,
        final_weight=final_weight,
        max_steps=max_steps,
        schedule_mode=schedule_mode,
        decay_start_progress=decay_start_progress,
        decay_end_progress=decay_end_progress,
    )
