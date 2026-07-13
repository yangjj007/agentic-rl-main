from __future__ import annotations

import math
from typing import Optional


def training_progress(global_step: int, max_steps: Optional[int]) -> float:
    """Return optimizer progress in [0, 1], or 0 when the horizon is unknown."""
    total = int(max_steps or 0)
    if total <= 0:
        return 0.0
    return max(0.0, min(float(global_step) / float(total), 1.0))


def resolve_schedule_step(
    *,
    mode: str,
    absolute_step: int,
    progress: float,
    max_steps: Optional[int],
) -> int:
    """Resolve a phase boundary to an optimizer step.

    Progress boundaries use ``ceil`` so a phase never activates before the
    configured fraction of the training horizon.
    """
    if str(mode or "step").lower() != "progress" or int(max_steps or 0) <= 0:
        return max(0, int(absolute_step))
    bounded_progress = max(0.0, min(float(progress), 1.0))
    return int(math.ceil(bounded_progress * int(max_steps)))


def schedule_active(
    *,
    global_step: int,
    mode: str,
    absolute_step: int,
    progress: float,
    max_steps: Optional[int],
) -> bool:
    """Return whether a resolved phase boundary has been reached."""
    return int(global_step) >= resolve_schedule_step(
        mode=mode,
        absolute_step=absolute_step,
        progress=progress,
        max_steps=max_steps,
    )


def boundary_reached(
    global_step: int,
    max_steps: Optional[int],
    *,
    mode: str,
    step_boundary: int,
    progress_boundary: float,
) -> bool:
    """Evaluate a legacy step or normalized-progress phase boundary."""
    return schedule_active(
        global_step=global_step,
        mode=mode,
        absolute_step=step_boundary,
        progress=progress_boundary,
        max_steps=max_steps,
    )
