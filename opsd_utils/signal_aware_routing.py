"""Signal-aware route guards for teacher-probe OPD training."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from opsd_utils.constants import MODE_GRPO, MODE_OPSD, MODE_SFT, MODE_SKIP
from opsd_utils.phase_schedule import boundary_reached


@dataclass(frozen=True)
class CompletionQuality:
    """Cheap per-completion quality flags used by routing overrides."""

    degenerate: bool = False
    clipped: bool = False
    force_sft: bool = False
    table_spam: bool = False

    @property
    def format_bad(self) -> bool:
        return self.degenerate or self.force_sft or self.table_spam


@dataclass(frozen=True)
class SignalAwareRoutingConfig:
    signal_aware: bool = False
    reward_std_min: float = 0.05
    degenerate_hard_override: bool = False
    clipped_hard_override: bool = False

    @property
    def enabled(self) -> bool:
        return self.signal_aware or self.degenerate_hard_override or self.clipped_hard_override


@dataclass
class SignalAwareRoutingStats:
    degenerate_hard_overrides: int = 0
    clipped_hard_overrides: int = 0
    teacher_correct_overrides: int = 0
    signal_aware_sft: int = 0

    @property
    def total_overrides(self) -> int:
        return (
            self.degenerate_hard_overrides
            + self.clipped_hard_overrides
            + self.signal_aware_sft
        )


@dataclass(frozen=True)
class OpdRouteCapConfig:
    enabled: bool = False
    max_per_prompt: int = 0
    after_step: int = 0
    overflow_route: str = "sft"
    schedule_mode: str = "step"
    start_progress: float = 0.5


@dataclass
class OpdRouteCapStats:
    capped: int = 0
    eligible_prompts: int = 0
    kept_opd: int = 0
    teacher_traj_removed: int = 0
    rerouted_grpo: int = 0
    skipped: int = 0


def _as_config(raw: SignalAwareRoutingConfig | dict | None) -> SignalAwareRoutingConfig:
    if isinstance(raw, SignalAwareRoutingConfig):
        return raw
    raw = raw or {}
    return SignalAwareRoutingConfig(
        signal_aware=bool(raw.get("signal_aware_routing", raw.get("signal_aware", False))),
        reward_std_min=float(raw.get("signal_reward_std_min", raw.get("reward_std_min", 0.05))),
        degenerate_hard_override=bool(raw.get("degenerate_hard_override", False)),
        clipped_hard_override=bool(raw.get("clipped_hard_override", False)),
    )


def _as_opd_cap_config(raw: OpdRouteCapConfig | dict | None) -> OpdRouteCapConfig:
    if isinstance(raw, OpdRouteCapConfig):
        return raw
    raw = raw or {}
    return OpdRouteCapConfig(
        enabled=bool(raw.get("enabled", raw.get("opd_route_cap", False))),
        max_per_prompt=max(0, int(raw.get("max_per_prompt", 0) or 0)),
        after_step=max(0, int(raw.get("after_step", 0) or 0)),
        overflow_route=str(raw.get("overflow_route", "sft") or "sft").lower(),
        schedule_mode=str(raw.get("schedule_mode", "step") or "step").lower(),
        start_progress=float(raw.get("start_progress", 0.5) or 0.0),
    )


def apply_opd_route_cap(
    *,
    completion_modes: Sequence[int],
    teacher_traj_indices: Iterable[int],
    group_has_correct: Sequence[bool] | None = None,
    num_generations: int,
    global_step: int,
    max_steps: int | None = None,
    config: OpdRouteCapConfig | dict | None,
) -> tuple[list[int], set[int], OpdRouteCapStats]:
    """Limit post-probe OPD completions per prompt after a configured step.

    This is deliberately applied after teacher-probe/repair routing: the teacher
    can still identify recoverable wrong completions, but late training can stop
    OPD from dominating every wrong slot in a prompt.
    """
    cfg = _as_opd_cap_config(config)
    modes = list(completion_modes)
    kept_trajs = set(int(i) for i in teacher_traj_indices)
    stats = OpdRouteCapStats()
    if (
        not cfg.enabled
        or cfg.max_per_prompt <= 0
        or not boundary_reached(
            global_step,
            max_steps,
            mode=cfg.schedule_mode,
            step_boundary=cfg.after_step,
            progress_boundary=cfg.start_progress,
        )
        or cfg.overflow_route not in {"sft", "mixed_grpo_all_wrong_skip"}
    ):
        return modes, kept_trajs, stats

    grouped: dict[int, list[int]] = {}
    for idx, mode in enumerate(modes):
        if mode != MODE_OPSD:
            continue
        prompt_idx = idx // max(int(num_generations), 1)
        grouped.setdefault(prompt_idx, []).append(idx)

    for prompt_idx, indices in grouped.items():
        if len(indices) <= cfg.max_per_prompt:
            stats.kept_opd += len(indices)
            continue
        stats.eligible_prompts += 1
        keep = set(indices[: cfg.max_per_prompt])
        stats.kept_opd += len(keep)
        for idx in indices:
            if idx in keep:
                continue
            if cfg.overflow_route == "mixed_grpo_all_wrong_skip":
                has_correct = bool(group_has_correct[prompt_idx]) if group_has_correct is not None else False
                if has_correct:
                    modes[idx] = MODE_GRPO
                    stats.rerouted_grpo += 1
                else:
                    modes[idx] = MODE_SKIP
                    stats.skipped += 1
            else:
                modes[idx] = MODE_SFT
            stats.capped += 1
            if idx in kept_trajs:
                kept_trajs.discard(idx)
                stats.teacher_traj_removed += 1

    return modes, kept_trajs, stats


def is_table_spam_completion(text: str) -> bool:
    """Detect obvious table/DePlot transcription spam without using it as reward."""
    s = text or ""
    pipe_lines = [line for line in s.splitlines() if line.count("|") >= 2]
    if len(pipe_lines) >= 3:
        return True
    if len(re.findall(r"\b\d{4}\s*\|", s)) >= 3:
        return True
    if len(re.findall(r"\b[A-Za-z][A-Za-z ]{0,24}\s*\|\s*[-+]?\d", s)) >= 4:
        return True
    return False


def apply_signal_aware_routing(
    *,
    completion_modes: Sequence[int],
    teacher_traj_indices: Iterable[int],
    qualities: Sequence[CompletionQuality],
    group_reward_std: Sequence[float],
    num_generations: int,
    config: SignalAwareRoutingConfig | dict | None,
) -> tuple[list[int], set[int], SignalAwareRoutingStats]:
    """Apply route guards after teacher-probe routing, before batch assembly.

    The function intentionally does not alter all-wrong probe candidate creation.
    It only downgrades unsafe post-probe routes to SFT and removes corresponding
    teacher trajectories.
    """
    cfg = _as_config(config)
    modes = list(completion_modes)
    kept_trajs = set(int(i) for i in teacher_traj_indices)
    stats = SignalAwareRoutingStats()
    if not cfg.enabled:
        return modes, kept_trajs, stats

    for i, mode in enumerate(list(modes)):
        q = qualities[i] if i < len(qualities) else CompletionQuality()
        prompt_idx = i // max(int(num_generations), 1)
        reward_std = (
            float(group_reward_std[prompt_idx])
            if prompt_idx < len(group_reward_std)
            else cfg.reward_std_min
        )
        teacher_correct = i in kept_trajs

        reason: str | None = None
        if (cfg.degenerate_hard_override or cfg.signal_aware) and q.format_bad:
            reason = "degenerate"
        elif (cfg.clipped_hard_override or cfg.signal_aware) and q.clipped:
            reason = "clipped"
        elif cfg.signal_aware and mode == MODE_GRPO and reward_std < cfg.reward_std_min:
            reason = "low_signal"

        if reason is None:
            continue

        modes[i] = MODE_SFT
        if teacher_correct:
            kept_trajs.discard(i)
            stats.teacher_correct_overrides += 1
        if reason == "clipped":
            stats.clipped_hard_overrides += 1
        elif reason == "low_signal":
            stats.signal_aware_sft += 1
        else:
            stats.degenerate_hard_overrides += 1

    return modes, kept_trajs, stats


def local_teacher_traj_indices(
    *,
    teacher_traj_mask: Sequence[bool],
    has_teacher_prompt_ids: bool,
) -> list[int]:
    """Return local teacher-trajectory rows only when teacher prompts exist.

    OPD prompts are built from local OPD indices. A rank may have teacher
    trajectory rows while route guards or routing leave it with no local OPD
    prompt tensors. In that case the rank must still enter distributed
    collectives, but it contributes no local trajectory loss.
    """
    if not has_teacher_prompt_ids:
        return []
    return [idx for idx, enabled in enumerate(teacher_traj_mask) if bool(enabled)]
def teacher_probe_failure_mode(*, group_has_correct: bool, route: str) -> int:
    policy = str(route or "sft").lower()
    if policy == "sft":
        return MODE_SFT
    if policy == "mixed_grpo_all_wrong_skip":
        return MODE_GRPO if group_has_correct else MODE_SKIP
    raise ValueError(f"unknown teacher probe failure route: {route}")

