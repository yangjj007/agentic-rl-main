"""Signal-aware route guards for teacher-probe OPD training."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

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


@dataclass(frozen=True)
class SignalUtilityRoutingConfig:
    enabled: bool = False
    reward_std_scale: float = 0.10
    allow_grpo_on_format_bad: bool = False
    grpo_base: float = 0.05
    grpo_correct_bonus: float = 1.00
    grpo_mixed_bonus: float = 0.25
    grpo_signal_weight: float = 0.40
    grpo_readiness_weight: float = 0.70
    opd_base: float = 0.10
    opd_teacher_bonus: float = 0.45
    opd_wrong_bonus: float = 0.20
    opd_gap_weight: float = 0.55
    opd_all_wrong_bonus: float = 0.20
    opd_teacher_need_weight: float = 0.60
    opd_format_penalty: float = 1.00
    sft_base: float = 0.02
    sft_format_bad_bonus: float = 1.10
    sft_clipped_bonus: float = 0.80
    sft_all_wrong_bonus: float = 0.20
    sft_low_signal_bonus: float = 0.25
    sft_correct_penalty: float = 1.00
    skip_clipped_without_teacher: bool = False
    mode_stable_enabled: bool = False
    mode_stable_ema_beta: float = 0.80
    mode_stable_switch_margin: float = 0.20
    mode_stable_min_hold_steps: int = 2


@dataclass(frozen=True)
class RouteUtilities:
    grpo: float
    opd: float
    sft: float
    selected: int
    margin: float


@dataclass(frozen=True)
class ModeStableRouteState:
    previous_mode: int | None = None
    utility_ema_grpo: float = 0.0
    utility_ema_opd: float = 0.0
    utility_ema_sft: float = 0.0
    hold_steps: int = 0
    last_step: int = -1
    switch_count: int = 0
    blocked_switch_count: int = 0


@dataclass
class SignalUtilityRoutingStats:
    routed_grpo: int = 0
    routed_opd: int = 0
    routed_sft: int = 0
    routed_skip: int = 0
    rerouted_grpo: int = 0
    rerouted_opd: int = 0
    rerouted_sft: int = 0
    teacher_traj_removed: int = 0
    utilities: list[RouteUtilities] = field(default_factory=list)
    updated_stable_states: dict[str, ModeStableRouteState] = field(default_factory=dict)
    switches: int = 0
    blocked_switches: int = 0
    stable_holds: int = 0
    invalid_current_switches: int = 0
    switch_gains: list[float] = field(default_factory=list)

    @property
    def candidate_count(self) -> int:
        return len(self.utilities)

    @property
    def grpo_mean(self) -> float:
        return _mean([u.grpo for u in self.utilities])

    @property
    def opd_mean(self) -> float:
        return _mean([u.opd for u in self.utilities])

    @property
    def sft_mean(self) -> float:
        return _mean([u.sft for u in self.utilities])

    @property
    def margin_mean(self) -> float:
        return _mean([u.margin for u in self.utilities])

    @property
    def switch_gain_mean(self) -> float:
        return _mean(self.switch_gains)

    @property
    def ema_grpo_mean(self) -> float:
        return _mean([s.utility_ema_grpo for s in self.updated_stable_states.values()])

    @property
    def ema_opd_mean(self) -> float:
        return _mean([s.utility_ema_opd for s in self.updated_stable_states.values()])

    @property
    def ema_sft_mean(self) -> float:
        return _mean([s.utility_ema_sft for s in self.updated_stable_states.values()])


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


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _as_signal_utility_config(
    raw: SignalUtilityRoutingConfig | dict | None,
) -> SignalUtilityRoutingConfig:
    if isinstance(raw, SignalUtilityRoutingConfig):
        return raw
    raw = raw or {}
    return SignalUtilityRoutingConfig(
        enabled=bool(raw.get("enabled", raw.get("signal_utility_routing", False))),
        reward_std_scale=max(1e-8, float(raw.get("reward_std_scale", 0.10) or 0.10)),
        allow_grpo_on_format_bad=bool(raw.get("allow_grpo_on_format_bad", False)),
        grpo_base=float(raw.get("grpo_base", 0.05)),
        grpo_correct_bonus=float(raw.get("grpo_correct_bonus", 1.00)),
        grpo_mixed_bonus=float(raw.get("grpo_mixed_bonus", 0.25)),
        grpo_signal_weight=float(raw.get("grpo_signal_weight", 0.40)),
        grpo_readiness_weight=float(raw.get("grpo_readiness_weight", 0.70)),
        opd_base=float(raw.get("opd_base", 0.10)),
        opd_teacher_bonus=float(raw.get("opd_teacher_bonus", 0.45)),
        opd_wrong_bonus=float(raw.get("opd_wrong_bonus", 0.20)),
        opd_gap_weight=float(raw.get("opd_gap_weight", 0.55)),
        opd_all_wrong_bonus=float(raw.get("opd_all_wrong_bonus", 0.20)),
        opd_teacher_need_weight=float(raw.get("opd_teacher_need_weight", 0.60)),
        opd_format_penalty=float(raw.get("opd_format_penalty", 1.00)),
        sft_base=float(raw.get("sft_base", 0.02)),
        sft_format_bad_bonus=float(raw.get("sft_format_bad_bonus", 1.10)),
        sft_clipped_bonus=float(raw.get("sft_clipped_bonus", 0.80)),
        sft_all_wrong_bonus=float(raw.get("sft_all_wrong_bonus", 0.20)),
        sft_low_signal_bonus=float(raw.get("sft_low_signal_bonus", 0.25)),
        sft_correct_penalty=float(raw.get("sft_correct_penalty", 1.00)),
        skip_clipped_without_teacher=bool(raw.get("skip_clipped_without_teacher", False)),
        mode_stable_enabled=bool(raw.get("mode_stable_enabled", False)),
        mode_stable_ema_beta=_clamp01(raw.get("mode_stable_ema_beta", 0.80)),
        mode_stable_switch_margin=max(
            0.0, float(raw.get("mode_stable_switch_margin", 0.20) or 0.0)
        ),
        mode_stable_min_hold_steps=max(
            0, int(raw.get("mode_stable_min_hold_steps", 2) or 0)
        ),
    )


def _mode_score(mode: int | None, grpo: float, opd: float, sft: float) -> float:
    if mode == MODE_GRPO:
        return float(grpo)
    if mode == MODE_OPSD:
        return float(opd)
    if mode == MODE_SFT:
        return float(sft)
    return float("-inf")


def _mode_valid(mode: int | None, grpo: float, opd: float, sft: float, invalid: float) -> bool:
    return _mode_score(mode, grpo, opd, sft) > invalid * 0.5


def apply_signal_utility_routing(
    *,
    completion_modes: Sequence[int],
    teacher_traj_indices: Iterable[int],
    student_correct: Sequence[bool],
    group_has_correct: Sequence[bool],
    group_reward_std: Sequence[float],
    qualities: Sequence[CompletionQuality],
    num_generations: int,
    readiness: float,
    config: SignalUtilityRoutingConfig | dict | None,
    teacher_correct_indices: Iterable[int] | None = None,
    state_keys: Sequence[str] | None = None,
    mode_stable_states: Mapping[str, ModeStableRouteState] | None = None,
    global_step: int = 0,
) -> tuple[list[int], set[int], SignalUtilityRoutingStats]:
    """Choose GRPO/OPD/SFT from signal-derived utility scores.

    OPD is treated as useful only for recoverable student failures: the student
    is wrong and the teacher probe is correct. Its utility decays with student
    readiness instead of relying on a fixed late-training cap.
    """
    cfg = _as_signal_utility_config(config)
    modes = list(completion_modes)
    kept_trajs = set(int(i) for i in teacher_traj_indices)
    teacher_correct = (
        set(int(i) for i in teacher_correct_indices)
        if teacher_correct_indices is not None
        else set(kept_trajs)
    )
    stats = SignalUtilityRoutingStats()
    if not cfg.enabled:
        return modes, kept_trajs, stats

    n_gen = max(int(num_generations), 1)
    readiness_i = _clamp01(readiness)
    teacher_need = 1.0 - readiness_i
    invalid = -1e9

    for idx, old_mode in enumerate(modes):
        if old_mode == MODE_SKIP:
            stats.routed_skip += 1
            continue

        prompt_idx = idx // n_gen
        correct_i = bool(student_correct[idx]) if idx < len(student_correct) else False
        group_correct_i = (
            bool(group_has_correct[prompt_idx])
            if prompt_idx < len(group_has_correct)
            else False
        )
        reward_std_i = (
            float(group_reward_std[prompt_idx])
            if prompt_idx < len(group_reward_std)
            else 0.0
        )
        signal_i = _clamp01(reward_std_i / cfg.reward_std_scale)
        quality_i = qualities[idx] if idx < len(qualities) else CompletionQuality()
        format_bad_i = bool(quality_i.format_bad)
        clipped_i = bool(quality_i.clipped)
        teacher_correct_i = idx in teacher_correct
        wrong_i = not correct_i

        raw_grpo = (
            cfg.grpo_base
            + cfg.grpo_correct_bonus * float(correct_i)
            + cfg.grpo_mixed_bonus * float(group_correct_i)
            + cfg.grpo_signal_weight * signal_i
            + cfg.grpo_readiness_weight * readiness_i
        )
        opd_recoverability = (
            cfg.opd_teacher_bonus * float(teacher_correct_i)
            + cfg.opd_wrong_bonus * float(wrong_i)
            + cfg.opd_gap_weight * float(wrong_i and teacher_correct_i)
            + cfg.opd_all_wrong_bonus * float(not group_correct_i)
            + cfg.opd_teacher_need_weight
        )
        raw_opd = cfg.opd_base + teacher_need * opd_recoverability
        raw_opd -= cfg.opd_format_penalty * float(format_bad_i or clipped_i)
        raw_sft = (
            cfg.sft_base
            + cfg.sft_format_bad_bonus * float(format_bad_i)
            + cfg.sft_clipped_bonus * float(clipped_i)
            + cfg.sft_all_wrong_bonus * float(not group_correct_i)
            + cfg.sft_low_signal_bonus * (1.0 - signal_i)
            - cfg.sft_correct_penalty * float(correct_i)
        )

        grpo = raw_grpo
        opd = raw_opd
        sft = raw_sft
        if (format_bad_i or clipped_i) and not cfg.allow_grpo_on_format_bad:
            grpo = invalid
        if correct_i or not teacher_correct_i:
            opd = invalid

        force_skip = bool(
            cfg.skip_clipped_without_teacher
            and clipped_i
            and not teacher_correct_i
        )
        scored = [(MODE_GRPO, grpo), (MODE_OPSD, opd), (MODE_SFT, sft)]
        scored.sort(key=lambda item: item[1], reverse=True)
        new_mode = MODE_SKIP if force_skip else scored[0][0]
        valid_scores = [score for _, score in scored if score > invalid * 0.5]
        margin = (
            float(valid_scores[0] - valid_scores[1])
            if len(valid_scores) >= 2
            else 0.0
        )

        if cfg.mode_stable_enabled and not force_skip:
            state_key = (
                str(state_keys[idx])
                if state_keys is not None and idx < len(state_keys)
                else str(idx)
            )
            previous_state = (
                mode_stable_states.get(state_key)
                if mode_stable_states is not None
                else None
            )
            beta = float(cfg.mode_stable_ema_beta)
            if previous_state is None:
                ema_grpo = float(raw_grpo)
                ema_opd = float(raw_opd)
                ema_sft = float(raw_sft)
                current_mode = None
                hold_steps = 0
                switch_count = 0
                blocked_count = 0
            else:
                ema_grpo = beta * previous_state.utility_ema_grpo + (1.0 - beta) * raw_grpo
                ema_opd = beta * previous_state.utility_ema_opd + (1.0 - beta) * raw_opd
                ema_sft = beta * previous_state.utility_ema_sft + (1.0 - beta) * raw_sft
                current_mode = previous_state.previous_mode
                hold_steps = int(previous_state.hold_steps)
                switch_count = int(previous_state.switch_count)
                blocked_count = int(previous_state.blocked_switch_count)

            stable_scores = [
                (MODE_GRPO, ema_grpo if _mode_valid(MODE_GRPO, grpo, opd, sft, invalid) else invalid),
                (MODE_OPSD, ema_opd if _mode_valid(MODE_OPSD, grpo, opd, sft, invalid) else invalid),
                (MODE_SFT, ema_sft if _mode_valid(MODE_SFT, grpo, opd, sft, invalid) else invalid),
            ]
            stable_scores.sort(key=lambda item: item[1], reverse=True)
            best_mode = stable_scores[0][0]
            if current_mode is None:
                new_mode = best_mode
                next_hold_steps = 1
            elif not _mode_valid(current_mode, grpo, opd, sft, invalid):
                new_mode = best_mode
                next_hold_steps = 1 if new_mode != current_mode else hold_steps + 1
                if new_mode != current_mode:
                    stats.invalid_current_switches += 1
                    stats.switches += 1
                    switch_count += 1
            else:
                switch_gain = _mode_score(best_mode, ema_grpo, ema_opd, ema_sft) - _mode_score(
                    current_mode, ema_grpo, ema_opd, ema_sft
                )
                stats.switch_gains.append(float(max(0.0, switch_gain)))
                should_hold = (
                    best_mode == current_mode
                    or hold_steps < cfg.mode_stable_min_hold_steps
                    or switch_gain < cfg.mode_stable_switch_margin
                )
                if should_hold:
                    new_mode = int(current_mode)
                    next_hold_steps = hold_steps + 1
                    stats.stable_holds += 1
                    if best_mode != current_mode:
                        stats.blocked_switches += 1
                        blocked_count += 1
                else:
                    new_mode = best_mode
                    next_hold_steps = 1
                    stats.switches += 1
                    switch_count += 1

            stats.updated_stable_states[state_key] = ModeStableRouteState(
                previous_mode=int(new_mode),
                utility_ema_grpo=float(ema_grpo),
                utility_ema_opd=float(ema_opd),
                utility_ema_sft=float(ema_sft),
                hold_steps=int(next_hold_steps),
                last_step=int(global_step),
                switch_count=int(switch_count),
                blocked_switch_count=int(blocked_count),
            )

        modes[idx] = new_mode
        stats.utilities.append(
            RouteUtilities(
                grpo=float(raw_grpo),
                opd=float(raw_opd),
                sft=float(raw_sft),
                selected=int(new_mode),
                margin=margin,
            )
        )
        if new_mode == MODE_GRPO:
            stats.routed_grpo += 1
            stats.rerouted_grpo += int(old_mode != MODE_GRPO)
        elif new_mode == MODE_OPSD:
            stats.routed_opd += 1
            stats.rerouted_opd += int(old_mode != MODE_OPSD)
        elif new_mode == MODE_SFT:
            stats.routed_sft += 1
            stats.rerouted_sft += int(old_mode != MODE_SFT)
        elif new_mode == MODE_SKIP:
            stats.routed_skip += 1

        if new_mode != MODE_OPSD and idx in kept_trajs:
            kept_trajs.discard(idx)
            stats.teacher_traj_removed += 1

    return modes, kept_trajs, stats


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
