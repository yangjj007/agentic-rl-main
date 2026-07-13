"""Group-level filtering for zero-signal DyME batches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping, Sequence


@dataclass(frozen=True)
class EffectiveGroupFilterConfig:
    enabled: bool = False
    after_step: int = 0
    all_wrong_keep_per_prompt: int = 1
    filter_all_correct: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "EffectiveGroupFilterConfig":
        raw = raw or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            after_step=max(0, int(raw.get("after_step", 0) or 0)),
            all_wrong_keep_per_prompt=max(0, int(raw.get("all_wrong_keep_per_prompt", 1) or 0)),
            filter_all_correct=bool(raw.get("filter_all_correct", True)),
        )


@dataclass
class EffectiveGroupFilterStats:
    total: int = 0
    kept_total: int = 0
    filtered_total: int = 0
    kept_all_wrong: int = 0
    filtered_all_wrong: int = 0
    kept_mixed: int = 0
    filtered_all_correct: int = 0


def compute_effective_group_keep_mask(
    *,
    correct_counts: Sequence[int],
    num_generations: int,
    global_step: int,
    config: EffectiveGroupFilterConfig | dict[str, Any] | None,
) -> tuple[list[bool], EffectiveGroupFilterStats]:
    cfg = (
        config
        if isinstance(config, EffectiveGroupFilterConfig)
        else EffectiveGroupFilterConfig.from_mapping(config)
    )
    n_gen = max(1, int(num_generations))
    keep = [True] * (len(correct_counts) * n_gen)
    stats = EffectiveGroupFilterStats(total=len(keep), kept_total=len(keep))
    if not cfg.enabled or int(global_step) < cfg.after_step:
        return keep, stats

    for prompt_idx, raw_count in enumerate(correct_counts):
        correct = int(raw_count)
        start = prompt_idx * n_gen
        end = start + n_gen
        if correct <= 0:
            for row in range(start, end):
                local = row - start
                row_keep = local < cfg.all_wrong_keep_per_prompt
                keep[row] = row_keep
                if row_keep:
                    stats.kept_all_wrong += 1
                else:
                    stats.filtered_all_wrong += 1
        elif correct >= n_gen and cfg.filter_all_correct:
            for row in range(start, end):
                keep[row] = False
                stats.filtered_all_correct += 1
        else:
            stats.kept_mixed += n_gen

    stats.kept_total = sum(1 for value in keep if value)
    stats.filtered_total = stats.total - stats.kept_total
    return keep, stats


def apply_effective_group_filter_to_routes(
    *,
    keep_mask: Sequence[bool],
    completion_masks: list[Any],
    advantages: list[Any],
    opsd_mask: list[bool],
    sft_replaced: list[bool],
    teacher_trajs: MutableMapping[int, Any] | None = None,
) -> int:
    """Zero loss-bearing route state for filtered rows.

    Returns the number of teacher trajectories removed.
    """
    removed_teacher_trajs = 0
    for row, keep_row in enumerate(keep_mask):
        if keep_row or row >= len(completion_masks):
            continue
        completion_masks[row] = completion_masks[row].new_zeros(completion_masks[row].shape)
        if row < len(advantages):
            advantages[row] = advantages[row].new_zeros(advantages[row].shape)
        if row < len(opsd_mask):
            opsd_mask[row] = False
        if row < len(sft_replaced):
            sft_replaced[row] = False
        if teacher_trajs is not None and row in teacher_trajs:
            teacher_trajs.pop(row, None)
            removed_teacher_trajs += 1
    return removed_teacher_trajs
