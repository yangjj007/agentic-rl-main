from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MixedGroupHardReplayPlan:
    targets: dict[int, tuple[torch.Tensor, torch.Tensor]]
    all_wrong_indices: set[int]
    mixed_wrong_count: int
    correct_source_indices: dict[int, int]


def build_mixed_group_hard_replay_plan(
    *,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    acc_rewards: torch.Tensor,
    num_generations: int,
    correct_threshold: float,
) -> MixedGroupHardReplayPlan:
    """Replay the shortest correct sequence as a hard target for mixed-wrong rows."""
    targets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    all_wrong_indices: set[int] = set()
    correct_source_indices: dict[int, int] = {}
    mixed_wrong_count = 0
    if num_generations <= 0:
        return MixedGroupHardReplayPlan(targets, all_wrong_indices, 0, correct_source_indices)

    num_prompts = int(acc_rewards.shape[0])
    for prompt_idx in range(num_prompts):
        start = prompt_idx * num_generations
        end = min(start + num_generations, int(completion_ids.shape[0]))
        if start >= end:
            continue
        rewards = acc_rewards[prompt_idx, : end - start]
        correct_offsets = [
            offset for offset, value in enumerate(rewards.tolist()) if float(value) > correct_threshold
        ]
        if not correct_offsets:
            all_wrong_indices.update(range(start, end))
            continue
        if len(correct_offsets) == end - start:
            continue

        def valid_len(offset: int) -> int:
            return int(completion_mask[start + offset].sum().item())

        source_offset = min(correct_offsets, key=lambda offset: (valid_len(offset), offset))
        source_idx = start + source_offset
        source_len = max(valid_len(source_offset), 1)
        target_ids = completion_ids[source_idx, :source_len].detach().clone()
        target_mask = completion_mask[source_idx, :source_len].detach().clone().bool()

        for offset in range(end - start):
            row_idx = start + offset
            if offset in correct_offsets:
                continue
            targets[row_idx] = (target_ids.clone(), target_mask.clone())
            correct_source_indices[row_idx] = source_idx
            mixed_wrong_count += 1

    return MixedGroupHardReplayPlan(
        targets=targets,
        all_wrong_indices=all_wrong_indices,
        mixed_wrong_count=mixed_wrong_count,
        correct_source_indices=correct_source_indices,
    )
