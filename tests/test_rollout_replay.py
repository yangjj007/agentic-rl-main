from __future__ import annotations

import torch


def test_rollout_replay_adds_positive_correct_rows_and_skips_weak_rows() -> None:
    from opsd_utils.rollout_replay import RolloutReplayBuffer, RolloutReplayConfig

    buffer = RolloutReplayBuffer(
        RolloutReplayConfig(enabled=True, capacity=8, batch_size=2, after_step=0, min_abs_advantage=0.2)
    )
    prompt_ids = torch.tensor([[1, 2], [1, 3], [1, 4]])
    prompt_mask = torch.ones_like(prompt_ids)
    completion_ids = torch.tensor([[10, 11], [20, 21], [30, 31]])
    completion_mask = torch.ones_like(completion_ids)
    old_logps = torch.zeros_like(completion_ids, dtype=torch.float)
    advantages = torch.tensor([[0.6], [-0.8], [0.1]])
    acc_rewards = torch.tensor([1.0, 0.0, 1.0])

    stats = buffer.add_batch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        old_per_token_logps=old_logps,
        advantages=advantages,
        acc_rewards=acc_rewards,
        global_step=5,
    )

    assert stats.added == 1
    assert stats.skipped_not_positive == 1
    assert stats.skipped_low_advantage == 1
    assert len(buffer) == 1
    sample = buffer.sample(global_step=6)
    assert len(sample) == 1
    assert sample[0].advantage == torch.tensor(0.6).item()
    assert torch.equal(sample[0].completion_ids, torch.tensor([10, 11]))


def test_rollout_replay_uses_priority_and_age_eviction() -> None:
    from opsd_utils.rollout_replay import RolloutReplayBuffer, RolloutReplayConfig, RolloutReplayEntry

    buffer = RolloutReplayBuffer(
        RolloutReplayConfig(enabled=True, capacity=3, batch_size=3, after_step=0, priority_alpha=1.0, max_age_steps=3, seed=3)
    )
    for idx, adv in enumerate([0.2, 1.0, 0.5, 0.7]):
        buffer.add_entry(
            RolloutReplayEntry(
                prompt_ids=torch.tensor([1, idx]),
                prompt_mask=torch.ones(2, dtype=torch.long),
                completion_ids=torch.tensor([idx + 10]),
                completion_mask=torch.ones(1, dtype=torch.long),
                old_per_token_logps=torch.zeros(1),
                advantage=adv,
                acc_reward=1.0,
                global_step=idx,
            )
        )

    assert len(buffer) == 3
    assert [entry.global_step for entry in buffer.entries] == [1, 2, 3]

    sample = buffer.sample(global_step=4)
    assert len(sample) == 3
    assert all(4 - entry.global_step <= 3 for entry in sample)
    assert any(entry.advantage == 1.0 for entry in sample)


def test_rollout_replay_disabled_without_capacity_or_weight() -> None:
    from opsd_utils.rollout_replay import RolloutReplayBuffer, RolloutReplayConfig

    assert not RolloutReplayBuffer(RolloutReplayConfig(enabled=True, capacity=0)).available
    assert not RolloutReplayBuffer(RolloutReplayConfig(enabled=True, weight=0.0)).available


def test_rollout_replay_optional_tensor_stack_rejects_incompatible_vision_shapes() -> None:
    from opsd_utils.rollout_replay import stack_optional_compatible_tensors

    rows = [
        torch.zeros(7, 3, 384, 384),
        torch.zeros(5, 3, 384, 384),
    ]

    assert stack_optional_compatible_tensors(rows, device=torch.device("cpu")) is None
