from __future__ import annotations

from pathlib import Path

import torch

from opsd_utils.hard_replay import build_mixed_group_hard_replay_plan


ROOT = Path(__file__).resolve().parents[1]


def test_hard_replay_plan_targets_mixed_wrong_with_shortest_correct_completion() -> None:
    completion_ids = torch.tensor(
        [
            [10, 11, 12, 0],
            [20, 21, 0, 0],
            [30, 31, 32, 0],
            [40, 41, 42, 0],
            [50, 51, 52, 0],
            [60, 61, 62, 0],
        ]
    )
    completion_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    acc_rewards = torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    plan = build_mixed_group_hard_replay_plan(
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        acc_rewards=acc_rewards,
        num_generations=3,
        correct_threshold=0.5,
    )

    assert set(plan.targets) == {0}
    target_ids, target_mask = plan.targets[0]
    assert target_ids.tolist() == [20, 21]
    assert target_mask.tolist() == [True, True]
    assert plan.all_wrong_indices == {3, 4, 5}
    assert plan.mixed_wrong_count == 1
    assert plan.correct_source_indices == {0: 1}


def test_trainer_uses_only_honest_hard_replay_internal_names() -> None:
    trainer_source = (ROOT / "trainer" / "DyMETrainer.py").read_text()

    assert "ssopd_plan" not in trainer_source.lower()
    assert "SSOPD mixed-group self-distill plan" not in trainer_source
