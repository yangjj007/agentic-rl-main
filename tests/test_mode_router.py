"""Routing parity: mode=dyme must match original DyME binary SFT/GRPO logic."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from opsd_utils.constants import MODE_GRPO, MODE_OPSD, MODE_SFT
from opsd_utils.mode_router import (
    route_prompt_modes,
    expand_modes_to_completions,
    route_completion_modes,
)


def test_dyme_mode_matches_binary_routing():
    acc = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cfg = {"enabled": True, "mode": "dyme", "gate": {"correct_threshold": 0.5}}
    modes = route_prompt_modes(acc, num_generations=2, opsd_config=cfg, recoverable_flags=[True, True, True])
    assert modes == [MODE_SFT, MODE_GRPO, MODE_GRPO]


def test_trimode_routes_sft_when_all_wrong():
    acc = torch.tensor([[0.0, 0.0]])
    cfg = {"enabled": True, "mode": "trimode", "gate": {"correct_threshold": 0.5}}
    modes = route_prompt_modes(acc, 2, cfg, recoverable_flags=[True])
    assert modes == [MODE_SFT]


def test_trimode_opsd_when_any_correct():
    acc = torch.tensor([[1.0, 0.0]])
    cfg = {"enabled": True, "mode": "trimode", "gate": {"correct_threshold": 0.5}}
    modes = route_prompt_modes(acc, 2, cfg, recoverable_flags=[False])
    assert modes == [MODE_OPSD]


def test_trimode_falls_back_to_sft():
    acc = torch.tensor([[0.0, 0.0]])
    cfg = {"enabled": True, "mode": "trimode", "gate": {"correct_threshold": 0.5}}
    modes = route_prompt_modes(acc, 2, cfg, recoverable_flags=[False])
    assert modes == [MODE_SFT]


def test_expand_modes_to_completions():
    modes = expand_modes_to_completions([MODE_OPSD, MODE_GRPO], num_generations=2, batch_size=4)
    assert modes == [MODE_OPSD, MODE_OPSD, MODE_GRPO, MODE_GRPO]


def test_trimode_per_completion_opsd_only_correct_and_formatted():
    acc = torch.tensor([[1.0, 0.0]])
    fmt = torch.tensor([[1.0, 1.0]])
    cfg = {
        "enabled": True,
        "mode": "trimode",
        "gate": {
            "correct_threshold": 0.5,
            "per_completion_opsd": True,
            "require_format_for_opsd": True,
        },
    }
    modes = route_completion_modes(acc, 2, 2, cfg, [True], format_rewards=fmt)
    assert modes == [MODE_OPSD, MODE_SFT]


def test_trimode_per_completion_skips_wrong_format_even_if_correct():
    acc = torch.tensor([[1.0, 1.0]])
    fmt = torch.tensor([[0.0, 1.0]])
    cfg = {
        "enabled": True,
        "mode": "trimode",
        "gate": {
            "correct_threshold": 0.5,
            "per_completion_opsd": True,
            "require_format_for_opsd": True,
        },
    }
    modes = route_completion_modes(acc, 2, 2, cfg, [True], format_rewards=fmt)
    assert modes == [MODE_SFT, MODE_OPSD]


if __name__ == "__main__":
    test_dyme_mode_matches_binary_routing()
    test_trimode_routes_sft_when_all_wrong()
    test_trimode_opsd_when_any_correct()
    test_trimode_falls_back_to_sft()
    test_expand_modes_to_completions()
    test_trimode_per_completion_opsd_only_correct_and_formatted()
    test_trimode_per_completion_skips_wrong_format_even_if_correct()
    print("All routing tests passed.")
