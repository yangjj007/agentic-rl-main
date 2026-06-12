"""RLSD / COPSD anti-leakage routing tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from opsd_utils.constants import MODE_GRPO, MODE_OPSD, MODE_SFT
from opsd_utils.mode_router import route_completion_modes, route_prompt_modes


def _rlsd_cfg(**gate):
    base = {
        "enabled": True,
        "mode": "rlsd",
        "gate": {
            "correct_threshold": 0.5,
            "per_completion_opsd": True,
            "require_format_for_opsd": False,
            **gate,
        },
    }
    return base


def test_rlsd_prompt_correct_grpo():
    acc = torch.tensor([[1.0, 0.0]])
    modes = route_prompt_modes(acc, 2, _rlsd_cfg(), recoverable_flags=[True])
    assert modes == [MODE_GRPO]


def test_rlsd_prompt_wrong_opsd_when_recoverable():
    acc = torch.tensor([[0.0, 0.0]])
    modes = route_prompt_modes(acc, 2, _rlsd_cfg(), recoverable_flags=[True])
    assert modes == [MODE_OPSD]


def test_rlsd_prompt_wrong_sft_when_not_recoverable():
    acc = torch.tensor([[0.0, 0.0]])
    modes = route_prompt_modes(acc, 2, _rlsd_cfg(), recoverable_flags=[False])
    assert modes == [MODE_SFT]


def test_rlsd_per_completion_routing():
    acc = torch.tensor([[1.0, 0.0]])
    fmt = torch.tensor([[1.0, 0.5]])
    modes = route_completion_modes(acc, 2, 2, _rlsd_cfg(), [True], format_rewards=fmt)
    assert modes == [MODE_GRPO, MODE_OPSD]


def test_copsd_opd_alias_matches_rlsd():
    acc = torch.tensor([[0.0, 1.0]])
    cfg = _rlsd_cfg()
    cfg["mode"] = "copsd_opd"
    modes = route_completion_modes(acc, 2, 2, cfg, [True])
    assert modes == [MODE_OPSD, MODE_GRPO]


if __name__ == "__main__":
    test_rlsd_prompt_correct_grpo()
    test_rlsd_prompt_wrong_opsd_when_recoverable()
    test_rlsd_prompt_wrong_sft_when_not_recoverable()
    test_rlsd_per_completion_routing()
    test_copsd_opd_alias_matches_rlsd()
    print("RLSD routing tests passed.")
