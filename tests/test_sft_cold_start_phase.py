"""Tests for embedded SFT cold-start gate and degenerate probe splits."""
import re

from opsd_utils.diagnostics import summarize_generate_probe_stats
from opsd_utils.gate_policy import (
    in_sft_cold_start,
    resolve_skip_degenerate_opsd,
    sft_cold_start_steps,
    sft_slots_for_step,
)
from reward_utils.format_checks import evaluate_format_reward


def _gate_cfg():
    return {
        "gate": {
            "skip_degenerate_for_opsd": True,
            "degen_skip_warmup_steps": 200,
            "sft_warmup_steps": 500,
            "sft_warmup_slots_per_group": 4,
            "sft_cold_start_frac": 0.08,
        }
    }


def test_sft_cold_start_steps_from_frac():
    cfg = _gate_cfg()
    assert sft_cold_start_steps(cfg, 1000) == 80
    assert in_sft_cold_start(cfg, 79, 1000) is True
    assert in_sft_cold_start(cfg, 80, 1000) is False


def test_skip_degenerate_after_cold_start_and_warmup():
    cfg = _gate_cfg()
    max_steps = 1000
    assert resolve_skip_degenerate_opsd(cfg, 50, max_steps) is False
    assert resolve_skip_degenerate_opsd(cfg, 280, max_steps) is True


def test_sft_slots_zero_during_cold_start():
    cfg = _gate_cfg()
    assert sft_slots_for_step(cfg, 10, 1000) == 0
    assert sft_slots_for_step(cfg, 200, 1000) == 4


def test_graded_chart_format_partial_credit():
    partial = "Goal: x\nAnswer: 42"
    assert (
        evaluate_format_reward(
            partial,
            "answer:",
            re.compile(r"(?i)answer:"),
            task="chart",
        )
        == 0.5
    )


def test_degenerate_probe_split_format_vs_repeat():
    import torch

    # No Answer: → format degenerate, not repeat degenerate
    ids = torch.tensor([[17, 15, 18, 15, 151645]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
    is_eos = ids == 151645
    stats = summarize_generate_probe_stats(
        ids,
        mask,
        is_eos,
        eos_id=151645,
        completions=["2030"],
        answer_flag="Answer:",
        max_completion_length=128,
    )
    assert stats["degenerate_rate_format"] == 1.0
    assert stats["degenerate_rate_repeat"] == 0.0
