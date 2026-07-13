"""Tests for ChartQA format reward hardening."""
import re

from reward_utils.format_checks import (
    evaluate_format_reward,
    is_digit_spam_after_answer,
    should_zero_chart_format,
)


def _chart_format(response: str, task: str = "chart") -> float:
    return evaluate_format_reward(
        response,
        "answer:",
        re.compile(r"(?i)answer:"),
        task=task,
    )


def test_digit_spam_zeros_format():
    spam = "Goal: x\nObservation: y\nAnswer:579683579683579683579683579683"
    assert is_digit_spam_after_answer(spam, "answer:")
    assert _chart_format(spam) == 0.0


def test_min_thinking_required_for_chart():
    short = "Answer: 42"
    assert _chart_format(short) == 0.0
    ok = "Goal: read chart\nObservation: bar at 42\nAnswer: 42"
    assert _chart_format(ok) == 1.0


def test_non_chart_unchanged_without_min_thinking():
    bare = "Answer: 42"
    assert _chart_format(bare, task="math") == 1.0


def test_should_zero_on_char_repeat():
    repeated = "Answer:" + "1" * 80
    assert should_zero_chart_format(repeated, "answer:", require_answer_flag_for_degen=False)


def test_config_opd_inherits_rlsd_dyme_args():
    import config.config_opd_7b_chartqa as opd
    import config.config_rlsd_chartqa as rlsd

    opd_dyme = opd.CONFIG["training"]["dyme_args"]
    rlsd_dyme = rlsd.CONFIG["training"]["dyme_args"]
    assert opd_dyme["max_completion_length"] == rlsd_dyme["max_completion_length"]
    assert opd_dyme["temperature"] == rlsd_dyme["temperature"]
    assert opd_dyme["repetition_penalty"] == rlsd_dyme["repetition_penalty"]


def test_trainer_skip_degenerate_warmup():
    from opsd_utils.gate_policy import resolve_skip_degenerate_opsd, sft_slots_for_step

    cfg = {
        "gate": {
            "skip_degenerate_for_opsd": True,
            "degen_skip_warmup_steps": 200,
            "sft_warmup_steps": 200,
            "sft_warmup_slots_per_group": 2,
            "sft_cold_start_frac": 0.0,
        }
    }
    assert resolve_skip_degenerate_opsd(cfg, 50, 1000) is False
    assert sft_slots_for_step(cfg, 50, 1000) == 2
    assert resolve_skip_degenerate_opsd(cfg, 250, 1000) is True
    assert sft_slots_for_step(cfg, 250, 1000) == 1
