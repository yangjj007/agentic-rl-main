"""Regression tests for completion degeneration heuristics."""
from opsd_utils.diagnostics import (
    _detect_degeneration,
    _detect_repeat_loop,
    _detect_single_token_repeat,
    _max_same_token_run,
    is_degenerate_completion,
)


def test_single_token_repeat_detects_cjk_loop():
    ids = [39992, 25, 7379] + [41146] * 40
    assert _detect_single_token_repeat(ids)
    assert _max_same_token_run(ids) == (40, 41146)


def test_ngram_repeat_not_limited_to_first_eight_tokens():
    prefix = list(range(20))
    gram = [9, 8, 7]
    ids = prefix + gram * 5
    assert _detect_repeat_loop(ids)


def test_is_degenerate_completion_detects_repeat():
    ids = [39992, 25] + [41146] * 20
    assert is_degenerate_completion(ids, "Goal: x\n" + "其" * 40)


def test_degeneration_flags_missing_answer():
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    text = "Goal: test\nObservation: x\nReasoning: y\nConclusion: z"
    is_deg, reasons = _detect_degeneration(ids, text, answer_flag="Answer:")
    assert is_deg
    assert any(r.startswith("ANSWER_FLAG_COUNT") for r in reasons)
