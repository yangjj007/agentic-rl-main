from __future__ import annotations

from reward_utils.eval_format_reward import score_eval_format_reward


def test_eval_format_reward_accepts_single_final_answer_line() -> None:
    assert score_eval_format_reward("Reasoning: compare values.\nAnswer: 70") == 1.0
    assert score_eval_format_reward("Answer: Brazil") == 1.0


def test_eval_format_reward_rejects_template_pollution() -> None:
    assert score_eval_format_reward("Reasoning style.\nAnswer:\n70") == 0.0
    assert score_eval_format_reward("[Oracle]\nAnswer: 70") == 0.0
    assert score_eval_format_reward("[Final Hard Rule]\nAnswer: 70") == 0.0
    assert score_eval_format_reward("[Verified Hint]\nAnswer: 70") == 0.0
    assert score_eval_format_reward("[DePlot]\nAnswer: 70") == 0.0


def test_eval_format_reward_rejects_ambiguous_answer_lines() -> None:
    assert score_eval_format_reward("Answer: 70\nAnswer: 71") == 0.0
    assert score_eval_format_reward("Reasoning: compare.\nAnswer:") == 0.0
    assert score_eval_format_reward("Goal:\nObservation:\nConclusion:") == 0.0


def test_eval_format_reward_rejects_heading_as_last_line() -> None:
    assert score_eval_format_reward("Answer: 70\nGoal:") == 0.0
    assert score_eval_format_reward("Answer: 70\nObservation:") == 0.0
    assert score_eval_format_reward("Answer: 70\nConclusion:") == 0.0
