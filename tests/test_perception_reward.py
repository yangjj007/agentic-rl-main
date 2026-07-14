from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_utils.perception_reward import (
    build_perception_judge_prompt,
    parse_perception_judge_score,
    score_perception_rewards,
    sanitize_trusted_hint,
)


def test_image_teacher_prompt_excludes_answer_hint_and_deplot() -> None:
    sample = {
        "question": "What is the value in 2020?",
        "answer": "Answer: 70",
        "hint": "Observation: 2020 is 70.\nAnswer: 70",
        "visual_fact_deplot": "Year | Value\n2020 | 13",
    }

    prompt = build_perception_judge_prompt(
        source="image_teacher",
        sample=sample,
        question="What is the value in 2020?",
        response="Goal: read the bar.\nAnswer: 70",
    )

    assert "Reference Answer" not in prompt
    assert "Answer: 70" not in prompt
    assert "Year | Value" not in prompt
    assert "visual_fact_deplot" not in prompt


def test_trusted_hint_sanitizer_removes_answer_lines() -> None:
    hint = "Goal: read chart\nObservation: 2020 is highest\nAnswer: 70\nConclusion: final is 70"

    sanitized = sanitize_trusted_hint(hint, reference_answer="70")

    assert "Answer:" not in sanitized
    assert "70" not in sanitized
    assert "Observation:" in sanitized


def test_trusted_hint_reward_does_not_use_deplot_when_hint_missing() -> None:
    result = score_perception_rewards(
        samples=[
            {
                "question": "What is the value?",
                "answer": "Answer: 70",
                "visual_fact_deplot": "Value | 70",
            }
        ],
        responses=["Goal: inspect the chart.\nAnswer: 70"],
        source="trusted_hint",
    )

    assert result.rewards == [0.0]
    assert result.stats["skipped_rate"] == 1.0
    assert result.stats["diagnostic_deplot_overlap_mean"] > 0.0


def test_trusted_hint_reward_scores_grounded_reasoning_without_answer_copy() -> None:
    result = score_perception_rewards(
        samples=[
            {
                "question": "Which bar is lowest?",
                "answer": "Answer: B",
                "trusted_hint": "Goal: find the lowest bar\nObservation: B is lower than A and C\nReasoning: compare the three bars\nAnswer: B",
                "visual_fact_deplot": "A | 10\nB | 1\nC | 5",
            }
        ],
        responses=["Goal: find the lowest bar\nObservation: B is lower than A and C\nReasoning: compare the three bars\nAnswer: B"],
        source="trusted_hint",
    )

    assert result.rewards[0] > 0.5
    assert result.stats["skipped_rate"] == 0.0


def test_image_teacher_judge_label_parser() -> None:
    assert parse_perception_judge_score("high")[0] == 1.0
    assert parse_perception_judge_score("The judgement is medium.")[0] == 0.5
    assert parse_perception_judge_score("low confidence")[0] == 0.0
    assert parse_perception_judge_score("not sure")[1] is False
