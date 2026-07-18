from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.constants import MODE_GRPO, MODE_OPSD, MODE_SFT
from opsd_utils.teacher_sft_repair import (
    TeacherSftRepairConfig,
    apply_teacher_sft_repair_routing,
    build_teacher_sft_repair_target,
    constrain_teacher_sft_repair_target,
    sanitize_teacher_sft_text,
    teacher_sft_target_quality,
    teacher_sft_repair_advantages,
)


def test_all_wrong_teacher_correct_becomes_sft_repair_and_removes_traj() -> None:
    modes, kept_trajs, repairs, stats = apply_teacher_sft_repair_routing(
        completion_modes=[MODE_OPSD, MODE_OPSD],
        teacher_traj_indices={0, 1},
        teacher_correct_indices={0, 1},
        group_has_correct=[False],
        num_generations=2,
        config=TeacherSftRepairConfig(repair_mode="traj_sft", scope="all_wrong", slots_per_prompt=1),
    )

    assert modes == [MODE_SFT, MODE_OPSD]
    assert repairs == {0}
    assert kept_trajs == {1}
    assert stats.teacher_sft_repairs == 1
    assert stats.teacher_sft_repair_all_wrong == 1
    assert stats.repair_slot_eligible == 1
    assert stats.teacher_correct_to_sft_repair == 1
    assert stats.teacher_correct_to_opd == 1


def test_mixed_teacher_correct_stays_opd_under_all_wrong_scope() -> None:
    modes, kept_trajs, repairs, stats = apply_teacher_sft_repair_routing(
        completion_modes=[MODE_GRPO, MODE_OPSD],
        teacher_traj_indices={1},
        teacher_correct_indices={1},
        group_has_correct=[True],
        num_generations=2,
        config=TeacherSftRepairConfig(repair_mode="traj_sft", scope="all_wrong", slots_per_prompt=1),
    )

    assert modes == [MODE_GRPO, MODE_OPSD]
    assert kept_trajs == {1}
    assert repairs == set()
    assert stats.teacher_correct_to_opd == 1


def test_default_opd_mode_preserves_existing_teacher_traj_routing() -> None:
    modes, kept_trajs, repairs, stats = apply_teacher_sft_repair_routing(
        completion_modes=[MODE_OPSD],
        teacher_traj_indices={0},
        teacher_correct_indices={0},
        group_has_correct=[False],
        num_generations=1,
        config=TeacherSftRepairConfig(),
    )

    assert modes == [MODE_OPSD]
    assert kept_trajs == {0}
    assert repairs == set()
    assert stats.teacher_sft_repairs == 0
    assert stats.teacher_correct_to_opd == 1


def test_refiner_sft_repair_promotes_all_wrong_teacher_correct_without_traj() -> None:
    modes, kept_trajs, repairs, stats = apply_teacher_sft_repair_routing(
        completion_modes=[MODE_OPSD, MODE_OPSD, MODE_GRPO, MODE_OPSD],
        teacher_traj_indices=set(),
        teacher_correct_indices={0, 1, 3},
        group_has_correct=[False, True],
        num_generations=2,
        config=TeacherSftRepairConfig(
            repair_mode="refiner_sft",
            scope="all_wrong",
            slots_per_prompt=1,
        ),
    )

    assert modes == [MODE_SFT, MODE_OPSD, MODE_GRPO, MODE_OPSD]
    assert repairs == {0}
    assert kept_trajs == set()
    assert stats.teacher_sft_repairs == 1
    assert stats.teacher_sft_repair_all_wrong == 1
    assert stats.repair_slot_eligible == 1
    assert stats.teacher_correct_to_sft_repair == 1
    assert stats.teacher_correct_to_opd == 2


def test_sanitize_teacher_sft_text_removes_privileged_tags_but_keeps_answer() -> None:
    text = """[Verified Hint]
Goal: Find the maximum.
[Reference Answer] 42
[DePlot] A | B | 42
[Visual Facts - DePlot] table text
Reasoning: The maximum visible value is 42.
Answer: 42"""

    cleaned = sanitize_teacher_sft_text(text)

    assert "[Verified Hint]" not in cleaned
    assert "[Reference Answer]" not in cleaned
    assert "[DePlot]" not in cleaned
    assert "[Visual Facts" not in cleaned
    assert "Reasoning: The maximum visible value is 42." in cleaned
    assert "Answer: 42" in cleaned


def test_teacher_sft_repair_advantages_are_unit_weighted_like_sft() -> None:
    mask = torch.tensor([1, 1, 0], dtype=torch.long)

    advantages = teacher_sft_repair_advantages(mask)

    assert torch.equal(advantages, torch.tensor([1.0, 1.0, 1.0]))


def test_constrain_teacher_sft_target_falls_back_to_verified_hint_when_teacher_is_clipped() -> None:
    sample = {
        "answer": "51",
        "hint": (
            "Goal: Find the rightmost value of the orange graph.\n"
            "Observation: The orange graph values are 2013: 73, 2014: 65, and 2015: 73.\n"
            "Reasoning: The rightmost value corresponds to the latest x-axis position.\n"
            "Conclusion: The rightmost value of the orange graph is 73."
        ),
    }
    raw_teacher = (
        "[Verified Hint]\n"
        "Goal: Find the rightmost value.\n"
        "Observation: DePlot table says 73, 65, 73.\n"
        "Reasoning: Use the rightmost point.\n"
        "Conclusion: The..."
    )

    constrained = constrain_teacher_sft_repair_target(
        raw_teacher,
        sample=sample,
        reference_answer="51",
        sanitize_privileged=True,
    )

    assert constrained.used_fallback_hint is True
    assert constrained.raw_clipped is True
    assert "[Verified Hint]" not in constrained.text
    assert "[DePlot]" not in constrained.text
    assert constrained.text.startswith("Goal: Find the rightmost value of the orange graph.")
    assert "\nObservation: The orange graph values are 2013: 73, 2014: 65, and 2015: 73." in constrained.text
    assert "\nConclusion: Therefore, the answer is 51." in constrained.text
    assert constrained.text.rstrip().endswith("Answer: 51")
    quality = teacher_sft_target_quality(constrained.text, "51")
    assert quality["full_hint_format"] is True
    assert quality["exact_reference_answer_line"] is True
    assert quality["privileged_tag_present"] is False


def test_constrain_teacher_sft_target_keeps_valid_teacher_content_but_patches_answer_line() -> None:
    sample = {
        "answer": "70",
        "hint": (
            "Goal: Find the lowest value of the red graph.\n"
            "Observation: The red values are 72, 70, and 77.\n"
            "Reasoning: Compare the values.\n"
            "Conclusion: The lowest value is 70."
        ),
    }
    raw_teacher = (
        "Goal: Identify the minimum red value.\\n"
        "Observation: The teacher sees red values 72, 70, and 77.\\n"
        "Reasoning: The smallest of these values is 70.\\n"
        "Conclusion: The minimum red value is 70.\\n"
        "Answer: 72"
    )

    constrained = constrain_teacher_sft_repair_target(
        raw_teacher,
        sample=sample,
        reference_answer="70",
    )

    assert constrained.used_fallback_hint is False
    assert "The teacher sees red values 72, 70, and 77." in constrained.text
    assert constrained.text.rstrip().endswith("Answer: 70")
    assert "Answer: 72" not in constrained.text


def test_build_teacher_sft_target_answer_only_is_exact_answer_line() -> None:
    target = build_teacher_sft_repair_target(
        "[Verified Hint]\nReasoning: Ignore privileged tags.\nAnswer: 12",
        sample={"answer": "51"},
        reference_answer="51",
        target_style="answer_only",
        sanitize_privileged=True,
    )

    assert target.text == "Answer: 51"
    assert target.answer_only_format is True
    assert target.student_short_format is False
    assert target.privileged_tag_present is False


def test_build_teacher_sft_target_student_short_uses_concise_reasoning_and_exact_answer() -> None:
    target = build_teacher_sft_repair_target(
        "[Verified Hint]\n"
        "Goal: Find the lowest value.\n"
        "Observation: The red values are 72, 70, and 77.\n"
        "Reasoning: Compare the values and choose the smallest one.\n"
        "Conclusion: The lowest value is 70.\n"
        "[DePlot] noisy table\n"
        "Answer: 72",
        sample={"answer": "70"},
        reference_answer="70",
        target_style="student_short",
        sanitize_privileged=True,
    )

    assert target.text == (
        "Reasoning: Compare the values and choose the smallest one.\n"
        "Answer: 70"
    )
    assert "Goal:" not in target.text
    assert "Observation:" not in target.text
    assert "Conclusion:" not in target.text
    assert "[Verified Hint]" not in target.text
    assert "[DePlot]" not in target.text
    assert target.student_short_format is True
    assert target.answer_only_format is False
    assert target.exact_reference_answer_line is True


def test_build_teacher_sft_target_student_short_falls_back_to_answer_only_without_reasoning() -> None:
    target = build_teacher_sft_repair_target(
        "[Visual Facts - DePlot]\nNo usable reasoning.\nAnswer: 99",
        sample={"answer": "51"},
        reference_answer="51",
        target_style="student_short",
        sanitize_privileged=True,
    )

    assert target.text == "Answer: 51"
    assert target.student_short_format is True
    assert target.answer_only_format is True


def test_build_teacher_sft_target_student_hint_short_prefers_verified_hint_reasoning() -> None:
    target = build_teacher_sft_repair_target(
        "Reasoning: Teacher says to follow a noisy DePlot-derived shortcut.\nAnswer: 12",
        sample={
            "answer": "51",
            "hint": (
                "Goal: Read the chart.\n"
                "Observation: The verified data supports 51.\n"
                "Reasoning: Use the verified hint reasoning instead of the teacher shortcut.\n"
                "Conclusion: The answer is 51."
            ),
        },
        reference_answer="51",
        target_style="student_hint_short",
        sanitize_privileged=True,
    )

    assert target.text == (
        "Reasoning: Use the verified hint reasoning instead of the teacher shortcut.\n"
        "Answer: 51"
    )
    assert "noisy DePlot-derived" not in target.text
    assert target.student_short_format is True
    assert target.answer_only_format is False
    assert target.exact_reference_answer_line is True


def test_build_teacher_sft_target_student_hint_short_falls_back_to_teacher_reasoning() -> None:
    target = build_teacher_sft_repair_target(
        "Reasoning: Teacher derives the answer from the visible bars.\nAnswer: 12",
        sample={"answer": "12", "hint": "No sectioned reasoning here."},
        reference_answer="12",
        target_style="student_hint_short",
        sanitize_privileged=True,
    )

    assert target.text == (
        "Reasoning: Teacher derives the answer from the visible bars.\n"
        "Answer: 12"
    )
    assert target.student_short_format is True
