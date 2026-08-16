import json

from opsd_utils.teacher_trajectory_log import (
    append_teacher_trajectory_record,
    build_teacher_trajectory_record,
)
from reward_utils.chart_cot_verifier import verify_chart_cot_trajectory


def test_trajectory_audit_keeps_semantic_outputs_and_opd_only_invariant(tmp_path):
    trajectory = (
        "Goal: Find minimum.\nObservation: 2018: 72, 2019: 70, and 2020: 77.\n"
        "Reasoning: The minimum is 70.\nConclusion: The lowest value is 70.\nAnswer: 70"
    )
    verification = verify_chart_cot_trajectory(
        trajectory,
        "Year | Rep/Lean Rep\n2018 | 72\n2019 | 70\n2020 | 77",
        answer_correct=True,
    )
    record = build_teacher_trajectory_record(
        sample={"question_wo_prompt": "What is lowest?", "image": "chart.png"},
        global_step=0,
        rank=0,
        global_idx=1,
        source_idx=0,
        prompt_idx=0,
        generation_idx=1,
        reference="Answer: 70",
        student_output="20",
        teacher_probe_output="Answer: 70",
        teacher_probe_correct=True,
        trajectory_output=trajectory,
        trajectory_raw_output=trajectory.removesuffix("\nAnswer: 70"),
        trajectory_evidence_retry_raw_output="",
        trajectory_prompt_profile="chartqa_structured_trajectory",
        trajectory_answer="70",
        trajectory_answer_source="teacher_probe_completion",
        trajectory_correct=True,
        trajectory_parse_failed=False,
        verification=verification,
    )

    assert record["student_output"] == "20"
    assert record["teacher_trajectory_output"] == trajectory
    assert record["teacher_trajectory_raw_output"].endswith("The lowest value is 70.")
    assert record["teacher_trajectory_evidence_retry_raw_output"] == ""
    assert record["trajectory_prompt_profile"] == "chartqa_structured_trajectory"
    assert record["schema_version"] == 2
    assert record["trajectory_answer_source"] == "teacher_probe_completion"
    assert record["quality"]["quality"] == "Q3"
    assert record["routing"] == {
        "opd_only": True,
        "opsd_selected": True,
        "grpo_selected": False,
        "sft_selected": False,
    }
    path = append_teacher_trajectory_record(
        output_dir=str(tmp_path),
        opsd_config={"teacher_trajectory": {"audit_log": {"enabled": True}}},
        rank=0,
        record=record,
    )
    assert path == str(tmp_path / "teacher_trajectories" / "rank0.jsonl")
    assert json.loads((tmp_path / "teacher_trajectories" / "rank0.jsonl").read_text()) == record
