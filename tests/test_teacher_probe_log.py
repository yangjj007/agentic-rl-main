import json
import importlib.util
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_utils.chart.deplot_pipeline import placeholder_deplot_table

_SPEC = importlib.util.spec_from_file_location(
    "teacher_probe_log",
    os.path.join(ROOT, "opsd_utils", "teacher_probe_log.py"),
)
teacher_probe_log = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(teacher_probe_log)
append_teacher_probe_record = teacher_probe_log.append_teacher_probe_record
build_teacher_probe_record = teacher_probe_log.build_teacher_probe_record


def test_teacher_probe_record_summarizes_route_and_deplot_placeholder():
    sample = {
        "question": "What is the highest bar?",
        "answer": "42",
        "image": "/tmp/chart.png",
        "visual_fact": "Goal: inspect chart",
        "visual_fact_deplot": placeholder_deplot_table({"question": "q"}),
    }

    record = build_teacher_probe_record(
        sample=sample,
        global_step=17,
        rank=2,
        global_idx=5,
        prompt_idx=1,
        generation_idx=2,
        provider_names=["format_only", "visual_facts_deplot"],
        reference="42",
        student_output="Answer: 41",
        teacher_output="Reasoning...\nAnswer: 42",
        score=1.0,
        final_route="opd",
        answer_flag="Answer:",
        max_text_chars=32,
    )

    assert record["global_step"] == 17
    assert record["rank"] == 2
    assert record["final_route"] == "opd"
    assert record["teacher_correct"] is True
    assert record["question"] == "What is the highest bar?"
    assert record["student_output"] == "Answer: 41"
    assert record["teacher_output"] == "Reasoning...\\nAnswer: 42"
    assert record["privileged"]["visual_fact_deplot_status"] == "placeholder"
    assert record["privileged"]["visual_fact_present"] is True
    assert record["provider_names"] == ["format_only", "visual_facts_deplot"]


def test_append_teacher_probe_record_is_disabled_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        path = append_teacher_probe_record(
            output_dir=tmp,
            opsd_config={},
            rank=0,
            record={"global_step": 1},
        )

        assert path is None
        assert not os.path.exists(os.path.join(tmp, "teacher_probe_candidates"))


def test_append_teacher_probe_record_writes_jsonl_when_enabled():
    with tempfile.TemporaryDirectory() as tmp:
        record = {"global_step": 1, "teacher_output": "Answer: 7"}

        path = append_teacher_probe_record(
            output_dir=tmp,
            opsd_config={"teacher_probe": {"candidate_log": {"enabled": True}}},
            rank=3,
            record=record,
        )

        assert path == os.path.join(tmp, "teacher_probe_candidates", "rank3.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert rows == [{"global_step": 1, "teacher_output": "Answer: 7"}]
