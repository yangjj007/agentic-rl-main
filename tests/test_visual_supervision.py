"""Unit tests for Visual Supervision (7B teacher path)."""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from opsd_utils.visual_supervision_log import VisualBatchRecorder
from reward_utils.template_pool import TemplatePool
from reward_utils.visual_checker_teacher import TeacherVisualChecker, _score_from_label
from reward_utils.visual_ic import _parse_ic_json, extract_visual_facts_teacher


def test_score_from_label():
    assert _score_from_label("high")[0] == 1.0
    assert _score_from_label("medium")[0] == 0.5
    assert _score_from_label("low")[0] == 0.0
    assert _score_from_label("unknown")[0] == 0.0


def test_parse_ic_json_extracts_object():
    text = 'Here is JSON: {"description": "chart", "objects": [{"name": "bar"}]}'
    obj, err = _parse_ic_json(text)
    assert err is None
    assert obj["description"] == "chart"
    assert len(obj["objects"]) == 1


def test_template_pool_cas_write():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "best_template.txt")
        pool = TemplatePool(template_path=path, lock_path=path + ".lock")
        written, label = pool.maybe_update("Goal: [x]", lambda _c, _n: True)
        assert written is True
        assert label == "YES"
        assert pool.get_template(force_refresh=True) == "Goal: [x]"
        written2, label2 = pool.maybe_update("Goal: [x]", lambda _c, _n: True)
        assert written2 is False
        assert label2 == "identical"


def test_visual_batch_recorder_log_format(capsys):
    recorder = VisualBatchRecorder(
        global_step=7,
        output_dir=tempfile.gettempdir(),
        log_cfg={"enabled": True, "sample_count": 1, "save_artifacts": False},
    )
    recorder.record_checker(
        sample_idx=0,
        score=1.0,
        label="high",
        thinking_len=42,
        thinking_preview="Goal: test",
    )
    summary = recorder.finish()
    captured = capsys.readouterr().out
    assert "[VISUAL-CHECKER]" in captured
    assert "[VISUAL-BATCH]" in captured
    assert summary["visual/checker_mean"] == 1.0


@patch("reward_utils.spacy_model.load_spacy_english")
@patch("reward_utils.visual_checker_teacher.teacher_generate_one")
@patch("reward_utils.visual_checker_teacher.extract_visual_facts_teacher")
def test_teacher_checker_high_triggers_pool(mock_ic, mock_gen, _mock_spacy):
    _mock_spacy.return_value = MagicMock()
    mock_ic.return_value = ('{"description":"d","objects":[]}', {})
    mock_gen.side_effect = [
        ("high", 1.0),
        ("Goal: [State objective]\nObservation: [data]", 2.0),
        ("YES", 3.0),
    ]
    rl_cfg = {"answer_flag": "Answer:"}
    checker = TeacherVisualChecker(rl_cfg, {}, gpu_id=0, visual_config={"checker": {"enabled": True}})
    checker.bind_teacher(MagicMock(), MagicMock())
    checker.begin_generate_batch(
        samples=[{"hint": "x", "visual_fact_hint": "vf"}],
        images=["img.png"],
        questions=["q?"],
        global_step=1,
        output_dir=tempfile.gettempdir(),
    )
    checker._current_sample_idx = 0
    score = checker.get_thinking_reward_prompt(
        "Goal: find max\nAnswer: 3",
        "prompt",
        "3",
        "hint",
        "chart",
    )
    assert score == 1.0
    stats = checker.end_generate_batch()
    assert stats["visual/checker_high"] == 1.0


def test_extract_visual_facts_fallback_without_teacher():
    sample = {"visual_fact_hint": json.dumps({"description": "x", "objects": []})}
    ic_text, meta = extract_visual_facts_teacher(
        teacher_model=None,
        processor=None,
        sample=sample,
        question="What?",
        image="a.png",
    )
    assert "description" in ic_text
    assert meta.get("fallback", "").startswith("hint_")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
