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
from reward_utils.visual_ic import _parse_ic_json, build_prompt_s1, extract_visual_facts_teacher


def test_score_from_label():
    assert _score_from_label("high")[0] == 1.0
    assert _score_from_label("medium")[0] == 0.5
    assert _score_from_label("low")[0] == 0.0
    assert _score_from_label("unknown")[0] == 0.0


def test_build_prompt_s1_with_json_braces():
    prompt = build_prompt_s1("What is the ratio between two countries?")
    assert "__QUESTION__" not in prompt
    assert "What is the ratio" in prompt
    assert '"description"' in prompt
    assert '{"name": "person"' in prompt


def test_parse_ic_json_extracts_object():
    text = 'Here is JSON: {"description": "chart", "objects": [{"name": "bar"}]}'
    obj, err = _parse_ic_json(text)
    assert err is None
    assert obj["description"] == "chart"
    assert len(obj["objects"]) == 1


def test_parse_ic_json_strips_markdown_fence():
    text = '```json\n{"description": "chart", "objects": []}\n```'
    obj, err = _parse_ic_json(text)
    assert err is None
    assert obj["description"] == "chart"


def test_refine_context_sequential_dedupes():
    from reward_utils.compute_rewards import refine_context_sequential

    class StubRefiner:
        requires_sequential = True
        visual_config = {"dedupe_per_batch": True}
        _batch_images = ["img/a.png", "img/a.png", "img/b.png"]

        def __init__(self):
            self.calls = 0

        def refine_hint(self, question, hint, answer, task, gpu_id):
            self.calls += 1
            return f"refined:{hint[:8]}"

        def record_refiner_dedupe(self, **kwargs):
            pass

    refiner = StubRefiner()
    out = refine_context_sequential(
        refiner,
        ["q1", "q1", "q2"],
        ["hint-aaaa", "hint-aaaa", "hint-bbbb"],
        ["a1", "a1", "a2"],
        "chart",
        0,
    )
    assert refiner.calls == 2
    assert out[0] == out[1]
    assert out[2] != out[0]


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


def test_visual_batch_recorder_teacher_timing():
    recorder = VisualBatchRecorder(
        global_step=1,
        output_dir=tempfile.gettempdir(),
        log_cfg={"enabled": True, "save_artifacts": False},
    )
    recorder.record_teacher_timing("ic", latency_ms=10.0, n_calls=2, batch_size=2)
    recorder.record_teacher_timing("checker", latency_ms=20.0, n_calls=1, batch_size=1)
    recorder.record_teacher_timing("refiner", latency_ms=30.0, n_calls=3, batch_size=3)
    summary = recorder.finish()
    assert summary["visual/ic_latency_ms"] == 10.0
    assert summary["visual/checker_latency_ms"] == 20.0
    assert summary["visual/refiner_latency_ms"] == 30.0
    assert summary["visual/ic_calls"] == 2.0
    assert summary["visual/teacher_batch_calls"] == 3.0


def test_refiner_skip_cold_start_only_when_active():
    from reward_utils.visual_refiner_teacher import TeacherVisualRefiner, _RefinerJob

    rl_cfg = {}
    refiner = TeacherVisualRefiner(
        rl_cfg,
        {},
        visual_config={
            "refiner": {"enabled": True, "skip_cold_start": True},
            "prefetch_ic": False,
        },
    )
    refiner.bind_teacher(MagicMock(), MagicMock())
    refiner.begin_generate_batch(
        samples=[{"hint": "h"}],
        images=["img.png"],
        questions=["q?"],
        global_step=1,
        output_dir=tempfile.gettempdir(),
        skip_cold_start=False,
    )
    assert refiner._skip_cold_start_active is False

    with patch.object(refiner, "batch_refine_hints", wraps=refiner.batch_refine_hints) as mock_batch:
        with patch(
            "reward_utils.visual_refiner_teacher.teacher_generate_batched_chunks",
            return_value=(["Goal: x\nObservation: y"], 1.0),
        ):
            with patch(
                "reward_utils.visual_refiner_teacher.extract_visual_facts_teacher",
                return_value=('{"description":"d","objects":[]}', {}),
            ):
                out = refiner.refine_hint("q?", "hint text", "ans", "chart", 0)
        mock_batch.assert_called()
    assert out != "hint text" or "Goal:" in out

    refiner.begin_generate_batch(
        samples=[{"hint": "h"}],
        images=["img.png"],
        questions=["q?"],
        global_step=2,
        output_dir=tempfile.gettempdir(),
        skip_cold_start=True,
    )
    passthrough = refiner.batch_refine_hints(
        [_RefinerJob(0, "q?", "hint text", "ans")],
        "chart",
    )
    assert passthrough[0] == "hint text"


def test_calculate_rewards_sequential_uses_batch_checker():
    from reward_utils.compute_rewards import calculate_rewards_sequential

    class StubChecker:
        requires_sequential = True
        answer_flag = "answer:"

        def __init__(self):
            self.batch_called = False
            self._thinking_score_cache = {}

        def prepare_thinking_jobs(self, responses, prompts, answers, hints, task):
            return [object()]

        def batch_score_thinking(self, jobs, task):
            self.batch_called = True
            self._thinking_score_cache[0] = 0.75
            return {0: 0.75}

        def get_format_reward(self, response, task="chart"):
            return 1.0

        def get_answer_reward(self, prediction, answer, task):
            return 0.5

        def get_thinking_reward_prompt(self, *args):
            raise AssertionError("should use cache after batch")

    checker = StubChecker()
    batch = {
        "response": ["think\nanswer: 3"],
        "prompt": ["p"],
        "answer": ["3"],
        "hints": [""],
    }
    rewards, fmt, ans, think = calculate_rewards_sequential(checker, batch, 0, "chart")
    assert checker.batch_called
    assert think[0] == 0.75
    assert rewards[0] == 1.0 + 0.5 + 0.75


def test_teacher_generate_one_delegates_batch():
    from reward_utils.teacher_generate import teacher_generate_one

    with patch("reward_utils.teacher_generate.teacher_generate_batch") as mock_batch:
        mock_batch.return_value = (["ok"], 5.0)
        text, ms = teacher_generate_one(
            MagicMock(),
            MagicMock(),
            "prompt",
            ["img.png"],
            recorder=MagicMock(),
            timing_kind="ic",
        )
        assert text == "ok"
        assert ms == 5.0
        mock_batch.assert_called_once()


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
@patch("reward_utils.visual_checker_teacher.teacher_generate_batched_chunks")
@patch("reward_utils.visual_checker_teacher.teacher_generate_one")
@patch("reward_utils.visual_checker_teacher.extract_visual_facts_teacher")
def test_teacher_checker_high_triggers_pool(mock_ic, mock_gen_one, mock_gen_batch, _mock_spacy):
    _mock_spacy.return_value = MagicMock()
    mock_ic.return_value = ('{"description":"d","objects":[]}', {})
    mock_gen_batch.return_value = (["high"], 1.0)
    mock_gen_one.side_effect = [
        ("Goal: [State objective]\nObservation: [data]", 2.0),
        ("YES", 3.0),
    ]
    rl_cfg = {"answer_flag": "Answer:"}
    checker = TeacherVisualChecker(
        rl_cfg,
        {},
        gpu_id=0,
        visual_config={"checker": {"enabled": True}, "prefetch_ic": False},
    )
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
