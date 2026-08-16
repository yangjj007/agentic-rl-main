"""Unit tests for Visual Supervision (7B teacher path)."""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from opsd_utils.visual_supervision_log import VisualBatchRecorder
from reward_utils.template_pool import TemplatePool, is_valid_reasoning_template
from reward_utils.visual_checker_teacher import (
    TeacherVisualChecker,
    _postprocess_checker_label,
    _score_from_label,
    _split_response_parts,
    build_image_primary_checker_prompt,
)
from reward_utils.visual_refiner_teacher import build_no_gold_refiner_prompt
from reward_utils.visual_ic import (
    _parse_ic_json,
    build_prompt_s1,
    extract_visual_facts_teacher,
    ic_text_from_offline_sample,
)


def test_score_from_label():
    assert _score_from_label("high")[0] == 1.0
    assert _score_from_label("medium")[0] == 0.5
    assert _score_from_label("low")[0] == 0.0
    assert _score_from_label("unknown")[0] == 0.0


def test_image_primary_checker_prompt_excludes_aux_by_default():
    prompt = build_image_primary_checker_prompt(
        question="What is the 2020 value?",
        answer="10",
        reasoning="The chart shows 2020 is 10.",
        student_answer="10",
        aux_evidence="WRONG DEPLOT: 2020 is 99",
        aux_mode="none",
    )
    low = prompt.lower()
    assert "attached chart image is the only authoritative visual source" in low
    assert "return exactly one token" in low
    assert "WRONG DEPLOT" not in prompt
    assert "2020 is 99" not in prompt


def test_image_primary_checker_prompt_marks_deplot_as_noisy_aux():
    prompt = build_image_primary_checker_prompt(
        question="What is the 2020 value?",
        answer="10",
        reasoning="The chart shows 2020 is 10.",
        student_answer="10",
        aux_evidence="DePlot says 2020 is 99",
        aux_mode="deplot_noisy",
    )
    low = prompt.lower()
    assert "optional noisy extracted text" in low
    assert "may be incomplete or wrong" in low
    assert "conflicts with the image" in low
    assert "DePlot says 2020 is 99" in prompt


def test_image_primary_checker_prompt_includes_student_final_answer():
    prompt = build_image_primary_checker_prompt(
        question="Which group is highest?",
        answer="18-24 years old",
        reasoning="The tallest bar is the 18-24 group at 40%.",
        student_answer="40%",
        aux_mode="none",
    )
    assert "Student final answer:" in prompt
    assert "\n40%\n" in prompt
    assert "supports both the student's final answer and the reference answer" in prompt


def test_no_gold_refiner_prompt_cannot_embed_reference_answer():
    prompt = build_no_gold_refiner_prompt(
        ic_text="The chart has a blue bar of 10.",
        question="What is the blue value?",
        template="Goal: ...\nObservation: ...\nReasoning: ...",
    )
    assert "Reference answer" not in prompt
    assert "Do not infer, state, or format a final answer" in prompt
    assert "Answer:" not in prompt


def test_no_gold_refiner_rejects_answer_only_or_answer_leaking_output():
    from reward_utils.visual_refiner_teacher import _is_usable_refined_hint

    assert not _is_usable_refined_hint("Yes", include_gold=False)
    assert not _is_usable_refined_hint(
        "Goal: inspect\nObservation: value is visible\nAnswer: 10",
        include_gold=False,
    )
    assert _is_usable_refined_hint(
        "Goal: inspect\nObservation: value is visible\nReasoning: compare values",
        include_gold=False,
    )


def test_template_pool_rejects_invalid_candidates_and_disk_templates():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "template.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("Yes")
        pool = TemplatePool(template_path=path)
        assert is_valid_reasoning_template(pool.get_template())
        assert pool.maybe_update("Yes", lambda _cur, _new: True) == (False, "none_template")


def test_checker_postprocess_caps_fragments_and_missing_answer_marker():
    score, label, reason = _postprocess_checker_label(
        score=1.0,
        label="high",
        reasoning="46%",
        has_answer_flag=False,
    )
    assert (score, label, reason) == (0.0, "low", "answer_fragment")

    score, label, reason = _postprocess_checker_label(
        score=1.0,
        label="high",
        reasoning="The chart shows Q3 is higher than Q2 because the bar is taller.",
        has_answer_flag=False,
        student_answer_correct=True,
    )
    assert (score, label, reason) == (0.5, "medium", "missing_answer_flag_high_cap")


def test_checker_postprocess_caps_teacher_high_when_student_answer_is_wrong():
    score, label, reason = _postprocess_checker_label(
        score=1.0,
        label="high",
        reasoning="The chart shows Q3 is higher than Q2 because the bar is taller.",
        has_answer_flag=True,
        student_answer_correct=False,
    )
    assert (score, label, reason) == (0.5, "medium", "answer_incorrect_high_cap")


def test_checker_postprocess_promotes_correct_answer_fragment_to_medium():
    score, label, reason = _postprocess_checker_label(
        score=1.0,
        label="high",
        reasoning="37.8",
        has_answer_flag=False,
        student_answer_correct=True,
    )
    assert (score, label, reason) == (0.5, "medium", "correct_answer_fragment")


def test_split_response_parts_preserves_original_case():
    thinking, answer, has_flag = _split_response_parts(
        "The chart shows DDT exposure was highest for Ages 18-24.\nAnSwEr: 40%",
        "answer:",
    )
    assert has_flag is True
    assert thinking == "The chart shows DDT exposure was highest for Ages 18-24."
    assert answer == "40%"


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


def test_parse_ic_json_accepts_valid_object_before_trailing_text():
    obj, err = _parse_ic_json('{"description": "chart", "objects": []}\nExplanation')
    assert err is None
    assert obj == {"description": "chart", "objects": []}


def test_parse_ic_json_rejects_a_truncated_outer_object_with_nested_json():
    obj, err = _parse_ic_json(
        '{"description": "truncated", "objects": [{"name": "bar"}]'
    )
    assert obj is None
    assert err == "json_decode"


def test_parse_ic_json_rejects_an_object_fragment_without_ic_schema():
    obj, err = _parse_ic_json(
        '{"name": "line graph", "attributes": ["title"], "position": "center"}'
    )
    assert obj is None
    assert err == "invalid_ic_schema"


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
        candidate = (
            "Goal: [x]\nObservation: [evidence]\n"
            "Reasoning: [compare]\nConclusion: [result]"
        )
        written, label = pool.maybe_update(candidate, lambda _c, _n: True)
        assert written is True
        assert label == "YES"
        assert pool.get_template(force_refresh=True) == candidate
        written2, label2 = pool.maybe_update(candidate, lambda _c, _n: True)
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
    assert summary["visual/ic_batch_calls"] == 1.0
    assert summary["visual/checker_batch_calls"] == 1.0
    assert summary["visual/refiner_batch_calls"] == 1.0


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


def test_calculate_rewards_sanitizes_non_finite_parallel_and_sequential():
    from reward_utils.compute_rewards import calculate_rewards_in_parallel, calculate_rewards_sequential

    class UnsafeChecker:
        requires_sequential = False
        answer_flag = "answer:"

        def get_format_reward(self, response, task="chart"):
            if "bad-format" in response:
                return float("nan")
            return 1.0

        def get_answer_reward(self, prediction, answer, task):
            return float("inf")

        def get_thinking_reward_prompt(self, response, prompt, answer, hint, task):
            return None

    batch = {
        "response": ["think\nanswer: 3", "bad-format\nanswer: 4"],
        "prompt": ["p1", "p2"],
        "answer": ["3", "4"],
        "hints": ["h1", "h2"],
    }
    for fn in (calculate_rewards_in_parallel, calculate_rewards_sequential):
        rewards, fmt, ans, think = fn(UnsafeChecker(), batch, 0, task="chart")
        assert rewards == [1.0, 0.0]
        assert fmt == [1.0, 0.0]
        assert ans == [0.0, 0.0]
        assert think == [0.0, 0.0]


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


def test_visual_batch_recorder_saves_route_artifacts(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        recorder = VisualBatchRecorder(
            global_step=3,
            output_dir=tmp,
            log_cfg={"enabled": True, "save_artifacts": True},
        )
        recorder.record_route(
            sample_idx=0,
            route="opd",
            checker_score=0.0,
            answer_reward=1.0,
            format_reward=0.0,
        )
        recorder.finish()
        captured = capsys.readouterr().out
        assert "visual/checker_mean=0.0" in captured

        path = os.path.join(tmp, "visual_supervision", "step_3", "rank0.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert rows == [
            {
                "kind": "route",
                "sample_idx": 0,
                "route": "opd",
                "checker_score": 0.0,
                "answer_reward": 1.0,
                "format_reward": 0.0,
            }
        ]


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
        "Goal: find max\nObservation: the chart shows category A has the highest value.\nAnswer: 3",
        "prompt",
        "3",
        "hint",
        "chart",
    )
    assert score == 1.0
    stats = checker.end_generate_batch()
    assert stats["visual/checker_high"] == 1.0


@patch("reward_utils.spacy_model.load_spacy_english")
@patch("reward_utils.visual_checker_teacher.teacher_generate_batched_chunks")
@patch("reward_utils.visual_checker_teacher.extract_visual_facts_teacher")
def test_teacher_checker_default_uses_image_prompt_without_deplot(
    mock_ic,
    mock_gen_batch,
    _mock_spacy,
):
    _mock_spacy.return_value = MagicMock()
    mock_gen_batch.return_value = (["medium"], 1.0)
    rl_cfg = {"answer_flag": "Answer:"}
    checker = TeacherVisualChecker(
        rl_cfg,
        {},
        gpu_id=0,
        visual_config={
            "checker": {"enabled": True},
            "prefetch_ic": False,
            "logging": {"enabled": True, "save_artifacts": False},
        },
    )
    checker.bind_teacher(MagicMock(), MagicMock())
    checker.begin_generate_batch(
        samples=[{"visual_fact_deplot": "WRONG DEPLOT TABLE 2020 | 99"}],
        images=["chart.png"],
        questions=["What is the 2020 value?"],
        global_step=1,
        output_dir=tempfile.gettempdir(),
    )
    checker._current_sample_idx = 0
    score = checker.get_thinking_reward_prompt(
        "The chart shows the 2020 value.\nAnswer: 10",
        "prompt",
        "10",
        "hint",
        "chart",
    )
    assert score == 0.5
    mock_ic.assert_not_called()
    request = mock_gen_batch.call_args.args[2][0]
    assert request.images == ["chart.png"]
    assert "WRONG DEPLOT" not in request.prompt_text
    assert "attached chart image is the only authoritative visual source" in request.prompt_text.lower()


@patch("reward_utils.spacy_model.load_spacy_english")
@patch("reward_utils.visual_checker_teacher.teacher_generate_batched_chunks")
def test_teacher_checker_missing_image_records_low_without_deplot_fallback(mock_gen_batch, _mock_spacy):
    _mock_spacy.return_value = MagicMock()
    rl_cfg = {"answer_flag": "Answer:"}
    checker = TeacherVisualChecker(
        rl_cfg,
        {},
        gpu_id=0,
        visual_config={
            "checker": {"enabled": True},
            "prefetch_ic": False,
            "logging": {"enabled": True, "save_artifacts": False},
        },
    )
    checker.bind_teacher(MagicMock(), MagicMock())
    checker.begin_generate_batch(
        samples=[{"visual_fact_deplot": "DePlot table exists but image is missing"}],
        images=[None],
        questions=["What is shown?"],
        global_step=1,
        output_dir=tempfile.gettempdir(),
    )
    checker._current_sample_idx = 0
    score = checker.get_thinking_reward_prompt(
        "The chart shows the value.\nAnswer: 10",
        "prompt",
        "10",
        "hint",
        "chart",
    )
    stats = checker.end_generate_batch()
    assert score == 0.0
    mock_gen_batch.assert_not_called()
    assert stats["visual/checker_low"] == 1.0
    assert stats["visual/checker_image_missing"] == 1.0


@patch("reward_utils.spacy_model.load_spacy_english")
@patch("reward_utils.visual_checker_teacher.teacher_generate_batched_chunks")
def test_teacher_checker_deplot_noisy_aux_includes_warning(mock_gen_batch, _mock_spacy):
    _mock_spacy.return_value = MagicMock()
    mock_gen_batch.return_value = (["low"], 1.0)
    rl_cfg = {"answer_flag": "Answer:"}
    checker = TeacherVisualChecker(
        rl_cfg,
        {},
        gpu_id=0,
        visual_config={
            "checker": {
                "enabled": True,
                "aux_evidence": "deplot_noisy",
            },
            "prefetch_ic": False,
            "logging": {"enabled": True, "save_artifacts": False},
        },
    )
    checker.bind_teacher(MagicMock(), MagicMock())
    checker.begin_generate_batch(
        samples=[{"visual_fact_hint": "Noisy extracted text: 2020 | 99"}],
        images=["chart.png"],
        questions=["What is shown?"],
        global_step=1,
        output_dir=tempfile.gettempdir(),
    )
    checker._current_sample_idx = 0
    checker.get_thinking_reward_prompt(
        "The chart shows the value.\nAnswer: 10",
        "prompt",
        "10",
        "hint",
        "chart",
    )
    request = mock_gen_batch.call_args.args[2][0]
    assert "Noisy extracted text: 2020 | 99" in request.prompt_text
    assert "may be incomplete or wrong" in request.prompt_text.lower()
    assert "conflicts with the image" in request.prompt_text.lower()


@patch("reward_utils.spacy_model.load_spacy_english")
@patch("reward_utils.visual_checker_teacher.teacher_generate_one")
@patch("reward_utils.visual_checker_teacher.teacher_generate_batched_chunks")
def test_teacher_checker_caps_teacher_high_for_answer_fragment(
    mock_gen_batch,
    mock_gen_one,
    _mock_spacy,
):
    _mock_spacy.return_value = MagicMock()
    mock_gen_batch.return_value = (["high"], 1.0)
    rl_cfg = {"answer_flag": "Answer:"}
    checker = TeacherVisualChecker(
        rl_cfg,
        {},
        gpu_id=0,
        visual_config={
            "checker": {"enabled": True},
            "prefetch_ic": False,
            "logging": {"enabled": True, "save_artifacts": False},
        },
    )
    checker.bind_teacher(MagicMock(), MagicMock())
    checker.begin_generate_batch(
        samples=[{}],
        images=["chart.png"],
        questions=["What is the value?"],
        global_step=1,
        output_dir=tempfile.gettempdir(),
    )
    checker._current_sample_idx = 0
    score = checker.get_thinking_reward_prompt(
        "47%",
        "prompt",
        "46%",
        "hint",
        "chart",
    )
    stats = checker.end_generate_batch()
    assert score == 0.0
    assert stats["visual/checker_high"] == 0.0
    assert stats["visual/checker_low"] == 1.0
    mock_gen_one.assert_not_called()


@patch("reward_utils.spacy_model.load_spacy_english")
def test_teacher_checker_promotes_correct_answer_only_to_medium(_mock_spacy):
    _mock_spacy.return_value = MagicMock()
    rl_cfg = {"answer_flag": "Answer:"}
    checker = TeacherVisualChecker(
        rl_cfg,
        {},
        gpu_id=0,
        visual_config={
            "checker": {"enabled": True},
            "prefetch_ic": False,
            "logging": {"enabled": True, "save_artifacts": False},
        },
    )
    checker.get_answer_reward = MagicMock(return_value=1.0)
    checker.bind_teacher(MagicMock(), MagicMock())
    checker.begin_generate_batch(
        samples=[{}],
        images=["chart.png"],
        questions=["What is the value?"],
        global_step=1,
        output_dir=tempfile.gettempdir(),
    )
    checker._current_sample_idx = 0
    score = checker.get_thinking_reward_prompt(
        "Answer: 37.8",
        "prompt",
        "37.8",
        "hint",
        "chart",
    )
    stats = checker.end_generate_batch()
    assert score == 0.5
    assert stats["visual/checker_medium"] == 1.0
    assert stats["visual/checker_low"] == 0.0


@patch("reward_utils.spacy_model.load_spacy_english")
@patch("reward_utils.visual_checker_teacher.teacher_generate_batched_chunks")
def test_teacher_checker_promotes_correct_bare_answer_fragment_without_answer_flag(
    mock_gen_batch,
    _mock_spacy,
):
    _mock_spacy.return_value = MagicMock()
    mock_gen_batch.return_value = (["high"], 1.0)
    rl_cfg = {"answer_flag": "Answer:"}
    checker = TeacherVisualChecker(
        rl_cfg,
        {},
        gpu_id=0,
        visual_config={
            "checker": {"enabled": True},
            "prefetch_ic": False,
            "logging": {"enabled": True, "save_artifacts": False},
        },
    )
    checker.get_answer_reward = MagicMock(return_value=1.0)
    checker.bind_teacher(MagicMock(), MagicMock())
    checker.begin_generate_batch(
        samples=[{}],
        images=["chart.png"],
        questions=["What is the value?"],
        global_step=1,
        output_dir=tempfile.gettempdir(),
    )
    checker._current_sample_idx = 0
    score = checker.get_thinking_reward_prompt(
        "37.8",
        "prompt",
        "37.8",
        "hint",
        "chart",
    )
    stats = checker.end_generate_batch()
    assert score == 0.5
    assert stats["visual/checker_medium"] == 1.0
    assert stats["visual/checker_low"] == 0.0


def test_ic_text_from_offline_prefers_deplot():
    from data_utils.chart.deplot_pipeline import build_deplot_visual_fact

    sample = {
        "visual_fact_hint": "hint-only text",
        "visual_fact_deplot": build_deplot_visual_fact(
            {"question": "q"}, "Year | Value\n2020 | 10"
        ),
    }
    ic_text, source = ic_text_from_offline_sample(sample)
    assert source == "deplot"
    assert "Year | Value" in ic_text
    assert "hint-only" not in ic_text


def test_extract_visual_facts_uses_offline_deplot_without_teacher():
    from data_utils.chart.deplot_pipeline import build_deplot_visual_fact

    sample = {
        "visual_fact_deplot": build_deplot_visual_fact(
            {"question": "q"}, "Col | Val\nA | 1"
        ),
    }
    ic_text, meta = extract_visual_facts_teacher(
        teacher_model=None,
        processor=None,
        sample=sample,
        question="What?",
        image="a.png",
    )
    assert "Col | Val" in ic_text
    assert meta.get("ic_source") == "offline_deplot"


@patch("reward_utils.visual_ic.teacher_generate_one")
def test_extract_visual_facts_teacher_image_prefers_image_over_offline(mock_teacher_one):
    from data_utils.chart.deplot_pipeline import build_deplot_visual_fact

    mock_teacher_one.return_value = (
        '{"description":"image says 2020 is 10","objects":[]}',
        2.0,
    )
    sample = {
        "visual_fact_deplot": build_deplot_visual_fact(
            {"question": "q"}, "Year | Value\n2020 | 99"
        ),
    }
    ic_text, meta = extract_visual_facts_teacher(
        teacher_model=MagicMock(),
        processor=MagicMock(),
        sample=sample,
        question="What is the 2020 value?",
        image="chart.png",
        ic_source="teacher_image",
    )
    assert "image says 2020 is 10" in ic_text
    assert "2020 | 99" not in ic_text
    assert meta.get("ic_source") == "teacher_image"
    mock_teacher_one.assert_called_once()


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
    assert meta.get("ic_source") == "offline_hint_visual_fact_hint"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
