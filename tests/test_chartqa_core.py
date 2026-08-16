from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval.chartqa_core as chartqa_core  # noqa: E402
from eval.chartqa_core import (  # noqa: E402
    ChartQAEvaluationConfig,
    evaluate_chartqa_in_memory,
    print_chartqa_evaluation,
    prepare_chartqa_examples,
    shard_chartqa_examples,
)


class _Accelerator:
    num_processes = 1
    process_index = 0
    is_main_process = True
    device = None


class _DistributedAccelerator:
    num_processes = 2
    process_index = 1
    is_main_process = False
    device = None


class _Tokenizer:
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 1


class _Processor:
    tokenizer = _Tokenizer()


class _TensorProcessor(_Processor):
    def __init__(self, torch) -> None:
        self.torch = torch
        self.tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, add_generation_prompt: bool):
        assert add_generation_prompt is True
        assert messages[0]["content"][0]["type"] == "image"
        return "prompt"

    def __call__(self, *, text, images, return_tensors, padding, truncation):
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is True
        assert len(text) == len(images)
        return {
            "input_ids": self.torch.ones((len(text), 2), dtype=self.torch.long),
            "pixel_values": self.torch.ones((len(text), 1), dtype=self.torch.float32),
        }

    def batch_decode(self, token_ids, skip_special_tokens: bool):
        assert skip_special_tokens is True
        return ["Answer: 1"] * token_ids.shape[0]


class _Model:
    training = True

    def __init__(self) -> None:
        self.eval_calls = 0
        self.train_calls: list[bool] = []

    def eval(self) -> None:
        self.eval_calls += 1
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.train_calls.append(mode)
        self.training = mode


class _GeneratingModel(_Model):
    def __init__(self, torch) -> None:
        super().__init__()
        self.torch = torch
        self.dtype = torch.float32
        self.generate_calls: list[dict] = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        completion = self.torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype)
        return self.torch.cat((input_ids, completion), dim=1)


class _FailingTensorProcessor(_TensorProcessor):
    def __call__(self, **kwargs):
        raise OSError("rank-local image/processor preparation failed")


def _install_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = ModuleType("data_utils.chart.evaluator")
    evaluator.eval_one_chart = lambda prediction, answer: float(prediction == answer)
    monkeypatch.setitem(sys.modules, "data_utils.chart.evaluator", evaluator)


def test_prepare_chartqa_examples_accepts_raw_and_prepared_rows() -> None:
    image = _Image()
    rows = [
        {"image": image, "query": "raw?", "label": ["1"]},
        {"image_path": image, "model_input_text": "prepared?", "answer": "2"},
        {"image": image, "query": "missing label", "label": []},
    ]

    prepared = prepare_chartqa_examples(rows)

    assert [row["answer"] for row in prepared] == ["1", "2"]
    assert [row["model_input_text"] for row in prepared] == ["raw?", "prepared?"]


def test_eval_discards_training_hint_and_answer_fields_before_model_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation must use only the public ChartQA image/question/label trio."""
    torch = pytest.importorskip("torch")

    class _CapturingProcessor(_TensorProcessor):
        def __init__(self) -> None:
            super().__init__(torch)
            self.messages = []

        def apply_chat_template(self, messages, add_generation_prompt: bool):
            self.messages.extend(messages)
            return super().apply_chat_template(messages, add_generation_prompt)

    private_hint = "PRIVATE_HINT_70_DO_NOT_SHOW_TO_EVAL"
    private_answer = "PRIVATE_SFT_TARGET_70_DO_NOT_SHOW_TO_EVAL"
    prepared = prepare_chartqa_examples(
        [
            {
                "image": _Image(),
                "query": "What is publicly asked?",
                "label": ["70"],
                "hint": private_hint,
                "answer": private_answer,
                "visual_fact_hint": private_hint,
                "visual_fact_deplot": private_hint,
            }
        ]
    )
    assert prepared == [
        {
            "image_path": prepared[0]["image_path"],
            "model_input_text": "What is publicly asked?",
            "answer": "70",
            "original_question": "What is publicly asked?",
        }
    ]

    processor = _CapturingProcessor()
    monkeypatch.setattr(chartqa_core, "_load_rgb_image", lambda value: value)
    chartqa_core.build_chartqa_batch_inputs(
        processor,
        prepared,
        device=None,
        input_dtype=None,
        prompt_template="Question: {question}",
    )
    rendered = str(processor.messages)
    assert "What is publicly asked?" in rendered
    assert private_hint not in rendered
    assert private_answer not in rendered


def test_shard_chartqa_examples_is_contiguous_and_exact() -> None:
    rows = [{"id": index} for index in range(10)]

    shards = [
        shard_chartqa_examples(rows, num_processes=3, process_index=rank)
        for rank in range(3)
    ]

    assert [[item["id"] for item in shard] for shard in shards] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]


def test_in_memory_evaluation_uses_live_model_and_restores_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_metric(monkeypatch)
    model = _Model()
    processor = _Processor()
    seen_models: list[object] = []

    @contextmanager
    def generation_context():
        yielded_model = object()
        seen_models.append(yielded_model)
        yield yielded_model

    def generate(generation_model, passed_processor, batch, config):
        assert generation_model is seen_models[0]
        assert passed_processor is processor
        assert processor.tokenizer.padding_side == "left"
        return ["reasoning\nAnswer: 1", "Answer: 2"][: len(batch)]

    image = _Image()
    result = evaluate_chartqa_in_memory(
        model=model,
        processor=processor,
        accelerator=_Accelerator(),
        dataset=[
            {"image": image, "query": "q1", "label": ["1"]},
            {"image": image, "query": "q2", "label": ["2"]},
        ],
        generation_context=generation_context,
        config=ChartQAEvaluationConfig(batch_size=2, batch_generator=generate),
    )

    assert result["accuracy"] == 1.0
    assert result["checkpoint_score"] == 1.0
    assert result["sample_count"] == 2
    assert result["output_type_counts"]["other"] == 2
    assert model.training is True
    assert model.train_calls == [True]
    assert processor.tokenizer.padding_side == "right"


def test_default_generation_path_uses_the_passed_live_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    _install_metric(monkeypatch)
    model = _GeneratingModel(torch)
    processor = _TensorProcessor(torch)

    result = evaluate_chartqa_in_memory(
        model=model,
        processor=processor,
        accelerator=_Accelerator(),
        dataset=[{"image": _Image(), "query": "q", "label": ["1"]}],
        config=ChartQAEvaluationConfig(batch_size=1, prompt_template="Q: {question}"),
    )

    assert result["accuracy"] == 1.0
    assert len(model.generate_calls) == 1
    call = model.generate_calls[0]
    assert call["do_sample"] is False
    assert call["repetition_penalty"] == 1.0
    assert call["pixel_values"].dtype == torch.float32


def test_single_process_batch_preparation_failure_stays_local_and_does_not_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The safety envelope adds no collective work to the normal one-GPU path."""
    torch = pytest.importorskip("torch")
    _install_metric(monkeypatch)
    model = _GeneratingModel(torch)

    def should_not_gather(*_args, **_kwargs):
        raise AssertionError("single-process evaluation must not gather readiness records")

    monkeypatch.setattr(chartqa_core, "_gather_batch_preparation_statuses", should_not_gather)
    with pytest.raises(OSError, match="rank-local image/processor preparation failed"):
        evaluate_chartqa_in_memory(
            model=model,
            processor=_FailingTensorProcessor(torch),
            accelerator=_Accelerator(),
            dataset=[{"image": _Image(), "query": "q", "label": ["1"]}],
            config=ChartQAEvaluationConfig(batch_size=1, prompt_template="Q: {question}"),
        )

    assert model.generate_calls == []


def test_distributed_worker_receives_remote_preparation_failure_before_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy worker exits from readiness, never entering collective generate."""
    torch = pytest.importorskip("torch")
    _install_metric(monkeypatch)
    model = _GeneratingModel(torch)
    observed_statuses: list[dict[str, object]] = []

    def gather_remote_failure(_accelerator, local_status):
        observed_statuses.append(dict(local_status))
        return [
            {
                "rank": 0,
                "ok": False,
                "error_type": "OSError",
                "error": "cannot read chart image",
            },
            dict(local_status),
        ]

    monkeypatch.setattr(chartqa_core, "_gather_batch_preparation_statuses", gather_remote_failure)
    with pytest.raises(
        RuntimeError,
        match=r"batch preparation failed before distributed generation .*batch=0, rank=0, OSError: cannot read chart image",
    ) as raised:
        evaluate_chartqa_in_memory(
            model=model,
            processor=_TensorProcessor(torch),
            accelerator=_DistributedAccelerator(),
            dataset=[
                {"image": _Image(), "query": "q1", "label": ["1"]},
                {"image": _Image(), "query": "q2", "label": ["1"]},
            ],
            config=ChartQAEvaluationConfig(batch_size=1, prompt_template="Q: {question}"),
        )

    assert raised.value.__cause__ is None
    assert observed_statuses == [{"rank": 1, "ok": True}]
    assert model.generate_calls == []


def test_distributed_local_preparation_failure_is_shared_before_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failing rank keeps its cause while peers receive the same envelope."""
    torch = pytest.importorskip("torch")
    original = OSError("cannot open local chart image")
    captured: list[dict[str, object]] = []

    def gather_local_failure(_accelerator, local_status):
        captured.append(dict(local_status))
        return [
            {
                "rank": 0,
                "ok": False,
                "error_type": "OSError",
                "error": "cannot open local chart image",
            },
            {"rank": 1, "ok": True},
        ]

    monkeypatch.setattr(chartqa_core, "_gather_batch_preparation_statuses", gather_local_failure)
    failing_accelerator = _DistributedAccelerator()
    failing_accelerator.process_index = 0
    failing_accelerator.is_main_process = True
    with pytest.raises(RuntimeError, match="rank=0, OSError: cannot open local chart image") as raised:
        chartqa_core._synchronize_batch_preparation(
            failing_accelerator,
            batch_index=3,
            local_error=original,
        )

    assert raised.value.__cause__ is original
    assert captured == [
        {
            "rank": 0,
            "ok": False,
            "error_type": "OSError",
            "error": "cannot open local chart image",
        }
    ]


def test_distributed_worker_receives_remote_scoring_failure_after_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-generation failure is shared before the next collective/reduce."""
    _install_metric(monkeypatch)
    gather_calls: list[dict[str, object]] = []

    def gather_remote_scoring_failure(_accelerator, local_status):
        gather_calls.append(dict(local_status))
        if len(gather_calls) == 1:
            # The pre-generation readiness exchange succeeds on both ranks.
            return [{"rank": 0, "ok": True}, dict(local_status)]
        return [
            {
                "rank": 0,
                "ok": False,
                "error_type": "ValueError",
                "error": "ChartQA metric rejected rank-zero prediction",
            },
            dict(local_status),
        ]

    monkeypatch.setattr(chartqa_core, "_gather_batch_preparation_statuses", gather_remote_scoring_failure)
    reduced = False

    def should_not_reduce(*_args, **_kwargs):
        nonlocal reduced
        reduced = True
        raise AssertionError("post-generation failure must stop before metric reduction")

    monkeypatch.setattr(chartqa_core, "_reduce_score_count", should_not_reduce)
    with pytest.raises(
        RuntimeError,
        match=(
            r"batch generation/scoring failed during distributed evaluation "
            r"\(batch=0, rank=0, ValueError: ChartQA metric rejected rank-zero prediction\)"
        ),
    ):
        evaluate_chartqa_in_memory(
            model=_Model(),
            processor=_Processor(),
            accelerator=_DistributedAccelerator(),
            dataset=[
                {"image": _Image(), "query": "q1", "label": ["1"]},
                {"image": _Image(), "query": "q2", "label": ["1"]},
            ],
            config=ChartQAEvaluationConfig(
                batch_size=1,
                batch_generator=lambda *_args: ["Answer: 1"],
            ),
        )

    assert gather_calls == [{"rank": 1, "ok": True}, {"rank": 1, "ok": True}]
    assert reduced is False


def test_in_memory_evaluation_pads_partial_batches_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_metric(monkeypatch)
    seen_batch_sizes: list[int] = []

    def generate(_model, _processor, batch, _config):
        seen_batch_sizes.append(len(batch))
        return ["Answer: 1"] * len(batch)

    image = _Image()
    result = evaluate_chartqa_in_memory(
        model=_Model(),
        processor=_Processor(),
        accelerator=_Accelerator(),
        dataset=[
            {"image": image, "query": "q1", "label": ["1"]},
            {"image": image, "query": "q2", "label": ["1"]},
            {"image": image, "query": "q3", "label": ["1"]},
        ],
        config=ChartQAEvaluationConfig(batch_size=2, batch_generator=generate),
    )

    assert seen_batch_sizes == [2, 2]
    assert result["sample_count"] == 3
    assert result["local_dummy_batches"] == 1


def test_in_memory_evaluation_rejects_empty_valid_dataset() -> None:
    with pytest.raises(ValueError, match="no valid examples"):
        evaluate_chartqa_in_memory(
            model=_Model(),
            processor=_Processor(),
            accelerator=_Accelerator(),
            dataset=[{"query": "missing image", "label": ["1"]}],
            config=ChartQAEvaluationConfig(batch_generator=lambda *_args: []),
        )


def test_final_summary_keeps_legacy_completed_evaluation_marker() -> None:
    lines: list[str] = []

    print_chartqa_evaluation(
        {
            "sample_count": 2,
            "total_items": 2,
            "accuracy": 1.0,
            "output_type_counts": {"other": 2},
            "template_behavior_counts": {"total": 2},
        },
        print_fn=lines.append,
    )

    assert lines[0] == "--- Final Report ---"
    assert lines[1] == "Global samples processed: 2 / 2"


class _Image:
    def convert(self, mode: str):
        assert mode == "RGB"
        return self
