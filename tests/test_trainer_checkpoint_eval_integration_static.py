"""Dependency-free guards for DyMETrainer's checkpoint-evaluation wiring.

The full trainer imports CUDA/TRL, so these tests inspect its AST/source while
the policy's state transitions are covered independently in
``test_checkpoint_eval_policy.py``.
"""
from __future__ import annotations

import ast
import __future__
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Optional

import pytest

from opsd_utils.checkpoint_eval import CheckpointEvaluationTriggerCallback


SOURCE_PATH = Path(__file__).resolve().parents[1] / "trainer" / "DyMETrainer.py"
CALLBACK_SOURCE_PATH = Path(__file__).resolve().parents[1] / "opsd_utils" / "checkpoint_eval.py"


def _method_source(name: str) -> str:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DyMETrainer"
    )
    method = next(
        node
        for node in trainer.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    text = ast.get_source_segment(source, method)
    assert text is not None
    return text


def _trainer_method_node(name: str) -> tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DyMETrainer"
    )
    method = next(
        node
        for node in trainer.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    return source, method


def _load_trainer_methods(
    names: tuple[str, ...], extra_globals: dict[str, Any] | None = None
) -> type:
    """Compile selected lightweight trainer methods without CUDA/TRL imports."""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    trainer = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DyMETrainer"
    )
    methods = [
        node
        for node in trainer.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    assert {method.name for method in methods} == set(names)
    extracted = ast.ClassDef(
        name="ExtractedDyMETrainer",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(body=[extracted], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Any": Any, "Optional": Optional}
    namespace.update(extra_globals or {})
    exec(
        compile(
            module,
            str(SOURCE_PATH),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        namespace,
    )
    return namespace["ExtractedDyMETrainer"]


def _contains_named_call(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(candidate, ast.Call)
        and (
            (isinstance(candidate.func, ast.Name) and candidate.func.id == name)
            or (isinstance(candidate.func, ast.Attribute) and candidate.func.attr == name)
        )
        for candidate in ast.walk(node)
    )


def test_save_time_evaluate_uses_live_student_and_base_trainer_log() -> None:
    evaluate = _method_source("evaluate")

    # The evaluation core receives the resident student and the wrapper-aware
    # generation context; no checkpoint path/model reload is part of this call.
    assert "model=self.model" in evaluate
    assert "generation_context=self._checkpoint_evaluation_generation_context" in evaluate
    assert "from_pretrained" not in evaluate
    # Avoid DyMETrainer.log(), which clears GRPO training metric buffers.
    assert "Trainer.log(self, metrics)" in evaluate


def test_non_save_time_evaluations_return_before_policy_or_live_generation() -> None:
    evaluate = _method_source("evaluate")
    gate = evaluate.index("save_time_evaluation = self._consume_checkpoint_evaluation_request()")
    non_save_return = evaluate.index("if not save_time_evaluation:", gate)
    live_eval = evaluate.index("result = evaluate_chartqa_in_memory(", non_save_return)
    policy_observe = evaluate.index("self.checkpoint_eval_policy.observe", live_eval)

    assert "return super().evaluate(" in evaluate[non_save_return:live_eval]
    assert policy_observe > live_eval


def test_policy_result_persists_only_without_native_save_then_commits_best_on_save() -> None:
    evaluate = _method_source("evaluate")
    recorded = evaluate.index("callback.record_checkpoint_evaluation_decision(decision)")
    refreshed = evaluate.index("self._refresh_checkpoint_evaluation_callback_state(")
    decision_applied = evaluate.index("apply_checkpoint_evaluation_decision(control, decision)")
    assert recorded < refreshed < decision_applied
    assert 'persist_sidecar=not bool(decision["should_save"])' in evaluate

    refresh_helper = _method_source("_refresh_checkpoint_evaluation_callback_state")
    assert "if persist_sidecar:" in refresh_helper
    assert "callback.persist(" in refresh_helper
    assert "is_world_process_zero" in refresh_helper
    # Sidecar persistence must not depend on Trainer's optional callback-state
    # mapping existing (otherwise non-save scores could be lost on resume).
    assert "if not isinstance(stateful_callbacks, dict):\n            return" not in refresh_helper

    source = CALLBACK_SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    callback = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CheckpointEvaluationTriggerCallback"
    )
    on_save = next(
        node for node in callback.body if isinstance(node, ast.FunctionDef) and node.name == "on_save"
    )
    on_save_source = ast.get_source_segment(source, on_save)
    assert on_save_source is not None
    assert on_save_source.index("self._wait_for_native_checkpoint_write()") < on_save_source.index(
        "self._commit_saved_checkpoint(step=saved_step)"
    )

    commit = next(
        node
        for node in callback.body
        if isinstance(node, ast.FunctionDef) and node.name == "_commit_saved_checkpoint"
    )
    commit_source = ast.get_source_segment(source, commit)
    assert commit_source is not None
    assert "update_final_checkpoint_link(self.output_dir, checkpoint_dir)" in commit_source
    assert "self.persist(is_world_process_zero=True)" in commit_source


def test_rank_zero_policy_observation_failure_is_sent_in_the_existing_decision_broadcast() -> None:
    """A bad rank-zero policy result must not strand worker ranks at broadcast."""
    broadcasts: list[tuple[dict[str, Any] | None, int]] = []

    def fake_broadcast(objects: list[Any], from_process: int = 0) -> None:
        broadcasts.append((objects[0], from_process))

    trainer_type = _load_trainer_methods(
        ("_broadcast_checkpoint_evaluation_payload",),
        {"broadcast_object_list": fake_broadcast},
    )
    trainer = trainer_type()
    trainer.accelerator = SimpleNamespace(is_main_process=True, num_processes=2)

    with pytest.raises(
        RuntimeError,
        match="policy observation failed on global rank zero \\(ValueError\\)",
    ) as raised:
        trainer._broadcast_checkpoint_evaluation_payload(
            None,
            rank_zero_error=ValueError("invalid checkpoint score"),
        )

    assert isinstance(raised.value.__cause__, ValueError)
    assert broadcasts == [
        (
            {
                "ok": False,
                "error_type": "ValueError",
                "error": "invalid checkpoint score",
            },
            0,
        )
    ]


def test_worker_raises_rank_zero_policy_observation_failure_from_decision_broadcast() -> None:
    """A worker receives the same policy failure through that one collective."""
    broadcasts: list[int] = []

    def fake_broadcast(objects: list[Any], from_process: int = 0) -> None:
        broadcasts.append(from_process)
        assert objects == [None]
        objects[0] = {
            "ok": False,
            "error_type": "ValueError",
            "error": "invalid checkpoint score",
        }

    trainer_type = _load_trainer_methods(
        ("_broadcast_checkpoint_evaluation_payload",),
        {"broadcast_object_list": fake_broadcast},
    )
    trainer = trainer_type()
    trainer.accelerator = SimpleNamespace(is_main_process=False, num_processes=2)

    with pytest.raises(
        RuntimeError,
        match="policy observation failed on global rank zero \\(ValueError\\)",
    ):
        trainer._broadcast_checkpoint_evaluation_payload(None)

    assert broadcasts == [0]


def test_evaluate_catches_rank_zero_observe_error_before_its_single_decision_collective() -> None:
    """The live evaluation path wraps observe instead of raising before broadcast."""
    broadcasts: list[tuple[dict[str, Any] | None, int]] = []

    class RaisingPolicy:
        def observe(self, score: float, *, step: int) -> None:
            assert (score, step) == (0.5, 7)
            raise ValueError("invalid checkpoint score")

    class FakeChartQAEvaluationConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def fake_evaluate_chartqa_in_memory(**kwargs: Any) -> dict[str, float]:
        return {"checkpoint_score": 0.5}

    def fake_broadcast(objects: list[Any], from_process: int = 0) -> None:
        broadcasts.append((objects[0], from_process))

    trainer_type = _load_trainer_methods(
        ("_broadcast_checkpoint_evaluation_payload", "evaluate"),
        {
            "ChartQAEvaluationConfig": FakeChartQAEvaluationConfig,
            "evaluate_chartqa_in_memory": fake_evaluate_chartqa_in_memory,
            "broadcast_object_list": fake_broadcast,
        },
    )
    trainer = trainer_type()
    trainer.checkpoint_eval_policy = RaisingPolicy()
    trainer.eval_dataset = ["example"]
    trainer.checkpoint_eval_config = {}
    trainer.accelerator = SimpleNamespace(is_main_process=True, num_processes=2, device=None)
    trainer.state = SimpleNamespace(global_step=7)
    trainer.model = object()
    trainer.processing_class = object()
    trainer._consume_checkpoint_evaluation_request = lambda: True
    trainer._checkpoint_evaluation_generation_context = lambda: None

    with pytest.raises(
        RuntimeError,
        match="policy observation failed on global rank zero \\(ValueError\\)",
    ) as raised:
        trainer.evaluate()

    assert isinstance(raised.value.__cause__, ValueError)
    assert broadcasts == [
        (
            {
                "ok": False,
                "error_type": "ValueError",
                "error": "invalid checkpoint score",
            },
            0,
        )
    ]


def test_non_saving_sidecar_persistence_uses_callback_failure_collective() -> None:
    """Lower/tied score persistence is guarded rather than a direct rank-zero write."""
    refresh_helper = _method_source("_refresh_checkpoint_evaluation_callback_state")
    assert "_run_on_rank_zero_and_broadcast_failure(" in refresh_helper
    assert "non-saving checkpoint evaluation sidecar persistence" in refresh_helper


def test_non_saving_sidecar_persistence_failure_is_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trainer invokes the callback envelope for a rank-zero write error."""
    torch_distributed = pytest.importorskip("torch.distributed")
    broadcasts: list[tuple[dict[str, Any] | None, int]] = []
    callback = CheckpointEvaluationTriggerCallback()
    callback._is_world_process_zero = True

    def fail_persist(*, is_world_process_zero: bool | None = None) -> None:
        assert is_world_process_zero is True
        raise OSError("cannot write checkpoint_eval_state.json")

    def fake_broadcast(payload: list[Any], src: int) -> None:
        broadcasts.append((payload[0], src))

    monkeypatch.setattr(callback, "persist", fail_persist)
    monkeypatch.setattr(torch_distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch_distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch_distributed, "broadcast_object_list", fake_broadcast)

    trainer_type = _load_trainer_methods(
        ("_refresh_checkpoint_evaluation_callback_state",),
        {"CheckpointEvaluationTriggerCallback": CheckpointEvaluationTriggerCallback},
    )
    trainer = trainer_type()
    trainer.state = SimpleNamespace(stateful_callbacks={})
    trainer.callback_handler = SimpleNamespace(callbacks=[callback])
    trainer.accelerator = SimpleNamespace(is_main_process=True)

    with pytest.raises(
        RuntimeError,
        match="non-saving checkpoint evaluation sidecar persistence failed on global rank zero \\(OSError\\)",
    ) as raised:
        trainer._refresh_checkpoint_evaluation_callback_state(persist_sidecar=True)

    assert isinstance(raised.value.__cause__, OSError)
    assert broadcasts == [
        (
            {
                "ok": False,
                "error_type": "OSError",
                "error": "cannot write checkpoint_eval_state.json",
            },
            0,
        )
    ]


def test_generation_context_unwraps_the_current_wrapped_student() -> None:
    context = _method_source("_checkpoint_evaluation_generation_context")
    assert "unwrap_model_for_generation(" in context
    assert "self.model_wrapped" in context
    assert "self.accelerator" in context
    assert "FSDP.summon_full_params" in context


def test_terminal_resume_preflight_returns_before_parent_training_loop() -> None:
    """A terminal sidecar/checkpoint must not enter HF's training lifecycle.

    ``Callback.on_train_begin`` occurs inside ``Trainer.train`` and is too
    late to prevent some setup/training work.  Keep a structural guard here
    so a terminal checkpoint-eval policy is checked before the parent loop;
    the terminal path returns a valid ``TrainOutput``, while the normal path
    still delegates to the parent implementation.
    """
    _, preflight = _trainer_method_node("_checkpoint_evaluation_preflight_should_stop")
    preflight_source = _method_source("_checkpoint_evaluation_preflight_should_stop")
    assert "resume_from_checkpoint" in preflight_source
    assert "CheckpointEvaluationTriggerCallback" in preflight_source
    assert "stop_requested" in preflight_source
    # A newer terminal result can live only in the checkpoint-eval sidecar,
    # rather than the last native Trainer checkpoint.  Restore uses the
    # failure-envelope helper so a rank-zero read error is collective too.
    assert "_load_and_broadcast_policy_state_from_world_zero" in preflight_source
    assert "callback.load(" not in preflight_source

    _, train = _trainer_method_node("train")
    preflight_calls = [
        call
        for call in ast.walk(train)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
        and call.func.attr == "_checkpoint_evaluation_preflight_should_stop"
    ]
    assert len(preflight_calls) == 1
    preflight_call = preflight_calls[0]

    parent_train_calls = [
        call
        for call in ast.walk(train)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "train"
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Name)
        and call.func.value.func.id == "super"
    ]
    # The unchanged, non-terminal behavior has exactly one parent-loop
    # delegate, and it is lexically after the preflight decision.
    assert len(parent_train_calls) == 1
    parent_train_call = parent_train_calls[0]
    assert preflight_call.lineno < parent_train_call.lineno

    terminal_returns = [
        returned
        for returned in ast.walk(train)
        if isinstance(returned, ast.Return)
        and returned.lineno < parent_train_call.lineno
        and returned.value is not None
        and _contains_named_call(returned.value, "TrainOutput")
    ]
    assert terminal_returns, "terminal preflight branch must return TrainOutput before super().train"

    parent_returns = [
        returned
        for returned in ast.walk(train)
        if isinstance(returned, ast.Return)
        and returned.value is parent_train_call
    ]
    assert len(parent_returns) == 1
