from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opsd_utils.checkpoint_eval import (
    CheckpointEvaluationPolicy,
    CheckpointEvaluationTriggerCallback,
    apply_checkpoint_evaluation_decision,
)


def test_first_and_strictly_higher_scores_are_the_only_save_events() -> None:
    policy = CheckpointEvaluationPolicy(patience=3)

    first = policy.observe(0.50, step=10)
    lower = policy.observe(0.40, step=20)
    better = policy.observe(0.60, step=30)

    assert (first.should_save, first.reason, first.best_step) == (True, "initial", 10)
    assert (lower.should_save, lower.should_stop, lower.lower_score_streak) == (False, False, 1)
    assert (better.should_save, better.reason, better.best_score, better.best_step) == (
        True,
        "improved",
        0.60,
        30,
    )
    assert policy.state.lower_score_streak == 0


def test_three_strictly_lower_scores_stop_on_fourth_evaluation() -> None:
    policy = CheckpointEvaluationPolicy(patience=3)
    decisions = [policy.observe(score, step=index) for index, score in enumerate((0.5, 0.4, 0.3, 0.2), 1)]

    assert [decision.should_save for decision in decisions] == [True, False, False, False]
    assert [decision.should_stop for decision in decisions] == [False, False, False, True]
    assert decisions[-1].reason == "patience_exhausted"
    assert decisions[-1].best_score == 0.5
    assert decisions[-1].best_step == 1
    assert policy.state.stop_requested is True


def test_tie_resets_low_score_streak_without_overwriting_best() -> None:
    policy = CheckpointEvaluationPolicy(patience=2, tie_policy="reset")
    policy.observe(0.5, step=1)
    policy.observe(0.4, step=2)
    tied = policy.observe(0.5, step=3)
    lower = policy.observe(0.3, step=4)

    assert (tied.should_save, tied.should_stop, tied.reason, tied.lower_score_streak) == (False, False, "tie_reset", 0)
    assert (lower.should_stop, lower.lower_score_streak) == (False, 1)
    assert policy.state.best_step == 1


def test_policy_state_round_trip_preserves_best_and_streak() -> None:
    policy = CheckpointEvaluationPolicy(patience=3)
    policy.observe(0.5, step=10)
    policy.observe(0.4, step=20)

    resumed = CheckpointEvaluationPolicy.from_state_dict(policy.state_dict())
    result = resumed.observe(0.3, step=30)

    assert result.best_score == 0.5
    assert result.best_step == 10
    assert result.lower_score_streak == 2
    assert result.evaluation_count == 3


def test_callback_state_round_trip_restores_policy_for_resume() -> None:
    callback = CheckpointEvaluationTriggerCallback(patience=3, tie_policy="reset")
    callback.policy.observe(0.5, step=10)
    callback.policy.observe(0.4, step=20)

    resumed = CheckpointEvaluationTriggerCallback.from_state(callback.state())
    result = resumed.policy.observe(0.3, step=30)

    assert result.best_score == 0.5
    assert result.best_step == 10
    assert result.lower_score_streak == 2


def test_callback_sidecar_persists_non_improved_streak_and_train_begin_restores_it(tmp_path: Path) -> None:
    # The sidecar may contain lower-score evaluations which occurred after the
    # retained native checkpoint.  Its best step must therefore exist before
    # a resume is allowed to trust it.
    (tmp_path / "checkpoint-10").mkdir()
    callback = CheckpointEvaluationTriggerCallback(output_dir=tmp_path, patience=3)
    callback.policy.observe(0.5, step=10)
    callback.policy.observe(0.4, step=20)  # no checkpoint is written for this score

    metadata_path = callback.persist(is_world_process_zero=True)

    assert metadata_path == tmp_path / "checkpoint_eval_state.json"
    assert metadata_path is not None and metadata_path.exists()
    assert list(tmp_path.glob(".checkpoint_eval_state.json.tmp-*")) == []
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["checkpoint_eval_state"]["state"] == {
        "best_score": 0.5,
        "best_step": 10,
        "lower_score_streak": 1,
        "evaluation_count": 2,
        "stop_requested": False,
    }

    resumed = CheckpointEvaluationTriggerCallback(patience=3)
    control = SimpleNamespace()
    resumed.on_train_begin(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(is_world_process_zero=True),
        control,
    )
    decision = resumed.policy.observe(0.3, step=30)
    assert (decision.best_score, decision.best_step, decision.lower_score_streak) == (0.5, 10, 2)


def test_new_best_is_published_only_after_native_checkpoint_on_save(tmp_path: Path) -> None:
    callback = CheckpointEvaluationTriggerCallback(output_dir=tmp_path)
    decision = callback.policy.observe(0.8, step=10)
    callback.record_checkpoint_evaluation_decision(decision)

    # Evaluation has selected the score, but its checkpoint has not been
    # written yet.  Publishing a sidecar here would make a crash resumable at
    # a model step that does not exist.
    assert callback.state_path is not None
    assert not callback.state_path.exists()
    assert not (tmp_path / "final_checkpoint").exists()

    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    control = SimpleNamespace()
    callback.on_save(None, SimpleNamespace(global_step=10), control)

    assert (tmp_path / "final_checkpoint").resolve() == checkpoint.resolve()
    assert callback.state_path.exists()
    state = json.loads(callback.state_path.read_text(encoding="utf-8"))["checkpoint_eval_state"]["state"]
    assert state == {
        "best_score": 0.8,
        "best_step": 10,
        "lower_score_streak": 0,
        "evaluation_count": 1,
        "stop_requested": False,
    }


def test_terminal_sidecar_state_stops_a_resumed_training_run(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-1").mkdir()
    callback = CheckpointEvaluationTriggerCallback(output_dir=tmp_path, patience=1)
    callback.policy.observe(0.8, step=1)
    terminal = callback.policy.observe(0.7, step=2)
    assert terminal.should_stop is True
    callback.persist(is_world_process_zero=True)

    resumed = CheckpointEvaluationTriggerCallback(patience=1)
    control = SimpleNamespace(should_training_stop=False)
    resumed.on_train_begin(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(is_world_process_zero=True),
        control,
    )

    assert resumed.policy.state.stop_requested is True
    assert control.should_training_stop is True


def test_nonzero_rank_receives_rank_zero_terminal_state_at_train_begin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch_distributed = pytest.importorskip("torch.distributed")
    source = CheckpointEvaluationPolicy(patience=1)
    source.observe(0.8, step=10)
    source.observe(0.7, step=20)
    source_payload = source.state_dict()

    callback = CheckpointEvaluationTriggerCallback(patience=1)
    callback.policy.observe(0.1, step=1)
    received_sources: list[int] = []

    monkeypatch.setattr(torch_distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch_distributed, "is_initialized", lambda: True)

    def fake_broadcast(payload: list[object], src: int) -> None:
        received_sources.append(src)
        if len(received_sources) == 1:
            # Sidecar restore now uses an outcome envelope so a rank-zero
            # parsing error can reach workers through this same collective.
            payload[0] = {
                "ok": True,
                "loaded": True,
                "policy_state": source_payload,
            }
        else:
            # ``on_train_begin`` also synchronizes rank-zero artifact-repair
            # failures after the policy has been restored.
            payload[0] = {"ok": True}

    monkeypatch.setattr(torch_distributed, "broadcast_object_list", fake_broadcast)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_train_begin(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(is_world_process_zero=False),
        control,
    )

    assert received_sources == [0, 0]
    assert callback.policy.state_dict() == source_payload
    assert control.should_training_stop is True


def test_rank_zero_sidecar_load_failure_is_broadcast_before_train_begin_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt sidecar must fail every rank, not strand workers at broadcast."""
    torch_distributed = pytest.importorskip("torch.distributed")
    (tmp_path / "checkpoint_eval_state.json").write_text("{not json", encoding="utf-8")
    callback = CheckpointEvaluationTriggerCallback(output_dir=tmp_path)
    transmitted: list[object] = []

    monkeypatch.setattr(torch_distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch_distributed, "is_initialized", lambda: True)

    def fake_broadcast(payload: list[object], src: int) -> None:
        assert src == 0
        transmitted.append(payload[0])

    monkeypatch.setattr(torch_distributed, "broadcast_object_list", fake_broadcast)

    with pytest.raises(
        RuntimeError,
        match="sidecar policy restore failed on global rank zero \\(ValueError\\)",
    ) as raised:
        callback.on_train_begin(
            SimpleNamespace(output_dir=str(tmp_path)),
            SimpleNamespace(is_world_process_zero=True),
            SimpleNamespace(should_training_stop=False),
        )

    assert isinstance(raised.value.__cause__, ValueError)
    assert transmitted == [
        {
            "ok": False,
            "error_type": "ValueError",
            "error": f"Cannot load checkpoint evaluation metadata from {callback.state_path}",
        }
    ]


def test_worker_raises_rank_zero_sidecar_load_failure_instead_of_continuing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workers consume the rank-zero failure envelope before the next hook."""
    torch_distributed = pytest.importorskip("torch.distributed")
    callback = CheckpointEvaluationTriggerCallback(output_dir=tmp_path)
    received_sources: list[int] = []

    monkeypatch.setattr(torch_distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch_distributed, "is_initialized", lambda: True)

    def fake_broadcast(payload: list[object], src: int) -> None:
        received_sources.append(src)
        payload[0] = {
            "ok": False,
            "error_type": "ValueError",
            "error": "corrupt checkpoint_eval_state.json",
        }

    monkeypatch.setattr(torch_distributed, "broadcast_object_list", fake_broadcast)

    with pytest.raises(
        RuntimeError,
        match="sidecar policy restore failed on global rank zero \\(ValueError\\)",
    ):
        callback.on_train_begin(
            SimpleNamespace(output_dir=str(tmp_path)),
            SimpleNamespace(is_world_process_zero=False),
            SimpleNamespace(should_training_stop=False),
        )

    # It raises at the restore collective, so the later artifact-repair
    # collective is never reached.
    assert received_sources == [0]


def test_rank_zero_artifact_repair_failure_is_broadcast_before_train_begin_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rank-zero publish failures must not leave workers past the repair hook."""
    torch_distributed = pytest.importorskip("torch.distributed")
    callback = CheckpointEvaluationTriggerCallback(output_dir=tmp_path)
    transmitted: list[object] = []

    monkeypatch.setattr(torch_distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch_distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        callback,
        "_recover_committed_checkpoint_artifacts",
        lambda: (_ for _ in ()).throw(OSError("cannot publish final_checkpoint")),
    )

    def fake_broadcast(payload: list[object], src: int) -> None:
        assert src == 0
        transmitted.append(payload[0])

    monkeypatch.setattr(torch_distributed, "broadcast_object_list", fake_broadcast)

    with pytest.raises(
        RuntimeError,
        match="committed checkpoint artifact recovery failed on global rank zero \\(OSError\\)",
    ) as raised:
        callback.on_train_begin(
            SimpleNamespace(output_dir=str(tmp_path)),
            SimpleNamespace(is_world_process_zero=True),
            SimpleNamespace(should_training_stop=False),
        )

    assert isinstance(raised.value.__cause__, OSError)
    # First collective shares the absent-sidecar state; second shares repair
    # failure before rank zero raises.
    assert transmitted == [
        {
            "ok": True,
            "loaded": False,
            "policy_state": callback.policy.state_dict(),
        },
        {
            "ok": False,
            "error_type": "OSError",
            "error": "cannot publish final_checkpoint",
        },
    ]


def test_rank_zero_saved_checkpoint_publication_failure_is_broadcast_before_on_save_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-barrier link/sidecar publication cannot let workers run ahead."""
    torch_distributed = pytest.importorskip("torch.distributed")
    callback = CheckpointEvaluationTriggerCallback(output_dir=tmp_path)
    decision = callback.policy.observe(0.8, step=10)
    callback.record_checkpoint_evaluation_decision(decision)
    transmitted: list[object] = []

    monkeypatch.setattr(torch_distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch_distributed, "is_initialized", lambda: True)
    # The native-write barrier is separately exercised by the existing save
    # lifecycle; keep this test focused on the post-barrier rank-zero action.
    monkeypatch.setattr(callback, "_wait_for_native_checkpoint_write", lambda: None)
    monkeypatch.setattr(
        callback,
        "_commit_saved_checkpoint",
        lambda *, step: (_ for _ in ()).throw(OSError(f"cannot publish step {step}")),
    )

    def fake_broadcast(payload: list[object], src: int) -> None:
        assert src == 0
        transmitted.append(payload[0])

    monkeypatch.setattr(torch_distributed, "broadcast_object_list", fake_broadcast)

    with pytest.raises(
        RuntimeError,
        match="saved checkpoint publication failed on global rank zero \\(OSError\\)",
    ) as raised:
        callback.on_save(None, SimpleNamespace(global_step=10), SimpleNamespace())

    assert isinstance(raised.value.__cause__, OSError)
    assert transmitted == [
        {
            "ok": False,
            "error_type": "OSError",
            "error": "cannot publish step 10",
        }
    ]


def test_callback_sidecar_skips_nonzero_ranks(tmp_path: Path) -> None:
    callback = CheckpointEvaluationTriggerCallback(output_dir=tmp_path)
    callback.policy.observe(0.5, step=1)

    assert callback.persist(is_world_process_zero=False) is None
    assert callback.load(is_world_process_zero=False) is False
    assert not (tmp_path / "checkpoint_eval_state.json").exists()


def test_callback_sidecar_rejects_invalid_metadata(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint_eval_state.json"
    path.write_text("{not json", encoding="utf-8")
    callback = CheckpointEvaluationTriggerCallback(output_dir=tmp_path)

    with pytest.raises(ValueError, match="Cannot load checkpoint evaluation metadata"):
        callback.load()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), "not-a-score"])
def test_non_finite_or_invalid_score_is_rejected_without_mutating_state(score: object) -> None:
    policy = CheckpointEvaluationPolicy()
    with pytest.raises(ValueError):
        policy.observe(score, step=1)
    assert policy.state.evaluation_count == 0
    assert policy.state.best_score is None


def test_trigger_callback_redirects_native_save_to_evaluation_and_can_be_consumed() -> None:
    callback = CheckpointEvaluationTriggerCallback(enabled=True)
    control = SimpleNamespace(should_save=True, should_evaluate=False)

    result = callback.on_step_end(None, SimpleNamespace(global_step=5), control)

    assert result is control
    assert (control.should_save, control.should_evaluate) == (False, True)
    assert callback.consume_checkpoint_evaluation() is True
    assert callback.consume_checkpoint_evaluation() is False
    assert callback.trigger_count == 1


def test_trigger_callback_leaves_non_save_events_alone_and_round_trips_state() -> None:
    callback = CheckpointEvaluationTriggerCallback(enabled=True)
    control = SimpleNamespace(should_save=False, should_evaluate=False)
    callback.on_epoch_end(None, SimpleNamespace(global_step=5), control)
    assert (control.should_save, control.should_evaluate, callback.trigger_count) == (False, False, 0)

    restored = CheckpointEvaluationTriggerCallback.from_state(
        {
            "args": {"enabled": True, "patience": 3, "tie_policy": "reset"},
            "attributes": {
                "pending_checkpoint_evaluation": True,
                "trigger_count": 4,
                "checkpoint_eval_state": {
                    "version": 1,
                    "patience": 3,
                    "tie_policy": "reset",
                    "state": {"best_score": 0.7, "best_step": 9, "lower_score_streak": 1, "evaluation_count": 2},
                },
            },
        }
    )
    assert restored.consume_checkpoint_evaluation() is True
    assert restored.trigger_count == 4
    assert restored.policy.state.best_score == 0.7
    assert restored.policy.state.best_step == 9


def test_apply_decision_vetoes_save_or_requests_clean_early_stop() -> None:
    policy = CheckpointEvaluationPolicy(patience=1)
    policy.observe(0.5, step=1)
    stopped = policy.observe(0.4, step=2)
    control = SimpleNamespace(should_save=True, should_evaluate=True, should_training_stop=False)

    apply_checkpoint_evaluation_decision(control, stopped.to_dict())

    assert (control.should_save, control.should_evaluate, control.should_training_stop) == (False, False, True)
