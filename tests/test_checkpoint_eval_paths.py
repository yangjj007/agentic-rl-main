from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from opsd_utils.checkpoint_eval_paths import (
    find_best_checkpoint_path,
    find_checkpoint_evaluation_policy,
    recover_interrupted_checkpoint_eval_save,
    update_final_checkpoint_link,
    validate_checkpoint_eval_output_dir,
)
from opsd_utils.checkpoint_eval import CheckpointEvaluationTriggerCallback


class _State:
    best_model_checkpoint = None


class _Trainer:
    state = _State()


def _checkpoint_eval_policy_state(
    *,
    step: int,
    score: float,
    evaluation_count: int,
    lower_score_streak: int = 0,
    stop_requested: bool = False,
) -> dict[str, Any]:
    """Build the callback payload HF serializes into trainer_state.json."""
    return {
        "version": 1,
        "patience": 3,
        "tie_policy": "reset",
        "state": {
            "best_score": score,
            "best_step": step,
            "lower_score_streak": lower_score_streak,
            "evaluation_count": evaluation_count,
            "stop_requested": stop_requested,
        },
    }


def _write_native_checkpoint_state(
    checkpoint: Path,
    *,
    step: int,
    policy: dict[str, Any],
    best_global_step: int | None = None,
) -> None:
    """Write only the durable state needed to prove a native HF save."""
    checkpoint.mkdir(exist_ok=True)
    state = {
        "global_step": step,
        "best_global_step": step if best_global_step is None else best_global_step,
        "stateful_callbacks": {
            "CheckpointEvaluationTriggerCallback": {
                "args": {"enabled": True, "patience": 3, "tie_policy": "reset"},
                "attributes": {
                    "pending_checkpoint_evaluation": False,
                    "trigger_count": policy["state"]["evaluation_count"],
                    "checkpoint_eval_state": policy,
                },
            }
        },
    }
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
    )


def _interrupted_native_save_layout(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Model HF's write-complete / rotation-not-yet-started crash window."""
    old_checkpoint = tmp_path / "checkpoint-100"
    new_checkpoint = tmp_path / "checkpoint-200"
    old_policy = _checkpoint_eval_policy_state(
        step=100,
        score=0.60,
        evaluation_count=1,
    )
    new_policy = _checkpoint_eval_policy_state(
        step=200,
        score=0.75,
        evaluation_count=2,
    )
    _write_native_checkpoint_state(old_checkpoint, step=100, policy=old_policy)
    # The callback's prior successful save is the durable pointer evidence
    # that checkpoint-100 was the retained best before native rotation.
    os.symlink(old_checkpoint.name, tmp_path / "final_checkpoint")
    _write_native_checkpoint_state(new_checkpoint, step=200, policy=new_policy)
    return old_checkpoint, new_checkpoint, old_policy, new_policy


def _rotated_before_callback_publication_layout(
    tmp_path: Path,
    *,
    with_stale_sidecar: bool = True,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Model HF rotation completing immediately before callback ``on_save``.

    This is the narrow single-directory crash state: HF has already deleted
    the old best but the callback has not yet repointed ``final_checkpoint``
    or published the new policy sidecar.
    """
    old_checkpoint, new_checkpoint, old_policy, new_policy = _interrupted_native_save_layout(
        tmp_path
    )
    if with_stale_sidecar:
        (tmp_path / "checkpoint_eval_state.json").write_text(
            json.dumps({"version": 1, "checkpoint_eval_state": old_policy}),
            encoding="utf-8",
        )
    shutil.rmtree(old_checkpoint)
    assert (tmp_path / "final_checkpoint").is_symlink()
    assert not (tmp_path / "final_checkpoint").exists()
    return old_checkpoint, new_checkpoint, old_policy, new_policy


def test_update_final_checkpoint_link_uses_relative_atomic_symlink(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")

    final = update_final_checkpoint_link(tmp_path, checkpoint)

    assert final.is_symlink()
    assert not Path(os.readlink(final)).is_absolute()
    assert final.resolve() == checkpoint.resolve()
    assert list(tmp_path.glob(".final_checkpoint.tmp-*")) == []


def test_find_best_checkpoint_path_uses_trainer_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    trainer = _Trainer()
    trainer.state.best_model_checkpoint = str(checkpoint)

    assert find_best_checkpoint_path(trainer, tmp_path) == checkpoint.resolve()


def test_find_best_checkpoint_path_uses_resumed_callback_policy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-9"
    checkpoint.mkdir()
    trainer = _Trainer()
    callback = CheckpointEvaluationTriggerCallback(output_dir=tmp_path)
    callback.policy.observe(0.8, step=9)

    class _Handler:
        callbacks = [callback]

    trainer.callback_handler = _Handler()
    assert find_best_checkpoint_path(trainer, tmp_path) == checkpoint.resolve()


def test_existing_ambiguous_layout_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-1").mkdir()
    (tmp_path / "checkpoint-2").mkdir()
    with pytest.raises(ValueError, match="multiple checkpoint"):
        validate_checkpoint_eval_output_dir(tmp_path)

    (tmp_path / "checkpoint-2").rmdir()
    (tmp_path / "final_checkpoint").mkdir()
    with pytest.raises(ValueError, match="real final_checkpoint"):
        validate_checkpoint_eval_output_dir(tmp_path)


def test_dangling_internal_best_link_can_be_replaced_after_checkpoint_rotation(tmp_path: Path) -> None:
    old_checkpoint = tmp_path / "checkpoint-1"
    old_checkpoint.mkdir()
    update_final_checkpoint_link(tmp_path, old_checkpoint)

    # With save_total_limit=1, native Trainer rotation can remove the old
    # checkpoint before the callback's on_save gets a chance to point the
    # compatibility link at the new one.
    old_checkpoint.rmdir()
    new_checkpoint = tmp_path / "checkpoint-2"
    new_checkpoint.mkdir()

    assert (tmp_path / "final_checkpoint").is_symlink()
    assert not (tmp_path / "final_checkpoint").exists()
    validate_checkpoint_eval_output_dir(tmp_path)
    final = update_final_checkpoint_link(tmp_path, new_checkpoint)

    assert final.resolve() == new_checkpoint.resolve()


def test_recover_post_rotation_dangling_link_uses_retained_native_policy(
    tmp_path: Path,
) -> None:
    old_checkpoint, new_checkpoint, old_policy, new_policy = (
        _rotated_before_callback_publication_layout(tmp_path)
    )

    retained = recover_interrupted_checkpoint_eval_save(
        tmp_path, patience=3, tie_policy="reset"
    )

    assert retained == new_checkpoint.resolve()
    assert not old_checkpoint.exists()
    final = tmp_path / "final_checkpoint"
    assert final.is_symlink()
    assert final.resolve() == new_checkpoint.resolve()
    assert os.readlink(final) == new_checkpoint.name
    sidecar = json.loads((tmp_path / "checkpoint_eval_state.json").read_text(encoding="utf-8"))
    assert sidecar == {"version": 1, "checkpoint_eval_state": new_policy}
    assert sidecar["checkpoint_eval_state"] != old_policy
    validate_checkpoint_eval_output_dir(tmp_path)


def test_recover_post_rotation_rejects_missing_sidecar_without_old_policy_proof(
    tmp_path: Path,
) -> None:
    _, new_checkpoint, _, new_policy = _rotated_before_callback_publication_layout(
        tmp_path, with_stale_sidecar=False
    )

    assert recover_interrupted_checkpoint_eval_save(tmp_path) is None
    final = tmp_path / "final_checkpoint"
    assert final.is_symlink()
    assert not final.exists()
    assert new_checkpoint.is_dir()
    assert not (tmp_path / "checkpoint_eval_state.json").exists()


def test_post_rotation_recovery_rejects_unproven_or_dangerous_artifacts(tmp_path: Path) -> None:
    old_checkpoint, new_checkpoint, _, new_policy = _rotated_before_callback_publication_layout(
        tmp_path
    )
    # A retained directory name is never enough: its policy must nominate and
    # prove this exact saved step.
    new_policy["state"]["best_step"] = 100
    _write_native_checkpoint_state(new_checkpoint, step=200, policy=new_policy)

    assert recover_interrupted_checkpoint_eval_save(tmp_path) is None
    final = tmp_path / "final_checkpoint"
    assert final.is_symlink()
    assert os.readlink(final) == old_checkpoint.name
    assert not final.exists()

    # Restore the valid native proof, then turn the sidecar into an external
    # symlink.  Recovery must neither follow nor replace it.
    new_policy["state"]["best_step"] = 200
    _write_native_checkpoint_state(new_checkpoint, step=200, policy=new_policy)
    external_sidecar = tmp_path / "external-sidecar.json"
    external_sidecar.write_text("do not replace", encoding="utf-8")
    sidecar = tmp_path / "checkpoint_eval_state.json"
    sidecar.unlink()
    sidecar.symlink_to(external_sidecar)

    assert recover_interrupted_checkpoint_eval_save(tmp_path) is None
    assert sidecar.is_symlink()
    assert external_sidecar.read_text(encoding="utf-8") == "do not replace"
    assert os.readlink(final) == old_checkpoint.name


@pytest.mark.parametrize(
    "target", ("../outside", "checkpoint-100/../checkpoint-100", "checkpoint-0100")
)
def test_post_rotation_recovery_never_replaces_noncanonical_final_link(
    tmp_path: Path, target: str
) -> None:
    _, new_checkpoint, _, _ = _rotated_before_callback_publication_layout(tmp_path)
    final = tmp_path / "final_checkpoint"
    final.unlink()
    final.symlink_to(target)

    assert recover_interrupted_checkpoint_eval_save(tmp_path) is None
    assert final.is_symlink()
    assert os.readlink(final) == target
    assert new_checkpoint.is_dir()


def test_post_rotation_recovery_requires_second_evaluation(
    tmp_path: Path,
) -> None:
    _, new_checkpoint, _, new_policy = _rotated_before_callback_publication_layout(tmp_path)
    new_policy["state"]["evaluation_count"] = 1
    _write_native_checkpoint_state(new_checkpoint, step=200, policy=new_policy)

    assert recover_interrupted_checkpoint_eval_save(tmp_path) is None
    final = tmp_path / "final_checkpoint"
    assert final.is_symlink()
    assert not final.exists()
    sidecar = json.loads((tmp_path / "checkpoint_eval_state.json").read_text(encoding="utf-8"))
    assert sidecar["checkpoint_eval_state"]["state"]["best_step"] == 100


def test_recover_interrupted_native_save_uses_trainer_state_proof_before_pruning(
    tmp_path: Path,
) -> None:
    old_checkpoint, new_checkpoint, _, new_policy = _interrupted_native_save_layout(tmp_path)

    retained = recover_interrupted_checkpoint_eval_save(tmp_path)

    assert retained == new_checkpoint.resolve()
    assert not old_checkpoint.exists()
    assert new_checkpoint.is_dir()
    final = tmp_path / "final_checkpoint"
    assert final.is_symlink()
    assert final.resolve() == new_checkpoint.resolve()
    sidecar = json.loads((tmp_path / "checkpoint_eval_state.json").read_text(encoding="utf-8"))
    assert sidecar == {"version": 1, "checkpoint_eval_state": new_policy}

    # Validation is intentionally the recovery entrypoint used by main, and
    # after the proof-driven repair it sees the normal one-checkpoint layout.
    validate_checkpoint_eval_output_dir(tmp_path)
    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == ["checkpoint-200"]


def test_validation_is_read_only_until_rank_zero_recovery_has_proven_the_save(
    tmp_path: Path,
) -> None:
    old_checkpoint, new_checkpoint, _, _ = _interrupted_native_save_layout(tmp_path)

    # main invokes recovery on rank zero and synchronizes before validation.
    # Validation itself must remain non-destructive: it may reject the two-dir
    # window, but it must never decide which model to delete on every rank.
    with pytest.raises(ValueError, match="multiple checkpoint"):
        validate_checkpoint_eval_output_dir(tmp_path)
    assert old_checkpoint.is_dir()
    assert new_checkpoint.is_dir()
    assert (tmp_path / "final_checkpoint").resolve() == old_checkpoint.resolve()

    assert recover_interrupted_checkpoint_eval_save(tmp_path) == new_checkpoint.resolve()
    validate_checkpoint_eval_output_dir(tmp_path)
    assert not old_checkpoint.exists()
    assert new_checkpoint.is_dir()


def test_interrupted_save_with_mismatched_new_policy_is_rejected_without_deletion(
    tmp_path: Path,
) -> None:
    old_checkpoint, new_checkpoint, _, new_policy = _interrupted_native_save_layout(tmp_path)
    # Directory names alone are never enough evidence: make the new callback
    # policy nominate the old step despite global_step/best_global_step=200.
    new_policy["state"]["best_step"] = 100
    _write_native_checkpoint_state(new_checkpoint, step=200, policy=new_policy)

    assert recover_interrupted_checkpoint_eval_save(tmp_path) is None
    assert old_checkpoint.is_dir()
    assert new_checkpoint.is_dir()
    assert (tmp_path / "final_checkpoint").resolve() == old_checkpoint.resolve()
    assert not (tmp_path / "checkpoint_eval_state.json").exists()

    with pytest.raises(ValueError, match="multiple checkpoint"):
        validate_checkpoint_eval_output_dir(tmp_path)
    # validate must not turn ambiguous evidence into a destructive repair.
    assert old_checkpoint.is_dir()
    assert new_checkpoint.is_dir()
    assert (tmp_path / "final_checkpoint").resolve() == old_checkpoint.resolve()


def test_interrupted_save_rejects_checkpoint_directory_symlinks(tmp_path: Path) -> None:
    old_checkpoint, new_checkpoint, _, _ = _interrupted_native_save_layout(tmp_path)
    external_root = tmp_path / "external"
    external_root.mkdir()
    external_new = external_root / "checkpoint-200"
    new_checkpoint.rename(external_new)
    new_checkpoint.symlink_to(external_new, target_is_directory=True)

    # A directory symlink must not become an escape hatch for recovery's
    # destructive prune or for a final_checkpoint link outside output_dir.
    assert recover_interrupted_checkpoint_eval_save(tmp_path) is None
    assert old_checkpoint.is_dir()
    assert new_checkpoint.is_symlink()
    assert external_new.is_dir()
    assert (tmp_path / "final_checkpoint").resolve() == old_checkpoint.resolve()


def test_interrupted_save_rejects_incompatible_serialized_policy(tmp_path: Path) -> None:
    old_checkpoint, new_checkpoint, _, new_policy = _interrupted_native_save_layout(tmp_path)
    new_policy["patience"] = 0
    _write_native_checkpoint_state(new_checkpoint, step=200, policy=new_policy)

    assert recover_interrupted_checkpoint_eval_save(tmp_path) is None
    assert old_checkpoint.is_dir()
    assert new_checkpoint.is_dir()
    assert (tmp_path / "final_checkpoint").resolve() == old_checkpoint.resolve()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("best_score", 0.60),
        ("best_score", 0.50),
        ("patience", 2),
        ("tie_policy", "ignore"),
    ],
)
def test_interrupted_save_requires_strictly_better_compatible_new_policy(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    old_checkpoint, new_checkpoint, _, new_policy = _interrupted_native_save_layout(tmp_path)
    if field in {"best_score"}:
        new_policy["state"][field] = value
    else:
        new_policy[field] = value
    _write_native_checkpoint_state(new_checkpoint, step=200, policy=new_policy)

    assert recover_interrupted_checkpoint_eval_save(
        tmp_path, patience=3, tie_policy="reset"
    ) is None
    assert old_checkpoint.is_dir()
    assert new_checkpoint.is_dir()
    assert (tmp_path / "final_checkpoint").resolve() == old_checkpoint.resolve()


def test_interrupted_save_prefers_newer_sidecar_baseline_before_pruning(tmp_path: Path) -> None:
    old_checkpoint, new_checkpoint, _, _ = _interrupted_native_save_layout(tmp_path)
    # A lower evaluation after checkpoint-100 has no native checkpoint, but
    # its sidecar is the current committed policy baseline.  The prospective
    # score must still beat that score, not merely the old trainer-state score.
    sidecar_policy = _checkpoint_eval_policy_state(
        step=100,
        score=0.80,
        evaluation_count=2,
        lower_score_streak=1,
    )
    (tmp_path / "checkpoint_eval_state.json").write_text(
        json.dumps({"version": 1, "checkpoint_eval_state": sidecar_policy}),
        encoding="utf-8",
    )

    assert recover_interrupted_checkpoint_eval_save(tmp_path) is None
    assert old_checkpoint.is_dir()
    assert new_checkpoint.is_dir()
    assert (tmp_path / "final_checkpoint").resolve() == old_checkpoint.resolve()


def test_find_checkpoint_evaluation_policy_prefers_live_callback_after_restore() -> None:
    trainer = _Trainer()
    stale_policy = CheckpointEvaluationTriggerCallback().policy
    active_callback = CheckpointEvaluationTriggerCallback()
    active_callback.policy.observe(0.9, step=4)
    trainer.checkpoint_eval_policy = stale_policy

    class _Handler:
        callbacks = [active_callback]

    trainer.callback_handler = _Handler()

    assert find_checkpoint_evaluation_policy(trainer) is active_callback.policy
