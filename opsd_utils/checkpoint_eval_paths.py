"""Filesystem helpers for checkpoint-evaluation best-checkpoint outputs.

The training callback decides *whether* a checkpoint is the best one.  This
module deliberately only deals with the small amount of durable filesystem
state needed by the main entrypoint: a single checkpoint directory and the
``final_checkpoint`` compatibility symlink that points at it.
"""
from __future__ import annotations

import os
import re
import json
import math
import shutil
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4


_CHECKPOINT_DIR_PATTERN = re.compile(r"checkpoint-(\d+)")
_CHECKPOINT_EVAL_CALLBACK_NAME = "CheckpointEvaluationTriggerCallback"
_TRAINER_STATE_NAME = "trainer_state.json"


def validate_checkpoint_eval_output_dir(output_dir: str | os.PathLike[str]) -> None:
    """Reject output layouts that cannot safely retain one checkpoint.

    A run can resume from its one retained ``checkpoint-*`` directory and an
    optional ``final_checkpoint`` symlink.  A real ``final_checkpoint``
    directory or multiple checkpoint directories are ambiguous legacy layouts.
    This validator is intentionally read-only because the training entrypoint
    calls it on every distributed rank.  Coordinated rank-zero recovery of
    the one proven native-save crash window lives in
    :func:`recover_interrupted_checkpoint_eval_save`.
    """
    root = Path(output_dir)
    if not root.exists():
        return
    if not root.is_dir():
        raise ValueError(
            "checkpoint_eval requires training.output_dir to be a directory: "
            f"{root}"
        )

    final_checkpoint = root / "final_checkpoint"
    if final_checkpoint.exists() and not final_checkpoint.is_symlink():
        raise ValueError(
            "checkpoint_eval cannot safely reuse an output directory containing "
            f"a real final_checkpoint directory/file: {final_checkpoint}. "
            "Move or remove the legacy checkpoint explicitly, then retry."
        )

    checkpoints = [path for path in root.glob("checkpoint-*") if path.is_dir()]
    if len(checkpoints) > 1:
        names = ", ".join(sorted(path.name for path in checkpoints))
        raise ValueError(
            "checkpoint_eval keeps exactly one checkpoint, but output_dir already "
            f"contains multiple checkpoint directories ({names}). "
            "Move or prune the legacy checkpoints explicitly, then retry."
        )

    if final_checkpoint.is_symlink():
        # Native checkpoint rotation happens before Trainer's ``on_save``.
        # When a new best replaces the old one with save_total_limit=1, the
        # existing final_checkpoint link is therefore briefly dangling.  That
        # is an expected internal state and must be replaceable atomically.
        # Still reject arbitrary/broken external links so this helper never
        # silently overwrites user-managed paths.
        target = _internal_final_checkpoint_target(root, final_checkpoint)
        if target is None or (target.exists() and not target.is_dir()):
            raise ValueError(
                "checkpoint_eval found a final_checkpoint symlink that is not "
                f"a replaceable internal checkpoint link: {final_checkpoint}. "
                "Repair or remove it explicitly, then retry."
            )


def _checkpoint_step(path: Path) -> int | None:
    """Return an exact HF checkpoint step, rejecting look-alike directories."""
    match = _CHECKPOINT_DIR_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match is not None else None


def _read_trainer_state(checkpoint_dir: Path) -> Mapping[str, Any] | None:
    """Read a native Trainer state without treating a partial file as valid."""
    state_path = checkpoint_dir / _TRAINER_STATE_NAME
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _callback_checkpoint_eval_policy(trainer_state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Extract the one callback policy emitted by current HF Trainer state."""
    callbacks = trainer_state.get("stateful_callbacks")
    if not isinstance(callbacks, Mapping):
        return None
    callback_state = callbacks.get(_CHECKPOINT_EVAL_CALLBACK_NAME)
    # HF stores a list only when a callback class appears more than once.  The
    # checkpoint-eval configuration installs exactly one such callback; accept
    # the unambiguous serialized representation but never guess between many.
    if isinstance(callback_state, list):
        if len(callback_state) != 1:
            return None
        callback_state = callback_state[0]
    if not isinstance(callback_state, Mapping):
        return None
    attributes = callback_state.get("attributes")
    if not isinstance(attributes, Mapping):
        return None
    policy = attributes.get("checkpoint_eval_state")
    return policy if isinstance(policy, Mapping) else None


def _strict_int(value: Any) -> int | None:
    """Accept JSON integers only; booleans and coercions are not proof."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _policy_best_step(policy: Mapping[str, Any]) -> int | None:
    """Return a valid saved-policy best step, without importing the callback."""
    state = policy.get("state")
    if not isinstance(state, Mapping):
        return None
    best_step = _strict_int(state.get("best_step"))
    if best_step is None or best_step < 0:
        return None
    try:
        best_score = float(state.get("best_score"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(best_score):
        return None
    return best_step


def _policy_score(policy: Mapping[str, Any]) -> float | None:
    """Return the finite best score from a serialized policy."""
    state = policy.get("state")
    if not isinstance(state, Mapping):
        return None
    try:
        score = float(state.get("best_score"))
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _policy_evaluation_count(policy: Mapping[str, Any]) -> int | None:
    """Return a non-negative serialized evaluation count without coercion."""
    state = policy.get("state")
    if not isinstance(state, Mapping):
        return None
    value = _strict_int(state.get("evaluation_count"))
    return value if value is not None and value >= 0 else None


def _valid_serialized_policy(policy: Mapping[str, Any]) -> bool:
    """Validate the policy envelope before publishing it as a sidecar."""
    if _strict_int(policy.get("version")) != 1:
        return False
    patience = _strict_int(policy.get("patience"))
    if patience is None or patience <= 0:
        return False
    if str(policy.get("tie_policy", "")).strip().lower() not in {"reset", "ignore", "stop"}:
        return False
    state = policy.get("state")
    if not isinstance(state, Mapping):
        return False
    lower_score_streak = _strict_int(state.get("lower_score_streak"))
    if lower_score_streak is None or lower_score_streak < 0:
        return False
    if not isinstance(state.get("stop_requested"), bool):
        return False
    return (
        _policy_best_step(policy) is not None
        and _policy_score(policy) is not None
        and _policy_evaluation_count(policy) is not None
    )


def _policy_matches_expected_config(
    policy: Mapping[str, Any],
    *,
    patience: int | None,
    tie_policy: str | None,
) -> bool:
    """Refuse recovery when serialized stop semantics differ from this run."""
    if patience is not None and _strict_int(policy.get("patience")) != patience:
        return False
    if tie_policy is not None and str(policy.get("tie_policy", "")).strip().lower() != tie_policy:
        return False
    return True


def _is_committed_checkpoint_state(
    trainer_state: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    step: int,
) -> bool:
    """Verify enough native state to identify a checkpoint without guessing."""
    if not _valid_serialized_policy(policy):
        return False
    if _strict_int(trainer_state.get("global_step")) != step:
        return False
    # A current HF Trainer state records this after `_determine_best_metric`.
    # Accept its absence for compatibility, but never accept a contradictory
    # value when it is present.
    best_global_step = trainer_state.get("best_global_step")
    if best_global_step is not None and _strict_int(best_global_step) != step:
        return False
    if _policy_best_step(policy) != step:
        return False
    raw_policy_state = policy.get("state")
    if not isinstance(raw_policy_state, Mapping):
        return False
    # The second checkpoint can only arise from an initial/improved policy
    # decision.  A terminal/lower result is never allowed to save a native
    # checkpoint.  We intentionally do not rely on HF's best_global_step:
    # custom save-time evaluation uses its own strict policy rather than
    # Trainer._determine_best_metric.
    if raw_policy_state.get("stop_requested") is not False:
        return False
    if _strict_int(raw_policy_state.get("lower_score_streak")) != 0:
        return False
    evaluation_count = _strict_int(raw_policy_state.get("evaluation_count"))
    if evaluation_count is None or evaluation_count < 1:
        return False
    return True


def _read_sidecar_policy(root: Path) -> tuple[bool, Mapping[str, Any] | None]:
    """Return ``(present, policy)`` without following user-controlled links.

    A sidecar can contain later lower-score evaluations than the last native
    checkpoint, so it is the preferred baseline for interrupted-save recovery.
    A malformed or linked sidecar is deliberately reported as present-but-
    invalid; falling back in that case could discard the only model it was
    meant to protect.
    """
    path = root / "checkpoint_eval_state.json"
    if not path.exists() and not path.is_symlink():
        return False, None
    if path.is_symlink() or not path.is_file():
        return True, None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return True, None
    if not isinstance(payload, Mapping) or _strict_int(payload.get("version")) != 1:
        return True, None
    policy = payload.get("checkpoint_eval_state")
    return True, policy if isinstance(policy, Mapping) else None


def _sidecar_path_is_replaceable(root: Path) -> bool:
    """Return whether recovery may atomically replace the local sidecar.

    A stale *regular* sidecar is expected after a crash between native
    checkpoint rotation and callback publication.  It is safe to replace
    because ``os.replace`` changes only this directory entry.  Never use a
    symlink, directory, device, or FIFO as recovery metadata: even though
    replacement would normally not follow a symlink, treating an arbitrary
    user-managed object as this run's policy file is needlessly surprising.
    """
    path = root / "checkpoint_eval_state.json"
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return stat.S_ISREG(mode)


def _committed_baseline_policy(
    root: Path,
    *,
    old_checkpoint: Path,
    old_step: int,
    expected_patience: int | None,
    expected_tie_policy: str | None,
) -> Mapping[str, Any] | None:
    """Find the committed policy state immediately preceding a new save.

    Prefer the sidecar because it survives no-save lower-score evaluations.
    If it is absent, the old native checkpoint is sufficient.  If it exists
    but is malformed or points at a different checkpoint, fail closed rather
    than silently selecting a weaker baseline.
    """
    sidecar_present, sidecar_policy = _read_sidecar_policy(root)
    if sidecar_present:
        if (
            sidecar_policy is None
            or not _valid_serialized_policy(sidecar_policy)
            or not _policy_matches_expected_config(
                sidecar_policy,
                patience=expected_patience,
                tie_policy=expected_tie_policy,
            )
            or _policy_best_step(sidecar_policy) != old_step
        ):
            return None
        sidecar_state = sidecar_policy.get("state")
        assert isinstance(sidecar_state, Mapping)
        if sidecar_state.get("stop_requested") is not False:
            return None
        return sidecar_policy

    old_trainer_state = _read_trainer_state(old_checkpoint)
    if old_trainer_state is None:
        return None
    old_policy = _callback_checkpoint_eval_policy(old_trainer_state)
    if old_policy is None or not _policy_matches_expected_config(
        old_policy,
        patience=expected_patience,
        tie_policy=expected_tie_policy,
    ):
        return None
    return (
        old_policy
        if _is_committed_checkpoint_state(old_trainer_state, old_policy, step=old_step)
        else None
    )


def _internal_final_checkpoint_target(root: Path, final_checkpoint: Path) -> Path | None:
    """Return the exact local target of a safe internal compatibility link.

    The project itself always writes ``final_checkpoint -> checkpoint-<step>``
    as a direct relative link.  Require that exact shape rather than merely a
    path which happens to resolve below ``output_dir``: recovery may replace a
    dangling link, so accepting ``..`` components or a child symlink would
    turn a user-managed path into a mutable checkpoint artifact.
    """
    if not final_checkpoint.is_symlink():
        return None
    try:
        raw_target = os.readlink(final_checkpoint)
        target_path = Path(raw_target)
        target_step = _checkpoint_step(target_path)
        if (
            target_path.is_absolute()
            or raw_target != target_path.name
            or target_step is None
            or target_path.name != f"checkpoint-{target_step}"
        ):
            return None
        target = root.resolve() / target_path
        # A live checkpoint child must be a real directory, not a route to an
        # arbitrary location.  A deleted old checkpoint simply does not exist
        # and remains eligible for the narrow post-rotation recovery below.
        if target.is_symlink():
            return None
        return target
    except (OSError, ValueError):
        return None


def _replace_final_checkpoint_link(root: Path, best_checkpoint_path: Path) -> Path:
    """Atomically write the known-safe relative compatibility symlink."""
    final_checkpoint = root / "final_checkpoint"
    if final_checkpoint.exists() and not final_checkpoint.is_symlink():
        raise ValueError(
            "checkpoint_eval recovery will not replace a real final_checkpoint path: "
            f"{final_checkpoint}"
        )
    target = os.path.relpath(best_checkpoint_path, start=root.resolve())
    temporary = root / f".final_checkpoint.tmp-{os.getpid()}"
    suffix = 0
    while temporary.exists() or temporary.is_symlink():
        suffix += 1
        temporary = root / f".final_checkpoint.tmp-{os.getpid()}-{suffix}"

    try:
        os.symlink(target, temporary)
        os.replace(temporary, final_checkpoint)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return final_checkpoint


def _persist_recovered_policy_sidecar(root: Path, policy: Mapping[str, Any]) -> Path:
    """Publish the policy from a proven native checkpoint after recovery."""
    path = root / "checkpoint_eval_state.json"
    if not _sidecar_path_is_replaceable(root):
        raise ValueError(
            "checkpoint_eval recovery will not replace a non-regular sidecar: "
            f"{path}"
        )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}")
    payload = {
        "version": policy.get("version"),
        "checkpoint_eval_state": policy,
    }
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _saved_checkpoint_policy(
    checkpoint: Path,
    *,
    step: int,
    expected_patience: int | None,
    expected_tie_policy: str | None,
) -> Mapping[str, Any] | None:
    """Return a policy only when this real native checkpoint proves it saved.

    This is intentionally stricter than merely finding a callback payload.
    Startup repair must never nominate a model based on a partially written
    state file or a policy which would not have been permitted to save.
    """
    trainer_state = _read_trainer_state(checkpoint)
    if trainer_state is None:
        return None
    policy = _callback_checkpoint_eval_policy(trainer_state)
    if policy is None or not _policy_matches_expected_config(
        policy,
        patience=expected_patience,
        tie_policy=expected_tie_policy,
    ):
        return None
    return (
        policy
        if _is_committed_checkpoint_state(trainer_state, policy, step=step)
        else None
    )


def _stale_sidecar_proves_preceding_policy(
    root: Path,
    *,
    old_step: int,
    retained_policy: Mapping[str, Any],
    expected_patience: int | None,
    expected_tie_policy: str | None,
) -> bool:
    """Check a stale sidecar without allowing it to outrank native state.

    A crash after HF rotation but before ``on_save`` leaves the previous
    sidecar behind.  The current checkpoint's serialized callback policy is
    authoritative, but the stale file supplies the missing chronology proof:
    it must nominate the deleted old best and precede the new strict
    improvement.  A normal checkpoint-eval run always has this sidecar after
    its first committed best, so an absent/current/malformed one fails closed
    instead of authorizing a repair of a manually created dangling link.
    """
    if not _sidecar_path_is_replaceable(root):
        return False
    sidecar_present, sidecar_policy = _read_sidecar_policy(root)
    if not sidecar_present:
        return False
    if sidecar_policy is None:
        return False
    if (
        not _valid_serialized_policy(sidecar_policy)
        or not _policy_matches_expected_config(
            sidecar_policy,
            patience=expected_patience,
            tie_policy=expected_tie_policy,
        )
        or _policy_best_step(sidecar_policy) != old_step
    ):
        return False
    sidecar_state = sidecar_policy.get("state")
    if not isinstance(sidecar_state, Mapping) or sidecar_state.get("stop_requested") is not False:
        return False
    old_score = _policy_score(sidecar_policy)
    retained_score = _policy_score(retained_policy)
    old_count = _policy_evaluation_count(sidecar_policy)
    retained_count = _policy_evaluation_count(retained_policy)
    return bool(
        old_score is not None
        and retained_score is not None
        and old_count is not None
        and retained_count is not None
        and retained_score > old_score
        and retained_count > old_count
    )


def _recover_rotated_checkpoint_eval_save(
    root: Path,
    *,
    checkpoint: Path,
    checkpoint_step: int,
    final_checkpoint: Path,
    expected_patience: int | None,
    expected_tie_policy: str | None,
) -> Path | None:
    """Repair the post-rotation / pre-``on_save`` single-checkpoint window.

    No model directory is deleted here.  The only mutable artifacts are this
    project's direct internal compatibility link and a regular local sidecar,
    and both are published only after the retained Trainer state proves that
    this checkpoint is the policy's saved best.
    """
    try:
        root_resolved = root.resolve()
        retained = checkpoint.resolve(strict=True)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if retained.parent != root_resolved or checkpoint.is_symlink():
        return None

    old_target = _internal_final_checkpoint_target(root, final_checkpoint)
    if old_target is None or old_target.exists() or old_target.is_symlink():
        return None
    old_step = _checkpoint_step(old_target)
    if old_step is None or old_step >= checkpoint_step:
        return None

    retained_policy = _saved_checkpoint_policy(
        checkpoint,
        step=checkpoint_step,
        expected_patience=expected_patience,
        expected_tie_policy=expected_tie_policy,
    )
    if retained_policy is None:
        return None
    # A deleted previous best necessarily means this is at least the second
    # checkpoint-evaluation observation.  It rules out repairing a manually
    # created dangling link next to a first-ever native save.
    retained_count = _policy_evaluation_count(retained_policy)
    if retained_count is None or retained_count < 2:
        return None
    if not _stale_sidecar_proves_preceding_policy(
        root,
        old_step=old_step,
        retained_policy=retained_policy,
        expected_patience=expected_patience,
        expected_tie_policy=expected_tie_policy,
    ):
        return None

    # Match the normal callback's publication order.  Once the atomic link is
    # valid, an interruption before the sidecar write is safe: the native
    # checkpoint state remains authoritative and ``on_train_begin`` repairs
    # the stale sidecar without needing this startup-recovery shape again.
    _replace_final_checkpoint_link(root, retained)
    _persist_recovered_policy_sidecar(root, retained_policy)
    return retained


def recover_interrupted_checkpoint_eval_save(
    output_dir: str | os.PathLike[str],
    *,
    patience: int | None = None,
    tie_policy: str | None = None,
) -> Path | None:
    """Recover the one safe HF write-before-rotation crash window.

    Native ``Trainer._save_checkpoint`` writes the new directory and its
    ``trainer_state.json`` before it rotates the old directory.  A process
    death before rotation leaves two checkpoints even with
    ``save_total_limit=1``; a death after rotation but before callback
    ``on_save`` leaves one new checkpoint and a dangling old compatibility
    link.  Do not infer intent from names or mtimes: this function removes the
    older directory only when every durable fact agrees:

    * exactly two numeric ``checkpoint-<step>`` directories exist;
    * ``final_checkpoint`` is the internal relative link to the older one;
    * the newer native Trainer state names that newer step as its policy's
      saved best; and
    * that new policy is a non-terminal, zero-streak, strict score improvement
      over the committed old policy (including a newer sidecar when present).

    For the single-checkpoint post-rotation form it touches no model
    directory; a direct, dangling internal link plus retained native callback
    state and a compatible stale sidecar are required before it repoints the
    link and restores the sidecar.

    It returns the retained new directory on recovery and ``None`` when the
    directory does not match either exact interrupted-save shape.  Ambiguous
    layouts remain the caller's responsibility and are rejected by
    :func:`validate_checkpoint_eval_output_dir`.
    """
    root = Path(output_dir)
    try:
        expected_patience = int(patience) if patience is not None else None
    except (TypeError, ValueError):
        return None
    if expected_patience is not None and expected_patience <= 0:
        return None
    expected_tie_policy = (
        str(tie_policy).strip().lower() if tie_policy is not None else None
    )
    if expected_tie_policy not in {None, "reset", "ignore", "stop"}:
        return None
    if not root.exists():
        return None
    if not root.is_dir():
        return None

    final_checkpoint = root / "final_checkpoint"
    # Match the public validator's safety boundary even when this helper is
    # called directly: never replace a user-managed real path.
    if final_checkpoint.exists() and not final_checkpoint.is_symlink():
        return None

    checkpoints = [path for path in root.glob("checkpoint-*") if path.is_dir()]
    if len(checkpoints) not in {1, 2}:
        return None
    # ``Path.is_dir()`` follows symlinks.  Recovery is the sole destructive
    # path here, so never use a symlinked checkpoint directory as evidence or
    # as a final-link target; otherwise a crafted child link could escape the
    # run directory when resolved below.
    if any(path.is_symlink() for path in checkpoints):
        return None
    checkpoint_steps = [(path, _checkpoint_step(path)) for path in checkpoints]
    if any(step is None for _, step in checkpoint_steps):
        return None
    if len(checkpoints) == 1:
        checkpoint, checkpoint_step = checkpoint_steps[0]
        assert checkpoint_step is not None
        return _recover_rotated_checkpoint_eval_save(
            root,
            checkpoint=checkpoint,
            checkpoint_step=checkpoint_step,
            final_checkpoint=final_checkpoint,
            expected_patience=expected_patience,
            expected_tie_policy=expected_tie_policy,
        )

    ordered = sorted((path, int(step)) for path, step in checkpoint_steps if step is not None)
    old_checkpoint, old_step = ordered[0]
    new_checkpoint, new_step = ordered[1]
    if new_step <= old_step:
        return None
    try:
        root_resolved = root.resolve()
        old_resolved = old_checkpoint.resolve(strict=True)
        new_resolved = new_checkpoint.resolve(strict=True)
        if old_resolved.parent != root_resolved or new_resolved.parent != root_resolved:
            return None
    except (FileNotFoundError, OSError, ValueError):
        return None

    final_target = _internal_final_checkpoint_target(root, final_checkpoint)
    if final_target is None or final_target != old_checkpoint.resolve():
        return None

    new_trainer_state = _read_trainer_state(new_checkpoint)
    if new_trainer_state is None:
        return None
    new_policy = _callback_checkpoint_eval_policy(new_trainer_state)
    if new_policy is None:
        return None
    if not _policy_matches_expected_config(
        new_policy,
        patience=expected_patience,
        tie_policy=expected_tie_policy,
    ):
        return None
    if not _is_committed_checkpoint_state(
        new_trainer_state, new_policy, step=new_step
    ):
        return None
    baseline_policy = _committed_baseline_policy(
        root,
        old_checkpoint=old_checkpoint,
        old_step=old_step,
        expected_patience=expected_patience,
        expected_tie_policy=expected_tie_policy,
    )
    if baseline_policy is None:
        return None
    baseline_score = _policy_score(baseline_policy)
    new_score = _policy_score(new_policy)
    baseline_evaluation_count = _policy_evaluation_count(baseline_policy)
    new_evaluation_count = _policy_evaluation_count(new_policy)
    if (
        baseline_score is None
        or new_score is None
        or baseline_evaluation_count is None
        or new_evaluation_count is None
        or new_score <= baseline_score
        or new_evaluation_count <= baseline_evaluation_count
    ):
        return None

    # Delete only after every proof above succeeds.  If the process dies in
    # the following small sequence, there is still exactly one valid new
    # checkpoint; a dangling old link/sidecar is accepted and the callback
    # repairs it when the resumed Trainer loads this native state.
    # Both final artifacts are known-safe before deleting the only old
    # checkpoint.  Check this up front so a conflicting user-managed sidecar
    # or final path cannot leave the run with one valid checkpoint but stale
    # metadata after the destructive rotation repair.
    if not _sidecar_path_is_replaceable(root):
        return None
    shutil.rmtree(old_checkpoint)
    retained = new_checkpoint.resolve()
    _replace_final_checkpoint_link(root, retained)
    _persist_recovered_policy_sidecar(root, new_policy)
    return retained


def _as_checkpoint_path(
    raw_path: Any,
    *,
    output_dir: Path,
) -> Path | None:
    if raw_path is None:
        return None
    try:
        path = Path(os.fspath(raw_path))
    except TypeError:
        return None
    if not path.is_absolute():
        path = output_dir / path
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(output_dir.resolve())
    except (FileNotFoundError, OSError, ValueError):
        return None
    return resolved if resolved.is_dir() else None


def _candidate_best_checkpoint_paths(trainer: Any) -> Iterable[Any]:
    """Yield compatible best-checkpoint attributes across trainer revisions."""
    for name in (
        "best_checkpoint_path",
        "checkpoint_eval_best_checkpoint_path",
        "best_model_checkpoint",
    ):
        yield getattr(trainer, name, None)

    state = getattr(trainer, "state", None)
    if state is not None:
        yield getattr(state, "best_model_checkpoint", None)

    callback_handler = getattr(trainer, "callback_handler", None)
    for callback in getattr(callback_handler, "callbacks", []) or []:
        for name in (
            "best_checkpoint_path",
            "checkpoint_eval_best_checkpoint_path",
            "best_model_checkpoint",
        ):
            yield getattr(callback, name, None)


def find_best_checkpoint_path(
    trainer: Any,
    output_dir: str | os.PathLike[str],
) -> Path | None:
    """Resolve the retained best checkpoint reported by a trainer/callback.

    The checkpoint-evaluation trainer exposes ``best_checkpoint_path``.  The
    small compatibility fallback list keeps the main entrypoint usable while
    resuming checkpoints produced by closely related trainer versions.
    """
    root = Path(output_dir)
    for raw_path in _candidate_best_checkpoint_paths(trainer):
        path = _as_checkpoint_path(raw_path, output_dir=root)
        if path is not None:
            return path

    # Policy state is intentionally serializable and may only retain best_step.
    # Check both the trainer and callback: on resume Transformers can replace
    # the callback instance before the trainer has consumed its first trigger.
    callback_handler = getattr(trainer, "callback_handler", None)
    callbacks = list(getattr(callback_handler, "callbacks", []) or [])
    owners = [trainer, getattr(trainer, "checkpoint_eval_policy", None), *callbacks]
    for callback in callbacks:
        owners.append(getattr(callback, "policy", None))
    for owner in owners:
        if owner is None:
            continue
        state = getattr(owner, "checkpoint_eval_state", None) or getattr(owner, "state", None)
        if isinstance(state, Mapping):
            state = state.get("state", state)
            best_step = state.get("best_step") if isinstance(state, Mapping) else None
        else:
            best_step = getattr(state, "best_step", None)
        if best_step is None:
            continue
        path = _as_checkpoint_path(f"checkpoint-{int(best_step)}", output_dir=root)
        if path is not None:
            return path
    return None


def find_checkpoint_evaluation_policy(trainer: Any) -> Any | None:
    """Return the active checkpoint-evaluation policy after HF callback restore.

    ``Trainer._load_callback_state`` replaces exportable callback instances on
    resume.  Prefer the policy held by the active callback over the trainer
    field that was captured during ``__init__`` so final reporting cannot show
    stale/empty values when no further save event occurs after resume.
    """
    callback_handler = getattr(trainer, "callback_handler", None)
    for callback in getattr(callback_handler, "callbacks", []) or []:
        policy = getattr(callback, "policy", None)
        if policy is not None and hasattr(policy, "state"):
            return policy
    policy = getattr(trainer, "checkpoint_eval_policy", None)
    return policy if policy is not None and hasattr(policy, "state") else None


def update_final_checkpoint_link(
    output_dir: str | os.PathLike[str],
    best_checkpoint_path: str | os.PathLike[str],
) -> Path:
    """Atomically point ``final_checkpoint`` at the retained best checkpoint.

    The symlink target is relative so a completed run remains relocatable.  A
    temporary symlink followed by ``os.replace`` avoids exposing a partially
    updated link to an evaluator or a user inspecting the run directory.
    """
    root = Path(output_dir)
    validate_checkpoint_eval_output_dir(root)
    best = _as_checkpoint_path(best_checkpoint_path, output_dir=root)
    if best is None:
        raise ValueError(
            "Cannot update final_checkpoint: best checkpoint must be an existing "
            f"directory inside output_dir ({best_checkpoint_path!r})."
        )

    final_checkpoint = root / "final_checkpoint"
    if final_checkpoint.exists() and not final_checkpoint.is_symlink():
        # Keep this explicit even though validation above handles it.  The file
        # system may have changed between validation and the atomic replacement.
        raise ValueError(
            "Cannot replace a real final_checkpoint directory/file: "
            f"{final_checkpoint}"
        )

    target = os.path.relpath(best, start=root.resolve())
    temporary = root / f".final_checkpoint.tmp-{os.getpid()}"
    suffix = 0
    while temporary.exists() or temporary.is_symlink():
        suffix += 1
        temporary = root / f".final_checkpoint.tmp-{os.getpid()}-{suffix}"

    try:
        os.symlink(target, temporary)
        os.replace(temporary, final_checkpoint)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return final_checkpoint
