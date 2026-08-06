"""Save-time evaluation policy for retaining one best training checkpoint.

The regular :class:`transformers.Trainer` save schedule is deliberately kept
as the clock for this feature.  ``CheckpointEvaluationTriggerCallback`` turns
each native save request into an evaluation request; the trainer then calls
``CheckpointEvaluationPolicy.observe`` with the score produced by its
in-memory evaluator.  Only an improving score is allowed to turn saving back
on.

Keeping the decision policy independent from Trainer makes it small, easy to
resume, and testable without loading a model or importing an accelerator.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from opsd_utils.checkpoint_eval_paths import update_final_checkpoint_link

try:  # Allow the lightweight policy tests to run without the training stack.
    from transformers import TrainerCallback
    from transformers.trainer_callback import ExportableState
except ImportError:  # pragma: no cover - exercised only in minimal tooling envs
    class TrainerCallback:  # type: ignore[no-redef]
        """Minimal compatibility base used when transformers is unavailable."""

    class ExportableState:  # type: ignore[no-redef]
        """Compatibility marker for environments that only test this module."""


@dataclass
class CheckpointEvaluationState:
    """Durable state required to make save decisions after a resume."""

    best_score: float | None = None
    best_step: int | None = None
    lower_score_streak: int = 0
    evaluation_count: int = 0
    # A terminal policy result must survive a crash before the trainer exits.
    # Otherwise a resumed job could perform another long interval of training
    # before noticing that the third lower score had already occurred.
    stop_requested: bool = False


@dataclass(frozen=True)
class CheckpointEvaluationDecision:
    """The result of comparing one evaluation score with the retained best."""

    score: float
    step: int
    should_save: bool
    should_stop: bool
    reason: str
    best_score: float
    best_step: int
    lower_score_streak: int
    evaluation_count: int

    @property
    def improved(self) -> bool:
        """Whether this result replaced the best checkpoint."""
        return self.should_save

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointEvaluationDecision":
        """Restore a decision sent over a distributed object broadcast."""
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint evaluation decision must be a mapping")
        required = {
            "score",
            "step",
            "should_save",
            "should_stop",
            "reason",
            "best_score",
            "best_step",
            "lower_score_streak",
            "evaluation_count",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"checkpoint evaluation decision missing fields: {sorted(missing)}")
        return cls(
            score=float(payload["score"]),
            step=int(payload["step"]),
            should_save=bool(payload["should_save"]),
            should_stop=bool(payload["should_stop"]),
            reason=str(payload["reason"]),
            best_score=float(payload["best_score"]),
            best_step=int(payload["best_step"]),
            lower_score_streak=int(payload["lower_score_streak"]),
            evaluation_count=int(payload["evaluation_count"]),
        )


class CheckpointEvaluationPolicy:
    """Compare checkpoint-time scores and implement best-only retention.

    A first finite score always establishes the initial recoverable
    checkpoint.  Later scores save only when strictly greater than the best.
    Strictly lower scores increment ``lower_score_streak`` and stop at
    ``patience``.  Equal scores are controlled by ``tie_policy``:

    * ``reset`` (the training default): reset the low-score streak;
    * ``ignore``: leave the streak unchanged;
    * ``stop``: stop immediately, without saving.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        *,
        patience: int = 3,
        tie_policy: str = "reset",
        state: CheckpointEvaluationState | None = None,
    ) -> None:
        try:
            patience = int(patience)
        except (TypeError, ValueError) as exc:
            raise ValueError("patience must be a positive integer") from exc
        if patience <= 0:
            raise ValueError(f"patience must be a positive integer, got {patience}")
        tie_policy = str(tie_policy).strip().lower()
        if tie_policy not in {"reset", "ignore", "stop"}:
            raise ValueError("tie_policy must be one of: reset, ignore, stop")
        self.patience = patience
        self.tie_policy = tie_policy
        self.state = state if state is not None else CheckpointEvaluationState()
        self._validate_state(self.state)

    @staticmethod
    def _coerce_finite_score(score: Any) -> float:
        try:
            value = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"checkpoint evaluation score must be numeric, got {score!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"checkpoint evaluation score must be finite, got {score!r}")
        return value

    @staticmethod
    def _coerce_step(step: Any) -> int:
        # bool is an int subclass but never a meaningful Trainer global step.
        if isinstance(step, bool):
            raise ValueError("checkpoint evaluation step must be a non-negative integer")
        try:
            result = int(step)
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint evaluation step must be a non-negative integer") from exc
        if result < 0 or result != step:
            raise ValueError("checkpoint evaluation step must be a non-negative integer")
        return result

    @classmethod
    def _validate_state(cls, state: CheckpointEvaluationState) -> None:
        if state.best_score is not None:
            cls._coerce_finite_score(state.best_score)
        if state.best_step is not None:
            cls._coerce_step(state.best_step)
        if state.best_score is None and state.best_step is not None:
            raise ValueError("checkpoint evaluation state has best_step without best_score")
        if state.best_score is not None and state.best_step is None:
            raise ValueError("checkpoint evaluation state has best_score without best_step")
        if int(state.lower_score_streak) < 0 or int(state.evaluation_count) < 0:
            raise ValueError("checkpoint evaluation counters cannot be negative")
        if int(state.evaluation_count) > 0 and state.best_score is None:
            raise ValueError(
                "checkpoint evaluation state has completed evaluations without a best score"
            )
        if state.stop_requested and state.best_score is None:
            raise ValueError("checkpoint evaluation stop state has no best score")

    def observe(self, score: Any, *, step: Any) -> CheckpointEvaluationDecision:
        """Record a score and return the corresponding save/stop decision.

        This mutates only policy state.  The caller remains responsible for
        saving the model on every process when ``should_save`` is true.
        """
        score_value = self._coerce_finite_score(score)
        step_value = self._coerce_step(step)
        state = self.state
        if state.stop_requested:
            raise RuntimeError(
                "checkpoint evaluation policy already requested training stop; "
                "create a new run instead of observing more scores"
            )
        state.evaluation_count += 1

        if state.best_score is None:
            state.best_score = score_value
            state.best_step = step_value
            state.lower_score_streak = 0
            state.stop_requested = False
            return self._decision(score_value, step_value, True, False, "initial")

        if score_value > state.best_score:
            state.best_score = score_value
            state.best_step = step_value
            state.lower_score_streak = 0
            state.stop_requested = False
            return self._decision(score_value, step_value, True, False, "improved")

        if score_value < state.best_score:
            state.lower_score_streak += 1
            should_stop = state.lower_score_streak >= self.patience
            state.stop_requested = should_stop
            return self._decision(
                score_value,
                step_value,
                False,
                should_stop,
                "patience_exhausted" if should_stop else "lower",
            )

        # Equality must not overwrite the old checkpoint.  The explicit
        # policy makes repeated plateaus predictable for long-running jobs.
        if self.tie_policy == "reset":
            state.lower_score_streak = 0
            state.stop_requested = False
            return self._decision(score_value, step_value, False, False, "tie_reset")
        if self.tie_policy == "stop":
            state.stop_requested = True
            return self._decision(score_value, step_value, False, True, "tie_stop")
        state.stop_requested = False
        return self._decision(score_value, step_value, False, False, "tie_ignored")

    def _decision(
        self,
        score: float,
        step: int,
        should_save: bool,
        should_stop: bool,
        reason: str,
    ) -> CheckpointEvaluationDecision:
        # best_score/best_step are guaranteed after the first branch of observe.
        assert self.state.best_score is not None
        assert self.state.best_step is not None
        return CheckpointEvaluationDecision(
            score=score,
            step=step,
            should_save=should_save,
            should_stop=should_stop,
            reason=reason,
            best_score=self.state.best_score,
            best_step=self.state.best_step,
            lower_score_streak=self.state.lower_score_streak,
            evaluation_count=self.state.evaluation_count,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-safe policy state suitable for Trainer checkpoint state."""
        return {
            "version": self.STATE_VERSION,
            "patience": self.patience,
            "tie_policy": self.tie_policy,
            "state": asdict(self.state),
        }

    @property
    def checkpoint_eval_state(self) -> dict[str, Any]:
        """Alias used by Trainer/callback checkpoint serialization glue."""
        return self.state_dict()

    @checkpoint_eval_state.setter
    def checkpoint_eval_state(self, payload: Mapping[str, Any]) -> None:
        self.load_state_dict(payload)

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Restore a state emitted by :meth:`state_dict`.

        Policy settings are included to prevent a resumed run from silently
        changing its early-stop semantics.  A caller may construct with the
        same settings (the normal path), or explicitly restore an older state
        whose settings match.
        """
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint evaluation state must be a mapping")
        version = payload.get("version", self.STATE_VERSION)
        if version != self.STATE_VERSION:
            raise ValueError(f"unsupported checkpoint evaluation state version: {version!r}")
        if int(payload.get("patience", self.patience)) != self.patience:
            raise ValueError("resumed checkpoint evaluation patience does not match configuration")
        if str(payload.get("tie_policy", self.tie_policy)).strip().lower() != self.tie_policy:
            raise ValueError("resumed checkpoint evaluation tie_policy does not match configuration")
        raw_state = payload.get("state", payload)
        if not isinstance(raw_state, Mapping):
            raise ValueError("checkpoint evaluation state payload must contain a state mapping")
        state = CheckpointEvaluationState(
            best_score=raw_state.get("best_score"),
            best_step=raw_state.get("best_step"),
            lower_score_streak=int(raw_state.get("lower_score_streak", 0)),
            evaluation_count=int(raw_state.get("evaluation_count", 0)),
            stop_requested=bool(raw_state.get("stop_requested", False)),
        )
        self._validate_state(state)
        self.state = state

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "CheckpointEvaluationPolicy":
        """Construct a policy with the settings and state in ``payload``."""
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint evaluation state must be a mapping")
        policy = cls(
            patience=payload.get("patience", 3),
            tie_policy=payload.get("tie_policy", "reset"),
        )
        policy.load_state_dict(payload)
        return policy


def apply_checkpoint_evaluation_decision(control: Any, decision: CheckpointEvaluationDecision | Mapping[str, Any]) -> Any:
    """Apply a policy decision to a Trainer control object.

    This tiny helper centralizes the required veto: non-improving evaluations
    must clear the original save event, while an exhausted patience counter
    requests a clean training-loop stop.  It accepts a mapping so rank-zero
    decisions can be broadcast as ordinary serializable objects.
    """
    if isinstance(decision, Mapping):
        decision = CheckpointEvaluationDecision.from_dict(decision)
    if not isinstance(decision, CheckpointEvaluationDecision):
        raise TypeError("decision must be CheckpointEvaluationDecision or a mapping")
    control.should_save = bool(decision.should_save)
    # Custom in-memory evaluation does not run Trainer.evaluate's normal
    # callback path unless its caller does so explicitly.  Clear this flag
    # here as a final lifecycle guard: otherwise the epoch-end
    # _maybe_log_save_evaluate invocation can evaluate the same save event a
    # second time (especially on the final optimizer step).
    if hasattr(control, "should_evaluate"):
        control.should_evaluate = False
    if decision.should_stop:
        control.should_training_stop = True
    return control


class CheckpointEvaluationTriggerCallback(TrainerCallback, ExportableState):
    """Redirect native Trainer save events to the in-memory evaluator.

    ``DefaultFlowCallback`` sets ``control.should_save`` at configured save
    steps/epochs.  This callback is appended after it, clears that flag and
    requests evaluation instead.  The trainer consumes the pending flag during
    its custom ``evaluate`` implementation and uses the policy decision to
    set ``should_save`` back to true only for an improved score.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        patience: int = 3,
        tie_policy: str = "reset",
        output_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.policy = CheckpointEvaluationPolicy(patience=patience, tie_policy=tie_policy)
        self.output_dir = os.fspath(output_dir) if output_dir is not None else None
        self.pending_checkpoint_evaluation = False
        self.trigger_count = 0
        # The in-flight save flag is process-local: a checkpoint is serialized
        # before ``on_save``, so persisting it would make a restored callback
        # believe a successful save was still in flight.  The terminal-stop
        # flag is mirrored in policy state and intentionally survives resume.
        self._pending_checkpoint_save_step: int | None = None
        self._policy_stop_requested = False
        # ``persist`` can be called by the trainer immediately after a policy
        # decision.  Cache the process role from on_train_begin so all ranks
        # may invoke it safely without racing on the metadata file.
        self._is_world_process_zero: bool | None = None

    @property
    def state_path(self) -> Path | None:
        """Path of the external policy metadata, or ``None`` when disabled."""
        return Path(self.output_dir) / "checkpoint_eval_state.json" if self.output_dir else None

    @staticmethod
    def _world_process_zero(state: Any) -> bool:
        """Use Trainer's global-rank flag, defaulting to single-process true."""
        return bool(getattr(state, "is_world_process_zero", True))

    def _set_output_dir_from_training_args(self, args: Any) -> None:
        if self.output_dir is None:
            output_dir = getattr(args, "output_dir", None)
            if output_dir:
                self.output_dir = os.fspath(output_dir)

    def _best_checkpoint_dir(self, *, policy: CheckpointEvaluationPolicy | None = None) -> Path | None:
        """Return the policy's retained checkpoint directory when it exists.

        The sidecar is allowed to be newer than the retained Trainer
        checkpoint after one or more lower-scoring evaluations.  It is *not*
        allowed to nominate a best step whose checkpoint was never committed:
        that can happen if a process dies between an evaluation and native
        Trainer checkpoint writing.  Checking this path is the commit barrier
        for sidecar recovery.
        """
        active_policy = self.policy if policy is None else policy
        step = active_policy.state.best_step
        if step is None or self.output_dir is None:
            return None
        return Path(self.output_dir) / f"checkpoint-{int(step)}"

    def _sidecar_policy_is_committed(self, policy: CheckpointEvaluationPolicy) -> bool:
        """Whether a sidecar policy can safely override checkpoint state."""
        # ``output_dir`` is always set by on_train_begin in real Trainer runs.
        # Keep direct, lightweight uses of ``load`` backwards compatible when
        # a caller intentionally has no filesystem location.
        if self.output_dir is None:
            return True
        checkpoint_dir = self._best_checkpoint_dir(policy=policy)
        return checkpoint_dir is not None and checkpoint_dir.is_dir()

    def persist(self, *, is_world_process_zero: bool | None = None) -> Path | None:
        """Atomically persist policy metadata outside model checkpoints.

        This is intentionally a sidecar under ``output_dir`` rather than an
        edit to ``checkpoint-*/trainer_state.json``.  A lower-scoring result
        has no checkpoint to write, but it still has to survive a crash so the
        three-strike early-stop counter remains correct after resuming.

        The trainer may call this on every rank.  Only global rank zero writes;
        all other ranks return without touching the filesystem.
        """
        is_main = self._is_world_process_zero if is_world_process_zero is None else bool(is_world_process_zero)
        if is_main is False:
            return None
        path = self.state_path
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CheckpointEvaluationPolicy.STATE_VERSION,
            "checkpoint_eval_state": self.checkpoint_eval_state,
        }
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}")
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

    def load(self, *, is_world_process_zero: bool | None = None) -> bool:
        """Load sidecar policy metadata, returning false when it is absent.

        Invalid or incompatible metadata fails loudly instead of silently
        resetting patience: silently doing so could retain a lower-scoring
        model after a resume.  Like :meth:`persist`, only rank zero reads it;
        the trainer's normal evaluation decision broadcast synchronizes the
        restored policy to other ranks before the next save decision.
        """
        is_main = self._is_world_process_zero if is_world_process_zero is None else bool(is_world_process_zero)
        if is_main is False:
            return False
        path = self.state_path
        if path is None or not path.exists():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load checkpoint evaluation metadata from {path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"Checkpoint evaluation metadata at {path} must be a mapping")
        if payload.get("version") != CheckpointEvaluationPolicy.STATE_VERSION:
            raise ValueError(
                f"Unsupported checkpoint evaluation metadata version at {path}: "
                f"{payload.get('version')!r}"
            )
        policy_state = payload.get("checkpoint_eval_state")
        if not isinstance(policy_state, Mapping):
            raise ValueError(f"Checkpoint evaluation metadata at {path} has no policy state")

        # A retained Trainer checkpoint is the durable model/optimizer source
        # of truth.  A sidecar may add later *lower* evaluations, but must
        # never roll the callback backwards or point it to a best model that
        # was not actually written.  Construct a temporary policy first so
        # incompatible/corrupt metadata still fails loudly without mutating
        # the live callback.
        candidate = CheckpointEvaluationPolicy(
            patience=self.policy.patience,
            tie_policy=self.policy.tie_policy,
        )
        candidate.load_state_dict(policy_state)
        if candidate.state.evaluation_count <= self.policy.state.evaluation_count:
            return False
        if not self._sidecar_policy_is_committed(candidate):
            return False

        self.policy.load_state_dict(candidate.state_dict())
        return True

    @property
    def checkpoint_eval_state(self) -> dict[str, Any]:
        """Serializable policy state restored by Transformers callback state."""
        return self.policy.state_dict()

    @checkpoint_eval_state.setter
    def checkpoint_eval_state(self, payload: Mapping[str, Any]) -> None:
        self.policy.load_state_dict(payload)

    def record_checkpoint_evaluation_decision(
        self, decision: CheckpointEvaluationDecision | Mapping[str, Any]
    ) -> None:
        """Record transient lifecycle state for the trainer's policy result.

        A saving decision becomes durable only from :meth:`on_save`, which is
        invoked after ``Trainer._save_checkpoint`` has written and rotated the
        checkpoint.  Non-saving decisions are persisted immediately by the
        trainer because otherwise their lower-score streak has no native
        checkpoint in which to live.
        """
        if isinstance(decision, Mapping):
            decision = CheckpointEvaluationDecision.from_dict(decision)
        if not isinstance(decision, CheckpointEvaluationDecision):
            raise TypeError("decision must be CheckpointEvaluationDecision or a mapping")

        if decision.should_save:
            if (
                self._pending_checkpoint_save_step is not None
                and self._pending_checkpoint_save_step != decision.step
            ):
                raise RuntimeError(
                    "checkpoint evaluation attempted to schedule a second save before "
                    "the prior best checkpoint committed"
                )
            self._pending_checkpoint_save_step = decision.step
        if decision.should_stop:
            self._policy_stop_requested = True

    def _commit_saved_checkpoint(self, *, step: int) -> None:
        """Publish an already-written best checkpoint to sidecar/link users."""
        is_main = self._is_world_process_zero
        if is_main is None:
            is_main = True
        if not is_main:
            return

        best_step = self.policy.state.best_step
        if best_step != step:
            raise RuntimeError(
                "checkpoint evaluation save commit does not match the policy best step: "
                f"saved_step={step}, best_step={best_step}"
            )
        checkpoint_dir = self._best_checkpoint_dir()
        if checkpoint_dir is None or not checkpoint_dir.is_dir():
            raise RuntimeError(
                "checkpoint evaluation expected native Trainer to write the new best "
                f"checkpoint before on_save: {checkpoint_dir}"
            )

        # Keep the compatibility final_checkpoint pointer valid throughout a
        # long run, not merely after ``train()`` returns.  Native rotation may
        # have just deleted the old target, so this is intentionally after
        # _save_checkpoint rather than at evaluation time.
        update_final_checkpoint_link(self.output_dir, checkpoint_dir)
        self.persist(is_world_process_zero=True)

    def _recover_committed_checkpoint_artifacts(self) -> None:
        """Repair sidecar/link state after resuming an already-written best."""
        is_main = self._is_world_process_zero
        if is_main is None:
            is_main = True
        if not is_main or self.policy.state.best_step is None:
            return
        checkpoint_dir = self._best_checkpoint_dir()
        if checkpoint_dir is None or not checkpoint_dir.is_dir():
            return
        update_final_checkpoint_link(self.output_dir, checkpoint_dir)
        self.persist(is_world_process_zero=True)

    @staticmethod
    def _wait_for_native_checkpoint_write() -> None:
        """Wait for rank-local checkpoint shards before publishing the link.

        Trainer invokes ``on_save`` on every rank after its own checkpoint
        writes.  A small distributed barrier here prevents rank zero from
        publishing ``final_checkpoint`` while another rank is still writing
        optimizer/RNG or sharded model data.  Single-process and non-torch
        backends intentionally take the no-op path.
        """
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except (ImportError, RuntimeError):
            # Trainer itself supports non-torch distributed backends.  Do not
            # make the policy callback less portable than the native saver.
            return

    def _broadcast_rank_zero_outcome(
        self,
        *,
        operation: str,
        success_payload: Mapping[str, Any] | None = None,
        rank_zero_error: Exception | None = None,
    ) -> dict[str, Any] | None:
        """Share a rank-zero operation's outcome before any rank can proceed.

        A sidecar is intentionally read and written only by global rank zero.
        That normally avoids filesystem races, but it has an important failure
        mode: if rank zero raises before a following collective, workers block
        forever at that collective.  Put both successful values and failures
        in one object broadcast so every torch-distributed rank either proceeds
        with the exact same result or raises a contextual error.

        ``None`` is returned only for an apparent nonzero rank when torch
        distributed is not initialized.  That case is not a real distributed
        run and preserves the callback's lightweight/direct-use behavior.
        """
        is_main = self._is_world_process_zero is not False
        outcome: list[Any] = [None]
        if is_main:
            if rank_zero_error is None:
                outcome[0] = {"ok": True, **dict(success_payload or {})}
            else:
                outcome[0] = {
                    "ok": False,
                    "error_type": type(rank_zero_error).__name__,
                    "error": str(rank_zero_error),
                }

        try:
            import torch.distributed as dist
        except ImportError:  # pragma: no cover - torch is present in training
            distributed = False
        else:
            distributed = bool(dist.is_available() and dist.is_initialized())
            if distributed:
                dist.broadcast_object_list(outcome, src=0)

        if not distributed and not is_main:
            return None

        received = outcome[0]
        if not isinstance(received, Mapping) or not isinstance(received.get("ok"), bool):
            raise RuntimeError(
                f"checkpoint evaluation {operation} did not receive a valid rank-zero outcome"
            )
        if not received["ok"]:
            error_type = received.get("error_type", "RuntimeError")
            detail = received.get("error", "unknown error")
            message = (
                f"checkpoint evaluation {operation} failed on global rank zero "
                f"({error_type}): {detail}"
            )
            if rank_zero_error is not None:
                raise RuntimeError(message) from rank_zero_error
            raise RuntimeError(message)
        return dict(received)

    def _run_on_rank_zero_and_broadcast_failure(
        self, *, operation: str, action: Callable[[], Any]
    ) -> None:
        """Run a rank-zero-only side effect without stranding other ranks."""
        is_main = self._is_world_process_zero is not False
        rank_zero_error: Exception | None = None
        if is_main:
            try:
                action()
            except Exception as exc:
                # Do not re-raise until every worker has received this result.
                rank_zero_error = exc
        self._broadcast_rank_zero_outcome(
            operation=operation,
            rank_zero_error=rank_zero_error,
        )

    def _load_and_broadcast_policy_state_from_world_zero(self) -> bool:
        """Load sidecar state on rank zero and share success *or* failure.

        ``load`` performs validation and may raise for corrupt or incompatible
        metadata.  It must therefore be inside the same error-envelope
        collective as the resulting policy state; calling ``load`` followed by
        a raw policy broadcast would otherwise leave every worker waiting when
        rank zero rejects a sidecar.
        """
        is_main = self._is_world_process_zero is not False
        loaded = False
        rank_zero_error: Exception | None = None
        success_payload: Mapping[str, Any] | None = None
        if is_main:
            try:
                loaded = self.load(is_world_process_zero=True)
                success_payload = {
                    "loaded": bool(loaded),
                    "policy_state": self.policy.state_dict(),
                }
            except Exception as exc:
                # Keep the original exception for rank zero's exception chain,
                # while workers receive its serializable description.
                rank_zero_error = exc

        received = self._broadcast_rank_zero_outcome(
            operation="sidecar policy restore",
            success_payload=success_payload,
            rank_zero_error=rank_zero_error,
        )
        if received is None:
            return False
        policy_state = received.get("policy_state")
        if not isinstance(policy_state, Mapping):
            raise RuntimeError(
                "checkpoint evaluation sidecar policy restore received no valid policy state"
            )
        self.policy.load_state_dict(policy_state)
        return bool(received.get("loaded", False))

    def _broadcast_policy_state_from_world_zero(self) -> None:
        """Synchronize sidecar-restored policy state to every torch rank.

        Callback state restored from ``trainer_state.json`` is normally the
        same on every rank.  The sidecar can be newer, however, because lower
        scores do not create a native checkpoint.  Only global rank zero reads
        that file, so immediately broadcast the resulting state before a
        resumed training loop can make a save or early-stop decision.

        Keep this helper local to the callback rather than importing
        ``accelerate`` here: the policy module is intentionally usable by its
        lightweight tests and by tooling which does not import the full
        training stack.
        """
        try:
            import torch.distributed as dist
        except ImportError:  # pragma: no cover - torch is present in training
            return

        if not dist.is_available() or not dist.is_initialized():
            return

        payload: list[Any] = [self.policy.state_dict() if self._is_world_process_zero else None]
        dist.broadcast_object_list(payload, src=0)
        policy_state = payload[0]
        if not isinstance(policy_state, Mapping):
            raise RuntimeError(
                "checkpoint evaluation did not receive valid policy state from global rank zero"
            )
        self.policy.load_state_dict(policy_state)

    def _redirect_save_to_evaluate(self, control: Any) -> Any:
        # HF still invokes on_epoch_end after a policy-requested early stop.
        # DefaultFlowCallback can set should_save there, which would otherwise
        # cause one extra ChartQA evaluation after the third lower score.  Do
        # not use control.should_training_stop here: at max_steps it is also
        # true and the final scheduled checkpoint must still be evaluated.
        if self._policy_stop_requested:
            control.should_save = False
            control.should_evaluate = False
            self.pending_checkpoint_evaluation = False
            return control
        if not self.enabled or not getattr(control, "should_save", False):
            return control
        control.should_save = False
        control.should_evaluate = True
        self.pending_checkpoint_evaluation = True
        self.trigger_count += 1
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return self._redirect_save_to_evaluate(control)

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return self._redirect_save_to_evaluate(control)

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Restore sidecar metadata on rank zero before any save-time eval."""
        self._set_output_dir_from_training_args(args)
        self._is_world_process_zero = self._world_process_zero(state)
        self._pending_checkpoint_save_step = None
        self._policy_stop_requested = bool(self.policy.state.stop_requested)
        # Do not call ``load`` followed by a separate raw broadcast: an invalid
        # sidecar would make rank zero raise before workers enter that latter
        # collective.  This helper broadcasts either the restored state or the
        # rank-zero error, so the whole job fails coherently.
        self._load_and_broadcast_policy_state_from_world_zero()
        self._policy_stop_requested = bool(self.policy.state.stop_requested)
        if self._policy_stop_requested:
            # ``CallbackHandler.on_train_begin`` resets this field before
            # invoking callbacks.  Restore the terminal policy decision here
            # so a resumed job exits without taking another optimizer step.
            control.should_training_stop = True
        # If a process died after native checkpoint writing but before on_save,
        # callback state in trainer_state.json is still valid.  Re-publish the
        # sidecar and final_checkpoint link before continuing.
        self._run_on_rank_zero_and_broadcast_failure(
            operation="committed checkpoint artifact recovery",
            action=self._recover_committed_checkpoint_artifacts,
        )
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Commit sidecar/link metadata only after native checkpoint writing."""
        expected_step = self._pending_checkpoint_save_step
        if expected_step is None:
            return control
        self._pending_checkpoint_save_step = None
        saved_step = int(getattr(state, "global_step", expected_step))
        if saved_step != expected_step:
            raise RuntimeError(
                "checkpoint evaluation received an unexpected native save event: "
                f"expected_step={expected_step}, saved_step={saved_step}"
            )
        self._wait_for_native_checkpoint_write()
        # Native shard writes have completed on every rank.  The final link
        # and sidecar are rank-zero-only publication work, so share a failure
        # from that phase before workers can advance into a later collective.
        self._run_on_rank_zero_and_broadcast_failure(
            operation="saved checkpoint publication",
            action=lambda: self._commit_saved_checkpoint(step=saved_step),
        )
        return control

    def consume_checkpoint_evaluation(self) -> bool:
        """Return and clear whether the next evaluation is a save-time one."""
        pending = self.pending_checkpoint_evaluation
        self.pending_checkpoint_evaluation = False
        return pending

    def state(self) -> dict[str, Any]:
        """Expose callback progress through Transformers' callback state API."""
        return {
            "args": {
                "enabled": self.enabled,
                "patience": self.policy.patience,
                "tie_policy": self.policy.tie_policy,
                "output_dir": self.output_dir,
            },
            "attributes": {
                "pending_checkpoint_evaluation": self.pending_checkpoint_evaluation,
                "trigger_count": self.trigger_count,
                "checkpoint_eval_state": self.checkpoint_eval_state,
            },
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CheckpointEvaluationTriggerCallback":
        args = state.get("args", {}) if isinstance(state, Mapping) else {}
        attrs = state.get("attributes", {}) if isinstance(state, Mapping) else {}
        callback = cls(
            enabled=bool(args.get("enabled", True)),
            patience=args.get("patience", 3),
            tie_policy=args.get("tie_policy", "reset"),
            output_dir=args.get("output_dir"),
        )
        callback.pending_checkpoint_evaluation = bool(attrs.get("pending_checkpoint_evaluation", False))
        callback.trigger_count = int(attrs.get("trigger_count", 0))
        if "checkpoint_eval_state" in attrs:
            callback.checkpoint_eval_state = attrs["checkpoint_eval_state"]
        callback._policy_stop_requested = bool(callback.policy.state.stop_requested)
        return callback


__all__ = [
    "CheckpointEvaluationDecision",
    "CheckpointEvaluationPolicy",
    "CheckpointEvaluationState",
    "CheckpointEvaluationTriggerCallback",
    "apply_checkpoint_evaluation_decision",
]
