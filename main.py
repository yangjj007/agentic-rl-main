# train_grpo.py
"""
Main script for training a Llava-based model using the custom MyGRPOTrainer.

This script handles:
1. Configuration loading.
2. Initialization of Weights & Biases (wandb) and Hugging Face Accelerate.
3. Loading the model and processor.
4. Preparing the training and evaluation datasets.
5. Setting up and running the GPRO trainer.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Any

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from datasets import Dataset, load_dataset
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from trl import GRPOConfig

from config.loader import load_config
from data_utils.commom_util import collate_fn, define_task_data_func
from data_utils.paths import (
    local_pretrained_kwargs,
    resolve_model_path,
    validate_local_model_dir,
)
from trainer.DyMETrainer import DyMETrainer
from reward_utils.visual_supervision_factory import build_visual_supervision, visual_supervision_needs_teacher
from opsd_utils import debug_log as opsd_debug
from opsd_utils.teacher_batching import (
    log_teacher_placement,
    resolve_teacher_device_map,
)
from opsd_utils.deepspeed_utils import (
    deepspeed_zero_stage,
    gradient_checkpointing_enable_kwargs,
    is_deepspeed_accelerate_config,
    should_disable_gradient_checkpointing,
    uses_deepspeed_json_file,
)
from opsd_utils.trusted_torch_load import maybe_allow_trusted_torch_load
from opsd_utils.checkpoint_eval import CheckpointEvaluationTriggerCallback
from opsd_utils.checkpoint_eval_paths import (
    find_best_checkpoint_path,
    find_checkpoint_evaluation_policy,
    recover_interrupted_checkpoint_eval_save,
    update_final_checkpoint_link,
    validate_checkpoint_eval_output_dir,
)


_CHECKPOINT_EVAL_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "split": "validation",
    "batch_size": 1,
    "max_new_tokens": 1024,
    "patience": 3,
    "tie_policy": "reset",
    "max_samples": None,
}


def resolve_checkpoint_eval_config(
    raw_config: dict[str, Any] | None,
    *,
    task: str,
    eval_dataset: Any,
    dyme_args: dict[str, Any],
) -> dict[str, Any]:
    """Validate and normalize save-time checkpoint evaluation settings.

    The evaluator intentionally receives an already-loaded dataset and the
    trainer's in-memory student model.  Main only validates the configuration,
    applies the Trainer save/metric invariants, and returns a detached config
    for the trainer/callback.
    """
    config = {**_CHECKPOINT_EVAL_DEFAULTS, **(raw_config or {})}
    enabled = bool(config.get("enabled", False))
    config["enabled"] = enabled
    if not enabled:
        return config

    task_name = str(task or "").strip().lower()
    if "chart" not in task_name:
        raise ValueError(
            "checkpoint_eval is enabled, but this training task is not ChartQA: "
            f"{task!r}. Set checkpoint_eval.enabled=false for non-ChartQA runs."
        )
    if eval_dataset is None:
        raise ValueError(
            "checkpoint_eval is enabled but no evaluation dataset was loaded. "
            "Configure dataset.eval_dataset (and the requested split), or disable checkpoint_eval."
        )
    try:
        eval_size = len(eval_dataset)
    except TypeError as exc:
        raise ValueError("checkpoint_eval requires a sized evaluation dataset") from exc
    if eval_size <= 0:
        raise ValueError("checkpoint_eval cannot run on an empty evaluation dataset")

    split = str(config.get("split", "validation")).strip().lower()
    # Model selection must never consume the held-out ChartQA test set.  The
    # standalone eval CLI may use test; the training-time policy may not.
    if split not in {"validation", "val"}:
        raise ValueError(
            "checkpoint_eval.split must be one of validation or val; "
            f"got {config.get('split')!r}"
        )
    config["split"] = split

    for key in ("batch_size", "max_new_tokens", "patience"):
        try:
            value = int(config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"checkpoint_eval.{key} must be a positive integer") from exc
        if value <= 0:
            raise ValueError(f"checkpoint_eval.{key} must be a positive integer, got {value}")
        config[key] = value

    if config.get("max_samples") is not None:
        try:
            max_samples = int(config["max_samples"])
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint_eval.max_samples must be a positive integer") from exc
        if max_samples <= 0:
            raise ValueError("checkpoint_eval.max_samples must be a positive integer")
        if max_samples > eval_size:
            max_samples = eval_size
        config["max_samples"] = max_samples

    tie_policy = str(config.get("tie_policy", "reset")).strip().lower()
    if tie_policy not in {"reset", "ignore", "stop"}:
        raise ValueError(
            "checkpoint_eval.tie_policy must be reset, ignore, or stop; "
            f"got {config.get('tie_policy')!r}"
        )
    config["tie_policy"] = tie_policy

    # ``TrainingArguments`` defaults to steps, and older project profiles
    # (notably config_7B.py) rely on that default while only setting
    # save_steps.  Normalize it here rather than rejecting a valid HF setup.
    save_strategy = str(dyme_args.get("save_strategy", "steps")).strip().lower()
    if save_strategy not in {"steps", "epoch"}:
        raise ValueError(
            "checkpoint_eval requires training.dyme_args.save_strategy to be "
            f"'steps' or 'epoch', got {dyme_args.get('save_strategy')!r}"
        )
    output_dir = dyme_args.get("output_dir")
    if not output_dir:
        raise ValueError("checkpoint_eval requires training.dyme_args.output_dir")

    # Native Trainer checkpoint writing remains the single writer for all
    # distributed ranks.  These invariants prevent generic HF best-model logic
    # from creating a second retention policy or dropping optimizer state.
    dyme_args["save_total_limit"] = 1
    dyme_args["save_only_model"] = False
    dyme_args["metric_for_best_model"] = "checkpoint_score"
    dyme_args["greater_is_better"] = True
    dyme_args["load_best_model_at_end"] = False
    # The native evaluation cadence is independent from save scheduling.  If
    # left enabled, an ordinary Trainer eval returns its normal GRPO metrics,
    # then HF's `_determine_best_metric` looks for `eval_checkpoint_score` and
    # raises KeyError.  The callback promotes every save event to the one
    # authoritative ChartQA evaluation instead.
    dyme_args["eval_strategy"] = "no"
    # The policy (best score + lower-score streak) is stateful.  Preserve it
    # when resuming the one retained checkpoint instead of treating the next
    # evaluation as a fresh baseline.
    dyme_args["restore_callback_states_from_checkpoint"] = True
    return config


def _explicit_checkpoint_resume_target(
    resume_from_checkpoint: Any,
    *,
    output_dir: str | os.PathLike[str],
) -> Path | None:
    """Return a directly requested retained-checkpoint path, if it exists.

    This deliberately excludes ``final_checkpoint``.  That compatibility link
    is atomically repointed by recovery and remains a valid resume path on its
    own; only a user-provided ``checkpoint-<step>`` directory can become stale
    when rank zero repairs the native write-before-rotation crash window.
    """
    if not isinstance(resume_from_checkpoint, (str, os.PathLike)):
        return None
    try:
        raw_path = Path(os.fspath(resume_from_checkpoint))
        if not raw_path.name.startswith("checkpoint-"):
            return None
        suffix = raw_path.name.removeprefix("checkpoint-")
        if not suffix.isdecimal():
            return None
        output_root = Path(output_dir).resolve(strict=False)
        candidate = raw_path.resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if candidate.parent != output_root or not candidate.is_dir():
        return None
    return candidate


def _rewrite_recovered_resume_checkpoint(
    resume_from_checkpoint: Any,
    *,
    resume_target_before_recovery: Path | None,
    recovered_checkpoint: str | os.PathLike[str] | None,
) -> str | bool | None:
    """Redirect only an explicit checkpoint path that recovery removed.

    The pre-recovery existence check is the proof that the caller selected the
    directory that native checkpoint rotation subsequently removed.  Do not
    redirect arbitrary missing paths: those should retain the normal Trainer
    error rather than being silently treated as a request for the best model.
    """
    if recovered_checkpoint is None or resume_target_before_recovery is None:
        return resume_from_checkpoint
    try:
        retained = Path(recovered_checkpoint).resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return resume_from_checkpoint
    if resume_target_before_recovery == retained or resume_target_before_recovery.exists():
        return resume_from_checkpoint
    return os.fspath(retained)


def _coordinate_checkpoint_eval_recovery(
    *,
    accelerator: Any,
    output_dir: str | os.PathLike[str],
    patience: int,
    tie_policy: str,
    resume_from_checkpoint: Any,
) -> tuple[str | bool | None, str | None]:
    """Run destructive checkpoint recovery once and share its exact outcome.

    A rank-zero exception followed by a plain ``wait_for_everyone`` strands
    every other worker at the barrier.  Instead, rank zero catches both
    recovery and layout-validation errors, broadcasts a small serializable
    result to all workers, and each worker raises the same contextual failure.
    On success the rank-zero-normalized resume path is also authoritative, so
    every process resumes the same retained checkpoint after recovery.
    """
    outcome: list[dict[str, Any] | None] = [None]
    rank_zero_error: Exception | None = None
    if bool(getattr(accelerator, "is_main_process", True)):
        resume_target_before_recovery = _explicit_checkpoint_resume_target(
            resume_from_checkpoint,
            output_dir=output_dir,
        )
        try:
            recovered_checkpoint = recover_interrupted_checkpoint_eval_save(
                output_dir,
                patience=patience,
                tie_policy=tie_policy,
            )
            validate_checkpoint_eval_output_dir(output_dir)
            normalized_resume = _rewrite_recovered_resume_checkpoint(
                resume_from_checkpoint,
                resume_target_before_recovery=resume_target_before_recovery,
                recovered_checkpoint=recovered_checkpoint,
            )
            if isinstance(normalized_resume, os.PathLike):
                normalized_resume = os.fspath(normalized_resume)
            outcome[0] = {
                "ok": True,
                "recovered_checkpoint": (
                    os.fspath(recovered_checkpoint)
                    if recovered_checkpoint is not None
                    else None
                ),
                "resume_from_checkpoint": normalized_resume,
            }
        except Exception as exc:
            # Do not re-raise before the collective: workers that reached the
            # next line must receive this failure rather than wait forever.
            rank_zero_error = exc
            outcome[0] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    broadcast_object_list(outcome, from_process=0)
    received = outcome[0]
    if not isinstance(received, dict) or not isinstance(received.get("ok"), bool):
        raise RuntimeError(
            "checkpoint_eval recovery did not receive a valid rank-zero outcome"
        )
    if not received["ok"]:
        error_type = received.get("error_type", "RuntimeError")
        detail = received.get("error", "unknown recovery or validation error")
        message = (
            "checkpoint_eval rank-zero recovery/layout validation failed "
            f"({error_type}): {detail}"
        )
        if rank_zero_error is not None:
            raise RuntimeError(message) from rank_zero_error
        raise RuntimeError(message)

    recovered = received.get("recovered_checkpoint")
    normalized_resume = received.get("resume_from_checkpoint")
    if recovered is not None and not isinstance(recovered, str):
        raise RuntimeError(
            "checkpoint_eval recovery received an invalid retained checkpoint path"
        )
    if normalized_resume is not None and not isinstance(normalized_resume, (str, bool)):
        raise RuntimeError(
            "checkpoint_eval recovery received an invalid resume_from_checkpoint value"
        )
    return normalized_resume, recovered


def apply_launch_config(launch_config: dict[str, Any]) -> None:
    """Apply optional runtime launch knobs from CONFIG['launch'] when env is unset."""
    if not launch_config:
        return
    env_mappings = {
        "opsd_detail_min_free_gb": "DYME_OPSD_DETAIL_MIN_FREE_GB",
        "opsd_detail_every": "DYME_OPSD_DETAIL_EVERY",
        "pytorch_cuda_alloc_conf": "PYTORCH_CUDA_ALLOC_CONF",
    }
    for key, env_name in env_mappings.items():
        if key in launch_config and env_name not in os.environ:
            os.environ[env_name] = str(launch_config[key])


def resolve_gradient_checkpointing_enabled(
    launch_config: dict[str, Any],
) -> bool:
    env_raw = os.environ.get("DYME_GRADIENT_CHECKPOINTING", "").strip().lower()
    if env_raw:
        return env_raw in ("1", "true", "yes", "on")
    return bool(launch_config.get("gradient_checkpointing_enable", False))


def _run_cross_model_vocab_checks(model, processor, teacher_model, model_config: Dict[str, Any]) -> None:
    """Startup checks for cross-model OPD vocab slice + tokenizer alignment."""
    from transformers import AutoProcessor

    from opsd_utils.vocab_align import print_vocab_align_report, verify_shared_tokenizer_alignment

    student_vocab = getattr(model.config, "vocab_size", len(processor.tokenizer))
    teacher_vocab = getattr(teacher_model.config, "vocab_size", student_vocab)
    shared = min(student_vocab, teacher_vocab)
    print(
        f"[OPSD-VOCAB] lm_head widths: student={student_vocab} teacher={teacher_vocab} "
        f"shared_slice={shared}",
        flush=True,
    )
    if student_vocab == teacher_vocab:
        print("[OPSD-VOCAB] vocab sizes match — no slice needed", flush=True)
        return

    teacher_path = model_config.get("teacher_model_path")
    teacher_processor = AutoProcessor.from_pretrained(teacher_path)
    full_scan = os.environ.get("DYME_VOCAB_ALIGN_FULL", "0").strip().lower() in ("1", "true", "yes")
    stride = int(os.environ.get("DYME_VOCAB_ALIGN_STRIDE", "500"))
    report = verify_shared_tokenizer_alignment(
        processor.tokenizer,
        teacher_processor.tokenizer,
        shared_vocab=shared,
        full_scan=full_scan,
        sample_stride=stride,
    )
    print_vocab_align_report(report)


def _wandb_disabled_by_env() -> bool:
    if os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1", "yes", "on"):
        return True
    if os.environ.get("WANDB_MODE", "").lower() in ("disabled", "off"):
        return True
    return False


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def _is_main_process() -> bool:
    return _local_rank() == 0


def _log_startup(message: str) -> None:
    if _is_main_process():
        print(f"[DyME] {message}", flush=True)


def _try_wandb_login() -> bool:
    """Return True if wandb credentials are available (env, offline, or prior login)."""
    if os.environ.get("WANDB_MODE", "").lower() == "offline":
        return True
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
        return True
    try:
        wandb.login(relogin=False)
        key = wandb.api.api_key
        return bool(key and len(key) >= 40)
    except Exception:
        return False


def _resolve_use_wandb(want_wandb: bool) -> bool:
    """Resolve wandb without interactive prompts on distributed worker ranks."""
    if not want_wandb or _wandb_disabled_by_env():
        return False
    if _local_rank() != 0:
        # Never call wandb.login() on worker ranks — shared stdin causes multi-process deadlocks.
        return (
            os.environ.get("WANDB_MODE", "").lower() == "offline"
            or bool(os.environ.get("WANDB_API_KEY"))
        )
    return _try_wandb_login()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _run_text_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        text = (result.stdout or "") + (result.stderr or "")
        return text.strip()
    except Exception as exc:
        return f"<command failed: {exc!r}>"


def write_run_config_snapshot(
    *,
    output_dir: str,
    config_path: str,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    rl_config: Dict[str, Any],
    client_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    opsd_config: Dict[str, Any],
    checkpoint_eval_config: Dict[str, Any] | None,
    training_args: GRPOConfig,
    generation_config,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    resolved = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_arg": config_path,
        "model": model_config,
        "training": training_config,
        "rl": rl_config,
        "client": client_config,
        "dataset": dataset_config,
        "opsd": opsd_config,
        "checkpoint_eval": checkpoint_eval_config or {},
        "training_args": training_args.to_dict() if hasattr(training_args, "to_dict") else vars(training_args),
    }
    files = {
        "resolved_config.json": resolved,
        "run_env.json": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "env": {
                k: v
                for k, v in sorted(os.environ.items())
                if k.startswith("DYME_")
                or k.startswith("CUDA")
                or k.startswith("ACCELERATE")
                or k in {"WANDB_MODE", "WANDB_DISABLED"}
            },
        },
        "generation_config.json": (
            generation_config.to_dict()
            if hasattr(generation_config, "to_dict")
            else _json_safe(generation_config)
        ),
    }
    for name, payload in files.items():
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "git_status.txt"), "w", encoding="utf-8") as f:
        f.write(_run_text_command(["git", "status", "--short"]))
        f.write("\n\n")
        f.write(_run_text_command(["git", "diff", "--stat"]))
        f.write("\n")
    with open(os.path.join(output_dir, "training_command.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv))
        f.write("\n")


def setup_accelerator_and_wandb(bf16, want_wandb: bool) -> tuple[Accelerator, bool]:
    """
    Initialize Accelerator and optionally wandb.

    Returns:
        (accelerator, use_wandb)
    """
    use_wandb = _resolve_use_wandb(want_wandb)

    accel_kwargs: dict = {}
    # bf16 for DDP/MULTI_GPU only; with deepspeed_config_file, precision lives in the JSON.
    if bf16 and not uses_deepspeed_json_file():
        accel_kwargs["mixed_precision"] = "bf16"
    if use_wandb:
        accel_kwargs["log_with"] = "wandb"
    return Accelerator(**accel_kwargs), use_wandb


def destroy_distributed_process_group() -> None:
    """Avoid NCCL teardown warnings / spurious non-zero exit after accelerate launch."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def load_model_and_processor(model_config: Dict[str, Any]):
    """
    Loads the pre-trained vision-language model and its associated processor.

    Args:
        model_config (Dict[str, Any]): Configuration dictionary for the model.

    Returns:
        Tuple[LlavaOnevisionForConditionalGeneration, PreTrainedProcessor]: The loaded model and processor.
    """
    model_id = validate_local_model_dir(
        resolve_model_path(model_config["pretrained_model_path"]),
        role="student",
    )
    local_kw = local_pretrained_kwargs(model_id)
    _log_startup(f"Loading student weights from: {model_id}")

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, model_config['torch_dtype']),
        attn_implementation='flash_attention_2' if model_config['use_flash_attention_2'] else 'sdpa',
        low_cpu_mem_usage=True,
        **local_kw,
    )

    # Freeze the vision tower to save memory and computation
    model.base_model.vision_tower.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(model_id, **local_kw)
    processor.tokenizer.padding_side = "left"
    _log_startup("Student model and processor ready")

    return model, processor


def load_teacher_model(model_config: Dict[str, Any], *, local_rank: int = 0, num_gpus: int = 1):
    """Load optional frozen teacher for cross-model OPD (e.g. LLaVA-OneVision 7B)."""
    teacher_path = model_config.get("teacher_model_path")
    if not teacher_path:
        return None

    teacher_path = validate_local_model_dir(
        resolve_model_path(teacher_path),
        role="teacher",
    )
    teacher_local_kw = local_pretrained_kwargs(teacher_path)
    if teacher_local_kw and os.environ.get("RANK", "0") == "0":
        print(f"[DyME] Loading teacher from local path: {teacher_path}", flush=True)

    dtype_name = model_config.get("teacher_dtype", model_config.get("torch_dtype", "bfloat16"))
    torch_dtype = getattr(torch, dtype_name)
    requested_map = model_config.get("teacher_device_map")
    if not requested_map:
        env_map = os.environ.get("DYME_TEACHER_DEVICE_MAP", "").strip()
        if env_map:
            requested_map = env_map

    device_map = resolve_teacher_device_map(
        requested_map,
        local_rank=local_rank,
        num_gpus=max(1, num_gpus),
    )
    log_teacher_placement(
        local_rank=local_rank,
        num_gpus=max(1, num_gpus),
        teacher_path=teacher_path,
        resolved_device=device_map,
        requested_map=requested_map,
    )

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    manual_teacher_device = None
    if (deepspeed_zero_stage() or 0) >= 3:
        manual_teacher_device = device_map
        if os.environ.get("RANK", "0") == "0":
            print(
                "[DyME] ZeRO-3 teacher load: skip device_map kw and move teacher after load "
                f"(target={device_map})",
                flush=True,
            )
    else:
        load_kwargs["device_map"] = device_map

    teacher = LlavaOnevisionForConditionalGeneration.from_pretrained(
        teacher_path,
        attn_implementation='flash_attention_2' if model_config.get('use_flash_attention_2') else 'sdpa',
        **load_kwargs,
        **teacher_local_kw,
    )
    if manual_teacher_device:
        teacher.to(manual_teacher_device)
    teacher.eval()
    teacher.requires_grad_(False)
    if hasattr(teacher, "base_model") and hasattr(teacher.base_model, "vision_tower"):
        teacher.base_model.vision_tower.requires_grad_(False)

    return teacher


def _select_chartqa_eval_split(dataset_bundle: Any, requested_split: str) -> Dataset:
    """Select validation data for training-time checkpoint evaluation.

    ChartQA mirrors have used both ``validation`` and ``val`` as the split
    name.  They are aliases for this purpose; ``test`` is deliberately never a
    fallback because it would leak the final benchmark into model selection.
    """
    requested = str(requested_split or "validation").strip().lower()
    if requested not in {"validation", "val"}:
        raise ValueError(
            "Training-time ChartQA checkpoint evaluation must use the validation "
            f"split (validation/val), got {requested_split!r}"
        )
    available = set(getattr(dataset_bundle, "keys", lambda: [])())
    for candidate in (requested, "validation", "val"):
        if candidate in available:
            return dataset_bundle[candidate]
    raise ValueError(
        "ChartQA dataset has no validation split (expected 'validation' or 'val'); "
        f"available splits: {sorted(available)}. Refusing to fall back to test."
    )


def prepare_datasets(
    task: str,
    dataset_config: Dict[str, Any],
    mode='rl',
    checkpoint_eval_config: Dict[str, Any] | None = None,
) -> (Dataset, Dataset):
    """
    Prepares the training and evaluation datasets based on the specified task.

    Args:
        task (str): The name of the task (e.g., 'chartqa').
        dataset_config (Dict[str, Any]): Configuration for datasets.

    Returns:
        Tuple[Dataset, Dataset]: The training and evaluation datasets.
    """
    data_func = define_task_data_func(task, mode=mode)

    eval_spec = dataset_config.get("eval_dataset")
    load_eval = eval_spec not in (None, "", False)

    # Create training dataset
    _log_startup(f"Loading training data: {dataset_config['train_dataset']}")
    train_data_list = data_func(json_path=dataset_config['train_dataset'])
    for idx, sample in enumerate(train_data_list):
        if isinstance(sample, dict):
            sample.setdefault("_dyme_index", idx)
    train_dataset = Dataset.from_list(train_data_list)
    max_n = dataset_config.get("max_train_samples")
    if max_n is not None and int(max_n) > 0:
        cap = min(int(max_n), len(train_dataset))
        train_dataset = train_dataset.select(range(cap))
        _log_startup(f"Training dataset capped: {cap} samples (max_train_samples)")
    _log_startup(f"Training dataset ready: {len(train_dataset)} samples")

    # Create evaluation dataset
    checkpoint_eval_enabled = bool((checkpoint_eval_config or {}).get("enabled", False))
    checkpoint_eval_split = (checkpoint_eval_config or {}).get("split", "validation")
    if 'chart' in task and load_eval:
        _log_startup(f"Loading eval dataset: {eval_spec}")
        dataset_bundle = load_dataset(eval_spec)
        if checkpoint_eval_enabled:
            eval_dataset = _select_chartqa_eval_split(dataset_bundle, checkpoint_eval_split)
            max_samples = (checkpoint_eval_config or {}).get("max_samples")
            if max_samples is not None:
                max_samples = min(int(max_samples), len(eval_dataset))
                eval_dataset = eval_dataset.select(range(max_samples))
            _log_startup(
                f"Checkpoint-eval dataset ready: split={checkpoint_eval_split} "
                f"samples={len(eval_dataset)}"
            )
        else:
            # Preserve the historical standalone/main behavior when the new
            # policy is explicitly disabled.  The training callback is not
            # attached in this mode, so test remains an ordinary eval dataset.
            try:
                eval_dataset = dataset_bundle['test']
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "ChartQA eval dataset does not provide a test split while "
                    "checkpoint_eval is disabled"
                ) from exc
        _log_startup(f"Eval dataset ready: {len(eval_dataset)} samples")
        # Note: You can uncomment the line below for quick testing/debugging.
        # eval_dataset = eval_dataset.select(range(1000, 1100))

    else:
        if 'chart' in task and not load_eval:
            _log_startup("Skipping eval dataset load (eval_dataset disabled in config)")
        eval_dataset = None

    return train_dataset, eval_dataset


def _deplot_status_from_text(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return "missing"
    if "deplot_placeholder" in text:
        return "placeholder"
    if "google/deplot" in text or '"source": "deplot"' in text or "'source': 'deplot'" in text:
        return "real"
    return "unknown"


def _log_run_config_summary(
    *,
    config_path: str,
    dataset_config: Dict[str, Any],
    training_config: Dict[str, Any],
    opsd_config: Dict[str, Any],
    model_config: Dict[str, Any],
    launch_config: Dict[str, Any],
    checkpoint_eval_config: Dict[str, Any] | None = None,
) -> None:
    if not _is_main_process():
        return
    dyme_args = training_config.get("dyme_args", {})
    probe_cfg = opsd_config.get("teacher_probe", {}) or {}
    traj_cfg = opsd_config.get("teacher_trajectory", {}) or {}
    visual_cfg = opsd_config.get("visual_supervision", {}) or {}
    utility_cfg = opsd_config.get("signal_utility_routing", {}) or {}
    payload = {
        "config": config_path,
        "train_dataset": dataset_config.get("train_dataset"),
        "output_dir": dyme_args.get("output_dir"),
        "num_train_epochs": dyme_args.get("num_train_epochs"),
        "max_steps": dyme_args.get("max_steps"),
        "resume_from_checkpoint": os.environ.get("DYME_RESUME_FROM_CHECKPOINT", "").strip() or None,
        "opsd_enabled": bool(opsd_config.get("enabled", False)),
        "checkpoint_eval": checkpoint_eval_config or {},
        "opsd_mode": opsd_config.get("mode"),
        "privileged_providers": opsd_config.get("privileged_providers", []),
        "text_include_gold": bool(opsd_config.get("text_include_gold", False)),
        "loss": opsd_config.get("loss", {}),
        "teacher_model_path": model_config.get("teacher_model_path"),
        "teacher_device_map": model_config.get("teacher_device_map", launch_config.get("teacher_device_map")),
        "teacher_probe": {
            "enabled": bool(probe_cfg.get("enabled", False)),
            "context_providers": probe_cfg.get("context_providers", []),
            "max_new_tokens": probe_cfg.get("max_new_tokens"),
            "max_per_batch": probe_cfg.get("max_per_batch"),
            "prompt_profile": probe_cfg.get("prompt_profile"),
            "answer_parser": probe_cfg.get("answer_parser"),
            "skip_no_evidence": probe_cfg.get("skip_no_evidence"),
            "candidate_log": probe_cfg.get("candidate_log", {}),
        },
        "teacher_trajectory": {
            "enabled": bool(traj_cfg.get("enabled", False)),
            "max_new_tokens": traj_cfg.get("max_new_tokens"),
            "loss_type": traj_cfg.get("loss_type"),
            "weight": traj_cfg.get("weight"),
        },
        "visual_supervision": {
            "enabled": bool(visual_cfg.get("enabled", False)),
            "checker_enabled": bool((visual_cfg.get("checker") or {}).get("enabled", False)),
            "refiner_enabled": bool((visual_cfg.get("refiner") or {}).get("enabled", False)),
            "prefetch_ic": bool(visual_cfg.get("prefetch_ic", False)),
            "logging_enabled": bool((visual_cfg.get("logging") or {}).get("enabled", False)),
        },
        "signal_utility_routing": {
            "enabled": bool(utility_cfg.get("enabled", False)),
            "reward_std_scale": utility_cfg.get("reward_std_scale"),
            "grpo_readiness_weight": utility_cfg.get("grpo_readiness_weight"),
            "opd_teacher_need_weight": utility_cfg.get("opd_teacher_need_weight"),
            "opd_format_penalty": utility_cfg.get("opd_format_penalty"),
            "sft_format_bad_bonus": utility_cfg.get("sft_format_bad_bonus"),
            "mode_stable_enabled": utility_cfg.get("mode_stable_enabled"),
            "mode_stable_ema_beta": utility_cfg.get("mode_stable_ema_beta"),
            "mode_stable_switch_margin": utility_cfg.get("mode_stable_switch_margin"),
            "mode_stable_min_hold_steps": utility_cfg.get("mode_stable_min_hold_steps"),
        },
        "hang_debug_env": {
            "DYME_OPSD_HANG_DEBUG": os.environ.get("DYME_OPSD_HANG_DEBUG", "<unset>"),
            "DYME_OPSD_HANG_FORCE": os.environ.get("DYME_OPSD_HANG_FORCE", "<unset>"),
        },
    }
    print(f"[DyME-RUN-CONFIG] {json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True)}", flush=True)


def _log_dataset_status(train_dataset: Dataset, dataset_config: Dict[str, Any]) -> None:
    if not _is_main_process():
        return
    counts = {
        "deplot_real": 0,
        "deplot_placeholder": 0,
        "deplot_missing": 0,
        "deplot_unknown": 0,
        "visual_fact_present": 0,
        "visual_fact_hint_present": 0,
    }
    total = len(train_dataset)
    for sample in train_dataset:
        status = _deplot_status_from_text(sample.get("visual_fact_deplot"))
        counts[f"deplot_{status}"] += 1
        if str(sample.get("visual_fact") or sample.get("visual_facts") or "").strip():
            counts["visual_fact_present"] += 1
        if str(sample.get("visual_fact_hint") or "").strip():
            counts["visual_fact_hint_present"] += 1
    payload = {
        "train_dataset": dataset_config.get("train_dataset"),
        "num_train_samples": total,
        **counts,
        "deplot_real_rate": counts["deplot_real"] / max(total, 1),
        "deplot_placeholder_rate": counts["deplot_placeholder"] / max(total, 1),
    }
    print(f"[DyME-DATA] {json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True)}", flush=True)


def main():
    """
    Main function to orchestrate the model training pipeline.
    """

    parser = argparse.ArgumentParser(description="Train a Llava model using either SFT or GRPO.")

    parser.add_argument(
        '--config', type=str, default='config/config.py',
        help="Python config path (e.g. config/config.py, config/config_trimode.py) "
             "or shorthand alias: norm | trimode | llavacot | low | aok",
    )
    parser.add_argument(
        '--mode', type=str, default='rl',
    )
    parser.add_argument(
        '--opsd_mode', type=str, default=None,
        help="OPSD routing mode: dyme | trimode | rlsd | copsd_opd | opsd_only | replace_sft | opsd_on_wrong | grpo_opsd_joint",
    )
    parser.add_argument(
        '--opsd_providers', type=str, default=None,
        help="Comma-separated privileged providers: text,visual_facts,crop,hybrid",
    )
    parser.add_argument(
        '--opsd_privilege_profile', type=str, default=None,
        help="Privileged profile preset: text | visual | hybrid (default hybrid in config_trimode)",
    )
    parser.add_argument(
        '--reward_weights', type=str, default=None,
        help="Comma-separated reward weights: format,context,acc (e.g. 0.5,1.5,1.0). "
             "Overrides config; env DYME_REWARD_WEIGHTS also supported in antidegen config.",
    )
    parser.add_argument(
        '--opsd_enabled', action='store_true',
        help="Enable OPSD / TriMode training extensions",
    )
    parser.add_argument(
        '--opsd_debug', action='store_true',
        help="Enable verbose OPSD debug logs (or set env DYME_OPSD_DEBUG=1)",
    )
    parser.add_argument(
        '--opsd_detail_every', type=int, default=None,
        help="Emit full weak-signal diagnostic bundle every N global steps on rank 0 "
             "(default 10; config opsd.debug.detail_every or env DYME_OPSD_DETAIL_EVERY)",
    )
    parser.add_argument(
        '--opsd_probe_on_generate', dest='opsd_probe_on_generate', action='store_true',
        help="Emit [OPSD-PROBE] on every (re)generate on rank 0 (config_trimode default on)",
    )
    parser.add_argument(
        '--no_opsd_probe_on_generate', dest='opsd_probe_on_generate', action='store_false',
        help="Disable per-generate [OPSD-PROBE] logs",
    )
    parser.set_defaults(opsd_probe_on_generate=None)
    parser.add_argument(
        '--no_opsd_probe_first_token_logits', dest='opsd_probe_first_token_logits', action='store_false',
        help="Disable pre-generate first-token logits probe ([OPSD-GENDBG])",
    )
    parser.set_defaults(opsd_probe_first_token_logits=None)
    parser.add_argument(
        '--wandb', dest='wandb', action='store_true',
        help="Force enable Weights & Biases logging",
    )
    parser.add_argument(
        '--no_wandb', dest='wandb', action='store_false',
        help="Disable Weights & Biases logging (or set WANDB_MODE=offline/disabled)",
    )
    parser.set_defaults(wandb=None)
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help="Resume Trainer state from a checkpoint directory. Env DYME_RESUME_FROM_CHECKPOINT is also supported.",
    )

    args = parser.parse_args()
    mode = args.mode
    _log_startup(
        f"Process start: argv={' '.join(sys.argv)} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK', '?')} RANK={os.environ.get('RANK', '?')}"
    )

    # 1. Load Configurations
    CONFIG = load_config(args.config)
    model_config = CONFIG['model']
    training_config = CONFIG['training']
    rl_config = CONFIG['rl']
    client_config = CONFIG['client']
    dataset_config = CONFIG['dataset']
    launch_config = dict(CONFIG.get("launch", {}))
    apply_launch_config(launch_config)
    task = training_config['task']
    checkpoint_eval_config = dict(CONFIG.get("checkpoint_eval", {}))
    opsd_config = dict(CONFIG.get('opsd', {"enabled": False, "mode": "dyme"}))
    if args.opsd_enabled:
        opsd_config["enabled"] = True
    if args.opsd_mode is not None:
        opsd_config["enabled"] = True
        opsd_config["mode"] = args.opsd_mode
    if args.opsd_providers is not None:
        opsd_config["privileged_providers"] = [p.strip() for p in args.opsd_providers.split(",") if p.strip()]
    if args.opsd_privilege_profile is not None:
        opsd_config["privileged_profile"] = args.opsd_privilege_profile.strip()
    reward_weights_raw = args.reward_weights or os.environ.get("DYME_REWARD_WEIGHTS")
    if reward_weights_raw:
        parts = [p.strip() for p in reward_weights_raw.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(
                f"reward_weights must have exactly 3 comma-separated values (format,context,acc), got: {reward_weights_raw!r}"
            )
        opsd_config["reward_weights"] = [float(p) for p in parts]
    debug_cfg = opsd_config.setdefault("debug", {})
    detail_every = debug_cfg.get("detail_every", 10)
    if args.opsd_detail_every is not None:
        detail_every = max(0, args.opsd_detail_every)
        debug_cfg["detail_every"] = detail_every
    probe_on_generate = debug_cfg.get("probe_on_generate", False)
    if args.opsd_probe_on_generate is not None:
        probe_on_generate = args.opsd_probe_on_generate
        debug_cfg["probe_on_generate"] = probe_on_generate
    probe_first_token_logits = debug_cfg.get("probe_first_token_logits", True)
    if args.opsd_probe_first_token_logits is not None:
        probe_first_token_logits = args.opsd_probe_first_token_logits
        debug_cfg["probe_first_token_logits"] = probe_first_token_logits

    debug_enabled = opsd_debug.configure(
        enabled=args.opsd_debug if args.opsd_debug else debug_cfg.get("verbose"),
        detail_every=detail_every,
        probe_on_generate=probe_on_generate,
        probe_first_token_logits=probe_first_token_logits,
        probe_prompt_tail_tokens=debug_cfg.get("probe_prompt_tail_tokens", 16),
        probe_log_model_context=debug_cfg.get("probe_log_model_context", True),
    )
    if debug_enabled:
        opsd_debug.log_config("main", "resolved OPSD config", opsd_config)
        opsd_debug.log("main", "training entry", mode=mode, config_path=args.config)
    _log_run_config_summary(
        config_path=args.config,
        dataset_config=dataset_config,
        training_config=training_config,
        opsd_config=opsd_config,
        model_config=model_config,
        launch_config=launch_config,
        checkpoint_eval_config=checkpoint_eval_config,
    )

    # 2. Setup Environment
    # Default off unless --wandb or config launch.wandb_enabled / WANDB_API_KEY is set.
    launch_wandb = bool(launch_config.get("wandb_enabled", False))
    want_wandb = launch_wandb if args.wandb is None else args.wandb
    _log_startup("Initializing Accelerator (DeepSpeed/DDP may take a minute)...")
    accelerator, use_wandb = setup_accelerator_and_wandb(
        bf16=training_config['dyme_args']['bf16'],
        want_wandb=want_wandb,
    )
    if want_wandb and not use_wandb and args.wandb is True:
        raise RuntimeError(
            "wandb was requested (--wandb) but no API key is configured. "
            "Run `wandb login`, set WANDB_API_KEY, or use WANDB_MODE=offline."
        )
    if accelerator.is_main_process:
        if use_wandb:
            print("[DyME] wandb enabled for training logs")
        elif want_wandb:
            print(
                "[DyME] wandb disabled (no credentials). Training continues with report_to=none. "
                "Run `wandb login`, export WANDB_API_KEY, or pass --wandb after configuring."
            )
    device_id = accelerator.process_index
    opsd_debug.configure(
        enabled=debug_enabled,
        detail_every=detail_every,
        probe_on_generate=probe_on_generate,
        probe_first_token_logits=probe_first_token_logits,
        probe_prompt_tail_tokens=debug_cfg.get("probe_prompt_tail_tokens", 16),
        probe_log_model_context=debug_cfg.get("probe_log_model_context", True),
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    if debug_enabled:
        opsd_debug.log(
            "main",
            "accelerator initialized",
            process_index=accelerator.process_index,
            local_process_index=accelerator.local_process_index,
            num_processes=accelerator.num_processes,
            device=str(accelerator.device),
        )

    visible_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
    if visible_gpus == 0:
        raise RuntimeError("No CUDA devices are visible to this process.")
    if accelerator.num_processes > visible_gpus:
        raise RuntimeError(
            f"GPU/process mismatch: launched {accelerator.num_processes} distributed processes "
            f"but only {visible_gpus} CUDA device(s) are visible "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}).\n"
            f"Fix: accelerate launch --num_processes {visible_gpus} ...\n"
            f"Or: bash scripts/train_local_gpus.sh  (auto-detects {visible_gpus} GPU(s))"
        )
    if local_rank >= visible_gpus:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {visible_gpus} GPU(s) visible. "
            f"Reduce --num_processes to {visible_gpus}."
        )
    if accelerator.is_main_process:
        print(
            f"[DyME] Distributed launch OK: num_processes={accelerator.num_processes}, "
            f"visible_gpus={visible_gpus}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
            flush=True,
        )

    _log_startup("Loading student model (CPU staging first; GPU memory rises after DeepSpeed wrap)...")
    ds_zero_stage = deepspeed_zero_stage()
    if accelerator.is_main_process and is_deepspeed_accelerate_config():
        print(
            f"[DyME] DeepSpeed enabled via ACCELERATE_CONFIG "
            f"({os.environ.get('ACCELERATE_CONFIG', '<unset>')}), ZeRO stage={ds_zero_stage}",
            flush=True,
        )

    model, processor = load_model_and_processor(model_config)
    if resolve_gradient_checkpointing_enabled(launch_config):
        if should_disable_gradient_checkpointing():
            if accelerator.is_main_process:
                print(
                    "[DyME] gradient checkpointing skipped: incompatible with DeepSpeed ZeRO-1/2 "
                    "(multiple student forwards / checkpoint backward). "
                    "Use ZeRO-3, DDP, or DYME_GRADIENT_CHECKPOINTING=0.",
                    flush=True,
                )
        else:
            gc_kwargs = gradient_checkpointing_enable_kwargs()
            if gc_kwargs:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
            else:
                model.gradient_checkpointing_enable()
            if accelerator.is_main_process:
                mode = f"use_reentrant={gc_kwargs['use_reentrant']}" if gc_kwargs else "default"
                print(
                    f"[DyME] gradient checkpointing enabled on student "
                    f"(config launch.gradient_checkpointing_enable, {mode})",
                    flush=True,
                )

    cold_start_frac = float(
        opsd_config.get("gate", {}).get("sft_cold_start_frac", 0.0) or 0.0
    )
    cold_start_steps = opsd_config.get("gate", {}).get("sft_cold_start_steps")
    visual_needs_teacher = visual_supervision_needs_teacher(opsd_config)
    lazy_teacher = (bool(cold_start_steps) or cold_start_frac > 0.0) and not visual_needs_teacher

    teacher_model = None
    teacher_model_config = None
    if lazy_teacher:
        teacher_model_config = dict(model_config)
        if accelerator.is_main_process:
            print(
                "[DyME] SFT cold-start enabled: deferring 7B teacher load until RL phase",
                flush=True,
            )
    else:
        teacher_model = load_teacher_model(
            model_config,
            local_rank=local_rank,
            num_gpus=visible_gpus,
        )
        if visual_needs_teacher and accelerator.is_main_process:
            print(
                "[DyME] Visual supervision enabled: 7B teacher loaded before training",
                flush=True,
            )
    if accelerator.is_main_process and teacher_model is not None:
        _run_cross_model_vocab_checks(
            model,
            processor,
            teacher_model,
            model_config,
        )

    # 4. Prepare Datasets
    train_dataset, eval_dataset = prepare_datasets(
        task,
        dataset_config,
        mode=mode,
        checkpoint_eval_config=checkpoint_eval_config,
    )
    _log_dataset_status(train_dataset, dataset_config)

    # 5. Initialize Reward Calculator
    # checker = RewardCalculator(rl_config, client_config.copy(), gpu_id=device_id)
    # refiner = ContextRefiner(rl_config, client_config.copy(), gpu_id=device_id)

    checker, refiner, visual_meta = build_visual_supervision(
        rl_config,
        client_config,
        opsd_config,
        gpu_id=device_id,
        teacher_model=teacher_model,
        processor=processor,
    )
    # 6. Define Training Arguments
    dyme_args = dict(training_config['dyme_args'])
    resume_from_checkpoint = (
        args.resume_from_checkpoint
        or dyme_args.pop("resume_from_checkpoint", None)
        or os.environ.get("DYME_RESUME_FROM_CHECKPOINT", "").strip()
        or None
    )
    checkpoint_eval_config = resolve_checkpoint_eval_config(
        checkpoint_eval_config,
        task=task,
        eval_dataset=eval_dataset,
        dyme_args=dyme_args,
    )
    if checkpoint_eval_config.get("enabled"):
        # HF writes a new native checkpoint before applying its rotation.  A
        # crash in that tiny interval can leave the old + new directories;
        # rank zero alone repairs it only when both serialized states prove
        # the exact internal save relationship.  Its recovery/validation
        # result is broadcast before any worker proceeds, avoiding a deadlock
        # if rank zero finds a malformed layout.  It also rewrites an explicit
        # old checkpoint path that recovery just pruned, so every rank resumes
        # the retained new best checkpoint.
        resume_from_checkpoint, recovered_checkpoint = _coordinate_checkpoint_eval_recovery(
            accelerator=accelerator,
            output_dir=dyme_args.get("output_dir"),
            patience=checkpoint_eval_config["patience"],
            tie_policy=checkpoint_eval_config["tie_policy"],
            resume_from_checkpoint=resume_from_checkpoint,
        )
        if accelerator.is_main_process and recovered_checkpoint is not None:
            _log_startup(
                "Recovered interrupted checkpoint-eval save: "
                f"retained {recovered_checkpoint}"
            )
        _log_startup(
            "Checkpoint evaluation enabled: "
            f"split={checkpoint_eval_config['split']} "
            f"patience={checkpoint_eval_config['patience']} "
            "(student model stays in memory)"
        )
    if ds_zero_stage is not None and ds_zero_stage >= 3:
        dyme_args.setdefault("ds3_gather_for_generation", True)
    if not use_wandb:
        dyme_args["report_to"] = "none"
    training_args = GRPOConfig(**dyme_args)

    collate_fn_with_processor = partial(collate_fn, processor=processor)
    # 7. Initialize the Trainer
    dyme_trainer = DyMETrainer(
        model=model,
        checker=checker,
        refiner=refiner,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        processing_func=collate_fn_with_processor,
        task_name=task,
        end_flag=rl_config['end_flag'],
        opsd_config=opsd_config,
        teacher_model=teacher_model,
        teacher_model_config=teacher_model_config,
        visual_supervision_meta=visual_meta,
        checkpoint_eval_config=checkpoint_eval_config,
        callbacks=(
            [
                CheckpointEvaluationTriggerCallback(
                    enabled=True,
                    patience=checkpoint_eval_config["patience"],
                    tie_policy=checkpoint_eval_config["tie_policy"],
                    output_dir=training_args.output_dir,
                )
            ]
            if checkpoint_eval_config.get("enabled")
            else None
        ),
    )
    if accelerator.is_main_process:
        write_run_config_snapshot(
            output_dir=training_args.output_dir,
            config_path=args.config,
            model_config=model_config,
            training_config=training_config,
            rl_config=rl_config,
            client_config=client_config,
            dataset_config=dataset_config,
            opsd_config=opsd_config,
            checkpoint_eval_config=checkpoint_eval_config,
            training_args=training_args,
            generation_config=dyme_trainer.generation_config,
    )

    # 8. Start Training
    if accelerator.is_main_process and resume_from_checkpoint:
        print(f"[DyME] Resuming training from checkpoint: {resume_from_checkpoint}", flush=True)
    maybe_allow_trusted_torch_load(
        resume_from_checkpoint=resume_from_checkpoint,
        log=print if accelerator.is_main_process else (lambda _: None),
    )
    dyme_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if checkpoint_eval_config.get("enabled"):
        # CheckpointEvaluationPolicy writes complete resumable checkpoints at
        # improvement time.  Do not perform an unconditional final save: it
        # could overwrite the best model with a lower-scoring terminal model.
        if accelerator.is_main_process:
            best_checkpoint = find_best_checkpoint_path(dyme_trainer, training_args.output_dir)
            if best_checkpoint is None:
                raise RuntimeError(
                    "Checkpoint evaluation was enabled but no best checkpoint was reported "
                    f"under {training_args.output_dir!r}."
                )
            final_link = update_final_checkpoint_link(training_args.output_dir, best_checkpoint)
            best_score = getattr(dyme_trainer, "best_checkpoint_score", None)
            best_step = getattr(dyme_trainer, "best_checkpoint_step", None)
            if best_score is None:
                policy = find_checkpoint_evaluation_policy(dyme_trainer)
                state = getattr(policy, "state", None)
                best_score = getattr(state, "best_score", None)
                best_step = best_step if best_step is not None else getattr(state, "best_step", None)
            print(
                "[DyME] Checkpoint evaluation summary: "
                f"best_score={best_score!r} best_step={best_step!r} "
                f"best_checkpoint={best_checkpoint} final_checkpoint={final_link}",
                flush=True,
            )
    else:
        # Preserve the historical behavior for configs which explicitly
        # disable checkpoint evaluation (e.g. no-eval smoke runs).
        output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
        if accelerator.is_main_process and is_deepspeed_accelerate_config():
            print(
                "[DyME] Saving consolidated student checkpoint (DeepSpeed ZeRO gather if configured)...",
                flush=True,
            )
        dyme_trainer.save_model(output_dir)
        if accelerator.is_main_process:
            processor.save_pretrained(output_dir)
            print(f"Model and processor saved to {output_dir}")
if __name__ == "__main__":
    try:
        main()
    finally:
        destroy_distributed_process_group()
