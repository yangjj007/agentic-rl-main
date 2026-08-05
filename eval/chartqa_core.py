"""ChartQA evaluation that can run against a model already resident in memory.

The command-line evaluator in :mod:`eval.eval_chartqa` deliberately keeps model
loading at its edge.  Training code should import this module instead, pass its
live student model, and use the returned score for checkpoint selection.

All ranks must invoke :func:`evaluate_chartqa_in_memory` together.  The helper
uses a deterministic contiguous shard and pads its final/empty rank batches
with discarded, real examples.  Consequently every rank enters ``generate``
the same number of times with the same batch geometry, which is important for
ZeRO-3/FSDP generation contexts whose parameter gathers are collective
operations.
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Mapping, Optional, Sequence

from eval.distributed_eval_utils import distributed_batch_plan


_TEMPLATE_BEHAVIOR_KEYS = (
    "full_cot_template",
    "partial_cot_template",
    "goal_without_answer",
    "empty_cot_skeleton",
    "malformed_answer_section",
)
_OUTPUT_TYPE_KEYS = (
    "char_repeat_guang",
    "goalie",
    "answer_flag",
    "full_cot",
    "other",
)


@dataclass
class ChartQAEvaluationConfig:
    """Generation settings for :func:`evaluate_chartqa_in_memory`.

    A mapping with the same keys is accepted as well.  ``dummy_batch_factory``
    is an escape hatch for datasets whose first example cannot be reused as a
    valid dummy input.  It is called as
    ``factory(examples=..., batch_index=..., batch_size=..., config=...)`` and
    must return exactly ``batch_size`` already-prepared ChartQA examples.

    ``batch_generator`` is primarily useful for tests or custom model backends.
    Its signature is ``(generation_model, processor, batch, config)`` and it
    must return one generated string per item in ``batch``.
    """

    batch_size: int = 1
    max_new_tokens: int = 1024
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    force_left_padding: bool = True
    input_dtype: Any = None
    device: Any = None
    synced_gpus: Optional[bool] = None
    prompt_template: Optional[str] = None
    dummy_example: Optional[Mapping[str, Any]] = None
    dummy_batch_factory: Optional[Callable[..., Sequence[Mapping[str, Any]]]] = None
    batch_generator: Optional[Callable[..., Sequence[str]]] = None


def _config_value(config: ChartQAEvaluationConfig | Mapping[str, Any] | None, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _empty_template_behavior_counts() -> dict[str, int]:
    return {"total": 0, **{key: 0 for key in _TEMPLATE_BEHAVIOR_KEYS}}


def classify_output_type(text: str) -> str:
    """Keep the historical ChartQA output-type labels used by the CLI logs."""
    text = text or ""
    lower = text.lower()
    if "光" * 6 in text:
        return "char_repeat_guang"
    if "goalie" in lower:
        return "goalie"
    if lower.count("answer:") != 1:
        return "answer_flag"
    if all(key in lower for key in ("goal", "observation", "reasoning")):
        return "full_cot"
    return "other"


def split_chartqa_initial_context(text: str) -> tuple[str, str]:
    """Match the historical ``split_initial_context`` ChartQA parsing exactly."""
    text = str(text or "").lower()
    flag = "answer:"
    if flag in text:
        answer = text.split(flag)[-1].strip().strip(".")
        context = text.split(flag)[0].strip()
    else:
        context = text
        answer = text
    return context, answer


def _first_answer(value: Any) -> Optional[str]:
    """Return the first ChartQA label without treating a string as characters."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    try:
        if len(value) == 0:
            return None
        value = value[0]
    except (TypeError, KeyError, IndexError):
        pass
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def prepare_chartqa_examples(dataset: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize raw ChartQA rows or previously prepared rows for generation.

    Raw Hugging Face rows use ``image``, ``query``, and ``label``.  The returned
    shape intentionally mirrors the old evaluator's ``eval_datasets_all_prepared``
    entries so callers can also supply their own pre-prepared examples.
    Invalid rows (missing image, question, or answer) are skipped, matching the
    previous behavior for missing labels.
    """
    if dataset is None:
        raise ValueError("ChartQA evaluation dataset is required")

    prepared: list[dict[str, Any]] = []
    for raw_item in dataset:
        if not isinstance(raw_item, Mapping):
            try:
                raw_item = dict(raw_item)
            except (TypeError, ValueError) as exc:
                raise TypeError("Each ChartQA evaluation item must be mapping-like") from exc

        is_prepared = "model_input_text" in raw_item or "image_path" in raw_item
        image = raw_item.get("image_path") if is_prepared else raw_item.get("image")
        question = raw_item.get("model_input_text") if is_prepared else raw_item.get("query")
        answer = _first_answer(raw_item.get("answer") if is_prepared else raw_item.get("label"))

        if image is None or question is None or answer is None:
            continue

        question = str(question).strip()
        if not question:
            continue
        prepared.append(
            {
                "image_path": image,
                "model_input_text": question,
                "answer": answer,
                "original_question": str(raw_item.get("original_question", question)),
            }
        )
    return prepared


def shard_chartqa_examples(
    examples: Sequence[Mapping[str, Any]], *, num_processes: int, process_index: int
) -> list[Mapping[str, Any]]:
    """Return this rank's exact, contiguous ChartQA shard."""
    if process_index < 0 or process_index >= num_processes:
        raise ValueError(
            f"process_index must be in [0, {num_processes}), got {process_index}"
        )
    plan = distributed_batch_plan(
        total_items=len(examples), num_processes=num_processes, batch_size=1
    )
    start = sum(plan.local_item_counts[:process_index])
    end = start + plan.local_item_counts[process_index]
    return list(examples[start:end])


def _get_prompt_template(config: ChartQAEvaluationConfig | Mapping[str, Any] | None) -> str:
    configured = _config_value(config, "prompt_template", None)
    if configured is not None:
        return str(configured)
    # Import lazily: importing the core must not instantiate a model or force the
    # full training configuration into light-weight unit tests.
    from data_utils.rl_prompt import PROMPT_TEMPLATE

    return PROMPT_TEMPLATE


def _load_rgb_image(image_source: Any) -> Any:
    """Load a path or normalize an already decoded PIL image to RGB."""
    if isinstance(image_source, (str, bytes, os.PathLike, Path)):
        from PIL import Image

        with Image.open(image_source) as image:
            return image.convert("RGB")
    if hasattr(image_source, "convert"):
        return image_source.convert("RGB")
    raise TypeError(
        "ChartQA image must be a filesystem path or an image object with convert('RGB')"
    )


def _try_import_torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _is_floating_tensor(value: Any) -> bool:
    predicate = getattr(value, "is_floating_point", None)
    if callable(predicate):
        return bool(predicate())
    return bool(predicate) if predicate is not None else False


def _infer_input_dtype(model: Any, configured_dtype: Any) -> Any:
    if configured_dtype is not None:
        if isinstance(configured_dtype, str):
            torch = _try_import_torch()
            if torch is None or not hasattr(torch, configured_dtype):
                raise ValueError(f"Unknown or unavailable input dtype: {configured_dtype!r}")
            return getattr(torch, configured_dtype)
        return configured_dtype

    # Most transformer wrappers expose dtype directly.  Fall back through
    # ``module`` (DDP/DeepSpeed) and finally the first parameter.
    current = model
    for _ in range(3):
        dtype = getattr(current, "dtype", None)
        if dtype is not None:
            return dtype
        current = getattr(current, "module", None)
        if current is None:
            break
    try:
        return next(model.parameters()).dtype
    except (AttributeError, StopIteration, TypeError):
        torch = _try_import_torch()
        return getattr(torch, "bfloat16", None) if torch is not None else None


def _move_inputs_to_device(inputs: Mapping[str, Any], *, device: Any, input_dtype: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        value = value.to(device) if device is not None else value
        if input_dtype is not None and _is_floating_tensor(value):
            value = value.to(input_dtype)
        moved[key] = value
    return moved


def build_chartqa_batch_inputs(
    processor: Any,
    batch: Sequence[Mapping[str, Any]],
    *,
    device: Any,
    input_dtype: Any,
    prompt_template: str,
) -> dict[str, Any]:
    """Build the same multimodal prompts and processor inputs as the old CLI."""
    images = []
    formatted_prompts = []
    for item in batch:
        question = str(item["model_input_text"]).strip()
        question_with_tags = prompt_template.format(question=question)
        images.append(_load_rgb_image(item["image_path"]))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_with_tags},
                ],
            }
        ]
        try:
            formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            formatted_prompt = formatted_prompt.strip()
        except Exception:
            formatted_prompt = f"USER: <image>\n{question_with_tags}\nASSISTANT:"
        formatted_prompts.append(formatted_prompt)

    inputs = processor(
        text=formatted_prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return _move_inputs_to_device(inputs, device=device, input_dtype=input_dtype)


@contextmanager
def _generation_model(model: Any, generation_context: Any) -> Iterable[Any]:
    """Open a caller-supplied model-unwrapping context exactly once per eval."""
    context_or_model = generation_context
    if context_or_model is None:
        yield model
        return

    # A factory is the convenient form for ``unwrap_model_for_generation``.
    # Do not call a context manager object even if it happens to be callable.
    if callable(context_or_model) and not hasattr(context_or_model, "__enter__"):
        context_or_model = context_or_model()

    if hasattr(context_or_model, "__enter__") and hasattr(context_or_model, "__exit__"):
        with context_or_model as unwrapped_model:
            yield unwrapped_model if unwrapped_model is not None else model
    else:
        yield context_or_model if context_or_model is not None else model


def _inference_mode() -> ContextManager[Any]:
    torch = _try_import_torch()
    if torch is not None and hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    return nullcontext()


def _generation_kwargs(
    processor: Any,
    config: ChartQAEvaluationConfig | Mapping[str, Any] | None,
    *,
    num_processes: int,
) -> dict[str, Any]:
    do_sample = bool(_config_value(config, "do_sample", False))
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(_config_value(config, "max_new_tokens", 1024)),
        "do_sample": do_sample,
        "repetition_penalty": float(_config_value(config, "repetition_penalty", 1.0)),
    }
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            kwargs["eos_token_id"] = eos_token_id
    if do_sample:
        kwargs["temperature"] = max(float(_config_value(config, "temperature", 0.0)), 1e-5)
        kwargs["top_p"] = float(_config_value(config, "top_p", 1.0))

    # Explicitly synchronize decoding in distributed mode unless the caller opts
    # out.  This avoids rank-dependent early-EOS loops under ZeRO-3/FSDP.
    synced_gpus = _config_value(config, "synced_gpus", None)
    if synced_gpus is None:
        synced_gpus = num_processes > 1
    if synced_gpus:
        kwargs["synced_gpus"] = True
    return kwargs


def _default_batch_generator(
    generation_model: Any,
    processor: Any,
    batch: Sequence[Mapping[str, Any]],
    config: ChartQAEvaluationConfig | Mapping[str, Any] | None,
    *,
    device: Any,
    input_dtype: Any,
    prompt_template: str,
    num_processes: int,
) -> list[str]:
    """Prepare and generate one local batch for non-distributed callers.

    The distributed evaluation loop prepares default batches itself before
    calling ``generate`` so it can synchronize preparation failures across
    ranks.  Keeping this convenience wrapper preserves the small standalone
    helper while routing both paths through the same decode implementation.
    """
    inputs = build_chartqa_batch_inputs(
        processor,
        batch,
        device=device,
        input_dtype=input_dtype,
        prompt_template=prompt_template,
    )
    return _generate_prepared_chartqa_batch(
        generation_model,
        processor,
        inputs,
        generation_kwargs=_generation_kwargs(processor, config, num_processes=num_processes),
    )


def _generate_prepared_chartqa_batch(
    generation_model: Any,
    processor: Any,
    inputs: Mapping[str, Any],
    *,
    generation_kwargs: Mapping[str, Any],
) -> list[str]:
    """Generate from inputs that were already prepared and safety-synchronized."""
    generated_ids = _generate_chartqa_ids(
        generation_model,
        inputs,
        generation_kwargs=generation_kwargs,
    )
    return _decode_generated_chartqa_ids(processor, inputs, generated_ids)


def _generate_chartqa_ids(
    generation_model: Any,
    inputs: Mapping[str, Any],
    *,
    generation_kwargs: Mapping[str, Any],
) -> Any:
    """Run model generation after the caller synchronized input preparation.

    In ZeRO-3/FSDP mode generation itself can contain collectives.  A rank-local
    failure *inside* that call is handled by the distributed backend; do not
    attempt a second object collective until every healthy rank has returned.
    Decode and scoring, which happen only after this returns, are synchronized
    separately by :func:`evaluate_chartqa_in_memory`.
    """
    _validate_prepared_chartqa_batch_inputs(inputs)
    generated_ids = generation_model.generate(
        **inputs,
        **generation_kwargs,
    )
    return getattr(generated_ids, "sequences", generated_ids)


def _decode_generated_chartqa_ids(
    processor: Any,
    inputs: Mapping[str, Any],
    generated_ids: Any,
) -> list[str]:
    """Decode a completed generation without performing model collectives."""
    input_ids = inputs.get("input_ids")
    # ``input_ids`` was validated before generation; the explicit assertion
    # gives type checkers a non-optional value here.
    assert input_ids is not None
    input_ids_length = input_ids.shape[1]
    newly_generated_ids = generated_ids[:, input_ids_length:]
    texts = processor.batch_decode(newly_generated_ids, skip_special_tokens=True)
    return [str(text).strip(".").strip() for text in texts]


def _validate_prepared_chartqa_batch_inputs(inputs: Mapping[str, Any]) -> None:
    """Reject malformed processor output before a distributed ``generate`` call."""
    if "input_ids" not in inputs:
        raise KeyError("ChartQA processor inputs must contain input_ids")
    input_ids = inputs["input_ids"]
    shape = getattr(input_ids, "shape", None)
    try:
        input_width = shape[1]
    except (TypeError, IndexError, KeyError) as exc:
        raise ValueError(
            "ChartQA processor input_ids must be a rank-2 tensor-like value"
        ) from exc
    try:
        if int(input_width) < 0:
            raise ValueError("ChartQA processor input_ids has an invalid negative width")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("ChartQA processor"):
            raise
        raise ValueError(
            "ChartQA processor input_ids must expose a numeric sequence width"
        ) from exc


def _gather_batch_preparation_statuses(
    accelerator: Any,
    local_status: Mapping[str, Any],
) -> list[Any]:
    """All-gather one small pre-generation readiness record per rank.

    This intentionally uses the process group's object collective instead of
    gathering a tensor flag and then trying to reconstruct a remote exception.
    The complete, tiny record lets every rank raise the same useful error
    *before* any rank reaches ZeRO-3/FSDP's collective ``generate`` path.
    """
    num_processes = int(getattr(accelerator, "num_processes", 1))
    torch = _try_import_torch()
    if torch is None:
        raise RuntimeError("Distributed ChartQA batch preparation requires torch")
    try:
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover - torch ships this in training environments
        raise RuntimeError(
            "Distributed ChartQA batch preparation requires torch.distributed"
        ) from exc
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Distributed ChartQA batch preparation requires an initialized "
            "torch.distributed process group"
        )
    world_size = int(dist.get_world_size())
    if world_size != num_processes:
        raise RuntimeError(
            "Distributed ChartQA batch preparation found a process-group size "
            f"mismatch: accelerator.num_processes={num_processes}, world_size={world_size}"
        )
    gathered: list[Any] = [None] * num_processes
    dist.all_gather_object(gathered, dict(local_status))
    return gathered


def _batch_stage_failure_message(
    statuses: Sequence[Any],
    *,
    batch_index: int,
    num_processes: int,
    stage: str,
) -> tuple[str, int | None]:
    """Validate gathered rank records and format one deterministic error."""
    if stage == "preparation":
        noun = "preparation"
        action = "before distributed generation"
    else:
        noun = stage
        action = "during distributed evaluation"
    if len(statuses) != num_processes:
        return (
            f"ChartQA distributed batch {noun} received an unexpected number "
            f"of rank records {action} for batch {batch_index}: "
            f"expected={num_processes}, actual={len(statuses)}",
            None,
        )

    failures: list[tuple[int, str, str]] = []
    seen_ranks: set[int] = set()
    for status in statuses:
        if not isinstance(status, Mapping):
            return (
                f"ChartQA distributed batch {noun} received an invalid rank "
                f"record {action} for batch {batch_index}",
                None,
            )
        rank = status.get("rank")
        ok = status.get("ok")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 0
            or rank >= num_processes
            or rank in seen_ranks
            or not isinstance(ok, bool)
        ):
            return (
                f"ChartQA distributed batch {noun} received an invalid rank "
                f"record {action} for batch {batch_index}",
                None,
            )
        seen_ranks.add(rank)
        if not ok:
            error_type = str(status.get("error_type") or "RuntimeError")
            error = str(status.get("error") or "unknown batch preparation error")
            failures.append((rank, error_type, error))

    if len(seen_ranks) != num_processes:
        return (
            f"ChartQA distributed batch {noun} received incomplete rank "
            f"records {action} for batch {batch_index}",
            None,
        )
    if not failures:
        return "", None

    # ``all_gather_object`` returns rank order, but sorting makes the exact
    # exception independent of backend implementation details.
    rank, error_type, error = min(failures, key=lambda value: value[0])
    return (
        f"ChartQA batch {noun} failed {action} "
        f"(batch={batch_index}, rank={rank}, {error_type}: {error})",
        rank,
    )


def _synchronize_batch_stage(
    accelerator: Any,
    *,
    batch_index: int,
    local_error: Exception | None,
    stage: str,
) -> None:
    """Raise a shared batch-stage error before ranks can diverge.

    ``preparation`` is synchronized before any rank can enter collective
    generation.  ``generation/scoring`` is synchronized after a batch's model
    call, decode, and metric work so a rank-local failure cannot strand healthy
    peers at the next batch's readiness or final reduction collective.  The
    single-GPU path deliberately performs no import or collective work.
    """
    num_processes = int(getattr(accelerator, "num_processes", 1))
    if num_processes <= 1:
        if local_error is not None:
            raise local_error
        return

    rank = int(getattr(accelerator, "process_index", 0))
    if rank < 0 or rank >= num_processes:
        raise RuntimeError(
            "ChartQA distributed batch preparation received an invalid accelerator "
            f"process_index={rank} for num_processes={num_processes}"
        )
    status: dict[str, Any] = {"rank": rank, "ok": local_error is None}
    if local_error is not None:
        status["error_type"] = type(local_error).__name__
        try:
            detail = str(local_error)
        except Exception:  # pragma: no cover - defensive for malformed exception classes
            detail = "unprintable batch preparation exception"
        status["error"] = detail or "batch preparation raised without a message"

    message, failing_rank = _batch_stage_failure_message(
        _gather_batch_preparation_statuses(accelerator, status),
        batch_index=batch_index,
        num_processes=num_processes,
        stage=stage,
    )
    if message:
        if local_error is not None and failing_rank == rank:
            raise RuntimeError(message) from local_error
        raise RuntimeError(message)


def _synchronize_batch_preparation(
    accelerator: Any,
    *,
    batch_index: int,
    local_error: Exception | None,
) -> None:
    """Raise a shared input-preparation error before distributed ``generate``."""
    _synchronize_batch_stage(
        accelerator,
        batch_index=batch_index,
        local_error=local_error,
        stage="preparation",
    )


def _synchronize_batch_generation_and_scoring(
    accelerator: Any,
    *,
    batch_index: int,
    local_error: Exception | None,
) -> None:
    """Synchronize post-generation/decode/scoring failures across ranks."""
    _synchronize_batch_stage(
        accelerator,
        batch_index=batch_index,
        local_error=local_error,
        stage="generation/scoring",
    )


def _dummy_batch(
    examples: Sequence[Mapping[str, Any]],
    *,
    batch_index: int,
    batch_size: int,
    config: ChartQAEvaluationConfig | Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    factory = _config_value(config, "dummy_batch_factory", None)
    if factory is not None:
        batch = list(
            factory(
                examples=examples,
                batch_index=batch_index,
                batch_size=batch_size,
                config=config,
            )
        )
    else:
        example = _config_value(config, "dummy_example", None)
        if example is None:
            example = examples[0] if examples else None
        batch = [example] * batch_size if example is not None else []
    if not batch:
        raise RuntimeError(
            "A distributed ChartQA rank has no real batch and no valid dummy example. "
            "Provide checkpoint_eval.dummy_example or dummy_batch_factory."
        )
    if len(batch) != batch_size:
        raise RuntimeError(
            "ChartQA dummy_batch_factory must return exactly one full dummy batch: "
            f"expected={batch_size}, actual={len(batch)}"
        )
    return batch


def _reduce_score_count(
    accelerator: Any,
    *,
    score_sum: float,
    count: int,
) -> tuple[float, int]:
    """All-reduce the only numeric values needed for exact global accuracy."""
    num_processes = int(getattr(accelerator, "num_processes", 1))
    if num_processes <= 1:
        return score_sum, count

    torch = _try_import_torch()
    if torch is None:
        raise RuntimeError("Distributed ChartQA evaluation requires torch")
    payload = torch.tensor(
        [score_sum, float(count)],
        dtype=torch.float64,
        device=getattr(accelerator, "device", None),
    )
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)
            return float(payload[0].item()), int(round(float(payload[1].item())))
    except (ImportError, RuntimeError):
        pass

    gather = getattr(accelerator, "gather", None)
    if callable(gather):
        gathered = gather(payload)
        gathered = gathered.reshape(-1, 2)
        return (
            float(gathered[:, 0].sum().item()),
            int(round(float(gathered[:, 1].sum().item()))),
        )
    raise RuntimeError(
        "Distributed ChartQA evaluation needs an initialized torch.distributed group "
        "or accelerator.gather()."
    )


def _reduce_fixed_count_mapping(
    accelerator: Any,
    value: Mapping[str, int],
    *,
    keys: Sequence[str],
) -> dict[str, int]:
    """All-reduce a fixed diagnostic counter mapping without gathering texts.

    The dynamic output-type labels are first made deterministic by sorting their
    union through ``all_gather_object`` only when necessary.  The actual counts
    (and all template counters) use a compact tensor all-reduce.  This keeps the
    training path's cross-rank traffic bounded regardless of response length.
    """
    num_processes = int(getattr(accelerator, "num_processes", 1))
    if num_processes <= 1:
        return {key: int(value.get(key, 0)) for key in keys}

    torch = _try_import_torch()
    if torch is None:
        raise RuntimeError("Distributed ChartQA evaluation requires torch")
    payload = torch.tensor(
        [int(value.get(key, 0)) for key in keys],
        dtype=torch.int64,
        device=getattr(accelerator, "device", None),
    )
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)
            return {key: int(payload[index].item()) for index, key in enumerate(keys)}
    except (ImportError, RuntimeError):
        pass

    gather = getattr(accelerator, "gather", None)
    if callable(gather):
        gathered = gather(payload).reshape(-1, len(keys))
        return {
            key: int(gathered[:, index].sum().item())
            for index, key in enumerate(keys)
        }
    raise RuntimeError(
        "Distributed ChartQA evaluation needs an initialized torch.distributed group "
        "or accelerator.gather()."
    )


def _summarize_template_behavior(texts: list[str]) -> dict[str, int]:
    try:
        from eval.output_behavior import summarize_output_behavior_counts

        return summarize_output_behavior_counts(texts)
    except (ImportError, ModuleNotFoundError):
        # Keep the core importable in lightweight environments.  Real training
        # installations have the diagnostics dependency and take the branch above.
        return _empty_template_behavior_counts() | {"total": len(texts)}


def _restore_model_mode(model: Any, was_training: Any) -> None:
    if was_training is None:
        return
    train = getattr(model, "train", None)
    if callable(train):
        try:
            train(bool(was_training))
            return
        except TypeError:
            pass
    if bool(was_training):
        if callable(train):
            train()
    else:
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()


def evaluate_chartqa_in_memory(
    model: Any,
    processor: Any,
    accelerator: Any,
    dataset: Iterable[Mapping[str, Any]],
    generation_context: Any = None,
    config: ChartQAEvaluationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a live student model on ChartQA without loading any weights.

    ``generation_context`` may be a context manager, a factory returning one, or
    an already-unwrapped generation model.  For the training path, pass a factory
    that opens ``unwrap_model_for_generation(model_wrapped, accelerator, ...)``;
    the core then uses that yielded model directly.

    The returned ``checkpoint_score``, ``accuracy``, and ``chartqa_accuracy`` are
    identical aliases so a training callback can log one and use the same value
    for best-checkpoint comparison.
    """
    examples = prepare_chartqa_examples(dataset)
    if not examples:
        raise ValueError("ChartQA evaluation has no valid examples after filtering")

    batch_size = int(_config_value(config, "batch_size", 1))
    if batch_size < 1:
        raise ValueError("ChartQA evaluation batch_size must be >= 1")
    num_processes = int(getattr(accelerator, "num_processes", 1))
    process_index = int(getattr(accelerator, "process_index", 0))
    if num_processes < 1:
        raise ValueError("accelerator.num_processes must be >= 1")
    local_examples = shard_chartqa_examples(
        examples, num_processes=num_processes, process_index=process_index
    )
    batch_plan = distributed_batch_plan(
        total_items=len(examples), num_processes=num_processes, batch_size=batch_size
    )

    device = _config_value(config, "device", None)
    if device is None:
        device = getattr(accelerator, "device", None)
    input_dtype = _infer_input_dtype(model, _config_value(config, "input_dtype", None))
    prompt_template = _get_prompt_template(config)
    custom_batch_generator = _config_value(config, "batch_generator", None)

    tokenizer = getattr(processor, "tokenizer", None)
    old_padding_side = getattr(tokenizer, "padding_side", None) if tokenizer is not None else None
    force_left_padding = bool(_config_value(config, "force_left_padding", True))
    was_training = getattr(model, "training", None)
    generation_was_training = None
    score_sum = 0.0
    local_texts: list[str] = []
    output_type_counts: dict[str, int] = {}
    dummy_batches = 0
    generation_model = model

    if tokenizer is not None and force_left_padding and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    evaluate = getattr(model, "eval", None)
    if callable(evaluate):
        evaluate()

    try:
        with _generation_model(model, generation_context) as generation_model:
            if generation_model is not model:
                generation_was_training = getattr(generation_model, "training", None)
                generation_evaluate = getattr(generation_model, "eval", None)
                if callable(generation_evaluate):
                    generation_evaluate()
            with _inference_mode():
                for batch_index in range(batch_plan.sync_batches):
                    start = batch_index * batch_size
                    current_batch: list[Mapping[str, Any]] = []
                    real_item_count = 0
                    prepared_inputs: Mapping[str, Any] | None = None
                    generation_kwargs: Mapping[str, Any] | None = None
                    preparation_error: Exception | None = None
                    try:
                        current_batch = list(local_examples[start : start + batch_size])
                        real_item_count = len(current_batch)
                        if real_item_count < batch_size:
                            dummy_items = _dummy_batch(
                                examples,
                                batch_index=batch_index,
                                batch_size=batch_size - real_item_count,
                                config=config,
                            )
                            current_batch = current_batch + dummy_items
                            dummy_batches += 1

                        # Processor/image work must finish on every rank before
                        # any one rank can enter collective generation.  A custom
                        # batch generator owns its own preparation by contract,
                        # so only its common batch construction is checked here.
                        if custom_batch_generator is None:
                            prepared_inputs = build_chartqa_batch_inputs(
                                processor,
                                current_batch,
                                device=device,
                                input_dtype=input_dtype,
                                prompt_template=prompt_template,
                            )
                            generation_kwargs = _generation_kwargs(
                                processor,
                                config,
                                num_processes=num_processes,
                            )
                            _validate_prepared_chartqa_batch_inputs(prepared_inputs)
                    except Exception as exc:
                        # Do not raise yet in distributed mode.  Every peer
                        # must enter the readiness collective below so it can
                        # avoid an otherwise deadlocking ``generate`` call.
                        preparation_error = exc

                    _synchronize_batch_preparation(
                        accelerator,
                        batch_index=batch_index,
                        local_error=preparation_error,
                    )
                    if custom_batch_generator is None:
                        # A successful local readiness record guarantees these
                        # values were built above.  Keep the checks explicit so
                        # a malformed custom processor cannot reach generate.
                        if prepared_inputs is None or generation_kwargs is None:
                            raise RuntimeError(
                                "ChartQA batch preparation succeeded without generation inputs"
                            )
                        generated_ids = _generate_chartqa_ids(
                            generation_model,
                            prepared_inputs,
                            generation_kwargs=generation_kwargs,
                        )
                        try:
                            predictions = _decode_generated_chartqa_ids(
                                processor,
                                prepared_inputs,
                                generated_ids,
                            )
                        except Exception as exc:
                            # Decoding happens after every rank has returned
                            # from the model's collective generation path.
                            generation_or_scoring_error: Exception | None = exc
                        else:
                            generation_or_scoring_error = None
                    else:
                        # Custom generators own their generation contract.  A
                        # custom distributed implementation must ensure its own
                        # model-call collectives complete; its returned texts
                        # and all scoring below are still failure-synchronized.
                        try:
                            predictions = list(
                                custom_batch_generator(
                                    generation_model, processor, current_batch, config
                                )
                            )
                        except Exception as exc:
                            generation_or_scoring_error = exc
                        else:
                            generation_or_scoring_error = None
                    if generation_or_scoring_error is None:
                        try:
                            if len(predictions) != len(current_batch):
                                raise RuntimeError(
                                    "ChartQA batch generator returned a different number of predictions "
                                    f"({len(predictions)}) than inputs ({len(current_batch)})"
                                )
                            for item, full_prediction in zip(
                                current_batch[:real_item_count], predictions[:real_item_count]
                            ):
                                full_prediction = str(full_prediction).strip(".").strip()
                                _, parsed_prediction = split_chartqa_initial_context(full_prediction)
                                if not parsed_prediction.strip():
                                    parsed_prediction = full_prediction
                                # Imported here to keep module import side-effect
                                # free for training/test environments that do not
                                # need scoring.
                                from data_utils.chart.evaluator import eval_one_chart

                                score = float(eval_one_chart(parsed_prediction, item["answer"]))
                                if not math.isfinite(score):
                                    raise ValueError(
                                        "ChartQA metric returned a non-finite score for "
                                        f"{item['original_question']!r}"
                                    )
                                score_sum += score
                                local_texts.append(full_prediction)
                                output_type = classify_output_type(full_prediction)
                                output_type_counts[output_type] = output_type_counts.get(output_type, 0) + 1
                        except Exception as exc:
                            # Decode/scoring failures happen after default
                            # generation returned, so every rank can rendezvous
                            # here before a peer advances to the next batch.
                            generation_or_scoring_error = exc
                    _synchronize_batch_generation_and_scoring(
                        accelerator,
                        batch_index=batch_index,
                        local_error=generation_or_scoring_error,
                    )
    finally:
        if generation_was_training is not None and generation_model is not model:
            _restore_model_mode(generation_model, generation_was_training)
        if tokenizer is not None and old_padding_side is not None and hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = old_padding_side
        _restore_model_mode(model, was_training)

    local_count = len(local_texts)
    if local_count != len(local_examples):
        raise RuntimeError(
            "ChartQA local processed count does not match its deterministic shard: "
            f"processed={local_count}, shard={len(local_examples)}"
        )
    global_score_sum, global_count = _reduce_score_count(
        accelerator, score_sum=score_sum, count=local_count
    )
    if global_count != len(examples):
        raise RuntimeError(
            "ChartQA global processed count does not match the prepared dataset: "
            f"processed={global_count}, expected={len(examples)}"
        )
    if global_count == 0:
        raise ValueError("ChartQA evaluation produced zero scored examples")

    global_output_types = _reduce_fixed_count_mapping(
        accelerator, output_type_counts, keys=_OUTPUT_TYPE_KEYS
    )
    local_template_behavior = _summarize_template_behavior(local_texts)
    global_template_behavior = _reduce_fixed_count_mapping(
        accelerator,
        local_template_behavior,
        keys=tuple(_empty_template_behavior_counts()),
    )
    accuracy = global_score_sum / global_count
    return {
        "checkpoint_score": accuracy,
        "accuracy": accuracy,
        "chartqa_accuracy": accuracy,
        "score_sum": global_score_sum,
        "sample_count": global_count,
        "processed_count": global_count,
        "total_items": len(examples),
        "output_type_counts": global_output_types,
        "template_behavior_counts": global_template_behavior,
        "local_score_sum": score_sum,
        "local_sample_count": local_count,
        "local_dummy_batches": dummy_batches,
        "num_processes": num_processes,
        "process_index": process_index,
        "is_main_process": bool(getattr(accelerator, "is_main_process", process_index == 0)),
    }


# Short alias for callers that prefer the old evaluator's terminology.
evaluate_chartqa = evaluate_chartqa_in_memory


def print_chartqa_evaluation(result: Mapping[str, Any], *, print_fn: Callable[..., None] = print) -> None:
    """Print the legacy final-summary lines consumed by existing log parsers."""
    # Several existing evaluation runners use this marker to distinguish a
    # completed evaluation from a partial/failed log.  Keep it even though the
    # in-memory core itself no longer emits intermediate progress reports.
    print_fn("--- Final Report ---")
    print_fn(
        f"Global samples processed: {int(result['sample_count'])} / {int(result['total_items'])}"
    )
    print_fn(f"Current Global Mean Accuracy: {float(result['accuracy']):.4f}")
    print_fn(f"Output type counts: {dict(result['output_type_counts'])}")
    print_fn(f"Template behavior counts: {dict(result['template_behavior_counts'])}")
