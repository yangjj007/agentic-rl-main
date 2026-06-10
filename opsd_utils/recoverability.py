from typing import Any, Optional

import torch
import torch.nn.functional as F

from opsd_utils import debug_log as opsd_debug
from opsd_utils.privileged import build_privileged_context
from opsd_utils.prompt_builder import tokenize_teacher_prompt


def privileged_context_available(
    sample: dict[str, Any],
    provider_names: list[str],
    opsd_config: Optional[dict[str, Any]] = None,
) -> bool:
    suffix, teacher_images = build_privileged_context(
        sample,
        provider_names,
        opsd_config=opsd_config or {},
    )
    has_visual = len(teacher_images) > 1
    available = bool(suffix.strip()) or has_visual
    opsd_debug.log(
        "recoverability",
        "privileged_context_available",
        available=available,
        suffix_len=len(suffix.strip()),
        num_teacher_images=len(teacher_images),
        has_privileged_visual=has_visual,
        provider_names=provider_names,
    )
    return available


def logprob_gain_recoverable(
    model,
    processor,
    sample: dict[str, Any],
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    student_prompt_ids: torch.Tensor,
    student_prompt_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_sizes,
    provider_names: list[str],
    tau: float = 0.5,
    opsd_config: Optional[dict[str, Any]] = None,
) -> bool:
    """Compare mean log-prob gain on completion tokens (teacher vs student)."""
    opsd_config = opsd_config or {}
    suffix, teacher_images = build_privileged_context(
        sample,
        provider_names,
        opsd_config=opsd_config,
    )
    if not suffix.strip() and len(teacher_images) <= 1:
        return False

    if not teacher_images:
        from opsd_utils.privileged.image_utils import load_rgb

        full = load_rgb(sample.get("image"))
        teacher_images = [full] if full is not None else []

    teacher_batch = tokenize_teacher_prompt(
        processor,
        sample["prompt"],
        suffix,
        teacher_images,
    )
    device = student_prompt_ids.device
    teacher_prompt_ids = teacher_batch["input_ids"].to(device)
    teacher_prompt_mask = teacher_batch["attention_mask"].to(device)
    teacher_pixel_values = teacher_batch.get("pixel_values", pixel_values).to(device)
    teacher_image_sizes = teacher_batch.get("image_sizes", image_sizes)

    comp_len = int(completion_mask.sum().item())
    if comp_len == 0:
        return False

    student_input = torch.cat([student_prompt_ids, completion_ids[:comp_len].unsqueeze(0)], dim=1)
    student_attn = torch.cat(
        [student_prompt_mask, completion_mask[:comp_len].unsqueeze(0).long()], dim=1
    )
    teacher_input = torch.cat([teacher_prompt_ids, completion_ids[:comp_len].unsqueeze(0)], dim=1)
    teacher_attn = torch.cat(
        [teacher_prompt_mask, completion_mask[:comp_len].unsqueeze(0).long()], dim=1
    )

    with torch.no_grad():
        s_logits = model(
            input_ids=student_input,
            attention_mask=student_attn,
            pixel_values=pixel_values[:1] if pixel_values is not None else None,
            image_sizes=image_sizes,
        ).logits[:, -comp_len - 1 : -1, :]
        t_logits = _teacher_forward_with_oom_retry(
            model,
            teacher_input,
            teacher_attn,
            teacher_pixel_values,
            teacher_image_sizes,
            comp_len,
        )

        targets = completion_ids[:comp_len].unsqueeze(0)
        s_logp = F.log_softmax(s_logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
        t_logp = F.log_softmax(t_logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
        gain = (t_logp - s_logp).mean().item()
    return gain > tau


def _teacher_forward_with_oom_retry(model, input_ids, attention_mask, pixel_values, image_sizes, comp_len):
    try:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        ).logits[:, -comp_len - 1 : -1, :]
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or pixel_values is None:
            raise
        opsd_debug.log(
            "teacher_forward_oom",
            "teacher recoverability forward OOM, clearing cache and retrying",
            micro_batch_size=1,
            oom_retries=1,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        ).logits[:, -comp_len - 1 : -1, :]


def estimate_recoverable_flags(
    samples: list[dict[str, Any]],
    num_generations: int,
    opsd_config: dict,
    model=None,
    processor=None,
    completions_tensors: Optional[dict] = None,
) -> list[bool]:
    """
    One recoverability flag per prompt group.
    """
    gate = opsd_config.get("gate", {})
    method = gate.get("teacher_recoverable", "privileged_available")
    providers = opsd_config.get("privileged_providers", ["text"])
    tau = gate.get("recoverable_tau", 0.5)

    num_prompts = len(samples) // num_generations
    flags: list[bool] = []
    opsd_debug.log(
        "recoverability",
        "estimate_recoverable_flags enter",
        method=method,
        num_prompts=num_prompts,
        num_generations=num_generations,
        providers=providers,
        privileged_profile=opsd_config.get("privileged_profile", "hybrid"),
        tau=tau,
    )

    for p in range(num_prompts):
        sample = samples[p * num_generations]
        if method == "privileged_available":
            flag = privileged_context_available(sample, providers, opsd_config=opsd_config)
        elif method == "logprob_gain" and model is not None and processor is not None:
            assert completions_tensors is not None
            idx = p * num_generations
            with opsd_debug.timed("recoverability", f"logprob_gain prompt={p}"):
                flag = logprob_gain_recoverable(
                    model=model,
                    processor=processor,
                    sample=sample,
                    completion_ids=completions_tensors["completion_ids"][idx],
                    completion_mask=completions_tensors["completion_mask"][idx],
                    student_prompt_ids=completions_tensors["prompt_ids"][idx : idx + 1],
                    student_prompt_mask=completions_tensors["prompt_mask"][idx : idx + 1],
                    pixel_values=completions_tensors["pixel_values"][idx : idx + 1],
                    image_sizes=completions_tensors["image_sizes"],
                    provider_names=providers,
                    tau=tau,
                    opsd_config=opsd_config,
                )
        else:
            flag = privileged_context_available(sample, providers, opsd_config=opsd_config)
        flags.append(flag)
        opsd_debug.log(
            "recoverability",
            "prompt recoverability",
            prompt_index=p,
            recoverable=flag,
            has_privileged_visual=flag and opsd_config.get("privileged_profile") in ("visual", "hybrid"),
        )

    opsd_debug.log("recoverability", "estimate_recoverable_flags done", flags=flags)
    return flags
