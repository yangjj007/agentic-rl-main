from typing import Any, Optional

import torch
import torch.nn.functional as F

from opsd_utils.privileged import build_privileged_context
from opsd_utils.prompt_builder import tokenize_teacher_prompt


def privileged_context_available(sample: dict[str, Any], provider_names: list[str]) -> bool:
    suffix, _ = build_privileged_context(sample, provider_names)
    return bool(suffix.strip())


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
) -> bool:
    """Compare mean log-prob gain on completion tokens (teacher vs student)."""
    suffix, teacher_image = build_privileged_context(sample, provider_names)
    if not suffix.strip():
        return False

    teacher_batch = tokenize_teacher_prompt(
        processor,
        sample["prompt"],
        suffix,
        teacher_image if teacher_image is not None else sample.get("image"),
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
        t_logits = model(
            input_ids=teacher_input,
            attention_mask=teacher_attn,
            pixel_values=teacher_pixel_values,
            image_sizes=teacher_image_sizes,
        ).logits[:, -comp_len - 1 : -1, :]

        targets = completion_ids[:comp_len].unsqueeze(0)
        s_logp = F.log_softmax(s_logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
        t_logp = F.log_softmax(t_logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
        gain = (t_logp - s_logp).mean().item()
    return gain > tau


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

    for p in range(num_prompts):
        sample = samples[p * num_generations]
        if method == "privileged_available":
            flags.append(privileged_context_available(sample, providers))
        elif method == "logprob_gain" and model is not None and processor is not None:
            assert completions_tensors is not None
            idx = p * num_generations
            flags.append(
                logprob_gain_recoverable(
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
                )
            )
        else:
            flags.append(privileged_context_available(sample, providers))

    return flags
