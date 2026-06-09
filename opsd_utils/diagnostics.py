"""Periodic full-detail diagnostics for weak reward / gradient signals."""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from opsd_utils import debug_log as opsd_debug


def _tensor_stats(t: torch.Tensor, name: str) -> dict[str, Any]:
    if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
        return {name: "empty"}
    with torch.no_grad():
        flat = t.detach().float().reshape(-1)
        return {
            f"{name}/shape": tuple(t.shape),
            f"{name}/mean": float(flat.mean().item()),
            f"{name}/std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
            f"{name}/min": float(flat.min().item()),
            f"{name}/max": float(flat.max().item()),
            f"{name}/abs_mean": float(flat.abs().mean().item()),
        }


def _preview_text(text: str, max_len: int = 320) -> str:
    text = (text or "").replace("\n", "\\n")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text or "<EMPTY>"


def jsd_token_stats(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.5,
    top_k: int = 5,
) -> dict[str, Any]:
    """Token-level JSD breakdown without building the graph."""
    with torch.no_grad():
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_probs = student_log_probs.exp()
        teacher_probs = teacher_log_probs.exp()

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

        per_token_jsd = jsd.sum(dim=-1)
        m = mask.float()
        valid = m.sum().clamp(min=1.0)
        per_token_jsd_masked = per_token_jsd * m

        # Top-1 agreement on completion tokens
        s_top = student_logits.argmax(dim=-1)
        t_top = teacher_logits.argmax(dim=-1)
        agree = ((s_top == t_top).float() * m).sum() / valid

        # Mean L2 distance of log-probs on gold completion tokens (if provided elsewhere)
        logprob_l2 = ((student_log_probs - teacher_log_probs) ** 2).sum(dim=-1)
        logprob_l2_masked = (logprob_l2 * m).sum() / valid

        # Entropy gap
        s_ent = -(student_probs * student_log_probs).sum(dim=-1)
        t_ent = -(teacher_probs * teacher_log_probs).sum(dim=-1)
        ent_gap = ((s_ent - t_ent).abs() * m).sum() / valid

        stats: dict[str, Any] = {
            "jsd_per_token_mean": float((per_token_jsd_masked.sum() / valid).item()),
            "jsd_per_token_max": float(per_token_jsd[m.bool()].max().item()) if m.any() else 0.0,
            "jsd_per_token_min": float(per_token_jsd[m.bool()].min().item()) if m.any() else 0.0,
            "top1_agreement": float(agree.item()),
            "logprob_l2_mean": float(logprob_l2_masked.item()),
            "entropy_gap_mean": float(ent_gap.item()),
            "mask_valid_tokens": int(valid.item()),
        }

        if m.any():
            idx = int((per_token_jsd * m).argmax().item())
            pos = idx % per_token_jsd.size(-1)
            stats["max_jsd_token_pos"] = pos
            stats["max_jsd_value"] = float(per_token_jsd.reshape(-1)[idx].item())
            s_topk = torch.topk(student_probs[0, pos], k=min(top_k, student_probs.size(-1)))
            t_topk = torch.topk(teacher_probs[0, pos], k=min(top_k, teacher_probs.size(-1)))
            stats["student_topk_at_max_jsd"] = [
                (int(i), float(p)) for i, p in zip(s_topk.indices.tolist(), s_topk.values.tolist())
            ]
            stats["teacher_topk_at_max_jsd"] = [
                (int(i), float(p)) for i, p in zip(t_topk.indices.tolist(), t_topk.values.tolist())
            ]
        return stats


def log_generation_diagnostics(
    *,
    global_step: int,
    completions: list[str],
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    is_eos: torch.Tensor,
    max_completion_length: int,
    num_generations: int,
    sample_count: int = 4,
) -> None:
    if not opsd_debug.should_log_detail(global_step):
        return

    opsd_debug.log_detail_banner(global_step, "GENERATION & COMPLETION")

    with torch.no_grad():
        lengths = completion_mask.sum(dim=1).float()
        has_eos = is_eos.any(dim=1)
        eos_rate = has_eos.float().mean().item()
        clipped = (~has_eos).float().mean().item()
        all_pad_or_zero = (lengths == 0).float().mean().item()
        at_max_len = (lengths >= max_completion_length - 1).float().mean().item()

    fields: dict[str, Any] = {
        "batch_size": completion_ids.size(0),
        "num_generations": num_generations,
        "completion_max_len": completion_ids.size(1),
        "effective_tokens_mean": float(lengths.mean().item()),
        "effective_tokens_min": float(lengths.min().item()),
        "effective_tokens_max": float(lengths.max().item()),
        "eos_terminated_rate": eos_rate,
        "clipped_no_eos_rate": clipped,
        "zero_length_rate": all_pad_or_zero,
        "at_max_length_rate": at_max_len,
    }
    opsd_debug.log_detail("generation", "completion mask summary", **fields)

    n = min(sample_count, len(completions))
    for i in range(n):
        gen_group = i // num_generations if num_generations else 0
        opsd_debug.log_detail(
            "generation",
            f"sample[{i}]",
            group=gen_group,
            effective_tokens=int(lengths[i].item()),
            has_eos=bool(has_eos[i].item()),
            text=_preview_text(completions[i]),
            raw_token_head=completion_ids[i, :12].tolist(),
        )


def log_reward_diagnostics(
    *,
    global_step: int,
    format_rewards: torch.Tensor,
    acc_rewards: torch.Tensor,
    context_rewards: torch.Tensor,
    all_rewards: torch.Tensor,
    advantages: torch.Tensor,
    reward_weights: torch.Tensor,
    num_generations: int,
    answers: Optional[list[str]] = None,
    completions: Optional[list[str]] = None,
    sample_count: int = 4,
) -> None:
    if not opsd_debug.should_log_detail(global_step):
        return

    opsd_debug.log_detail_banner(global_step, "REWARD & ADVANTAGE")

    w = reward_weights.detach().float()
    weighted = (
        format_rewards * w[0] + context_rewards * w[1] + acc_rewards * w[2]
    )

    fields: dict[str, Any] = {
        "reward_weights": w.tolist(),
        "format_sum": float(format_rewards.sum().item()),
        "acc_sum": float(acc_rewards.sum().item()),
        "context_sum": float(context_rewards.sum().item()),
        "total_sum": float(all_rewards.sum().item()),
        "format_zero_rate": float((format_rewards == 0).float().mean().item()),
        "acc_zero_rate": float((acc_rewards == 0).float().mean().item()),
        "context_zero_rate": float((context_rewards == 0).float().mean().item()),
        "weighted_mean": float(weighted.mean().item()),
        "weighted_std": float(weighted.std(unbiased=False).item()) if weighted.numel() > 1 else 0.0,
    }
    adv_flat = advantages.reshape(-1)
    fields.update(
        {
            "advantage_mean": float(adv_flat.mean().item()),
            "advantage_std": float(adv_flat.std(unbiased=False).item()) if adv_flat.numel() > 1 else 0.0,
            "advantage_zero_rate": float((adv_flat.abs() < 1e-8).float().mean().item()),
            "advantage_min": float(adv_flat.min().item()),
            "advantage_max": float(adv_flat.max().item()),
        }
    )
    opsd_debug.log_detail("reward", "aggregate reward stats", **fields)

    n = min(sample_count, format_rewards.numel())
    for i in range(n):
        g = i // num_generations if num_generations else 0
        extra: dict[str, Any] = {
            "group": g,
            "format": float(format_rewards.view(-1)[i].item()),
            "acc": float(acc_rewards.view(-1)[i].item()),
            "context": float(context_rewards.view(-1)[i].item()),
            "weighted": float(weighted.view(-1)[i].item()),
            "advantage": float(adv_flat.view(-1)[i].item()) if i < adv_flat.numel() else None,
        }
        if answers and i < len(answers):
            extra["gold_answer"] = _preview_text(str(answers[i // num_generations]), 80)
        if completions and i < len(completions):
            extra["completion"] = _preview_text(completions[i], 160)
        opsd_debug.log_detail("reward", f"per_sample[{i}]", **extra)


def log_routing_diagnostics(
    *,
    global_step: int,
    opsd_active: bool,
    opsd_mask_list: list[bool],
    has_correct: torch.Tensor,
    completion_modes: Optional[list[int]] = None,
    recoverable_flags: Optional[list[bool]] = None,
    completion_advantages: Optional[torch.Tensor] = None,
    completion_mask: Optional[torch.Tensor] = None,
) -> None:
    if not opsd_debug.should_log_detail(global_step):
        return

    opsd_debug.log_detail_banner(global_step, "OPSD ROUTING & MASK")

    opsd_true = sum(opsd_mask_list)
    fields: dict[str, Any] = {
        "opsd_active": opsd_active,
        "opsd_mask_true": opsd_true,
        "opsd_mask_false": len(opsd_mask_list) - opsd_true,
        "opsd_mask_ratio": opsd_true / max(len(opsd_mask_list), 1),
        "has_correct": has_correct.tolist() if hasattr(has_correct, "tolist") else has_correct,
    }
    if recoverable_flags is not None:
        fields["recoverable_flags"] = recoverable_flags
    if completion_modes is not None:
        from opsd_utils.debug_log import MODE_NAMES

        counts = {MODE_NAMES.get(c, str(c)): completion_modes.count(c) for c in set(completion_modes)}
        fields["completion_mode_counts"] = counts

    if completion_advantages is not None and completion_mask is not None:
        with torch.no_grad():
            pos = ((completion_advantages > 0) & (completion_mask > 0)).float().sum(dim=1)
            neg = ((completion_advantages < 0) & (completion_mask > 0)).float().sum(dim=1)
            zero_adv = ((completion_advantages.abs() < 1e-8) & (completion_mask > 0)).float().sum(dim=1)
            fields["adv_pos_tokens_mean"] = float(pos.mean().item())
            fields["adv_neg_tokens_mean"] = float(neg.mean().item())
            fields["adv_zero_tokens_mean"] = float(zero_adv.mean().item())

    opsd_debug.log_detail("routing", "mode routing summary", **fields)


def log_loss_diagnostics(
    *,
    global_step: int,
    grpo_loss: torch.Tensor,
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    coef_1: torch.Tensor,
    per_token_loss: torch.Tensor,
    opsd_loss: Optional[torch.Tensor] = None,
    combined_loss: Optional[torch.Tensor] = None,
    opsd_mask: Optional[torch.Tensor] = None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
) -> None:
    if not opsd_debug.should_log_detail(global_step):
        return

    opsd_debug.log_detail_banner(global_step, "LOSS & GRADIENT SIGNAL")

    with torch.no_grad():
        m = completion_mask.float()
        valid_per_sample = m.sum(dim=1).clamp(min=1.0)
        sample_grpo = (per_token_loss * m).sum(dim=1) / valid_per_sample

        fields: dict[str, Any] = {
            "grpo_loss_scalar": float(grpo_loss.detach().item()),
            "grpo_per_sample_mean": float(sample_grpo.mean().item()),
            "grpo_per_sample_max": float(sample_grpo.max().item()),
            "completion_mask_tokens_mean": float(valid_per_sample.mean().item()),
        }
        fields.update(_tensor_stats(advantages, "advantages"))
        fields.update(_tensor_stats(per_token_logps, "per_token_logps"))
        fields.update(_tensor_stats(old_per_token_logps, "old_per_token_logps"))
        fields.update(_tensor_stats((per_token_logps - old_per_token_logps) * m, "logps_delta_masked"))
        fields.update(_tensor_stats(coef_1 * m, "coef_1_masked"))

        low_clip = ((coef_1 < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0) & (m > 0)).float().sum()
        high_clip = ((coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0) & (m > 0)).float().sum()
        fields["clipped_low_tokens"] = int(low_clip.item())
        fields["clipped_high_tokens"] = int(high_clip.item())
        fields["grpo_zero_loss_rate"] = float((sample_grpo.abs() < 1e-12).float().mean().item())

        if opsd_loss is not None:
            fields["opsd_loss_scalar"] = float(opsd_loss.detach().item())
        if combined_loss is not None:
            fields["combined_loss_scalar"] = float(combined_loss.detach().item())
        if opsd_mask is not None:
            fields["opsd_batch_count"] = int(opsd_mask.sum().item())

        # Weak-signal hints
        hints: list[str] = []
        if fields.get("advantages/abs_mean", 1.0) < 1e-6:
            hints.append("advantages≈0 → GRPO per-token loss≈0")
        if fields.get("grpo_zero_loss_rate", 0) > 0.9:
            hints.append("most samples have ~0 GRPO loss")
        if fields.get("completion_mask_tokens_mean", 0) < 2:
            hints.append("very few effective completion tokens in mask")
        if opsd_loss is not None and abs(fields.get("opsd_loss_scalar", 0)) < 1e-8:
            hints.append("OPSD JSD≈0 → student/teacher nearly identical on completion")
        fields["weak_signal_hints"] = hints or ["none"]

    opsd_debug.log_detail("loss", "GRPO / OPSD loss breakdown", **fields)


def log_opsd_jsd_diagnostics(
    *,
    global_step: int,
    model,
    inputs: dict,
    opsd_indices: list[int],
    beta: float,
    tokenizer,
    max_samples: int = 2,
) -> None:
    if not opsd_debug.should_log_detail(global_step) or not opsd_indices:
        return

    opsd_debug.log_detail_banner(global_step, "OPSD JSD DECOMPOSITION")

    from opsd_utils.opsd_loss import _slice_image_sizes

    for k, global_idx in enumerate(opsd_indices[:max_samples]):
        local = global_idx
        prompt_ids = inputs["prompt_ids"][local : local + 1]
        prompt_mask = inputs["prompt_mask"][local : local + 1]
        completion_ids = inputs["completion_ids"][local : local + 1]
        completion_mask = inputs["completion_mask"][local : local + 1]
        pixel_values = inputs["pixel_values"][local : local + 1]
        img_sizes = _slice_image_sizes(inputs.get("img_sizes"), local)

        teacher_prompt_ids = inputs["teacher_prompt_ids"][local : local + 1]
        teacher_prompt_mask = inputs["teacher_prompt_mask"][local : local + 1]
        t_pixel = inputs["pixel_values"][local : local + 1]
        teacher_sizes = _slice_image_sizes(inputs.get("img_sizes"), local)
        if inputs.get("teacher_pixel_values") is not None:
            t_pixel = inputs["teacher_pixel_values"][local : local + 1]
        if inputs.get("teacher_image_sizes") is not None:
            teacher_sizes = _slice_image_sizes(inputs["teacher_image_sizes"], local)

        student_input = torch.cat([prompt_ids, completion_ids], dim=1)
        student_attn = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attn = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            student_logits = model(
                input_ids=student_input,
                attention_mask=student_attn,
                pixel_values=pixel_values,
                image_sizes=img_sizes,
            ).logits[:, -logits_to_keep - 1 : -1, :]
            teacher_logits = model(
                input_ids=teacher_input,
                attention_mask=teacher_attn,
                pixel_values=t_pixel,
                image_sizes=teacher_sizes,
            ).logits[:, -logits_to_keep - 1 : -1, :]

        stats = jsd_token_stats(student_logits, teacher_logits, completion_mask.float(), beta=beta)
        decoded = tokenizer.decode(
            completion_ids[0][completion_mask[0].bool()],
            skip_special_tokens=True,
        )
        stats["sample_index"] = global_idx
        stats["completion_text"] = _preview_text(decoded)
        stats["student_prompt_len"] = int(prompt_mask.sum().item())
        stats["teacher_prompt_len"] = int(teacher_prompt_mask.sum().item())
        opsd_debug.log_detail("opsd_jsd", f"sample[{k}] token-level JSD", **stats)
