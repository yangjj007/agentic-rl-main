import torch
import torch.nn.functional as F

from opsd_utils import debug_log as opsd_debug
from opsd_utils import diagnostics as opsd_diagnostics
from opsd_utils.teacher_batching import (
    align_teacher_prompt_image_tokens,
    as_batch_num_images_tensor,
    get_teacher_vision_for_sample,
    model_inference_device,
    move_batch_num_images_to_model_device,
    move_pixel_values_to_model_device,
    student_batch_num_images_tensor,
)
from opsd_utils.vocab_align import align_cross_model_logits


def _slice_image_sizes(image_sizes, index: int):
    """Slice per-sample image_sizes for student path (one image per batch row)."""
    if image_sizes is None:
        return None
    if isinstance(image_sizes, torch.Tensor):
        if image_sizes.dim() == 0:
            return image_sizes
        return image_sizes[index : index + 1]
    if isinstance(image_sizes, (list, tuple)):
        return image_sizes[index]
    return image_sizes


def _slice_image_sizes_batch(image_sizes, start: int, end: int):
    """Slice image_sizes for a micro-batch row range [start, end)."""
    if image_sizes is None:
        return None
    if isinstance(image_sizes, torch.Tensor):
        if image_sizes.dim() == 0:
            return image_sizes
        if image_sizes.shape[0] >= end:
            return image_sizes[start:end]
        return image_sizes
    if isinstance(image_sizes, (list, tuple)):
        return image_sizes[start:end] if len(image_sizes) >= end else image_sizes
    return image_sizes


def _teacher_image_counts(inputs: dict, batch_size: int) -> list[int]:
    """Number of teacher images per batch sample (LLaVA-OV stacks images on dim 0)."""
    counts = inputs.get("teacher_num_images")
    if counts is None:
        return [1] * batch_size
    if isinstance(counts, torch.Tensor):
        return [int(max(1, c)) for c in counts.detach().cpu().tolist()]
    return [int(max(1, c)) for c in counts]


def _teacher_row(inputs: dict, batch_local_idx: int) -> int:
    """Map a batch row to a row in compact teacher tensors (if used)."""
    compact = inputs.get("teacher_compact_indices")
    if compact is None:
        return batch_local_idx
    if batch_local_idx in compact:
        return compact.index(batch_local_idx)
    return 0


def _teacher_image_count_for_row(inputs: dict, teacher_row: int) -> int:
    counts = inputs.get("teacher_num_images")
    if counts is None:
        return 1
    if isinstance(counts, torch.Tensor):
        return int(max(1, counts[teacher_row].item()))
    return int(max(1, counts[teacher_row]))


def _trim_to_effective_completion(
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    student_logits=None,
):
    """Drop padded completion tail so teacher/student OPSD only run on valid tokens."""
    eff_len = max(int(completion_mask.sum().item()), 1)
    width = int(completion_ids.size(1))
    if eff_len >= width:
        trimmed_logits = student_logits
        if student_logits is not None and student_logits.size(1) > eff_len:
            trimmed_logits = student_logits[:, :eff_len, :]
        return completion_ids, completion_mask, trimmed_logits, eff_len
    comp_ids = completion_ids[:, :eff_len]
    comp_mask = completion_mask[:, :eff_len]
    trimmed_logits = None
    if student_logits is not None:
        trimmed_logits = student_logits[:, :eff_len, :]
    return comp_ids, comp_mask, trimmed_logits, eff_len


def slice_teacher_vision_inputs(
    teacher_pixel_values,
    teacher_image_sizes,
    local: int,
    num_images_per_sample: list[int],
):
    """
    Slice teacher pixel_values / image_sizes for one batch sample.
    LLaVA-OneVision uses dim-0 = total images across batch (not batch size).
    """
    if teacher_pixel_values is None:
        return None, None
    start = sum(num_images_per_sample[:local])
    end = start + num_images_per_sample[local]
    t_pixel = teacher_pixel_values[start:end]
    t_sizes = None
    if teacher_image_sizes is not None and isinstance(teacher_image_sizes, torch.Tensor):
        t_sizes = teacher_image_sizes[start:end]
    return t_pixel, t_sizes


def _aligned_log_probs(student_logits, teacher_logits, mask):
    # Cross-model OPD: teacher logits already live on the teacher GPU; avoid
    # copying them onto the student GPU (vocab × seq is multi-hundred MiB per sample).
    loss_device = teacher_logits.device
    if student_logits.device != loss_device:
        student_logits = student_logits.to(loss_device, non_blocking=True)
    mask = mask.to(device=loss_device, non_blocking=True)

    comp_dtype = student_logits.dtype
    if comp_dtype == torch.float32:
        comp_dtype = torch.bfloat16
    if student_logits.dtype != comp_dtype:
        student_logits = student_logits.to(comp_dtype)
    if teacher_logits.dtype != comp_dtype:
        teacher_logits = teacher_logits.to(comp_dtype)

    student_logits, teacher_logits = align_cross_model_logits(student_logits, teacher_logits)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    return student_log_probs, teacher_log_probs, mask


def _masked_token_kl(token_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_loss = token_loss.sum(dim=-1)
    token_loss = token_loss * mask
    denom = mask.sum().clamp(min=1.0)
    return token_loss.sum() / denom


def _build_token_reliability_mask(
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    tokenizer,
    token_weighting: dict | None,
) -> torch.Tensor:
    """Build continuous OPD weights without banning reasoning-format tokens."""
    valid_mask = completion_mask.float()
    cfg = token_weighting or {}
    if not cfg.get("enabled", False) or tokenizer is None:
        return valid_mask

    min_weight = float(cfg.get("min_weight", 0.75))
    numeric_weight = float(cfg.get("numeric_weight", 2.0))
    answer_weight = float(cfg.get("answer_weight", 1.5))
    mode = str(cfg.get("mode", "reliability") or "reliability").strip().lower()
    weights = torch.full_like(valid_mask, min_weight)

    for row in range(completion_ids.shape[0]):
        after_answer_anchor = False
        for col in range(completion_ids.shape[1]):
            if not bool(completion_mask[row, col].item()):
                weights[row, col] = 0.0
                continue
            piece = tokenizer.decode(
                [int(completion_ids[row, col].item())],
                skip_special_tokens=False,
            )
            normalized = str(piece).strip().lower()
            is_numeric = any(char.isdigit() for char in normalized) or any(
                marker in normalized for marker in ("%", "$", "€", "£")
            )
            is_answer_anchor = "answer" in normalized
            if mode == "answer_anchor":
                if is_answer_anchor:
                    after_answer_anchor = True
                    weights[row, col] = answer_weight
                elif after_answer_anchor:
                    weights[row, col] = numeric_weight if is_numeric else answer_weight
            elif is_numeric:
                weights[row, col] = numeric_weight
            elif is_answer_anchor:
                weights[row, col] = answer_weight

    return weights * valid_mask


def generalized_jsd_loss(student_logits, teacher_logits, mask, beta=0.5):
    """Token-level generalized JSD on completion positions."""
    student_log_probs, teacher_log_probs, mask = _aligned_log_probs(student_logits, teacher_logits, mask)
    opsd_debug.log(
        "vocab_align",
        "generalized_jsd_loss log_softmax on aligned vocab",
        student_log_prob_shape=tuple(student_log_probs.shape),
        teacher_log_prob_shape=tuple(teacher_log_probs.shape),
        student_exp_sum=float(torch.exp(student_log_probs[0, 0]).sum().item()) if student_log_probs.numel() else None,
        teacher_exp_sum=float(torch.exp(teacher_log_probs[0, 0]).sum().item()) if teacher_log_probs.numel() else None,
    )

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

    return _masked_token_kl(jsd, mask)


def forward_kl_loss(student_logits, teacher_logits, mask):
    """Forward KL: KL(P_teacher || P_student)."""
    student_log_probs, teacher_log_probs, mask = _aligned_log_probs(student_logits, teacher_logits, mask)
    token_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    return _masked_token_kl(token_loss, mask)


def reverse_kl_loss(student_logits, teacher_logits, mask):
    """Reverse KL: KL(P_student || P_teacher)."""
    student_log_probs, teacher_log_probs, mask = _aligned_log_probs(student_logits, teacher_logits, mask)
    token_loss = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    return _masked_token_kl(token_loss, mask)


def skew_reverse_kl_loss(student_logits, teacher_logits, mask, alpha=0.1):
    """Skew reverse KL: KL(P_student || (1-alpha)P_teacher + alpha P_student)."""
    student_log_probs, teacher_log_probs, mask = _aligned_log_probs(student_logits, teacher_logits, mask)
    alpha_t = torch.tensor(
        min(max(float(alpha), 1e-6), 1.0 - 1e-6),
        dtype=student_log_probs.dtype,
        device=student_log_probs.device,
    )
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                teacher_log_probs + torch.log1p(-alpha_t),
                student_log_probs + torch.log(alpha_t),
            ]
        ),
        dim=0,
    )
    token_loss = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
    return _masked_token_kl(token_loss, mask)


def token_distillation_loss(
    student_logits,
    teacher_logits,
    mask,
    *,
    loss_type: str = "jsd",
    beta: float = 0.5,
    srkl_alpha: float = 0.1,
) -> torch.Tensor:
    """Dispatch token-level OPD divergence."""
    loss_name = (loss_type or "jsd").lower()
    if loss_name == "jsd":
        return generalized_jsd_loss(student_logits, teacher_logits, mask, beta=beta)
    if loss_name == "fkl":
        return forward_kl_loss(student_logits, teacher_logits, mask)
    if loss_name == "rkl":
        return reverse_kl_loss(student_logits, teacher_logits, mask)
    if loss_name == "srkl":
        return skew_reverse_kl_loss(student_logits, teacher_logits, mask, alpha=srkl_alpha)
    raise ValueError(f"Unknown OPD loss_type: {loss_type}")


def _combine_grpo_opsd_losses(
    grpo_loss: torch.Tensor,
    *,
    grpo_weight: float,
    opsd_loss: torch.Tensor | None,
    opsd_weight: float,
) -> torch.Tensor:
    """Apply GRPO and OPD weights even when a batch has no local OPD samples."""
    combined = float(grpo_weight) * grpo_loss
    if opsd_loss is not None:
        combined = combined + float(opsd_weight) * opsd_loss
    return combined


def _teacher_logits_with_oom_retry(
    model,
    processor,
    teacher_prompt_ids,
    teacher_prompt_mask,
    completion_ids,
    completion_mask,
    t_pixel,
    t_sizes,
    logits_to_keep: int,
    teacher_batch_num_images=None,
):
    """Teacher forward with OOM micro-batch halving (decision E). Batch dim is already 1 in OPSD loop."""
    if processor is not None:
        teacher_prompt_ids, teacher_prompt_mask = align_teacher_prompt_image_tokens(
            model,
            processor,
            teacher_prompt_ids,
            teacher_prompt_mask,
            t_pixel,
            t_sizes,
            batch_num_images=teacher_batch_num_images,
        )
    teacher_device = model_inference_device(model)
    teacher_prompt_ids = teacher_prompt_ids.to(teacher_device)
    teacher_prompt_mask = teacher_prompt_mask.to(teacher_device)
    completion_ids = completion_ids.to(teacher_device)
    completion_mask = completion_mask.to(teacher_device)
    t_pixel = move_pixel_values_to_model_device(model, t_pixel)
    teacher_batch_num_images = move_batch_num_images_to_model_device(model, teacher_batch_num_images)
    teacher_input = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
    teacher_attn = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
    oom_retries = 0
    opsd_debug.hang_probe(
        "teacher_forward_start",
        teacher_input_shape=tuple(teacher_input.shape),
        logits_to_keep=logits_to_keep,
        has_pixel_values=t_pixel is not None,
    )
    while True:
        try:
            with torch.no_grad():
                out = model(
                    input_ids=teacher_input,
                    attention_mask=teacher_attn,
                    pixel_values=t_pixel,
                    image_sizes=t_sizes,
                    batch_num_images=teacher_batch_num_images,
                ).logits[:, -logits_to_keep - 1 : -1, :]
            opsd_debug.hang_probe(
                "teacher_forward_done",
                teacher_logits_shape=tuple(out.shape),
                oom_retries=oom_retries,
            )
            return out
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            oom_retries += 1
            opsd_debug.log(
                "teacher_forward_oom",
                "teacher OPSD forward OOM, clearing cache and retrying",
                micro_batch_size=teacher_input.shape[0],
                oom_retries=oom_retries,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if oom_retries >= 3:
                raise


def slice_student_completion_logits(full_logits: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
    """Completion-token logits aligned with ``_get_per_token_logps`` / OPSD JSD."""
    logits = full_logits[:, -logits_to_keep - 1 :, :]
    logits = logits[:, :-1, :]
    return logits[:, -logits_to_keep:, :]


def compute_vlm_opsd_loss(
    model,
    student_prompt_ids,
    student_prompt_mask,
    student_pixel_values,
    student_image_sizes,
    teacher_prompt_ids,
    teacher_prompt_mask,
    teacher_pixel_values,
    completion_ids,
    completion_mask,
    beta=0.5,
    teacher_image_sizes=None,
    processor=None,
    teacher_batch_num_images=None,
    teacher_model=None,
    global_idx: int | None = None,
    capture_jsd_detail: bool = False,
    tokenizer=None,
    student_logits=None,
    loss_type: str = "jsd",
    srkl_alpha: float = 0.1,
    token_weighting: dict | None = None,
) -> torch.Tensor:
    """
    OPSD / OPD: student vs teacher prompt, shared student completion.
    When teacher_model is set, cross-model OPD (e.g. frozen 7B teacher); else self-OPSD.
    """
    teacher_model = teacher_model if teacher_model is not None else model
    opsd_debug.log(
        "opsd_loss",
        "compute_vlm_opsd_loss enter",
        beta=beta,
        loss_type=loss_type,
        srkl_alpha=srkl_alpha,
        student_prompt_shape=tuple(student_prompt_ids.shape),
        teacher_prompt_shape=tuple(teacher_prompt_ids.shape),
        completion_shape=tuple(completion_ids.shape),
        has_teacher_pixel_values=teacher_pixel_values is not None,
        teacher_pixel_values_shape=(
            tuple(teacher_pixel_values.shape) if teacher_pixel_values is not None else None
        ),
    )
    student_batch_num_images = student_batch_num_images_tensor(
        student_pixel_values, student_prompt_ids.shape[0]
    )
    padded_width = int(completion_ids.size(1))
    completion_ids, completion_mask, student_logits, eff_len = _trim_to_effective_completion(
        completion_ids,
        completion_mask,
        student_logits,
    )
    opsd_debug.hang_probe(
        "opsd_trim_completion",
        global_idx=global_idx,
        padded_width=padded_width,
        effective_tokens=eff_len,
    )
    if processor is not None and student_pixel_values is not None and student_logits is None:
        student_prompt_ids, student_prompt_mask = align_teacher_prompt_image_tokens(
            model,
            processor,
            student_prompt_ids,
            student_prompt_mask,
            student_pixel_values,
            student_image_sizes,
            batch_num_images=student_batch_num_images,
        )

    student_input = torch.cat([student_prompt_ids, completion_ids], dim=1)
    student_attn = torch.cat([student_prompt_mask, completion_mask], dim=1)

    logits_to_keep = completion_ids.size(1)

    if student_logits is None:
        with opsd_debug.timed("opsd_loss", "student forward (grad)"):
            student_logits = model(
                input_ids=student_input,
                attention_mask=student_attn,
                pixel_values=student_pixel_values,
                image_sizes=student_image_sizes,
                batch_num_images=student_batch_num_images,
            ).logits[:, -logits_to_keep - 1 : -1, :]
    else:
        opsd_debug.log(
            "opsd_loss",
            "reuse GRPO student completion logits (DeepSpeed single-forward)",
            student_logits_shape=tuple(student_logits.shape),
        )

    t_pixel = teacher_pixel_values if teacher_pixel_values is not None else student_pixel_values
    t_sizes = teacher_image_sizes if teacher_image_sizes is not None else student_image_sizes
    cross_model = teacher_model is not model
    opsd_debug.hang_probe(
        "opsd_sample_forward",
        global_idx=global_idx,
        student_prompt_len=int(student_prompt_mask.sum().item()),
        teacher_prompt_len=int(teacher_prompt_mask.sum().item()),
        completion_tokens=int(completion_mask.sum().item()),
        cross_model=cross_model,
    )
    with opsd_debug.timed("opsd_loss", "teacher forward (no grad)"):
        teacher_logits = _teacher_logits_with_oom_retry(
            teacher_model,
            processor,
            teacher_prompt_ids,
            teacher_prompt_mask,
            completion_ids,
            completion_mask,
            t_pixel,
            t_sizes,
            logits_to_keep,
            teacher_batch_num_images=teacher_batch_num_images,
        )

    if cross_model:
        opsd_debug.log(
            "opsd_loss",
            "cross-model OPD logits",
            student_vocab=student_logits.size(-1),
            teacher_vocab=teacher_logits.size(-1),
        )

    reliability_mask = _build_token_reliability_mask(
        completion_ids,
        completion_mask,
        tokenizer=tokenizer,
        token_weighting=token_weighting,
    )
    loss = token_distillation_loss(
        student_logits,
        teacher_logits,
        reliability_mask,
        loss_type=loss_type,
        beta=beta,
        srkl_alpha=srkl_alpha,
    )

    if capture_jsd_detail and global_idx is not None:
        opsd_diagnostics.maybe_capture_opsd_jsd_detail(
            global_idx=global_idx,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            completion_mask=completion_mask,
            completion_ids=completion_ids,
            beta=beta,
            tokenizer=tokenizer,
            student_prompt_len=int(student_prompt_mask.sum().item()),
            teacher_prompt_len=int(teacher_prompt_mask.sum().item()),
        )

    del teacher_logits

    opsd_debug.log("opsd_loss", "compute_vlm_opsd_loss done", loss=float(loss.detach().item()))
    return loss


def compute_vlm_opsd_loss_masked_batch(
    model,
    opsd_indices: list[int],
    all_indices: list[int],
    inputs: dict,
    beta: float = 0.5,
    processor=None,
    teacher_model=None,
    acc_gate: bool = True,
    pad_to_count: int | None = None,
    global_step: int | None = None,
    tokenizer=None,
    detail_max_samples: int = 2,
    student_completion_logits=None,
    loss_type: str = "jsd",
    srkl_alpha: float = 0.1,
    token_weighting: dict | None = None,
) -> torch.Tensor:
    """Compute mean OPSD loss over opsd_indices within a batch.

    Each rank runs teacher forwards only for its own OPSD samples. Ranks with
    zero local OPSD skip the loop; ``DyMETrainer`` barriers before
    ``gather_for_metrics`` so fast ranks do not enter NCCL while slow ranks
    are still in 7B teacher forwards.
    """
    real_count = len(opsd_indices)
    if real_count <= 0:
        opsd_debug.log("opsd_loss", "compute_vlm_opsd_loss_masked_batch skipped (no OPSD samples)")
        return torch.tensor(0.0, device=inputs["prompt_ids"].device, requires_grad=True)

    opsd_debug.hang_probe(
        "opsd_masked_batch_enter",
        real_count=real_count,
        opsd_indices=opsd_indices,
        pad_to_count=pad_to_count,
    )
    opsd_debug.log(
        "opsd_loss",
        "compute_vlm_opsd_loss_masked_batch enter",
        opsd_indices=opsd_indices,
        all_indices=all_indices,
        beta=beta,
        loss_type=loss_type,
        srkl_alpha=srkl_alpha,
        real_count=real_count,
    )
    capture_jsd_detail = (
        global_step is not None and opsd_debug.should_log_detail(global_step)
    )
    if capture_jsd_detail:
        opsd_diagnostics.begin_opsd_jsd_detail_capture(
            global_step,
            opsd_indices,
            max_samples=detail_max_samples,
        )
    losses = []
    idx_map = {g: i for i, g in enumerate(all_indices)}
    batch_size = inputs["prompt_ids"].shape[0]
    teacher_img_counts = _teacher_image_counts(inputs, batch_size)

    for step_idx, global_idx in enumerate(opsd_indices):
        local = idx_map[global_idx]
        teacher_row = _teacher_row(inputs, local)
        student_sizes = _slice_image_sizes(inputs.get("img_sizes"), local)
        t_pixel, teacher_sizes = get_teacher_vision_for_sample(
            inputs, teacher_row, teacher_img_counts
        )
        if t_pixel is None:
            t_pixel = inputs["pixel_values"][local : local + 1]
            teacher_sizes = student_sizes
        n_img = _teacher_image_count_for_row(inputs, teacher_row)
        comp_ids = inputs["completion_ids"][local : local + 1]
        comp_mask = inputs["completion_mask"][local : local + 1]
        precomputed_student_logits = None
        if student_completion_logits is not None:
            precomputed_student_logits = student_completion_logits[local : local + 1]
        opsd_debug.hang_probe(
            "opsd_loop_iter_start",
            step_idx=step_idx,
            global_idx=global_idx,
            local_idx=local,
            completion_tokens=int(comp_mask.sum().item()),
        )
        opsd_debug.log(
            "opsd_loss",
            "compute sample OPSD loss",
            global_idx=global_idx,
            local_idx=local,
            teacher_row=teacher_row,
            completion_tokens=int(comp_mask.sum().item()),
            teacher_num_images=n_img,
            student_image_sizes=student_sizes,
            teacher_image_sizes=teacher_sizes,
            teacher_pixel_values_shape=tuple(t_pixel.shape) if t_pixel is not None else None,
        )
        teacher_batch_num_images = as_batch_num_images_tensor(n_img, t_pixel)
        with opsd_debug.timed("opsd_loss", f"sample_opsd_loss idx={global_idx}"):
            loss = compute_vlm_opsd_loss(
                model,
                inputs["prompt_ids"][local : local + 1],
                inputs["prompt_mask"][local : local + 1],
                inputs["pixel_values"][local : local + 1],
                student_sizes,
                inputs["teacher_prompt_ids"][teacher_row : teacher_row + 1],
                inputs["teacher_prompt_mask"][teacher_row : teacher_row + 1],
                t_pixel,
                comp_ids,
                comp_mask,
                beta=beta,
                teacher_image_sizes=teacher_sizes,
                processor=processor,
                teacher_batch_num_images=teacher_batch_num_images,
                teacher_model=teacher_model,
                global_idx=global_idx,
                capture_jsd_detail=capture_jsd_detail,
                tokenizer=tokenizer,
                student_logits=precomputed_student_logits,
                loss_type=loss_type,
                srkl_alpha=srkl_alpha,
                token_weighting=token_weighting,
            )
            if acc_gate and "acc_rewards" in inputs:
                acc_val = float(inputs["acc_rewards"][global_idx].item())
                loss = loss * max(0.0, 1.0 - acc_val)
        losses.append(loss)
        opsd_debug.hang_probe(
            "opsd_loop_iter_done",
            step_idx=step_idx,
            global_idx=global_idx,
            loss=float(loss.detach().item()),
        )

    mean_loss = torch.stack(losses).sum() / real_count
    opsd_debug.hang_probe(
        "opsd_masked_batch_done",
        real_count=real_count,
        mean_loss=float(mean_loss.detach().item()),
    )
    opsd_debug.log(
        "opsd_loss",
        "compute_vlm_opsd_loss_masked_batch done",
        mean_loss=float(mean_loss.detach().item()),
        real_count=real_count,
    )
    return mean_loss
