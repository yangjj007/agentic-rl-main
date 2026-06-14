import torch
import torch.nn.functional as F

from opsd_utils import debug_log as opsd_debug
from opsd_utils import diagnostics as opsd_diagnostics
from opsd_utils.deepspeed_utils import deepspeed_requires_single_student_forward
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


def generalized_jsd_loss(student_logits, teacher_logits, mask, beta=0.5):
    """Token-level generalized JSD on completion positions."""
    # Cross-model OPD: teacher logits already live on the teacher GPU; avoid
    # copying them onto the student GPU (vocab × seq is multi-hundred MiB per sample).
    jsd_device = teacher_logits.device
    if student_logits.device != jsd_device:
        student_logits = student_logits.to(jsd_device, non_blocking=True)
    mask = mask.to(device=jsd_device, non_blocking=True)

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

    jsd = jsd.sum(dim=-1)
    jsd = jsd * mask
    denom = mask.sum().clamp(min=1.0)
    return jsd.sum() / denom


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
    while True:
        try:
            with torch.no_grad():
                return model(
                    input_ids=teacher_input,
                    attention_mask=teacher_attn,
                    pixel_values=t_pixel,
                    image_sizes=t_sizes,
                    batch_num_images=teacher_batch_num_images,
                ).logits[:, -logits_to_keep - 1 : -1, :]
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
    if processor is not None and student_pixel_values is not None:
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

    cross_model = teacher_model is not model
    if cross_model:
        opsd_debug.log(
            "opsd_loss",
            "cross-model OPD logits",
            student_vocab=student_logits.size(-1),
            teacher_vocab=teacher_logits.size(-1),
        )

    loss = generalized_jsd_loss(student_logits, teacher_logits, completion_mask.float(), beta=beta)

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
    if cross_model and torch.cuda.is_available():
        torch.cuda.empty_cache()

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
) -> torch.Tensor:
    """Compute mean OPSD loss over opsd_indices within a batch.

    Under DDP every rank must run the *same* number of student/teacher
    forwards, otherwise the per-forward buffer broadcast (and gradient
    reduction) collectives desync across ranks and NCCL eventually times out.
    ``pad_to_count`` is the global-max OPSD sample count across ranks; ranks
    with fewer (or zero) real samples run extra zero-weighted "dummy" forwards
    on a valid local row so the collective sequence stays aligned.
    """
    real_count = len(opsd_indices)
    target_count = pad_to_count if pad_to_count is not None else real_count
    if target_count <= 0:
        opsd_debug.log("opsd_loss", "compute_vlm_opsd_loss_masked_batch skipped (no OPSD samples)")
        return torch.tensor(0.0, device=inputs["prompt_ids"].device, requires_grad=True)

    opsd_debug.log(
        "opsd_loss",
        "compute_vlm_opsd_loss_masked_batch enter",
        opsd_indices=opsd_indices,
        all_indices=all_indices,
        beta=beta,
        real_count=real_count,
        target_count=target_count,
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

    for step_idx in range(target_count):
        is_real = step_idx < real_count
        # Dummy iterations reuse the first available row so shapes stay valid;
        # their contribution is zeroed out below.
        global_idx = opsd_indices[step_idx] if is_real else all_indices[0]
        local = idx_map[global_idx]
        student_sizes = _slice_image_sizes(inputs.get("img_sizes"), local)
        t_pixel, teacher_sizes = get_teacher_vision_for_sample(
            inputs, local, teacher_img_counts
        )
        if t_pixel is None:
            t_pixel = inputs["pixel_values"][local : local + 1]
            teacher_sizes = student_sizes
        opsd_debug.log(
            "opsd_loss",
            "compute sample OPSD loss",
            global_idx=global_idx,
            local_idx=local,
            teacher_num_images=teacher_img_counts[local],
            student_image_sizes=student_sizes,
            teacher_image_sizes=teacher_sizes,
            teacher_pixel_values_shape=tuple(t_pixel.shape) if t_pixel is not None else None,
        )
        n_img = teacher_img_counts[local]
        teacher_batch_num_images = as_batch_num_images_tensor(n_img, t_pixel)
        if not is_real and deepspeed_requires_single_student_forward():
            # ZeRO-1/2: avoid extra student forwards (even loss*0 still backprops).
            losses.append(torch.zeros((), device=inputs["prompt_ids"].device, requires_grad=True))
            continue
        precomputed_student_logits = None
        if student_completion_logits is not None:
            precomputed_student_logits = student_completion_logits[local : local + 1]
        with opsd_debug.timed("opsd_loss", f"sample_opsd_loss idx={global_idx}"):
            loss = compute_vlm_opsd_loss(
                model,
                inputs["prompt_ids"][local : local + 1],
                inputs["prompt_mask"][local : local + 1],
                inputs["pixel_values"][local : local + 1],
                student_sizes,
                inputs["teacher_prompt_ids"][local : local + 1],
                inputs["teacher_prompt_mask"][local : local + 1],
                t_pixel,
                inputs["completion_ids"][local : local + 1],
                inputs["completion_mask"][local : local + 1],
                beta=beta,
                teacher_image_sizes=teacher_sizes,
                processor=processor,
                teacher_batch_num_images=teacher_batch_num_images,
                teacher_model=teacher_model,
                global_idx=global_idx if is_real else None,
                capture_jsd_detail=capture_jsd_detail and is_real,
                tokenizer=tokenizer,
                student_logits=precomputed_student_logits,
            )
            if not is_real:
                # Keep the autograd graph / DDP collective alive but contribute nothing.
                loss = loss * 0.0
            elif acc_gate and "acc_rewards" in inputs:
                acc_val = float(inputs["acc_rewards"][global_idx].item())
                loss = loss * max(0.0, 1.0 - acc_val)
        losses.append(loss)

    # Mean over real samples only; dummy (zero-weighted) forwards keep the
    # collective sequence aligned across ranks without skewing the loss scale.
    mean_loss = torch.stack(losses).sum() / max(real_count, 1)
    opsd_debug.log(
        "opsd_loss",
        "compute_vlm_opsd_loss_masked_batch done",
        mean_loss=float(mean_loss.detach().item()),
        real_count=real_count,
        target_count=target_count,
    )
    return mean_loss
