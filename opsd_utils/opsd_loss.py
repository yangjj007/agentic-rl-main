import torch
import torch.nn.functional as F

from opsd_utils import debug_log as opsd_debug


def _slice_image_sizes(image_sizes, index: int):
    """Slice per-sample image_sizes to match a batch-1 pixel_values forward."""
    if image_sizes is None:
        return None
    if isinstance(image_sizes, torch.Tensor):
        if image_sizes.dim() == 0:
            return image_sizes
        return image_sizes[index : index + 1]
    if isinstance(image_sizes, (list, tuple)):
        return image_sizes[index]
    return image_sizes


def generalized_jsd_loss(student_logits, teacher_logits, mask, beta=0.5):
    """Token-level generalized JSD on completion positions."""
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

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
) -> torch.Tensor:
    """
    Self-OPSD: same model, student vs privileged teacher prompt, shared student completion.
    """
    device = student_prompt_ids.device
    opsd_debug.log(
        "opsd_loss",
        "compute_vlm_opsd_loss enter",
        beta=beta,
        student_prompt_shape=tuple(student_prompt_ids.shape),
        teacher_prompt_shape=tuple(teacher_prompt_ids.shape),
        completion_shape=tuple(completion_ids.shape),
        has_teacher_pixel_values=teacher_pixel_values is not None,
    )
    student_input = torch.cat([student_prompt_ids, completion_ids], dim=1)
    student_attn = torch.cat([student_prompt_mask, completion_mask], dim=1)
    teacher_input = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
    teacher_attn = torch.cat([teacher_prompt_mask, completion_mask], dim=1)

    logits_to_keep = completion_ids.size(1)

    with opsd_debug.timed("opsd_loss", "student forward (grad)"):
        student_logits = model(
            input_ids=student_input,
            attention_mask=student_attn,
            pixel_values=student_pixel_values,
            image_sizes=student_image_sizes,
        ).logits[:, -logits_to_keep - 1 : -1, :]

    with torch.no_grad():
        t_pixel = teacher_pixel_values if teacher_pixel_values is not None else student_pixel_values
        t_sizes = teacher_image_sizes if teacher_image_sizes is not None else student_image_sizes
        with opsd_debug.timed("opsd_loss", "teacher forward (no grad)"):
            teacher_logits = model(
                input_ids=teacher_input,
                attention_mask=teacher_attn,
                pixel_values=t_pixel,
                image_sizes=t_sizes,
            ).logits[:, -logits_to_keep - 1 : -1, :]

    loss = generalized_jsd_loss(student_logits, teacher_logits, completion_mask.float(), beta=beta)
    opsd_debug.log("opsd_loss", "compute_vlm_opsd_loss done", loss=float(loss.detach().item()))
    return loss


def compute_vlm_opsd_loss_masked_batch(
    model,
    opsd_indices: list[int],
    all_indices: list[int],
    inputs: dict,
    beta: float = 0.5,
) -> torch.Tensor:
    """Compute mean OPSD loss over opsd_indices within a batch."""
    if not opsd_indices:
        opsd_debug.log("opsd_loss", "compute_vlm_opsd_loss_masked_batch skipped (empty opsd_indices)")
        return torch.tensor(0.0, device=inputs["prompt_ids"].device, requires_grad=True)

    opsd_debug.log(
        "opsd_loss",
        "compute_vlm_opsd_loss_masked_batch enter",
        opsd_indices=opsd_indices,
        all_indices=all_indices,
        beta=beta,
    )
    losses = []
    idx_map = {g: i for i, g in enumerate(all_indices)}

    for global_idx in opsd_indices:
        local = idx_map[global_idx]
        t_pixel = inputs["pixel_values"][local : local + 1]
        t_sizes = _slice_image_sizes(inputs.get("img_sizes"), local)
        if inputs.get("teacher_pixel_values") is not None:
            t_pixel = inputs["teacher_pixel_values"][local : local + 1]
        teacher_sizes = None
        if inputs.get("teacher_image_sizes") is not None:
            teacher_sizes = _slice_image_sizes(inputs["teacher_image_sizes"], local)
        opsd_debug.log(
            "opsd_loss",
            "compute sample OPSD loss",
            global_idx=global_idx,
            local_idx=local,
            student_image_sizes=_slice_image_sizes(inputs.get("img_sizes"), local),
            teacher_image_sizes=teacher_sizes,
        )
        with opsd_debug.timed("opsd_loss", f"sample_opsd_loss idx={global_idx}"):
            loss = compute_vlm_opsd_loss(
                model,
                inputs["prompt_ids"][local : local + 1],
                inputs["prompt_mask"][local : local + 1],
                inputs["pixel_values"][local : local + 1],
                t_sizes,
                inputs["teacher_prompt_ids"][local : local + 1],
                inputs["teacher_prompt_mask"][local : local + 1],
                t_pixel,
                inputs["completion_ids"][local : local + 1],
                inputs["completion_mask"][local : local + 1],
                beta=beta,
                teacher_image_sizes=teacher_sizes,
            )
        losses.append(loss)

    mean_loss = torch.stack(losses).mean()
    opsd_debug.log("opsd_loss", "compute_vlm_opsd_loss_masked_batch done", mean_loss=float(mean_loss.detach().item()))
    return mean_loss
