import os
from typing import Any, Optional

import torch
from PIL import Image

from opsd_utils import debug_log as opsd_debug
from opsd_utils.privileged import build_privileged_context, maybe_save_privileged_images
from opsd_utils.teacher_batching import (
    count_image_tokens,
    process_teacher_sample,
    stack_teacher_processor_batches,
)


def _build_teacher_text(student_prompt: str, privileged_suffix: str) -> str:
    teacher_text = student_prompt
    if privileged_suffix.strip():
        teacher_text = f"{student_prompt}\n\n{privileged_suffix.strip()}"
    return teacher_text


def tokenize_teacher_prompt(
    processor,
    student_prompt: str,
    privileged_suffix: str,
    images: Any,
) -> dict:
    """Tokenize teacher multimodal prompt = student question + privileged suffix + N images."""
    if isinstance(images, list):
        pil_images = [img for img in images if isinstance(img, Image.Image)]
    else:
        from opsd_utils.privileged.image_utils import load_rgb

        one = load_rgb(images)
        pil_images = [one] if one is not None else []

    teacher_text = _build_teacher_text(student_prompt, privileged_suffix)

    opsd_debug.log(
        "teacher_prompt",
        "tokenize_teacher_prompt",
        num_images=len(pil_images),
        suffix_len=len(privileged_suffix.strip()),
        teacher_text_len=len(teacher_text),
    )

    batch = process_teacher_sample(processor, teacher_text, pil_images)

    opsd_debug.log(
        "teacher_prompt",
        "tokenize_teacher_prompt result",
        input_ids_shape=tuple(batch["input_ids"].shape),
        has_pixel_values="pixel_values" in batch,
        pixel_values_shape=tuple(batch["pixel_values"].shape) if "pixel_values" in batch else None,
        image_token_count=count_image_tokens(batch["input_ids"], processor),
    )
    return batch


def build_teacher_prompt_batch(
    processor,
    samples: list[dict[str, Any]],
    indices: list[int],
    provider_names: list[str],
    device,
    *,
    opsd_config: Optional[dict[str, Any]] = None,
    global_step: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Build padded teacher prompt tensors for OPSD samples at given indices."""
    opsd_config = opsd_config or {}
    privileged_profile = opsd_config.get("privileged_profile", "hybrid")
    crop_cfg = opsd_config.get("privileged_image") or {}
    privileged_debug_cfg = opsd_config.get("privileged_debug") or {}

    opsd_debug.log(
        "teacher_prompt",
        "build_teacher_prompt_batch enter",
        num_indices=len(indices),
        indices=indices,
        num_samples=len(samples),
        provider_names=provider_names,
        privileged_profile=privileged_profile,
        device=str(device),
        global_step=global_step,
    )

    if not indices:
        opsd_debug.log("teacher_prompt", "empty indices, return {}")
        return {}

    sample_payloads: list[dict[str, Any]] = []
    for idx in indices:
        sample = samples[idx]
        suffix, teacher_images = build_privileged_context(
            sample,
            provider_names,
            privileged_profile=privileged_profile,
            crop_cfg=crop_cfg,
            opsd_config=opsd_config,
        )
        if not teacher_images:
            from opsd_utils.privileged.image_utils import load_rgb

            full = load_rgb(sample.get("image"))
            teacher_images = [full] if full is not None else []

        full_img = teacher_images[0] if teacher_images else None
        crop_img = teacher_images[1] if len(teacher_images) > 1 else None
        maybe_save_privileged_images(
            global_step,
            idx,
            full_img,
            crop_img,
            meta={
                "privileged_profile": privileged_profile,
                "num_teacher_images": len(teacher_images),
                "suffix_len": len(suffix.strip()),
            },
            output_dir=output_dir,
            privileged_debug_cfg=privileged_debug_cfg,
        )

        teacher_text = _build_teacher_text(sample["prompt"], suffix)
        sample_payloads.append(
            {
                "teacher_text": teacher_text,
                "images": teacher_images,
                "suffix_len": len(suffix.strip()),
                "num_teacher_images": len(teacher_images),
            }
        )

    batch = _build_teacher_batch_with_oom_retry(processor, sample_payloads)

    out = {
        "teacher_prompt_ids": batch["input_ids"].to(device),
        "teacher_prompt_mask": batch["attention_mask"].to(device),
    }
    if batch.get("pixel_values_list"):
        out["teacher_pixel_values_list"] = [pv.to(device) for pv in batch["pixel_values_list"]]
    if batch.get("image_sizes_list"):
        out["teacher_image_sizes_list"] = [sz.to(device) for sz in batch["image_sizes_list"]]

    teacher_num_images = [int(max(0, n)) for n in batch.get("batch_num_images", [])]
    if not teacher_num_images:
        teacher_num_images = [p["num_teacher_images"] for p in sample_payloads]
    out["teacher_num_images"] = torch.tensor(teacher_num_images, device=device, dtype=torch.long)

    student_len = None
    if indices and samples[indices[0]].get("prompt"):
        student_messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": samples[indices[0]]["prompt"]}],
            }
        ]
        student_text = processor.apply_chat_template(student_messages, add_generation_prompt=True)
        student_len = len(processor(text=[student_text], return_tensors="pt")["input_ids"][0])

    teacher_len = int(out["teacher_prompt_ids"].shape[1])
    opsd_debug.log(
        "teacher_prompt",
        "build_teacher_prompt_batch done",
        teacher_prompt_ids_shape=tuple(out["teacher_prompt_ids"].shape),
        teacher_prompt_mask_shape=tuple(out["teacher_prompt_mask"].shape),
        has_teacher_pixel_values=bool(out.get("teacher_pixel_values_list")),
        teacher_pixel_values_shapes=[
            tuple(pv.shape) for pv in out.get("teacher_pixel_values_list", [])[:4]
        ],
        teacher_images_count=sample_payloads[0]["num_teacher_images"] if sample_payloads else 0,
        teacher_num_images=teacher_num_images,
        teacher_image_token_counts=batch.get("image_token_counts"),
        teacher_prompt_len=teacher_len,
        vision_placeholder_delta=(teacher_len - student_len) if student_len else None,
    )
    opsd_debug.log_detail(
        "teacher_prompt",
        "teacher prompt batch built",
        global_step=global_step,
        batch_size=len(indices),
        teacher_prompt_len=teacher_len,
        teacher_pixel_values_shapes=[
            tuple(pv.shape) for pv in out.get("teacher_pixel_values_list", [])[:4]
        ],
        teacher_image_token_counts=batch.get("image_token_counts"),
    )

    from opsd_utils.leakage import privileged_suffix_has_gold

    vf_empty = 0
    gold_suffix_count = 0
    for idx in indices:
        sample = samples[idx]
        vf = (
            sample.get("visual_fact_hint")
            or sample.get("visual_fact")
            or sample.get("visual_facts")
            or ""
        )
        if not str(vf).strip():
            vf_empty += 1
        priv_suffix, _ = build_privileged_context(
            sample,
            provider_names,
            privileged_profile=privileged_profile,
            crop_cfg=crop_cfg,
            opsd_config=opsd_config,
        )
        if privileged_suffix_has_gold(priv_suffix, sample):
            gold_suffix_count += 1
    suffix_lens = [p["suffix_len"] for p in sample_payloads]
    n_idx = max(len(indices), 1)
    out["teacher_stats"] = {
        "teacher_suffix_len_mean": float(sum(suffix_lens) / len(suffix_lens)) if suffix_lens else 0.0,
        "visual_fact_empty_rate": vf_empty / n_idx,
        "privileged_suffix_has_gold_rate": gold_suffix_count / n_idx,
        "num_teacher_images_mean": float(
            sum(p["num_teacher_images"] for p in sample_payloads) / len(sample_payloads)
        )
        if sample_payloads
        else 0.0,
    }
    return out


def _build_teacher_batch_with_oom_retry(
    processor,
    sample_payloads: list[dict[str, Any]],
) -> dict:
    """Process each teacher sample separately; on OOM halve micro-batch and retry."""
    n = len(sample_payloads)
    if n == 0:
        return {}
    micro = n
    while micro >= 1:
        try:
            per_sample_batches: list[dict[str, Any]] = []
            for start in range(0, n, micro):
                end = min(start + micro, n)
                for payload in sample_payloads[start:end]:
                    per_sample_batches.append(
                        process_teacher_sample(
                            processor,
                            payload["teacher_text"],
                            payload["images"],
                        )
                    )
            return stack_teacher_processor_batches(processor, per_sample_batches)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or micro == 1:
                raise
            opsd_debug.log(
                "teacher_forward_oom",
                "teacher prompt batch OOM, halving micro-batch",
                original_batch=n,
                micro_batch_size=micro,
                new_micro_batch_size=max(1, micro // 2),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            micro = max(1, micro // 2)
    return {}
