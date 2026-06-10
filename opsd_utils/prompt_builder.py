import os
from typing import Any, Optional

import torch
from PIL import Image

from opsd_utils import debug_log as opsd_debug
from opsd_utils.privileged import build_privileged_context, maybe_save_privileged_images


def _build_teacher_text(student_prompt: str, privileged_suffix: str) -> str:
    teacher_text = student_prompt
    if privileged_suffix.strip():
        teacher_text = f"{student_prompt}\n\n{privileged_suffix.strip()}"
    return teacher_text


def _messages_for_teacher(teacher_text: str, num_images: int) -> list[dict]:
    content: list[dict] = [{"type": "image"} for _ in range(max(num_images, 1))]
    content.append({"type": "text", "text": teacher_text})
    return [{"role": "user", "content": content}]


def _processor_batch(processor, texts: list[str], images: list[list[Image.Image]]):
    has_images = any(len(imgs) > 0 for imgs in images)
    if has_images:
        return processor(text=texts, images=images, return_tensors="pt", padding=True)
    return processor(text=texts, return_tensors="pt", padding=True)


def _merge_batches(batches: list[dict]) -> dict:
    if len(batches) == 1:
        return batches[0]
    out: dict[str, Any] = {}
    keys = batches[0].keys()
    for key in keys:
        parts = [b[key] for b in batches if key in b]
        if not parts:
            continue
        if isinstance(parts[0], torch.Tensor):
            out[key] = torch.cat(parts, dim=0)
        else:
            out[key] = parts[0]
    return out


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
    num_images = len(pil_images) if pil_images else 1

    opsd_debug.log(
        "teacher_prompt",
        "tokenize_teacher_prompt",
        num_images=len(pil_images),
        suffix_len=len(privileged_suffix.strip()),
        teacher_text_len=len(teacher_text),
    )

    messages = _messages_for_teacher(teacher_text, num_images)
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    if pil_images:
        batch = processor(text=[text], images=pil_images, return_tensors="pt", padding=True)
    else:
        batch = processor(text=[text], return_tensors="pt", padding=True)

    opsd_debug.log(
        "teacher_prompt",
        "tokenize_teacher_prompt result",
        input_ids_shape=tuple(batch["input_ids"].shape),
        has_pixel_values="pixel_values" in batch,
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
        num_images = len(teacher_images) if teacher_images else 1
        messages = _messages_for_teacher(teacher_text, num_images)
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        sample_payloads.append(
            {
                "text": text.strip() if isinstance(text, str) else text,
                "images": teacher_images,
                "suffix_len": len(suffix.strip()),
                "num_teacher_images": len(teacher_images),
            }
        )

    texts = [p["text"] for p in sample_payloads]
    images = [p["images"] for p in sample_payloads]

    batch = _build_teacher_batch_with_oom_retry(processor, texts, images)

    out = {
        "teacher_prompt_ids": batch["input_ids"].to(device),
        "teacher_prompt_mask": batch["attention_mask"].to(device),
    }
    if "pixel_values" in batch:
        out["teacher_pixel_values"] = batch["pixel_values"].to(device)
    if "image_sizes" in batch:
        out["teacher_image_sizes"] = batch["image_sizes"].to(device)

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
        has_teacher_pixel_values="teacher_pixel_values" in out,
        teacher_pixel_values_shape=(
            tuple(out["teacher_pixel_values"].shape) if "teacher_pixel_values" in out else None
        ),
        teacher_images_count=sample_payloads[0]["num_teacher_images"] if sample_payloads else 0,
        teacher_prompt_len=teacher_len,
        vision_placeholder_delta=(teacher_len - student_len) if student_len else None,
    )
    opsd_debug.log_detail(
        "teacher_prompt",
        "teacher prompt batch built",
        global_step=global_step,
        batch_size=len(indices),
        teacher_prompt_len=teacher_len,
        teacher_pixel_values_shape=(
            tuple(out["teacher_pixel_values"].shape) if "teacher_pixel_values" in out else None
        ),
    )
    return out


def _build_teacher_batch_with_oom_retry(
    processor,
    texts: list[str],
    images: list[list[Image.Image]],
) -> dict:
    """Process teacher prompts; on CUDA OOM halve micro-batch and retry (decision E)."""
    n = len(texts)
    if n == 0:
        return {}
    micro = n
    while micro >= 1:
        try:
            batches = []
            for start in range(0, n, micro):
                end = min(start + micro, n)
                batches.append(_processor_batch(processor, texts[start:end], images[start:end]))
            return _merge_batches(batches)
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
