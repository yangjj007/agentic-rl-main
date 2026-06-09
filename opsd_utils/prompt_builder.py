import os
from typing import Any, Optional

from PIL import Image

from data_utils.paths import resolve_image_path
from opsd_utils import debug_log as opsd_debug


def _load_image(image: Any) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, str):
        path = resolve_image_path(image)
        if os.path.exists(path):
            img = Image.open(path)
            return img.convert("RGB")
    return None


def _build_teacher_text(student_prompt: str, privileged_suffix: str) -> str:
    teacher_text = student_prompt
    if privileged_suffix.strip():
        teacher_text = f"{student_prompt}\n\n{privileged_suffix.strip()}"
    return teacher_text


def tokenize_teacher_prompt(processor, student_prompt: str, privileged_suffix: str, image: Any) -> dict:
    """Tokenize teacher multimodal prompt = student question + privileged suffix."""
    pil_image = _load_image(image)
    teacher_text = _build_teacher_text(student_prompt, privileged_suffix)

    opsd_debug.log(
        "teacher_prompt",
        "tokenize_teacher_prompt",
        has_image=pil_image is not None,
        suffix_len=len(privileged_suffix.strip()),
        teacher_text_len=len(teacher_text),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": teacher_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    if pil_image is not None:
        batch = processor(text=[text], images=[pil_image], return_tensors="pt", padding=True)
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
) -> dict[str, Any]:
    """Build padded teacher prompt tensors for OPSD samples at given indices."""
    from opsd_utils.privileged import build_privileged_context

    opsd_debug.log(
        "teacher_prompt",
        "build_teacher_prompt_batch enter",
        num_indices=len(indices),
        indices=indices,
        num_samples=len(samples),
        provider_names=provider_names,
        device=str(device),
    )

    if not indices:
        opsd_debug.log("teacher_prompt", "empty indices, return {}")
        return {}

    texts: list[str] = []
    images: list[Optional[Image.Image]] = []
    for idx in indices:
        sample = samples[idx]
        suffix, teacher_image = build_privileged_context(sample, provider_names)
        image = teacher_image if teacher_image is not None else sample.get("image")
        teacher_text = _build_teacher_text(sample["prompt"], suffix)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": teacher_text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        texts.append(text.strip() if isinstance(text, str) else text)
        images.append(_load_image(image))

    # Batch through processor so variable image patch counts are padded consistently.
    has_images = any(img is not None for img in images)
    if has_images:
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    else:
        batch = processor(text=texts, return_tensors="pt", padding=True)

    out = {
        "teacher_prompt_ids": batch["input_ids"].to(device),
        "teacher_prompt_mask": batch["attention_mask"].to(device),
    }
    if "pixel_values" in batch:
        out["teacher_pixel_values"] = batch["pixel_values"].to(device)
    if "image_sizes" in batch:
        out["teacher_image_sizes"] = batch["image_sizes"].to(device)

    opsd_debug.log(
        "teacher_prompt",
        "build_teacher_prompt_batch done",
        teacher_prompt_ids_shape=tuple(out["teacher_prompt_ids"].shape),
        teacher_prompt_mask_shape=tuple(out["teacher_prompt_mask"].shape),
        has_teacher_pixel_values="teacher_pixel_values" in out,
        teacher_pixel_values_shape=(
            tuple(out["teacher_pixel_values"].shape) if "teacher_pixel_values" in out else None
        ),
    )
    return out
