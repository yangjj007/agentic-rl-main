"""Image loading and crop utilities for privileged teacher dual-image forward."""
from __future__ import annotations

from typing import Any, Optional

from PIL import Image

from data_utils.paths import resolve_image_path
from data_utils.privileged_schema import resolve_crop_bbox
from opsd_utils import debug_log as opsd_debug


def load_rgb(image: Any) -> Optional[Image.Image]:
    """Load sample image as RGB PIL from path or in-memory object."""
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(image, str):
        path = resolve_image_path(image)
        try:
            img = Image.open(path)
            return img.convert("RGB")
        except (FileNotFoundError, OSError):
            opsd_debug.log("privileged_image", "load_rgb failed", path=path)
            return None
    return None


def center_crop(img: Image.Image, margin_ratio: float = 0.25) -> Image.Image:
    w, h = img.size
    margin_w = int(w * margin_ratio)
    margin_h = int(h * margin_ratio)
    return img.crop((margin_w, margin_h, w - margin_w, h - margin_h))


def crop_image(
    img: Image.Image,
    bbox_norm: Optional[list[float]] = None,
    strategy: str = "center",
    margin_ratio: float = 0.25,
    fallback_reason: Optional[str] = None,
) -> tuple[Image.Image, str]:
    """
    Crop image using C2 normalized bbox or center fallback.
    Returns (cropped_image, crop_strategy_used).
    """
    if bbox_norm is not None and strategy in ("bbox", "heuristic", "bbox_then_center"):
        w, h = img.size
        x0 = int(bbox_norm[0] * w)
        y0 = int(bbox_norm[1] * h)
        x1 = int(bbox_norm[2] * w)
        y1 = int(bbox_norm[3] * h)
        x0, x1 = max(0, min(x0, w - 1)), max(1, min(x1, w))
        y0, y1 = max(0, min(y0, h - 1)), max(1, min(y1, h))
        if x1 > x0 and y1 > y0:
            used = strategy if strategy != "bbox_then_center" else "bbox"
            opsd_debug.log(
                "privileged_image",
                "crop_image bbox",
                strategy=used,
                bbox_norm=bbox_norm,
                crop_px=(x0, y0, x1, y1),
                fallback_reason=fallback_reason,
            )
            return img.crop((x0, y0, x1, y1)), used

    crop = center_crop(img, margin_ratio=margin_ratio)
    used = "center_fallback" if fallback_reason else "center"
    opsd_debug.log(
        "privileged_image",
        "crop_image center",
        strategy=used,
        bbox_norm=bbox_norm,
        margin_ratio=margin_ratio,
        fallback_reason=fallback_reason,
    )
    return crop, used


def heuristic_crop_from_visual_fact(
    img: Image.Image,
    sample: dict[str, Any],
    crop_cfg: Optional[dict[str, Any]] = None,
) -> tuple[Image.Image, str, Optional[list[float]]]:
    """D2 with D1 fallback: heuristic bbox from visual_fact, else center crop."""
    crop_cfg = crop_cfg or {}
    margin_ratio = float(crop_cfg.get("margin_ratio", 0.25))
    bbox_norm, strategy = resolve_crop_bbox(sample, crop_cfg)
    fallback_reason = None
    if strategy == "center" and sample.get("visual_fact"):
        fallback_reason = "heuristic_failed"
    crop, used = crop_image(
        img,
        bbox_norm=bbox_norm,
        strategy=strategy if bbox_norm else "center",
        margin_ratio=margin_ratio,
        fallback_reason=fallback_reason,
    )
    return crop, used, bbox_norm


def resolve_teacher_images(
    sample: dict[str, Any],
    profile: str,
    crop_cfg: Optional[dict[str, Any]] = None,
) -> tuple[list[Image.Image], dict[str, Any]]:
    """
    Build teacher image list for privileged forward.
    text -> [full]; visual/hybrid -> [full, crop].
    Returns (images, debug_meta).
    """
    crop_cfg = crop_cfg or {}
    image = sample.get("image")
    if image is None:
        return [], {"crop_strategy": "none", "num_teacher_images": 0, "has_bbox": False}

    full = load_rgb(image)
    if full is None:
        return [], {"crop_strategy": "load_failed", "num_teacher_images": 0, "has_bbox": False}

    if profile == "text":
        meta = {
            "crop_strategy": "single_full",
            "num_teacher_images": 1,
            "has_bbox": False,
            "bbox_norm": None,
        }
        return [full], meta

    crop, crop_strategy, bbox_norm = heuristic_crop_from_visual_fact(full, sample, crop_cfg)
    meta = {
        "crop_strategy": crop_strategy,
        "num_teacher_images": 2,
        "has_bbox": bbox_norm is not None,
        "bbox_norm": bbox_norm,
        "full_size": full.size,
        "crop_size": crop.size,
    }
    return [full, crop], meta
