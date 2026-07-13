"""Save privileged teacher images to disk on detail steps."""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from PIL import Image

from opsd_utils import debug_log as opsd_debug

_saved_counts: dict[int, int] = {}
_output_dir: Optional[str] = None
_cfg: dict[str, Any] = {}


def configure(output_dir: Optional[str] = None, privileged_debug_cfg: Optional[dict[str, Any]] = None) -> None:
    global _output_dir, _cfg
    _output_dir = output_dir
    _cfg = dict(privileged_debug_cfg or {})
    _saved_counts.clear()


def _image_subdir() -> str:
    return _cfg.get("image_subdir", "logs/images")


def maybe_save_privileged_images(
    global_step: Optional[int],
    sample_idx: int,
    full_img: Optional[Image.Image],
    crop_img: Optional[Image.Image],
    meta: Optional[dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    privileged_debug_cfg: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """
    Save teacher privileged images when should_log_detail(global_step) is true.
    Returns base path prefix if saved, else None.
    """
    if global_step is None:
        return None
    if not opsd_debug.should_log_detail(global_step):
        return None

    cfg = privileged_debug_cfg if privileged_debug_cfg is not None else _cfg
    if not cfg.get("save_images", True):
        return None

    max_samples = int(cfg.get("max_samples_per_detail", 2))
    count = _saved_counts.get(global_step, 0)
    if count >= max_samples:
        return None

    base_out = output_dir or _output_dir
    if not base_out:
        opsd_debug.log(
            "privileged_debug",
            "skip image save (no output_dir)",
            global_step=global_step,
            sample_idx=sample_idx,
        )
        return None

    subdir = os.path.join(base_out, _image_subdir() if cfg is _cfg else cfg.get("image_subdir", "logs/images"))
    os.makedirs(subdir, exist_ok=True)

    prefix = os.path.join(subdir, f"step_{int(global_step):06d}_idx_{sample_idx}")
    saved_paths: list[str] = []

    if full_img is not None:
        full_path = f"{prefix}_full.png"
        full_img.save(full_path)
        saved_paths.append(full_path)

    if crop_img is not None:
        crop_path = f"{prefix}_crop.png"
        crop_img.save(crop_path)
        saved_paths.append(crop_path)

    meta_path = f"{prefix}_meta.json"
    meta_payload = dict(meta or {})
    meta_payload.update(
        {
            "global_step": global_step,
            "sample_idx": sample_idx,
            "saved_paths": saved_paths,
        }
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, ensure_ascii=False, indent=2)

    _saved_counts[global_step] = count + 1
    opsd_debug.log_detail(
        "privileged_debug",
        "privileged images saved",
        global_step=global_step,
        sample_idx=sample_idx,
        prefix=prefix,
        saved_paths=saved_paths,
        meta_path=meta_path,
        **{k: v for k, v in (meta or {}).items() if k not in ("full_size", "crop_size")},
    )
    return prefix
