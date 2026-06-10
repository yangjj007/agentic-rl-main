"""Unified privileged sample field parsing (G3 adapter interface)."""
from __future__ import annotations

import json
from typing import Any, Optional


def normalize_evidence_bbox(bbox: Any) -> Optional[list[float]]:
    """Validate and normalize evidence_bbox to C2 normalized [0,1] coordinates."""
    if bbox is None:
        return None
    if isinstance(bbox, str):
        try:
            bbox = json.loads(bbox)
        except json.JSONDecodeError:
            return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        coords = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if any(c < 0.0 or c > 1.0 for c in coords):
        return None
    x0, y0, x1, y1 = coords
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def parse_visual_fact(raw: Any) -> str:
    """B1: serialize visual_fact as raw JSON string for teacher suffix."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    return json.dumps(raw, ensure_ascii=False)


def _position_to_bbox_norm(position: str) -> list[float]:
    """Map A-OKVQA object position label to normalized crop box (D2)."""
    pos = (position or "center").strip().lower()
    mapping = {
        "center": (0.25, 0.25, 0.75, 0.75),
        "top": (0.1, 0.0, 0.9, 0.5),
        "bottom": (0.1, 0.5, 0.9, 1.0),
        "left": (0.0, 0.1, 0.5, 0.9),
        "right": (0.5, 0.1, 1.0, 0.9),
        "middle": (0.25, 0.25, 0.75, 0.75),
    }
    return list(mapping.get(pos, mapping["center"]))


def heuristic_bbox_from_visual_fact(raw: Any) -> Optional[list[float]]:
    """D2: derive normalized bbox from visual_fact.objects[].position."""
    if raw is None:
        return None
    data = raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(data, dict):
        objects = data.get("objects")
        if isinstance(objects, list) and objects:
            first = objects[0]
            if isinstance(first, dict):
                position = first.get("position", "center")
                return _position_to_bbox_norm(str(position))
    return None


def resolve_crop_bbox(
    sample: dict[str, Any],
    crop_cfg: Optional[dict[str, Any]] = None,
) -> tuple[Optional[list[float]], str]:
    """
    Resolve crop bbox and strategy for a sample.
    Returns (bbox_norm_or_none, crop_strategy).
    """
    crop_cfg = crop_cfg or {}
    for key in ("evidence_bbox", "bbox"):
        bbox = normalize_evidence_bbox(sample.get(key))
        if bbox is not None:
            return bbox, "bbox"

    if crop_cfg.get("crop_strategy") in ("bbox_then_center", "heuristic"):
        try:
            vf = sample.get("visual_fact") or sample.get("visual_facts")
            bbox = heuristic_bbox_from_visual_fact(vf)
            if bbox is not None:
                return bbox, "heuristic"
        except Exception:
            pass

    return None, "center"
