from typing import Any

from PIL import Image

from data_utils.privileged_schema import parse_visual_fact
from opsd_utils.privileged.base import PrivilegedContextProvider
from opsd_utils.privileged.image_utils import heuristic_crop_from_visual_fact, load_rgb


class TextProvider(PrivilegedContextProvider):
    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        parts = []
        hint = (sample.get("hint") or "").strip()
        answer = (sample.get("answer") or "").strip()
        if hint:
            parts.append(f"[Reference Reasoning]\n{hint}")
        if answer:
            parts.append(f"[Reference Answer]\n{answer}")
        return "\n\n".join(parts)


class VisualFactsProvider(PrivilegedContextProvider):
    """B1: raw JSON visual facts; F1+F2 merge hint and deplot sources."""

    def _collect_visual_fact_parts(self, sample: dict[str, Any]) -> list[str]:
        parts: list[str] = []
        hint_vf = sample.get("visual_fact_hint")
        if hint_vf:
            text = parse_visual_fact(hint_vf)
            if text:
                parts.append(f"[Visual Facts - Hint]\n{text}")

        deplot_vf = sample.get("visual_fact_deplot")
        if deplot_vf:
            text = parse_visual_fact(deplot_vf)
            if text:
                parts.append(f"[Visual Facts - DePlot]\n{text}")

        primary = sample.get("visual_fact") or sample.get("visual_facts")
        if primary and not (hint_vf or deplot_vf):
            text = parse_visual_fact(primary)
            if text:
                parts.append(f"[Visual Facts]\n{text}")
        elif primary and (hint_vf or deplot_vf):
            text = parse_visual_fact(primary)
            if text:
                parts.append(f"[Visual Facts - Combined]\n{text}")

        return parts

    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        parts = self._collect_visual_fact_parts(sample)
        return "\n\n".join(parts)


class CropProvider(PrivilegedContextProvider):
    """Returns evidence crop as second teacher image (dual-image path uses image_utils)."""

    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        return ""

    def build_teacher_images(self, sample: dict[str, Any], crop_cfg: dict[str, Any] | None = None) -> list[Image.Image]:
        image = sample.get("image")
        if image is None:
            return []
        full = load_rgb(image)
        if full is None:
            return []
        crop, _, _ = heuristic_crop_from_visual_fact(full, sample, crop_cfg)
        return [crop]


class HybridProvider(PrivilegedContextProvider):
    def __init__(self, provider_names: list[str], crop_cfg: dict[str, Any] | None = None):
        self._providers: list[PrivilegedContextProvider] = []
        self._crop_cfg = crop_cfg or {}
        for name in provider_names:
            if name == "text":
                self._providers.append(TextProvider())
            elif name == "visual_facts":
                self._providers.append(VisualFactsProvider())
            elif name == "crop":
                self._providers.append(CropProvider())

    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        chunks = [p.build_teacher_suffix(sample) for p in self._providers]
        chunks = [c for c in chunks if c.strip()]
        return "\n\n".join(chunks)

    def build_teacher_images(self, sample: dict[str, Any]) -> list[Image.Image]:
        for p in self._providers:
            if isinstance(p, CropProvider):
                imgs = p.build_teacher_images(sample, self._crop_cfg)
                if imgs:
                    return imgs
        return []
