from typing import Any, Optional

from PIL import Image

from opsd_utils.privileged.base import PrivilegedContextProvider


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
    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        vf = (sample.get("visual_fact") or sample.get("visual_facts") or "").strip()
        if not vf:
            return ""
        return f"[Visual Facts]\n{vf}"


class CropProvider(PrivilegedContextProvider):
    """v1 stub: returns center crop when bbox missing."""

    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        return "[Visual Facts]\nEvidence region crop provided."

    def build_teacher_images(self, sample: dict[str, Any]) -> Optional[Image.Image]:
        image = sample.get("image")
        if image is None:
            return None
        if isinstance(image, str):
            return None
        w, h = image.size
        margin_w, margin_h = w // 4, h // 4
        return image.crop((margin_w, margin_h, w - margin_w, h - margin_h))


class HybridProvider(PrivilegedContextProvider):
    def __init__(self, provider_names: list[str]):
        self._providers = []
        for name in provider_names:
            if name == "text":
                self._providers.append(TextProvider())
            elif name == "visual_facts":
                self._providers.append(VisualFactsProvider())
            elif name == "crop":
                self._providers.append(CropProvider())

    def build_teacher_suffix(self, sample: dict[str, Any]) -> str:
        chunks = [p.build_teacher_suffix(sample) for p in self._providers if p.build_teacher_suffix(sample)]
        return "\n\n".join(chunks)

    def build_teacher_images(self, sample: dict[str, Any]) -> Optional[Image.Image]:
        for p in self._providers:
            if isinstance(p, CropProvider):
                img = p.build_teacher_images(sample)
                if img is not None:
                    return img
        return None
