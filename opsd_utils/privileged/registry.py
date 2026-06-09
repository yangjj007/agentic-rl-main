from typing import Any

from opsd_utils.privileged.base import PrivilegedContextProvider
from opsd_utils.privileged.providers import (
    CropProvider,
    HybridProvider,
    TextProvider,
    VisualFactsProvider,
)

PROVIDER_REGISTRY: dict[str, type[PrivilegedContextProvider]] = {
    "text": TextProvider,
    "visual_facts": VisualFactsProvider,
    "crop": CropProvider,
    "hybrid": HybridProvider,
}


def get_providers(names: list[str]) -> list[PrivilegedContextProvider]:
    if not names:
        names = ["text"]
    if len(names) == 1 and names[0] == "hybrid":
        return [HybridProvider(["text", "visual_facts"])]
    if "hybrid" in names:
        sub = [n for n in names if n != "hybrid"]
        return [HybridProvider(sub or ["text", "visual_facts"])]
    return [PROVIDER_REGISTRY[n]() for n in names if n in PROVIDER_REGISTRY]


def build_privileged_context(sample: dict[str, Any], provider_names: list[str]) -> tuple[str, Any]:
    """Return (privileged_suffix, teacher_images)."""
    providers = get_providers(provider_names)
    if len(providers) == 1 and not isinstance(providers[0], HybridProvider):
        p = providers[0]
        return p.build_teacher_suffix(sample), p.build_teacher_images(sample)

    hybrid = HybridProvider(
        [n for n in provider_names if n != "hybrid"] or ["text", "visual_facts"]
    )
    return hybrid.build_teacher_suffix(sample), hybrid.build_teacher_images(sample)
