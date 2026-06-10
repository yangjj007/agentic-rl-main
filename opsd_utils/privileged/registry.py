from typing import Any, Optional

from opsd_utils import debug_log as opsd_debug
from opsd_utils.privileged.base import PrivilegedContextProvider
from opsd_utils.privileged.image_utils import resolve_teacher_images
from opsd_utils.privileged.profiles import DEFAULT_PROFILE, effective_profile, resolve_profile_config
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


def get_providers(names: list[str], crop_cfg: Optional[dict[str, Any]] = None) -> list[PrivilegedContextProvider]:
    if not names:
        names = ["text"]
    if len(names) == 1 and names[0] == "hybrid":
        return [HybridProvider(["text", "visual_facts"], crop_cfg=crop_cfg)]
    if "hybrid" in names:
        sub = [n for n in names if n != "hybrid"]
        return [HybridProvider(sub or ["text", "visual_facts"], crop_cfg=crop_cfg)]
    return [PROVIDER_REGISTRY[n]() for n in names if n in PROVIDER_REGISTRY]


def build_privileged_context(
    sample: dict[str, Any],
    provider_names: Optional[list[str]] = None,
    *,
    privileged_profile: str = DEFAULT_PROFILE,
    crop_cfg: Optional[dict[str, Any]] = None,
    opsd_config: Optional[dict[str, Any]] = None,
) -> tuple[str, list[Any]]:
    """
    Return (privileged_suffix, teacher_images).
    teacher_images: list[PIL.Image] — [full] for text profile, [full, crop] for visual/hybrid.
    """
    cfg = opsd_config or {}
    profile = effective_profile(sample, cfg.get("privileged_profile", privileged_profile))
    crop_cfg = crop_cfg or cfg.get("privileged_image") or {}

    profile_cfg = resolve_profile_config(profile, provider_names)
    providers = profile_cfg["providers"]

    opsd_debug.log(
        "privileged",
        "build_privileged_context",
        privileged_profile=profile,
        provider_names=providers,
        resolved_provider_types=[type(p).__name__ for p in get_providers(providers, crop_cfg)],
        sample_keys=list(sample.keys()),
    )

    hybrid = HybridProvider(providers, crop_cfg=crop_cfg)
    suffix = hybrid.build_teacher_suffix(sample)
    teacher_images, image_meta = resolve_teacher_images(sample, profile, crop_cfg)

    vf_raw = sample.get("visual_fact") or sample.get("visual_facts")
    if isinstance(vf_raw, str):
        visual_fact_len = len(vf_raw.strip())
    elif vf_raw is not None:
        from data_utils.privileged_schema import parse_visual_fact

        visual_fact_len = len(parse_visual_fact(vf_raw))
    else:
        visual_fact_len = 0

    meta = {
        "privileged_profile": profile,
        "num_teacher_images": len(teacher_images),
        "suffix_len": len(suffix.strip()),
        "visual_fact_len": visual_fact_len,
        **image_meta,
    }
    opsd_debug.log(
        "privileged",
        "build_privileged_context result",
        has_privileged_visual=len(teacher_images) > 1,
        **meta,
    )
    return suffix, teacher_images
