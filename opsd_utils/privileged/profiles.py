"""Privileged profile resolution (text / visual / hybrid)."""
from __future__ import annotations

from typing import Any

from opsd_utils import debug_log as opsd_debug

PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "text": {
        "providers": ["text"],
        "dual_image": False,
    },
    "visual": {
        "providers": ["visual_facts"],
        "dual_image": True,
    },
    "hybrid": {
        "providers": ["text", "visual_facts"],
        "dual_image": True,
    },
}

DEFAULT_PROFILE = "hybrid"


def effective_profile(sample: dict[str, Any], profile: str) -> str:
    """Force text profile for samples without images (e.g. math_lm)."""
    profile = profile if profile in PROFILE_SPECS else DEFAULT_PROFILE
    if not sample.get("image"):
        if profile in ("visual", "hybrid"):
            opsd_debug.log(
                "privileged_profile",
                "downgrade profile (no image)",
                requested_profile=profile,
                effective_profile="text",
            )
            return "text"
    return profile


def resolve_profile_config(
    profile: str,
    provider_override: list[str] | None = None,
) -> dict[str, Any]:
    spec = PROFILE_SPECS.get(profile, PROFILE_SPECS[DEFAULT_PROFILE])
    providers = list(provider_override) if provider_override else list(spec["providers"])
    return {
        "profile": profile,
        "providers": providers,
        "dual_image": spec["dual_image"],
    }
