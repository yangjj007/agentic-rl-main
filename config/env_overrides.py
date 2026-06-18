"""Optional environment-variable overrides for config defaults."""
from __future__ import annotations

import os
from typing import Optional


def env_str(key: str, default: str) -> str:
    value = os.environ.get(key, "").strip()
    return value if value else default


def env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key, "").strip().lower()
    if not value:
        return default
    return value not in ("0", "false", "no", "off")


def env_int(key: str, default: int) -> int:
    value = os.environ.get(key, "").strip()
    if not value:
        return default
    return int(value)


def env_float(key: str, default: float) -> float:
    value = os.environ.get(key, "").strip()
    if not value:
        return default
    return float(value)


def env_list(key: str, default: list[str]) -> list[str]:
    value = os.environ.get(key, "").strip()
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def env_optional_int(key: str, default: Optional[int] = None) -> Optional[int]:
    value = os.environ.get(key, "").strip()
    if not value:
        return default
    return int(value)


def env_optional_float(key: str, default: Optional[float] = None) -> Optional[float]:
    value = os.environ.get(key, "").strip()
    if not value:
        return default
    return float(value)
