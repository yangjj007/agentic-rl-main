"""Thread-safe spaCy model load for multi-process training (one download, all ranks wait)."""
from __future__ import annotations

import os
import subprocess
import sys

from filelock import FileLock

SPACY_MODEL = "en_core_web_sm"
# Not on PyPI as ``en-core-web-sm`` — install from GitHub release wheel (spaCy docs).
SPACY_MODEL_WHEEL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
SPACY_MODEL_WHEEL_MIRROR = (
    "https://ghproxy.com/https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
_LOCK_PATH = os.path.join(os.path.expanduser("~"), ".cache", "dyme_en_core_web_sm.lock")


def _install_spacy_model() -> None:
    """Install en_core_web_sm: GitHub wheel first (mirror then direct), then spacy CLI."""
    errors: list[str] = []
    for label, url in (
        ("ghproxy wheel", SPACY_MODEL_WHEEL_MIRROR),
        ("GitHub wheel", SPACY_MODEL_WHEEL),
    ):
        print(f"[DyME] Installing {SPACY_MODEL} via {label}...", flush=True)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", url])
            return
        except subprocess.CalledProcessError as exc:
            errors.append(f"{label}: exit {exc.returncode}")

    print(f"[DyME] Wheel install failed ({'; '.join(errors)}); trying spacy download...", flush=True)
    subprocess.check_call([sys.executable, "-m", "spacy", "download", SPACY_MODEL])


def ensure_spacy_english_model(timeout: float = 600.0) -> None:
    """Download ``en_core_web_sm`` once if missing (safe before ``accelerate launch``)."""
    import spacy

    try:
        spacy.load(SPACY_MODEL)
        return
    except OSError:
        pass

    os.makedirs(os.path.dirname(_LOCK_PATH), exist_ok=True)
    with FileLock(_LOCK_PATH, timeout=timeout):
        try:
            spacy.load(SPACY_MODEL)
            return
        except OSError:
            _install_spacy_model()
            spacy.load(SPACY_MODEL)


def load_spacy_english(timeout: float = 600.0):
    """Load spaCy English model; concurrent ranks share one download via file lock."""
    import spacy

    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        pass

    os.makedirs(os.path.dirname(_LOCK_PATH), exist_ok=True)
    with FileLock(_LOCK_PATH, timeout=timeout):
        try:
            return spacy.load(SPACY_MODEL)
        except OSError:
            _install_spacy_model()
            return spacy.load(SPACY_MODEL)
