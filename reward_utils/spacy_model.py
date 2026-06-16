"""Thread-safe spaCy model load for multi-process training (one download, all ranks wait)."""
from __future__ import annotations

import os
import subprocess
import sys

from filelock import FileLock

SPACY_MODEL = "en_core_web_sm"
_LOCK_PATH = os.path.join(os.path.expanduser("~"), ".cache", "dyme_en_core_web_sm.lock")


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
            print(f"[DyME] Downloading spaCy model {SPACY_MODEL} (single process)...", flush=True)
            subprocess.check_call([sys.executable, "-m", "spacy", "download", SPACY_MODEL])
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
            print(f"[DyME] Downloading spaCy model {SPACY_MODEL}...", flush=True)
            subprocess.check_call([sys.executable, "-m", "spacy", "download", SPACY_MODEL])
            return spacy.load(SPACY_MODEL)
