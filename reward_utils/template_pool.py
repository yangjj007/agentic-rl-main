"""Shared dynamic template pool (best_template.txt) for Visual Refiner / Checker."""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

from filelock import FileLock

DEFAULT_TEMPLATE = """Goal: [State the user's objective, e.g., Find the year with the highest sales]
Observation: [List key data points from the chart, e.g., 2020: 150, 2021: 200, 2022: 180]
Reasoning: [State the logical step, e.g., Compare the values. 200 is the maximum.]
Conclusion: [Draw the conclusion, e.g., The year with the highest sales was 2021.]
"""

DEFAULT_TEMPLATE_FILE = "best_template.txt"
DEFAULT_LOCK_FILE = "best_template.txt.lock"
_REQUIRED_TEMPLATE_HEADINGS = ("goal:", "observation:", "reasoning:", "conclusion:")


def is_valid_reasoning_template(text: str) -> bool:
    """Return whether a template can safely guide an online SFT refiner."""
    lowered = str(text or "").lower()
    return all(heading in lowered for heading in _REQUIRED_TEMPLATE_HEADINGS)


def _comparison_prompt(current_template: str, new_template: str) -> str:
    return f"""You are an expert in AI prompt engineering. Your task is to compare two reasoning templates. You must decide if the 'New Template' should replace the 'Current Template' as the single 'best' template.

My goal is to keep only the *best*, *clearest*, and *most general* template.

---
**Current Template:** {current_template}
---
**New Template:** {new_template}
---

**Instructions:**
1.  **Check for Novelty:** Is the 'New Template' *semantically different*?
2.  **Check for Quality:** If different, is the 'New Template' *objectively better* or *more general*?
3.  **Decision:** Should the 'New Template' **replace** the 'Current Template**?

Respond with **only** the word "YES" or "NO".

**Decision:**"""


class TemplatePool:
    """Process-safe single-template pool with optimistic CAS updates."""

    def __init__(
        self,
        template_path: str = DEFAULT_TEMPLATE_FILE,
        lock_path: Optional[str] = None,
        default_template: str = DEFAULT_TEMPLATE,
        refresh_interval_sec: float = 60.0,
    ):
        self.template_path = template_path
        self.lock_path = lock_path or f"{template_path}.lock"
        self.default_template = default_template.strip()
        self.refresh_interval_sec = refresh_interval_sec
        self._lock = FileLock(self.lock_path)
        self._cached_template = self.default_template
        self._last_refresh_time = 0.0
        self.pool_updates = 0

    def read_locked(self) -> str:
        try:
            with self._lock.acquire(timeout=5):
                if not os.path.exists(self.template_path):
                    return ""
                with open(self.template_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
        except Exception as exc:
            print(f"[Process {os.getpid()}] TemplatePool read failed: {exc}", flush=True)
            return ""

    def get_template(self, *, force_refresh: bool = False) -> str:
        now = time.time()
        if not force_refresh and (now - self._last_refresh_time) < self.refresh_interval_sec:
            return self._cached_template
        disk = self.read_locked()
        if is_valid_reasoning_template(disk):
            self._cached_template = disk
        else:
            self._cached_template = self.default_template
        self._last_refresh_time = now
        return self._cached_template

    def cas_write(self, new_template: str, original_template: str) -> bool:
        clean = (new_template or "").strip()
        if not clean:
            return False
        try:
            with self._lock.acquire(timeout=10):
                current = ""
                if os.path.exists(self.template_path):
                    with open(self.template_path, "r", encoding="utf-8") as f:
                        current = f.read().strip()
                if current != original_template:
                    return False
                with open(self.template_path, "w", encoding="utf-8") as f:
                    f.write(clean)
                self._cached_template = clean
                self.pool_updates += 1
                return True
        except Exception as exc:
            print(f"[Process {os.getpid()}] TemplatePool write failed: {exc}", flush=True)
            return False

    def maybe_update(
        self,
        new_template: str,
        compare_fn: Callable[[str, str], bool],
    ) -> tuple[bool, str]:
        """Compare via callback; CAS-write if accepted. Returns (written, compare_result_label)."""
        clean = (new_template or "").strip()
        if not clean or "none" in clean.lower() or not is_valid_reasoning_template(clean):
            return False, "none_template"
        original = self.read_locked()
        if original == clean:
            return False, "identical"
        is_better = compare_fn(original, clean)
        if not is_better:
            return False, "NO"
        written = self.cas_write(clean, original)
        return written, "YES" if written else "cas_conflict"


def compare_templates_via_client(client, system_prompt: str, current: str, new: str) -> bool:
    try:
        response = client.get_completion(
            _comparison_prompt(current, new),
            system_prompt=system_prompt,
            max_tokens=30,
        )
        return response.strip().upper() == "YES"
    except Exception:
        return False


def update_best_template_if_different(client, system_prompt: str, new_template: str, pool: Optional[TemplatePool] = None):
    """Backward-compatible helper used by API RewardCalculator."""
    tpl_pool = pool or TemplatePool()
    tpl_pool.maybe_update(
        new_template,
        lambda cur, new: compare_templates_via_client(client, system_prompt, cur, new),
    )
