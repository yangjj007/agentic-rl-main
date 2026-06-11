"""Offline DePlot (google/deplot) batch pipeline for ChartQA visual_fact_deplot."""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from data_utils.paths import resolve_image_path

DEFAULT_MODEL_ID = "google/deplot"
DEFAULT_PROMPT = "Generate underlying data table of the figure below:"
PLACEHOLDER_SOURCE = "deplot_placeholder"
REAL_SOURCE = "google/deplot"


def _parse_vf(raw: Any) -> Optional[dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None
    return None


def is_deplot_placeholder(vf: Any) -> bool:
    data = _parse_vf(vf)
    if data is None:
        return False
    return data.get("source") == PLACEHOLDER_SOURCE


def has_real_deplot(vf: Any) -> bool:
    data = _parse_vf(vf)
    if data is None:
        return False
    if data.get("source") == PLACEHOLDER_SOURCE:
        return False
    table = (data.get("parsed_table") or "").strip()
    return bool(table) and data.get("source") in (REAL_SOURCE, "google/deplot", "deplot")


def format_deplot_for_teacher(vf: Any) -> str:
    """Teacher-facing text from visual_fact_deplot; empty if missing/placeholder."""
    data = _parse_vf(vf)
    if data is None:
        return ""
    if data.get("source") == PLACEHOLDER_SOURCE:
        return ""
    table = (data.get("parsed_table") or "").strip()
    if table:
        return table
    return ""


def placeholder_deplot_table(entry: dict[str, Any], error: Optional[str] = None) -> str:
    question = entry.get("question", entry.get("question_wo_prompt", ""))
    payload: dict[str, Any] = {
        "source": PLACEHOLDER_SOURCE,
        "question": question,
        "parsed_table": {"note": "DePlot unavailable or image missing"},
    }
    if error:
        payload["error"] = error
    return json.dumps(payload, ensure_ascii=False)


def build_deplot_visual_fact(
    entry: dict[str, Any],
    parsed_table: str,
    *,
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    question = entry.get("question", entry.get("question_wo_prompt", ""))
    payload = {
        "source": REAL_SOURCE,
        "model_id": model_id,
        "question": question,
        "parsed_table": parsed_table.strip(),
    }
    return json.dumps(payload, ensure_ascii=False)


def cache_key_for_entry(entry: dict[str, Any]) -> str:
    image = entry.get("image", "")
    return os.path.abspath(resolve_image_path(image)) if image else ""


def load_deplot_cache(path: str) -> dict[str, str]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_deplot_cache(path: str, cache: dict[str, str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def needs_deplot_processing(
    entry: dict[str, Any],
    *,
    replace_placeholder: bool = True,
    only_missing: bool = False,
) -> bool:
    vf = entry.get("visual_fact_deplot")
    if not vf:
        return True
    if is_deplot_placeholder(vf):
        return replace_placeholder or only_missing
    if has_real_deplot(vf):
        return replace_placeholder and not only_missing
    return only_missing or replace_placeholder


class DePlotRunner:
    """Lazy-loaded batched DePlot inference."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 384,
    ):
        self.model_id = model_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._dtype = dtype
        self._processor = None
        self._model = None

    def _resolve_device(self):
        import torch

        if self._device and self._device != "auto":
            return torch.device(self._device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _resolve_dtype(self, device):
        import torch

        if self._dtype == "float32":
            return torch.float32
        if self._dtype == "float16":
            return torch.float16
        if self._dtype == "bfloat16":
            return torch.bfloat16
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

            device = self._resolve_device()
            dtype = self._resolve_dtype(device)
            self._processor = Pix2StructProcessor.from_pretrained(self.model_id)
            self._model = Pix2StructForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            ).to(device)
            self._model.eval()
            self._device_obj = device
            return True
        except Exception as exc:
            print(f"[DePlot] model load failed: {exc}")
            self._model = None
            return False

    def generate_batch(self, image_paths: list[str]) -> list[str]:
        if not image_paths:
            return []
        if not self.load():
            return [""] * len(image_paths)

        import torch
        from PIL import Image

        images = []
        valid_indices: list[int] = []
        results: list[str] = [""] * len(image_paths)
        for i, path in enumerate(image_paths):
            if not path or not os.path.isfile(path):
                continue
            try:
                images.append(Image.open(path).convert("RGB"))
                valid_indices.append(i)
            except OSError:
                continue

        if not images:
            return results

        device = self._device_obj
        texts = [self.prompt] * len(images)
        with torch.inference_mode():
            inputs = self._processor(images=images, text=texts, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)

        for idx, text in zip(valid_indices, decoded):
            results[idx] = (text or "").strip()
        return results

    def generate_batch_with_oom_retry(
        self,
        image_paths: list[str],
        batch_size: int = 8,
        max_retries: int = 3,
    ) -> list[str]:
        if not image_paths:
            return []
        import torch

        bs = max(1, batch_size)
        out: list[str] = []
        pos = 0
        retries_left = max_retries
        while pos < len(image_paths):
            chunk_paths = image_paths[pos : pos + bs]
            try:
                chunk_out = self.generate_batch(chunk_paths)
                out.extend(chunk_out)
                pos += len(chunk_paths)
                retries_left = max_retries
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or bs <= 1 or retries_left <= 0:
                    out.extend([""] * len(chunk_paths))
                    pos += len(chunk_paths)
                    continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                retries_left -= 1
        return out


def enrich_entries_with_deplot(
    entries: list[dict[str, Any]],
    *,
    enabled: bool = True,
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 8,
    max_new_tokens: int = 384,
    cache_path: str = "",
    replace_placeholder: bool = True,
    only_missing: bool = False,
    max_samples: int = 0,
    device: Optional[str] = None,
) -> dict[str, int]:
    """
    Fill visual_fact_deplot on entries in-place.
    Returns stats dict: real, placeholder, skipped, failed, cached.
    """
    stats = {"real": 0, "placeholder": 0, "skipped": 0, "failed": 0, "cached": 0}
    work_entries = entries[:max_samples] if max_samples > 0 else entries

    if not enabled:
        for entry in work_entries:
            if not needs_deplot_processing(
                entry, replace_placeholder=replace_placeholder, only_missing=only_missing
            ):
                stats["skipped"] += 1
                continue
            entry["visual_fact_deplot"] = placeholder_deplot_table(entry, error="deplot_disabled")
            stats["placeholder"] += 1
        return stats

    cache = load_deplot_cache(cache_path)
    runner = DePlotRunner(model_id=model_id, device=device, max_new_tokens=max_new_tokens)
    model_ok = runner.load()

    pending: list[tuple[int, str, str]] = []

    for idx, entry in enumerate(work_entries):
        if not needs_deplot_processing(
            entry, replace_placeholder=replace_placeholder, only_missing=only_missing
        ):
            stats["skipped"] += 1
            continue

        key = cache_key_for_entry(entry)
        if key and key in cache and cache[key].strip():
            entry["visual_fact_deplot"] = build_deplot_visual_fact(entry, cache[key], model_id=model_id)
            stats["cached"] += 1
            stats["real"] += 1
            continue

        if not key or not os.path.isfile(key):
            entry["visual_fact_deplot"] = placeholder_deplot_table(entry, error="image_missing")
            stats["placeholder"] += 1
            continue

        if not model_ok:
            entry["visual_fact_deplot"] = placeholder_deplot_table(entry, error="model_load_failed")
            stats["placeholder"] += 1
            continue

        pending.append((idx, key, key))

    if pending and model_ok:
        bs = max(1, batch_size)
        for start in range(0, len(pending), bs):
            chunk = pending[start : start + bs]
            paths = [p[2] for p in chunk]
            tables = runner.generate_batch_with_oom_retry(paths, batch_size=bs)
            for (entry_idx, key, _), table in zip(chunk, tables):
                entry = work_entries[entry_idx]
                if table:
                    entry["visual_fact_deplot"] = build_deplot_visual_fact(
                        entry, table, model_id=model_id
                    )
                    if key:
                        cache[key] = table
                    stats["real"] += 1
                else:
                    entry["visual_fact_deplot"] = placeholder_deplot_table(
                        entry, error="inference_failed"
                    )
                    stats["failed"] += 1
                    stats["placeholder"] += 1
            if cache_path and cache:
                save_deplot_cache(cache_path, cache)

    return stats
