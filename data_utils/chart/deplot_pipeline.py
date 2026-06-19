"""Offline DePlot (google/deplot) batch pipeline for ChartQA visual_fact_deplot."""
from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import queue as queue_module
import time
from collections import Counter
from typing import Any, Optional

from tqdm import tqdm

from data_utils.paths import resolve_image_path, resolve_model_path

DEFAULT_MODEL_ID = "google/deplot"
DEFAULT_PROMPT = "Generate underlying data table of the figure below:"
PLACEHOLDER_SOURCE = "deplot_placeholder"
REAL_SOURCE = "google/deplot"
HF_FONT_REPO = "ybelkada/fonts"
HF_FONT_FILE = "Arial.TTF"
MAX_ERROR_LOG_LINES = 20


class DePlotErrorTracker:
    """Collect inference failure reasons for end-of-run reporting."""

    def __init__(self, *, max_log_lines: int = MAX_ERROR_LOG_LINES):
        self.counts: Counter[str] = Counter()
        self.samples: list[str] = []
        self._max_log_lines = max_log_lines

    def record(self, reason: str, detail: str = "", path: str = "") -> None:
        self.counts[reason] += 1
        if len(self.samples) >= self._max_log_lines:
            return
        parts = [f"[DePlot][{reason}]"]
        if path:
            parts.append(os.path.basename(path))
        if detail:
            parts.append(str(detail)[:240])
        self.samples.append(" ".join(parts))

    def merge(self, other: "DePlotErrorTracker") -> None:
        self.counts.update(other.counts)
        for line in other.samples:
            if len(self.samples) >= self._max_log_lines:
                break
            if line not in self.samples:
                self.samples.append(line)

    def emit(self, *, show_progress: bool = True) -> None:
        if not self.counts:
            return
        writer = tqdm.write if show_progress else print
        writer("[DePlot] failure summary:")
        for reason, count in self.counts.most_common():
            writer(f"  - {reason}: {count}")
        if self.samples:
            writer("[DePlot] sample errors (first {}):".format(len(self.samples)))
            for line in self.samples:
                writer(f"  {line}")


def _local_pretrained_kwargs(model_id: str) -> dict[str, bool]:
    resolved = resolve_model_path(model_id)
    if os.path.isdir(resolved):
        return {"local_files_only": True}
    return {}


def _hf_hub_font_cache_candidates() -> list[str]:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return [
        os.path.join(hf_home, "hub", "models--ybelkada--fonts", "snapshots", "*", HF_FONT_FILE),
        os.path.join(hf_home, "hub", "models--ybelkada--fonts", "snapshots", "*", "arial.ttf"),
    ]


def resolve_deplot_font_path(model_dir: str = "") -> Optional[str]:
    """
    Pix2Struct renders prompt text onto chart images and defaults to downloading
    ``ybelkada/fonts/Arial.TTF``. Resolve a local font for offline runs.
    """
    for env_key in ("DEPLOT_FONT_PATH", "PIX2STRUCT_FONT_PATH"):
        env_path = (os.environ.get(env_key) or "").strip()
        if env_path and os.path.isfile(env_path):
            return os.path.abspath(env_path)

    if model_dir:
        for name in (HF_FONT_FILE, "arial.ttf", "Arial.ttf"):
            candidate = os.path.join(model_dir, name)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

    for pattern in _hf_hub_font_cache_candidates():
        matches = sorted(glob.glob(pattern))
        if matches:
            return os.path.abspath(matches[0])

    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(HF_FONT_REPO, HF_FONT_FILE, local_files_only=True)
        if cached and os.path.isfile(cached):
            return os.path.abspath(cached)
    except Exception:
        pass

    for candidate in (
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.isfile(candidate):
            return candidate

    return None


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
        error_tracker: Optional[DePlotErrorTracker] = None,
    ):
        self.model_id = model_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._dtype = dtype
        self._processor = None
        self._model = None
        self._font_path: Optional[str] = None
        self._error_tracker = error_tracker or DePlotErrorTracker()
        self._logged_timing = False

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
            model_id = resolve_model_path(self.model_id)
            local_kw = _local_pretrained_kwargs(model_id)
            model_dir = model_id if os.path.isdir(model_id) else ""
            self._font_path = resolve_deplot_font_path(model_dir)
            if os.environ.get("HF_HUB_OFFLINE") == "1" and not self._font_path:
                print(
                    "[DePlot] offline mode but no local Arial font found; "
                    "set DEPLOT_FONT_PATH or place Arial.TTF next to the model."
                )
            elif self._font_path:
                print(f"[DePlot] using font: {self._font_path}")

            self._processor = Pix2StructProcessor.from_pretrained(model_id, **local_kw)
            load_kw = dict(local_kw)
            try:
                self._model = Pix2StructForConditionalGeneration.from_pretrained(
                    model_id,
                    dtype=dtype,
                    **load_kw,
                ).to(device)
            except TypeError:
                self._model = Pix2StructForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    **load_kw,
                ).to(device)
            self._model.eval()
            self._device_obj = device
            self.model_id = model_id
            print(f"[DePlot] loaded on {device} (dtype={dtype})")
            return True
        except Exception as exc:
            print(f"[DePlot] model load failed: {exc}")
            self._error_tracker.record("model_load_failed", str(exc))
            self._model = None
            return False

    def _processor_call(self, images: list[Any], texts: list[str]) -> Any:
        images_kwargs: dict[str, Any] = {}
        if self._font_path:
            images_kwargs["font_path"] = self._font_path
        try:
            return self._processor(
                images=images,
                text=texts,
                return_tensors="pt",
                images_kwargs=images_kwargs,
            )
        except TypeError:
            if self._font_path:
                return self._processor(
                    images=images,
                    text=texts,
                    return_tensors="pt",
                    font_path=self._font_path,
                )
            return self._processor(images=images, text=texts, return_tensors="pt")

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
                self._error_tracker.record("image_missing_at_infer", path=path or "<empty>")
                continue
            try:
                images.append(Image.open(path).convert("RGB"))
                valid_indices.append(i)
            except OSError as exc:
                self._error_tracker.record("image_open_failed", str(exc), path)
                continue

        if not images:
            return results

        device = self._device_obj
        texts = [self.prompt] * len(images)
        try:
            with torch.inference_mode():
                t0 = time.perf_counter()
                inputs = self._processor_call(images, texts)
                t_prep = time.perf_counter() - t0
                inputs = {k: v.to(device) for k, v in inputs.items()}
                t1 = time.perf_counter()
                outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                t_gen = time.perf_counter() - t1
                if hasattr(outputs, "sequences"):
                    outputs = outputs.sequences
                decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
                if not self._logged_timing:
                    print(
                        f"[DePlot] batch timing: preprocess={t_prep:.1f}s "
                        f"generate={t_gen:.1f}s images={len(images)} device={device}",
                        flush=True,
                    )
                    self._logged_timing = True
        except Exception as exc:
            for i, path in enumerate(image_paths):
                if path:
                    self._error_tracker.record(
                        type(exc).__name__,
                        str(exc)[:240],
                        path,
                    )
            return results

        for out_idx, (img_idx, text) in enumerate(zip(valid_indices, decoded)):
            cleaned = (text or "").strip()
            results[img_idx] = cleaned
            if not cleaned:
                token_len = 0
                try:
                    token_len = int(outputs[out_idx].numel())
                except Exception:
                    pass
                self._error_tracker.record(
                    "empty_decode",
                    f"token_len={token_len} preview={repr((text or '')[:80])}",
                    image_paths[img_idx],
                )
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
                msg = str(exc)
                if "out of memory" in msg.lower() and bs > 1 and retries_left > 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    retries_left -= 1
                    self._error_tracker.record(
                        "oom_retry",
                        f"reducing batch_size to {bs}: {msg[:160]}",
                        chunk_paths[0] if chunk_paths else "",
                    )
                    continue
                for path in chunk_paths:
                    self._error_tracker.record("runtime_error", msg[:240], path)
                out.extend([""] * len(chunk_paths))
                pos += len(chunk_paths)
        return out


def resolve_deplot_devices(
    *,
    devices: Optional[list[str]] = None,
    device: Optional[str] = None,
    use_all_gpus: bool = False,
) -> list[str]:
    """Resolve device list for DePlot inference."""
    if devices:
        return [d.strip() for d in devices if d and d.strip()]
    if device and device != "auto":
        return [device]
    try:
        import torch

        if use_all_gpus and torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available():
            return ["cuda:0"]
    except Exception:
        pass
    return ["cpu"]


def _worker_cuda_device(device: str) -> str:
    """Pin a spawned worker to one physical GPU via CUDA_VISIBLE_DEVICES."""
    dev = (device or "").strip()
    if dev.startswith("cuda:"):
        gpu_id = dev.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return "cuda"
    if dev.startswith("cuda"):
        return dev
    return dev or "cpu"


def _deplot_shard_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Process one pending shard on a single GPU (spawn-safe entrypoint)."""
    shard: list[tuple[int, str, str]] = payload["shard"]
    device_label = str(payload.get("device", "?"))
    progress_queue = payload.get("progress_queue")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    worker_device = _worker_cuda_device(device_label)
    runner = DePlotRunner(
        model_id=payload["model_id"],
        device=worker_device,
        max_new_tokens=payload["max_new_tokens"],
    )
    if not runner.load():
        return {
            "rows": [(idx, key, "", "model_load_failed") for idx, key, _ in shard],
            "counts": {"model_load_failed": len(shard)},
            "samples": runner._error_tracker.samples,
        }

    bs = max(1, int(payload["batch_size"]))
    rows: list[tuple[int, str, str, str]] = []
    for batch_idx, start in enumerate(range(0, len(shard), bs)):
        chunk = shard[start : start + bs]
        paths = [p[2] for p in chunk]
        t0 = time.perf_counter()
        tables = runner.generate_batch_with_oom_retry(paths, batch_size=bs)
        elapsed = time.perf_counter() - t0
        for (entry_idx, key, _), table in zip(chunk, tables):
            rows.append((entry_idx, key, table, "" if table else "inference_failed"))
        if progress_queue is not None:
            try:
                progress_queue.put(
                    {
                        "n": len(chunk),
                        "device": device_label,
                        "elapsed": elapsed,
                        "first": batch_idx == 0,
                    }
                )
            except Exception:
                pass
    return {
        "rows": rows,
        "counts": dict(runner._error_tracker.counts),
        "samples": runner._error_tracker.samples,
    }


def _apply_deplot_rows(
    work_entries: list[dict[str, Any]],
    rows: list[tuple[int, str, str, str]],
    *,
    model_id: str,
    cache: dict[str, str],
    stats: dict[str, int],
) -> None:
    for entry_idx, key, table, _err in rows:
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
    devices: Optional[list[str]] = None,
    use_all_gpus: bool = False,
    show_progress: bool = True,
) -> dict[str, int]:
    """
    Fill visual_fact_deplot on entries in-place.
    Returns stats dict: real, placeholder, skipped, failed, cached.
    """
    stats = {"real": 0, "placeholder": 0, "skipped": 0, "failed": 0, "cached": 0}
    error_tracker = DePlotErrorTracker()
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
    device_list = resolve_deplot_devices(
        devices=devices,
        device=device,
        use_all_gpus=use_all_gpus,
    )
    if show_progress:
        print(f"[DePlot] devices: {', '.join(device_list)}")

    runner: Optional[DePlotRunner] = None
    model_ok = False
    if len(device_list) == 1:
        runner = DePlotRunner(
            model_id=model_id,
            device=device_list[0],
            max_new_tokens=max_new_tokens,
            error_tracker=error_tracker,
        )
        model_ok = runner.load()
    elif device_list:
        model_ok = True

    if model_ok and show_progress:
        print(f"[DePlot] model ready; scanning {len(work_entries)} records")

    pending: list[tuple[int, str, str]] = []

    scan_bar = tqdm(
        work_entries,
        desc="DePlot prep",
        unit="rec",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for idx, entry in enumerate(scan_bar):
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

        if show_progress:
            scan_bar.set_postfix(
                pending=len(pending),
                cached=stats["cached"],
                skipped=stats["skipped"],
                placeholder=stats["placeholder"],
            )

    if pending and model_ok:
        bs = max(1, batch_size)
        n_pending = len(pending)
        if show_progress:
            tqdm.write(
                f"[DePlot] inference: {n_pending} images "
                f"(cached={stats['cached']} skipped={stats['skipped']} "
                f"placeholder={stats['placeholder']}, batch_size={bs}, "
                f"workers={len(device_list)})"
            )

        if len(device_list) > 1:
            from concurrent.futures import ProcessPoolExecutor

            shards = [[] for _ in device_list]
            for i, item in enumerate(pending):
                shards[i % len(device_list)].append(item)
            mp_ctx = mp.get_context("spawn")
            progress_queue = mp_ctx.Queue()
            payloads = [
                {
                    "shard": shard,
                    "model_id": model_id,
                    "device": device_list[i],
                    "max_new_tokens": max_new_tokens,
                    "batch_size": bs,
                    "progress_queue": progress_queue,
                }
                for i, shard in enumerate(shards)
                if shard
            ]
            infer_bar = tqdm(
                total=n_pending,
                desc="DePlot infer",
                unit="img",
                disable=not show_progress,
                dynamic_ncols=True,
            )
            with ProcessPoolExecutor(max_workers=len(payloads), mp_context=mp_ctx) as pool:
                futures = [pool.submit(_deplot_shard_worker, payload) for payload in payloads]
                pending_futures = set(futures)
                while pending_futures:
                    try:
                        while True:
                            msg = progress_queue.get_nowait()
                            if isinstance(msg, dict):
                                infer_bar.update(int(msg.get("n", 0)))
                                if msg.get("first") and show_progress:
                                    tqdm.write(
                                        f"[DePlot] {msg.get('device', '?')} first batch "
                                        f"({msg.get('n', 0)} img) in {float(msg.get('elapsed', 0)):.1f}s"
                                    )
                            else:
                                infer_bar.update(int(msg))
                            infer_bar.set_postfix(
                                processed=infer_bar.n,
                                real=stats["real"],
                                failed=stats["failed"],
                            )
                    except queue_module.Empty:
                        pass

                    done = [f for f in pending_futures if f.done()]
                    for fut in done:
                        pending_futures.remove(fut)
                        result = fut.result()
                        error_tracker.counts.update(result.get("counts") or {})
                        for line in result.get("samples") or []:
                            if len(error_tracker.samples) < error_tracker._max_log_lines:
                                error_tracker.samples.append(line)
                        rows = result.get("rows") or []
                        _apply_deplot_rows(
                            work_entries,
                            rows,
                            model_id=model_id,
                            cache=cache,
                            stats=stats,
                        )
                        if cache_path and cache:
                            save_deplot_cache(cache_path, cache)
                        if show_progress:
                            infer_bar.set_postfix(
                                processed=infer_bar.n,
                                real=stats["real"],
                                failed=stats["failed"],
                            )

                    if pending_futures:
                        time.sleep(0.1)
            if show_progress:
                infer_bar.close()
        else:
            assert runner is not None
            infer_bar = tqdm(
                total=n_pending,
                desc="DePlot infer",
                unit="img",
                disable=not show_progress,
                dynamic_ncols=True,
            )
            for start in range(0, n_pending, bs):
                chunk = pending[start : start + bs]
                paths = [p[2] for p in chunk]
                tables = runner.generate_batch_with_oom_retry(paths, batch_size=bs)
                rows = [
                    (entry_idx, key, table, "" if table else "inference_failed")
                    for (entry_idx, key, _), table in zip(chunk, tables)
                ]
                _apply_deplot_rows(
                    work_entries,
                    rows,
                    model_id=model_id,
                    cache=cache,
                    stats=stats,
                )
                if cache_path and cache:
                    save_deplot_cache(cache_path, cache)
                if show_progress:
                    infer_bar.update(len(chunk))
                    infer_bar.set_postfix(
                        real=stats["real"],
                        failed=stats["failed"],
                        cached=stats["cached"],
                    )
            if show_progress:
                infer_bar.close()

    if stats["failed"] > 0 or error_tracker.counts:
        error_tracker.emit(show_progress=show_progress)

    return stats
