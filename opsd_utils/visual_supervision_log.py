"""Structured [VISUAL-*] logging for checker / refiner / I_c extraction."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from opsd_utils import debug_log as opsd_debug


def _enabled(cfg: Optional[dict]) -> bool:
    if cfg is not None and cfg.get("enabled") is False:
        return False
    import os as _os
    raw = _os.environ.get("DYME_VISUAL_LOG", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def _sample_count(cfg: Optional[dict]) -> int:
    if cfg and cfg.get("sample_count") is not None:
        return int(cfg["sample_count"])
    import os as _os
    raw = _os.environ.get("DYME_VISUAL_LOG_SAMPLES", "").strip()
    return int(raw) if raw.isdigit() else 3


def _preview_chars(cfg: Optional[dict]) -> int:
    if cfg and cfg.get("preview_chars") is not None:
        return int(cfg["preview_chars"])
    import os as _os
    raw = _os.environ.get("DYME_VISUAL_LOG_PREVIEW_CHARS", "").strip()
    return int(raw) if raw.isdigit() else 400


def _save_artifacts(cfg: Optional[dict]) -> bool:
    if cfg is not None and "save_artifacts" in cfg:
        return bool(cfg["save_artifacts"])
    import os as _os
    return _os.environ.get("DYME_VISUAL_SAVE_ARTIFACTS", "1").strip().lower() not in ("0", "false", "no")


def _prefix(tag: str) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    step = opsd_debug._DETAIL_STEP if opsd_debug._DETAIL_STEP is not None else "?"
    return (
        f"[{tag}][{ts}][rank={opsd_debug._RANK}/{opsd_debug._WORLD_SIZE}]"
        f"[global_step={step}]"
    )


def _fmt_preview(text: Any, max_len: int) -> str:
    s = str(text or "").replace("\n", "\\n")
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def log_visual(tag: str, msg: str, *, cfg: Optional[dict] = None, **fields: Any) -> None:
    if not _enabled(cfg):
        return
    if opsd_debug._RANK != 0:
        return
    preview = _preview_chars(cfg)
    extra = ""
    if fields:
        extra = " | " + " | ".join(
            f"{k}={_fmt_preview(v, preview)}" for k, v in fields.items()
        )
    print(f"{_prefix(tag)} {msg}{extra}", flush=True)


@dataclass
class VisualBatchRecorder:
    """Accumulates per-generate-batch visual supervision stats."""

    global_step: int
    output_dir: str
    log_cfg: dict = field(default_factory=dict)
    ic_ok: int = 0
    ic_fail: int = 0
    ic_cache_hit: int = 0
    ic_chars: list[float] = field(default_factory=list)
    checker_scores: list[float] = field(default_factory=list)
    checker_high: int = 0
    checker_medium: int = 0
    checker_low: int = 0
    checker_skipped_no_thinking: int = 0
    checker_local_fallback: int = 0
    refiner_changed: int = 0
    refiner_unchanged: int = 0
    refiner_in_lens: list[int] = field(default_factory=list)
    refiner_out_lens: list[int] = field(default_factory=list)
    refiner_fallback: int = 0
    pool_updates: int = 0
    artifacts: list[dict] = field(default_factory=list)
    route_bindings: list[dict] = field(default_factory=list)
    ic_latency_ms: float = 0.0
    checker_latency_ms: float = 0.0
    refiner_latency_ms: float = 0.0
    ic_calls: int = 0
    checker_calls: int = 0
    refiner_calls: int = 0
    teacher_batch_calls: int = 0

    def record_teacher_timing(
        self,
        kind: str,
        *,
        latency_ms: float,
        n_calls: int = 1,
        batch_size: int = 1,
    ) -> None:
        self.teacher_batch_calls += 1
        if kind == "ic":
            self.ic_latency_ms += float(latency_ms)
            self.ic_calls += int(n_calls)
        elif kind == "checker":
            self.checker_latency_ms += float(latency_ms)
            self.checker_calls += int(n_calls)
        elif kind == "refiner":
            self.refiner_latency_ms += float(latency_ms)
            self.refiner_calls += int(n_calls)
        else:
            self.checker_latency_ms += float(latency_ms)
            self.checker_calls += int(n_calls)

    def record_ic(self, **fields: Any) -> None:
        if fields.get("cache_hit"):
            self.ic_cache_hit += 1
        if fields.get("parse_ok"):
            self.ic_ok += 1
            self.ic_chars.append(float(fields.get("ic_chars", 0)))
        else:
            self.ic_fail += 1
        n = _sample_count(self.log_cfg)
        if fields.get("sample_idx", 999) < n:
            log_visual("VISUAL-IC", f"sample[{fields.get('sample_idx')}]", cfg=self.log_cfg, **fields)
        if _save_artifacts(self.log_cfg):
            self.artifacts.append({"kind": "ic", **fields})

    def record_checker(self, **fields: Any) -> None:
        score = float(fields.get("score", 0.0))
        self.checker_scores.append(score)
        label = fields.get("label", "low")
        if label == "high":
            self.checker_high += 1
        elif label == "medium":
            self.checker_medium += 1
        else:
            self.checker_low += 1
        if fields.get("skipped_no_thinking"):
            self.checker_skipped_no_thinking += 1
        if fields.get("local_fallback"):
            self.checker_local_fallback += 1
        n = _sample_count(self.log_cfg)
        idx = int(fields.get("sample_idx", 0))
        show = idx < n or fields.get("force_log")
        if show:
            log_visual("VISUAL-CHECKER", f"sample[{idx}]", cfg=self.log_cfg, **fields)
        if _save_artifacts(self.log_cfg):
            self.artifacts.append({"kind": "checker", **fields})

    def record_refiner(self, **fields: Any) -> None:
        changed = bool(fields.get("changed"))
        if changed:
            self.refiner_changed += 1
        else:
            self.refiner_unchanged += 1
        if fields.get("passthrough"):
            self.refiner_fallback += 1
        in_len = int(fields.get("in_len", 0))
        out_len = int(fields.get("out_len", 0))
        self.refiner_in_lens.append(in_len)
        self.refiner_out_lens.append(out_len)
        n = _sample_count(self.log_cfg)
        idx = int(fields.get("sample_idx", 0))
        if idx < n:
            log_visual("VISUAL-REFINER", f"sample[{idx}]", cfg=self.log_cfg, **fields)
        if _save_artifacts(self.log_cfg):
            self.artifacts.append({"kind": "refiner", **fields})

    def record_pool(self, **fields: Any) -> None:
        if fields.get("written"):
            self.pool_updates += 1
        payload = dict(fields)
        msg = payload.pop("msg", "template_event")
        log_visual("VISUAL-POOL", msg, cfg=self.log_cfg, **payload)

    def record_route(self, **fields: Any) -> None:
        if not self.log_cfg.get("log_route_binding", True):
            return
        if not opsd_debug.should_log_detail(self.global_step):
            return
        self.route_bindings.append(fields)
        log_visual("VISUAL-ROUTE", f"sample[{fields.get('sample_idx')}]", cfg=self.log_cfg, **fields)

    def finish(self) -> dict[str, Any]:
        n_ic = self.ic_ok + self.ic_fail
        ic_ok_rate = self.ic_ok / max(n_ic, 1)
        n_chk = len(self.checker_scores)
        checker_mean = sum(self.checker_scores) / max(n_chk, 1)
        n_ref = self.refiner_changed + self.refiner_unchanged
        refiner_changed_rate = self.refiner_changed / max(n_ref, 1)
        mean_in = sum(self.refiner_in_lens) / max(len(self.refiner_in_lens), 1)
        mean_out = sum(self.refiner_out_lens) / max(len(self.refiner_out_lens), 1)
        mean_delta = mean_out - mean_in
        mean_ic_chars = sum(self.ic_chars) / max(len(self.ic_chars), 1)

        log_visual(
            "VISUAL-IC",
            "batch_summary",
            cfg=self.log_cfg,
            extract_ok=self.ic_ok,
            extract_fail=self.ic_fail,
            mean_ic_chars=round(mean_ic_chars, 1),
            cache_hit=self.ic_cache_hit,
        )
        log_visual(
            "VISUAL-CHECKER",
            "batch_summary",
            cfg=self.log_cfg,
            n=n_chk,
            mean_score=round(checker_mean, 4),
            high=self.checker_high,
            medium=self.checker_medium,
            low=self.checker_low,
            skipped_no_thinking=self.checker_skipped_no_thinking,
            local_fallback=self.checker_local_fallback,
        )
        log_visual(
            "VISUAL-REFINER",
            "batch_summary",
            cfg=self.log_cfg,
            n=n_ref,
            changed=self.refiner_changed,
            unchanged=self.refiner_unchanged,
            mean_in_len=round(mean_in, 1),
            mean_out_len=round(mean_out, 1),
            mean_delta=round(mean_delta, 1),
            fallback_passthrough=self.refiner_fallback,
        )
        summary = {
            "visual/ic_ok_rate": ic_ok_rate,
            "visual/ic_fail_count": float(self.ic_fail),
            "visual/checker_mean": checker_mean,
            "visual/checker_high": float(self.checker_high),
            "visual/checker_medium": float(self.checker_medium),
            "visual/checker_low": float(self.checker_low),
            "visual/refiner_changed_rate": refiner_changed_rate,
            "visual/refiner_mean_delta_len": mean_delta,
            "visual/pool_updates": float(self.pool_updates),
            "visual/fallback_checker": float(self.checker_local_fallback),
            "visual/fallback_refiner": float(self.refiner_fallback),
            "visual/ic_latency_ms": round(self.ic_latency_ms, 1),
            "visual/checker_latency_ms": round(self.checker_latency_ms, 1),
            "visual/refiner_latency_ms": round(self.refiner_latency_ms, 1),
            "visual/ic_calls": float(self.ic_calls),
            "visual/checker_calls": float(self.checker_calls),
            "visual/refiner_calls": float(self.refiner_calls),
            "visual/teacher_batch_calls": float(self.teacher_batch_calls),
        }
        log_visual("VISUAL-BATCH", "generate_summary", cfg=self.log_cfg, **summary)

        if _save_artifacts(self.log_cfg) and self.artifacts:
            art_dir = os.path.join(
                self.output_dir,
                "visual_supervision",
                f"step_{self.global_step}",
            )
            os.makedirs(art_dir, exist_ok=True)
            path = os.path.join(art_dir, f"rank{opsd_debug._RANK}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                for row in self.artifacts:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return summary
