from __future__ import annotations

import json
import os
from typing import Any, Optional


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _truncate_text(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", "\\n")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _candidate_log_config(opsd_config: dict[str, Any]) -> dict[str, Any]:
    probe_cfg = opsd_config.get("teacher_probe") or {}
    cfg = probe_cfg.get("candidate_log", {})
    if isinstance(cfg, bool):
        cfg = {"enabled": cfg}
    elif not isinstance(cfg, dict):
        cfg = {}
    enabled = bool(cfg.get("enabled", False)) or _truthy_env("DYME_TEACHER_PROBE_CANDIDATE_LOG")
    return {
        **cfg,
        "enabled": enabled,
        "max_text_chars": int(
            os.environ.get(
                "DYME_TEACHER_PROBE_CANDIDATE_LOG_MAX_CHARS",
                cfg.get("max_text_chars", 512),
            )
        ),
    }


def _deplot_status(sample: dict[str, Any]) -> str:
    raw = sample.get("visual_fact_deplot")
    if not str(raw or "").strip():
        return "missing"
    text = str(raw)
    if '"source": "deplot_placeholder"' in text or "'source': 'deplot_placeholder'" in text:
        return "placeholder"
    if "google/deplot" in text or '"source": "deplot"' in text:
        return "real"
    return "unknown"


def build_teacher_probe_record(
    *,
    sample: dict[str, Any],
    global_step: int,
    rank: int,
    global_idx: int,
    prompt_idx: int,
    generation_idx: int,
    provider_names: list[str],
    reference: str,
    student_output: str,
    teacher_output: str,
    score: float,
    final_route: str,
    answer_flag: str,
    prompt_profile: str = "",
    harness: str = "",
    harness_version: str = "",
    max_new_tokens: int | None = None,
    source_idx: int | None = None,
    parsed_answer: str = "",
    parse_failed: bool = False,
    has_answer_flag: bool = False,
    evidence_status: Optional[dict[str, Any]] = None,
    group_has_correct: Optional[bool] = None,
    group_reward_std: Optional[float] = None,
    is_all_wrong_probe_candidate: bool = False,
    is_mixed_wrong_probe_candidate: bool = False,
    route_reason: str = "",
    strict_rejected: bool = False,
    strict_reject_reasons: Optional[list[str]] = None,
    generated_clipped: bool = False,
    max_text_chars: int = 512,
) -> dict[str, Any]:
    question = sample.get("question") or sample.get("question_wo_prompt") or sample.get("prompt", "")
    teacher_correct = bool(score > 0)
    return {
        "schema_version": 1,
        "global_step": int(global_step),
        "rank": int(rank),
        "global_idx": int(global_idx),
        "source_idx": int(source_idx) if source_idx is not None else int(global_idx),
        "prompt_idx": int(prompt_idx),
        "generation_idx": int(generation_idx),
        "provider_names": list(provider_names),
        "prompt_profile": str(prompt_profile or ""),
        "harness": str(harness or ""),
        "harness_version": str(harness_version or ""),
        "max_new_tokens": int(max_new_tokens) if max_new_tokens is not None else None,
        "final_route": str(final_route),
        "teacher_correct": teacher_correct,
        "student_correct": False,
        "score": float(score),
        "teacher_accepted": bool(score > 0 and not strict_rejected),
        "strict_rejected": bool(strict_rejected),
        "strict_reject_reasons": list(strict_reject_reasons or []),
        "generated_clipped": bool(generated_clipped),
        "group_has_correct": bool(group_has_correct) if group_has_correct is not None else None,
        "group_all_wrong": (not bool(group_has_correct)) if group_has_correct is not None else None,
        "group_reward_std": float(group_reward_std) if group_reward_std is not None else None,
        "is_all_wrong_probe_candidate": bool(is_all_wrong_probe_candidate),
        "is_mixed_wrong_probe_candidate": bool(is_mixed_wrong_probe_candidate),
        "route_reason": str(route_reason),
        "answer_flag": str(answer_flag),
        "question": _truncate_text(question, max_text_chars),
        "reference": _truncate_text(reference, max_text_chars),
        "student_output": _truncate_text(student_output, max_text_chars),
        "teacher_output": _truncate_text(teacher_output, max_text_chars),
        "parsed_answer": _truncate_text(parsed_answer, max_text_chars),
        "parse_failed": bool(parse_failed),
        "has_answer_flag": bool(has_answer_flag),
        "image": _truncate_text(sample.get("image", ""), max_text_chars),
        "privileged": {
            "visual_fact_present": bool(str(sample.get("visual_fact") or sample.get("visual_facts") or "").strip()),
            "visual_fact_hint_present": bool(str(sample.get("visual_fact_hint") or "").strip()),
            "visual_fact_deplot_status": _deplot_status(sample),
            "evidence_status": dict(evidence_status or {}),
        },
    }


def append_teacher_probe_record(
    *,
    output_dir: Optional[str],
    opsd_config: dict[str, Any],
    rank: int,
    record: dict[str, Any],
) -> Optional[str]:
    cfg = _candidate_log_config(opsd_config)
    if not cfg["enabled"] or not output_dir:
        return None

    log_dir = cfg.get("dir") or os.environ.get("DYME_TEACHER_PROBE_CANDIDATE_LOG_DIR")
    if not log_dir:
        log_dir = os.path.join(output_dir, "teacher_probe_candidates")
    os.makedirs(log_dir, exist_ok=True)
    path = cfg.get("path") or os.path.join(log_dir, f"rank{int(rank)}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path
