"""Structured, semantic audit records for auxiliary teacher trajectories."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from typing import Any


def _trajectory_log_config(opsd_config: dict[str, Any]) -> dict[str, Any]:
    cfg = (opsd_config.get("teacher_trajectory") or {}).get("audit_log", {})
    if isinstance(cfg, bool):
        cfg = {"enabled": cfg}
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        **cfg,
        "enabled": bool(cfg.get("enabled", False)),
        "max_text_chars": int(cfg.get("max_text_chars", 0) or 0),
    }


def _maybe_truncate(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _plain(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else value


def build_teacher_trajectory_record(
    *,
    sample: dict[str, Any],
    global_step: int,
    rank: int,
    global_idx: int,
    source_idx: int,
    prompt_idx: int,
    generation_idx: int,
    reference: str,
    student_output: str,
    teacher_probe_output: str,
    teacher_probe_correct: bool | None,
    trajectory_output: str,
    trajectory_raw_output: str = "",
    trajectory_evidence_retry_raw_output: str = "",
    trajectory_prompt_profile: str = "",
    trajectory_answer: str,
    trajectory_answer_source: str = "trajectory",
    trajectory_correct: bool | None,
    trajectory_parse_failed: bool,
    verification: Any = None,
    loss_eligible: bool = True,
    max_text_chars: int = 0,
) -> dict[str, Any]:
    """Return a complete question/student/probe/trajectory audit record."""
    question = sample.get("question") or sample.get("question_wo_prompt") or sample.get("prompt", "")
    payload: dict[str, Any] = {
        "schema_version": 2,
        "global_step": int(global_step),
        "rank": int(rank),
        "global_idx": int(global_idx),
        "source_idx": int(source_idx),
        "prompt_idx": int(prompt_idx),
        "generation_idx": int(generation_idx),
        "question": _maybe_truncate(question, max_text_chars),
        "image": _maybe_truncate(sample.get("image", ""), max_text_chars),
        "reference": _maybe_truncate(reference, max_text_chars),
        "student_output": _maybe_truncate(student_output, max_text_chars),
        "teacher_probe_output": _maybe_truncate(teacher_probe_output, max_text_chars),
        "teacher_probe_correct": teacher_probe_correct,
        "teacher_trajectory_output": _maybe_truncate(trajectory_output, max_text_chars),
        "teacher_trajectory_raw_output": _maybe_truncate(trajectory_raw_output, max_text_chars),
        "teacher_trajectory_evidence_retry_raw_output": _maybe_truncate(
            trajectory_evidence_retry_raw_output, max_text_chars
        ),
        "trajectory_prompt_profile": str(trajectory_prompt_profile),
        "trajectory_answer": _maybe_truncate(trajectory_answer, max_text_chars),
        "trajectory_answer_source": str(trajectory_answer_source),
        "trajectory_correct": trajectory_correct,
        "trajectory_parse_failed": bool(trajectory_parse_failed),
        "trajectory_loss_eligible": bool(loss_eligible),
        # OPD-only invariant: diagnostics and auxiliary FKL never route rows.
        "routing": {
            "opd_only": True,
            "opsd_selected": True,
            "grpo_selected": False,
            "sft_selected": False,
        },
    }
    if verification is not None:
        payload["quality"] = {
            "quality": getattr(verification, "quality", ""),
            "reason_codes": list(getattr(verification, "reason_codes", ()) or ()),
            "structure_valid": bool(getattr(getattr(verification, "parsed", None), "structure_valid", False)),
            "deplot_available": bool(getattr(verification, "deplot_available", False)),
            "grounded_claims": [_plain(item) for item in getattr(verification, "grounded_claims", ())],
            "reasoning_checks": [_plain(item) for item in getattr(verification, "reasoning_checks", ())],
            "conclusion_answer": _plain(getattr(verification, "conclusion_answer", None)),
        }
    return payload


def append_teacher_trajectory_record(
    *,
    output_dir: str | None,
    opsd_config: dict[str, Any],
    rank: int,
    record: dict[str, Any],
) -> str | None:
    cfg = _trajectory_log_config(opsd_config)
    if not cfg["enabled"] or not output_dir:
        return None
    directory = cfg.get("dir") or os.path.join(output_dir, "teacher_trajectories")
    os.makedirs(directory, exist_ok=True)
    path = cfg.get("path") or os.path.join(directory, f"rank{int(rank)}.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path
