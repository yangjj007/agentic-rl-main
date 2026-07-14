"""Teacher-correct trajectory SFT repair helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from opsd_utils.constants import MODE_OPSD, MODE_SFT


@dataclass(frozen=True)
class TeacherSftRepairConfig:
    repair_mode: str = "opd"
    scope: str = "all_wrong"
    slots_per_prompt: int = 1
    target_max_tokens: int = 256
    sanitize_privileged: bool = True

    @property
    def enabled(self) -> bool:
        return self.repair_mode == "traj_sft" and self.slots_per_prompt > 0


@dataclass
class TeacherSftRepairStats:
    teacher_sft_repairs: int = 0
    teacher_sft_repair_all_wrong: int = 0
    repair_slot_eligible: int = 0
    teacher_correct_to_opd: int = 0
    teacher_correct_to_sft_repair: int = 0


_PRIVILEGED_LINE_RE = re.compile(
    r"^\s*\[(?:Verified Hint|Reference Answer|DePlot|Visual Facts[^\]]*)\].*$",
    re.IGNORECASE,
)
_PRIVILEGED_TAG_RE = re.compile(
    r"\[(?:Verified Hint|Reference Answer|DePlot|Visual Facts[^\]]*)\]",
    re.IGNORECASE,
)
_CHARTQA_SECTION_RE = re.compile(
    r"(?is)(goal|observation|reasoning|conclusion)\s*:\s*(.*?)(?=(?:\n\s*)?(?:goal|observation|reasoning|conclusion|answer)\s*:|$)"
)
_ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*:\s*(.*?)\s*$")
_REQUIRED_SECTION_KEYS = ("goal", "observation", "reasoning", "conclusion")


@dataclass(frozen=True)
class ConstrainedTeacherSftTarget:
    text: str
    raw_full_hint_format: bool
    full_hint_format: bool
    exact_reference_answer_line: bool
    privileged_tag_present: bool
    used_fallback_hint: bool
    raw_clipped: bool
    student_short_format: bool = False
    answer_only_format: bool = False


def _as_config(raw: TeacherSftRepairConfig | dict | None) -> TeacherSftRepairConfig:
    if isinstance(raw, TeacherSftRepairConfig):
        return raw
    raw = raw or {}
    return TeacherSftRepairConfig(
        repair_mode=str(raw.get("mode", raw.get("repair_mode", "opd")) or "opd"),
        scope=str(raw.get("scope", "all_wrong") or "all_wrong"),
        slots_per_prompt=max(0, int(raw.get("slots_per_prompt", raw.get("slots", 1)) or 0)),
        target_max_tokens=max(1, int(raw.get("target_max_tokens", 256) or 256)),
        sanitize_privileged=bool(raw.get("sanitize_privileged", True)),
    )


def sanitize_teacher_sft_text(text: str) -> str:
    """Remove privileged-section labels from teacher text before student SFT."""
    kept: list[str] = []
    for line in _decode_escaped_newlines(text).splitlines():
        if _PRIVILEGED_LINE_RE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _decode_escaped_newlines(text: Any) -> str:
    value = str(text or "")
    return (
        value.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
    )


def _clean_answer_text(answer: Any) -> str:
    text = str(answer or "").strip()
    text = re.sub(r"(?i)^\s*answer\s*:\s*", "", text).strip()
    return text


def _normalize_section_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _extract_chartqa_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    normalized = _decode_escaped_newlines(text)
    for match in _CHARTQA_SECTION_RE.finditer(normalized):
        key = match.group(1).lower()
        value = _normalize_section_text(match.group(2))
        if value:
            sections[key] = value
    return sections


def _sample_hint_text(sample: dict[str, Any] | None) -> str:
    sample = sample or {}
    value = sample.get("hint")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _sample_verified_hint_text(sample: dict[str, Any] | None) -> str:
    sample = sample or {}
    value = sample.get("hint")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _sample_question_text(sample: dict[str, Any] | None) -> str:
    sample = sample or {}
    for key in ("question", "question_wo_prompt", "prompt"):
        value = str(sample.get(key) or "").strip()
        if value:
            return value
    return "Answer the chart question."


def _looks_clipped_or_template(text: str) -> bool:
    cleaned = _decode_escaped_newlines(text).strip()
    if not cleaned:
        return True
    lower = cleaned.lower()
    if lower.endswith("...") or lower.endswith("…"):
        return True
    if re.search(r"(?i)(goal|observation|reasoning|conclusion)\s*:\s*(?:\.|\.{3}|…|\([^)]*\))\s*$", cleaned):
        return True
    if re.search(r"(?i)\b(the|a|an|this|that)\.\.\.$", cleaned):
        return True
    return False


def _has_all_chartqa_sections(sections: dict[str, str]) -> bool:
    return all(bool(sections.get(key)) for key in _REQUIRED_SECTION_KEYS)


def _answer_last_line(text: str) -> bool:
    lines = [line.strip() for line in _decode_escaped_newlines(text).splitlines() if line.strip()]
    return bool(lines and lines[-1].lower().startswith("answer:"))


def _has_required_headings_in_order(text: str) -> bool:
    lower = _decode_escaped_newlines(text).lower()
    pos = -1
    for heading in ("goal:", "observation:", "reasoning:", "conclusion:", "answer:"):
        next_pos = lower.find(heading, pos + 1)
        if next_pos < 0:
            return False
        pos = next_pos
    return True


def _exact_reference_answer_line(text: str, reference_answer: Any) -> bool:
    answer = _clean_answer_text(reference_answer)
    if not answer:
        return False
    expected = f"Answer: {answer}"
    return any(line.strip() == expected for line in _decode_escaped_newlines(text).splitlines())


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in _decode_escaped_newlines(text).splitlines() if line.strip()]


def _answer_only_format(text: str, reference_answer: Any = "") -> bool:
    lines = _nonempty_lines(text)
    if len(lines) != 1 or not lines[0].lower().startswith("answer:"):
        return False
    answer = _clean_answer_text(reference_answer)
    return not answer or lines[0] == f"Answer: {answer}"


def _student_short_format(text: str, reference_answer: Any = "") -> bool:
    lines = _nonempty_lines(text)
    if not lines:
        return False
    lower = "\n".join(lines).lower()
    if any(heading in lower for heading in ("goal:", "observation:", "conclusion:")):
        return False
    if len(lines) == 1:
        return _answer_only_format(text, reference_answer)
    if len(lines) != 2 or not lines[0].lower().startswith("reasoning:"):
        return False
    if not lines[-1].lower().startswith("answer:"):
        return False
    answer = _clean_answer_text(reference_answer)
    return not answer or lines[-1] == f"Answer: {answer}"


def _section_answer_aligned(text: str, answer: str) -> bool:
    if not answer:
        return True
    answer_norm = re.sub(r"\s+", " ", answer).strip().lower()
    text_norm = re.sub(r"\s+", " ", text).strip().lower()
    return bool(answer_norm and answer_norm in text_norm)


def _build_default_sections(sample: dict[str, Any] | None, answer: str) -> dict[str, str]:
    question = _sample_question_text(sample)
    return {
        "goal": question,
        "observation": "Use the verified chart evidence for this question.",
        "reasoning": "Follow the verified reasoning and align the final result with the reference answer.",
        "conclusion": f"Therefore, the answer is {answer}." if answer else "Therefore, use the verified answer.",
    }


def _format_chartqa_target(sections: dict[str, str], answer: str) -> str:
    conclusion = _normalize_section_text(sections.get("conclusion"))
    if answer and not _section_answer_aligned(conclusion, answer):
        conclusion = f"Therefore, the answer is {answer}."
    rows = [
        ("Goal", sections.get("goal")),
        ("Observation", sections.get("observation")),
        ("Reasoning", sections.get("reasoning")),
        ("Conclusion", conclusion),
        ("Answer", answer),
    ]
    return "\n".join(f"{name}: {_normalize_section_text(value)}" for name, value in rows).strip()


def teacher_sft_target_quality(text: str, reference_answer: Any = "") -> dict[str, bool]:
    """Return format checks for a teacher SFT repair target."""
    normalized = _decode_escaped_newlines(text)
    sections = _extract_chartqa_sections(normalized)
    return {
        "full_hint_format": _has_all_chartqa_sections(sections)
        and _has_required_headings_in_order(normalized)
        and _answer_last_line(normalized)
        and bool(_ANSWER_LINE_RE.search(normalized)),
        "answer_last_line": _answer_last_line(normalized),
        "exact_reference_answer_line": _exact_reference_answer_line(normalized, reference_answer),
        "privileged_tag_present": bool(_PRIVILEGED_TAG_RE.search(normalized)),
        "raw_clipped": _looks_clipped_or_template(normalized),
        "student_short_format": _student_short_format(normalized, reference_answer),
        "answer_only_format": _answer_only_format(normalized, reference_answer),
    }


def constrain_teacher_sft_repair_target(
    text: str,
    *,
    sample: dict[str, Any] | None = None,
    reference_answer: Any = "",
    sanitize_privileged: bool = True,
) -> ConstrainedTeacherSftTarget:
    """Constrain a teacher-generated repair target to ChartQA hint style.

    This mirrors the visual refiner's guarded pattern: accept teacher output only
    when it is structurally usable, otherwise fall back to the verified hint. In
    both cases the final answer line is deterministic and reference-answer based.
    """
    answer = _clean_answer_text(reference_answer or (sample or {}).get("answer"))
    raw = sanitize_teacher_sft_text(text) if sanitize_privileged else _decode_escaped_newlines(text).strip()
    raw_sections = _extract_chartqa_sections(raw)
    raw_quality = teacher_sft_target_quality(raw, answer)
    raw_clipped = bool(raw_quality["raw_clipped"])
    use_raw = _has_all_chartqa_sections(raw_sections) and not raw_clipped
    used_fallback_hint = not use_raw

    if use_raw:
        sections = raw_sections
    else:
        hint_sections = _extract_chartqa_sections(_sample_hint_text(sample))
        if _has_all_chartqa_sections(hint_sections):
            sections = hint_sections
        else:
            sections = _build_default_sections(sample, answer)

    target = _format_chartqa_target(sections, answer)
    quality = teacher_sft_target_quality(target, answer)
    return ConstrainedTeacherSftTarget(
        text=target,
        raw_full_hint_format=bool(raw_quality["full_hint_format"]),
        full_hint_format=bool(quality["full_hint_format"]),
        exact_reference_answer_line=bool(quality["exact_reference_answer_line"]),
        privileged_tag_present=bool(quality["privileged_tag_present"]),
        used_fallback_hint=used_fallback_hint,
        raw_clipped=raw_clipped,
        student_short_format=bool(quality["student_short_format"]),
        answer_only_format=bool(quality["answer_only_format"]),
    )


def _build_student_short_target(raw: str, sample: dict[str, Any] | None, answer: str) -> str:
    raw_sections = _extract_chartqa_sections(raw)
    reasoning = _normalize_section_text(raw_sections.get("reasoning"))
    if not reasoning:
        hint_sections = _extract_chartqa_sections(_sample_hint_text(sample))
        reasoning = _normalize_section_text(hint_sections.get("reasoning"))
    if reasoning:
        return f"Reasoning: {reasoning}\nAnswer: {answer}".strip()
    return f"Answer: {answer}".strip()


def _build_student_hint_short_target(raw: str, sample: dict[str, Any] | None, answer: str) -> str:
    hint_sections = _extract_chartqa_sections(_sample_verified_hint_text(sample))
    reasoning = _normalize_section_text(hint_sections.get("reasoning"))
    if not reasoning:
        raw_sections = _extract_chartqa_sections(raw)
        reasoning = _normalize_section_text(raw_sections.get("reasoning"))
    if reasoning:
        return f"Reasoning: {reasoning}\nAnswer: {answer}".strip()
    return f"Answer: {answer}".strip()


def build_teacher_sft_repair_target(
    text: str,
    *,
    sample: dict[str, Any] | None = None,
    reference_answer: Any = "",
    target_style: str = "chartqa_hint",
    sanitize_privileged: bool = True,
) -> ConstrainedTeacherSftTarget:
    """Build the student-visible teacher repair target for the selected style."""
    style = (target_style or "chartqa_hint").lower()
    answer = _clean_answer_text(reference_answer or (sample or {}).get("answer"))
    if style in ("chartqa_hint", "constrained", "constraint"):
        return constrain_teacher_sft_repair_target(
            text,
            sample=sample,
            reference_answer=answer,
            sanitize_privileged=sanitize_privileged,
        )

    raw = sanitize_teacher_sft_text(text) if sanitize_privileged else _decode_escaped_newlines(text).strip()
    raw_quality = teacher_sft_target_quality(raw, answer)
    if style == "answer_only":
        target = f"Answer: {answer}".strip()
        used_fallback_hint = False
    elif style == "student_short":
        target = _build_student_short_target(raw, sample, answer)
        used_fallback_hint = not bool(_extract_chartqa_sections(raw).get("reasoning"))
    elif style in ("student_hint_short", "hint_short", "verified_hint_short"):
        target = _build_student_hint_short_target(raw, sample, answer)
        used_fallback_hint = bool(_extract_chartqa_sections(_sample_verified_hint_text(sample)).get("reasoning"))
    else:
        target = raw
        used_fallback_hint = False

    quality = teacher_sft_target_quality(target, answer)
    student_short_format = bool(quality["student_short_format"])
    answer_only_format = bool(quality["answer_only_format"])
    if style == "answer_only":
        student_short_format = False
    return ConstrainedTeacherSftTarget(
        text=target,
        raw_full_hint_format=bool(raw_quality["full_hint_format"]),
        full_hint_format=bool(quality["full_hint_format"]),
        exact_reference_answer_line=bool(quality["exact_reference_answer_line"]),
        privileged_tag_present=bool(quality["privileged_tag_present"]),
        used_fallback_hint=used_fallback_hint,
        raw_clipped=bool(raw_quality["raw_clipped"]),
        student_short_format=student_short_format,
        answer_only_format=answer_only_format,
    )


def teacher_sft_repair_advantages(mask: torch.Tensor) -> torch.Tensor:
    """Return unit SFT-style advantages for a replacement target."""
    return torch.ones_like(mask, dtype=torch.float, device=mask.device)


def apply_teacher_sft_repair_routing(
    *,
    completion_modes: Sequence[int],
    teacher_traj_indices: Iterable[int],
    group_has_correct: Sequence[bool],
    num_generations: int,
    config: TeacherSftRepairConfig | dict | None,
) -> tuple[list[int], set[int], set[int], TeacherSftRepairStats]:
    """Promote selected all-wrong teacher-correct trajectories to SFT repair.

    Mixed wrong completions intentionally remain OPD. This preserves DyME's
    memorize/explore split while replacing all-wrong noisy student trajectories
    with verified teacher trajectories for a small number of repair slots.
    """
    cfg = _as_config(config)
    modes = list(completion_modes)
    kept_trajs = set(int(i) for i in teacher_traj_indices)
    repair_indices: set[int] = set()
    stats = TeacherSftRepairStats()
    per_prompt_used: dict[int, int] = {}

    if cfg.enabled and cfg.scope == "all_wrong":
        eligible_counts: dict[int, int] = {}
        for idx in sorted(kept_trajs):
            prompt_idx = idx // max(int(num_generations), 1)
            is_all_wrong = (
                not bool(group_has_correct[prompt_idx])
                if prompt_idx < len(group_has_correct)
                else False
            )
            if is_all_wrong:
                eligible_counts[prompt_idx] = eligible_counts.get(prompt_idx, 0) + 1
        stats.repair_slot_eligible = sum(
            min(cfg.slots_per_prompt, count)
            for count in eligible_counts.values()
        )

    for idx in sorted(kept_trajs):
        prompt_idx = idx // max(int(num_generations), 1)
        is_all_wrong = not bool(group_has_correct[prompt_idx]) if prompt_idx < len(group_has_correct) else False
        eligible_scope = cfg.scope == "all_wrong" and is_all_wrong
        if cfg.enabled and eligible_scope:
            used = per_prompt_used.get(prompt_idx, 0)
            if used < cfg.slots_per_prompt:
                per_prompt_used[prompt_idx] = used + 1
                modes[idx] = MODE_SFT
                repair_indices.add(idx)
                stats.teacher_sft_repairs += 1
                stats.teacher_sft_repair_all_wrong += 1
                stats.teacher_correct_to_sft_repair += 1
                continue

        if idx < len(modes) and modes[idx] == MODE_OPSD:
            stats.teacher_correct_to_opd += 1

    kept_trajs -= repair_indices
    return modes, kept_trajs, repair_indices, stats
