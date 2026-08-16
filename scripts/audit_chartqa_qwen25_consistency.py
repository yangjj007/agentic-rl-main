#!/usr/bin/env python3
"""Audit and directly repair the Qwen2.5-rewritten ChartQA corpus.

The input annotation, its Qwen rewrite, and the cached DePlot table are not
independent sources.  This tool therefore uses two *blind* visual-teacher
controls for only the rows with a deterministic hard-risk signal:

* image only;
* image plus a clearly-labelled, potentially noisy DePlot table.

Neither prompt receives the gold answer or the rewritten hint. When the two
controls agree exactly with a hard gold-vs-hint conflict, the resolved answer
is written into a single corrected training corpus. For a gold-confirming
repair, a label-consistent original ChartQA rationale is normalized into the
Qwen four-section schema. For an image-confirming label correction, the
existing Qwen rationale is retained only when it validates against the
resolved answer. No sample is deleted and no retain/filter/manual-review
split is produced; the JSONL patch ledger is the audit trail.

Run six workers (one frozen local teacher per idle GPU), then merge:

  for gpu in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$gpu /data/junjie/.miniforge3/envs/eval3d/bin/python \
      scripts/audit_chartqa_qwen25_consistency.py --worker-id $gpu --num-workers 6 &
  done; wait
  /data/junjie/.miniforge3/envs/eval3d/bin/python \
    scripts/audit_chartqa_qwen25_consistency.py --merge
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils.chart.deplot_pipeline import format_deplot_for_teacher
from data_utils.chart.evaluator import eval_teacher_probe_chart, parse_teacher_probe_answer
from data_utils.paths import resolve_image_path
from main import load_teacher_model
from opsd_utils.privileged.providers import CHARTQA_SHORT_ANSWER_HINT
from reward_utils.chart_cot_verifier import parse_number, verify_chart_cot_trajectory
from reward_utils.teacher_generate import TeacherGenerateRequest, teacher_generate_batched_chunks
from transformers import AutoProcessor


DEFAULT_DATASET = (
    PROJECT_ROOT / "data/chartqa/train_new_prerefine_vf_full_real_deplot_fp32_qwen25.json"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/chartqa-qwen25-consistency-audit"
DEFAULT_TEACHER = Path("/data/junjie/models/opd_eval3d/teacher-7b")
DEFAULT_CORRECTED_DATASET = (
    PROJECT_ROOT / "data/chartqa/train_new_prerefine_vf_full_real_deplot_fp32_qwen25_corrected.json"
)
_HINT_SECTION_RE = re.compile(r"(?im)^\s*(Goal|Observation|Reasoning|Conclusion)\s*:\s*")
_HIGH_SECTION_RE = re.compile(r"(?is)<(SUMMARY|CAPTION|REASONING|CONCLUSION)>\s*(.*?)\s*</\1>")

# A handful of the blind conflicts are caused by a known ChartQA artifact:
# the short-answer probe can answer the *label* named in a question (for
# example, ``No``) instead of the comparison, while the cached DePlot table
# and the chart make the comparison unambiguous.  These corrections are
# deliberately literal, auditable, and limited to the seven rows emitted by
# the completed blind audit.  They are not a general model-generated label
# rewrite.
_DIRECT_EVIDENCE_CORRECTIONS: dict[int, dict[str, str]] = {
    750: {
        "answer": "Yes",
        "hint": (
            "Goal: Determine whether the sum of the two smallest bars is greater "
            "than the largest bar.\n"
            "Observation: The chart values are Trinidad and Tobago 15.01, South "
            "Sudan 14.32, Western Sub-Saharan Africa 9.92, Venezuela 9.29, and "
            "Chad 9.17.\n"
            "Reasoning: The two smallest values are 9.17 and 9.29, whose sum is "
            "18.46. The largest value is 15.01, and 18.46 is greater than 15.01.\n"
            "Conclusion: Yes, the sum of the two smallest bars is greater than the "
            "largest bar."
        ),
    },
    1021: {
        "answer": "[26,15,11,5,3]",
        "hint": (
            "Goal: Report the peak values of all bars.\n"
            "Observation: The bars are Partisanship 26, Race 15, Geography 11, "
            "Gender 5, and Age 3.\n"
            "Reasoning: Reading the bars from highest to lowest gives 26, 15, 11, "
            "5, and 3.\n"
            "Conclusion: The peak values of all the bars are [26,15,11,5,3]."
        ),
    },
    1464: {
        "answer": "1.8",
        "hint": (
            "Goal: Find the ratio between Adult content and Twitter.com.\n"
            "Observation: Adult content is 90.0 and Twitter.com is 50.0.\n"
            "Reasoning: Divide 90.0 by 50.0: 90.0 / 50.0 = 1.8.\n"
            "Conclusion: The ratio between Adult content and Twitter.com is 1.8."
        ),
    },
    2170: {
        "answer": "No",
        "hint": (
            "Goal: Determine whether the share in 2010 is greater than the share "
            "in 2012.\n"
            "Observation: The chart shows 55% in 2010 and 57% in 2012.\n"
            "Reasoning: 55% is less than 57%, so the 2010 share is not greater.\n"
            "Conclusion: No, the share in 2010 is not greater than the share in 2012."
        ),
    },
    2550: {
        "answer": "Yes",
        "hint": (
            "Goal: Determine whether the mode is greater than the median.\n"
            "Observation: The operating percentages are 30%, 26%, 26%, 22%, "
            "19%, and 20%.\n"
            "Reasoning: The mode is 26%. After sorting the values as 19%, 20%, "
            "22%, 26%, 26%, 30%, the median is (22% + 26%) / 2 = 24%. Since "
            "26% is greater than 24%, the statement is true.\n"
            "Conclusion: Yes, the mode is greater than the median."
        ),
    },
    3628: {
        "answer": "2020",
        "hint": (
            "Goal: Find the year with the highest average age of marriage for males.\n"
            "Observation: The male values rise to 39.5 in 2020, above 39.3 in 2019 "
            "and all earlier years.\n"
            "Reasoning: The maximum male average age is 39.5, which occurs in 2020.\n"
            "Conclusion: The year with the highest average age of marriage for males is 2020."
        ),
    },
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"dataset must be a JSON list of objects: {path}")
    return data


def _clean_answer(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"(?i)^\s*answer\s*:\s*", "", text).strip()


def _answer_kind(answer: str) -> str:
    answer = _clean_answer(answer)
    if answer.lower() in {"yes", "no"}:
        return "binary"
    if parse_number(answer) is not None:
        return "numeric"
    return "text"


def _answer_matches(candidate: str, reference: str) -> bool:
    """Use the production ChartQA relaxed comparison on a short answer."""
    candidate = _clean_answer(candidate)
    reference = _clean_answer(reference)
    if not candidate or not reference:
        return False
    # eval_teacher_probe_chart supplies the production normalization for text,
    # numeric tolerance, percentages, and short answer parsing.
    score, parsed = eval_teacher_probe_chart(
        f"Answer: {candidate}",
        reference,
        max_relative_change=0.05,
    )
    return not parsed.parse_failed and bool(score >= 1.0)


def _answers_strictly_equivalent(left: str, right: str) -> bool:
    """Equality for two independent VLM controls; never apply 5% tolerance."""
    left = _clean_answer(left)
    right = _clean_answer(right)
    if not left or not right:
        return False
    left_num = parse_number(left)
    right_num = parse_number(right)
    if left_num is not None or right_num is not None:
        return bool(
            left_num is not None
            and right_num is not None
            and left_num.value == right_num.value
            and left_num.is_percent == right_num.is_percent
        )
    normalize = lambda value: " ".join(re.sub(r"[\W_]+", " ", value.casefold()).split())
    return normalize(left) == normalize(right)


def _hint_conflict(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return only high-precision gold-vs-Qwen-hint conflict candidates."""
    answer = _clean_answer(row.get("answer"))
    hint = str(row.get("hint") or "")
    if not answer or not hint:
        return None

    # The structured verifier adds independent hard signals (e.g. exact
    # table-cell contradiction or an invalid arithmetic statement).  It is
    # deliberately not used to edit a row without blind VLM agreement.
    verification = verify_chart_cot_trajectory(
        f"{hint}\nAnswer: {answer}",
        row.get("visual_fact_deplot"),
        answer_correct=True,
        require_two_bindings_for_multirow=False,
    )
    verifier_codes = set(verification.reason_codes)
    deterministic = sorted(
        verifier_codes & {"grounding_contradiction", "reasoning_invalid"}
    )
    conclusion_conflict = verification.conclusion_answer.status == "inconsistent"
    if not conclusion_conflict and not deterministic:
        return None
    if conclusion_conflict:
        reason = "hint_conclusion_conflicts_gold"
    else:
        reason = "deterministic_" + "_and_".join(deterministic)
    return {
        "risk_reason": reason,
        "gold_answer": answer,
        "hint_conclusion": verification.parsed.conclusion,
        "hint_conclusion_value": verification.conclusion_answer.conclusion_value,
        "answer_kind": _answer_kind(answer),
        "deterministic_reason_codes": list(verification.reason_codes),
        "deterministic_quality": verification.quality,
    }


def _candidate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        if row.get("human_or_machine", 0) != 0:
            continue
        conflict = _hint_conflict(row)
        if conflict is not None:
            candidates.append({"source_index": source_index, **conflict})
    return candidates


def _audit_prompt(row: dict[str, Any], *, include_deplot: bool) -> str:
    prompt = (
        "You are independently auditing one ChartQA example. The attached chart image "
        "is authoritative. Answer the user's question from the chart. Do not infer any "
        "hidden reference answer and do not discuss this audit.\n\n"
        f"Question:\n{str(row.get('question') or row.get('question_wo_prompt') or '').strip()}\n\n"
    )
    if include_deplot:
        table = format_deplot_for_teacher(row.get("visual_fact_deplot")).strip()
        if table:
            prompt += (
                "The following is an automatic OCR table. It can contain transcription "
                "errors, so resolve conflicts in favor of the image.\n"
                f"[OCR table]\n{table}\n\n"
            )
    return prompt + CHARTQA_SHORT_ANSWER_HINT


def _request_pair(row: dict[str, Any], max_new_tokens: int) -> list[TeacherGenerateRequest]:
    image = resolve_image_path(str(row.get("image") or ""))
    if not Path(image).is_file():
        raise FileNotFoundError(f"missing ChartQA image for audit: {image}")
    common = {
        "images": [image],
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
    }
    return [
        TeacherGenerateRequest(prompt_text=_audit_prompt(row, include_deplot=False), **common),
        TeacherGenerateRequest(prompt_text=_audit_prompt(row, include_deplot=True), **common),
    ]


def _parsed_answer(text: str) -> str:
    parsed = parse_teacher_probe_answer(text)
    return parsed.answer if not parsed.parse_failed else ""


def _decision(candidate: dict[str, Any], image_answer: str, deplot_answer: str) -> tuple[str, str]:
    """Return a direct correction decision plus its blind-evidence reason."""
    if not image_answer or not deplot_answer:
        return "unchanged", "one_or_both_vlm_controls_unparseable"
    if not _answers_strictly_equivalent(image_answer, deplot_answer):
        return "unchanged", "vlm_controls_disagree"

    agreed = image_answer
    matches_gold = _answer_matches(agreed, candidate["gold_answer"])
    matches_hint = _answer_matches(agreed, candidate["hint_conclusion_value"])
    # Repair only when the two blinded VLM readings settle the specific
    # gold-vs-Qwen conflict.  A DePlot OCR mismatch by itself is not enough:
    # it could be a harmless OCR rounding error rather than a bad annotation.
    if matches_gold and candidate["risk_reason"] == "hint_conclusion_conflicts_gold":
        return "repair_hint_keep_gold", "blind_vlm_agrees_with_gold_not_qwen_hint"
    if matches_hint and not matches_gold:
        return "repair_answer_and_hint", "blind_vlm_agrees_with_qwen_hint_not_gold"
    if matches_gold and matches_hint:
        return "unchanged", "gold_and_hint_equivalent_under_chartqa_metric"
    return "unchanged", "blind_vlm_agrees_with_neither_side"


def _repair_hint_is_valid(hint: str, answer: str) -> bool:
    matches = list(_HINT_SECTION_RE.finditer(str(hint or "")))
    if [match.group(1).lower() for match in matches] != [
        "goal", "observation", "reasoning", "conclusion"
    ]:
        return False
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(hint)
        sections[match.group(1).lower()] = hint[match.end() : end].strip()
    if not all(sections.get(name) for name in ("goal", "observation", "reasoning", "conclusion")):
        return False
    # The conclusion is normally a sentence ("The rightmost value is 51"),
    # while the training answer is a short label.  The production answer
    # parser intentionally rejects prose here, so accept an explicit short
    # answer occurrence using the same numeric/text normalization instead.
    conclusion = sections["conclusion"]
    normalized_answer = _clean_answer(answer)
    if _answer_kind(normalized_answer) == "numeric":
        target = parse_number(normalized_answer)
        return bool(
            target is not None
            and any(
                parse_number(token) is not None
                and parse_number(token).value == target.value
                and parse_number(token).is_percent == target.is_percent
                for token in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", conclusion)
            )
        )
    return bool(
        re.search(
            rf"(?i)(?<!\w){re.escape(normalized_answer)}(?!\w)",
            conclusion,
        )
    )


def _high_hint_to_qwen_schema(raw_hint: str, answer: str) -> str:
    """Convert an answer-verified original ChartQA rationale to Qwen schema.

    ``train_high.json`` is the pre-Qwen source annotation for this corpus.  It
    is preferable to a fresh VLM long-form generation whenever its own
    conclusion exactly agrees with the official ChartQA label: no new visual
    inference is introduced, and the factual chain comes from the original
    dataset annotation.  This function only changes heading syntax.
    """
    sections = {name.lower(): body.strip() for name, body in _HIGH_SECTION_RE.findall(str(raw_hint or ""))}
    if not all(sections.get(key) for key in ("summary", "caption", "reasoning", "conclusion")):
        return ""
    # The original summary is a natural Goal.  Caption names visual evidence;
    # preserve it as Observation rather than inventing a new table reading.
    result = "\n".join(
        [
            f"Goal: {sections['summary']}",
            f"Observation: {sections['caption']}",
            f"Reasoning: {sections['reasoning']}",
            f"Conclusion: {sections['conclusion']}",
        ]
    )
    return result if _repair_hint_is_valid(result, answer) else ""


def _high_source_by_question() -> dict[str, list[dict[str, Any]]]:
    path = PROJECT_ROOT / "data/chartqa/train_high.json"
    if not path.is_file():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _load_rows(path):
        question = str(row.get("question") or "").strip()
        if question:
            grouped.setdefault(question, []).append(row)
    return grouped


def _verified_high_repair(
    row: dict[str, Any],
    answer: str,
    high_by_question: dict[str, list[dict[str, Any]]],
) -> str:
    """Return a schema-normalized original rationale if its conclusion is gold-consistent."""
    question = str(row.get("question") or row.get("question_wo_prompt") or "").strip()
    for source in high_by_question.get(question, []):
        candidate_answer = str(source.get("answer") or "").strip()
        # Use the production relaxed metric for the original ChartQA label
        # mapping, then exact conclusion validation after schema conversion.
        if not _answer_matches(candidate_answer, answer):
            continue
        normalized = _high_hint_to_qwen_schema(str(source.get("hint") or ""), answer)
        if normalized:
            return normalized
    return ""


def _apply_audit_corrections(args: argparse.Namespace) -> int:
    """Apply direct evidence-confirmed corrections without deleting rows."""
    rows = _load_rows(args.dataset)
    path = args.out_dir / "qwen25_consistency_vlm_report.jsonl"
    reports = {
        int(row["source_index"]): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    high_by_question = _high_source_by_question()
    corrected_rows: list[dict[str, Any]] = []
    patches: list[dict[str, Any]] = []
    for source_index, source in enumerate(rows):
        row = dict(source)
        record: dict[str, Any] | None = None
        direct = _DIRECT_EVIDENCE_CORRECTIONS.get(source_index)
        if direct is not None:
            # These rows were manually checked against the concrete chart and
            # its cached real DePlot table after the blind probe exposed a
            # probe-vs-question semantic failure.  Keep the correction
            # explicit and auditable rather than asking a model to regenerate
            # a rationale.
            record = {
                "source_index": source_index,
                "decision": "direct_evidence_correction",
                "decision_reason": "chart_and_real_deplot_manual_consistency_audit",
                "repair_answer": direct["answer"],
                "repair_hint": direct["hint"],
                "repair_status": "applied",
                "image_only_answer": "",
                "image_plus_deplot_answer": "",
            }
        # The only answer correction rule is deliberately narrow: two blind
        # reads (image-only and image+clearly-noisy-OCR) must produce exactly
        # the same short answer, and that answer must also agree with the
        # existing Qwen conclusion while disagreeing with the source label.
        # This keeps the correction independent of the original label while
        # requiring three mutually reinforcing signals.
        report = reports.get(source_index)
        if record is not None and direct is not None and report is not None:
            record["image_only_answer"] = report.get("image_only_answer", "")
            record["image_plus_deplot_answer"] = report.get("image_plus_deplot_answer", "")
        if record is None and report is not None:
            decision, decision_reason = _decision(
                report,
                str(report.get("image_only_answer") or ""),
                str(report.get("image_plus_deplot_answer") or ""),
            )
            if decision == "repair_hint_keep_gold":
                high_hint = _verified_high_repair(row, report["gold_answer"], high_by_question)
                if high_hint:
                    record = {
                        "source_index": source_index,
                        "decision": decision,
                        "decision_reason": decision_reason + "_verified_original_chartqa_rationale",
                        "repair_answer": report["gold_answer"],
                        "repair_hint": high_hint,
                        "repair_status": "applied",
                        "image_only_answer": report["image_only_answer"],
                        "image_plus_deplot_answer": report["image_plus_deplot_answer"],
                    }
            # Generic model ``repair_answer_and_hint`` decisions are not
            # applied here. The completed audit exposed probe semantic
            # failures, so answer edits must come from the explicit,
            # chart-reviewed correction table above.
        if record is not None:
            old_answer, old_hint = str(row.get("answer") or ""), str(row.get("hint") or "")
            if record["decision"] in {"repair_answer_and_hint", "direct_evidence_correction"}:
                row["answer"] = record["repair_answer"]
            row["hint"] = record["repair_hint"]
            meta = dict(row.get("dyme_rewrite") or {})
            meta.update(
                {
                    "status": "ok",
                    "audit": "qwen25_visual_consistency_repaired",
                    "audit_model": str(args.teacher),
                    "rewritten_hint_chars": len(row["hint"]),
                    "audit_original_answer": old_answer,
                    "audit_original_hint_chars": len(old_hint),
                    "audit_resolved_answer": str(row["answer"]),
                    "audit_decision": record["decision"],
                    "audit_evidence": record["decision_reason"],
                }
            )
            row["dyme_rewrite"] = meta
            patches.append(
                {
                    "source_index": source_index,
                    "decision": record["decision"],
                    "decision_reason": record["decision_reason"],
                    "old_answer": old_answer,
                    "new_answer": row["answer"],
                    "old_hint": old_hint,
                    "new_hint": row["hint"],
                    "image_only_answer": record["image_only_answer"],
                    "image_plus_deplot_answer": record["image_plus_deplot_answer"],
                    "blind_image_only_output": (report or {}).get("image_only_output", ""),
                    "blind_image_plus_deplot_output": (report or {}).get(
                        "image_plus_deplot_output", ""
                    ),
                    "blind_decision": (report or {}).get("decision", ""),
                    "blind_decision_reason": (report or {}).get("decision_reason", ""),
                    "gold_answer_from_audit": (report or {}).get("gold_answer", old_answer),
                    "qwen_hint_conclusion_value": (report or {}).get(
                        "hint_conclusion_value", ""
                    ),
                }
            )
        corrected_rows.append(row)
    corrected_path = DEFAULT_CORRECTED_DATASET
    corrected_path.parent.mkdir(parents=True, exist_ok=True)
    corrected_path.write_text(json.dumps(corrected_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    corrections_path = args.out_dir / "qwen25_consistency_corrections.jsonl"
    with corrections_path.open("w", encoding="utf-8") as handle:
        for patch in patches:
            handle.write(json.dumps(patch, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = args.out_dir / "qwen25_consistency_summary.json"
    old_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    applied_by_decision = Counter(patch["decision"] for patch in patches)
    report_decisions = Counter(record.get("decision", "") for record in reports.values())
    old_summary.update(
        {
            "corrected_dataset": str(corrected_path),
            "corrections": str(corrections_path),
            "repaired_rows": len(patches),
            "unchanged_rows": len(rows) - len(patches),
            "decision_counts": dict(report_decisions),
            "applied_decision_counts": dict(applied_by_decision),
            "answer_corrected_rows": applied_by_decision["repair_answer_and_hint"]
            + applied_by_decision["direct_evidence_correction"],
            "hint_corrected_rows": applied_by_decision["repair_hint_keep_gold"]
            + applied_by_decision["direct_evidence_correction"],
            "direct_evidence_corrections": sorted(_DIRECT_EVIDENCE_CORRECTIONS),
            "repair_status_counts": {
                "applied": len(patches),
                "not_requested": len(rows) - len(reports),
                "rejected_or_unapplied": len(reports) - len(patches),
            },
            "unapplied_model_answer_corrections": sum(
                record.get("decision") == "repair_answer_and_hint"
                for record in reports.values()
            ),
        }
    )
    summary_path.write_text(json.dumps(old_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[QWEN25-REPAIR] apply " + json.dumps(old_summary, ensure_ascii=False), flush=True)
    return 0


def _worker(args: argparse.Namespace) -> int:
    rows = _load_rows(args.dataset)
    candidates = _candidate_rows(rows)
    shard = [candidate for idx, candidate in enumerate(candidates) if idx % args.num_workers == args.worker_id]
    out_dir = args.out_dir / "vlm_workers"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"worker_{args.worker_id:02d}.jsonl"
    print(
        f"[QWEN25-AUDIT] worker={args.worker_id}/{args.num_workers} "
        f"candidates_total={len(candidates)} shard={len(shard)} device=cuda:0",
        flush=True,
    )

    model_cfg = {
        "pretrained_model_path": str(args.teacher),
        "teacher_model_path": str(args.teacher),
        "use_flash_attention_2": bool(args.flash_attention),
        "torch_dtype": "bfloat16",
        "teacher_dtype": "bfloat16",
        "teacher_device_map": "cuda:0",
    }
    # Do not load a second copy of the 7B model merely to obtain its processor:
    # every audit worker owns exactly one frozen VLM on its assigned GPU.
    processor = AutoProcessor.from_pretrained(str(args.teacher), local_files_only=True)
    processor.tokenizer.padding_side = "left"
    teacher = load_teacher_model(model_cfg, local_rank=0, num_gpus=1)

    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(shard), args.batch_size):
            batch_meta = shard[start : start + args.batch_size]
            requests: list[TeacherGenerateRequest] = []
            for meta in batch_meta:
                requests.extend(_request_pair(rows[meta["source_index"]], args.max_new_tokens))
            texts, latency_ms = teacher_generate_batched_chunks(
                teacher,
                processor,
                requests,
                chunk_size=max(1, args.batch_size * 2),
                timing_kind="qwen25_consistency_audit",
            )
            if len(texts) != len(requests):
                raise RuntimeError(f"teacher returned {len(texts)} outputs for {len(requests)} requests")
            for offset, meta in enumerate(batch_meta):
                image_output, deplot_output = texts[offset * 2 : offset * 2 + 2]
                image_answer = _parsed_answer(image_output)
                deplot_answer = _parsed_answer(deplot_output)
                decision, decision_reason = _decision(meta, image_answer, deplot_answer)
                record = {
                    **meta,
                    "question": str(rows[meta["source_index"]].get("question") or ""),
                    "image": str(rows[meta["source_index"]].get("image") or ""),
                    "image_only_output": image_output,
                    "image_only_answer": image_answer,
                    "image_plus_deplot_output": deplot_output,
                    "image_plus_deplot_answer": deplot_answer,
                    "decision": decision,
                    "decision_reason": decision_reason,
                    "repair_answer": (
                        meta["gold_answer"]
                        if decision == "repair_hint_keep_gold"
                        else image_answer
                        if decision == "repair_answer_and_hint"
                        else ""
                    ),
                    "batch_latency_ms": round(latency_ms, 2),
                    "audit_model": str(args.teacher),
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            done = min(start + len(batch_meta), len(shard))
            print(
                f"[QWEN25-AUDIT] worker={args.worker_id} progress={done}/{len(shard)} "
                f"batch_latency_s={latency_ms / 1000.0:.2f}",
                flush=True,
            )
    del teacher
    return 0


def _merge(args: argparse.Namespace) -> int:
    rows = _load_rows(args.dataset)
    candidates = _candidate_rows(rows)
    worker_dir = args.out_dir / "vlm_workers"
    records: dict[int, dict[str, Any]] = {}
    for path in sorted(worker_dir.glob("worker_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            source_index = int(record["source_index"])
            if source_index in records:
                raise ValueError(f"duplicate worker result for source index {source_index}")
            records[source_index] = record
    expected = {int(candidate["source_index"]) for candidate in candidates}
    missing = sorted(expected - set(records))
    if missing:
        raise RuntimeError(
            f"cannot merge audit: missing {len(missing)} VLM results; first source index={missing[0]}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = args.out_dir / "qwen25_consistency_vlm_report.jsonl"
    with report.open("w", encoding="utf-8") as handle:
        for source_index in sorted(records):
            handle.write(json.dumps(records[source_index], ensure_ascii=False, sort_keys=True) + "\n")
    # Applying never loads a second VLM or asks it for a long rationale: this
    # prevents a short-answer-correct but self-contradictory generation from
    # entering the training corpus.  It reuses the auditable blind outputs.
    return _apply_audit_corrections(args)


def _write_candidates(args: argparse.Namespace) -> int:
    candidates = _candidate_rows(_load_rows(args.dataset))
    payload = args.out_dir / "deterministic_risk_candidates.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with payload.open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        "[QWEN25-AUDIT] deterministic "
        + json.dumps({"candidates": len(candidates), "output": str(payload)}, ensure_ascii=False),
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--write-candidates", action="store_true")
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.out_dir = args.out_dir.resolve()
    args.teacher = args.teacher.resolve()
    if args.num_workers <= 0 or not 0 <= args.worker_id < args.num_workers:
        parser.error("worker-id must be in [0, num-workers)")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("batch-size and max-new-tokens must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.merge:
        return _merge(args)
    if args.write_candidates:
        return _write_candidates(args)
    return _worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
