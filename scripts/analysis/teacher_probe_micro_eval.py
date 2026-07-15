#!/usr/bin/env python
"""Micro-evaluate teacher probe prompts on existing PCD candidates.

The script compares a no-gold DePlot-only teacher prompt with an intentional
oracle-hint prompt on the same candidate completions. Use --dry-run for prompt
inspection, --fake-teacher for fast metric plumbing tests, or run without either
flag to load the configured teacher model.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils.chart.evaluator import eval_teacher_probe_chart
from opsd_utils.evidence_harness import EvidenceAction, HarnessStatus
from opsd_utils.evidence_harness.chartqa import (
    build_chartqa_arithmetic_recovery_suffix,
    build_chartqa_executable_deplot_recovery_suffix,
    build_chartqa_executable_deplot_response_prefix,
    build_chartqa_scale_unit_recovery_suffix,
    build_chartqa_target_phrase_recovery_suffix,
    build_chartqa_target_phrase_response_prefix,
    build_chartqa_candidate,
    build_visual_base_suffix,
    build_visual_deplot_suffix,
    build_visual_recovery_suffix,
    decide_after_parallel_attempts,
    decide_after_recovery,
    validate_chartqa_candidate,
)
from opsd_utils.privileged import build_privileged_context
from opsd_utils.privileged.image_utils import load_rgb
from opsd_utils.privileged.providers import split_teacher_response_prefix
from opsd_utils.prompt_builder import _build_teacher_text

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional in tiny test envs.
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_CANDIDATE_GLOB = (
    "outputs/test-fast/pcd-no-visual/pcd_no_visual_aligned_4epoch/"
    "deplot_no_vs_opd_pcd/teacher_probe_candidates/rank*.jsonl"
)
DEFAULT_DATASET = "data/chartqa/train_medium_vf_full.json"
DEFAULT_OUT_DIR = "outputs/test-fast/teacher-probe-micro-eval/oracle_hint_official"

CONTROL_SPECS: dict[str, dict[str, Any]] = {
    "visual_short_answer": {
        "providers": ["format_only"],
        "prompt_profile": "chartqa_visual_short",
    },
    "visual_reasoned_answer": {
        "providers": ["format_only"],
        "prompt_profile": "chartqa_visual_reasoned",
    },
    "visual_chain_of_charts": {
        "providers": ["format_only"],
        "prompt_profile": "chartqa_visual_chain_of_charts",
    },
    "visual_zoom_short_answer": {
        "providers": ["format_only"],
        "prompt_profile": "chartqa_visual_zoom_short",
        "privileged_profile": "visual",
        "privileged_image": {
            "mode": "dual",
            "crop_strategy": "center",
            "margin_ratio": 0.12,
        },
    },
    "visual_answer_prefix": {
        "providers": ["format_only"],
        "prompt_profile": "chartqa_visual_answer_prefix",
    },
    "visual_answer_prefix_numeric": {
        "providers": ["format_only"],
        "prompt_profile": "chartqa_visual_answer_prefix_numeric",
    },
    "visual_operation_answer_prefix": {
        "providers": ["format_only"],
        "prompt_profile": "chartqa_visual_operation_answer_prefix",
    },
    "visual_deplot_answer_prefix": {
        "providers": ["visual_facts_deplot", "format_only"],
        "prompt_profile": "chartqa_deplot_answer_prefix",
    },
    "deplot_operation_answer_prefix": {
        "providers": ["visual_facts_deplot", "format_only"],
        "prompt_profile": "chartqa_deplot_operation_answer_prefix",
    },
    "baseline_deplot_only": {
        "providers": ["format_only", "visual_facts_deplot"],
        "prompt_profile": "chartqa_short_answer",
    },
    "reasoned_deplot_only": {
        "providers": ["visual_facts_deplot", "format_only"],
        "prompt_profile": "chartqa_deplot_reasoned",
        "canonicalize_draft": True,
    },
    "oracle_hint_deplot": {
        "providers": ["format_only", "visual_facts_deplot", "oracle_hint"],
        "prompt_profile": "chartqa_oracle_hint",
    },
}
DEFAULT_CONTROL_NAMES = ["baseline_deplot_only", "oracle_hint_deplot"]

REQUIRED_HINT_HEADINGS = ("goal:", "observation:", "reasoning:", "conclusion:", "answer:")


def _json_load(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _basename(value: Any) -> str:
    return Path(str(value or "")).name


def _norm_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _clean_answer(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"(?i)^\s*answer\s*:\s*", "", text).strip()
    return text


def _norm_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean_answer(value).strip()).lower()


def _record_question(record: dict[str, Any]) -> str:
    return str(
        record.get("question")
        or record.get("question_wo_prompt")
        or record.get("prompt")
        or ""
    ).strip()


def _record_prompt(record: dict[str, Any]) -> str:
    return str(record.get("prompt") or _record_question(record)).strip()


class DatasetIndex:
    def __init__(self, records: list[dict[str, Any]]):
        self.exact: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.by_question_image: dict[tuple[str, str], dict[str, Any]] = {}
        self.by_question_answer: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records:
            question = _norm_question(_record_question(record))
            image = _basename(record.get("image"))
            answer = _norm_answer(record.get("answer"))
            if question and image and answer:
                self.exact.setdefault((question, image, answer), record)
            if question and image:
                self.by_question_image.setdefault((question, image), record)
            if question and answer:
                self.by_question_answer.setdefault((question, answer), record)

    def lookup(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        question = _norm_question(candidate.get("question"))
        image = _basename(candidate.get("image"))
        answer = _norm_answer(candidate.get("reference") or candidate.get("answer"))
        if question and image and answer:
            found = self.exact.get((question, image, answer))
            if found is not None:
                return found
        if question and image:
            found = self.by_question_image.get((question, image))
            if found is not None:
                return found
        if question and answer:
            return self.by_question_answer.get((question, answer))
        return None


def _candidate_scope(row: dict[str, Any]) -> str:
    if row.get("is_all_wrong_probe_candidate") is True or row.get("group_all_wrong") is True:
        return "all_wrong"
    if row.get("is_mixed_wrong_probe_candidate") is True:
        return "mixed_wrong"
    route = str(row.get("route_reason") or "")
    if "mixed" in route:
        return "mixed_wrong"
    if "all_wrong" in route:
        return "all_wrong"
    return "other"


def _qtype(question: str) -> str:
    q = question.lower()
    if any(word in q for word in ("average", "mean")):
        return "average"
    if any(word in q for word in ("difference", "change", "more than", "less than")):
        return "difference"
    if any(word in q for word in ("percent", "percentage", "%", "ratio")):
        return "percent"
    if any(word in q for word in ("how many", "number of", "count")):
        return "count"
    if any(word in q for word in ("highest", "lowest", "maximum", "minimum", "largest", "smallest")):
        return "extreme"
    if q.startswith("is ") or q.startswith("are ") or q.startswith("does ") or q.startswith("do "):
        return "yes_no"
    return "other"


def _load_candidates(pattern: str) -> list[dict[str, Any]]:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_iter_jsonl(path))
    return rows


def _sample_candidates(rows: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if max_samples <= 0:
        return []
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {"all_wrong": [], "mixed_wrong": [], "other": []}
    for row in rows:
        buckets.setdefault(_candidate_scope(row), []).append(row)

    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)

    half = max_samples // 2
    selected = buckets["all_wrong"][:half] + buckets["mixed_wrong"][: max_samples - half]
    shortfall = max_samples - len(selected)
    if shortfall > 0:
        selected_ids = {id(row) for row in selected}
        remainder = [
            row
            for scope in ("all_wrong", "mixed_wrong", "other")
            for row in buckets.get(scope, [])
            if id(row) not in selected_ids
        ]
        selected.extend(remainder[:shortfall])
    return selected[:max_samples]


def _enrich_candidate(candidate: dict[str, Any], dataset_record: dict[str, Any]) -> dict[str, Any]:
    sample = dict(dataset_record)
    question = _record_question(dataset_record) or str(candidate.get("question") or "").strip()
    sample["question"] = question
    sample["prompt"] = _record_prompt(dataset_record) or question
    sample["image"] = candidate.get("image") or dataset_record.get("image")
    sample["answer"] = _clean_answer(dataset_record.get("answer") or candidate.get("reference") or candidate.get("answer"))
    sample["_candidate"] = candidate
    sample["_scope"] = _candidate_scope(candidate)
    sample["_answer_flag"] = str(candidate.get("answer_flag") or "answer:").strip() or "answer:"
    return sample


def _build_prompt(
    sample: dict[str, Any],
    control: dict[str, Any],
    *,
    load_images: bool,
) -> tuple[str, list[Any], str, str]:
    opsd_config = {
        "text_include_gold": False,
        "privileged_profile": control.get("privileged_profile", "text"),
        "teacher_probe": {"prompt_profile": control["prompt_profile"]},
    }
    if control.get("privileged_image"):
        opsd_config["privileged_image"] = control["privileged_image"]
    context_sample = sample if load_images else {**sample, "image": None}
    suffix, teacher_images = build_privileged_context(
        context_sample,
        control["providers"],
        privileged_profile=control.get("privileged_profile", "text"),
        opsd_config=opsd_config,
    )
    suffix, response_prefix = split_teacher_response_prefix(suffix)
    if load_images and not teacher_images:
        full = load_rgb(sample.get("image"))
        teacher_images = [full] if full is not None else []
    prompt = _build_teacher_text(sample["prompt"], suffix)
    return prompt, teacher_images, suffix, response_prefix


def _fake_teacher_output(control_name: str, sample: dict[str, Any]) -> str:
    if control_name.startswith("oracle_hint"):
        answer = _clean_answer(sample.get("answer"))
        hint = str(sample.get("hint") or "").strip()
        observation = hint.splitlines()[1] if len(hint.splitlines()) > 1 else f"Reference answer is {answer}."
        return (
            "Goal: Follow the verified training hint.\n"
            f"{observation}\n"
            "Reasoning: The oracle hint and reference answer are authoritative.\n"
            f"Conclusion: {answer}.\n"
            f"Answer: {answer}"
        )
    if control_name == "reasoned_deplot_only":
        return (
            "Goal: Solve the chart question.\n"
            "Observation: Use only the chart and DePlot evidence.\n"
            "Reasoning: Inspect the relevant labels and values.\n"
            "Conclusion: The evidence gives the result.\n"
            "Answer: __reasoned_wrong__"
        )
    if control_name == "deplot_operation_answer_prefix":
        return f"Answer: {_clean_answer(sample.get('answer'))}"
    return "Answer: __baseline_wrong__"


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _has_full_hint_format(text: str) -> bool:
    lower = str(text or "").lower()
    pos = -1
    for heading in REQUIRED_HINT_HEADINGS:
        next_pos = lower.find(heading, pos + 1)
        if next_pos < 0:
            return False
        pos = next_pos
    return True


def _answer_last_line(text: str) -> bool:
    lines = _nonempty_lines(text)
    return bool(lines and lines[-1].lower().startswith("answer:"))


def _exact_reference_answer_line(text: str, reference: Any) -> bool:
    expected = f"Answer: {_clean_answer(reference)}".strip()
    if not expected or expected == "Answer:":
        return False
    return any(line == expected for line in _nonempty_lines(text))


def _load_teacher(args: argparse.Namespace):
    if args.teacher_backend == "qwen25vl":
        return _load_qwen25vl_teacher(args)

    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    from data_utils.paths import local_pretrained_kwargs, resolve_model_path, validate_local_model_dir

    model_path = validate_local_model_dir(resolve_model_path(args.teacher_model), role="teacher")
    local_kw = local_pretrained_kwargs(model_path)
    processor = AutoProcessor.from_pretrained(model_path, **local_kw)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        **local_kw,
    }
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation=args.attn_implementation,
        **load_kwargs,
    )
    if not args.device_map and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    model.requires_grad_(False)
    return model, processor


def _load_qwen25vl_teacher(args: argparse.Namespace):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_path = args.teacher_model
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation=args.attn_implementation,
        **load_kwargs,
    )
    if not args.device_map and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    model.requires_grad_(False)
    return model, processor


def _iter_progress_chunks(items: list[Any], *, chunk_size: int, desc: str):
    chunk_size = max(1, int(chunk_size))
    total = math.ceil(len(items) / chunk_size) if items else 0
    starts = range(0, len(items), chunk_size)
    for start in tqdm(starts, total=total, desc=desc, unit="batch"):
        yield items[start : start + chunk_size]


def _run_real_teacher(
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
) -> list[str]:
    teacher_model, processor = _load_teacher(args)
    return _run_real_teacher_loaded(args, jobs, teacher_model, processor)


def _run_real_teacher_loaded(
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
    teacher_model: Any,
    processor: Any,
    *,
    desc: str = "teacher micro-eval",
) -> list[str]:
    if getattr(args, "teacher_backend", "llava_onevision") == "qwen25vl":
        return _run_qwen25vl_teacher_loaded(args, jobs, teacher_model, processor, desc=desc)

    from reward_utils.teacher_generate import TeacherGenerateRequest, teacher_generate_batch

    requests = [
        TeacherGenerateRequest(
            prompt_text=job["prompt"],
            images=job["images"],
            response_prefix=job.get("response_prefix", ""),
            max_new_tokens=int(job.get("max_new_tokens") or args.max_new_tokens),
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        for job in jobs
    ]
    texts: list[str] = []
    for chunk in _iter_progress_chunks(
        requests,
        chunk_size=args.batch_size,
        desc=f"{desc} bs={max(1, int(args.batch_size))}",
    ):
        chunk_texts, _ = teacher_generate_batch(
            teacher_model,
            processor,
            chunk,
            timing_kind="teacher_probe_micro_eval",
        )
        texts.extend(chunk_texts)
    return texts


def _run_qwen25vl_teacher_loaded(
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
    teacher_model: Any,
    processor: Any,
    *,
    desc: str = "qwen25vl teacher micro-eval",
) -> list[str]:
    import torch
    from qwen_vl_utils import process_vision_info

    device = next(teacher_model.parameters()).device
    texts: list[str] = []
    for chunk in _iter_progress_chunks(
        jobs,
        chunk_size=args.batch_size,
        desc=f"{desc} bs={max(1, int(args.batch_size))}",
    ):
        rendered_texts: list[str] = []
        image_inputs_all: list[Any] = []
        for job in chunk:
            content = []
            for image in job.get("images") or []:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": job["prompt"]})
            messages = [{"role": "user", "content": content}]
            rendered = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix = str(job.get("response_prefix") or "")
            if prefix:
                rendered = f"{rendered}{prefix}"
            image_inputs, _ = process_vision_info(messages)
            rendered_texts.append(rendered)
            image_inputs_all.extend(image_inputs or [])

        inputs = processor(
            text=rendered_texts,
            images=image_inputs_all or None,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            generated = teacher_model.generate(
                **inputs,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else None,
                top_p=args.top_p if args.do_sample else None,
                repetition_penalty=args.repetition_penalty,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for job, text in zip(chunk, decoded):
            prefix = str(job.get("response_prefix") or "")
            texts.append(f"{prefix}{text}" if prefix else text)
    return texts


def _build_canonicalization_jobs(
    jobs: list[dict[str, Any]],
    drafts: list[str],
    controls: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    canonical_jobs: list[dict[str, Any]] = []
    canonical_indices: list[int] = []
    for idx, (job, draft) in enumerate(zip(jobs, drafts)):
        control = controls.get(job["control"]) or {}
        if not control.get("canonicalize_draft"):
            continue
        canonical_prompt = (
            f"{job['prompt']}\n\n"
            "[Teacher Draft Reasoning]\n"
            f"{str(draft).strip()}\n\n"
            "Extract the single final answer supported by the chart evidence and the draft. "
            "Return only the short answer after the provided Answer: prefix. Do not add "
            "reasoning, labels, alternatives, or any other text."
        )
        canonical_jobs.append(
            {
                **job,
                "prompt": canonical_prompt,
                "response_prefix": "Answer:",
                "max_new_tokens": 32,
            }
        )
        canonical_indices.append(idx)
    return canonical_jobs, canonical_indices


def _merge_canonicalized_outputs(
    drafts: list[str],
    canonical_outputs: list[str],
    canonical_indices: list[int],
) -> list[str]:
    merged = list(drafts)
    for idx, canonical in zip(canonical_indices, canonical_outputs):
        merged[idx] = f"{str(drafts[idx]).rstrip()}\n{str(canonical).strip()}".strip()
    return merged


def _metric_row(records: list[dict[str, Any]], keys: dict[str, str]) -> dict[str, str]:
    n = len(records)
    correct = sum(1 for row in records if row.get("teacher_correct") is True)
    parse_fail = sum(1 for row in records if row.get("parse_failed") is True)
    answer_flag = sum(1 for row in records if row.get("has_answer_flag") is True)
    full_hint_format = sum(1 for row in records if row.get("full_hint_format") is True)
    answer_last_line = sum(1 for row in records if row.get("answer_last_line") is True)
    exact_reference_answer_line = sum(
        1 for row in records if row.get("exact_reference_answer_line") is True
    )
    token_values = [float(row.get("generated_tokens") or 0) for row in records]
    token_mean = sum(token_values) / n if n else 0.0
    row = dict(keys)
    row.update(
        {
            "n": str(n),
            "teacher_correct_rate": f"{(correct / n) if n else 0.0:.4f}",
            "parse_fail_rate": f"{(parse_fail / n) if n else 0.0:.4f}",
            "answer_flag_rate": f"{(answer_flag / n) if n else 0.0:.4f}",
            "full_hint_format_rate": f"{(full_hint_format / n) if n else 0.0:.4f}",
            "answer_last_line_rate": f"{(answer_last_line / n) if n else 0.0:.4f}",
            "exact_reference_answer_line_rate": f"{(exact_reference_answer_line / n) if n else 0.0:.4f}",
            "generated_tokens_mean": f"{token_mean:.2f}",
            "status": "ok" if n else "empty",
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_summaries(
    out_dir: Path,
    records: list[dict[str, Any]],
    controls: dict[str, dict[str, Any]],
) -> None:
    summary_rows: list[dict[str, str]] = []
    for control in controls:
        subset = [row for row in records if row["control"] == control]
        summary_rows.append(_metric_row(subset, {"control": control}))
    _write_csv(
        out_dir / "summary.csv",
        summary_rows,
        [
            "control",
            "n",
            "teacher_correct_rate",
            "parse_fail_rate",
            "answer_flag_rate",
            "full_hint_format_rate",
            "answer_last_line_rate",
            "exact_reference_answer_line_rate",
            "generated_tokens_mean",
            "status",
        ],
    )

    by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_qtype: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_scope[(row["control"], row["scope"])].append(row)
        by_qtype[(row["control"], row["qtype"])].append(row)

    scope_rows = [
        _metric_row(group, {"control": control, "scope": scope})
        for (control, scope), group in sorted(by_scope.items())
    ]
    _write_csv(
        out_dir / "by_scope.csv",
        scope_rows,
        [
            "control",
            "scope",
            "n",
            "teacher_correct_rate",
            "parse_fail_rate",
            "answer_flag_rate",
            "full_hint_format_rate",
            "answer_last_line_rate",
            "exact_reference_answer_line_rate",
            "generated_tokens_mean",
            "status",
        ],
    )

    qtype_rows = [
        _metric_row(group, {"control": control, "qtype": qtype})
        for (control, qtype), group in sorted(by_qtype.items())
    ]
    _write_csv(
        out_dir / "by_qtype.csv",
        qtype_rows,
        [
            "control",
            "qtype",
            "n",
            "teacher_correct_rate",
            "parse_fail_rate",
            "answer_flag_rate",
            "full_hint_format_rate",
            "answer_last_line_rate",
            "exact_reference_answer_line_rate",
            "generated_tokens_mean",
            "status",
        ],
    )


def _select_verifier_first_correct(
    records: list[dict[str, Any]],
    control_order: list[str],
) -> list[dict[str, Any]]:
    by_sample: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        by_sample[int(row["sample_idx"])][str(row["control"])] = row

    selected_rows: list[dict[str, Any]] = []
    for sample_idx in sorted(by_sample):
        sample_rows = by_sample[sample_idx]
        ordered = [sample_rows[name] for name in control_order if name in sample_rows]
        if not ordered:
            continue

        selected = ordered[0]
        status = "abstained"
        attempt_count = len(ordered)
        for position, row in enumerate(ordered, start=1):
            if row.get("teacher_correct") is True:
                selected = row
                status = "accepted"
                attempt_count = position
                break

        attempts_used = ordered[:attempt_count]
        selected_rows.append(
            {
                "sample_idx": sample_idx,
                "scope": selected.get("scope", ""),
                "qtype": selected.get("qtype", ""),
                "question": selected.get("question", ""),
                "image_basename": selected.get("image_basename", ""),
                "reference": selected.get("reference", ""),
                "status": status,
                "selected_control": selected.get("control", ""),
                "selected_output": selected.get("teacher_output", ""),
                "parsed_answer": selected.get("parsed_answer", ""),
                "teacher_correct": bool(selected.get("teacher_correct") is True),
                "attempt_count": attempt_count,
                "controls_attempted": [row.get("control", "") for row in attempts_used],
                "oracle_any_attempt_correct": any(
                    row.get("teacher_correct") is True for row in ordered
                ),
                "attempts": [
                    {
                        "control": row.get("control", ""),
                        "teacher_output": row.get("teacher_output", ""),
                        "parsed_answer": row.get("parsed_answer", ""),
                        "teacher_correct": bool(row.get("teacher_correct") is True),
                        "parse_failed": bool(row.get("parse_failed") is True),
                    }
                    for row in ordered
                ],
            }
        )
    return selected_rows


def _write_selected_summary(path: Path, selected_rows: list[dict[str, Any]]) -> None:
    n = len(selected_rows)
    accepted = [row for row in selected_rows if row["status"] == "accepted"]
    attempts = [float(row.get("attempt_count") or 0) for row in selected_rows]
    by_control = Counter(row["selected_control"] for row in accepted)
    summary = {
        "n": str(n),
        "selected_coverage_rate": f"{(len(accepted) / n) if n else 0.0:.4f}",
        "selected_precision": f"{(sum(row['teacher_correct'] for row in accepted) / len(accepted)) if accepted else 0.0:.4f}",
        "abstain_rate": f"{(sum(row['status'] != 'accepted' for row in selected_rows) / n) if n else 0.0:.4f}",
        "oracle_union_accuracy": f"{(sum(row['oracle_any_attempt_correct'] for row in selected_rows) / n) if n else 0.0:.4f}",
        "mean_attempts": f"{(sum(attempts) / n) if n else 0.0:.2f}",
        "accepted_by_control": json.dumps(dict(sorted(by_control.items())), sort_keys=True),
        "status": "ok" if n else "empty",
    }
    _write_csv(path, [summary], list(summary.keys()))


def _prepare_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    dataset_records = _json_load(Path(args.dataset))
    if isinstance(dataset_records, dict):
        dataset_records = dataset_records.get("data") or dataset_records.get("records") or []
    index = DatasetIndex(list(dataset_records))
    candidates = _sample_candidates(_load_candidates(args.candidate_glob), args.max_samples, args.seed)

    samples: list[dict[str, Any]] = []
    missing = 0
    for candidate in candidates:
        record = index.lookup(candidate)
        if record is None:
            missing += 1
            continue
        samples.append(_enrich_candidate(candidate, record))
    return samples, {
        "candidate_rows": len(candidates),
        "matched_samples": len(samples),
        "missing_dataset_matches": missing,
    }


def _build_jobs(
    samples: list[dict[str, Any]],
    controls: dict[str, dict[str, Any]],
    *,
    load_images: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        for control_name, control in controls.items():
            prompt, images, suffix, response_prefix = _build_prompt(sample, control, load_images=load_images)
            job = {
                "control": control_name,
                "sample_idx": sample_idx,
                "sample": sample,
                "prompt": prompt,
                "images": images,
                "suffix": suffix,
                "response_prefix": response_prefix,
            }
            jobs.append(job)
            previews.append(
                {
                    "control": control_name,
                    "sample_idx": sample_idx,
                    "scope": sample["_scope"],
                    "question": sample["question"],
                    "image_basename": _basename(sample.get("image")),
                    "reference": sample["answer"],
                    "prompt": prompt,
                    "response_prefix": response_prefix,
                }
            )
    return jobs, previews


def _harness_images(sample: dict[str, Any], *, load_images: bool) -> list[Any]:
    if not load_images:
        return []
    full = load_rgb(sample.get("image"))
    return [full] if full is not None else []


def _build_chartqa_harness_initial_jobs(
    samples: list[dict[str, Any]],
    *,
    load_images: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        configurations = (
            ("visual_base", EvidenceAction.VISUAL_BASE, build_visual_base_suffix()),
            (
                "visual_deplot",
                EvidenceAction.ATTACH_DEPLOT,
                build_visual_deplot_suffix(sample.get("visual_fact_deplot")),
            ),
        )
        for configuration, action, suffix in configurations:
            prompt = _build_teacher_text(sample["question"], suffix)
            images = _harness_images(sample, load_images=load_images)
            job = {
                "configuration": configuration,
                "action": action,
                "sample_idx": sample_idx,
                "sample": sample,
                "prompt": prompt,
                "images": images,
                "response_prefix": "",
            }
            jobs.append(job)
            previews.append(
                {
                    "configuration": configuration,
                    "sample_idx": sample_idx,
                    "scope": sample.get("_scope", "other"),
                    "question": sample["question"],
                    "image_basename": _basename(sample.get("image")),
                    "native_image_available": sample.get("image") is not None,
                    "loaded_image_count": len(images),
                    "prompt": prompt,
                }
            )
    return jobs, previews


def _build_chartqa_harness_recovery_job(
    *,
    sample_idx: int,
    sample: dict[str, Any],
    base_output: str,
    deplot_output: str,
    load_images: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    suffix = build_visual_recovery_suffix(
        deplot_value=sample.get("visual_fact_deplot"),
        base_output=base_output,
        deplot_output=deplot_output,
    )
    prompt = _build_teacher_text(sample["question"], suffix)
    images = _harness_images(sample, load_images=load_images)
    job = {
        "configuration": "visual_recovery",
        "action": EvidenceAction.VISUAL_RECOVERY,
        "sample_idx": sample_idx,
        "sample": sample,
        "prompt": prompt,
        "images": images,
        "response_prefix": "",
    }
    preview = {
        "configuration": "visual_recovery",
        "sample_idx": sample_idx,
        "scope": sample.get("_scope", "other"),
        "question": sample["question"],
        "image_basename": _basename(sample.get("image")),
        "native_image_available": sample.get("image") is not None,
        "loaded_image_count": len(images),
        "prompt": prompt,
    }
    return job, preview


def _fake_chartqa_harness_output(job: dict[str, Any]) -> str:
    answer = _clean_answer(job["sample"].get("answer"))
    configuration = job["configuration"]
    if configuration == "visual_deplot" and int(job["sample_idx"]) % 2 == 0:
        return "The auxiliary table suggests a different value.\nAnswer: __deplot_conflict__"
    if configuration == "visual_recovery":
        return f"I rechecked the full chart image and resolved the conflict.\nAnswer: {answer}"
    return f"I inspected the full chart image.\nAnswer: {answer}"


def _score_harness_output(output: str, sample: dict[str, Any]) -> tuple[bool, str]:
    score, parsed = eval_teacher_probe_chart(
        output,
        sample["answer"],
        0.05,
        answer_flag=str(sample.get("_answer_flag") or "answer:").lower(),
    )
    return bool(score > 0.0), parsed.answer


def _write_chartqa_harness_summary(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    n = len(records)

    def rate(predicate) -> float:
        return sum(1 for row in records if predicate(row)) / n if n else 0.0

    accepted = [row for row in records if row["status"] == HarnessStatus.ACCEPTED.value]
    recovered = [row for row in records if row["recovery_triggered"]]
    agreements = [row for row in records if row["initial_reason"] == "cross_attempt_agreement"]
    attempts = [float(row["attempt_count"]) for row in records]
    summary = {
        "n": str(n),
        "base_accuracy": f"{rate(lambda row: row['base_correct']):.4f}",
        "deplot_accuracy": f"{rate(lambda row: row['deplot_correct']):.4f}",
        "selected_accuracy": f"{rate(lambda row: row['teacher_correct']):.4f}",
        "oracle_union_accuracy": f"{rate(lambda row: row['oracle_any_attempt_correct']):.4f}",
        "agreement_rate": f"{(len(agreements) / n) if n else 0.0:.4f}",
        "agreement_accuracy": f"{(sum(row['teacher_correct'] for row in agreements) / len(agreements)) if agreements else 0.0:.4f}",
        "recovery_trigger_rate": f"{(len(recovered) / n) if n else 0.0:.4f}",
        "recovered_accuracy": f"{(sum(row['teacher_correct'] for row in recovered) / len(recovered)) if recovered else 0.0:.4f}",
        "accepted_precision": f"{(sum(row['teacher_correct'] for row in accepted) / len(accepted)) if accepted else 0.0:.4f}",
        "false_accept_rate": f"{(sum(not row['teacher_correct'] for row in accepted) / len(accepted)) if accepted else 0.0:.4f}",
        "abstain_rate": f"{rate(lambda row: row['status'] != HarnessStatus.ACCEPTED.value):.4f}",
        "mean_attempts": f"{(sum(attempts) / n) if n else 0.0:.2f}",
        "status": "ok" if n else "empty",
    }
    _write_csv(path, [summary], list(summary.keys()))


_CLOSED_LOOP_OPERATION_QTYPES = {"average", "difference", "percent", "count"}
_CLOSED_LOOP_ACTIONS = [
    "visual_answer",
    "visual_operation_recovery",
    "deplot_operation_recovery",
    "executable_deplot_recovery",
    "reasoned_recovery",
    "target_phrase_recovery",
    "arithmetic_recovery",
    "scale_unit_recovery",
]
_INTEGRATED_CLOSED_LOOP_CONTROLLER = "integrated_closed_loop_recovery_controller"


def _closed_loop_verifier_event(output: str, sample: dict[str, Any]) -> dict[str, Any]:
    score, parsed = eval_teacher_probe_chart(
        output,
        sample["answer"],
        0.05,
        answer_flag=str(sample.get("_answer_flag") or "answer:").lower(),
    )
    teacher_correct = bool(score > 0.0)
    qtype = _qtype(sample["question"])
    if teacher_correct:
        event = "accepted"
    elif parsed.parse_failed:
        event = "canonical_repair_required"
    elif qtype in _CLOSED_LOOP_OPERATION_QTYPES:
        event = "operation_recovery_required"
    else:
        event = "evidence_recovery_required"
    return {
        "event": event,
        "qtype": qtype,
        "parsed_answer": parsed.answer,
        "teacher_correct": teacher_correct,
        "score": float(score),
        "parse_failed": bool(parsed.parse_failed),
        "has_answer_flag": bool(parsed.has_answer_flag),
    }


def _closed_loop_next_action(
    event: str,
    qtype: str,
    attempted_actions: list[str],
) -> str | None:
    if event == "accepted":
        return None
    if event == "canonical_repair_required" and "reasoned_recovery" not in attempted_actions:
        return "reasoned_recovery"
    needs_recovery = (
        event in {"operation_recovery_required", "evidence_recovery_required"}
        or qtype in _CLOSED_LOOP_OPERATION_QTYPES
    )
    if needs_recovery and "visual_operation_recovery" not in attempted_actions:
        return "visual_operation_recovery"
    if needs_recovery and "deplot_operation_recovery" not in attempted_actions:
        return "deplot_operation_recovery"
    if needs_recovery and "executable_deplot_recovery" not in attempted_actions:
        return "executable_deplot_recovery"
    if "reasoned_recovery" not in attempted_actions:
        return "reasoned_recovery"
    if needs_recovery and "target_phrase_recovery" not in attempted_actions:
        return "target_phrase_recovery"
    if needs_recovery and "arithmetic_recovery" not in attempted_actions:
        return "arithmetic_recovery"
    if needs_recovery and "scale_unit_recovery" not in attempted_actions:
        return "scale_unit_recovery"
    return None


def _closed_loop_previous_answer(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return ""
    return str(attempts[-1].get("teacher_output") or "").strip()


class ClosedLoopRecoveryController:
    """Execute one sample's verifier-observe-recover loop.

    The runner batches the jobs emitted by this object, but the recovery
    process itself owns its evidence state, verifier events, next action, and
    final record. This keeps recovery as one executable trajectory whose later
    prompts are conditioned on earlier verifier failures.
    """

    controller_name = _INTEGRATED_CLOSED_LOOP_CONTROLLER

    def __init__(
        self,
        *,
        sample_idx: int,
        sample: dict[str, Any],
        load_images: bool,
    ) -> None:
        self.sample_idx = int(sample_idx)
        self.sample = sample
        self.load_images = bool(load_images)
        self.attempts: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.actions: list[str] = []
        self.status = "active"
        self.next_action: str | None = "visual_answer"
        self.next_event = "initial"
        self.selected_attempt: dict[str, Any] | None = None
        self._pending_job: dict[str, Any] | None = None

    def build_next_job(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if self.status != "active":
            return None
        if not self.next_action:
            self.status = "abstained"
            return None

        job, preview = _build_closed_loop_job(
            sample_idx=self.sample_idx,
            sample=self.sample,
            action=self.next_action,
            attempts=self.attempts,
            event=self.next_event,
            load_images=self.load_images,
        )
        step = len(self.attempts) + 1
        job["controller"] = self.controller_name
        job["controller_step"] = step
        preview["controller"] = self.controller_name
        preview["controller_step"] = step
        self._pending_job = job
        return job, preview

    def observe(self, output: str) -> dict[str, Any]:
        if self._pending_job is None:
            raise RuntimeError("observe() requires a job from build_next_job()")

        job = self._pending_job
        event = _closed_loop_verifier_event(output, self.sample)
        action = str(job["action"])
        next_action: str | None = None
        if not event["teacher_correct"]:
            next_action = _closed_loop_next_action(
                str(event["event"]),
                str(event["qtype"]),
                list(self.actions) + [action],
            )

        attempt = {
            "sample_idx": self.sample_idx,
            "controller": self.controller_name,
            "controller_step": len(self.attempts) + 1,
            "action": action,
            "event": event["event"],
            "prompt_event": job.get("event", ""),
            "teacher_output": output,
            "parsed_answer": event["parsed_answer"],
            "teacher_correct": event["teacher_correct"],
            "score": event["score"],
            "parse_failed": event["parse_failed"],
            "has_answer_flag": event["has_answer_flag"],
            "next_action": next_action or "",
        }
        self.attempts.append(attempt)
        self.events.append(event["event"])
        self.actions.append(action)
        self._pending_job = None

        if event["teacher_correct"]:
            self.status = "accepted"
            self.selected_attempt = attempt
            self.next_action = None
        elif next_action is None:
            self.status = "abstained"
            self.selected_attempt = attempt
            self.next_action = None
        else:
            self.next_action = next_action
            self.next_event = event["event"]
        return event

    def finish(self) -> None:
        if self.status != "active":
            return
        self.status = "abstained"
        if self.attempts and self.selected_attempt is None:
            self.selected_attempt = self.attempts[-1]
        self.next_action = None

    def attempt_rows(self) -> list[dict[str, Any]]:
        return [dict(attempt) for attempt in self.attempts]

    def to_record(self) -> dict[str, Any]:
        self.finish()
        selected = self.selected_attempt or {}
        return {
            "sample_idx": self.sample_idx,
            "controller": self.controller_name,
            "scope": self.sample.get("_scope", "other"),
            "qtype": _qtype(self.sample["question"]),
            "question": self.sample["question"],
            "image_basename": _basename(self.sample.get("image")),
            "reference": self.sample["answer"],
            "status": self.status,
            "attempt_count": len(self.attempts),
            "events": list(self.events),
            "actions": list(self.actions),
            "selected_action": selected.get("action", ""),
            "selected_output": selected.get("teacher_output", ""),
            "parsed_answer": selected.get("parsed_answer", ""),
            "teacher_correct": bool(selected.get("teacher_correct") is True),
            "oracle_any_attempt_correct": any(
                bool(attempt.get("teacher_correct")) for attempt in self.attempts
            ),
            "attempts": self.attempt_rows(),
        }


def _build_closed_loop_job(
    *,
    sample_idx: int,
    sample: dict[str, Any],
    action: str,
    attempts: list[dict[str, Any]],
    event: str,
    load_images: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    previous_answer = _closed_loop_previous_answer(attempts)
    max_new_tokens: int | None = None
    if action == "visual_answer":
        control_name = "visual_answer_prefix"
        prompt, images, _suffix, response_prefix = _build_prompt(
            sample,
            CONTROL_SPECS[control_name],
            load_images=load_images,
        )
        max_new_tokens = 32
    elif action in {"visual_operation_recovery", "deplot_operation_recovery", "reasoned_recovery"}:
        control_name = {
            "visual_operation_recovery": "visual_operation_answer_prefix",
            "deplot_operation_recovery": "deplot_operation_answer_prefix",
            "reasoned_recovery": "reasoned_deplot_only",
        }[action]
        prompt, images, _suffix, response_prefix = _build_prompt(
            sample,
            CONTROL_SPECS[control_name],
            load_images=load_images,
        )
        max_new_tokens = 160 if action == "reasoned_recovery" else 96
        if action == "reasoned_recovery":
            prompt += (
                "\n\n[Closed-Loop Recovery State]\n"
                f"Verifier event: {str(event)}\n"
                f"Previous teacher answer:\n{previous_answer}\n\n"
                "The previous answer failed the verifier. Repair the specific evidence "
                "obligation without using any reference answer or hidden hint."
            )
    elif action == "arithmetic_recovery":
        control_name = "arithmetic_recovery"
        suffix = build_chartqa_arithmetic_recovery_suffix(sample.get("visual_fact_deplot"))
        prompt = _build_teacher_text(sample["question"], suffix)
        images = _harness_images(sample, load_images=load_images)
        response_prefix = ""
        max_new_tokens = 160
    elif action == "target_phrase_recovery":
        control_name = "target_phrase_recovery"
        suffix = build_chartqa_target_phrase_recovery_suffix(
            sample["question"],
            sample.get("visual_fact_deplot"),
        )
        prompt = _build_teacher_text(sample["question"], suffix)
        images = _harness_images(sample, load_images=load_images)
        response_prefix = build_chartqa_target_phrase_response_prefix(
            sample["question"],
            sample.get("visual_fact_deplot"),
        )
        max_new_tokens = 96
    elif action == "executable_deplot_recovery":
        control_name = "executable_deplot_recovery"
        suffix = build_chartqa_executable_deplot_recovery_suffix(
            sample["question"],
            sample.get("visual_fact_deplot"),
        )
        prompt = _build_teacher_text(sample["question"], suffix)
        images = []
        response_prefix = build_chartqa_executable_deplot_response_prefix(
            sample["question"],
            sample.get("visual_fact_deplot"),
        )
        max_new_tokens = 1 if response_prefix != "Answer:" else 48
    elif action == "scale_unit_recovery":
        control_name = "scale_unit_recovery"
        suffix = build_chartqa_scale_unit_recovery_suffix(
            deplot_value=sample.get("visual_fact_deplot"),
            attempts=attempts,
        )
        prompt = _build_teacher_text(sample["question"], suffix)
        images = _harness_images(sample, load_images=load_images)
        response_prefix = "Answer:"
        max_new_tokens = 48
    else:
        raise ValueError(f"Unknown closed-loop action: {action}")

    job = {
        "control": control_name,
        "action": action,
        "sample_idx": sample_idx,
        "sample": sample,
        "prompt": prompt,
        "images": images,
        "response_prefix": response_prefix,
        "event": event,
    }
    if max_new_tokens is not None:
        job["max_new_tokens"] = max_new_tokens
    preview = {
        "action": action,
        "sample_idx": sample_idx,
        "scope": sample.get("_scope", "other"),
        "qtype": _qtype(sample["question"]),
        "question": sample["question"],
        "image_basename": _basename(sample.get("image")),
        "native_image_available": sample.get("image") is not None,
        "loaded_image_count": len(images),
        "event": event,
        "prompt": prompt,
        "response_prefix": response_prefix,
    }
    return job, preview


def _fake_closed_loop_output(job: dict[str, Any]) -> str:
    answer = _clean_answer(job["sample"].get("answer"))
    action = str(job["action"])
    if action == "visual_answer" and int(job["sample_idx"]) % 2 == 1:
        return "Answer: __visual_wrong__"
    if action == "visual_operation_recovery":
        return "Answer: __visual_operation_wrong__"
    if action == "deplot_operation_recovery":
        return f"Answer: {answer}"
    if action == "reasoned_recovery":
        return f"I repaired the failed evidence step.\nAnswer: {answer}"
    if action == "target_phrase_recovery":
        return f"Target phrase: requested chart item\nEvidence: matched label\nAnswer: {answer}"
    if action == "executable_deplot_recovery":
        return f"Answer: {answer}"
    if action == "arithmetic_recovery":
        return f"Operands: evidence values\nOperation: repair\nEquation: none\nAnswer: {answer}"
    if action == "scale_unit_recovery":
        return f"Answer: {answer}"
    return f"Answer: {answer}"


def _write_closed_loop_summary(path: Path, records: list[dict[str, Any]]) -> None:
    n = len(records)
    accepted = [row for row in records if row["status"] == "accepted"]
    attempts = [float(row.get("attempt_count") or 0) for row in records]
    recovered = [row for row in accepted if int(row.get("attempt_count") or 0) > 1]
    recovery_triggered = [row for row in records if int(row.get("attempt_count") or 0) > 1]
    event_counts = Counter(event for row in records for event in row.get("events", []))
    action_counts = Counter(action for row in records for action in row.get("actions", []))
    accepted_by_action = Counter(row["selected_action"] for row in accepted)
    parse_fail_events = sum(1 for row in records if "canonical_repair_required" in row.get("events", []))
    summary = {
        "n": str(n),
        "selected_accuracy": f"{(sum(row['teacher_correct'] for row in records) / n) if n else 0.0:.4f}",
        "accepted_coverage_rate": f"{(len(accepted) / n) if n else 0.0:.4f}",
        "abstain_rate": f"{(sum(row['status'] != 'accepted' for row in records) / n) if n else 0.0:.4f}",
        "mean_attempts": f"{(sum(attempts) / n) if n else 0.0:.2f}",
        "recovery_success_rate": f"{(len(recovered) / len(recovery_triggered)) if recovery_triggered else 0.0:.4f}",
        "parse_fail_event_rate": f"{(parse_fail_events / n) if n else 0.0:.4f}",
        "oracle_any_attempt_accuracy": f"{(sum(row['oracle_any_attempt_correct'] for row in records) / n) if n else 0.0:.4f}",
        "accepted_by_action": json.dumps(dict(sorted(accepted_by_action.items())), sort_keys=True),
        "event_counts": json.dumps(dict(sorted(event_counts.items())), sort_keys=True),
        "action_counts": json.dumps(dict(sorted(action_counts.items())), sort_keys=True),
        "status": "ok" if n else "empty",
    }
    _write_csv(path, [summary], list(summary.keys()))


def _closed_loop_generation_controls() -> dict[str, dict[str, Any]]:
    controls = {
        name: CONTROL_SPECS[name]
        for name in (
            "visual_answer_prefix",
            "visual_operation_answer_prefix",
            "deplot_operation_answer_prefix",
            "reasoned_deplot_only",
        )
    }
    controls["arithmetic_recovery"] = {"canonicalize_draft": True}
    controls["target_phrase_recovery"] = {"canonicalize_draft": True}
    controls["executable_deplot_recovery"] = {}
    controls["scale_unit_recovery"] = {"canonicalize_draft": True}
    return controls


def _run_chartqa_closed_loop_recovery(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    out_dir: Path,
    manifest: dict[str, Any],
) -> int:
    load_images = not (args.dry_run or args.fake_teacher)
    manifest.update(
        {
            "harness": "chartqa_closed_loop_recovery",
            "controller": _INTEGRATED_CLOSED_LOOP_CONTROLLER,
            "controller_contract": "verifier_observe_act_recover",
            "native_input": ["question", "full_chart_image"],
            "max_teacher_attempts": len(_CLOSED_LOOP_ACTIONS),
            "actions": list(_CLOSED_LOOP_ACTIONS),
            "dry_run": args.dry_run,
            "fake_teacher": args.fake_teacher,
            "max_samples": args.max_samples,
            "seed": args.seed,
        }
    )

    controllers = [
        ClosedLoopRecoveryController(
            sample_idx=sample_idx,
            sample=sample,
            load_images=load_images,
        )
        for sample_idx, sample in enumerate(samples)
    ]
    prompt_previews: list[dict[str, Any]] = []
    if args.dry_run:
        for controller in controllers:
            built = controller.build_next_job()
            if built is not None:
                _job, preview = built
                prompt_previews.append(preview)
        _write_jsonl(out_dir / "prompt_previews.jsonl", prompt_previews)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 0

    teacher_model = processor = None
    if not args.fake_teacher:
        teacher_model, processor = _load_teacher(args)

    closed_loop_controls = _closed_loop_generation_controls()

    for _round_idx in range(len(_CLOSED_LOOP_ACTIONS)):
        jobs: list[dict[str, Any]] = []
        pending_controllers: list[ClosedLoopRecoveryController] = []
        for controller in controllers:
            built = controller.build_next_job()
            if built is None:
                continue
            job, preview = built
            jobs.append(job)
            pending_controllers.append(controller)
            prompt_previews.append(preview)
        if not jobs:
            break

        if args.fake_teacher:
            outputs = [_fake_closed_loop_output(job) for job in jobs]
        else:
            outputs = _generate_outputs_for_jobs(
                args,
                jobs,
                closed_loop_controls,
                teacher_model=teacher_model,
                processor=processor,
                desc=f"chartqa closed-loop round {_round_idx + 1}",
            )

        for controller, output in zip(pending_controllers, outputs):
            controller.observe(output)

    for controller in controllers:
        controller.finish()

    records = [controller.to_record() for controller in controllers]
    attempt_rows: list[dict[str, Any]] = []
    for controller in controllers:
        attempt_rows.extend(controller.attempt_rows())

    _write_jsonl(out_dir / "prompt_previews.jsonl", prompt_previews)
    _write_jsonl(out_dir / "closed_loop_attempts.jsonl", attempt_rows)
    _write_jsonl(out_dir / "closed_loop_records.jsonl", records)
    _write_closed_loop_summary(out_dir / "closed_loop_summary.csv", records)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


def _run_chartqa_recoverable_harness(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    out_dir: Path,
    manifest: dict[str, Any],
) -> int:
    load_images = not (args.dry_run or args.fake_teacher)
    initial_jobs, previews = _build_chartqa_harness_initial_jobs(
        samples,
        load_images=load_images,
    )
    manifest.update(
        {
            "harness": "chartqa_recoverable",
            "native_input": ["question", "full_chart_image"],
            "max_teacher_attempts": 3,
            "dry_run": args.dry_run,
            "fake_teacher": args.fake_teacher,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "selection_policy": args.selection_policy,
        }
    )

    if args.dry_run:
        _write_jsonl(out_dir / "prompt_previews.jsonl", previews)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 0

    teacher_model = processor = None
    if args.fake_teacher:
        initial_outputs = [_fake_chartqa_harness_output(job) for job in initial_jobs]
    else:
        teacher_model, processor = _load_teacher(args)
        initial_outputs = _run_real_teacher_loaded(
            args,
            initial_jobs,
            teacher_model,
            processor,
            desc="chartqa base+deplot",
        )

    outputs_by_sample: dict[int, dict[str, str]] = defaultdict(dict)
    for job, output in zip(initial_jobs, initial_outputs):
        outputs_by_sample[job["sample_idx"]][job["configuration"]] = output

    runtime: dict[int, dict[str, Any]] = {}
    recovery_jobs: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        base_output = outputs_by_sample[sample_idx]["visual_base"]
        deplot_output = outputs_by_sample[sample_idx]["visual_deplot"]
        base_candidate = build_chartqa_candidate(
            attempt_id="base",
            action=EvidenceAction.VISUAL_BASE,
            output=base_output,
        )
        deplot_candidate = build_chartqa_candidate(
            attempt_id="deplot",
            action=EvidenceAction.ATTACH_DEPLOT,
            output=deplot_output,
        )
        base_validation = validate_chartqa_candidate(
            base_candidate,
            sample.get("visual_fact_deplot"),
        )
        deplot_validation = validate_chartqa_candidate(
            deplot_candidate,
            sample.get("visual_fact_deplot"),
        )
        decision = decide_after_parallel_attempts(base_candidate, deplot_candidate)
        runtime[sample_idx] = {
            "candidates": {"base": base_candidate, "deplot": deplot_candidate},
            "validations": {"base": base_validation, "deplot": deplot_validation},
            "initial_decision": decision,
        }
        if decision.next_action is EvidenceAction.VISUAL_RECOVERY:
            recovery_job, recovery_preview = _build_chartqa_harness_recovery_job(
                sample_idx=sample_idx,
                sample=sample,
                base_output=base_output,
                deplot_output=deplot_output,
                load_images=load_images,
            )
            recovery_jobs.append(recovery_job)
            previews.append(recovery_preview)

    if args.fake_teacher:
        recovery_outputs = [_fake_chartqa_harness_output(job) for job in recovery_jobs]
    elif recovery_jobs:
        recovery_outputs = _run_real_teacher_loaded(
            args,
            recovery_jobs,
            teacher_model,
            processor,
            desc="chartqa conflict recovery",
        )
    else:
        recovery_outputs = []

    for job, output in zip(recovery_jobs, recovery_outputs):
        sample_idx = job["sample_idx"]
        sample = job["sample"]
        recovery_candidate = build_chartqa_candidate(
            attempt_id="recovery",
            action=EvidenceAction.VISUAL_RECOVERY,
            output=output,
        )
        recovery_validation = validate_chartqa_candidate(
            recovery_candidate,
            sample.get("visual_fact_deplot"),
        )
        runtime[sample_idx]["candidates"]["recovery"] = recovery_candidate
        runtime[sample_idx]["validations"]["recovery"] = recovery_validation
        runtime[sample_idx]["final_decision"] = decide_after_recovery(
            recovery_candidate,
            base=runtime[sample_idx]["candidates"]["base"],
            deplot=runtime[sample_idx]["candidates"]["deplot"],
            validation=recovery_validation,
        )

    records: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        state = runtime[sample_idx]
        initial_decision = state["initial_decision"]
        final_decision = state.get("final_decision") or initial_decision
        candidates = state["candidates"]
        validations = state["validations"]
        selected = candidates.get(final_decision.selected_attempt_id or "")
        selected_output = selected.raw_output if selected is not None else ""
        teacher_correct, parsed_answer = _score_harness_output(selected_output, sample)
        base_correct, base_parsed = _score_harness_output(candidates["base"].raw_output, sample)
        deplot_correct, deplot_parsed = _score_harness_output(candidates["deplot"].raw_output, sample)
        recovery_candidate = candidates.get("recovery")
        recovery_correct = False
        recovery_parsed = ""
        if recovery_candidate is not None:
            recovery_correct, recovery_parsed = _score_harness_output(
                recovery_candidate.raw_output,
                sample,
            )

        runtime_attempts = []
        for attempt_id in ("base", "deplot", "recovery"):
            candidate = candidates.get(attempt_id)
            if candidate is None:
                continue
            attempt_payload = {
                "candidate": candidate.to_dict(),
                "validation": validations[attempt_id].to_dict(),
            }
            runtime_attempts.append(attempt_payload)
            attempt_rows.append(
                {
                    "sample_idx": sample_idx,
                    "configuration": candidate.action.value,
                    **attempt_payload,
                }
            )

        record = {
            "sample_idx": sample_idx,
            "scope": sample.get("_scope", "other"),
            "qtype": _qtype(sample["question"]),
            "question": sample["question"],
            "image_basename": _basename(sample.get("image")),
            "reference": sample["answer"],
            "status": final_decision.status.value,
            "initial_reason": initial_decision.reason_code,
            "recovery_triggered": "recovery" in candidates,
            "attempt_count": len(candidates),
            "selected_attempt_id": final_decision.selected_attempt_id,
            "selected_output": selected_output,
            "parsed_answer": parsed_answer,
            "teacher_correct": teacher_correct,
            "base_parsed_answer": base_parsed,
            "base_correct": base_correct,
            "deplot_parsed_answer": deplot_parsed,
            "deplot_correct": deplot_correct,
            "recovery_parsed_answer": recovery_parsed,
            "recovery_correct": recovery_correct,
            "oracle_any_attempt_correct": base_correct or deplot_correct or recovery_correct,
            "attempts": runtime_attempts,
            "decision": final_decision.to_dict(),
        }
        records.append(record)

    _write_jsonl(out_dir / "prompt_previews.jsonl", previews)
    _write_jsonl(out_dir / "harness_attempts.jsonl", attempt_rows)
    _write_jsonl(out_dir / "harness_records.jsonl", records)
    _write_chartqa_harness_summary(out_dir / "harness_summary.csv", records)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


def _records_from_outputs(jobs: list[dict[str, Any]], outputs: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job, output in zip(jobs, outputs):
        sample = job["sample"]
        answer_flag = sample["_answer_flag"]
        score, parsed = eval_teacher_probe_chart(
            output,
            sample["answer"],
            0.05,
            answer_flag=answer_flag.lower(),
        )
        candidate = sample.get("_candidate") or {}
        records.append(
            {
                "control": job["control"],
                "sample_idx": job["sample_idx"],
                "scope": sample["_scope"],
                "qtype": _qtype(sample["question"]),
                "question": sample["question"],
                "image_basename": _basename(sample.get("image")),
                "reference": sample["answer"],
                "answer_flag": answer_flag,
                "teacher_output": output,
                "parsed_answer": parsed.answer,
                "teacher_correct": bool(score > 0.0),
                "score": score,
                "parse_failed": bool(parsed.parse_failed),
                "has_answer_flag": bool(parsed.has_answer_flag),
                "full_hint_format": _has_full_hint_format(output),
                "answer_last_line": _answer_last_line(output),
                "exact_reference_answer_line": _exact_reference_answer_line(output, sample["answer"]),
                "generated_tokens": len(output.split()),
                "source_idx": candidate.get("source_idx"),
                "global_step": candidate.get("global_step"),
                "route_reason": candidate.get("route_reason"),
                "provider_names": candidate.get("provider_names"),
            }
        )
    return records


def _generate_outputs_for_jobs(
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
    controls: dict[str, dict[str, Any]],
    *,
    teacher_model: Any | None = None,
    processor: Any | None = None,
    desc: str = "teacher draft",
) -> list[str]:
    if args.fake_teacher:
        return [_fake_teacher_output(job["control"], job["sample"]) for job in jobs]
    if teacher_model is None or processor is None:
        teacher_model, processor = _load_teacher(args)
    drafts = _run_real_teacher_loaded(
        args,
        jobs,
        teacher_model,
        processor,
        desc=desc,
    )
    canonical_jobs, canonical_indices = _build_canonicalization_jobs(jobs, drafts, controls)
    canonical_outputs = _run_real_teacher_loaded(
        args,
        canonical_jobs,
        teacher_model,
        processor,
        desc=f"{desc} answer canonicalization",
    ) if canonical_jobs else []
    return _merge_canonicalized_outputs(drafts, canonical_outputs, canonical_indices)


def _run_verifier_early_stop(
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
    controls: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if args.selection_policy != "verifier_first_correct":
        raise ValueError("verifier_early_stop requires --selection-policy verifier_first_correct")

    jobs_by_control: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    sample_indices = sorted({int(job["sample_idx"]) for job in jobs})
    for job in jobs:
        jobs_by_control[str(job["control"])][int(job["sample_idx"])] = job

    teacher_model = processor = None
    if not args.fake_teacher:
        teacher_model, processor = _load_teacher(args)

    unresolved = set(sample_indices)
    records: list[dict[str, Any]] = []
    for control_name in controls:
        control_jobs = [
            jobs_by_control[control_name][sample_idx]
            for sample_idx in sample_indices
            if sample_idx in unresolved and sample_idx in jobs_by_control[control_name]
        ]
        if not control_jobs:
            continue
        outputs = _generate_outputs_for_jobs(
            args,
            control_jobs,
            controls,
            teacher_model=teacher_model,
            processor=processor,
            desc=f"teacher {control_name}",
        )
        control_records = _records_from_outputs(control_jobs, outputs)
        records.extend(control_records)
        for row in control_records:
            if row.get("teacher_correct") is True:
                unresolved.discard(int(row["sample_idx"]))
        if not unresolved:
            break
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-glob", default=DEFAULT_CANDIDATE_GLOB)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fake-teacher", action="store_true")
    parser.add_argument(
        "--harness",
        choices=("chartqa_recoverable", "chartqa_closed_loop_recovery"),
        default=None,
        help="Run a bounded reference-free evidence-recovery harness instead of controls.",
    )
    parser.add_argument(
        "--controls",
        default=",".join(DEFAULT_CONTROL_NAMES),
        help=(
            "Comma-separated controls to run. Available: "
            + ",".join(CONTROL_SPECS.keys())
        ),
    )
    parser.add_argument(
        "--selection-policy",
        choices=("none", "verifier_first_correct"),
        default="none",
        help="Optional post-generation selector for multi-view teacher attempts.",
    )
    parser.add_argument(
        "--execution-policy",
        choices=("eager", "verifier_early_stop"),
        default="eager",
        help=(
            "How to schedule multi-view controls. eager runs every requested control; "
            "verifier_early_stop runs controls in order and skips later controls once "
            "the verifier accepts a sample."
        ),
    )
    parser.add_argument("--prompt-preview-limit", type=int, default=16)
    parser.add_argument(
        "--teacher-model",
        default=os.environ.get("DYME_TEACHER_MODEL", "/home/deepseek_VG/deepseek/models/llava-7b-ov"),
    )
    parser.add_argument(
        "--teacher-backend",
        choices=("llava_onevision", "qwen25vl"),
        default=os.environ.get("DYME_TEACHER_BACKEND", "llava_onevision"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--device-map", default=os.environ.get("DYME_TEACHER_DEVICE_MAP", "auto"))
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser.parse_args(argv)


def _selected_controls(raw: str) -> dict[str, dict[str, Any]]:
    names = [name.strip() for name in str(raw or "").split(",") if name.strip()]
    controls: dict[str, dict[str, Any]] = {}
    unknown = [name for name in names if name not in CONTROL_SPECS]
    if unknown:
        raise ValueError(f"Unknown controls: {','.join(unknown)}")
    for name in names:
        controls[name] = CONTROL_SPECS[name]
    return controls


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.execution_policy == "verifier_early_stop" and args.selection_policy != "verifier_first_correct":
        raise ValueError("verifier_early_stop requires --selection-policy verifier_first_correct")

    controls = _selected_controls(args.controls)
    samples, manifest = _prepare_samples(args)
    if args.harness == "chartqa_recoverable":
        return _run_chartqa_recoverable_harness(args, samples, out_dir, manifest)
    if args.harness == "chartqa_closed_loop_recovery":
        return _run_chartqa_closed_loop_recovery(args, samples, out_dir, manifest)

    jobs, previews = _build_jobs(
        samples,
        controls,
        load_images=not (args.dry_run or args.fake_teacher),
    )
    if args.prompt_preview_limit >= 0:
        per_control: dict[str, int] = defaultdict(int)
        limited_previews: list[dict[str, Any]] = []
        for row in previews:
            if per_control[row["control"]] < args.prompt_preview_limit:
                limited_previews.append(row)
                per_control[row["control"]] += 1
        previews = limited_previews
    _write_jsonl(out_dir / "prompt_previews.jsonl", previews)

    manifest.update(
        {
            "controls": {
                name: {
                    "providers": spec["providers"],
                    "prompt_profile": spec["prompt_profile"],
                    "canonicalize_draft": bool(spec.get("canonicalize_draft", False)),
                }
                for name, spec in controls.items()
            },
            "dry_run": args.dry_run,
            "fake_teacher": args.fake_teacher,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "selection_policy": args.selection_policy,
            "execution_policy": args.execution_policy,
        }
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        return 0

    if args.execution_policy == "verifier_early_stop":
        records = _run_verifier_early_stop(args, jobs, controls)
    else:
        outputs = _generate_outputs_for_jobs(args, jobs, controls, desc="teacher draft")
        records = _records_from_outputs(jobs, outputs)

    _write_jsonl(out_dir / "records.jsonl", records)
    _write_summaries(out_dir, records, controls)
    if args.selection_policy == "verifier_first_correct":
        selected_rows = _select_verifier_first_correct(records, list(controls.keys()))
        _write_jsonl(out_dir / "selected_records.jsonl", selected_rows)
        _write_selected_summary(out_dir / "selected_summary.csv", selected_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
