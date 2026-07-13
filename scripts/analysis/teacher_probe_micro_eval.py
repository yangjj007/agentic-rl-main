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
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils.chart.evaluator import eval_teacher_probe_chart
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
    "baseline_deplot_only": {
        "providers": ["format_only", "visual_facts_deplot"],
        "prompt_profile": "chartqa_short_answer",
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
        "privileged_profile": "text",
        "teacher_probe": {"prompt_profile": control["prompt_profile"]},
    }
    context_sample = sample if load_images else {**sample, "image": None}
    suffix, teacher_images = build_privileged_context(
        context_sample,
        control["providers"],
        privileged_profile="text",
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
        hint = str(sample.get("hint") or sample.get("visual_fact_hint") or "").strip()
        observation = hint.splitlines()[1] if len(hint.splitlines()) > 1 else f"Reference answer is {answer}."
        return (
            "Goal: Follow the verified training hint.\n"
            f"{observation}\n"
            "Reasoning: The oracle hint and reference answer are authoritative.\n"
            f"Conclusion: {answer}.\n"
            f"Answer: {answer}"
        )
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
    from reward_utils.teacher_generate import TeacherGenerateRequest, teacher_generate_batch

    teacher_model, processor = _load_teacher(args)
    requests = [
        TeacherGenerateRequest(
            prompt_text=job["prompt"],
            images=job["images"],
            response_prefix=job.get("response_prefix", ""),
            max_new_tokens=args.max_new_tokens,
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
        desc=f"teacher micro-eval bs={max(1, int(args.batch_size))}",
    ):
        chunk_texts, _ = teacher_generate_batch(
            teacher_model,
            processor,
            chunk,
            timing_kind="teacher_probe_micro_eval",
        )
        texts.extend(chunk_texts)
    return texts


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
        "--controls",
        default=",".join(DEFAULT_CONTROL_NAMES),
        help=(
            "Comma-separated controls to run. Available: "
            + ",".join(CONTROL_SPECS.keys())
        ),
    )
    parser.add_argument("--prompt-preview-limit", type=int, default=16)
    parser.add_argument(
        "--teacher-model",
        default=os.environ.get("DYME_TEACHER_MODEL", "/home/deepseek_VG/deepseek/models/llava-7b-ov"),
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

    controls = _selected_controls(args.controls)
    samples, manifest = _prepare_samples(args)
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
                }
                for name, spec in controls.items()
            },
            "dry_run": args.dry_run,
            "fake_teacher": args.fake_teacher,
            "max_samples": args.max_samples,
            "seed": args.seed,
        }
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        return 0

    if args.fake_teacher:
        outputs = [_fake_teacher_output(job["control"], job["sample"]) for job in jobs]
    else:
        outputs = _run_real_teacher(args, jobs)

    records = _records_from_outputs(jobs, outputs)
    _write_jsonl(out_dir / "records.jsonl", records)
    _write_summaries(out_dir, records, controls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
