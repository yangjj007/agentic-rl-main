#!/usr/bin/env python3
"""Fail-fast validation for ChartQA DyME training data.

This is intentionally run before model/DeepSpeed startup.  It catches stale
vf_full files (especially DePlot placeholders), missing images, and datasets
whose hints were not produced by the Qwen rewrite pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Allow direct execution as ``python scripts/...`` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_utils.chart.deplot_pipeline import (
    has_real_deplot,
    is_deplot_placeholder,
)
from data_utils.paths import resolve_image_path


_HINT_SECTIONS = ("Goal", "Observation", "Reasoning", "Conclusion")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_FIELDS = ("question", "answer", "image", "hint")
_HINT_SECTION_RE = re.compile(
    r"(?im)^\s*(Goal|Observation|Reasoning|Conclusion)\s*:\s*"
)


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", data.get("examples", []))
    if not isinstance(data, list) or not data:
        raise ValueError("dataset must be a non-empty JSON list (or data/examples wrapper)")
    if not all(isinstance(x, dict) for x in data):
        raise ValueError("dataset contains a non-object record")
    return data


def _hint_sections(hint: str) -> dict[str, str]:
    matches = list(_HINT_SECTION_RE.finditer(hint or ""))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(hint)
        body = hint[match.end() : end].strip()
        if body:
            sections[match.group(1).lower()] = body
    return sections


def _deplot_source(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("source") or "")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return ""
        return str(parsed.get("source") or "") if isinstance(parsed, dict) else ""
    return ""


def _has_value(value: Any) -> bool:
    """Return whether a required value is present.

    ChartQA answers are usually strings, but numeric zero is also a valid
    answer in general-purpose fixtures; do not reject it via truthiness.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _is_qwen_model(model: str) -> bool:
    """Recognize an actual Qwen model id without accepting ``notqwen``."""
    parts = [part for part in re.split(r"[/\\\\]", model.strip()) if part]
    return any(re.match(r"qwen(?:$|[0-9._-])", part, re.I) for part in parts)


def _image_is_readable(path: str) -> bool:
    """Match the trainer's PIL image load contract without decoding pixels."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def validate(
    path: Path,
    *,
    require_qwen: bool = True,
    require_real_deplot: bool = True,
    expected_samples: int = 0,
) -> tuple[dict[str, Any], list[str]]:
    """Validate the rows that the ChartQA collector will actually consume."""
    if not path.is_file():
        return (
            {
                "path": str(path),
                "total": 0,
                "effective_rows": 0,
                "excluded_rows": 0,
                "image_missing": 0,
                "image_unreadable": 0,
                "required_missing": 0,
                "question_missing": 0,
                "answer_missing": 0,
                "hint_missing": 0,
                "deplot_real": 0,
                "deplot_placeholder": 0,
                "deplot_missing": 0,
                "deplot_unknown": 0,
                "qwen_rewrite_ok": 0,
                "hint_invalid": 0,
            },
            [f"configured training dataset does not exist: {path}"],
        )
    rows = _rows(path)
    effective_rows = [row for row in rows if row.get("human_or_machine", 0) == 0]
    missing_images = 0
    unreadable_images = 0
    real_deplot = 0
    placeholder_deplot = 0
    missing_deplot = 0
    unknown_deplot = 0
    qwen_ok = 0
    bad_hints = 0
    missing_fields: Counter[str] = Counter()
    errors: list[str] = []
    if not effective_rows:
        errors.append("dataset has no effective training rows (human_or_machine == 0)")
    if expected_samples < 0:
        errors.append(f"expected sample count must be non-negative, got {expected_samples}")
    if expected_samples and len(effective_rows) != expected_samples:
        errors.append(
            f"effective training rows={len(effective_rows)}, expected={expected_samples}"
        )
    for i, row in enumerate(rows):
        if row.get("human_or_machine", 0) != 0:
            continue
        missing = [
            field for field in _REQUIRED_FIELDS
            if not _has_value(row.get(field))
        ]
        if missing:
            for field in missing:
                missing_fields[field] += 1
            if len(errors) < 8:
                errors.append(f"row {i}: missing required field(s): {', '.join(missing)}")
        image = str(row.get("image") or "")
        resolved_image = resolve_image_path(image)
        if not image or not Path(resolved_image).is_file():
            missing_images += 1
            if len(errors) < 8:
                errors.append(
                    f"row {i}: image not found: {image!r} (resolved={resolved_image!r})"
                )
        elif not _image_is_readable(resolved_image):
            unreadable_images += 1
            if len(errors) < 8:
                errors.append(
                    f"row {i}: image is not readable by PIL: {image!r} "
                    f"(resolved={resolved_image!r})"
                )

        deplot = row.get("visual_fact_deplot")
        if not deplot:
            missing_deplot += 1
        else:
            try:
                deplot_is_real = has_real_deplot(deplot)
            except (AttributeError, TypeError, ValueError):
                deplot_is_real = False
            if deplot_is_real:
                real_deplot += 1
            else:
                source = _deplot_source(deplot)
                if is_deplot_placeholder(deplot) or source == "deplot_placeholder":
                    placeholder_deplot += 1
                else:
                    unknown_deplot += 1

        meta = row.get("dyme_rewrite")
        model = str(meta.get("model", "")) if isinstance(meta, dict) else ""
        status = str(meta.get("status", "")) if isinstance(meta, dict) else ""
        task = str(meta.get("task", "")) if isinstance(meta, dict) else ""
        hint = str(row.get("hint") or "").strip()
        rewritten_chars = meta.get("rewritten_hint_chars") if isinstance(meta, dict) else None
        provenance_ok = (
            status == "ok"
            and task in ("chart", "chartqa")
            and _is_qwen_model(model)
            and type(rewritten_chars) is int
            and rewritten_chars == len(hint)
        )
        structure_ok = all(name.lower() in _hint_sections(hint) for name in _HINT_SECTIONS)
        is_qwen = provenance_ok and bool(hint) and structure_ok
        if is_qwen:
            qwen_ok += 1
        if not hint or (require_qwen and not is_qwen):
            bad_hints += 1
            if len(errors) < 8:
                errors.append(
                    f"row {i}: hint is not a valid Qwen rewrite "
                    f"(status={status!r}, task={task!r}, model={model!r}, "
                    f"sections={sorted(_hint_sections(hint))})"
                )

    if missing_images:
        errors.append(f"{missing_images} effective rows have missing images")
    if unreadable_images:
        errors.append(f"{unreadable_images} effective rows have unreadable images")
    if require_real_deplot and real_deplot != len(effective_rows):
        errors.append(
            "real DePlot coverage is incomplete: "
            f"real={real_deplot}, placeholder={placeholder_deplot}, "
            f"missing={missing_deplot}, unknown={unknown_deplot}, "
            f"effective={len(effective_rows)}"
        )
    if require_qwen and qwen_ok != len(effective_rows):
        errors.append(
            f"Qwen rewrite provenance/structure is incomplete: "
            f"valid={qwen_ok}, effective={len(effective_rows)}"
        )

    total = len(rows)
    stats = {
        "path": str(path), "total": total,
        "effective_rows": len(effective_rows),
        "excluded_rows": total - len(effective_rows),
        "image_missing": missing_images,
        "image_unreadable": unreadable_images,
        "required_missing": sum(missing_fields.values()),
        "question_missing": missing_fields.get("question", 0),
        "answer_missing": missing_fields.get("answer", 0),
        "hint_missing": missing_fields.get("hint", 0),
        "deplot_real": real_deplot, "deplot_placeholder": placeholder_deplot,
        "deplot_missing": missing_deplot, "deplot_unknown": unknown_deplot,
        "qwen_rewrite_ok": qwen_ok, "hint_invalid": bad_hints,
    }
    if errors or missing_fields or missing_images or unreadable_images or (require_real_deplot and real_deplot != len(effective_rows)) or (
        require_qwen and qwen_ok != len(effective_rows)
    ):
        return stats, errors
    return stats, []


def _config_input(config_arg: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    # Import only when --config is requested; standalone validation remains
    # useful in lightweight data-preparation environments.  Some training
    # configs import the full OPSD package (and therefore torch), so fall back
    # to a small strict-recipe metadata resolver when optional training dependencies are not
    # installed.  The fallback is intentionally read-only and only extracts
    # the literal dataset path/validation contract.
    try:
        from config.loader import load_config

        config = load_config(config_arg)
    except ModuleNotFoundError as exc:
        # The known strict-config dependency chain imports torch through
        # ``opsd_utils``.  Do not mask unrelated missing-module/configuration
        # errors with a hard-coded fallback.
        if getattr(exc, "name", "") != "torch":
            raise
        config = _load_config_metadata_without_training_deps(config_arg, exc)
    dataset = config.get("dataset") or {}
    raw_path = dataset.get("train_dataset")
    if not raw_path:
        raise ValueError(f"config {config_arg!r} has no dataset.train_dataset")
    path = Path(os.path.expanduser(str(raw_path)))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path, config, config.get("data_validation") or {}


def _load_config_metadata_without_training_deps(
    config_arg: str, import_error: Exception
) -> dict[str, Any]:
    """Resolve strict preflight metadata without importing model/OPSD modules.

    This fallback supports the strict image-checker alias in CPU-only data
    environments.  It deliberately does not execute arbitrary config code;
    if the expected literal contract cannot be recovered, the original import
    error is surfaced instead of guessing a dataset.
    """
    if config_arg not in {
        "opd_7b_dyme_probe_image_checker",
        "opd_7b_probe_image_checker",
    } and not config_arg.endswith("config_opd_7b_dyme_probe_image_checker.yaml"):
        raise import_error
    real_path = _PROJECT_ROOT / "data/chartqa/train_new_prerefine_vf_full_real_deplot_fp32_qwen25.json"
    return {
        "dataset": {
            "train_dataset": str(real_path)
        },
        "data_validation": {
            "strict": True,
            "require_real_deplot": True,
            "require_qwen_rewrite": True,
            "expected_samples": 4576,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--input")
    source.add_argument("--config", help="Validate the exact dataset selected by a config alias/path")
    ap.add_argument("--allow-non-qwen", action="store_true")
    ap.add_argument("--expected-samples", type=int, default=0)
    args = ap.parse_args()
    try:
        config_validation: dict[str, Any] = {}
        config_name = ""
        if args.config:
            path, _config, config_validation = _config_input(args.config)
            config_name = args.config
        else:
            path = Path(os.path.expanduser(args.input))
            if not path.is_absolute() and not path.exists():
                path = _PROJECT_ROOT / path
        expected_samples = args.expected_samples or int(config_validation.get("expected_samples", 0) or 0)
        require_qwen = bool(config_validation.get("require_qwen_rewrite", True)) and not args.allow_non_qwen
        require_real_deplot = bool(config_validation.get("require_real_deplot", True))
        stats, errors = validate(
            path,
            require_qwen=require_qwen,
            require_real_deplot=require_real_deplot,
            expected_samples=expected_samples,
        )
    except Exception as exc:
        print("[DyME-DATA-WARNING] refusing to start training", file=sys.stderr)
        print(f"[DyME-DATA-ERROR] {args.config or args.input}: {exc}", file=sys.stderr)
        return 2
    print(
        f"[DyME-DATA-PREFLIGHT] config={config_name or '<input>'} "
        f"train_dataset={path}"
    )
    print("[DyME-DATA-CHECK] " + json.dumps(stats, ensure_ascii=False, sort_keys=True))
    if errors:
        print("[DyME-DATA-WARNING] refusing to start training", file=sys.stderr)
        print("[DyME-DATA-ERROR] training data gate failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print("  Require every row to have an existing image, real DePlot, and successful Qwen rewrite.", file=sys.stderr)
        return 2
    print("[DyME-DATA-CHECK] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
