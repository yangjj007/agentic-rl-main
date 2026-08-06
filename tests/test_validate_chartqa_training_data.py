"""Regression tests for the fail-fast ChartQA training-data gate.

These tests use temporary files and pure validator helpers. They do not load a
DePlot model or import torch, so they also run on CPU-only data hosts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.validate_chartqa_training_data as validator


_HINT = (
    "Goal: Find the value.\n"
    "Observation: The chart reports A as 1.\n"
    "Reasoning: Read the value for A.\n"
    "Conclusion: The value is 1."
)
_MISSING = object()


def _deplot(*, source: str = "google/deplot", table: object = "A | 1") -> str:
    return json.dumps(
        {"source": source, "model_id": "google/deplot", "parsed_table": table}
    )


def _qwen_meta(hint: str = _HINT, **overrides: object) -> dict[str, object]:
    meta: dict[str, object] = {
        "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "task": "chart",
        "status": "ok",
        "rewritten_hint_chars": len(hint),
    }
    meta.update(overrides)
    return meta


def _row(
    tmp_path: Path,
    *,
    image: bool = True,
    deplot: object = "real",
    hint: str = _HINT,
    metadata: object = _MISSING,
) -> dict[str, object]:
    image_path = tmp_path / "chart.png"
    if image:
        Image.new("RGB", (2, 2)).save(image_path)
    if deplot == "real":
        deplot_value: object = _deplot()
    elif deplot == "placeholder":
        deplot_value = _deplot(
            source="deplot_placeholder",
            table={"note": "DePlot unavailable or image missing"},
        )
    elif deplot == "missing":
        deplot_value = None
    else:
        deplot_value = deplot
    row: dict[str, object] = {
        "question": "What is the value?",
        "answer": "1",
        "image": str(image_path),
        "hint": hint,
        "visual_fact_deplot": deplot_value,
    }
    if metadata is _MISSING:
        row["dyme_rewrite"] = _qwen_meta(hint)
    elif metadata is not None:
        row["dyme_rewrite"] = metadata
    return row


def _write_dataset(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "train_vf_full.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


@pytest.fixture
def local_image_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from depending on this checkout's image directories."""

    monkeypatch.setattr(validator, "resolve_image_path", lambda value: value)


def test_valid_real_deplot_and_qwen_rewrite_pass(
    tmp_path: Path, local_image_resolver: None
) -> None:
    stats, errors = validator.validate(
        _write_dataset(tmp_path, [_row(tmp_path)]), expected_samples=1
    )

    assert errors == []
    assert stats["effective_rows"] == 1
    assert stats["image_missing"] == 0
    assert stats["deplot_real"] == 1
    assert stats["deplot_placeholder"] == 0
    assert stats["qwen_rewrite_ok"] == 1


@pytest.mark.parametrize(
    ("deplot_kind", "stat_key"),
    [("placeholder", "deplot_placeholder"), ("missing", "deplot_missing")],
)
def test_placeholder_or_missing_deplot_is_rejected(
    tmp_path: Path,
    local_image_resolver: None,
    deplot_kind: str,
    stat_key: str,
) -> None:
    stats, errors = validator.validate(
        _write_dataset(tmp_path, [_row(tmp_path, deplot=deplot_kind)])
    )

    assert errors
    assert stats[stat_key] == 1
    assert stats["deplot_real"] == 0


def test_missing_image_is_rejected(tmp_path: Path, local_image_resolver: None) -> None:
    stats, errors = validator.validate(
        _write_dataset(tmp_path, [_row(tmp_path, image=False)])
    )

    assert errors
    assert stats["image_missing"] == 1


def test_unreadable_image_is_rejected(tmp_path: Path, local_image_resolver: None) -> None:
    row = _row(tmp_path)
    Path(str(row["image"])).write_bytes(b"not an image")
    stats, errors = validator.validate(_write_dataset(tmp_path, [row]))

    assert errors
    assert stats["image_missing"] == 0
    assert stats["image_unreadable"] == 1


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"model": "Qwen/Qwen2.5-14B-Instruct-AWQ", "task": "chart", "status": "fallback"},
        {"model": "meta-llama/Llama-3.1-8B-Instruct", "task": "chart", "status": "ok"},
    ],
)
def test_non_qwen_or_missing_rewrite_metadata_is_rejected(
    tmp_path: Path,
    local_image_resolver: None,
    metadata: object,
) -> None:
    stats, errors = validator.validate(
        _write_dataset(tmp_path, [_row(tmp_path, metadata=metadata)])
    )

    assert errors
    assert stats["qwen_rewrite_ok"] == 0
    assert stats["hint_invalid"] == 1


def test_expected_sample_count_mismatch_is_rejected(
    tmp_path: Path, local_image_resolver: None
) -> None:
    stats, errors = validator.validate(
        _write_dataset(tmp_path, [_row(tmp_path)]), expected_samples=2
    )

    assert errors
    assert any("expected=2" in error for error in errors)
    assert stats["effective_rows"] == 1


def test_machine_rows_are_excluded_like_chart_collector(
    tmp_path: Path, local_image_resolver: None
) -> None:
    machine = _row(tmp_path, deplot="missing")
    machine["human_or_machine"] = 1
    stats, errors = validator.validate(
        _write_dataset(tmp_path, [machine]), expected_samples=0
    )

    assert errors
    assert stats["effective_rows"] == 0
    assert stats["excluded_rows"] == 1
    assert stats["deplot_missing"] == 0


def test_required_fields_are_checked_for_effective_rows(
    tmp_path: Path, local_image_resolver: None
) -> None:
    row = _row(tmp_path)
    row.pop("answer")
    stats, errors = validator.validate(_write_dataset(tmp_path, [row]))

    assert errors
    assert stats["answer_missing"] == 1
    assert stats["required_missing"] == 1


def test_malformed_deplot_payload_is_rejected(
    tmp_path: Path, local_image_resolver: None
) -> None:
    stats, errors = validator.validate(
        _write_dataset(tmp_path, [_row(tmp_path, deplot="{not valid json")])
    )

    assert errors
    assert stats["deplot_unknown"] == 1
    assert stats["deplot_real"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"source": "google/deplot", "parsed_table": {"not": "text"}},
        {"source": "google/deplot", "parsed_table": "A | 1", "error": "inference_failed"},
    ],
)
def test_non_text_or_error_deplot_payload_is_rejected_without_crashing(
    tmp_path: Path, local_image_resolver: None, payload: object
) -> None:
    stats, errors = validator.validate(
        _write_dataset(tmp_path, [_row(tmp_path, deplot=payload)])
    )

    assert errors
    assert stats["deplot_unknown"] == 1
    assert stats["deplot_real"] == 0


def test_malformed_dataset_json_is_reported_by_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "malformed.json"
    path.write_text("[{ definitely not json", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["validator", "--input", str(path)])

    result = validator.main()

    captured = capsys.readouterr()
    assert result == 2
    assert "DyME-DATA-ERROR" in captured.err
