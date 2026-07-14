#!/usr/bin/env python3
"""Repair ChartQA visual-fact fields so hint-derived text is not teacher evidence."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any


VISUAL_FACT_FIELDS = ("visual_fact", "visual_fact_hint")


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("data", payload.get("examples", []))
    else:
        records = []
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("input must be a JSON list or a dict with data/examples records")
    return records


def repair_chartqa_visual_fact_records(records: list[dict[str, Any]]) -> dict[str, int]:
    """Set ChartQA legacy visual_fact fields to JSON null in-place."""
    stats = {
        "records": len(records),
        "visual_fact_null": 0,
        "visual_fact_hint_null": 0,
        "visual_fact_overwritten": 0,
        "visual_fact_hint_overwritten": 0,
    }
    for row in records:
        for field in VISUAL_FACT_FIELDS:
            if row.get(field) is not None:
                stats[f"{field}_overwritten"] += 1
            row[field] = None
            stats[f"{field}_null"] += 1
    return stats


def repair_payload(payload: Any) -> tuple[Any, dict[str, int]]:
    records = _records_from_payload(payload)
    stats = repair_chartqa_visual_fact_records(records)
    return payload, stats


def _write_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Set ChartQA visual_fact and visual_fact_hint to null. "
            "Use visual_fact_deplot or future image-derived fields for teacher visual evidence."
        )
    )
    parser.add_argument("--input", required=True, help="Input ChartQA JSON file")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--in-place", action="store_true", help="Rewrite --input in place")
    args = parser.parse_args()

    if args.in_place and args.output:
        raise SystemExit("--in-place and --output are mutually exclusive")
    if not args.in_place and not args.output:
        raise SystemExit("provide --output or --in-place")

    with open(args.input, encoding="utf-8") as f:
        payload = json.load(f)

    repaired, stats = repair_payload(payload)
    out_path = args.input if args.in_place else args.output
    _write_json(out_path, repaired)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
