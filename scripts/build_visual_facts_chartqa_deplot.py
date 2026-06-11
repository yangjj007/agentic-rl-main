#!/usr/bin/env python3
"""
F2: ChartQA visual_fact_deplot via offline DePlot (google/deplot) batch pipeline.
Falls back to placeholder when disabled, on missing images, or inference failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.chart.deplot_pipeline import enrich_entries_with_deplot
from data_utils.paths import project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run real DePlot inference (default: enabled)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--cache",
        default=project_path("data/chartqa/deplot_cache.json"),
        help="Incremental cache keyed by resolved image path",
    )
    parser.add_argument(
        "--replace-placeholder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--only-missing", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all entries")
    parser.add_argument("--model-id", default="google/deplot")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    stats = enrich_entries_with_deplot(
        data,
        enabled=args.enabled,
        model_id=args.model_id,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        cache_path=args.cache,
        replace_placeholder=args.replace_placeholder,
        only_missing=args.only_missing,
        max_samples=args.max_samples,
        device=None if args.device == "auto" else args.device,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(data)} records to {args.output} | "
        f"real={stats['real']} cached={stats['cached']} "
        f"placeholder={stats['placeholder']} skipped={stats['skipped']} failed={stats['failed']}"
    )


if __name__ == "__main__":
    main()
