#!/usr/bin/env python3
"""Set legacy ChartQA visual_fact fields to null before optional DePlot enrichment."""
import argparse
import json

try:
    from scripts.repair_chartqa_visual_facts import repair_payload
except ModuleNotFoundError:  # pragma: no cover - direct script execution from scripts/
    from repair_chartqa_visual_facts import repair_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--also-set-visual-fact",
        action="store_true",
        help="Deprecated no-op kept for backward-compatible launch commands",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    data, stats = repair_payload(data)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        "Wrote "
        f"{stats['records']} records with visual_fact/visual_fact_hint=null to {args.output}; "
        f"overwritten={stats['visual_fact_overwritten'] + stats['visual_fact_hint_overwritten']}"
    )


if __name__ == "__main__":
    main()
