#!/usr/bin/env python3
"""
F1: add visual_fact_hint field to ChartQA JSON using hint as fallback.
Run before training with hybrid/visual privileged profiles.
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--also-set-visual-fact",
        action="store_true",
        help="Also populate visual_fact when missing (backward compatible)",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        hint = entry.get("hint", "")
        if not entry.get("visual_fact_hint"):
            entry["visual_fact_hint"] = hint
        if args.also_set_visual_fact and not entry.get("visual_fact"):
            entry["visual_fact"] = hint

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} records with visual_fact_hint to {args.output}")


if __name__ == "__main__":
    main()
