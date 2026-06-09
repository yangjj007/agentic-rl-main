#!/usr/bin/env python3
"""
Optional offline script: add visual_fact field to ChartQA JSON using hint as fallback.
Replace with DePlot/BiomedGPT pipeline when available.
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        if not entry.get("visual_fact"):
            entry["visual_fact"] = entry.get("hint", "")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} records to {args.output}")


if __name__ == "__main__":
    main()
