#!/usr/bin/env python3
"""
F2: optional DePlot/table parsing stub for ChartQA visual_fact_deplot.
Replace the placeholder implementation with a real DePlot pipeline when available.
"""
import argparse
import json


def _placeholder_deplot_table(entry: dict) -> str:
    """Minimal structured placeholder until DePlot is wired in."""
    question = entry.get("question", entry.get("question_wo_prompt", ""))
    answer = entry.get("answer", "")
    payload = {
        "source": "deplot_placeholder",
        "question": question,
        "parsed_table": {"note": "Replace with DePlot output"},
        "answer_hint": answer[:80] if answer else "",
    }
    return json.dumps(payload, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        if not entry.get("visual_fact_deplot"):
            entry["visual_fact_deplot"] = _placeholder_deplot_table(entry)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} records with visual_fact_deplot to {args.output}")


if __name__ == "__main__":
    main()
