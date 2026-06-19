"""Build a deterministic small ChartQA subset for test/ fast training."""
from __future__ import annotations

import argparse
import json
import os
import random


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice a shuffled ChartQA subset for fast training.")
    parser.add_argument(
        "--input",
        default=os.environ.get(
            "DYME_CHARTQA_VF_FULL",
            "data/chartqa/train_medium_vf_full.json",
        ),
    )
    parser.add_argument(
        "--output",
        default=os.environ.get(
            "DYME_FAST_TRAIN_JSON",
            "data/chartqa/train_fast_512.json",
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.environ.get("DYME_FAST_MAX_SAMPLES", "512")),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(
            f"Source dataset not found: {args.input}. "
            "Run scripts/launch_utils.sh ensure_chartqa_vf_full or bash test/prepare_fast_dataset.sh first."
        )

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {args.input}")

    rng = random.Random(args.seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    take = min(args.max_samples, len(indices))
    subset = [data[i] for i in indices[:take]]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False)

    print(f"Wrote {take} samples -> {args.output} (from {len(data)} total, seed={args.seed})")


if __name__ == "__main__":
    main()
