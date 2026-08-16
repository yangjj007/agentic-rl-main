#!/usr/bin/env python3
"""Materialize exactly the ChartQA images referenced by the Qwen25 OPD corpus.

The rewritten annotation file deliberately stores stable ``train_XXXXXX.png``
paths instead of embedding image bytes.  This utility streams the public
HuggingFaceM4/ChartQA training split, verifies each selected row's question
and answer against the rewritten corpus, and saves only the referenced images.
It avoids downloading validation/test images or unrelated training examples.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATASET = _PROJECT_ROOT / "data/chartqa/train_new_prerefine_vf_full_real_deplot_fp32_qwen25.json"
_IMAGE_NAME_RE = re.compile(r"train_(\d{6})\.png$")


def _answer_text(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def _target_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    targets: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("human_or_machine", 0) != 0:
            continue
        match = _IMAGE_NAME_RE.search(str(row.get("image") or ""))
        if match is None:
            raise ValueError(f"row has no canonical ChartQA train image path: {row.get('image')!r}")
        index = int(match.group(1))
        if index in targets:
            raise ValueError(f"duplicate ChartQA image index in rewritten corpus: {index}")
        targets[index] = row
    if not targets:
        raise ValueError(f"no effective rows in {path}")
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=_PROJECT_ROOT / "data/images/chartqa/images",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    targets = _target_rows(dataset_path)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = {
        index: args.image_dir / f"train_{index:06d}.png" for index in targets
    }
    existing = sum(path.is_file() for path in expected_paths.values())
    print(
        f"[ChartQA-Qwen25] dataset={dataset_path} targets={len(targets)} "
        f"already_present={existing}",
        flush=True,
    )

    stream = load_dataset("HuggingFaceM4/ChartQA", split="train", streaming=True)
    saved = 0
    verified = 0
    max_index = max(targets)
    for index, source in enumerate(stream):
        if index > max_index:
            break
        target = targets.get(index)
        if target is None:
            continue

        source_question = str(source.get("query") or "").strip()
        target_question = str(target.get("question") or "").strip()
        source_answer = _answer_text(source.get("label"))
        target_answer = _answer_text(target.get("answer"))
        if source_question != target_question or source_answer != target_answer:
            raise RuntimeError(
                "ChartQA row mismatch at train index "
                f"{index}: source=({source_question!r}, {source_answer!r}) "
                f"rewritten=({target_question!r}, {target_answer!r})"
            )
        verified += 1
        destination = expected_paths[index]
        if not destination.is_file():
            image = source.get("image")
            if image is None:
                raise RuntimeError(f"ChartQA source row {index} has no image")
            temporary = destination.with_suffix(".tmp.png")
            image.convert("RGB").save(temporary)
            temporary.replace(destination)
            saved += 1
        if verified % 500 == 0 or verified == len(targets):
            print(
                f"[ChartQA-Qwen25] verified={verified}/{len(targets)} saved={saved}",
                flush=True,
            )

    if verified != len(targets):
        raise RuntimeError(
            f"source train split ended early: verified {verified}/{len(targets)} target rows"
        )
    missing = [str(path) for path in expected_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"failed to materialize {len(missing)} target images; first={missing[0]}")
    print(
        f"[ChartQA-Qwen25] complete verified={verified} saved={saved} "
        f"image_dir={args.image_dir.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
