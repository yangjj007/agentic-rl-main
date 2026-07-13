#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.adaptive_supervision import (
    AdaptiveSupervisionConfig,
    AdaptiveSupervisionController,
)


SCENARIOS: dict[str, list[tuple[float, float]]] = {
    "cold_start": [(0.0, 1.0)] * 20,
    "gradual_learning": (
        [(0.0, 1.0)] * 10
        + [(0.10, 0.85)] * 10
        + [(0.25, 0.55)] * 15
        + [(0.40, 0.25)] * 20
    ),
    "regression": [(0.45, 0.20)] * 40 + [(0.0, 1.0)] * 20,
    "single_spike": [(1.0, 0.0)] + [(0.0, 1.0)] * 20,
}


def run_signals(signals: Iterable[tuple[float, float]]) -> list[dict[str, float | int]]:
    controller = AdaptiveSupervisionController(AdaptiveSupervisionConfig())
    return [
        asdict(controller.update(step=step, mixed_rate=mixed, zero_loss_rate=zero))
        for step, (mixed, zero) in enumerate(signals)
    ]


def run_scenario(name: str) -> list[dict[str, float | int]]:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {name}")
    return run_signals(SCENARIOS[name])


def load_signals_jsonl(path: str | Path) -> list[tuple[float, float]]:
    signals: list[tuple[float, float]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
                mixed = float(row["mixed_rate"])
                zero = float(row["zero_loss_rate"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid signal at line {line_number}") from exc
            if not math.isfinite(mixed) or not math.isfinite(zero):
                raise ValueError(f"invalid signal at line {line_number}")
            signals.append((mixed, zero))
    if not signals:
        raise ValueError("signal JSONL contains no rows")
    return signals


def _print_scenario(name: str) -> None:
    states = run_scenario(name)
    for state in states:
        print(json.dumps({"scenario": name, **state}, sort_keys=True))
    final = states[-1]
    print(
        json.dumps(
            {
                "scenario": name,
                "summary": True,
                "steps": len(states),
                "final_mastery": final["mastery"],
                "final_supervision": final["supervision"],
                "final_opsd_weight": final["opsd_weight"],
                "final_teacher_traj_weight": final["teacher_traj_weight"],
                "final_opd_max_per_prompt": final["opd_max_per_prompt"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--input-jsonl")
    args = parser.parse_args()
    if args.input_jsonl:
        states = run_signals(load_signals_jsonl(args.input_jsonl))
        for state in states:
            print(json.dumps({"scenario": "jsonl", **state}, sort_keys=True))
        return
    names = sorted(SCENARIOS) if args.all else [args.scenario or "gradual_learning"]
    for name in names:
        _print_scenario(name)


if __name__ == "__main__":
    main()
