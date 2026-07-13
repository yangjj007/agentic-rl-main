#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_ENV = {
    "DYME_NUM_TRAIN_EPOCHS": "4",
    "DYME_FAST_NUM_TRAIN_EPOCHS": "4",
    "DYME_SAVE_STRATEGY": "steps",
    "DYME_SAVE_STEPS": "50",
    "DYME_SAVE_TOTAL_LIMIT": "3",
    "DYME_TEACHER_TRAJECTORY": "0",
    "DYME_DISABLE_ONLINE_SFT_SLOTS": "1",
    "DYME_ONLINE_SFT_ON_ALL_WRONG": "0",
    "DYME_TEACHER_PROBE_FAILURE_ROUTE": "mixed_grpo_all_wrong_skip",
    "DYME_ADAPTIVE_SUPERVISION": "1",
    "DYME_ADAPTIVE_READINESS_SOURCE": "global_grpo_route",
    "DYME_ADAPTIVE_EMA_ALPHA": "0.10",
    "DYME_ADAPTIVE_TARGET_READINESS": "0.30",
    "DYME_ADAPTIVE_OPSD_INITIAL_WEIGHT": "1.5",
    "DYME_ADAPTIVE_OPSD_FINAL_WEIGHT": "0.5",
    "DYME_ADAPTIVE_TEACHER_INITIAL_WEIGHT": "0.0",
    "DYME_ADAPTIVE_TEACHER_FINAL_WEIGHT": "0.0",
    "DYME_ADAPTIVE_OPSD_INITIAL_CAP": "8",
    "DYME_ADAPTIVE_OPSD_FINAL_CAP": "2",
    "DYME_EFFECTIVE_SAMPLING": "1",
    "DYME_EFFECTIVE_SAMPLING_AFTER_STEP": "0",
    "DYME_EFFECTIVE_SAMPLING_START_PROGRESS": "0.0",
    "DYME_OPSD_SKIP_DEGENERATE": "0",
    "DYME_GLOBAL_SIGNAL_LOGGING": "1",
    "DYME_EVAL_FORMAT_REWARD": "0",
}


def audit_run_env(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    env = payload.get("env", {})
    violations = [
        {"key": key, "expected": expected, "actual": env.get(key)}
        for key, expected in EXPECTED_ENV.items()
        if env.get(key) != expected
    ]
    return {
        "status": "violation" if violations else "ok",
        "path": str(path),
        "checked": len(EXPECTED_ENV),
        "violations": violations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the frozen no-full-hint OPD runtime environment.")
    parser.add_argument("run_env", type=Path)
    args = parser.parse_args(argv)
    result = audit_run_env(args.run_env)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
