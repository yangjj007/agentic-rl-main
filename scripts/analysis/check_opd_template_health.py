#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path


METRIC_LINE = re.compile(r"(\{.*'loss'\s*:.*\})")
PARTIAL_SECTION = re.compile(
    r"(?im)^\s*(goal(?:\s+statement)?|observation|reasoning|conclusion)\b\s*(?::|=|-)?\s*"
)
ANSWER_HEADING = re.compile(r"(?im)^\s*answer\s*[:=]\s*")


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = METRIC_LINE.search(line)
        if not match:
            continue
        try:
            value = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict) and "loss" in value:
            rows.append(value)
    return rows


def mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else 0.0


def load_candidate_outputs(path: Path, window_steps: int) -> list[str]:
    by_step: dict[int, list[str]] = defaultdict(list)
    if not path.exists():
        return []
    for candidate_file in path.glob("rank*.jsonl"):
        for line in candidate_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
                step = int(row["global_step"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            by_step[step].append(str(row.get("student_output", "")))
    latest_steps = sorted(by_step)[-max(window_steps, 1):]
    return [text for step in latest_steps for text in by_step[step]]


def candidate_template_rates(texts: list[str]) -> tuple[float, float]:
    if not texts:
        return 0.0, 0.0
    partial = 0
    goal_without_answer = 0
    for text in texts:
        names = {
            match.group(1).lower().replace(" statement", "")
            for match in PARTIAL_SECTION.finditer(text)
        }
        has_goal = "goal" in names
        has_answer = ANSWER_HEADING.search(text) is not None
        if has_goal and len(names) >= 2 and not has_answer:
            partial += 1
        elif has_goal and not has_answer:
            partial += 1
        if has_goal and not has_answer:
            goal_without_answer += 1
    n = len(texts)
    return partial / n, goal_without_answer / n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--candidate-window-steps", type=int, default=5)
    parser.add_argument(
        "--allow-teacher-sft-repair",
        action="store_true",
        help="Treat teacher SFT repair routing as an expected mechanism.",
    )
    args = parser.parse_args()

    rows = load_rows(args.log)
    window = rows[-max(args.window, 1):]
    candidate_outputs = (
        load_candidate_outputs(args.candidate_dir, args.candidate_window_steps)
        if args.candidate_dir
        else []
    )
    candidate_partial_rate, candidate_goal_without_answer_rate = candidate_template_rates(
        candidate_outputs
    )
    payload = {
        "rows": len(window),
        "teacher_traj_weight_max": max(
            (float(row.get("loss/teacher_traj_effective_weight", 0.0)) for row in window),
            default=0.0,
        ),
        "teacher_sft_repair_rate_max": max(
            (float(row.get("routing/teacher_sft_repair_rate", 0.0)) for row in window),
            default=0.0,
        ),
        "full_cot_template_rate_mean": mean(window, "completions/full_cot_template_rate"),
        "empty_cot_skeleton_rate_mean": mean(window, "completions/empty_cot_skeleton_rate"),
        "malformed_answer_section_rate_mean": mean(
            window, "completions/malformed_answer_section_rate"
        ),
        "accuracy_reward_mean": mean(window, "rewards/accuracy/mean"),
        "grpo_route_rate_mean": mean(window, "routing/grpo_route_rate"),
        "opd_route_rate_mean": mean(window, "routing/opd_route_rate"),
        "sft_route_rate_mean": mean(window, "routing/sft_route_rate"),
        "grpo_zero_loss_rate_mean": mean(window, "signal/grpo_zero_loss_rate"),
        "all_wrong_rate_mean": mean(window, "signal/group_all_wrong_rate"),
        "degenerate_rate_mean": mean(window, "completions/degenerate_rate"),
        "clipped_ratio_mean": mean(window, "completions/clipped_ratio"),
        "eos_rate_mean": mean(window, "completions/eos_rate"),
        "candidate_rows": len(candidate_outputs),
        "candidate_partial_cot_template_rate": candidate_partial_rate,
        "candidate_goal_without_answer_rate": candidate_goal_without_answer_rate,
    }
    if payload["teacher_traj_weight_max"] > 1e-12 or (
        payload["teacher_sft_repair_rate_max"] > 1e-12
        and not args.allow_teacher_sft_repair
    ):
        payload["status"] = "mechanism_violation"
        code = 2
    elif payload["full_cot_template_rate_mean"] > 0.8 and (
        payload["empty_cot_skeleton_rate_mean"] > 0.2
        or payload["malformed_answer_section_rate_mean"] > 0.2
    ):
        payload["status"] = "template_collapse"
        payload["template_collapse_reason"] = (
            "empty_cot_skeleton"
            if payload["empty_cot_skeleton_rate_mean"] > 0.2
            else "malformed_answer_section"
        )
        code = 3
    elif (
        payload["candidate_partial_cot_template_rate"] > 0.6
        and payload["candidate_goal_without_answer_rate"] > 0.6
    ):
        payload["status"] = "template_drift"
        code = 5
    elif not window:
        payload["status"] = "insufficient"
        code = 4
    else:
        payload["status"] = "ok"
        code = 0
    print(json.dumps(payload, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
