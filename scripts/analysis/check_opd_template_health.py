#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

METRIC_LINE = re.compile(r"(\{.*'loss'\s*:.*\})")
COT_SECTION = re.compile(
    r"(?im)^\s*(goal|observation|reasoning|conclusion|answer)\s*[:.,]\s*"
)
COT_SECTION_ORDER = ("goal", "observation", "reasoning", "conclusion", "answer")
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


def preferred_route_mean(
    rows: list[dict], *, global_key: str, local_key: str
) -> tuple[float, float, float]:
    values = []
    global_count = 0
    local_count = 0
    for row in rows:
        if global_key in row:
            values.append(float(row[global_key]))
            global_count += 1
        elif local_key in row:
            values.append(float(row[local_key]))
            local_count += 1
    count = len(values)
    if not count:
        return 0.0, 0.0, 0.0
    return sum(values) / count, global_count / count, local_count / count


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


def normalize_candidate_preview(text: str) -> str:
    return str(text or "").replace("\\r\\n", "\n").replace("\\n", "\n")


def candidate_template_rates(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {
            "full_cot_template_rate": 0.0,
            "partial_cot_template_rate": 0.0,
            "goal_without_answer_rate": 0.0,
            "canonical_answer_rate": 0.0,
            "malformed_answer_rate": 0.0,
        }
    full = 0
    partial = 0
    goal_without_answer = 0
    canonical_answer = 0
    malformed_answer = 0
    for text in texts:
        text = normalize_candidate_preview(text)
        matches = list(COT_SECTION.finditer(text))
        section_names = tuple(match.group(1).lower() for match in matches)
        is_full = section_names == COT_SECTION_ORDER
        is_goal_without_answer = (
            PARTIAL_SECTION.search(text) is not None and ANSWER_HEADING.search(text) is None
        )
        full += int(is_full)
        if is_full:
            answer_match = matches[-1]
            answer_text = text[answer_match.end():].strip()
            malformed_answer += int(not answer_text or answer_text[0] in ".,;:")
        canonical_answer += int(ANSWER_HEADING.search(text) is not None)
        names = {
            match.group(1).lower().replace(" statement", "")
            for match in PARTIAL_SECTION.finditer(text)
        }
        has_goal = "goal" in names
        has_answer = ANSWER_HEADING.search(text) is not None
        if not is_full and has_goal and not has_answer:
            partial += 1
        if is_goal_without_answer:
            goal_without_answer += 1
    n = len(texts)
    return {
        "full_cot_template_rate": full / n,
        "partial_cot_template_rate": partial / n,
        "goal_without_answer_rate": goal_without_answer / n,
        "canonical_answer_rate": canonical_answer / n,
        "malformed_answer_rate": malformed_answer / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--candidate-window-steps", type=int, default=5)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--recovery-gate-step", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(args.log)
    window = rows[-max(args.window, 1):]
    candidate_outputs = (
        load_candidate_outputs(args.candidate_dir, args.candidate_window_steps)
        if args.candidate_dir
        else []
    )
    candidate_rates = candidate_template_rates(candidate_outputs)
    grpo_route_mean, grpo_global_fraction, grpo_local_fraction = preferred_route_mean(
        window,
        global_key="global_signal/grpo_route_rate",
        local_key="routing/grpo_route_rate",
    )
    payload = {
        "total_rows": len(rows),
        "rows": len(window),
        "teacher_traj_weight_max": max(
            (float(row.get("loss/teacher_traj_effective_weight", 0.0)) for row in window),
            default=0.0,
        ),
        "teacher_sft_repair_rate_max": max(
            (float(row.get("routing/teacher_sft_repair_rate", 0.0)) for row in window),
            default=0.0,
        ),
        "routing/legacy_online_sft_rate_max": max(
            (float(row.get("routing/legacy_online_sft_rate", 0.0)) for row in window),
            default=0.0,
        ),
        "routing/full_hint_hard_target_rate_max": max(
            (float(row.get("routing/full_hint_hard_target_rate", 0.0)) for row in window),
            default=0.0,
        ),
        "full_cot_template_rate_mean": mean(window, "completions/full_cot_template_rate"),
        "empty_cot_skeleton_rate_mean": mean(window, "completions/empty_cot_skeleton_rate"),
        "malformed_answer_section_rate_mean": mean(
            window, "completions/malformed_answer_section_rate"
        ),
        "accuracy_reward_mean": mean(window, "rewards/accuracy/mean"),
        "grpo_route_rate_mean": grpo_route_mean,
        "grpo_route_global_fraction": grpo_global_fraction,
        "grpo_route_local_fallback_fraction": grpo_local_fraction,
        "opd_route_rate_mean": mean(window, "routing/opd_route_rate"),
        "sft_route_rate_mean": mean(window, "routing/sft_route_rate"),
        "grpo_zero_loss_rate_mean": mean(window, "signal/grpo_zero_loss_rate"),
        "all_wrong_rate_mean": mean(window, "signal/group_all_wrong_rate"),
        "degenerate_rate_mean": mean(window, "completions/degenerate_rate"),
        "clipped_ratio_mean": mean(window, "completions/clipped_ratio"),
        "eos_rate_mean": mean(window, "completions/eos_rate"),
        "candidate_rows": len(candidate_outputs),
        "candidate_full_cot_template_rate": candidate_rates["full_cot_template_rate"],
        "candidate_partial_cot_template_rate": candidate_rates["partial_cot_template_rate"],
        "candidate_goal_without_answer_rate": candidate_rates["goal_without_answer_rate"],
        "candidate_canonical_answer_rate": candidate_rates["canonical_answer_rate"],
        "candidate_malformed_answer_rate": candidate_rates["malformed_answer_rate"],
    }
    recovery_stop = False
    if args.recovery_gate_step > 0 and len(rows) >= args.recovery_gate_step:
        recovery_stop = (
            payload["degenerate_rate_mean"] > 0.60
            and payload["accuracy_reward_mean"] < 0.02
            and payload["grpo_route_rate_mean"] < 0.02
        )
        payload["recovery_gate"] = {
            "step": args.recovery_gate_step,
            "accuracy_max": 0.02,
            "grpo_route_max": 0.02,
            "degenerate_min": 0.6,
            "decision": "stop" if recovery_stop else "continue",
        }
    if any(
        payload[key] > 1e-12
        for key in (
            "teacher_traj_weight_max",
            "teacher_sft_repair_rate_max",
            "routing/legacy_online_sft_rate_max",
            "routing/full_hint_hard_target_rate_max",
        )
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
    elif recovery_stop:
        payload["status"] = "recovery_failure"
        code = 6
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

    if args.snapshot_dir is not None:
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        for gate in (20, 40, 60, 100):
            path = args.snapshot_dir / f"gate_{gate}.json"
            if len(rows) >= gate and not path.exists():
                snapshot = {**payload, "gate": gate}
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(snapshot, sort_keys=True) + "\n", encoding="utf-8")
                tmp.replace(path)
    print(json.dumps(payload, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
