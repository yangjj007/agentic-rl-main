"""Smoke checks for the OPD teacher-probe ablation path.

These tests intentionally avoid launching training or loading VLM weights. They
cover the cheap checks needed before spending GPU time on the no-probe ablation:
config overrides, routing semantics, and log-level anti-leakage/health signals.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.constants import MODE_GRPO, MODE_OPSD, MODE_SFT
from opsd_utils.mode_router import route_completion_modes


ROOT = Path(__file__).resolve().parents[1]
CLEAN_NO_GOLD_LOG = ROOT / "outputs/test-fast/logs/train_test_opd_20260621_212323.log"
OLD_VS_NO_PROBE_LIKE_LOG = ROOT / "outputs/test-fast/logs/train_test_opd_20260620_111531.log"


def _metric_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"\{.*\}", line)
        if not match:
            continue
        try:
            row = ast.literal_eval(match.group())
        except (SyntaxError, ValueError):
            continue
        if isinstance(row, dict) and any(key.startswith(("routing/", "rewards/", "completions/")) for key in row):
            rows.append(row)
    return rows


def _values(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if key in row and row[key] is not None]


def _mean(values: list[float]) -> float:
    assert values, "metric is missing from parsed log rows"
    return sum(values) / len(values)


def _last_mean(rows: list[dict], key: str, window: int = 20) -> float:
    values = _values(rows, key)
    assert values, f"{key} is missing from parsed log rows"
    tail = values[-window:]
    return sum(tail) / len(tail)


def _teacher_probe_cfg(**gate_overrides: object) -> dict:
    return {
        "enabled": True,
        "mode": "dyme_teacher_probe_opd",
        "gate": {
            "correct_threshold": 0.5,
            "per_completion_opsd": True,
            "require_format_for_opsd": False,
            **gate_overrides,
        },
        "teacher_probe": {"enabled": True},
        "text_include_gold": False,
    }


def test_teacher_probe_route_marks_mixed_wrong_completions_for_probe() -> None:
    acc = torch.tensor([[1.0, 0.0, 0.0]])
    modes = route_completion_modes(acc, 3, 3, _teacher_probe_cfg(), [True])

    assert modes == [MODE_GRPO, MODE_OPSD, MODE_OPSD]


def test_teacher_probe_route_keeps_all_wrong_groups_on_sft() -> None:
    acc = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    modes = route_completion_modes(acc, 4, 4, _teacher_probe_cfg(), [True])

    assert modes == [MODE_SFT, MODE_SFT, MODE_SFT, MODE_SFT]


def test_teacher_probe_route_can_probe_all_wrong_after_configured_step() -> None:
    acc = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    cfg = _teacher_probe_cfg()
    cfg["teacher_probe"]["probe_all_wrong_after_step"] = 100
    cfg["global_step"] = 150

    modes = route_completion_modes(acc, 4, 4, cfg, [True])

    assert modes == [MODE_OPSD, MODE_OPSD, MODE_OPSD, MODE_OPSD]


def test_clean_opd_config_defaults_to_no_gold_deplot_probe() -> None:
    code = """
import json
from config.loader import load_config

c = load_config("opd_7b_dyme_probe")
opsd = c["opsd"]
probe = opsd["teacher_probe"]
payload = {
    "providers": opsd["privileged_providers"],
    "probe_providers": probe["context_providers"],
    "prompt_profile": probe["prompt_profile"],
    "answer_parser": probe["answer_parser"],
    "skip_no_evidence": probe["skip_no_evidence"],
    "candidate_log_enabled": probe["candidate_log"]["enabled"],
    "text_include_gold": opsd["text_include_gold"],
    "max_new_tokens": probe["max_new_tokens"],
}
print(json.dumps(payload, sort_keys=True))
"""
    env = {
        **os.environ,
        "DYME_VISUAL_CHECKER": "0",
        "DYME_VISUAL_REFINER": "0",
        "DYME_VISUAL_PREFETCH_IC": "0",
        "DYME_DEPLOT_ENABLED": "0",
        "DYME_OUTPUT_DIR": str(ROOT / "outputs/test-fast/opd-clean-config-dryrun"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    cfg = json.loads(result.stdout)

    assert cfg["providers"] == ["format_only", "visual_facts_deplot"]
    assert cfg["probe_providers"] == ["format_only", "visual_facts_deplot"]
    assert cfg["prompt_profile"] == "chartqa_short_answer"
    assert cfg["answer_parser"] == "chartqa_final_answer"
    assert cfg["skip_no_evidence"] is True
    assert cfg["candidate_log_enabled"] is True
    assert cfg["text_include_gold"] is False
    assert cfg["max_new_tokens"] == 96


def test_deplot_ablation_runner_dry_run_lists_three_aligned_variants() -> None:
    script = ROOT / "scripts/test/run_opd_deplot_ablation.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--run-id",
            "pytest",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "Variant: deplot_no_vs_opd" in out
    assert "Variant: deplot_vs_opd" in out
    assert "Variant: deplot_vs_srkl" in out
    assert out.index("Variant: deplot_vs_opd") < out.index("Variant: deplot_vs_srkl")
    assert out.index("Variant: deplot_vs_srkl") < out.index("Variant: deplot_no_vs_opd")
    assert "DYME_NUM_TRAIN_EPOCHS=4" in out
    assert "DYME_STUDENT_MODEL=/home/deepseek_VG/deepseek/models/llava-0.5b-ov" in out
    assert "DYME_TEACHER_MODEL=/home/deepseek_VG/deepseek/models/llava-7b-ov" in out
    assert "DYME_TEACHER_PROBE_MAX_NEW_TOKENS=96" in out
    assert "DYME_TEACHER_TRAJ_MAX_NEW_TOKENS=128" in out
    assert "DYME_TEACHER_PROBE_PROVIDERS=format_only,visual_facts_deplot" in out
    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot" in out
    assert "DYME_OPSD_HANG_DEBUG=0" in out
    assert "DYME_OPSD_HANG_FORCE=0" in out
    assert "TRANSFORMERS_OFFLINE=1" in out
    assert "HF_HUB_OFFLINE=1" in out
    assert "DYME_VISUAL_CHECKER=0" in out
    assert "DYME_VISUAL_CHECKER=1" in out
    assert out.count("DYME_OPSD_LOSS_TYPE=jsd") == 2
    assert "DYME_OPSD_LOSS_TYPE=srkl" in out
    assert "bash scripts/train_opd_7b_dyme_probe.sh" in out


def test_deplot_ablation_runner_rejects_one_step_smoke() -> None:
    script = ROOT / "scripts/test/run_opd_deplot_ablation.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--smoke",
            "--smoke-steps",
            "1",
            "--run-id",
            "pytest",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--smoke-steps must be >= 2" in result.stderr


def test_no_probe_ablation_config_is_cpu_dry_runnable() -> None:
    code = """
import json
from config.loader import load_config

c = load_config("opd_7b_dyme_probe")
opsd = c["opsd"]
payload = {
    "mode": opsd["mode"],
    "teacher_probe_enabled": opsd["teacher_probe"]["enabled"],
    "teacher_trajectory_enabled": opsd["teacher_trajectory"]["enabled"],
    "visual_supervision_enabled": opsd.get("visual_supervision", {}).get("enabled"),
    "text_include_gold": opsd["text_include_gold"],
    "privileged_providers": opsd["privileged_providers"],
    "max_steps": c["training"]["dyme_args"].get("max_steps"),
    "output_dir": c["training"]["dyme_args"]["output_dir"],
}
print(json.dumps(payload, sort_keys=True))
"""
    env = {
        **os.environ,
        "DYME_TEACHER_PROBE": "0",
        "DYME_TEACHER_TRAJECTORY": "0",
        "DYME_VISUAL_CHECKER": "0",
        "DYME_VISUAL_REFINER": "0",
        "DYME_VISUAL_PREFETCH_IC": "0",
        "DYME_DEPLOT_ENABLED": "0",
        "DYME_MAX_STEPS": "2",
        "DYME_OUTPUT_DIR": str(ROOT / "outputs/test-fast/opd-no-probe-smoke-dryrun"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    cfg = json.loads(result.stdout)

    assert cfg["mode"] == "dyme_teacher_probe_opd"
    assert cfg["teacher_probe_enabled"] is False
    assert cfg["teacher_trajectory_enabled"] is False
    assert cfg["visual_supervision_enabled"] is False
    assert cfg["text_include_gold"] is False
    assert cfg["max_steps"] == 2
    assert cfg["output_dir"].endswith("outputs/test-fast/opd-no-probe-smoke-dryrun")


def test_clean_no_gold_opd_log_has_expected_probe_and_health_signals() -> None:
    if not CLEAN_NO_GOLD_LOG.exists():
        pytest.skip(f"clean OPD log not found: {CLEAN_NO_GOLD_LOG}")

    rows = _metric_rows(CLEAN_NO_GOLD_LOG)
    assert len(rows) >= 500

    assert _last_mean(rows, "rewards/accuracy/mean") > 0.20
    assert _last_mean(rows, "rewards/format/mean") > 0.90
    assert _last_mean(rows, "completions/degenerate_rate") < 0.10

    candidate = _values(rows, "routing/teacher_probe_candidate_rate")
    correct = _values(rows, "routing/teacher_probe_correct_rate")
    wrong = _values(rows, "routing/teacher_probe_wrong_rate")
    opd_call = _values(rows, "routing/opd_teacher_call_rate")
    assert _mean(candidate) > 0.0
    assert _mean(wrong) > _mean(correct)
    assert abs(_mean(opd_call) - _mean(correct)) < 1e-8

    gold = _values(rows, "teacher/privileged_suffix_has_gold_rate")
    assert gold and max(gold) == 0.0
    assert not _values(rows, "visual/ic_ok_rate")


def test_old_vs_no_probe_like_log_is_not_a_clean_ablation_control() -> None:
    if not OLD_VS_NO_PROBE_LIKE_LOG.exists():
        pytest.skip(f"old OPD diagnostic log not found: {OLD_VS_NO_PROBE_LIKE_LOG}")

    rows = _metric_rows(OLD_VS_NO_PROBE_LIKE_LOG)
    visual = _values(rows, "visual/ic_ok_rate")
    candidate = _values(rows, "routing/teacher_probe_candidate_rate")
    opd_call = _values(rows, "routing/opd_teacher_call_rate")

    assert visual and min(visual) == 1.0
    assert candidate and max(candidate) == 0.0
    assert _mean(opd_call) > 0.10
