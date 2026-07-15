from __future__ import annotations

import os
import importlib.util
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = ROOT / "config/config_opd_7b_dyme_probe.py"


def _quality_variant_dry_run(tmp_path: Path, variant: str) -> str:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": f"pytest_{variant}",
    }
    env.pop("DYME_SAVE_STRATEGY", None)
    env.pop("DYME_SAVE_STEPS", None)
    env.pop("DYME_SAVE_TOTAL_LIMIT", None)
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual.sh",
            "4",
            "--dry-run",
            "--variant",
            variant,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.replace("\\,", ",")


def test_full_cot_quality_diagnostic_variant_exports_diagnostics(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path, "deplot_no_vs_opd_pcd_oracle_hint_full_cot_quality_diagnostic"
    )

    assert "DYME_CHART_COT_VERIFY=1" in out
    assert "DYME_CHART_COT_GATE_MODE=diagnostic" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=chartqa_hint" in out
    assert "DYME_PHASE_SCHEDULE_MODE=progress" in out
    assert "DYME_EFFECTIVE_SAMPLING=1" in out
    assert "DYME_OPSD_OVERFLOW_ROUTE=mixed_grpo_all_wrong_skip" in out
    assert "DYME_EVAL_FORMAT_REWARD=0" in out


def test_full_cot_quality_gate_variant_exports_q3_gate(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path, "deplot_no_vs_opd_pcd_oracle_hint_full_cot_quality_gate"
    )

    assert "DYME_CHART_COT_VERIFY=1" in out
    assert "DYME_CHART_COT_GATE_MODE=gate" in out


def test_full_cot_adaptive_supervision_variant_removes_legacy_schedules(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path, "deplot_no_vs_opd_pcd_oracle_hint_full_cot_adaptive_supervision"
    )

    assert "DYME_ADAPTIVE_SUPERVISION=1" in out
    assert "DYME_ADAPTIVE_READINESS_SOURCE=global_grpo_route" in out
    assert "DYME_ADAPTIVE_EMA_ALPHA=0.10" in out
    assert "DYME_ADAPTIVE_TARGET_READINESS=0.30" in out
    assert "DYME_ADAPTIVE_OPSD_INITIAL_WEIGHT=1.5" in out
    assert "DYME_ADAPTIVE_OPSD_FINAL_WEIGHT=0.5" in out
    assert "DYME_ADAPTIVE_TEACHER_INITIAL_WEIGHT=0.5" in out
    assert "DYME_ADAPTIVE_TEACHER_FINAL_WEIGHT=0.0" in out
    assert "DYME_ADAPTIVE_OPSD_INITIAL_CAP=8" in out
    assert "DYME_ADAPTIVE_OPSD_FINAL_CAP=2" in out
    assert "DYME_GLOBAL_SIGNAL_LOGGING=1" in out
    assert "DYME_EFFECTIVE_SAMPLING=1" in out
    assert "DYME_EFFECTIVE_SAMPLING_AFTER_STEP=0" in out
    assert "DYME_EFFECTIVE_SAMPLING_START_PROGRESS=0.0" in out
    assert "DYME_OPSD_OVERFLOW_ROUTE=mixed_grpo_all_wrong_skip" in out
    assert "DYME_OPSD_WEIGHT_DECAY=0" in out
    assert "DYME_TEACHER_TRAJ_WEIGHT_DECAY=0" in out
    assert "DYME_DYNAMIC_TRIGGER_MONITOR=0" in out
    assert "DYME_CHART_COT_GATE_MODE=gate" in out
    assert "DYME_CHART_COT_REQUIRE_Q3=1" in out
    assert "DYME_CHART_COT_LOG_SAMPLES=1" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=chartqa_hint" in out
    assert "DYME_EVAL_FORMAT_REWARD=0" in out


def test_opd_no_hard_imitation_variant_disables_trajectory_and_sft_repair(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path, "deplot_no_vs_opd_pcd_oracle_hint_opd_no_hard_imitation_adaptive_supervision"
    )

    assert "DYME_TEACHER_TRAJECTORY=0" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=opd" not in out
    assert "DYME_OPSD_SKIP_DEGENERATE=0" in out
    assert "DYME_ADAPTIVE_SUPERVISION=1" in out
    assert "DYME_ADAPTIVE_READINESS_SOURCE=global_grpo_route" in out
    assert "DYME_EFFECTIVE_SAMPLING=1" in out
    assert "DYME_EFFECTIVE_SAMPLING_AFTER_STEP=0" in out
    assert "DYME_EFFECTIVE_SAMPLING_START_PROGRESS=0.0" in out
    assert "DYME_CHART_COT_GATE_MODE=gate" in out
    assert "DYME_EVAL_FORMAT_REWARD=0" in out


def test_opd_no_full_hint_hard_sft_variant_exports_all_invariants(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_oracle_hint_opd_no_full_hint_hard_sft_adaptive_supervision",
    )

    assert "DYME_TEACHER_TRAJECTORY=0" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=opd" not in out
    assert "DYME_DISABLE_ONLINE_SFT_SLOTS=1" in out
    assert "DYME_ONLINE_SFT_ON_ALL_WRONG=0" in out
    assert "DYME_TEACHER_PROBE_FAILURE_ROUTE=mixed_grpo_all_wrong_skip" in out
    assert "DYME_OPSD_SKIP_DEGENERATE=0" in out
    assert "DYME_ADAPTIVE_SUPERVISION=1" in out
    assert "DYME_ADAPTIVE_READINESS_SOURCE=global_grpo_route" in out
    assert "DYME_EFFECTIVE_SAMPLING=1" in out
    assert "DYME_EFFECTIVE_SAMPLING_AFTER_STEP=0" in out
    assert "DYME_EFFECTIVE_SAMPLING_START_PROGRESS=0.0" in out
    assert "DYME_CHART_COT_GATE_MODE=gate" in out
    assert "DYME_EVAL_FORMAT_REWARD=0" in out
    assert "DYME_SAVE_STRATEGY=steps" in out
    assert "DYME_SAVE_STEPS=50" in out
    assert "DYME_SAVE_TOTAL_LIMIT=3" in out


def test_gold_hidden_fixed_routed_opd_variant_is_clean_and_non_adaptive(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_fixed",
    )

    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot" in out
    assert "DYME_TEACHER_PROBE_PROMPT_PROFILE=chartqa_short_answer" in out
    assert "oracle gold suffix expected: 0" in out
    assert "DYME_TEACHER_TRAJECTORY=0" in out
    assert "DYME_DISABLE_ONLINE_SFT_SLOTS=1" in out
    assert "DYME_ONLINE_SFT_ON_ALL_WRONG=0" in out
    assert "DYME_TEACHER_PROBE_FAILURE_ROUTE=mixed_grpo_all_wrong_skip" in out
    assert "DYME_OPSD_SKIP_DEGENERATE=0" in out
    assert "DYME_OPSD_MAX_PER_PROMPT=8" in out
    assert "DYME_ADAPTIVE_SUPERVISION=0" in out
    assert "DYME_EFFECTIVE_SAMPLING_AFTER_STEP=0" in out
    assert "DYME_EFFECTIVE_SAMPLING_START_PROGRESS=0.0" in out
    assert "DYME_GLOBAL_SIGNAL_LOGGING=1" in out
    assert "DYME_SAVE_STRATEGY=steps" in out
    assert "DYME_SAVE_STEPS=50" in out
    assert "DYME_SAVE_TOTAL_LIMIT=3" in out


def test_gold_hidden_adaptive_routed_opd_only_adds_controller(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision",
    )

    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot" in out
    assert "DYME_TEACHER_PROBE_HARNESS=chartqa_closed_loop_recovery" in out
    assert "DYME_TEACHER_PROBE_HARNESS_VERSION=v12_executable_deplot" in out
    assert "DYME_TEACHER_PROBE_PROMPT_PROFILE=chartqa_deplot_operation_answer_prefix" in out
    assert "DYME_TEACHER_PROBE_PROMPT_LOG=1" in out
    assert "DYME_TEACHER_PROBE_CANDIDATE_LOG=1" in out
    assert "oracle gold suffix expected: 0" in out
    assert "DYME_TEACHER_TRAJECTORY=0" in out
    assert "DYME_DISABLE_ONLINE_SFT_SLOTS=1" in out
    assert "DYME_ONLINE_SFT_ON_ALL_WRONG=0" in out
    assert "DYME_TEACHER_PROBE_FAILURE_ROUTE=mixed_grpo_all_wrong_skip" in out
    assert "DYME_OPSD_SKIP_DEGENERATE=0" in out
    assert "DYME_ADAPTIVE_SUPERVISION=1" in out
    assert "DYME_ADAPTIVE_READINESS_SOURCE=global_grpo_route" in out
    assert "DYME_ADAPTIVE_TARGET_READINESS=0.30" in out
    assert "DYME_ADAPTIVE_OPSD_INITIAL_CAP=8" in out
    assert "DYME_ADAPTIVE_OPSD_FINAL_CAP=2" in out
    assert "DYME_GLOBAL_SIGNAL_LOGGING=1" in out
    assert "DYME_SAVE_STRATEGY=steps" in out
    assert "DYME_SAVE_STEPS=50" in out
    assert "DYME_SAVE_TOTAL_LIMIT=3" in out


def test_gold_hidden_no_opd_variant_disables_opd_and_hard_sft(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_no_opd",
    )

    assert "DYME_OPSD_MODE=dyme_teacher_probe_opd" in out
    assert "DYME_OPSD_WEIGHT=0.0" in out
    assert "DYME_GRPO_WEIGHT=1.0" in out
    assert "DYME_TEACHER_PROBE=1" in out
    assert "DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP=0" in out
    assert "DYME_OPSD_MAX_PER_PROMPT=8" in out
    assert "DYME_TEACHER_TRAJECTORY=0" in out
    assert "DYME_DISABLE_ONLINE_SFT_SLOTS=1" in out
    assert "DYME_ONLINE_SFT_ON_ALL_WRONG=0" in out
    assert "oracle gold suffix expected: 0" in out


def test_gold_hidden_grpo_only_variant_removes_teacher_probe_and_opd(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_grpo_only",
    )

    assert "DYME_OPSD_MODE=dyme" in out
    assert "DYME_OPSD_WEIGHT=0.0" in out
    assert "DYME_GRPO_WEIGHT=1.0" in out
    assert "DYME_TEACHER_MODEL= " in out
    assert "DYME_TEACHER_PROBE=0" in out
    assert "DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP=-1" in out
    assert "DYME_OPSD_MAX_PER_PROMPT=0" in out
    assert "DYME_DISABLE_ONLINE_SFT_SLOTS=1" in out
    assert "DYME_ONLINE_SFT_ON_ALL_WRONG=0" in out


def test_gold_hidden_unconditional_opd_skips_teacher_verifier(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_uncond_opd_no_full_hint_hard_sft",
    )

    assert "DYME_OPSD_MODE=dyme_teacher_probe_opd" in out
    assert "DYME_TEACHER_PROBE=0" in out
    assert "DYME_OPSD_WEIGHT=1.5" in out
    assert "DYME_TEACHER_TRAJECTORY=0" in out
    assert "DYME_DISABLE_ONLINE_SFT_SLOTS=1" in out
    assert "DYME_OPSD_MAX_PER_PROMPT=8" in out
    assert "DYME_ADAPTIVE_SUPERVISION=0" in out


def test_gold_hidden_target020_changes_only_controller_target(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_target020",
    )

    assert "DYME_ADAPTIVE_SUPERVISION=1" in out
    assert "DYME_ADAPTIVE_READINESS_SOURCE=global_grpo_route" in out
    assert "DYME_ADAPTIVE_TARGET_READINESS=0.20" in out
    assert "DYME_DISABLE_ONLINE_SFT_SLOTS=1" in out
    assert "DYME_TEACHER_TRAJECTORY=0" in out


def test_gold_hidden_opd_only_variant_zeros_grpo_weight(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_opd_only_no_full_hint_hard_sft",
    )

    assert "DYME_OPSD_MODE=dyme_teacher_probe_opd" in out
    assert "DYME_OPSD_WEIGHT=1.5" in out
    assert "DYME_GRPO_WEIGHT=0.0" in out
    assert "DYME_TEACHER_PROBE=1" in out
    assert "DYME_DISABLE_ONLINE_SFT_SLOTS=1" in out
    assert "DYME_ONLINE_SFT_ON_ALL_WRONG=0" in out


def test_gold_hidden_fallback_only_variant_exports_requested_zero_weights(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_fallback_only",
    )

    assert "DYME_OPSD_MODE=dyme" in out
    assert "DYME_TEACHER_MODEL= " in out
    assert "DYME_TEACHER_PROBE=0" in out
    assert "DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP=-1" in out
    assert "DYME_OPSD_WEIGHT=0.0" in out
    assert "DYME_GRPO_WEIGHT=0.0" in out
    assert "DYME_DISABLE_ONLINE_SFT_SLOTS=0" in out
    assert "DYME_ONLINE_SFT_ON_ALL_WRONG=1" in out


def test_empty_teacher_model_env_disables_teacher_loading(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DYME_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("DYME_TEACHER_MODEL", "")

    spec = importlib.util.spec_from_file_location(
        "config_opd_7b_dyme_probe_no_teacher", PROBE_CONFIG
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.CONFIG["model"]["teacher_model_path"] == ""


def test_gold_hidden_token_reliability_variant_enables_token_weighting(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_token_reliability_clrc",
    )

    assert "DYME_ADAPTIVE_SUPERVISION=1" in out
    assert "DYME_OPSD_TOKEN_WEIGHTING=1" in out
    assert "DYME_OPSD_TOKEN_NUMERIC_WEIGHT=2.0" in out
    assert "DYME_OPSD_TOKEN_ANSWER_WEIGHT=1.5" in out
    assert "DYME_OPSD_TOKEN_MIN_WEIGHT=0.75" in out


def test_gold_hidden_answer_anchor_variant_downweights_rationale_tokens(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_answer_anchor_clrc",
    )

    assert "DYME_ADAPTIVE_SUPERVISION=1" in out
    assert "DYME_TEACHER_PROBE_HARNESS=chartqa_closed_loop_recovery" in out
    assert "DYME_TEACHER_PROBE_PROMPT_PROFILE=chartqa_deplot_operation_answer_prefix" in out
    assert "DYME_TEACHER_PROBE_PROMPT_LOG=1" in out
    assert "DYME_ADAPTIVE_READINESS_SOURCE=global_grpo_route" in out
    assert "DYME_ADAPTIVE_TARGET_READINESS=0.20" in out
    assert "DYME_OPSD_TOKEN_WEIGHTING=1" in out
    assert "DYME_OPSD_TOKEN_WEIGHTING_MODE=answer_anchor" in out
    assert "DYME_OPSD_TOKEN_NUMERIC_WEIGHT=3.0" in out
    assert "DYME_OPSD_TOKEN_ANSWER_WEIGHT=2.0" in out
    assert "DYME_OPSD_TOKEN_MIN_WEIGHT=0.05" in out


def test_gold_hidden_confidence_weighted_variant_uses_strict_teacher_probe(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_confidence_weighted_clrc",
    )

    assert "DYME_ADAPTIVE_SUPERVISION=1" in out
    assert "DYME_TEACHER_PROBE_HARNESS=chartqa_closed_loop_recovery" in out
    assert "DYME_TEACHER_PROBE_PROMPT_PROFILE=chartqa_deplot_operation_answer_prefix" in out
    assert "DYME_TEACHER_PROBE_PROMPT_LOG=1" in out
    assert "DYME_TEACHER_PROBE_STRICT_ACCEPT=1" in out
    assert "DYME_TEACHER_PROBE_REQUIRE_ANSWER_FLAG=1" in out
    assert "DYME_TEACHER_PROBE_REJECT_PARSE_FAIL=1" in out
    assert "DYME_TEACHER_PROBE_REJECT_CLIPPED=1" in out
    assert "DYME_TEACHER_PROBE_RELAXED_TOL=0.0" in out
    assert "DYME_TEACHER_PROBE_MAX_NEW_TOKENS=64" in out


def test_gold_hidden_grpo_recovery_boost_variant_prioritizes_grpo_recovery(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_grpo_recovery_boost_clrc",
    )

    assert "DYME_ADAPTIVE_SUPERVISION=1" in out
    assert "DYME_ADAPTIVE_TARGET_READINESS=0.15" in out
    assert "DYME_ADAPTIVE_OPSD_INITIAL_WEIGHT=1.0" in out
    assert "DYME_ADAPTIVE_OPSD_FINAL_WEIGHT=0.25" in out
    assert "DYME_ADAPTIVE_OPSD_INITIAL_CAP=4" in out
    assert "DYME_ADAPTIVE_OPSD_FINAL_CAP=1" in out
    assert "DYME_OPSD_OVERFLOW_ROUTE=mixed_grpo_all_wrong_skip" in out
    assert "DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT=6.0" in out


def test_gold_hidden_evidence_adaptive_variant_requires_high_quality_evidence(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_evidence_adaptive_clrc",
    )

    assert "DYME_ADAPTIVE_SUPERVISION=1" in out
    assert "DYME_TEACHER_PROBE_PROVIDERS=format_only,visual_facts_deplot" in out
    assert "DYME_TEACHER_PROBE_HARNESS=chartqa_closed_loop_recovery" in out
    assert "DYME_TEACHER_PROBE_PROMPT_PROFILE=chartqa_deplot_operation_answer_prefix" in out
    assert "DYME_TEACHER_PROBE_PROMPT_LOG=1" in out
    assert "DYME_TEACHER_PROBE_SKIP_NO_EVIDENCE=1" in out
    assert "DYME_CHART_COT_VERIFY=1" in out
    assert "DYME_CHART_COT_GATE_MODE=gate" in out
    assert "DYME_OPSD_TOKEN_WEIGHTING=1" in out
    assert "DYME_OPSD_TOKEN_WEIGHTING_MODE=answer_anchor" in out


def test_gold_hidden_mixed_group_hard_replay_variant_is_honestly_isolated(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_mixed_group_shortest_correct_hard_replay",
    )

    assert "DYME_MIXED_GROUP_HARD_REPLAY=1" in out
    assert "DYME_SSOPD_MIXED_GROUP" not in out
    assert "DYME_TEACHER_PROBE=0" in out
    assert "DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP=-1" in out
    assert "DYME_OPSD_WEIGHT=0.0" in out
    assert "DYME_GRPO_WEIGHT=1.0" in out
    assert "DYME_DISABLE_ONLINE_SFT_SLOTS=1" in out
    assert "DYME_ONLINE_SFT_ON_ALL_WRONG=0" in out


def test_pcd_runner_rejects_retired_near_neighbor_variants(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_retired_near_neighbor",
    }

    for retired_variant in (
        "deplot_no_vs_opd_pcd_gold_hidden_vold_cold_start",
        "deplot_no_vs_opd_pcd_gold_hidden_ssopd_mixed_group",
    ):
        result = subprocess.run(
            [
                "bash",
                "scripts/test/run_pcd_no_visual.sh",
                "4",
                "--dry-run",
                "--variant",
                retired_variant,
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        assert result.returncode == 2
        assert "retired near-neighbor variant" in result.stderr


def test_legacy_variant_keeps_epoch_checkpoint_default(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(tmp_path, "deplot_no_vs_opd_pcd")

    assert "DYME_SAVE_STRATEGY=epoch" in out
    assert "DYME_SAVE_STEPS=" not in out
    assert "DYME_SAVE_TOTAL_LIMIT=" not in out


def test_pcd_no_visual_dry_run_defaults_to_local_model_paths(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_default_models",
    }
    result = subprocess.run(
        ["bash", "scripts/test/run_pcd_no_visual.sh", "4", "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert f"DYME_STUDENT_MODEL={ROOT}/models/llava-0.5b-ov" in out
    assert f"DYME_TEACHER_MODEL={ROOT}/models/llava-7b-ov" in out


def test_pcd_no_visual_dry_run_canonical_speed_profile(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_canonical_speed",
    }
    result = subprocess.run(
        ["bash", "scripts/test/run_pcd_no_visual.sh", "4", "--dry-run", "--speed-profile", "canonical"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "speed profile: canonical" in out
    assert "DYME_PERF_TIMING=1" not in out
    assert "DYME_TEACHER_PROBE_BATCH_SIZE=1" in out
    assert "DYME_TEACHER_PROBE_MAX_PER_BATCH=0" in out
    assert "DYME_TEACHER_PROBE_CANDIDATE_LOG_MAX_CHARS=256" in out
    assert "DYME_TEACHER_TRAJECTORY=1" in out
    assert "DYME_TEACHER_PROBE_CANDIDATE_LOG=1" in out
    assert "DYME_VISUAL_PREFETCH_IC=0" in out
    assert "DYME_VISUAL_LOG_SAMPLES=1" in out
    assert "DYME_ONLINE_SFT_TARGET" not in out


def test_pcd_no_visual_dry_run_fast60_speed_profile(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_fast60_speed",
    }
    result = subprocess.run(
        ["bash", "scripts/test/run_pcd_no_visual.sh", "4", "--dry-run", "--speed-profile", "fast60"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "speed profile: fast60" in out
    assert "DYME_PERF_TIMING=1" not in out
    assert "DYME_TEACHER_PROBE_BATCH_SIZE=8" in out
    assert "DYME_TEACHER_PROBE_MAX_PER_BATCH=16" in out
    assert "DYME_TEACHER_TRAJECTORY=0" in out
    assert "DYME_TEACHER_PROBE_CANDIDATE_LOG=0" in out
    assert "DYME_ONLINE_SFT_TARGET" not in out


def test_pcd_no_visual_dry_run_oracle_hint_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_oracle_hint",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint" in out
    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot,oracle_hint" in out
    assert "DYME_TEACHER_PROBE_PROVIDERS=format_only,visual_facts_deplot,oracle_hint" in out
    assert "DYME_TEACHER_PROBE_PROMPT_PROFILE=chartqa_oracle_hint" in out
    assert "DYME_TEACHER_PROBE_MAX_NEW_TOKENS=500" in out
    assert "DYME_TEACHER_TRAJ_MAX_NEW_TOKENS=500" in out
    assert "oracle gold suffix expected: 1" in out


def test_pcd_no_visual_dry_run_oracle_eval_format_reward_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_eval_format_reward",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward" in out
    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot,oracle_hint" in out
    assert "DYME_EVAL_FORMAT_REWARD=1" in out
    assert "DYME_EVAL_FORMAT_REWARD_WEIGHT=0.1" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" not in out


def test_pcd_no_visual_dry_run_oracle_late_traj_decay_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_late_traj_decay",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay" in out
    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot,oracle_hint" in out
    assert "DYME_TEACHER_TRAJ_WEIGHT_DECAY=1" in out
    assert "DYME_TEACHER_TRAJ_DECAY_START_STEP=294" in out
    assert "DYME_TEACHER_TRAJ_DECAY_END_STEP=441" in out
    assert "DYME_TEACHER_TRAJ_FINAL_WEIGHT=0.0" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" not in out


def test_pcd_no_visual_dry_run_oracle_eval_format_late_traj_decay_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_eval_format_late_traj_decay",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay" in out
    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot,oracle_hint" in out
    assert "DYME_EVAL_FORMAT_REWARD=1" in out
    assert "DYME_EVAL_FORMAT_REWARD_WEIGHT=0.1" in out
    assert "DYME_TEACHER_TRAJ_WEIGHT_DECAY=1" in out
    assert "DYME_TEACHER_TRAJ_DECAY_START_STEP=294" in out
    assert "DYME_TEACHER_TRAJ_DECAY_END_STEP=441" in out
    assert "DYME_TEACHER_TRAJ_FINAL_WEIGHT=0.0" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" not in out


def test_pcd_no_visual_dry_run_route_guard_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_route_guard",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_route_guard",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_route_guard" in out
    assert "DYME_SIGNAL_AWARE_ROUTING=1" in out
    assert "DYME_DEGENERATE_HARD_OVERRIDE=1" in out
    assert "DYME_CLIPPED_HARD_OVERRIDE=1" in out
    assert "DYME_PERCEPTION_REWARD=0" in out
    assert "DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP=0" in out
    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot" in out


def test_pcd_no_visual_dry_run_oracle_teacher_sft_repair_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_teacher_sft_repair",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair" in out
    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot,oracle_hint" in out
    assert "DYME_TEACHER_PROBE_PROMPT_PROFILE=chartqa_oracle_hint" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" in out
    assert "DYME_TEACHER_SFT_REPAIR_SCOPE=all_wrong" in out
    assert "DYME_TEACHER_SFT_REPAIR_SLOTS=1" in out
    assert "DYME_TEACHER_SFT_TARGET_MAX_TOKENS=256" in out
    assert "DYME_TEACHER_SFT_SANITIZE_PRIVILEGED=1" in out
    assert "DYME_TEACHER_SFT_TARGET_CONSTRAINT=chartqa_hint" in out
    assert "DYME_SIGNAL_AWARE_ROUTING=1" not in out
    assert "DYME_PERCEPTION_REWARD=1" not in out


def test_pcd_no_visual_dry_run_oracle_teacher_sft_repair_student_style_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_teacher_sft_repair_student_style",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_style",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_style" in out
    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot,oracle_hint" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=student_short" in out


def test_pcd_no_visual_dry_run_oracle_teacher_sft_repair_student_hint_short_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_teacher_sft_repair_student_hint_short",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short" in out
    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot,oracle_hint" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=student_hint_short" in out


def test_pcd_no_visual_dry_run_student_hint_short_opd_decay_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_student_hint_opd_decay",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay" in out
    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot,oracle_hint" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=student_hint_short" in out
    assert "DYME_OPSD_WEIGHT_DECAY=1" in out
    assert "DYME_OPSD_DECAY_START_STEP=294" in out
    assert "DYME_OPSD_DECAY_END_STEP=441" in out
    assert "DYME_OPSD_FINAL_WEIGHT=0.5" in out
    assert "DYME_OPSD_MAX_PER_PROMPT_AFTER_STEP=294" in out
    assert "DYME_OPSD_MAX_PER_PROMPT=2" in out
    assert "DYME_TEACHER_TRAJ_WEIGHT_DECAY=1" in out
    assert "DYME_SIGNAL_AWARE_ROUTING=1" not in out


def test_pcd_no_visual_dry_run_student_hint_short_opd_decay_effective_sampling_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_student_hint_opd_decay_sampling",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=student_hint_short" in out
    assert "DYME_OPSD_WEIGHT_DECAY=1" in out
    assert "DYME_OPSD_MAX_PER_PROMPT=2" in out
    assert "DYME_EFFECTIVE_SAMPLING=1" in out
    assert "DYME_EFFECTIVE_SAMPLING_AFTER_STEP=294" in out
    assert "DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT=4.0" in out
    assert "DYME_EFFECTIVE_SAMPLING_ALL_WRONG_WEIGHT=1.0" in out
    assert "DYME_EFFECTIVE_SAMPLING_ALL_CORRECT_WEIGHT=0.7" in out


def test_pcd_no_visual_dry_run_first_run_eval_format_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_first_run_eval_format",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_eval_format",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_eval_format" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=student_hint_short" in out
    assert "DYME_OPSD_WEIGHT_DECAY=1" in out
    assert "DYME_OPSD_MAX_PER_PROMPT=2" in out
    assert "DYME_EFFECTIVE_SAMPLING=1" in out
    assert "DYME_EVAL_FORMAT_REWARD=1" in out
    assert "DYME_EVAL_FORMAT_REWARD_WEIGHT=0.2" in out


def test_pcd_no_visual_dry_run_progress_grpo_overflow_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_progress_grpo_overflow",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_grpo_overflow",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "phase schedule: mode=progress" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=student_hint_short" in out
    assert "DYME_PHASE_SCHEDULE_MODE=progress" in out
    assert "DYME_TEACHER_TRAJ_DECAY_START_PROGRESS=0.25" in out
    assert "DYME_TEACHER_TRAJ_DECAY_END_PROGRESS=0.50" in out
    assert "DYME_OPSD_DECAY_START_PROGRESS=0.50" in out
    assert "DYME_OPSD_DECAY_END_PROGRESS=0.75" in out
    assert "DYME_EFFECTIVE_SAMPLING_START_PROGRESS=0.50" in out
    assert "DYME_OPSD_ROUTE_CAP_START_PROGRESS=0.50" in out
    assert "DYME_OPSD_OVERFLOW_ROUTE=mixed_grpo_all_wrong_skip" in out
    assert "DYME_OPSD_MAX_PER_PROMPT=2" in out
    assert "DYME_EFFECTIVE_SAMPLING=1" in out
    assert "DYME_DYNAMIC_TRIGGER_MONITOR=1" in out
    assert "DYME_DYNAMIC_TRIGGER_SAMPLING_ZERO_MIN=0.70" in out
    assert "DYME_DYNAMIC_TRIGGER_RL_MIXED_MIN=0.30" in out
    assert "DYME_EVAL_FORMAT_REWARD=0" in out
    assert "DYME_POSITIVE_REPLAY=0" in out
    assert "DYME_ROLLOUT_REPLAY=0" in out


def test_pcd_no_visual_dry_run_student_hint_short_opd_decay_sampling_replay_mix_variant(tmp_path: Path) -> None:
    replay_dataset = tmp_path / "replay_train.json"
    replay_dataset.write_text("[]", encoding="utf-8")
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_student_hint_opd_decay_sampling_replay",
        "DYME_POSITIVE_REPLAY_DATASET": str(replay_dataset),
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_replay_mix",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_replay_mix" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=student_hint_short" in out
    assert "DYME_EFFECTIVE_SAMPLING=1" in out
    assert "DYME_POSITIVE_REPLAY=1" in out
    assert f"DYME_POSITIVE_REPLAY_DATASET={replay_dataset}" in out
    assert "DYME_POSITIVE_REPLAY_WEIGHT=0.1" in out
    assert "DYME_POSITIVE_REPLAY_BATCH_SIZE=1" in out
    assert "DYME_POSITIVE_REPLAY_AFTER_STEP=0" in out


def test_pcd_no_visual_dry_run_student_hint_short_opd_decay_sampling_rollout_replay_variant(tmp_path: Path) -> None:
    replay_dataset = tmp_path / "replay_train.json"
    replay_dataset.write_text("[]", encoding="utf-8")
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_student_hint_rollout_replay",
        "DYME_POSITIVE_REPLAY_DATASET": str(replay_dataset),
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay" in out
    assert "DYME_POSITIVE_REPLAY=1" in out
    assert "DYME_ROLLOUT_REPLAY=1" in out
    assert "DYME_ROLLOUT_REPLAY_WEIGHT=0.05" in out
    assert "DYME_ROLLOUT_REPLAY_BATCH_SIZE=2" in out
    assert "DYME_ROLLOUT_REPLAY_AFTER_STEP=50" in out
    assert "DYME_ROLLOUT_REPLAY_CAPACITY=256" in out


def test_pcd_no_visual_dry_run_student_hint_short_opd_decay_sampling_filter_variant(tmp_path: Path) -> None:
    replay_dataset = tmp_path / "replay_train.json"
    replay_dataset.write_text("[]", encoding="utf-8")
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_student_hint_filter",
        "DYME_POSITIVE_REPLAY_DATASET": str(replay_dataset),
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter" in out
    assert "DYME_ROLLOUT_REPLAY=1" in out
    assert "DYME_EFFECTIVE_GROUP_FILTER=1" in out
    assert "DYME_EFFECTIVE_GROUP_FILTER_AFTER_STEP=294" in out
    assert "DYME_EFFECTIVE_GROUP_FILTER_ALL_WRONG_KEEP=1" in out
    assert "DYME_EFFECTIVE_GROUP_FILTER_ALL_CORRECT=1" in out


def test_pcd_no_visual_dry_run_student_hint_short_rl_transition_variant(tmp_path: Path) -> None:
    replay_dataset = tmp_path / "replay_train.json"
    replay_dataset.write_text("[]", encoding="utf-8")
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_student_hint_rl_transition",
        "DYME_POSITIVE_REPLAY_DATASET": str(replay_dataset),
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert (
        "variant: "
        "deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition"
    ) in out
    assert "DYME_POSITIVE_REPLAY=0" in out
    assert "DYME_POSITIVE_REPLAY_WEIGHT=0.0" in out
    assert "DYME_POSITIVE_REPLAY_BATCH_SIZE=0" in out
    assert "DYME_POSITIVE_REPLAY_UNTIL_STEP=0" in out
    assert "DYME_OPSD_FINAL_WEIGHT=0.0" in out
    assert "DYME_EFFECTIVE_GROUP_FILTER=1" in out
    assert "DYME_EFFECTIVE_GROUP_FILTER_AFTER_STEP=294" in out
    assert "DYME_EFFECTIVE_GROUP_FILTER_ALL_WRONG_KEEP=0" in out
    assert "DYME_EFFECTIVE_GROUP_FILTER_ALL_CORRECT=1" in out


def test_pcd_no_visual_dry_run_oracle_teacher_sft_repair_answer_only_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_teacher_sft_repair_answer_only",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_answer_only",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_answer_only" in out
    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot,oracle_hint" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=answer_only" in out


def test_pcd_no_visual_dry_run_route_guard_perception_hint_variant(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_route_guard_perception",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_route_guard_perception_hint",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout.replace("\\,", ",")

    assert "variant: deplot_no_vs_opd_pcd_route_guard_perception_hint" in out
    assert "DYME_SIGNAL_AWARE_ROUTING=1" in out
    assert "DYME_PERCEPTION_REWARD=1" in out
    assert "DYME_PERCEPTION_REWARD_SOURCE=trusted_hint" in out
    assert "DYME_PERCEPTION_REWARD_WEIGHT=0.2" in out
    assert "DYME_PERCEPTION_REWARD_SOURCE=visual_fact_deplot" not in out


def test_pcd_no_visual_dry_run_baseline_does_not_enable_route_guard(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_baseline_no_route_guard",
    }
    result = subprocess.run(
        ["bash", "scripts/test/run_pcd_no_visual_4epoch.sh", "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "DYME_SIGNAL_AWARE_ROUTING=1" not in out
    assert "DYME_DEGENERATE_HARD_OVERRIDE=1" not in out
    assert "DYME_PERCEPTION_REWARD=1" not in out


def test_pcd_no_visual_dry_run_can_set_max_steps_for_smoke(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_route_guard_10step",
        "DYME_PCD_MAX_STEPS": "10",
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/test/run_pcd_no_visual_4epoch.sh",
            "--dry-run",
            "--variant",
            "deplot_no_vs_opd_pcd_route_guard",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "max steps override: 10" in out
    assert "DYME_TRAIN_MAX_STEPS=10" in out
    assert "-u DYME_TRAIN_MAX_STEPS" not in out


def test_pcd_no_visual_rejects_legacy_oracle_variant_names(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": "pytest_oracle_hint_legacy",
    }

    for variant in (
        "deplot_no_vs_opd_pcd_oracle_hint_v3",
        "deplot_no_vs_opd_pcd_oracle_hint_v4",
    ):
        result = subprocess.run(
            [
                "bash",
                "scripts/test/run_pcd_no_visual_4epoch.sh",
                "--dry-run",
                "--variant",
                variant,
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 2
        assert f"Unknown PCD variant: {variant}" in result.stderr


def test_pcd_no_visual_4epoch_dry_run_matches_deplot_pcd_ablation_env(tmp_path: Path) -> None:
    pcd_env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "pcd_out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "pcd_logs"),
        "DYME_PCD_RUN_ID": "pytest_pcd_align",
    }
    ablation_env = {
        **os.environ,
        "DYME_DEPLOT_ABLATION_OUTPUT_ROOT": str(tmp_path / "ablation_out"),
        "DYME_DEPLOT_ABLATION_LOG_ROOT": str(tmp_path / "ablation_logs"),
    }
    pcd = subprocess.run(
        ["bash", "scripts/test/run_pcd_no_visual_4epoch.sh", "--dry-run"],
        cwd=ROOT,
        env=pcd_env,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.replace("\\,", ",")
    ablation = subprocess.run(
        [
            "bash",
            "scripts/test/run_opd_deplot_ablation.sh",
            "--dry-run",
            "--run-id",
            "pytest_pcd_align",
            "--variants",
            "deplot_no_vs_opd_pcd",
        ],
        cwd=ROOT,
        env=ablation_env,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    aligned_tokens = [
        "DYME_NUM_TRAIN_EPOCHS=4",
        "DYME_FAST_NUM_TRAIN_EPOCHS=4",
        "DYME_OPSD_PRIVILEGE_PROFILE=text",
        "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot",
        "DYME_TEACHER_PROBE_PROVIDERS=format_only,visual_facts_deplot",
        "DYME_TEACHER_PROBE=1",
        "DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP=0",
        "DYME_TEACHER_PROBE_BATCH_SIZE=1",
        "DYME_TEACHER_PROBE_MAX_PER_BATCH=0",
        "DYME_TEACHER_PROBE_MAX_NEW_TOKENS=96",
        "DYME_TEACHER_PROBE_CANDIDATE_LOG=1",
        "DYME_TEACHER_PROBE_CANDIDATE_LOG_MAX_CHARS=256",
        "DYME_TEACHER_TRAJECTORY=1",
        "DYME_TEACHER_TRAJ_MAX_NEW_TOKENS=128",
        "DYME_OPSD_LOSS_TYPE=jsd",
        "DYME_OPSD_WEIGHT=1.5",
        "DYME_OPSD_VARIANCE_ADAPTIVE=0",
        "DYME_OPSD_ADAPTIVE_STD_TARGET=0.25",
        "DYME_OPSD_ADAPTIVE_MAX_MULT=2.0",
        "DYME_GRPO_WEIGHT=1.0",
        "DYME_OPSD_SRKL_ALPHA=0.1",
        "DYME_VISUAL_CHECKER=0",
        "DYME_VISUAL_REFINER=0",
        "DYME_VISUAL_PREFETCH_IC=0",
        "DYME_VISUAL_LOG=0",
        "DYME_VISUAL_SAVE_ARTIFACTS=0",
        "DYME_VISUAL_LOG_SAMPLES=1",
        "DYME_DEPLOT_ENABLED=0",
        "DYME_OPSD_HANG_DEBUG=0",
        "DYME_OPSD_HANG_FORCE=0",
        "DYME_OPSD_DETAIL_EVERY=0",
        "TRANSFORMERS_OFFLINE=1",
        "HF_HUB_OFFLINE=1",
        "WANDB_MODE=disabled",
    ]
    for token in aligned_tokens:
        assert token in pcd
        assert token in ablation
