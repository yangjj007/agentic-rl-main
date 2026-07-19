from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _quality_variant_dry_run(tmp_path: Path, variant: str) -> str:
    env = {
        **os.environ,
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_PCD_RUN_ID": f"pytest_{variant}",
    }
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
    assert "DYME_CHART_COT_GATE_MODE=gate" in out
    assert "DYME_EVAL_FORMAT_REWARD=0" in out


def test_gold_hidden_clrc_restores_grpo_recovery_defaults(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision",
    )

    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot" in out
    assert "DYME_TEACHER_PROBE_PROVIDERS=format_only,visual_facts_deplot" in out
    assert "DYME_TEACHER_TRAJECTORY=0" in out
    assert "teacher correct repair: mode=opd" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" not in out
    assert "DYME_ADAPTIVE_SUPERVISION=1" in out
    assert "DYME_ADAPTIVE_READINESS_SOURCE=global_grpo_route" in out
    assert "DYME_ADAPTIVE_TARGET_READINESS=0.15" in out
    assert "DYME_ADAPTIVE_OPSD_INITIAL_CAP=4" in out
    assert "DYME_ADAPTIVE_OPSD_FINAL_CAP=1" in out
    assert "DYME_ADAPTIVE_OPSD_INITIAL_WEIGHT=1.0" in out
    assert "DYME_ADAPTIVE_OPSD_FINAL_WEIGHT=0.25" in out
    assert "DYME_ADAPTIVE_TEACHER_INITIAL_WEIGHT=0.0" in out
    assert "DYME_ADAPTIVE_TEACHER_FINAL_WEIGHT=0.0" in out
    assert "DYME_EFFECTIVE_SAMPLING=1" in out
    assert "DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT=6.0" in out
    assert "DYME_OPSD_OVERFLOW_ROUTE=mixed_grpo_all_wrong_skip" in out
    assert "DYME_GLOBAL_SIGNAL_LOGGING=1" in out
    assert "DYME_VISUAL_REFINER=0" in out
    assert "oracle gold suffix expected: 0" in out
    assert "oracle_hint" not in out


def test_gold_hidden_clrc_sft_repair_uses_stronger_answer_repair_without_oracle(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair",
    )

    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot" in out
    assert "DYME_TEACHER_PROBE_PROVIDERS=format_only,visual_facts_deplot" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=refiner_sft" in out
    assert "DYME_TEACHER_SFT_REPAIR_SCOPE=all_wrong" in out
    assert "DYME_TEACHER_SFT_REPAIR_SLOTS=4" in out
    assert "DYME_TEACHER_SFT_TARGET_MAX_TOKENS=64" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=answer_only" in out
    assert "DYME_TEACHER_SFT_TARGET_CONSTRAINT=chartqa_hint" in out
    assert "teacher probe max new tokens: 320" in out
    assert "DYME_TEACHER_PROBE_MAX_NEW_TOKENS=320" in out
    assert "DYME_TEACHER_PROBE_CANDIDATE_LOG_MAX_CHARS=1024" in out
    assert "DYME_TEACHER_TRAJECTORY=0" in out
    assert "DYME_ADAPTIVE_TARGET_READINESS=0.15" in out
    assert "DYME_ADAPTIVE_OPSD_INITIAL_CAP=2" in out
    assert "DYME_ADAPTIVE_OPSD_FINAL_CAP=1" in out
    assert "DYME_SIGNAL_AWARE_ROUTING=1" in out
    assert "DYME_DEGENERATE_HARD_OVERRIDE=1" in out
    assert "DYME_CLIPPED_HARD_OVERRIDE=1" in out
    assert "DYME_EFFECTIVE_SAMPLING=1" in out
    assert "DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT=6.0" in out
    assert "DYME_OPSD_OVERFLOW_ROUTE=mixed_grpo_all_wrong_skip" in out
    assert "DYME_VISUAL_CHECKER=0" in out
    assert "DYME_VISUAL_REFINER=1" in out
    assert "DYME_VISUAL_PREFETCH_IC=1" in out
    assert "DYME_VISUAL_LOG=1" in out
    assert "oracle gold suffix expected: 0" in out
    assert "oracle_hint" not in out


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

    assert "DYME_STUDENT_MODEL=/home/deepseek_VG/deepseek/models/llava-0.5b-ov" in out
    assert "DYME_TEACHER_MODEL=/home/deepseek_VG/deepseek/models/llava-7b-ov" in out


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
