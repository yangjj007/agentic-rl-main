# Main5 CLRC SFT Repair Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the lost main5 gold-hidden CLRC OPD recovery runner contract and add a stronger short teacher-SFT repair route.

**Status:** Implemented and smoke-tested locally.

**Architecture:** Keep `scripts/test/run_pcd_no_visual.sh` as the single source of variant wiring. Add two gold-hidden variants to the existing bash runner and lock their dry-run env contract in `tests/test_pcd_no_visual_runner.py`.

**Tech Stack:** Bash runner scripts, pytest dry-run tests, existing DyME teacher-SFT repair controls.

---

### Task 1: Lock The Restored CLRC Contract

**Files:**
- Modify: `tests/test_pcd_no_visual_runner.py`

- [x] **Step 1: Write failing tests**

Add two dry-run tests:

```python
def test_gold_hidden_clrc_restores_grpo_recovery_defaults(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision",
    )

    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot" in out
    assert "DYME_TEACHER_TRAJECTORY=0" in out
    assert "teacher correct repair: mode=opd" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" not in out
    assert "DYME_ADAPTIVE_TARGET_READINESS=0.15" in out
    assert "DYME_ADAPTIVE_OPSD_INITIAL_CAP=4" in out
    assert "DYME_ADAPTIVE_OPSD_FINAL_CAP=1" in out
    assert "DYME_ADAPTIVE_OPSD_INITIAL_WEIGHT=1.0" in out
    assert "DYME_ADAPTIVE_OPSD_FINAL_WEIGHT=0.25" in out
    assert "DYME_EFFECTIVE_SAMPLING=1" in out
    assert "DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT=6.0" in out
    assert "DYME_OPSD_OVERFLOW_ROUTE=mixed_grpo_all_wrong_skip" in out
    assert "DYME_VISUAL_REFINER=0" in out
    assert "oracle gold suffix expected: 0" in out
```

```python
def test_gold_hidden_clrc_sft_repair_uses_short_repair_without_oracle(tmp_path: Path) -> None:
    out = _quality_variant_dry_run(
        tmp_path,
        "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair",
    )

    assert "DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot" in out
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft" in out
    assert "DYME_TEACHER_SFT_REPAIR_SCOPE=all_wrong" in out
    assert "DYME_TEACHER_SFT_REPAIR_SLOTS=1" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=student_hint_short" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=answer_only" not in out
    assert "DYME_ADAPTIVE_TARGET_READINESS=0.15" in out
    assert "DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT=6.0" in out
    assert "oracle gold suffix expected: 0" in out
    assert "oracle_hint" not in out
```

- [x] **Step 2: Verify red**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest \
  tests/test_pcd_no_visual_runner.py::test_gold_hidden_clrc_restores_grpo_recovery_defaults \
  tests/test_pcd_no_visual_runner.py::test_gold_hidden_clrc_sft_repair_uses_short_repair_without_oracle \
  -q
```

Expected: fail because both variants are not accepted by the runner yet.

### Task 2: Restore Runner Variants

**Files:**
- Modify: `scripts/test/run_pcd_no_visual.sh`

- [x] **Step 1: Add variants to usage and validation**

Add these names to the examples and variant allowlist:

```bash
deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision
deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair
```

- [x] **Step 2: Add shared gold-hidden CLRC block**

Add a block that applies to both variants:

```bash
TEACHER_PROVIDERS="format_only,visual_facts_deplot"
TEACHER_PROBE_PROMPT_PROFILE="${DYME_TEACHER_PROBE_PROMPT_PROFILE:-chartqa_short_answer}"
TEACHER_PROBE_MAX_NEW_TOKENS="${DYME_TEACHER_PROBE_MAX_NEW_TOKENS:-96}"
TEACHER_TRAJECTORY=0
TEACHER_CORRECT_REPAIR_MODE=opd
OPSD_SKIP_DEGENERATE=0
EFFECTIVE_SAMPLING="${DYME_EFFECTIVE_SAMPLING:-1}"
EFFECTIVE_SAMPLING_MIXED_WEIGHT="${DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT:-6.0}"
OPSD_OVERFLOW_ROUTE="${DYME_OPSD_OVERFLOW_ROUTE:-mixed_grpo_all_wrong_skip}"
OPSD_WEIGHT_DECAY=0
TEACHER_TRAJ_WEIGHT_DECAY=0
OPSD_MAX_PER_PROMPT=0
DYNAMIC_TRIGGER_MONITOR=0
EVAL_FORMAT_REWARD=0
CHART_COT_VERIFY=1
CHART_COT_GATE_MODE=gate
ADAPTIVE_SUPERVISION=1
ADAPTIVE_READINESS_SOURCE=global_grpo_route
ADAPTIVE_TARGET_READINESS=0.15
ADAPTIVE_OPSD_INITIAL_WEIGHT="${DYME_ADAPTIVE_OPSD_INITIAL_WEIGHT:-1.0}"
ADAPTIVE_OPSD_FINAL_WEIGHT="${DYME_ADAPTIVE_OPSD_FINAL_WEIGHT:-0.25}"
ADAPTIVE_OPSD_INITIAL_CAP="${DYME_ADAPTIVE_OPSD_INITIAL_CAP:-4}"
ADAPTIVE_OPSD_FINAL_CAP="${DYME_ADAPTIVE_OPSD_FINAL_CAP:-1}"
ADAPTIVE_TEACHER_INITIAL_WEIGHT=0.0
ADAPTIVE_TEACHER_FINAL_WEIGHT=0.0
GLOBAL_SIGNAL_LOGGING=1
ORACLE_GOLD_SUFFIX_EXPECTED=0
```

- [x] **Step 3: Add SFT repair override**

For the `_sft_repair` variant, override:

```bash
TEACHER_CORRECT_REPAIR_MODE="${DYME_TEACHER_CORRECT_REPAIR_MODE:-traj_sft}"
TEACHER_SFT_TARGET_STYLE="${DYME_TEACHER_SFT_TARGET_STYLE:-student_hint_short}"
TEACHER_SFT_REPAIR_SCOPE="${DYME_TEACHER_SFT_REPAIR_SCOPE:-all_wrong}"
TEACHER_SFT_REPAIR_SLOTS="${DYME_TEACHER_SFT_REPAIR_SLOTS:-1}"
```

- [x] **Step 4: Verify green**

Run the same focused pytest command. Expected: pass.

### Task 3: Smoke And Sync

**Files:**
- No code files beyond the runner and tests.

- [x] **Step 1: Run broader checks**

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/test_pcd_no_visual_runner.py -q
bash -n scripts/test/run_pcd_no_visual.sh
bash -n scripts/test/run_pcd_no_visual_10epoch.sh
```

- [x] **Step 2: Run smoke dry-run**

```bash
DYME_PCD_RUN_ID=smoke_main5_sft_route \
DYME_PCD_OUTPUT_ROOT=outputs/test-fast/main5-smoke/sft_route \
DYME_PCD_LOG_ROOT=outputs/test-fast/logs/main5-smoke/sft_route \
DYME_PCD_MAX_STEPS=2 \
bash scripts/test/run_pcd_no_visual.sh 10 --resume none --dry-run \
  --variant deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair
```

Expected: output contains the SFT repair envs and adaptive recovery defaults; no GPU training starts.

- [ ] **Step 3: Commit and push**

```bash
git add docs/superpowers/specs/2026-07-18-main5-clrc-sft-repair-recovery-design.md \
  docs/superpowers/plans/2026-07-18-main5-clrc-sft-repair-recovery.md \
  tests/test_pcd_no_visual_runner.py \
  scripts/test/run_pcd_no_visual.sh
git commit -m "Restore main5 CLRC SFT repair recovery"
git push origin main
```
