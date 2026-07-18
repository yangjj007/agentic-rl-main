# Main5 CLRC SFT Repair Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the lost main5 gold-hidden CLRC OPD recovery runner contract, add a refiner-backed SFT repair route, and prevent incomplete model-shard training starts.

**Status:** Implemented and smoke-tested locally.

**Architecture:** Keep `scripts/test/run_pcd_no_visual.sh` as the single source of variant wiring. Add refiner-backed repair in `opsd_utils/teacher_sft_repair.py` and connect it from `trainer/DyMETrainer.py`; lock the shell/env contract in dry-run tests. Harden `scripts/test/run_main5_10epoch_campaign.sh` so the long campaign retries model downloads until every indexed safetensors shard exists.

**Tech Stack:** Bash runner scripts, pytest dry-run tests, existing DyME teacher-probe/Visual Refiner controls.

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
    assert "DYME_TEACHER_CORRECT_REPAIR_MODE=refiner_sft" in out
    assert "DYME_TEACHER_SFT_REPAIR_SCOPE=all_wrong" in out
    assert "DYME_TEACHER_SFT_REPAIR_SLOTS=1" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=student_hint_short" in out
    assert "DYME_TEACHER_SFT_TARGET_STYLE=answer_only" not in out
    assert "DYME_ADAPTIVE_TARGET_READINESS=0.15" in out
    assert "DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT=6.0" in out
    assert "DYME_VISUAL_CHECKER=0" in out
    assert "DYME_VISUAL_REFINER=1" in out
    assert "DYME_VISUAL_PREFETCH_IC=1" in out
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

- [x] **Step 3: Add refiner-backed SFT repair override**

For the `_sft_repair` variant, override:

```bash
TEACHER_CORRECT_REPAIR_MODE="${DYME_TEACHER_CORRECT_REPAIR_MODE:-refiner_sft}"
TEACHER_SFT_TARGET_STYLE="${DYME_TEACHER_SFT_TARGET_STYLE:-student_hint_short}"
TEACHER_SFT_REPAIR_SCOPE="${DYME_TEACHER_SFT_REPAIR_SCOPE:-all_wrong}"
TEACHER_SFT_REPAIR_SLOTS="${DYME_TEACHER_SFT_REPAIR_SLOTS:-1}"
VISUAL_CHECKER="${DYME_VISUAL_CHECKER:-0}"
VISUAL_REFINER="${DYME_VISUAL_REFINER:-1}"
VISUAL_PREFETCH_IC="${DYME_VISUAL_PREFETCH_IC:-1}"
VISUAL_LOG="${DYME_VISUAL_LOG:-1}"
```

- [x] **Step 4: Verify green**

Run the same focused pytest command. Expected: pass.

### Task 3: Implement Refiner SFT Repair Routing

**Files:**
- Modify: `opsd_utils/teacher_sft_repair.py`
- Modify: `trainer/DyMETrainer.py`
- Modify: `tests/test_teacher_sft_repair.py`

- [x] **Step 1: Write failing test**

Add a test showing `refiner_sft` can promote an all-wrong teacher-correct slot without any teacher trajectory:

```python
def test_refiner_sft_repair_promotes_all_wrong_teacher_correct_without_traj() -> None:
    modes, kept_trajs, repairs, stats = apply_teacher_sft_repair_routing(
        completion_modes=[MODE_OPSD, MODE_OPSD, MODE_GRPO, MODE_OPSD],
        teacher_traj_indices=set(),
        teacher_correct_indices={0, 1, 3},
        group_has_correct=[False, True],
        num_generations=2,
        config=TeacherSftRepairConfig(
            repair_mode="refiner_sft",
            scope="all_wrong",
            slots_per_prompt=1,
        ),
    )

    assert modes == [MODE_SFT, MODE_OPSD, MODE_GRPO, MODE_OPSD]
    assert repairs == {0}
    assert kept_trajs == set()
    assert stats.teacher_sft_repairs == 1
```

- [x] **Step 2: Implement routing and trainer connection**

Allow `TeacherSftRepairConfig.enabled` for `refiner_sft`, accept `teacher_correct_indices`, and have `DyMETrainer` pass OPD teacher-probe-correct indices into `apply_teacher_sft_repair_routing`. For `refiner_sft`, the trainer does not tokenize teacher trajectory text; the normal SFT branch consumes the refined hint target.

- [x] **Step 3: Verify green**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest -q tests/test_teacher_sft_repair.py
```

Expected: pass.

### Task 4: Harden Main5 Campaign Model Readiness

**Files:**
- Modify: `scripts/test/run_main5_10epoch_campaign.sh`
- Modify: `tests/test_main5_campaign_runner.py`

- [x] **Step 1: Write failing tests**

Add tests that require `model_ready` to inspect `model.safetensors.index.json` / `weight_map`, report `missing model shard`, and use `max_workers=1` for resumable downloads.

- [x] **Step 2: Implement shard-aware readiness**

When `model.safetensors.index.json` exists, parse it with Python and require every unique `weight_map` shard file to exist before training starts.

- [x] **Step 3: Prioritize the stronger SFT route**

Keep the campaign variant list ordered with `deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair` before `deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision`, so the long run tests the refiner-backed route first.

- [x] **Step 4: Verify green**

Run:

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest -q tests/test_main5_campaign_runner.py
```

Expected: pass.

### Task 5: Smoke And Sync

**Files:**
- Modified code/tests/docs listed above.

- [x] **Step 1: Run broader checks**

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/test_pcd_no_visual_runner.py -q
/home/deepseek_VG/.conda/envs/dyme/bin/python -m pytest tests/test_main5_campaign_runner.py tests/test_teacher_sft_repair.py tests/test_opd_probe_ablation_smoke.py -q
bash -n scripts/test/run_pcd_no_visual.sh
bash -n scripts/test/run_pcd_no_visual_10epoch.sh
bash -n scripts/test/run_main5_10epoch_campaign.sh
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
  opsd_utils/teacher_sft_repair.py \
  trainer/DyMETrainer.py \
  scripts/test/run_main5_10epoch_campaign.sh \
  tests/test_pcd_no_visual_runner.py \
  tests/test_main5_campaign_runner.py \
  tests/test_teacher_sft_repair.py \
  scripts/test/run_pcd_no_visual.sh
git commit -m "Add main5 refiner SFT repair route"
git push origin main
```
