# Grounded Full-CoT OPD Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a deterministic ChartQA full-CoT verifier, offline G0 audit, and G1 teacher process-supervision quality gate without changing answer reward or GRPO/OPD/SFT routes.

**Architecture:** Keep parsing and verification in a pure `reward_utils` module with frozen result dataclasses. Add a small `opsd_utils` gate module that converts verification results into a shared eligible-index set and aggregate metrics. `DyMETrainer` computes quality immediately after teacher probe generation, uses the eligible set for both raw teacher trajectory and repair paths, and preserves legacy behavior when the feature is off or diagnostic-only.

**Tech Stack:** Python 3, dataclasses, regular expressions, JSON/CSV, PyTorch trainer integration, pytest.

---

### Task 1: Structured CoT and DePlot Parsing

**Files:**
- Create: `reward_utils/chart_cot_verifier.py`
- Create: `tests/test_chart_cot_verifier.py`

- [x] **Step 1: Write failing parser tests**

Cover complete sections, escaped newlines, duplicate/missing/empty sections, final `Answer:`, JSON-encoded DePlot, placeholder rejection, row-width normalization, and percent-aware numeric equivalence.

- [x] **Step 2: Run parser tests and verify RED**

Run: `pytest -q tests/test_chart_cot_verifier.py -k 'parse or numeric'`

Expected: collection/import failure because `reward_utils.chart_cot_verifier` does not exist.

- [x] **Step 3: Implement minimal parser dataclasses and functions**

Define `ParsedChartCoT`, `ChartTable`, `NormalizedNumber`, `parse_chart_cot`, `parse_deplot_table`, `parse_number`, and `numbers_equivalent`.

- [x] **Step 4: Run parser tests and verify GREEN**

Run: `pytest -q tests/test_chart_cot_verifier.py -k 'parse or numeric'`

Expected: PASS.

### Task 2: Grounding and CoT Consistency

**Files:**
- Modify: `reward_utils/chart_cot_verifier.py`
- Modify: `tests/test_chart_cot_verifier.py`

- [x] **Step 1: Write failing grounding and consistency tests**

Cover supported `2019: 70`, contradicted `2019: 71`, bare derived numbers as unknown, textual conclusion-answer agreement, numeric mismatch, and `0.39` versus `39%` equivalence.

- [x] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_chart_cot_verifier.py -k 'grounding or consistency'`

Expected: FAIL because claim and consistency APIs are absent.

- [x] **Step 3: Implement claim and consistency APIs**

Define `GroundedClaim`, `ConsistencyResult`, `verify_grounded_claims`, and `verify_conclusion_answer_consistency`. Use only explicit high-confidence label/value bindings for contradictions.

- [x] **Step 4: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_chart_cot_verifier.py -k 'grounding or consistency'`

Expected: PASS.

### Task 3: Reasoning Checks and Q Classification

**Files:**
- Modify: `reward_utils/chart_cot_verifier.py`
- Modify: `tests/test_chart_cot_verifier.py`

- [x] **Step 1: Write failing reasoning and quality tests**

Cover valid/invalid/unknown minimum, maximum, count-threshold, sum, difference, comparison, and Q3/Q2/Q1/Q0 classification. Explicitly test that unknown reasoning is neutral and that an answer-correct but malformed trajectory is Q1.

- [x] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_chart_cot_verifier.py -k 'reasoning or quality'`

Expected: FAIL because reasoning and quality APIs are absent.

- [x] **Step 3: Implement deterministic checks and classifier**

Define `ReasoningCheck`, `ChartCoTVerification`, `verify_reasoning`, and `verify_chart_cot_trajectory`. Accept `answer_correct` as an explicit input so the verifier never redefines official correctness.

- [x] **Step 4: Run full verifier tests and verify GREEN**

Run: `pytest -q tests/test_chart_cot_verifier.py`

Expected: PASS.

### Task 4: Template Diagnostics and Offline G0 Audit

**Files:**
- Create: `scripts/audit_chart_cot_quality.py`
- Create: `tests/test_chart_cot_quality_audit.py`
- Modify: `reward_utils/chart_cot_verifier.py`

- [x] **Step 1: Write failing audit tests**

Create a temporary dataset with Q3/Q0 hints and repeated templates. Assert deterministic sampling and the three required artifacts: summary JSON, rows JSONL, and conflicts CSV.

- [x] **Step 2: Run audit tests and verify RED**

Run: `pytest -q tests/test_chart_cot_quality_audit.py`

Expected: import/file failure because the audit script does not exist.

- [x] **Step 3: Implement template normalization and CLI audit**

Add `normalize_reasoning_template` and `summarize_template_diversity`. Implement dataset and candidate-JSONL loading, deterministic sampling, row serialization, conflict extraction, and summary aggregation.

- [x] **Step 4: Run audit tests and verify GREEN**

Run: `pytest -q tests/test_chart_cot_quality_audit.py`

Expected: PASS.

### Task 5: Shared Process-Supervision Gate

**Files:**
- Create: `opsd_utils/chart_cot_quality_gate.py`
- Create: `tests/test_chart_cot_quality_gate.py`

- [x] **Step 1: Write failing gate tests**

Assert `off` and `diagnostic` preserve all candidate indices, `gate` retains only Q3, empty candidates are zero-safe, and aggregated Q rates sum correctly.

- [x] **Step 2: Run gate tests and verify RED**

Run: `pytest -q tests/test_chart_cot_quality_gate.py`

Expected: import failure because the gate module does not exist.

- [x] **Step 3: Implement gate configuration, filtering, and metrics**

Define `ChartCoTQualityGateConfig`, `ChartCoTQualityGateResult`, `filter_quality_eligible_indices`, and `aggregate_chart_cot_verifications`.

- [x] **Step 4: Run gate tests and verify GREEN**

Run: `pytest -q tests/test_chart_cot_quality_gate.py`

Expected: PASS.

### Task 6: Trainer Integration

**Files:**
- Modify: `trainer/DyMETrainer.py`
- Create: `tests/test_chart_cot_trainer_integration.py`

- [x] **Step 1: Write failing integration-helper tests**

Test a pure helper that receives teacher trajectory text, samples, answer-correct indices, and config; assert diagnostic preservation, gate rejection, missing DePlot neutrality, and bounded sample records.

- [x] **Step 2: Run integration tests and verify RED**

Run: `pytest -q tests/test_chart_cot_trainer_integration.py`

Expected: FAIL because trainer integration helpers are absent.

- [x] **Step 3: Implement trainer quality evaluation**

After `_apply_teacher_probe_routing`, verify teacher trajectory text, create the shared eligible set, remove rejected raw trajectories before repair routing, restrict repair indices to the same eligible set, and append `cot_verify/*` metrics without introducing new distributed collectives.

- [x] **Step 4: Run integration and existing routing tests**

Run: `pytest -q tests/test_chart_cot_trainer_integration.py tests/test_signal_aware_routing.py tests/test_teacher_sft_repair.py`

Expected: PASS.

### Task 7: Configuration and Runner Variants

**Files:**
- Modify: `config/config.py`
- Modify: `scripts/test/run_pcd_no_visual.sh`
- Modify: `tests/test_config_antidegen.py`
- Modify: `tests/test_run_pcd_no_visual_script.py`

- [x] **Step 1: Write failing config and runner tests**

Assert environment overrides for `off/diagnostic/gate`, Q3 requirement, sample logging bounds, both new variant names, full-CoT `chartqa_hint` target style, and disabled eval-format reward.

- [x] **Step 2: Run config/runner tests and verify RED**

Run: `pytest -q tests/test_config_antidegen.py tests/test_run_pcd_no_visual_script.py -k 'cot_quality'`

Expected: FAIL because config keys and variants are absent.

- [x] **Step 3: Add config and runner wiring**

Add `chart_cot_quality_gate` defaults and environment parsing. Register diagnostic and gate variants derived from the progress-scheduled OPD variant and export all quality-gate environment variables.

- [x] **Step 4: Run config/runner tests and verify GREEN**

Run: `pytest -q tests/test_config_antidegen.py tests/test_run_pcd_no_visual_script.py`

Expected: PASS.

### Task 8: Regression and Smoke Audit

**Files:**
- Modify only if a failing test exposes a covered defect.

- [x] **Step 1: Run focused suite**

Run: `pytest -q tests/test_chart_cot_verifier.py tests/test_chart_cot_quality_audit.py tests/test_chart_cot_quality_gate.py tests/test_chart_cot_trainer_integration.py tests/test_config_antidegen.py tests/test_run_pcd_no_visual_script.py tests/test_teacher_sft_repair.py tests/test_signal_aware_routing.py`

Expected: PASS.

- [x] **Step 2: Run offline G0 smoke audit**

Run: `python scripts/audit_chart_cot_quality.py --dataset data/chartqa/train_medium_vf_full.json --out-dir outputs/chart_cot_quality_g0_smoke --max-samples 500 --seed 42`

Expected: three output artifacts, 500 rows, Q-class totals equal to 500, and no input modification.

- [x] **Step 3: Run dry-run for both variants**

Run: `bash scripts/test/run_pcd_no_visual.sh 4 --variant deplot_no_vs_opd_pcd_oracle_hint_full_cot_quality_diagnostic --dry-run`

Run: `bash scripts/test/run_pcd_no_visual.sh 4 --variant deplot_no_vs_opd_pcd_oracle_hint_full_cot_quality_gate --dry-run`

Expected: diagnostic exports `DYME_CHART_COT_GATE_MODE=diagnostic`; gate exports `gate`, full-CoT repair style, progress schedule, effective sampling, condition-aware overflow, and disabled eval-format reward.

- [x] **Step 4: Record verification limits**

Summarize focused tests, smoke-audit quality distribution, conflict examples, and any unrun GPU training checks. Do not claim the 4-epoch accuracy target until a real training/eval run completes.
