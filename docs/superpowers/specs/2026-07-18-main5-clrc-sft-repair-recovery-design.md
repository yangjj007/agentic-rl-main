# Main5 CLRC SFT Repair Recovery Design

## Problem

The `trainer_state.json` run for the main5 OPD method ends near 25% ChartQA accuracy. The run does not look like an output-degeneration failure: the tail degenerate rate is near zero. The failure is that late training still has too little effective GRPO signal:

- `global_signal/accuracy_reward_mean` tail50: about `0.2528`
- `signal/grpo_zero_loss_rate` tail50: about `0.94`
- `routing/teacher_probe_candidate_accuracy` tail50: about `0.4377`
- `routing/opd_route_rate` tail50: about `0.0931`
- `routing/grpo_route_rate` tail50: about `0.2406`

The old `clrc_full` adaptive OPD schedule was too OPD-heavy early and did not make mixed/GRPO recovery dominant enough. Lowering OPD pressure is necessary but not sufficient. Existing DyME notes show that `student_hint_short` teacher-SFT repair is useful because it reduces long full-CoT contamination and gives a shorter visual-reasoning target, though it still needs OPD decay/cap and mixed sampling to avoid turning into pure imitation.

## Design

Restore the lost gold-hidden CLRC runner contract and add a stronger SFT-routed variant.

### Restored CLRC Main Variant

Variant:

`deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision`

This variant uses gold-hidden teacher inputs (`format_only,visual_facts_deplot`), disables hard imitation trajectory, keeps teacher-correct repair in OPD mode, and restores the GRPO-recovery defaults:

- `DYME_TEACHER_TRAJECTORY=0`
- `DYME_TEACHER_CORRECT_REPAIR_MODE=opd` by omission from exported env
- `DYME_OPSD_SKIP_DEGENERATE=0`
- `DYME_ADAPTIVE_SUPERVISION=1`
- `DYME_ADAPTIVE_READINESS_SOURCE=global_grpo_route`
- `DYME_ADAPTIVE_TARGET_READINESS=0.15`
- `DYME_ADAPTIVE_OPSD_INITIAL_WEIGHT=1.0`
- `DYME_ADAPTIVE_OPSD_FINAL_WEIGHT=0.25`
- `DYME_ADAPTIVE_OPSD_INITIAL_CAP=4`
- `DYME_ADAPTIVE_OPSD_FINAL_CAP=1`
- `DYME_ADAPTIVE_TEACHER_INITIAL_WEIGHT=0.0`
- `DYME_ADAPTIVE_TEACHER_FINAL_WEIGHT=0.0`
- `DYME_EFFECTIVE_SAMPLING=1`
- `DYME_EFFECTIVE_SAMPLING_MIXED_WEIGHT=6.0`
- `DYME_OPSD_OVERFLOW_ROUTE=mixed_grpo_all_wrong_skip`
- `DYME_GLOBAL_SIGNAL_LOGGING=1`

### Strong SFT Route Variant

Variant:

`deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair`

This inherits the restored CLRC main variant and changes teacher-correct recoverable all-wrong rows from OPD repair to short SFT repair:

- `DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft`
- `DYME_TEACHER_SFT_REPAIR_SCOPE=all_wrong`
- `DYME_TEACHER_SFT_REPAIR_SLOTS=1`
- `DYME_TEACHER_SFT_TARGET_STYLE=student_hint_short`
- `DYME_TEACHER_SFT_TARGET_CONSTRAINT=chartqa_hint`
- `DYME_TEACHER_SFT_SANITIZE_PRIVILEGED=1`

The variant remains gold-hidden because teacher providers stay `format_only,visual_facts_deplot`. It does not use `oracle_hint`, answer-only targets, or visual supervision/refiner envs.

## Testing

Use dry-run tests as the contract:

- restored CLRC main exports the GRPO-recovery adaptive defaults and does not export teacher-SFT repair envs;
- strong SFT repair variant exports the same adaptive defaults plus `traj_sft` and `student_hint_short`;
- both variants keep `DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot`;
- both variants keep `DYME_VISUAL_REFINER=0` and `DYME_DEPLOT_ENABLED=0`;
- no smoke command starts real GPU training.

Smoke:

```bash
DYME_PCD_RUN_ID=smoke_main5_sft_route \
DYME_PCD_OUTPUT_ROOT=outputs/test-fast/main5-smoke/sft_route \
DYME_PCD_LOG_ROOT=outputs/test-fast/logs/main5-smoke/sft_route \
DYME_PCD_MAX_STEPS=2 \
bash scripts/test/run_pcd_no_visual.sh 10 --resume none --dry-run \
  --variant deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair
```
