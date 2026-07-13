# CLRC Experiment Ledger

更新时间：2026-07-13

本账本是论文实验状态的统一事实源。`Teacher sees gold` 表示 teacher/privileged
训练上下文是否直接包含 gold answer；`Verifier uses reference` 表示路由或奖励 verifier
是否使用 reference label。两者必须分开披露。

## 已完成与正在运行的主线实验

| Run | Method role | Epoch | Student | Teacher | Evidence | Teacher sees gold | Verifier uses reference | Controller | Status | Final acc | Processed | Artifact |
|---|---|---:|---|---|---|---|---|---|---|---:|---:|---|
| `pcd_no_visual_aligned_4epoch/deplot_no_vs_opd_pcd` | gold-hidden-teacher PCD aligned baseline | 4 | LLaVA-OV 0.5B | LLaVA-OV 7B | DePlot/visual facts | no | yes | none | complete | 0.5420 | 2500/2500 | `outputs/test-fast/pcd-no-visual/pcd_no_visual_aligned_4epoch/deplot_no_vs_opd_pcd/eval_chartqa/summary.csv` |
| `route_guard_oracle_hint_4epoch_7gpu/...oracle_hint_route_guard` | oracle route-guard baseline | 4 | LLaVA-OV 0.5B | LLaVA-OV 7B | DePlot + oracle hint | yes | yes | rule guard | complete | 0.5592 | 2500/2500 | `outputs/test-fast/pcd-no-visual/route_guard_oracle_hint_4epoch_7gpu/deplot_no_vs_opd_pcd_oracle_hint_route_guard/eval_chartqa/summary.csv` |
| `pcd_oracle_teacher_sft_repair_4epoch/deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair` | oracle full-template repair | 4 | LLaVA-OV 0.5B | LLaVA-OV 7B | DePlot + oracle hint | yes | yes | none | complete | 0.5624 | 2500/2500 | `outputs/test-fast/pcd-no-visual/pcd_oracle_teacher_sft_repair_4epoch/deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair/eval_chartqa/summary.csv` |
| `pcd_oracle_teacher_sft_repair_constrained_4epoch_rerun/deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair` | oracle constrained repair | 4 | LLaVA-OV 0.5B | LLaVA-OV 7B | DePlot + oracle hint | yes | yes | fixed constraint | complete | 0.5656 | 2500/2500 | `outputs/test-fast/pcd-no-visual/pcd_oracle_teacher_sft_repair_constrained_4epoch_rerun/deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair/eval_chartqa/summary.csv` |
| `pcd_oracle_teacher_sft_repair_student_hint_short_4epoch/deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short` | strongest completed internal oracle baseline | 4 | LLaVA-OV 0.5B | LLaVA-OV 7B | DePlot + oracle hint | yes | yes | fixed schedule | complete | 0.5800 | 2500/2500 | `outputs/test-fast/pcd-no-visual/pcd_oracle_teacher_sft_repair_student_hint_short_4epoch/deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short/eval_chartqa/summary.csv`, valid row `eval_final_checkpoint_bsz1_gpu0_20260709_192652` |
| `pcd_oracle_hint_official_4epoch/deplot_no_vs_opd_pcd_oracle_hint` | oracle official upper baseline | 4 | LLaVA-OV 0.5B | LLaVA-OV 7B | oracle hint | yes | yes | official DyME-aligned | complete | 0.5872 | 2500/2500 | `outputs/test-fast/pcd-no-visual/pcd_oracle_hint_official_4epoch/deplot_no_vs_opd_pcd_oracle_hint/eval_chartqa/summary.csv` |
| `global_grpo_route_full_4epoch_20260712_205549/...full_cot_adaptive_supervision` | oracle OPD + full-trajectory negative result | 4 | LLaVA-OV 0.5B | LLaVA-OV 7B | DePlot + `oracle_hint` | yes | yes | realized global GRPO route, `alpha=.10`, `tau=.30` | complete | 0.5120 | 2500/2500 | `outputs/test-fast/pcd-no-visual/global_grpo_route_full_4epoch_20260712_205549/deplot_no_vs_opd_pcd_oracle_hint_full_cot_adaptive_supervision/eval_chartqa/summary.csv` |
| `oracle_opd_no_hard_imitation_adaptive_4epoch_20260713_121946/...opd_no_hard_imitation_adaptive_supervision` | oracle verifier-routed OPD without hard teacher imitation | 4 | LLaVA-OV 0.5B | LLaVA-OV 7B | DePlot + `oracle_hint` | yes | yes | realized global GRPO route, trajectory/repair disabled | running | — | — | `outputs/test-fast/long-runs/oracle_opd_no_hard_imitation_adaptive_4epoch_20260713_121946/` |

## CLRC 主论文必要实验槽位

| Planned run | Method role | Epoch | Teacher sees gold | Verifier uses reference | Only changed factor | Required result |
|---|---|---:|---|---|---|---|
| `gold_hidden_fixed_schedule_4epoch` | gold-hidden-teacher fixed-schedule control | 4 | no | yes | fixed OPD/trajectory/cap schedule | 与 gold-hidden CLRC 同数据、模型、teacher evidence |
| `gold_hidden_recoverability_without_controller_4epoch` | local routing only | 4 | no | yes | 关闭 global controller | 隔离 completion-level recoverability routing |
| `gold_hidden_clrc_full_4epoch` | paper main method | 4 | no | yes | 开启 realized-autonomy controller | `gold_rate=0.0`；必须完整 eval |
| `gold_hidden_clrc_weight_only_4epoch` | action ablation | 4 | no | yes | 只控制 OPD weight | action-split table |
| `gold_hidden_clrc_budget_only_4epoch` | action ablation | 4 | no | yes | 只控制 teacher cap | compute/accuracy Pareto |
| `gold_hidden_clrc_changed_scale` | robustness | 4 or matched updates | no | yes | 改 effective batch 或 data scale | 对比 fixed-step 动作发生状态 |

## 当前长程任务

### Active OPD isolation run

- Train tmux: `dyme_opd_no_hard_full_121946`
- Template monitor tmux: `dyme_opd_no_hard_monitor_121946`
- Run ID: `oracle_opd_no_hard_imitation_adaptive_4epoch_20260713_121946`
- Hard-imitation invariants: `loss/teacher_traj_effective_weight=0` and `routing/teacher_sft_repair_rate=0`.
- Behavior metrics: `completions/full_cot_template_rate`, `empty_cot_skeleton_rate`, and `malformed_answer_section_rate`.
- Monitor checks the latest 20 steps every 10 minutes and records accuracy, GRPO/OPD/SFT
  routes, zero-loss, all-wrong, degenerate, clip, EOS, and template behavior. Any
  hard-imitation invariant violation stops immediately; full-template rate `>0.8` plus
  either empty-skeleton or malformed-answer rate `>0.2` in two consecutive windows stops
  the run. Candidate-only partial drift is warning-only.
- Successful training automatically launches 8-GPU ChartQA eval with `DYME_EVAL_BATCH_SIZE=1`.
- The final eval now preserves `Template behavior counts` in `summary.csv`, covering
  full/partial templates, Goal-without-Answer, empty skeletons and malformed Answer sections.
- Step-20 gate: hard-imitation invariants remain zero and no complete empty/malformed template
  collapse is present. Wrong probed completions show strong partial Goal-without-Answer drift,
  while GRPO zero-loss/degenerate/clipped remain at `1.0`; continue through steps 21--30 because
  the candidate-only drift is also present in historical successful oracle runs.
- Steps 21--30: accuracy/GRPO remain approximately `0.0004/0.0000`; degenerate/clip/EOS are
  `0.9563/0.9773/0.0187`. This is slightly healthier than the `0.5120` full-trajectory run
  but materially worse than the `0.5800` run. Continue through steps 31--50, where the
  historical failed run first recovered generation health.
- Steps 31--40: current accuracy/GRPO reach `0.0176/0.0187`, exceeding both the `0.5120`
  run (`0.0051/0.0031`) and the `0.5800` run (`0.0094/0.0094`) in the matched window.
  Clip/EOS improve to `0.6277/0.4250`; continue unchanged.

### Completed full-trajectory run

- Train tmux: `dyme_grpo_route_full_205549`
- Monitor tmux: `dyme_grpo_route_monitor_205549`
- Total training steps: `592`
- Training state: `outputs/test-fast/long-runs/global_grpo_route_full_4epoch_20260712_205549/status`
- Console: `outputs/test-fast/long-runs/global_grpo_route_full_4epoch_20260712_205549/train_console.log`
- Monitor: `outputs/test-fast/long-runs/global_grpo_route_full_4epoch_20260712_205549/monitor.log`
- Successful training automatically launches 8-GPU ChartQA eval with `DYME_EVAL_BATCH_SIZE=1`.
- Final result must be read from the generated `eval_chartqa/summary.csv`; training reward is not an eval substitute.
- Training completed at `2026-07-13 09:39 CST`; automatic 8-GPU eval completed at `10:12 CST` with accuracy `0.5120` over `2500/2500`.
- Eval output types were `full_cot=2397`, `answer_flag=103`; `Goal:` appeared 2415 times, with many empty/malformed section templates.
- Last50 training accuracy/GRPO route reached about `0.445/0.486`, so the low held-out score is a train-eval/style generalization failure rather than a failure to optimize the training reward.
- The next matched intervention disables teacher trajectory and teacher-SFT repair while retaining verifier-routed OPD, effective sampling, and the same adaptive controller.

## 更新规则

1. `running` 行不得预填 accuracy。
2. 失败/OOM eval 行不得与有效行平均。
3. 8-GPU eval 若稳定处理 `2496/2500`，可作快速决策结果，但正式论文主表优先补齐 `2500/2500`。
4. 同一 run 的 oracle/gold-hidden-teacher 角色由实际 leakage metric 决定，不由 variant 名字决定；verifier reference access 另列。
5. 每次更新结果时同步修改 `claim_evidence_matrix.md`，正文只读取已验证状态。
