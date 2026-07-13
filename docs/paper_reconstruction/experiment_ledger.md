# CLRC Experiment Ledger

更新时间：2026-07-14 00:19 CST

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
| `oracle_opd_no_hard_imitation_adaptive_4epoch_20260713_121946/...opd_no_hard_imitation_adaptive_supervision` | oracle OPD without teacher trajectory/repair, but with legacy gold-hint online SFT | 4 planned, stopped at step 60 | LLaVA-OV 0.5B | LLaVA-OV 7B | DePlot + `oracle_hint` + dataset hint | yes | yes | realized global GRPO route, teacher trajectory/repair disabled, online SFT slots retained | stopped unhealthy | — | — | `outputs/test-fast/long-runs/oracle_opd_no_hard_imitation_adaptive_4epoch_20260713_121946/status` |
| `oracle_opd_no_full_hint_hard_sft_adaptive_4epoch_20260713_150545/...opd_no_full_hint_hard_sft_adaptive_supervision` | clean oracle OPD causal run with every dataset full-hint hard-target path disabled | 4 planned, interrupted at step 86 | LLaVA-OV 0.5B | LLaVA-OV 7B | DePlot + `oracle_hint`; soft OPD only on verified wrong states | yes | yes | realized global GRPO route; teacher trajectory/repair, online-SFT slots, all-wrong SFT and forced SFT disabled | hardware-transient interrupted; no result | — | — | `outputs/test-fast/long-runs/oracle_opd_no_full_hint_hard_sft_adaptive_4epoch_20260713_150545/status` |
| `oracle_opd_no_full_hint_hard_sft_adaptive_resilient_4epoch_20260713_181613/...opd_no_full_hint_hard_sft_adaptive_supervision` | matched resilient rerun of the clean oracle OPD causal experiment | 4 | LLaVA-OV 0.5B | LLaVA-OV 7B | matched to interrupted clean run | yes | yes | approved frozen method; 50-step checkpoints and GPU/runtime recovery only | waiting for GPU gate; ScaleDivide remains on GPU 3, external video jobs rotate across GPUs, and the one-step P0 matrix smoke currently uses GPU 1; no rows/checkpoint yet (`2026-07-14 00:59 CST`) | — | — | `outputs/test-fast/long-runs/oracle_opd_no_full_hint_hard_sft_adaptive_resilient_4epoch_20260713_181613/launch.info` |

## CLRC 主论文必要实验槽位

| Planned run | Method role | Epoch | Teacher sees gold | Verifier uses reference | Only changed factor | Required result |
|---|---|---:|---|---|---|---|
| `gold_hidden_fixed_schedule_4epoch` | gold-hidden-teacher fixed-schedule control | 4 | no | yes | fixed OPD/trajectory/cap schedule | 与 gold-hidden CLRC 同数据、模型、teacher evidence |
| `gold_hidden_recoverability_without_controller_4epoch` | local routing only | 4 | no | yes | 关闭 global controller | 隔离 completion-level recoverability routing |
| `gold_hidden_clrc_full_4epoch` | paper main method | 4 | no | yes | 开启 realized-autonomy controller | `gold_rate=0.0`；必须完整 eval |
| `gold_hidden_clrc_weight_only_4epoch` | action ablation | 4 | no | yes | 只控制 OPD weight | action-split table |
| `gold_hidden_clrc_budget_only_4epoch` | action ablation | 4 | no | yes | 只控制 teacher cap | compute/accuracy Pareto |
| `gold_hidden_clrc_changed_scale` | robustness | 4 or matched updates | no | yes | 改 effective batch 或 data scale | 对比 fixed-step 动作发生状态 |

### 10epoch ChartQA matrix contract

| Slot | Explicit variant | Runtime contract | Status |
|---|---|---|---|
| P0-E1 | `deplot_no_vs_opd_pcd_gold_hidden_no_opd` | route-matched no-OPD: verifier/probe and routes retained, OPD weight `0`, GRPO `1`, no full-hint hard SFT | 1-step smoke passed; default 10epoch matrix ready |
| P0-E2 | `deplot_no_vs_opd_pcd_gold_hidden_uncond_opd_no_full_hint_hard_sft` | no verifier call, eligible wrong completion retains OPD, fixed cap `8` | ready for 2-step smoke |
| P0-E3 | `deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_fixed` | gold-hidden verifier routing, fixed OPD `1.5`, cap `8`, sampling from step `0` | ready for 2-step smoke |
| P0-E4 | `deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision` | E3 plus global-GRPO controller, target `.30`, cap `8->2` | ready for 2-step smoke |
| P0-E4 target ablation | `deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_target020` | only controller target changes to `.20` | ready, lower priority |
| P0-E5a | `deplot_no_vs_opd_pcd_gold_hidden_grpo_only` | teacher-probe off, OPD off, hard SFT off, pure matched GRPO | ready; runner contract tested |
| P0-E5b | `deplot_no_vs_opd_pcd_gold_hidden_opd_only_no_full_hint_hard_sft` | GRPO weight `0`, OPD contribution only | loss mixer fixed so no-OPD batches cannot leak GRPO; runner/unit contract tested; 1-step smoke after fix has OPD route `0.25`, `loss/opsd=0.08646`, `train_loss=0.12969` |
| P0-E5c | `deplot_no_vs_opd_pcd_gold_hidden_fallback_only` | requested OPD/GRPO off, legacy fallback diagnostic | runner env contract tested, but `mode=dyme` bypasses the mixer because no `opsd_mask` is produced; invalid as a strict single-signal result |
| P0-E6 | `deplot_no_vs_opd_pcd_gold_hidden_token_reliability_clrc` | lexical OPD mask prototype: base `.75`, numeric `2.0`, answer `1.5`; reward/routes unchanged | 1-step runtime smoke passed; include as exploratory 10epoch row, not main claim unless numeric-token audit is clean |
| P0-E9 diagnostic | `deplot_no_vs_opd_pcd_gold_hidden_mixed_group_shortest_correct_hard_replay` | shortest-correct completion hard replay for mixed-wrong rows; all-wrong rows skip | 1-step smoke passed on 2026-07-14 with all-wrong skip metric; not SSOPD |

The sequential one-step matrix smoke `smoke_all14_20260714_v4` started at `2026-07-14 00:53 CST`
on one GPU. The DyME pure/full, oracle-official, and route-matched no-OPD rows have completed;
the unconditional-OPD row is currently running. The route-matched no-OPD runtime evidence is
clean: teacher probe remains enabled, OPD route is `0.25`, effective OPD loss weight is `0.0`,
skip route is `0.75`, teacher-input gold rate is `0.0`, and all full-hint hard-target rates are
zero. This is contract evidence only, not an accuracy result. The smoke may execute
`fallback_only_matched` to expose its current behavior, but that row remains excluded from full
training and paper claims until the top-level base-loss mask is implemented.

The focused 10epoch matrix regression suite passes `29/29` on `2026-07-14`. The suite covers
DyME epoch override, default matrix labels, retired near-neighbor rejection, route-matched no-OPD,
pure GRPO, OPD-only/fallback-only runner contracts, token reliability, hard replay, and the
unconditional GRPO/OPD loss mixer. The 1-step smokes above are runtime evidence only and do not
provide an accuracy estimate.

The `2026-07-13 23:55 CST` semantic audit found that `opd_only_matched` could leak GRPO on batches
with no local OPD sample because `grpo_weight` was applied only inside the nonempty-OPD branch.
This is fixed on `2026-07-14`: `opsd_utils.opsd_loss._combine_grpo_opsd_losses` applies
`DYME_GRPO_WEIGHT` even when `opsd_loss is None`, and the unit/runner tests now pin the behavior.
Fallback-only remains invalid: `mode=dyme` produces no `opsd_mask`, so compute_loss bypasses the
mixer and the exported zero GRPO weight is not yet an effective runtime guarantee.

The runnable mixed-group diagnostic is honestly isolated as
`mixed_group_shortest_correct_hard_replay`; it is not a matched SSOPD distribution objective.
The misleading `ssopd_mixed_group` and `vold_cold_start` executable labels are retired and fail
explicitly.

At `2026-07-14 00:47 CST`, near-neighbor semantic isolation passes `105` focused regressions and
`12` paper checks without changing the frozen oracle runtime contract.

## 当前长程任务

### Active resilient OPD isolation run

- Train tmux: `dyme_no_full_hint_resilient_181613`
- Template/health monitor tmux: `dyme_no_full_hint_resilient_181613_watch`
- Post-eval forensic tmux: `dyme_no_full_hint_resilient_181613_forensics`
- Run ID: `oracle_opd_no_full_hint_hard_sft_adaptive_resilient_4epoch_20260713_181613`
- Current state: `waiting_for_gpu_gate`; as of `2026-07-14 00:47 CST`, external ScaleDivide PID
  `214423` remains on GPU 3 and multiple short-lived external `vcecf` video jobs rotate across
  GPUs 0/4/7. The compute-process-empty gate
  therefore remains closed, and no training row, checkpoint, or output directory exists. This
  job is outside this experiment and is not terminated by the runner. Its own log has advanced
  through epoch 55 with a new best validation MSE `0.5485025431`, so it is slow but making
  progress rather than demonstrably stalled. The new best resets its early-stopping patience;
  no reliable wall-time estimate is currently available.
- Hard-target invariants cover teacher trajectory, teacher-SFT repair, legacy online-SFT slot,
  forced replacement, and aggregate full-hint hard-target exposure; every rate must remain zero.
- Adaptive effective sampling is active from step zero. Runner dry-run exports
  `DYME_EFFECTIVE_SAMPLING_AFTER_STEP=0` and `DYME_EFFECTIVE_SAMPLING_START_PROGRESS=0.0`;
  trainer `always_active` provides a second runtime safeguard. The relevant adaptive/config/
  resilient regression suite passes `70` tests.
- Exact variant dry-run confirms `DYME_OPSD_LOSS_TYPE=jsd`,
  `DYME_OPSD_SKIP_DEGENERATE=0`, adaptive OPD weight/cap `1.5->0.5` and `8->2`, both adaptive
  teacher weights at `0.0`, and probe-failure routing `mixed_grpo_all_wrong_skip`. Thus the
  paper's clean recipe describes effective runtime values rather than base-config defaults.
- Prelaunch matched forensic compares the interrupted clean run, the `0.5800` baseline, and
  the `0.5120` full-trajectory negative result under
  `outputs/test-fast/long-runs/oracle_opd_no_full_hint_hard_sft_adaptive_resilient_4epoch_20260713_181613/prelaunch_forensics/`.
  Clean all-wrong teacher correctness is `0.9294`; steps 71--80 reach accuracy/GRPO
  `0.0797/0.0844` versus `0.0707/0.0781` for the `0.5800` baseline. This evidence supports
  completing the frozen recipe rather than reacting to its high early OPD route.
- Behavior metrics: `completions/full_cot_template_rate`, `empty_cot_skeleton_rate`, and `malformed_answer_section_rate`.
- Monitor checks the latest 10 optimizer rows every 5 minutes and records accuracy, GRPO/OPD/SFT
  routes, zero-loss, all-wrong, degenerate, clip, EOS, and template behavior. Any
  hard-imitation invariant violation stops immediately; full-template rate `>0.8` plus
  either empty-skeleton or malformed-answer rolling mean `>0.2` over the same latest-10-row
  window stops the run. Candidate-only partial drift is warning-only. The recovery gate uses
  `global_signal/grpo_route_rate` when available and explicitly falls back to rank-local
  routing only for legacy logs; the queued run must report global source fraction `1.0`.
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
- Steps 41--50: hard-imitation invariants remain exactly zero, so full teacher-trajectory
  hard supervision has not returned. Nevertheless, the current run regresses to
  accuracy/GRPO `0.0102/0.0000`, zero-loss/all-wrong `1.0/1.0`, and
  degenerate/clip/EOS `0.8438/0.5992/0.4094`. This is substantially more degenerate than
  the `0.5800` run in the matched window (`0.3563`). Wrong probed completions have
  near-universal partial `Goal`-without-`Answer` drift, which is evidence of OPD-mediated
  teacher-style transfer, not evidence that the disabled hard-trajectory path reactivated.
  Keep only steps 51--60 as a recovery gate because both historical controls improve in
  that interval. Stop if the 10-step degenerate rate does not fall below approximately
  `0.30` or task accuracy/GRPO does not materially recover.
- Steps 51--60 fail the recovery gate: accuracy/GRPO are `0.0070/0.0156`, while
  degenerate/clip/EOS are `0.9156/0.5977/0.3844`. The matched `0.5800` run reaches
  `0.0625/0.0313` accuracy/GRPO with degenerate rate `0.1500`. Training was stopped after
  step 60 (step 61 had begun) and no final eval was launched. The stopped run is evidence
  that removing hard teacher trajectories alone is insufficient; it is not a completed
  accuracy result.
- Post-stop code audit identifies the confounder: `trainer/DyMETrainer.py` builds legacy
  online-SFT targets as `hint + answer`, and ChartQA dataset hints contain the complete
  `Goal/Observation/Reasoning/Conclusion` trajectory. The fast profile reserves four SFT
  slots per all-wrong group during warmup; observed SFT route rate is approximately
  `0.49--0.53`. Thus the run removed teacher-generated hard trajectories but did not remove
  full-hint hard supervision. The next causal intervention must disable legacy online-SFT
  slots before modifying OPD token weights.
- The first no-full-hint distributed smoke then exposed a second bypass: malformed but
  otherwise GRPO-eligible completions entered `_should_force_sft_replace`, producing
  `legacy_online_sft_forced_rate=0.125`. A failing trainer regression test reproduced the
  event; the no-full-hint gate now disables this forced replacement while preserving legacy
  behavior. The focused suite passes `164` tests.
- The repeated 4-step, 8-GPU smoke completed cleanly. Across all logged rows,
  `teacher_traj_weight`, `teacher_sft_repair_rate`, `legacy_online_sft_rate`, and
  `full_hint_hard_target_rate` have maximum `0.0`; the external candidate monitor parsed
  `951` records and returned `status=ok`. The clean 4epoch run launched at
  `2026-07-13 15:05 CST` with automatic gates at rows 20, 40, 60, and 100 and automatic
  8-GPU final evaluation.
- Step-20 clean-run gate: hard-target invariants remain exactly zero and no full/partial
  template collapse is observed. The latest ten rows have accuracy/GRPO/OPD
  `0.0043/0.0094/0.9906`, zero-loss `1.0`, and degenerate/clip/EOS
  `0.8531/0.8340/0.1563`. This is unhealthy but still marginally better than the historical
  full-trajectory negative control in steps 21--30, so the preregistered decision is to
  continue unchanged through step 40 before selecting the answer-anchor contingency.
- Step-40 clean-run gate: the latest ten rows reach accuracy/GRPO `0.0160/0.0063`,
  zero-loss `0.95`, and degenerate/clip/EOS `0.4250/0.7129/0.2500`. Relative to the
  historical `0.5800` window, task accuracy and clipping are better but GRPO coverage and
  degeneration are worse. Step 41 briefly lowers zero-loss to `0.5`; therefore the run
  remains diagnostically mixed rather than irrecoverable and continues to the registered
  step-60 recovery gate.
- Step-60 clean-run recovery gate passes: latest-ten accuracy is `0.0684`, above the
  historical `0.5800` run's matched `0.0625`; degenerate/clip/EOS recover to
  `0.1219/0.3891/0.6281`. Realized GRPO remains only `0.0156` and task zero-loss remains
  `1.0`, so the next question is whether the global controller can convert higher task
  success into mixed groups and GRPO coverage by step 100. Continue unchanged.
- Steps 72--81 show sustained recovery: latest-ten accuracy/GRPO/OPD are
  `0.0836/0.0906/0.7500`, while degenerate/clip/EOS are `0.1406/0.2871/0.6656`.
  Controller mastery reaches `0.0842`, OPD weight falls to `1.308`, and the per-prompt cap
  moves from 8 to 7. This is stronger than the historical `0.5800` run's steps 81--90
  task signal and supports continuing to the full step-100 gate.
- A post-interruption audit separates teacher formatting from student behavior across 21,183
  wrong probe candidates. In steps 70--79, `44.82%` of `student_output` values contain at
  least one reasoning heading, but `0.0%` contain the full four-section teacher template;
  `88.50%` retain an `Answer:` heading and only `0.76%` have empty or malformed answers.
  Steps 80--86 increase Goal-without-Answer to `8.01%`, still without any complete four-heading
  copy. This supports a partial soft style-transfer risk, not a claim that full teacher hard
  supervision or full-template collapse reappeared. Token-selective OPD remains conditional on
  the complete run's final error taxonomy.

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
