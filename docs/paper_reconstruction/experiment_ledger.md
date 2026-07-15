# CLRC Experiment Ledger

更新时间：2026-07-15 17:35 CST

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
| `clean_visualfact_fix_opd_main_4epoch_20260714_235824/...gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision` | leakage-fixed gold-hidden CLRC main run | 4 | LLaVA-OV 0.5B | LLaVA-OV 7B | real DePlot evidence; no dataset visual-fact/gold hint in prompt | no | yes | realized global GRPO route, `alpha=.10`, `tau=.30`, OPD weight `1.5->0.5`, cap `8->2` | complete negative result; hard-target/template leakage stayed zero, but teacher recoverability and held-out accuracy failed | 0.2352 | 2500/2500 | `outputs/test-fast/pcd-no-visual/clean_visualfact_fix_opd_main_4epoch_20260714_235824/deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision/eval_chartqa/summary.csv` |

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

### Structured gold-hidden teacher-only diagnostic

- The TDD implementation adds `chartqa_deplot_reasoned` and a teacher-only two-pass
  canonicalization diagnostic. The first pass receives only the chart image, question, and
  DePlot evidence and produces `Goal/Observation/Reasoning/Conclusion`; the second pass receives
  the same evidence plus the teacher's own draft and emits canonical `Answer:`. Neither pass
  receives the reference answer, dataset hint, verifier result, or oracle provider.
- Focused provider/micro-eval tests pass (`25 passed`), and an eight-sample prompt audit finds no
  `[Verified Hint]`, `[Reference Answer]`, oracle-hint, reference-answer, or dataset-hint marker.
- On the same fixed-seed 128 difficult probe candidates, raw teacher correctness is
  `0.0469` for the old short-answer micro-eval, `0.3203` for structured two-pass gold-hidden, and
  `0.9219` for oracle-hint. Structured two-pass removes parse failure (`0.8984 -> 0.0`) and raises
  raw correctness by `27.34` points over this micro-eval control, but misses the preregistered
  `0.90` admission gate by a wide margin. All-wrong/mixed-wrong correctness is
  `0.2969/0.3438`.
- Artifacts: `outputs/test-fast/teacher-probe-micro-eval/paired128_20260715/`. Because the
  teacher-only gate fails, this profile is not promoted to a full 4epoch run. The next evidence
  intervention must improve visual/table recovery itself rather than spend another training
  budget on formatting or canonicalization.
- A follow-up sampled best-of-4 test uses four independent no-gold structured rollouts on the
  same 64 candidates. Per-rollout accuracy ranges from `0.2656` to `0.3594`; verifier union
  improves only to `0.5000` (`32/64`), with identical `0.50` coverage on all-wrong and
  mixed-wrong subsets. Half the prompts remain wrong in every rollout, so errors are strongly
  correlated and cannot be solved by teacher resampling alone. Artifact:
  `outputs/test-fast/teacher-probe-micro-eval/reasoned_sampled_bon4_64_20260715/`.
- A stratified manual audit checks 30 records / 29 unique chart images across count, extreme,
  difference, percent, and other questions, balanced between all-wrong and mixed-wrong. DePlot
  evidence is directly correct in `12/30` and sufficient for a derived answer in another `7/30`:
  `63.3%` is therefore usable without table reconstruction. The observed primary failures are
  question/reference ambiguity or annotation conflict `4/30`, missing series/color mapping
  `4/30`, missing cells `2/30`, and OCR numeric error `1/30`; no row/column alignment failure is
  found in this sample. This audit rejects the blanket claim that DePlot column alignment is the
  dominant problem. It instead motivates question-conditioned evidence views: table-focused for
  numeric/derived questions, image-focused for legend/color questions, and fused fallback for
  anomalous or incomplete tables.

### Cross-dataset harness: ChartQA adapter teacher-only gate

- The approved cross-dataset design is implemented first for ChartQA as a shared typed harness
  plus a dataset adapter. Every production attempt retains the question and full chart image.
  `visual_base` uses the image directly, `visual_deplot` adds fallible DePlot evidence, and
  `visual_recovery` is generated only when the first two parsed answers disagree or fail to
  parse. Runtime decisions contain no reference answer, hint, verifier score, or correctness
  label; references are joined only after selection for offline metrics.
- TDD covers typed contracts, bare short-answer parsing, safe arithmetic validation, prompt
  leakage audits, image preservation, bounded three-attempt routing, recovery verification, and
  artifact generation. The focused regression suite passes `43` tests. A real four-example
  smoke exposed and fixed an initial parser bug that treated bare outputs such as `64.86` and
  `No` as failures and unnecessarily triggered recovery.
- The fixed-seed 128-example calibration run uses the same difficult candidate population and
  `seed=13` as the earlier paired probe. Raw accuracy is `0.6875` for image-native
  `visual_base`, `0.6328` for image-plus-DePlot, and `0.6484` for the initial permissive
  reference-free selector. Agreement
  covers `64/128` and is highly reliable (`0.8906` accuracy). The other `64/128` trigger
  recovery, where accuracy is only `0.4062`. Eight examples abstain. Accepted precision is
  `0.6917`, false-accept rate is `0.3083`, and mean teacher generations are `2.50`.
- The three-attempt offline oracle union reaches `0.7656`, materially above every individual
  configuration. Complementarity therefore exists, but the current reference-free selector
  cannot recover it and is worse than simply using `visual_base`. On conflict examples, base,
  DePlot, and recovery accuracy are `31/64`, `24/64`, and `26/64`; the oracle union is `41/64`.
  The admission gates fail because accepted precision is below `0.80` and the selected policy
  does not improve the base teacher. This adapter is not promoted into a four-epoch run.
- By scope, the calibration selector accuracy is `0.5938` on all-wrong and `0.7031` on
  mixed-wrong examples;
  oracle union is `0.7188/0.8125`. A post-hoc rule analysis is diagnostic only: preferring the
  image-native answer and allowing a DePlot override for table-friendly/validated cases could
  reach up to `0.7266` on this same set, but this number is not a valid held-out result and must
  not be reported as method accuracy. The next step is a separately calibrated selector and a
  held-out probe, not post-hoc tuning on these 128 examples.
- Artifact:
  `outputs/test-fast/teacher-probe-micro-eval/chartqa_recoverable_128_20260715_122656/`.
  Primary files are `harness_summary.csv`, `harness_records.jsonl`,
  `harness_attempts.jsonl`, `prompt_previews.jsonl`, and `manifest.json`.
- The selector is then frozen to an image-primary risk policy: accept initial cross-configuration
  agreement; on conflict, accept only when the recovery attempt confirms the image-native base
  answer; otherwise abstain. On a `seed=29` 128-example near-held-out probe, accepted precision
  is `0.8378` (`62/74`) with selected raw accuracy `0.4844`, abstention `0.4219`, false-accept
  `0.1622`, and mean generations `2.51`. Base/DePlot/oracle-union accuracy is
  `0.6484/0.5938/0.7656`; agreement accuracy remains `0.8889`. The precision target `>=0.80`
  passes, but false acceptance `<0.10` and mean generations `<=2.2` fail, so training integration
  remains blocked. This set overlaps the calibration set on `2/128` examples; a paper-grade
  confirmation must enforce exact disjointness rather than call this fully held out. Artifact:
  `outputs/test-fast/teacher-probe-micro-eval/chartqa_recoverable_heldout128_seed29_20260715_123550/`.
- A Qwen2.5-VL-7B teacher backend plus image-native `Answer:` prefill is a useful cross-family
  prompt diagnostic, but it is not OPD-admissible for the LLaVA student because OPD requires the
  teacher/student tokenization and model family to be compatible. After fixing teacher-probe-only
  normalization for ChartQA surface forms such as footnote stars, parenthesized labels, and
  bracketed list answers, Qwen single-prompt `visual_answer_prefix` accuracy on 128-example probes
  is `0.8594/0.9375/0.8594` for seeds `13/29/31`. Qwen three-view agreement accepts `244/384`
  examples (`63.5%` coverage) with precision `229/244 = 0.9385`. These numbers show that the
  prompt harness can elicit high-quality answers from a stronger VLM, but they must not be used as
  the teacher-quality claim for OPD training. Qwen artifacts include
  `qwen25vl_answer_prefix128_seed13_20260715_132041`,
  `qwen25vl_answer_prefix128_seed29_20260715_131932`,
  `qwen25vl_answer_prefix128_seed31_20260715_135958`,
  `qwen25vl_visual_short128_seed13_20260715_141226`,
  `qwen25vl_visual_short128_seed29_20260715_141227`,
  `qwen25vl_visual_variants128_seed31_20260715_140141`,
  `qwen25vl_answer_prefix_numeric128_seed13_20260715_141449`,
  `qwen25vl_answer_prefix_numeric128_seed29_20260715_141543`, and
  `qwen25vl_answer_prefix_numeric128_seed31_20260715_140921` under
  `outputs/test-fast/teacher-probe-micro-eval/`.
- Re-running the same three-view agreement with the OPD-compatible `llava-7b-ov` teacher fails
  the teacher-quality gate. For seeds `13/29/31`, all3 precision is
  `0.7901/0.8101/0.8171`; aggregate precision is `195/242 = 0.8058` at `242/384 = 0.6302`
  coverage. By scope, all-wrong precision is only `73/98 = 0.7449`, and mixed-wrong precision is
  `122/144 = 0.8472`. Single-view aggregate accuracy is `0.6354` for
  `visual_answer_prefix`, `0.6667` for `visual_short_answer`, and `0.6120` for
  `visual_answer_prefix_numeric`. Therefore the agreement gate remains a default-off diagnostic
  and should not be used for a 4epoch OPD run until an OPD-compatible teacher reaches at least
  `>=0.90` accepted precision, especially on all-wrong candidates. Artifacts:
  `outputs/test-fast/teacher-probe-micro-eval/llava7b_same_lineage_agreement_3seed_summary.csv`
  and `outputs/test-fast/teacher-probe-micro-eval/llava7b_same_lineage_agreement_3seed_accepted.csv`.
- A verifier-backed multi-view harness is a better diagnostic than no-verifier agreement for the
  current RLVR setting. The new micro-eval controls keep the teacher gold-hidden and
  OPD-compatible (`llava-7b-ov`) but add operation-aware image and DePlot prompts; a
  `verifier_first_correct` selector accepts the first output that passes the ChartQA verifier.
  On the same seed31 128-example probe, old three-view oracle union is `95/128 = 0.7422`, while
  the new harness reaches `102/128 = 0.7969` with mean attempts `1.64`. All-wrong coverage
  improves from `39/64 = 0.6094` to `46/64 = 0.7188`; mixed-wrong remains `56/64 = 0.8750`.
  The gains are concentrated in operation-heavy categories: count `0.7143 -> 0.8571`,
  difference `0.5556 -> 0.6667`, average `0.6667 -> 0.7500`, and extreme `0.8125 -> 0.8750`.
  Artifact:
  `outputs/test-fast/teacher-probe-micro-eval/llava7b_verifier_harness128_seed31_20260715_153817/`.
- Teacher-probe-only ChartQA scoring was tightened for benign answer surfaces without changing
  student evaluation: numeric answers with leading unit/context text such as `3 TWh 2000` may
  verify against `3`, and simple trend synonyms such as `decreased` may verify against
  `decreasing`. Rescoring the same seed31 artifact raises selected coverage to
  `104/128 = 0.8125`, rescuing two previously abstained examples. Artifact:
  `outputs/test-fast/teacher-probe-micro-eval/llava7b_verifier_harness128_seed31_20260715_153817/rescore_teacher_probe_parser_20260715/`.
- Adding the existing structured `reasoned_deplot_only` prompt as a verifier-gated final
  fallback raises post-hoc selected coverage to `108/128 = 0.8438`, with all-wrong/mixed-wrong
  coverage `49/64 = 0.7656` and `59/64 = 0.9219`. The four unique fallback wins are
  operation-heavy examples, but the raw fallback view is slow and weak by itself
  (`0.3359` standalone accuracy, high full-hint-format surface), so it should be used only as a
  last-resort verifier-gated fallback. Artifact:
  `outputs/test-fast/teacher-probe-micro-eval/llava7b_verifier_harness_reasoned128_seed31_20260715_160456/`.
- The same four-view policy now has a true `verifier_early_stop` execution mode. On seed31 it
  exactly matches the post-hoc selected result (`108/128 = 0.8438`, mean attempts `1.80`,
  accepted by control: `visual_answer_prefix=87`, `visual_operation_answer_prefix=4`,
  `deplot_operation_answer_prefix=13`, `reasoned_deplot_only=4`) while reducing actual draft
  generations from `512` to `230`; canonicalization is run only for the `24` samples reaching
  `reasoned_deplot_only` instead of all `128`. Artifact:
  `outputs/test-fast/teacher-probe-micro-eval/llava7b_verifier_early_stop_reasoned128_seed31_20260715_161843/`.
  Multi-seed confirmation on seeds `13/29/31` is lower than the seed31 smoke: selected coverage
  is `97/128 = 0.7578`, `105/128 = 0.8203`, and `108/128 = 0.8438`, for an aggregate
  `310/384 = 0.8073` with mean attempts `1.88` and `723` draft generations instead of `1536`.
  Aggregate accepted-by-control is `visual_answer_prefix=249`,
  `visual_operation_answer_prefix=16`, `deplot_operation_answer_prefix=34`, and
  `reasoned_deplot_only=11`. The remaining `74` abstains are concentrated in all-wrong states
  (`54/74`) and operation-heavy qtypes: difference `15`, percent `15`, count `13`, average `7`.
  Artifact:
  `outputs/test-fast/teacher-probe-micro-eval/llava7b_verifier_early_stop_reasoned_3seed_20260715/`.
  This fails the `>=0.90` teacher-quality admission target, so it should not be promoted to a
  full 4epoch training run without another teacher/evidence intervention.
- The follow-up ChartQA harness changes the framing from post-hoc selection to integrated
  closed-loop recovery. Each sample is one verifier-observed trajectory: verifier failures emit
  runtime events, the controller chooses the next evidence recovery operator, and wrong
  candidates are rejected before later operators run. The new table-executable DePlot operator
  is fused into this loop rather than used as an offline answer selector. Full same-lineage
  image-in-the-loop confirmation on seeds `13/29/31` reaches `114/128 = 0.8906`,
  `119/128 = 0.9297`, and `119/128 = 0.9297`, for `352/384 = 0.9167`. This exceeds the
  `>=0.90` teacher-probe target and improves over the previous closed-loop aggregate
  `330/384 = 0.8594` by `+22/384 = +5.73` points. Accepted-by-action confirms that
  `executable_deplot_recovery` contributes `26/384` accepted recoveries inside the trajectory.
- A cross-dataset micro-eval-only harness now supports ChartQA, A-OKVQA, and GSM8K without
  touching trainer routing, OPD loss, student eval, or 4epoch runners. It writes
  `prompt_previews.jsonl`, `records.jsonl`, `summary.csv`, and `manifest.json`, and keeps gold
  answer/hint fields out of teacher prompts. The final smoke uses the OPD-compatible
  `/home/deepseek_VG/deepseek/models/llava-7b-ov` teacher on `64` fixed-seed examples per
  dataset. Results are ChartQA `0.5781` accuracy, parse-fail `0.1250`, DePlot support `0.7344`;
  A-OKVQA `0.8125` accuracy with parse-fail `0.0000`; and GSM8K `0.2656` accuracy with
  parse-fail `0.0000`. A-OKVQA local image paths do not resolve, so that row measures question
  plus choices plus `visual_fact`, not image-native VQA. Artifact:
  `outputs/test-fast/teacher-probe-micro-eval/cross_dataset_llava7b_smoke64_seed13_final_harness_20260715_173251/`.
- Harness ablation note: ChartQA should not decoder-prefill `Operands:`. An intermediate 64-case
  smoke with that prefill dropped ChartQA accuracy to `0.0938` by inducing table-wide sums or
  averages. The final harness keeps the arithmetic schema in the prompt and leaves ChartQA
  `response_prefix` empty; A-OKVQA and GSM8K use `Answer:` response prefixes to keep outputs
  short and parseable. The ChartQA DePlot validator was also hardened from whole-output equation
  regex scanning to bounded line-based parsing after a real run exposed regex backtracking on
  long operator-like teacher output. Focused tests pass `22/22`.

### Completed leakage-fixed gold-hidden CLRC run

- Train tmux: `dyme_gold_hidden_clrc_4e`
- Template/health monitor tmux: `dyme_gold_hidden_clrc_4e_watch`
- Automatic final-eval tmux: `dyme_gold_hidden_clrc_4e_eval`
- Post-eval forensic tmux: `dyme_gold_hidden_clrc_4e_forensics`
- Run ID: `clean_visualfact_fix_opd_main_4epoch_20260714_235824`
- Variant: `deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision`
- Final state: completed on eight H800 GPUs. After two pre-training launch failures exposed a
  missing repository-local model path and stale tmux `CUDA_HOME`, the active launch fixes both
  with explicit existing model paths and `/usr/local/cuda`. Neither failed launch initialized a
  training rank or consumed an optimizer update. The valid final eval
  `eval_final_checkpoint_bsz1_gpuall_20260715_081238` processed `2500/2500` and reached
  accuracy `0.2352`.
- This is a decisive negative result, not an evaluation-format failure. Eval reports all 2,500
  outputs as `other`, with full-CoT, partial-CoT, empty-skeleton, and malformed-Answer template
  counters all zero. The run therefore removes the previous hard-trajectory template collapse
  but does not preserve the `0.5420--0.5800` held-out capability of earlier recipes.
- Final teacher funnel contains `124,450` candidate completions. Teacher correctness is only
  `48.78%` overall, `48.10%` on all-wrong states, and `58.89%` on mixed-wrong states; parse-fail
  is `8.11%`. In contrast, the completed `0.5800` oracle-hint run has `92.48%` teacher
  correctness. This selects gold-hidden teacher recoverability as the next single-factor branch.
- The last 50 optimizer rows are locally healthy but insufficient: accuracy reward `0.2690`,
  global GRPO route about `0.30`, OPD route `0.096`, degenerate rate below `0.02`, and
  GRPO zero-loss `0.94`. Thus the controller does exit OPD and output degeneration is controlled,
  while useful per-group advantage remains sparse. The next run must not change controller,
  student sampling, OPD objective, or optimizer simultaneously with the teacher intervention.
- Frozen runtime audit: `scripts/analysis/check_frozen_run_env.py` checked all 25 invariants with
  no violation. `text_include_gold=false`, teacher trajectory and every hard-SFT route are
  disabled, effective sampling starts at step zero, and the controller reads realized global
  GRPO route with target `.30`, OPD weight `1.5->0.5`, and cap `8->2`.
- Initial latest-two health window: accuracy `0.0801`, global GRPO route `0.0957`, OPD route
  `0.5625`, zero-loss `1.0`, and degenerate rate `0.9219`. This is an early cold-start warning,
  not a stop decision. Full/partial-template, malformed-Answer, teacher trajectory, teacher-SFT,
  legacy online-SFT, and aggregate full-hint hard-target rates are all exactly zero.
- First-10 matched forensic at `2026-07-15 00:12 CST` records accuracy/global-GRPO/OPD
  `0.0555/0.0828/0.5531`, versus historical first-50 `0.0311/0.0338/0.9156` for the interrupted
  clean Oracle run and `0.0120/~0.009/0.4681` for the `0.5800` short-hint baseline. The window is
  too short for an effect claim, but it is not on the registered early-failure trajectory:
  useful global GRPO coverage is higher while OPD is less dominant. Candidate full/partial
  template rates remain zero. Snapshot artifacts are under `gate9_forensics/`.
- Step-20 gate passes. Steps 11--20 reach accuracy/global-GRPO/OPD
  `0.0508/0.0746/0.6094`, all-wrong `0.80`, clipped `0.3480`, degenerate `0.0406`, and EOS
  `0.7250`. The same steps in the interrupted clean Oracle run are
  `0.0043/0.0055/0.9906` with clipped/degenerate `0.8340/0.8531`; the full-trajectory negative
  control is `0/0` with clipped/degenerate approximately `1.0/0.9844`. Thus current clipping is
  a monitored risk, but the joint early-failure pattern is absent. All hard-target and student
  full/partial-template metrics remain zero. Continue unchanged to step 40.
- Step-40 gate is unhealthy for task signal but does not meet the registered stop rule. The
  run-local `gate_40.json` latest-ten window records accuracy/global-GRPO/OPD
  `0.0191/0.0285/0.3750`, all-wrong `1.0`, zero-loss `1.0`, clipped `0.2363`, and degenerate
  `0.0438`. Hard-target, legacy-SFT, teacher-trajectory, full-template, partial-template, and
  malformed-Answer rates remain exactly zero. Because the preregistered recovery stop is only
  evaluated at step 60 and requires degenerate `>0.60`, accuracy `<0.02`, and global GRPO
  `<0.02` simultaneously, the run continues unchanged. Step 41 immediately rebounds to
  accuracy/global-GRPO `0.0664/0.1055`, reinforcing the decision not to stop on a single
  low-signal window. The next decisions are the step-50 checkpoint and step-60 recovery gate.
- Gate-40 candidate forensics isolates the likely bottleneck. Across `10,745` logged probe
  completions, teacher correctness is `50.61%` and parse-fail is `8.45%`; all-wrong teacher
  correctness is `48.91%`, while mixed-wrong reaches `70.83%`. After deduplicating repeated
  completion probes to `(rank, step, prompt)`, correctness is `51.41%`. Real DePlot evidence is
  present, but failures include counting the header as a category, returning multiple table
  values, copying the literal `<short answer>` placeholder, and producing semantically plausible
  long text that the final-answer parser cannot verify. This is far below the old oracle probe
  precision and selects the gold-hidden evidence/probe-quality branch of the preregistered
  contingency if step 60 fails. It does not justify restoring full hints or hard teacher targets.
  Artifacts are under `gate40_forensics/`.
- Checkpoint 50 is complete and contains the model, processor/tokenizer files, scheduler,
  trainer state, and all eight RNG states. Steps 41--50 recover to latest-ten
  accuracy/global-GRPO/OPD `0.0824/0.1074/0.3500`, with degenerate `0.0094` and clipped
  `0.1066`. No intermediate eval is launched because the frozen protocol evaluates the final
  checkpoint; checkpoint 50 is retained for post-final optimization/retention forensics only.
- The registered step-60 recovery gate passes decisively. Steps 51--60 record
  accuracy/global-GRPO/OPD `0.1582/0.1758/0.2500`, all-wrong `0.80`, zero-loss `0.80`,
  clipped `0.1516`, and degenerate `0.0063`. The stop conjunction
  `(degenerate > 0.60) AND (accuracy < 0.02) AND (global GRPO < 0.02)` is false on every
  relevant dimension. Hard-target and template invariants remain zero. The frozen 4epoch run
  therefore continues unchanged; the low-precision gold-hidden teacher remains a final-forensic
  concern, not a reason to interrupt a run whose student task signal is now recovering.
- A same-step historical diagnostic strengthens that decision. Over steps 41--60, the current
  run reaches accuracy/global-GRPO `0.1449/0.1689` with degenerate `0.0078`. The completed
  `0.5800` student-hint-short run reaches `0.0406/0.0187` with degenerate `0.2531` in the same
  optimizer-step window; the `0.5120` full-trajectory negative control reaches
  `0.0184/0.0189` with degenerate `0.3750`. These historical runs are not gold-access-matched
  causal controls, so the comparison is health evidence only. It nevertheless shows that the
  step-60 recovery is materially stronger than the early trajectory of both completed recipes.
- Output behavior is becoming short rather than templated. Mean completion length falls from
  `43.89` tokens in steps 1--20 to `18.28` in steps 41--60; EOS rises from `0.7016` to
  `0.8875`, and `format_without_thinking_rate` reaches `0.9969`. Full/partial-template and
  malformed-Answer rates remain zero. This is not the previous fixed full-CoT collapse, but it
  may represent an answer-shortcut policy. It is warning-only during training: final eval must
  report output types, answer extraction failures, and accuracy conditioned on short versus
  structured outputs before the paper describes the learned behavior as improved reasoning.
- Checkpoint 100 is complete. Steps 91--100 record accuracy/global-GRPO/OPD
  `0.1570/0.1773/0.2715`, zero-loss `0.80`, degenerate `0.0031`, clipped `0.0`, and EOS `1.0`.
  Mean completion length is now `6.62` tokens. Over the same steps 81--100, current accuracy is
  `0.1404`, versus `0.0715` for the completed `0.5800` student-hint-short run and `0.1008` for
  the `0.5120` full-trajectory negative control. Current global GRPO is `0.1688`, versus
  `0.0703/0.1014`. These are historical health comparisons, not matched causal evidence. The
  run continues because short outputs terminate cleanly and improve task accuracy without
  template or hard-target leakage; the final paper must call this answer reliability unless
  held-out output analysis demonstrates genuine structured reasoning gains.
- Candidate-log diversity forensics reveals a distinct late risk. Among complete all-wrong
  groups, the mean number of unique student completions falls from `7.42/8` in steps 0--20 to
  `3.47/8` in steps 91--103. The fraction whose eight samples are all identical rises from
  `0.69%` to `28.83%`, and mean modal-answer share rises from `0.191` to `0.641`. All recent
  probed outputs contain an `Answer:` marker, so this is not parser failure or empty-output
  collapse; it is within-prompt exploration contraction. The current run remains above the
  historical health trajectories and therefore continues. If final accuracy is below `0.60`
  with high zero-loss, this evidence selects generation diversity/entropy as the next single
  intervention rather than stronger teacher supervision.
- Checkpoint 150 is complete. Steps 126--150 reach accuracy/global-GRPO/OPD
  `0.1236/0.1503/0.2911`, versus `0.0653/0.1013/0.4500` for the completed `0.5800`
  historical recipe in the same optimizer window. Zero-loss remains high at `0.96`, but
  degenerate is `0.0125` and mean completion length remains a clean `6.54` tokens. Importantly,
  all-wrong diversity no longer decreases monotonically: mean unique completions recover from
  `3.03/8` in steps 101--125 to `3.35/8` in steps 126--150 and `3.56/8` in steps 141--150;
  all-identical group rate falls to `21.05%` in the latest ten. Controller mastery is `0.1817`,
  OPD weight `0.844`, and cap `5`. The run continues; the current middle-stage limitation is
  effective GRPO advantage, not template, clipping, degeneration, or irreversible diversity
  collapse.
- The monitor evaluates the latest ten rows every five minutes and records GPU telemetry. Hard
  imitation or the registered joint template-collapse condition stops the train session. A
  successful final checkpoint automatically launches 8-GPU ChartQA eval with batch size one;
  after a valid summary, the forensic stage compares the run against the `0.5800` short-hint
  baseline and the `0.5120` full-trajectory negative control.

### Superseded oracle resilient queue

- Train tmux: `dyme_no_full_hint_resilient_181613`
- Template/health monitor tmux: `dyme_no_full_hint_resilient_181613_watch`
- Post-eval forensic tmux: `dyme_no_full_hint_resilient_181613_forensics`
- Run ID: `oracle_opd_no_full_hint_hard_sft_adaptive_resilient_4epoch_20260713_181613`
- Current state: superseded by the leakage-fixed gold-hidden run above. The original gate opened at `2026-07-14 15:01 CST`,
  but the first launch stopped before model loading because the inherited default student path
  pointed to the nonexistent repository-local `models/` directory. This was a non-training
  prelaunch failure: no optimizer row, checkpoint, or candidate record was produced. The failed
  attempt log is preserved as `resilient/attempt_1_model_path_failure.log`. At
  `2026-07-14 19:24 CST`, the complete train, preflight, health-watch, automatic 8-GPU eval, and
  post-eval forensic chain was restored with explicit existing student/teacher model paths and
  candidate logging enabled. It was not allowed to start after the visual-fact leakage audit
  promoted the gold-hidden variant to the main run. No final result is assigned to this queue.
- Hard-target invariants cover teacher trajectory, teacher-SFT repair, legacy online-SFT slot,
  forced replacement, and aggregate full-hint hard-target exposure; every rate must remain zero.
- A `2026-07-14` observability audit found that clean variants with
  `teacher_trajectory.enabled=false` could still log the configured base value `0.5` as
  `loss/teacher_traj_effective_weight`, despite never constructing or applying trajectory loss.
  The trainer now logs the controller/decay value separately as
  `loss/teacher_traj_scheduled_weight`; `loss/teacher_traj_effective_weight` is non-zero only
  when the feature is enabled and the synchronized global trajectory count is positive. A
  one-step GPU smoke of `gold_hidden_uncond_opd` records scheduled/effective `0.5/0.0`, and the
  external health checker returns `status=ok`. This is a monitoring-semantics correction, not
  a recipe or optimization change; the frozen Oracle run remains numerically unchanged.
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
