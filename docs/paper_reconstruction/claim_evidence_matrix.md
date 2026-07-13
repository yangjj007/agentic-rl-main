# CLRC Claim-Evidence Matrix

更新时间：2026-07-12

本文档是论文强主张的准入表。中文初稿、摘要、贡献和结论只能把状态为
`verified` 的效果主张写成既成事实。

## 状态定义

- `verified`：完整训练与可复现 eval artifact 已直接支持。
- `partial`：实现、测试或训练日志支持机制存在，但尚无完整主结果或必要消融。
- `running`：实验正在运行，不得在摘要中引用结果。
- `missing`：尚无直接证据，必须安排实验。

## Gold-access 术语审计

当前 teacher-probe 实现需要区分两个层面的 gold access：

1. **Teacher-input access**：`format_only,visual_facts_deplot` 不把 reference answer 拼入 teacher prompt；oracle variant 额外使用 `oracle_hint`。
2. **Routing-verifier access**：`DyMETrainer._apply_teacher_probe_routing` 在 teacher 生成答案后，将 `reference` 传给 ChartQA verifier，并以 `score > 0` 决定 OPD，否则进入 SFT。

因此非 oracle variant 的准确名称是 **gold-hidden teacher / verifier-available routing**，而不是“完全无 gold recoverability estimator”。这与 RLVR 使用 reference/verifier 计算 student reward 的设置兼容，但论文和主表必须分别披露 `Teacher sees gold` 与 `Verifier uses reference`。

## 方法与实现主张

| Claim | Status | Direct evidence | Required next evidence |
|---|---|---|---|
| 跨 data-parallel rank 归约最终互斥 completion routes，并生成一致的 global training snapshot | `verified` | `opsd_utils/global_training_signal.py`; `tests/test_global_training_signal.py`; `tests/test_global_training_signal_trainer.py`; 8-GPU smoke `global_signal_smoke_20260712_1830` | 无 |
| realized global GRPO completion route rate 可作为单一控制信号 | `verified` | `opsd_utils/adaptive_supervision.py`; `tests/test_adaptive_supervision.py`; `tests/test_adaptive_supervision_trainer.py` | 仍需效果消融证明它优于其他状态变量 |
| step `t` 的最终 route snapshot 只控制 step `t+1`，避免同一步循环反馈 | `verified` | `trainer/DyMETrainer.py`; direct-route trainer tests | 无 |
| mastery 单调，OPD weight/cap 可由同一 snapshot 连续控制 | `verified` | controller unit tests；4-step smoke metrics | weight-only、cap-only 与 joint OPD action 消融；hard trajectory 不属于主 controller action |
| 控制器语义不依赖固定 epoch 或绝对 step boundary | `verified`（机制） | controller API 不读取 epoch/max steps；duplicate-step/epoch-label tests | 改变 epoch、batch size、数据量的效果稳健性实验 |
| completion-level GRPO/OPD/SFT 三路路由已实现 | `verified`（机制） | trainer final route accounting；global snapshot route counts；teacher probe routing logs | routing ablation 与统一预算 final eval |
| no-hard-imitation OPD run 可在线区分完整 CoT、partial Goal drift、空骨架与 malformed Answer | `verified`（机制） | `opsd_utils/diagnostics.py`; `scripts/analysis/check_opd_template_health.py`; 31 tests；active candidate/log monitor | 完整 run 与 final eval，验证这些行为指标和 held-out error 的关系 |
| 本文是首个将 OPD 用于 VLM reasoning 的工作 | `contradicted` | VOLD (`2510.23497`)、Decomposed OPD (`2606.00564`) 与 VA-OPD (`2605.21924`) 已直接研究 VLM on-policy distillation | 禁止使用该表述 |
| 本文首次在 sub-1B VLM 可验证推理中系统引入和评估 OPD | `partial` | 当前 student 为 0.5B；现有近邻主要使用更大 VLM 或研究 grounding/token weighting；已有 OPD/no-OPD 内部对照线索 | 完整文献排查；统一 4epoch OPD vs no-OPD；至少一个 sub-1B baseline；最终稿使用“to our knowledge”限定 |
| OPD 与 GRPO/SFT 在 sub-1B VLM 中具有互补性 | `partial` | 三路 route 与 loss 已实现；step 71-100 显示 GRPO coverage/accuracy 恢复 | no-OPD、unconditional OPD、OPD-only、GRPO-only、full three-route 消融及 final eval |
| gold-hidden-teacher multimodal recoverability routing 是论文主方法 | `partial` | teacher provider 路径与 leakage metrics 已存在；routing verifier 仍使用 reference | `teacher/privileged_suffix_has_gold_rate=0.0` 的完整 4epoch CLRC 主实验，并披露 verifier access |
| 当前 global-GRPO controller full run 属于 oracle/privileged CLRC，而非 gold-hidden-teacher CLRC | `verified` | 当前 variant 使用 `oracle_hint`，并要求记录 `teacher/privileged_suffix_has_gold_rate` | 该 run 只登记为 oracle CLRC upper bound；完成后核验实际 gold-rate |
| recoverability routing 完全不使用 reference answer | `contradicted` | `trainer/DyMETrainer.py:1983-1999` 将 `reference` 传给 teacher-probe correctness verifier，并以 `score > 0` 决定 OPD/SFT | 若要支持该 claim，需实现 reference-free evidence-consistency/uncertainty gate；否则保持 gold-hidden teacher / verifier-available routing 表述 |

## 效果主张

| Claim | Status | Direct evidence | Required next evidence |
|---|---|---|---|
| gold-hidden-teacher PCD aligned 4epoch 达到 ChartQA `0.5420` | `verified` | legacy run ID `pcd_no_visual_aligned_4epoch`；`outputs/test-fast/pcd-no-visual/pcd_no_visual_aligned_4epoch/deplot_no_vs_opd_pcd/eval_chartqa/summary.csv`，`2500/2500` | 主表注明 verifier uses reference |
| oracle route_guard 4epoch 达到 `0.5592` | `verified` | `outputs/test-fast/pcd-no-visual/route_guard_oracle_hint_4epoch_7gpu/deplot_no_vs_opd_pcd_oracle_hint_route_guard/eval_chartqa/summary.csv`，`2500/2500` | 无 |
| oracle teacher-SFT repair full template 达到 `0.5624` | `verified` | `outputs/test-fast/pcd-no-visual/pcd_oracle_teacher_sft_repair_4epoch/deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair/eval_chartqa/summary.csv`，`2500/2500` | 无 |
| constrained repair 达到 `0.5656` | `verified` | `outputs/test-fast/pcd-no-visual/pcd_oracle_teacher_sft_repair_constrained_4epoch_rerun/deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair/eval_chartqa/summary.csv`，`2500/2500` | 无 |
| student_hint_short 达到 `0.5800` | `verified` | `outputs/test-fast/pcd-no-visual/pcd_oracle_teacher_sft_repair_student_hint_short_4epoch/deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short/eval_chartqa/summary.csv`，有效行为 `eval_final_checkpoint_bsz1_gpu0_20260709_192652`，`2500/2500` | 无 |
| oracle official final 达到 `0.5872` | `verified` | `outputs/test-fast/pcd-no-visual/pcd_oracle_hint_official_4epoch/deplot_no_vs_opd_pcd_oracle_hint/eval_chartqa/summary.csv`，checkpoint-588 与 final 均为 `0.5872` | 无 |
| oracle OPD + full-trajectory adaptive run 超过 oracle official | `contradicted` | run `global_grpo_route_full_4epoch_20260712_205549` final eval 为 `0.5120`，低于 oracle official `0.5872` | 将该 run 作为 hard trajectory 与 OPD 混合的负结果；下一轮隔离 OPD，关闭 trajectory/SFT repair |
| gold-hidden-teacher CLRC 优于统一预算 DyME | `missing` | 无完整 gold-hidden-teacher CLRC run | gold-hidden DyME/recoverability-only/full-CLRC 统一 4epoch 对照，并固定 verifier reference access |
| CLRC ChartQA test accuracy 超过 `0.60` | `missing` | 当前无任何 `>0.60` 完整 eval artifact | 完整 `summary.csv`，processed 至少 `2496/2500`，优先要求 `2500/2500` |
| full teacher-trajectory supervision 与 OPD 联合会改善 held-out accuracy | `contradicted` | 该 run last50 train accuracy 约 `0.445`，但 final eval 仅 `0.5120`，且 `2397/2500` 输出为 full-CoT 模板 | matched no-hard-imitation OPD run；按正确/错误交叉分析模板，而非惩罚 full-CoT 本身 |
| no-hard-imitation OPD 改善早期 task signal relative to matched historical runs | `running` | steps 31--40 accuracy/GRPO 为 `0.0176/0.0187`，高于 full-trajectory `0.0051/0.0031` 与 student-hint-short `0.0094/0.0094` | 后续窗口、完整训练与 final eval；早期窗口不得写入摘要 |
| CLRC 降低 all-wrong 与 task zero-loss | `running` | 当前 run 已记录 global task/route metrics | 完整训练窗口对照及相同配置无 controller 消融 |
| CLRC 以更少 teacher compute 达到相同或更高准确率 | `missing` | 现有日志有 teacher calls/tokens 字段但未统一汇总 | teacher calls、generated tokens、GPU hours 的 Pareto 对照 |
| CLRC 对 epoch/batch/data scale 变化比 fixed-step schedule 稳健 | `missing` | 机制不依赖固定 step，但无效果证据 | 至少两个改变训练尺度的配对实验 |

## 写作准入规则

1. `verified` 的机制主张可以写入方法与贡献。
2. `verified` 的效果主张可以写入结果章节，但必须保留预算、gold access 和 processed count。
3. `partial` 只能写成“我们实现/提出并将在消融中验证”，不能写“显著提升”。
4. `running` 只能出现在实验进度或附录复现说明。
5. `missing` 不得进入摘要的结果句。
6. oracle/privileged 结果不能用于证明 gold-hidden-teacher effectiveness 或无 teacher-input leakage。
7. `teacher/privileged_suffix_has_gold_rate=0` 只证明 teacher prompt 不含 gold，不证明 routing verifier 未使用 reference。
8. 禁止写“first VLM OPD”；允许的候选表述是“to our knowledge, the first systematic introduction and evaluation of OPD for sub-1B VLM reasoning under verifiable rewards”，且在消融与文献排查完成前只能作为待验证定位。
9. 论文只保留一个核心创新：OPD 在 sub-1B VLM 可验证推理中的引入、有效性与互补性。三路路由和闭环控制器应写成使 OPD 可靠介入并逐步退出的支撑机制，不得与 OPD 并列为互不相关的主创新。

## 当前摘要允许写入的结果

在当前 CLRC eval 完成前，摘要最多可以写：

> 现有 oracle-guided 4epoch 基线最高达到 58.00%，oracle official 为 58.72%；CLRC 的完整效果仍在统一预算实验中验证。

不得写“CLRC 已超过 60%”“CLRC 已优于 DyME”或“CLRC 已减少 teacher compute”。
