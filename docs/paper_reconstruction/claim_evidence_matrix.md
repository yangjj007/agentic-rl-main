# CLRC Claim-Evidence Matrix

更新时间：2026-07-14

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
| 当前 adaptive cap 可降低 teacher calls 或 generated tokens | `contradicted` | `DyMETrainer._apply_teacher_probe_routing` 先完成 teacher generation，随后才调用 `apply_opd_route_cap`；该函数 docstring 明确为 post-probe | 论文只能称其控制 OPD loss exposure；若需 compute claim，TDD 实现 pre-probe candidate cap 并做 matched eval |
| 控制器语义不依赖固定 epoch 或绝对 step boundary | `verified`（机制） | controller API 不读取 epoch/max steps；duplicate-step/epoch-label tests | 改变 epoch、batch size、数据量的效果稳健性实验 |
| completion-level GRPO/OPD/SFT 三路路由已实现 | `verified`（机制） | trainer final route accounting；global snapshot route counts；teacher probe routing logs | routing ablation 与统一预算 final eval |
| no-hard-imitation OPD run 可在线区分完整 CoT、partial Goal drift、空骨架与 malformed Answer | `verified`（机制） | `opsd_utils/diagnostics.py`; `scripts/analysis/check_opd_template_health.py`; 31 tests；clean interrupted run 的 21,183 条 wrong-candidate `student_output` 审计显示 late partial heading `44.82%`、full four-heading `0.0%`、Answer heading `88.50%` | 完整 run 与 final eval，验证这些行为指标和 held-out error 的条件关系；teacher-output template rate 不得冒充 student-output collapse rate |
| no-full-hint OPD 模式可严格阻断 slot、route、forced 三类 dataset-hint hard replacement | `verified`（机制） | forced bypass 的 RED 测试；`tests/test_no_full_hint_hard_sft_gate.py`; `tests/test_online_sft_source_metrics.py`; 164-test focused suite；4-step 8-GPU smoke及 clean run 至 step 86 的 hard-target rate 最大值均为 `0.0` | resilient 4epoch rerun 全程 invariant 与 final eval |
| P0 gold-hidden matched 因果矩阵可由当前 runner 直接复现 | `partial/mechanism-ready` | route-matched no-OPD、pure GRPO、OPD-only、fallback-only、unconditional OPD、fixed/adaptive routed OPD、target `.20`、token-reliability 与 hard-replay diagnostic 已显式拆分；`2026-07-14` focused matrix/loss suite `29 passed`。route-matched no-OPD 的 1-step smoke 保留 OPD route `0.75` 但 OPD weight/loss 为零；hard-replay smoke 在 all-wrong batch 上 skip rate `1.0`。SSOPD/VOLD 完整语义仍未实现 | 先为 E1--E6 做 2-step 8-GPU route/runtime smoke；token reliability 只作探索项；不得把 hard-replay 诊断标记为 SSOPD |
| OPD-only 与 fallback-only 标签提供严格单信号消融 | `partial; fallback contradicted` | `_combine_grpo_opsd_losses` 已修复 OPD-only 在无 OPD sample batch 上的 GRPO 泄漏；但 fallback-only 使用 `mode=dyme`，生成阶段不附加 `opsd_mask`，compute_loss 不进入该 mixer，因此 runner 导出的 `GRPO_WEIGHT=0` 尚不能证明实际 RL loss 为零 | 在 loss 顶层、独立于 `opsd_mask` 应用 GRPO base weight，或为 fallback-only 提供显式 base-loss mask；增加真实 compute_loss 测试和 effective loss-contribution 日志后才可称严格单信号 |
| 当前 mixed-group hard replay 是 SSOPD matched baseline | `contradicted/isolated` | 当前 `opsd_utils/hard_replay.py` 选择 shortest-correct token sequence；trainer 将其作为 hard CE target 替换所有 mixed-wrong completion。它已重命名为 `mixed_group_shortest_correct_hard_replay`，旧 SSOPD/VOLD runner 标签会明确失败；仍没有 longest-wrong target、wrong-prefix conditional teacher distribution 或 frontier weighting | 只把该路径作为 hard-replay 机制诊断；按 SSOPD 原目标完整实现并通过 matched smoke 后，才可恢复 SSOPD 论文标签 |
| 本文是首个将 OPD 用于 VLM reasoning 的工作 | `contradicted` | VOLD (`2510.23497`)、REOPOLD (`2603.11137`)、Decomposed OPD (`2606.00564`) 与 VA-OPD (`2605.21924`) 已直接研究 VLM/visual-reasoning on-policy distillation；REOPOLD 明确覆盖 3B/7B student | 禁止使用该表述 |
| 本文首次提出 multimodal recoverability routing 或首次将 OPD 与 RLVR 结合 | `contradicted` | ViCuR (`2606.05718`) 已研究 recoverable visual privilege；VOLD (`2510.23497`) 已联合 cold-start、GRPO 与 OPD | 禁止把 recoverability、OPD+RLVR 或 curriculum 作为无条件 first claim；差异必须限定到 sub-1B、all-wrong verifier-confirmed completion routing 与 realized route occupancy controller |
| 本文首次提出 verifier/reward-gated OPD | `contradicted` | RG-OPD (`2607.04037`) 已使用 verifier feedback 估计 teacher 相对 student 的可靠性并门控 OPD；其任务为 reasoning/coding LLM，而非 sub-1B VLM | 禁止把 verifier gate 本身作为 first claim；本文只能检验它在 0.5B multimodal all-wrong route 中是否有效，并以 matched unconditional OPD 消融证明必要性 |
| 本文首次利用 correct/wrong completion 状态构造稠密过程监督 | `contradicted` | SSOPD (`2605.17497`) 已在 mixed group 中用最短正确 completion 的条件 teacher distribution 纠正最长错误 completion | 禁止使用宽泛 completion-state first claim；只讨论 SSOPD 信号缺失的 all-wrong group，以及外部 privileged teacher recoverability 是否提供净收益 |
| 本文首次在 sub-1B VLM 可验证推理中系统引入和评估 OPD | `partial` | 当前 student 为 0.5B；截至 2026-07-14 的定向 arXiv 检索已纳入 REOPOLD 与 privileged-OPD failure analysis，仍未发现直接覆盖 sub-1B VLM RLVR matched OPD 净收益/互补性的问题设置；VOLD/REOPOLD/ViCuR/VA-OPD/Decomposed-OPD 已明确否定宽泛 first claim | 统一 4epoch matched OPD vs no-OPD；至少一个 sub-1B baseline；OPD/GRPO/fallback 互补性消融；最终稿必须使用“to our knowledge”限定，并在投稿前再次检索 |
| privileged teacher OPD 在关键推理 token 上天然安全 | `contradicted` | Kaur et al. (`2607.05184`) 报告 privileged teacher/student 在 reasoning fork token 上的策略差异可使 naive OPD 伤害 thinking model；本文自己的 full hard-trajectory 负对照也出现 `0.5120` 与固定模板污染 | 必须保留 teacher correctness/recoverability gate、hard-trajectory isolation 和 token/style forensics；若主实验仍低于 no-OPD，优先检查 teacher-student state mismatch 而非扩大 OPD exposure |
| OPD 与 GRPO/SFT 在 sub-1B VLM 中具有互补性 | `partial` | 三路 route 与 loss 已实现；step 71-100 显示 GRPO coverage/accuracy 恢复 | no-OPD、unconditional OPD、OPD-only、GRPO-only、full three-route 消融及 final eval |
| CLRC 优于 VOLD-style cold-start alignment + online joint GRPO/OPD | `missing` | 当前 clean variant 没有 matched VOLD-style cold-start comparator | 相同初始化来源、总 updates、generation、teacher tokens 与 hard-target exposure；单独匹配 cold-start 预算，最强近邻完整 4epoch eval |
| answer-verifier completion recoverability 优于 ViCuR-style visual recoverability | `missing` | 当前实现 answer-verifier routing；没有 matched visual-cue recoverability gate | 匹配 teacher/evidence/calls/reference disclosure 的 comparator，报告 accepted precision/coverage 与 final accuracy |
| all-wrong privileged-teacher OPD 优于 SSOPD-style mixed-group self-distillation | `missing` | 当前 routing 同时记录 mixed/all-wrong group，但没有 matched SSOPD comparator | mixed groups 使用 correct-to-wrong self-distillation、all-wrong groups无外部 teacher的 matched screening；与完整方法报告按 group type 的 accuracy/zero-loss/route coverage |
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
| CLRC 达到或超过 DyME 的 ChartQA 效果 | `missing` | DyME 原文在 LLaVA-OV-S 0.5B、ChartQA relaxed correctness 上报告 Pure DyME `0.649`（Medium CoT）和 full DyME `0.675`（含 Visual Supervision）；matched Pure/Full DyME config 与统一 runner 均已实现，配置深比较确认 Full 只增加 Visual Supervision，`135` 项组合回归及无 GPU dry-run 通过，但尚未排队或产生 eval；当前内部最好 `0.5872` | 分别运行统一模型、数据、decode、4epoch 与 eval 的 Pure DyME 和 Full DyME reproduction；`>0.60` 仅是工程突破线，不能作为 DyME parity |
| 主方法相对 oracle official 或 matched no-OPD 的提升具有配对统计可靠性 | `missing/not implemented` | 当前 eval 日志没有稳定 sample ID，只打印 prediction/reference/correct；不能安全对齐不同 run | 保存确定性逐样本 JSONL；paired bootstrap 95% CI、McNemar test、四格正确性计数与分层 error delta |
| full teacher-trajectory supervision 与 OPD 联合会改善 held-out accuracy | `contradicted` | 该 run last50 train accuracy 约 `0.445`，但 final eval 仅 `0.5120`，且 `2397/2500` 输出为 full-CoT 模板 | matched no-hard-imitation OPD run；按正确/错误交叉分析模板，而非惩罚 full-CoT 本身 |
| no-hard-imitation OPD 改善早期 task signal relative to matched historical runs | `contradicted` | steps 31--40 曾短暂升至 `0.0176/0.0187`，但 steps 51--60 accuracy/GRPO 仅 `0.0070/0.0156`，degenerate rate `0.9156`，恢复闸门失败并停跑 | 该 run 只作为“关闭硬轨迹仍不足”的负结果，不进入效果主表 |
| 关闭 full teacher trajectory 足以消除固定模板污染 | `contradicted` | 当前 run 的 teacher trajectory/SFT-repair 权重严格为零，但 legacy online SFT 仍以 dataset `hint + answer` 硬监督约一半 completion；ChartQA hint 本身是完整四段 CoT | 下一轮先关闭 legacy online-SFT slots，才能把残余 drift 归因于 OPD soft token matching |
| 关闭全部 full-hint hard SFT 后，OPD 能在统一 4epoch 预算超过 `0.60` | `missing` | clean smoke 已证明机制隔离；首个正式 run 在 step 86 因 rank 7 CUDA 瞬态错误中止，无 checkpoint/final eval | 相同配方 resilient 4epoch 重跑；自动 final ChartQA eval，processed 至少 `2496/2500`，优先 `2500/2500` |
| CLRC 降低 all-wrong 与 task zero-loss | `partial` | 中止 run 在 steps 72--81 显示 accuracy/GRPO recovery，但 task zero-loss 仍高；不能作为 final 效果 | 完整训练窗口对照及相同配置无 controller 消融 |
| CLRC controller 以更少 teacher compute 达到相同或更高准确率 | `missing/not implemented` | 当前 controller 只调 OPD weight 和 post-probe route cap，不减少已发生的 teacher generation | 先实现 pre-probe adaptive candidate cap，再做 teacher calls、generated tokens、GPU hours 的 matched Pareto；否则禁止该 claim |
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
10. `>0.60` 不等于优于或达到 DyME。DyME parity 必须引用统一协议的 reproduction；
    外部原文 `0.649/0.675` 只作为量级检查，并明确 Pure/full 与 Visual Supervision 差异。

## 当前摘要允许写入的结果

在当前 CLRC eval 完成前，摘要最多可以写：

> 现有 oracle-guided 4epoch 基线最高达到 58.00%，oracle official 为 58.72%；CLRC 的完整效果仍在统一预算实验中验证。

不得写“CLRC 已超过 60%”“CLRC 已优于 DyME”或“CLRC 已减少 teacher compute”。即使
未来得到 `60.x%`，在 matched DyME reproduction 完成前也只能写成超过内部工程阈值。
