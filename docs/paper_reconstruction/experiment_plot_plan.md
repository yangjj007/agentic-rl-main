# OPD for Sub-Billion VLM 实验与图表设计计划

更新时间：2026-07-13

本文件只定义论文图表需要回答的问题、数据字段和准入条件。实验命令与 run 状态见
`docs/opd_experiment_plan.md`，已完成数字见 `experiment_ledger.md`。

## 1. 图表叙事原则

论文的证据链按以下顺序展开：

1. matched no-OPD 对照证明 OPD 在统一 4epoch 预算下具有净收益。
2. OPD-only、GRPO-only、fallback-only 与联合训练证明信号互补性。
3. unconditional OPD 与 verifier-routed OPD 证明可靠性路由不是额外 teacher loss 的包装。
4. 全局控制器和动作拆分解释如何让 OPD 随学生自主性介入和退出，并报告 teacher compute。
5. token-selective OPD 近邻基线与训练尺度分析界定本文方法边界。

任何 oracle/privileged 结果都必须在主图中显式标记，不与 gold-hidden-teacher 方法合并平均。`Teacher sees gold` 与 `Verifier uses reference` 必须分列。

## 1.1 Motivation-to-Evidence Contract

每条 motivation 必须产生一个可证伪预测。若对应实验不成立，应收缩论文 claim，不能仅用 final accuracy 覆盖机制失败。

| Motivation | CLRC prediction | Required comparison | Supporting evidence | Falsification condition |
|---|---|---|---|---|
| total reward 可掩盖 task signal 消失 | realized GRPO route 比 total-reward mixed/variance 更准确地反映学生自主覆盖率 | controller state: total-reward proxy vs task mixed/zero proxy vs global final GRPO route | state 与 next-window task accuracy、all-wrong、zero-loss 的相关/滞后预测；global/local disagreement | total-reward proxy 在多 seed 上同样或更好地预测 task recovery 与 final accuracy |
| OPD 填补 SFT 与 RLVR 之间的学生错误状态 | matched verifier-routed OPD 优于 no-OPD | no-OPD vs verifier-routed OPD，固定其余训练流程与 teacher budget accounting | final accuracy、zero-loss、all-wrong recovery、OPD loss/coverage | matched no-OPD 相同或更好 |
| student-state guidance 比 teacher-sequence imitation 更适合小模型 | OPD-only guidance 优于或至少不弱于 OPD + hard trajectory，并减少空模板塌缩 | verifier-routed OPD without hard imitation vs matched OPD + full trajectory | held-out accuracy、train accuracy、full-template/empty-skeleton/malformed-answer rates | hard trajectory 在相同预算下提高 held-out accuracy且不增加模板失败 |
| OPD 与现有信号互补 | 完整联合训练优于任一单信号或两信号组合 | OPD-only、GRPO-only、fallback-only、OPD+GRPO、full | final accuracy、route occupancy、各状态分层收益 | 任一单信号在相同预算下达到同等或更高效果 |
| all-wrong 中只有部分状态适合 OPD | verifier-routed OPD 优于 unconditional OPD | all-wrong 全 OPD vs verifier-confirmed OPD | accepted teacher accuracy、coverage、teacher tokens、final accuracy | unconditional OPD 在 matched compute 下同样或更好 |
| fixed clock 不代表学生能力 | 改变 batch/data scale 后，CLRC 在相近 autonomy state 触发动作，而 fixed-step 在不同能力状态触发 | fixed step vs normalized progress vs CLRC，至少两种 scale | trigger step、normalized progress、GRPO EMA、task accuracy、final accuracy | CLRC trigger autonomy state 方差不低于 fixed/progress，且无 final/compute 收益 |
| teacher support 应随真实自主性减少 | joint controller 在保持或提升 accuracy 时降低 teacher calls/tokens | fixed support vs OPD-weight-only vs cap-only vs joint actions | accuracy/teacher-token Pareto、last50 routes、controller trajectory | joint controller 被 fixed support Pareto 支配 |

主结果超过 baseline 但上述预测均不成立时，只能将方法描述为一个有效 recipe，不能声称 recoverability curriculum 或 closed-loop mechanism 得到验证。

## 2. Figure 1：方法总览

### 左侧：Completion-Level Recoverability Routing

显示同一 prompt 的 `K` 个 student completions：

- correct completion -> GRPO；
- wrong + gold-hidden teacher answer verified correct -> OPD；
- wrong + teacher unrecoverable/quality fail -> fallback or skip。

图中 teacher 输入只画 image、question、DePlot/visual facts 和 format constraint，不画 gold answer；teacher 输出后单独画 verifier，并明确其在当前 RLVR 设置下可访问 reference。

### 右侧：Global Realized-Autonomy Feedback

显示各 rank 最终互斥 route counts 经 all-reduce 得到：

```text
a_t = N_grpo / N_total
```

再经 EMA、monotonic mastery、smoothstep 输出 OPD weight 和 OPD cap。图旁明确标注
主方法的 hard-trajectory weight 恒为零；用箭头标明 step `t` snapshot 控制 step `t+1`。

## 3. Table 1：统一主结果

列定义：

| Method | Student | Teacher | Epoch | Evidence | Teacher sees gold | Verifier uses reference | Controller | Teacher calls | Teacher tokens | ChartQA acc | Processed |
|---|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|

最低行集合：

- Base/SFT；
- DyME official；
- gold-hidden-teacher PCD aligned；
- gold-hidden-teacher fixed schedule；
- gold-hidden-teacher recoverability-only；
- gold-hidden-teacher full CLRC；
- oracle student_hint_short；
- oracle official；
- oracle CLRC upper bound。

主表或紧邻的核心消融表必须直接出现 matched `no-OPD` 与
`verifier-routed OPD`；否则主结果无法支撑论文标题中的 OPD claim。

主表不使用训练 reward 替代 test accuracy。正式数字要求 `2500/2500`；若先使用 `2496/2500` 快速结果，必须在 Processed 列标记并在最终稿补齐。

## 4. Figure 2：Accuracy / Teacher-Compute Pareto

横轴：每个训练 prompt 的 teacher generated tokens，或总 teacher generated tokens。

纵轴：ChartQA held-out accuracy。

点大小：GPU hours。点颜色：gold-hidden teacher、oracle、no teacher；点形状区分 verifier 是否使用 reference。比较 fixed schedule、recoverability-only、full CLRC 和 action ablations。

该图只有在统一统计 teacher calls/tokens 后才能进入主文。若当前日志缺失总量，先放 appendix 计划，不用 candidate accuracy 代替 compute。

## 5. Figure 3：闭环训练动力学

使用同一横轴和相同平滑窗口，至少包含四个 panel：

1. `global_signal/grpo_route_rate`、`opd_route_rate`、`sft_route_rate`；
2. `adaptive/signal_rate`、`signal_ema`、`mastery`、`supervision`；
3. OPD weight、OPD cap，以及作为负对照记录的 trajectory weight；主 OPD recipe
   中 trajectory weight 应始终为零；
4. task all-wrong、task zero-loss、total-reward zero-loss、disagreement、accuracy。

辅助 panel：clip、EOS、degenerate 和 teacher candidate/correct/accepted funnel。

行为 panel 额外绘制 `full_cot_template_rate`、`partial_cot_template_rate`、
`goal_without_answer_rate`、`empty_cot_skeleton_rate` 和
`malformed_answer_section_rate`。完整 CoT 比例本身不作为失败；只有空骨架或异常答案
与完整模板共同升高时才定义为 template collapse。来自 teacher-probe candidate JSONL 的
partial/Goal-without-Answer 曲线必须标注为“conditioned on wrong probed completions”，不得
与全体 rollout 比例混用。另以虚线画出
`teacher_traj_effective_weight` 和 `teacher_sft_repair_rate`，证明 OPD isolation run
没有重新引入 hard imitation。

禁止只画 rank-local `routing/*` 来证明全局控制器状态；主曲线必须来自 `global_signal/*`。

## 6. Figure 4：训练尺度稳健性

比较 fixed-step schedule 与 CLRC，在至少两种训练尺度下记录动作发生时的：

- absolute step；
- normalized progress；
- global GRPO route EMA；
- task accuracy/all-wrong；
- final accuracy。

理想证据是 fixed-step 在不同规模下对应不同学生能力，而 CLRC 在接近的 realized autonomy state 下产生相近动作。

## 7. Table 2：OPD 核心消融

### A. Signal Contribution

- no-OPD；
- OPD-only；
- GRPO-only；
- fallback-only；
- OPD + GRPO；
- OPD + GRPO + fallback。

这一 block 是论文最重要的机制表，优先于 controller state/action ablation。

### B. OPD Routing

- all-wrong SFT fallback；
- unconditional OPD；
- verifier/reward gate；
- token teachability / position-weighted selective OPD；
- gold-hidden-teacher multimodal recoverability（reference-verifier）。

### B2. OPD Token Reliability

- uniform-token OPD；
- non-answer-heading selective OPD；
- token-teachability / position-weighted selective OPD near-neighbor baseline。

报告保留 token 比例、non-answer-heading mask rate、答案/视觉/数值 token coverage 与
final accuracy。该 block 不对模型生成结构标题施加 reward penalty，只改变 teacher
distribution loss 在不同 token 位置的可信度权重。

### C. Controller State

- fixed weights；
- fixed step；
- normalized progress；
- mixed-rate × nonzero-loss；
- realized global GRPO route。

### D. Controller Actions

- OPD weight only；
- OPD cap only；
- OPD weight + cap。

`OPD + hard trajectory` 单独作为 supervision-type 负对照，不混入 controller action
消融，避免把已观察到的模板污染误写成主方法中的课程动作。

每个 block 内只能改变对应因素。若多个因素同时变化，该行只能作为 recipe，不作为因果消融。

## 8. Figure 5：Failure Taxonomy

从 eval outputs 统计并交叉：

- correct/incorrect；
- answer_flag/full_cot/other；
- clipped/EOS；
- parse failure；
- empty or malformed `Answer:`；
- repeated section template；
- privileged tag leakage。

上述 full/partial/empty/malformed 行为直接读取 final eval 的 `Template behavior counts`，
并与 correct/incorrect 交叉；不得用 teacher-probe candidate 条件统计替代 held-out 全体输出。

full-CoT 不是默认错误类。图中应回答“哪一种输出行为与错误相关”，而不是“推理越短越好”。

## 9. Qualitative Cases

至少选择：

1. student wrong、teacher 从 gold-hidden evidence 正确恢复、训练后 student 修正；
2. teacher 也不可恢复、fallback 避免错误蒸馏；
3. oracle hint 能恢复但 gold-hidden evidence 不足的上界差异；
4. full reasoning 正确案例；
5. 模板化 reasoning + malformed answer 的失败案例。

每例披露 teacher 可见信息，不展示会造成误解的隐藏 gold prompt。

## 10. Artifact 路由

- Run truth: `docs/paper_reconstruction/experiment_ledger.md`
- Claim status: `docs/paper_reconstruction/claim_evidence_matrix.md`
- Training logs: `outputs/test-fast/logs/pcd_no_visual_<run>/.../train_*.log`
- Eval truth: `<run>/<variant>/eval_chartqa/summary.csv`
- Candidate funnel: `<run>/<variant>/training_health/teacher_candidate_funnel.csv`
- Health windows: `<run>/<variant>/training_health/training_health_summary.csv`

图表脚本必须从这些 artifact 读取，不从论文 Markdown 反向解析数字。
