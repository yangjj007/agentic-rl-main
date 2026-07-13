# Closed-Loop Recoverability Curriculum 论文重构设计

日期：2026-07-12

> **Superseded positioning note (2026-07-13):** 本文件记录已批准并执行过的 CLRC
> 设计阶段，但不再定义最终论文的核心创新。当前论文以 OPD 在 sub-1B VLM
> verifiable reasoning 中的系统引入、有效性与互补性为唯一核心主张；routing 与
> controller 降为支撑机制。本文中的 `no-gold` 统一解释为 gold-hidden teacher，且
> verifier reference access 必须另列。

## 1. 目标

将当前以 recoverability-aware OPD 为中心的中文初稿，重构为一篇围绕
**Closed-Loop Recoverability Curriculum (CLRC)** 的训练方法论文。

论文必须同时满足两个条件：

1. **方法创新性**：贡献不能退化为“给 DyME 增加一个 OPD loss/gate”，而应形成局部 completion 路由与全局课程控制相互闭环的统一算法。
2. **实验有效性**：在统一 4epoch ChartQA 预算下，主方法至少稳定优于 DyME 与现有最强内部基线；论文主结果目标为 ChartQA test accuracy `> 0.60`。

在最终实验尚未完成前，正文必须区分：

- 已由日志或 eval 直接验证的事实；
- 当前实现但尚待完整消融验证的机制；
- 论文假设或目标，不得提前写成已证实结论。

## 2. 核心研究问题

小 VLM 的 on-policy reasoning training 同时面临两个失配：

1. **局部监督失配**：错误 completion 并不均质。部分错误可由 no-gold teacher 根据视觉证据恢复，部分错误连 teacher 也无法可靠恢复；统一 SFT 回退或统一 OPD 都会浪费信号或放大噪声。
2. **全局课程失配**：固定的 OPD/SFT 权重或按训练步数衰减，不能反映学生何时真正具备自主 on-policy exploration 能力。改变 epoch、batch size 或数据量后，固定 step boundary 还会改变方法语义。

CLRC 的核心问题是：

> 能否用无 gold 的多模态可恢复性诊断选择每个 completion 的监督路径，并用学生实际获得的全局 GRPO completion coverage 闭环控制 teacher support，使小模型从 recoverable failure repair 平滑过渡到自主探索？

## 3. 方法定义

### 3.1 局部层：No-Gold Multimodal Recoverability Routing

对于 prompt `x`，学生策略采样 completion 集合：

`Y = {y_1, ..., y_K}`。

可验证奖励给出每个 completion 的 outcome correctness。对错误 completion，teacher probe 只能访问：

- 原始问题；
- 图表图像；
- DePlot/visual evidence；
- 输出格式约束；
- 学生当前 completion 或其可用状态。

teacher probe 不允许访问 gold answer。根据 outcome reward 与 teacher recoverability，将 completion 分为三个学习状态：

1. **Autonomous exploration / GRPO**：学生 completion 正确，保留组内相对 RL 更新。
2. **Recoverable failure / OPD**：学生 completion 错误，但 no-gold teacher 可恢复正确答案，对学生当前 on-policy token states 施加 distribution guidance。
3. **Unrecoverable failure / fallback**：学生与 teacher 均无法可靠恢复，或 teacher 输出未通过质量门，使用受控 SFT/trajectory fallback 或跳过不可信监督。

该层的贡献不是 teacher 是否正确这一条规则本身，而是将可恢复性作为
**on-policy learning-state estimator**，让同一 rollout batch 内的 completion 接受不同性质的监督。

### 3.2 全局层：Realized-Autonomy Feedback

局部路由完成后，在所有 data-parallel rank 上统计互斥的最终 completion routes：

`N_grpo, N_opd, N_sft, N_skip`。

定义学生的实际自主学习覆盖率：

`a_t = N_grpo / N_total`。

控制器使用 EMA：

`z_t = alpha * a_t + (1 - alpha) * z_(t-1)`。

其中当前实现 `alpha = 0.10`。为避免同一步路由结果反向改变同一步路由，step `t` 的全局快照只控制 step `t+1` 的动作。

定义单调 mastery：

`m_t = max(m_(t-1), z_t)`。

再通过平滑映射得到 teacher support：

`s_t = 1 - smoothstep(clip(m_t / tau, 0, 1))`。

当前主配置 `tau = 0.30`。`s_t` 同时控制：

- OPD loss weight：`1.5 -> 0.5`；
- teacher trajectory loss weight：`0.5 -> 0`；
- 每 prompt OPD completion cap：`8 -> 2`。

因此全局课程由学生实际进入 GRPO 的能力驱动，而不是由 epoch 或绝对 step 驱动。

### 3.3 双时间尺度闭环

局部层与全局层构成闭环：

1. 当前 teacher support 影响 completion 的最终监督路由与更新强度；
2. 更新后的学生策略改变下一批 rollout 的正确 completion 数量；
3. 全局 realized GRPO route rate 衡量学生实际自主覆盖率；
4. 控制器据此调整下一步 teacher support。

论文应把这一结构写成双时间尺度课程，而不是三个独立 trick：

- fast timescale：completion-level recoverability routing；
- slow timescale：global autonomy feedback controller。

## 4. 与最邻近工作的边界

### 4.1 DyME

DyME 的核心是根据 batch learning state 在 memorization/SFT 与 exploration/RL 之间动态切换。CLRC 的差异必须落在：

- completion-level 三路监督，而非 batch-level 二路状态；
- no-gold multimodal recoverability，而非只依赖学生 batch correctness；
- realized GRPO completion coverage 驱动连续控制，而非离散阶段选择。

### 4.2 RG-OPD

RG-OPD 已使用 verifier reward 估计 teacher reliability 并调节 on-policy distillation。因此论文不得把“reward-gated teacher logits”作为主要 novelty。

CLRC 的区分点应是：

- recoverability 来自 no-gold multimodal teacher probe 与 evidence quality gate；
- routing 同时包含 GRPO、OPD、fallback 三种学习状态；
- 全局课程由最终 realized route feedback 闭环驱动；
- teacher compute/budget 本身也是控制动作，而不只是 token loss 权重。

### 4.3 CHORD 与其他 SFT/RL hybrid

CHORD 等方法动态组合 off-policy expert data 与 on-policy RL。CLRC 不应泛称“动态权重”创新，而应强调：

- 状态变量是完成局部路由后的实际 autonomous completion coverage；
- 控制对象是 recoverability-based teacher support；
- 控制器不依赖预定义训练进度，因此在 epoch/data/batch 改变时保持语义。

## 5. 论文主张

### 5.0 当前实现与论文主方法的已知缺口

当前正在运行的 `oracle_hint` 变体记录了
`teacher/privileged_suffix_has_gold_rate = 1.0`，因此该 run 只能用于：

- 验证全局 route reduction 与闭环控制器训练动力学；
- 估计 privileged/oracle 条件下的性能上界；
- 为后续 no-gold 主实验选择控制器超参数。

它不能作为 no-gold CLRC 的主结果，也不能用于证明无答案泄漏。论文主方法必须在
`teacher/privileged_suffix_has_gold_rate = 0.0` 的配置下重新完成 4epoch 训练与完整 eval。
实验表中要把 oracle CLRC 与 no-gold CLRC 分列，不能只在脚注说明。

### 5.1 可在方法层直接主张

1. 提出一个局部 recoverability routing 与全局 autonomy feedback 组成的双时间尺度训练框架。
2. 全局控制信号来自跨 rank 归约后的最终 completion routes，避免 rank-local health signal 与真实训练状态不一致。
3. 控制器使用滞后一拍的单调连续控制，避免同一步循环依赖和随 epoch 改变失效的固定 step boundary。
4. teacher probe 的主路径不访问 gold answer，并记录 leakage diagnostics。

### 5.2 必须由实验后才能主张

1. CLRC 在 4epoch 预算下超过 DyME、oracle official 和内部最强基线。
2. CLRC 达到 ChartQA test accuracy `> 0.60`。
3. CLRC 降低 all-wrong、zero-loss、clip/degenerate rate，并提高 last50 GRPO route rate。
4. CLRC 在相同性能下减少 teacher calls/tokens，或在相同 teacher compute 下提高性能。
5. 控制器对 epoch、batch size、数据规模变化比固定 step schedule 更稳健。

若实验不支持某条，该条必须从摘要、贡献和结论中删除或降级为分析。

## 6. AAAI 级证据矩阵

### E0：主结果

统一 4epoch、相同数据与 student initialization：

- base/SFT；
- DyME official；
- oracle official；
- recoverability routing without closed loop；
- no-gold full CLRC；
- oracle/privileged CLRC upper bound。

报告 ChartQA accuracy、训练时间、teacher calls、teacher generated tokens 和 GPU hours。

### E1：局部路由消融

- all wrong -> SFT；
- unconditional OPD；
- reward/verifier gate；
- no-gold multimodal recoverability routing。

目的：证明收益来自可恢复性状态划分，而不只是增加 teacher loss。

### E2：全局控制器消融

- fixed weights/cap；
- fixed step decay；
- normalized progress decay；
- mixed-rate/zero-loss controller；
- realized global GRPO route controller。

目的：证明真实 autonomous coverage 是更合适的课程状态变量。

### E3：闭环动作拆分

- 只控制 OPD weight；
- 只控制 trajectory weight；
- 只控制 teacher budget/cap；
- 三动作联合。

### E4：控制器稳健性

改变 epoch、有效 batch size 或训练集规模，比较固定 step boundary 与 CLRC。

核心指标不是只看 final accuracy，还要比较动作发生时的 realized GRPO route rate。

### E5：no-gold 与 evidence 消融

- format only；
- DePlot evidence；
- visual facts；
- oracle hint/no-gold evidence；
- gold leakage diagnostic（只作上界，不作为主方法）。

### E6：训练动力学

统一绘制：

- global GRPO/OPD/SFT route rates；
- controller signal EMA、mastery、support；
- OPD/trajectory weights 与 cap；
- task all-wrong、task zero-loss、total-reward zero-loss 与 disagreement；
- accuracy、clip、EOS、degenerate；
- teacher candidate/correct/accepted funnel。

### E7：效率与质性分析

- teacher calls per solved sample；
- teacher generated tokens；
- wrong-but-recoverable 与 unrecoverable 示例；
- 模板化 reasoning、答案空行、privileged leakage 等失败类型。

## 7. 中文稿重构结构

1. 摘要：问题、双时间尺度 CLRC、主要结果、效率结果。
2. 引言：局部失败异质性 + 固定课程失配两个缺口。
3. 相关工作：小 VLM RLVR、SFT/RL hybrid、on-policy distillation、adaptive curricula。
4. 预备知识：DyME/GRPO 与 on-policy distillation。
5. 方法：局部 recoverability state、三路 loss、全局 route reduction、连续控制器、训练算法。
6. 实验：主结果、消融、动力学、效率、稳健性。
7. 分析与局限：teacher quality、evidence dependence、控制器单调性、任务泛化。
8. 结论。

## 8. 文档职责

- `docs/paper_reconstruction/chinese_draft.md`：完整中文论文正文。
- `docs/paper_reconstruction/references_seed.bib`：可核验的相关工作条目。
- `docs/paper_reconstruction/experiment_plot_plan.md`：图表与表格定义。
- `docs/opd_experiment_plan.md`：可执行实验矩阵、优先级和命令。
- `docs/opd_main_training_eval_results.md`：只记录已完成实验事实。
- 新建 `docs/paper_reconstruction/claim_evidence_matrix.md`：逐条 claim 对应证据、状态和缺口。
- 新建 `docs/paper_reconstruction/experiment_ledger.md`：所有 run、配置、状态、结果和论文用途的统一账本。

## 9. 写作约束

1. 不把 oracle gold suffix 训练写成 no-gold 主方法；任何 privileged context 必须在表中显式披露。
2. 不把训练 reward 当成 held-out accuracy。
3. 不用单次短跑支持最终性能结论。
4. 不把 full-CoT 输出比例本身定义为坏；只分析错误模板、截断、parse failure 与答案准确性的关系。
5. 不把尚未超过 `0.60` 的方法写成已达到目标。
6. 所有表格必须给出模型、数据、epoch、有效 batch、teacher、视觉证据、gold access 和 eval processed count。

## 10. 验收标准

论文重构阶段完成时应满足：

1. 中文稿的标题、摘要、引言、方法和贡献均统一到 CLRC。
2. 方法公式与当前代码实现一致，不保留固定 step boundary 的旧叙事。
3. Related Work 明确讨论 DyME、CHORD、RG-OPD 和最邻近 OPD/RLVR 工作。
4. claim-evidence matrix 中不存在把待验证结果标成已验证的条目。
5. experiment ledger 包含当前 `0.5800` 基线、oracle official `0.5872`、正在运行的 global-GRPO controller run，以及后续所有迭代。
6. 实验计划能直接验证局部路由贡献、全局闭环贡献、teacher compute 与稳健性。
7. 最终摘要中的效果数字只来自完整、可复现的 eval artifact。
