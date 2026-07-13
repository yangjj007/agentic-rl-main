# 训练方法论文行文、实验与图表设计复盘

更新时间：2026-07-12

本文件按用户反馈重写：移除 DePlot 与 ChartQA 作为“优秀论文样例”的位置。它们是任务/数据/工具背景，不适合作为训练方法论文的写作模板。本文只保留训练方法类论文，用来指导我们的论文结构、相关工作、方法表达和实验图表。

已结合：

- 本地 `papers/DyME_2506.23061.pdf`：逐页渲染检查。
- 本地新增 `papers/best_paper.pdf`：VAR，作为优秀论文写法模板，逐页渲染检查。
- 多 agent 联网调研：VLM/RLVR 训练方法、GRPO/SFT-RL 混合、OPD/KD/teacher-guided training。

## 0. 我们论文应学习的总写法

`best_paper.pdf` 对应 VAR 论文。它不是同领域任务，但很适合作为方法论文写法模板：

1. 用一张强 teaser 图展示最终效果，而不是先堆实现细节。
2. 用一张范式对比图说明“我们重新定义了训练问题”。
3. 方法部分小标题少，公式集中，先讲核心建模，再讲训练目标。
4. 实验不是小消融列表，而是主结果、扩展规律、效率、泛化、消融、可视化逐层推进。
5. 图表承担论文论点：主表证明有效，训练/scale 曲线证明规律，消融表证明机制，qualitative 图证明行为变化。

迁移到我们的论文：

- 不把 `deplot_no_vs_opd_va_pcd` 这类实现名当论文主线。
- 不反复围绕 DyME 讲“二路/三路路由”，而是提出一个宏观训练范式：recoverability-guided dense supervision for sparse verifiable RL。
- 方法图应画 sparse RL、recoverability probe、dense on-policy guidance 三个训练信号如何组合，而不是画 shell/env 变量或具体 route if-else。
- 实验计划应证明整体算法：训练稳定性、有效更新密度、样本利用率、泛化和 teacher cost。

## 1. VAR: Visual Autoregressive Modeling

来源：本地 `papers/best_paper.pdf`

检查方式：用 PyMuPDF 渲染 19 页 PDF 到 `/tmp/best_paper_contact.png`，逐页查看图表布局；用文本抽取定位 Figure 1-11、Table 1-3、Algorithm 1-2。

### 行文结构

1. 摘要直接提出范式重定义：把图像 AR 从 raster-scan next-token prediction 改成 coarse-to-fine next-scale prediction。
2. Introduction 先讲 AR 与 diffusion 的大背景，再指出视觉 AR 的核心瓶颈，最后给出 VAR 的直观定义和数量级提升。
3. Related Work 简短服务主线，只覆盖 autoregressive image generation 与 scaling laws。
4. Method 分成核心建模、multi-scale VQ tokenizer、VAR transformer training，公式围绕 next-scale factorization 展开。
5. Experiments 先主结果表，再 scaling law，再 efficiency，再 zero-shot generalization，最后消融。

### 图表逐项作用

- Figure 1：强 teaser。大量生成样例和 zero-shot editing，立即展示方法能力。
- Figure 2：AR vs VAR 范式对比图。它是整篇论文最重要的方法定位图，清楚说明“不是改小模块，而是改预测范式”。
- Figure 3：不同模型族的 scaling behavior 对比，支持 VAR 与 LLM 类似可扩展。
- Figure 4：两阶段训练图：multi-scale VQ autoencoder 与 VAR transformer。图中只画核心训练流程，不画工程细节。
- Figure 5-6：scaling law 曲线，用模型规模和 compute 证明规律。
- Figure 7：模型规模和训练 compute 增大带来的视觉质量变化，定性支持 scaling。
- Figure 8：zero-shot downstream tasks，证明方法不是只在主 benchmark 上拟合。
- Figure 9：attention dependency heatmap，解释 VAR 学到了 coarse-to-fine 依赖。
- Figure 10-11：更多样例。
- Table 1：ImageNet 256 主结果大表，和 GAN/diffusion/AR 全面对比。
- Table 2：512 分辨率结果，证明高分辨率有效。
- Table 3：ablation，验证 tokenizer、multi-scale design、loss 等关键组件。
- Algorithm 1-2：multi-scale VQVAE 编解码，放在方法细节处而非主线中喧宾夺主。

### 对我们论文的借鉴

- Figure 2 式范式图：我们应画“sparse RL only / hard imitation only / recoverability-guided OPD”的范式对比。
- Figure 5-6 式规律图：我们应画训练预算、模型规模或 teacher compute 下的性能曲线。
- Table 1 式主表：主表应比较 SFT、GRPO、SFT-RL hybrid、generic OPD、ours，而不是只列四个环境变量消融。
- Table 3 式消融：消融只验证关键机制：recoverability gate、adaptive dense guidance、fallback、teacher evidence。

## 2. DyME: Dynamic Memorization and Exploration

来源：本地 `papers/DyME_2506.23061.pdf`；https://arxiv.org/abs/2506.23061

检查方式：本地 PDF 24 页渲染到 `/tmp/dyme_contact.png` 并逐页检查。已检查主文 Figure 1-5、Table 1-4，附录 Figure S1-S5、Table S1-S9、Prompt S1-S3。

### 行文结构

DyME 的写法可借鉴但不应成为我们全文的叙述中心。它从 small VLM 的训练失败出发，先展示 SFT 伪推理和 RLVR advantage collapse，再提出 dynamic memorization/exploration 的 training paradigm。实验分两部分：Pure DyME 验证动态切换，Full DyME 验证 visual supervision 跨任务有效。

### 图表逐项作用

- Figure 1：问题动机图，用红/绿输出对照说明 SFT/RL 对 LVLM 有效、对 SVLM 失败。
- Figure 2：小型 ChartQA teaser result，证明常规 SFT/RL/two-stage 可能降性能。
- Figure 3：方法主图，展示 switcher、GRPO/SFT 两分支和 visual checker/refiner。
- Figure 4：训练 reward 曲线，证明动态切换改善训练稳定性。
- Figure 5：定性样例，红色错误/绿色 grounded reasoning。
- Table 1：算法验证，拆数据质量与切换策略。
- Table 2：跨任务主结果。
- Table 3：memorization、exploration、visual refiner/checker 消融。
- Table 4：成本收益。
- 附录图表：refiner 前后 GT、off-policy influence、模型案例、训练设置、人工 CoT 评估等。

### 对我们论文的借鉴与边界

可借鉴的是“先证明 small VLM 训练失败，再提出训练范式”的结构；不应照搬的是二路 switch 叙事。我们的中心应是：

> sparse RL 中错误轨迹存在可恢复子集；teacher-verifiable recoverability 可以把这部分轨迹转化为 dense on-policy supervision。

因此正文可以把 DyME 放在相关工作和 baseline 中，不反复说“在 DyME 上扩展”。

## 3. DAPO: Open-Source LLM Reinforcement Learning System at Scale

来源：https://arxiv.org/abs/2503.14476

agent 已检查 Figure 1-7、Table 1-3、Algorithm 1。

### 行文结构

DAPO 的主线很适合作为训练方法论文模板：先指出 naive GRPO 复现 R1 类效果不足，再把问题分解成 entropy collapse、无效 group、sample-level loss 偏置、overlong reward noise，最后逐项提出 Clip-Higher、Dynamic Sampling、Token-level Loss、Overlong Reward Shaping。

### 图表作用

- Figure 1：主训练曲线，avg@32/pass@32/cons@32 随 step 提升。
- Figure 2：Clip-Higher 对 entropy 和 AIME 的影响。
- Figure 3：upper clipping 与全对 prompt 占比，说明无效 group 问题。
- Figure 4：token-level loss 对 entropy 和 response length 的影响。
- Figure 5：overlong filtering 对 reward noise 的影响。
- Figure 6：dynamic sampling 前后训练效率。
- Figure 7：response length、reward、entropy、mean probability dashboard。
- Table 1：逐项累加消融，是训练方法论文的强范式。
- Algorithm 1：dynamic sampling buffer 与过滤条件。

### 对我们论文的借鉴

- 我们的 Figure 3 可借鉴 DAPO Fig.7：用少量核心指标展示训练动力学。
- 我们的机制图可借鉴 DAPO Fig.6：强调低信号/无效更新区域如何被处理。
- Table 2 消融应像 DAPO Table 1 一样逐项说明机制贡献，而不是用实现名堆表。

## 4. SFT or RL? Training R1-like Reasoning Large Vision-Language Models

来源：https://arxiv.org/abs/2504.11468

调研结论：该论文比 ChartQA/DePlot 更适合作为样例，因为它直接研究训练方法选择。

### 行文结构

论文围绕 VLM reasoning post-training 中 SFT 与 RL 的取舍展开。它不是把任务数据作为主角，而是比较 SFT、RL、SFT+RL 对 reasoning behavior 的影响，重点分析 pseudo reasoning、native reasoning、GRPO 退化等训练现象。

### 图表亮点

- Figure 1：直接对比 SFT pseudo reasoning 与 RL native reasoning，是非常强的 motivation 图。
- Figure 2：数据生成/训练 pipeline，说明如何构造 SFT/RL 对照。
- Figure 3 与 Table 2/3：展示 SFT 单独或 SFT+GRPO 可能退化，支持“不是所有失败都应回退 SFT”的观点。

### 对我们论文的借鉴

这篇最适合支撑引言中的关键句：

> hard rationale imitation can stabilize format but may narrow exploration and induce pseudo reasoning; sparse RL can discover reasoning behavior but suffers from low-density reward.

我们的 Figure 1 可以仿照它：一侧展示 SFT-like 模板错误，另一侧展示 recoverability-guided training 后的修正轨迹。

## 5. Visual-RFT / VLM-R1 / Reason-RFT

来源：

- Visual-RFT：https://arxiv.org/abs/2503.01785
- VLM-R1：https://arxiv.org/abs/2504.07615
- Reason-RFT：https://arxiv.org/abs/2503.20752

这些论文应作为 VLM RLVR 训练方法组来学习，而不是单独照抄。

### 共同结构

1. 先定义多模态推理任务中的 verifiable reward。
2. 比较 SFT 与 RL/RFT/GRPO。
3. 报告多个视觉任务或多个 benchmark。
4. 分析 reward design、训练阶段和泛化。

### 可借鉴图表

- Visual-RFT：方法图对比 visual instruction tuning 与 visual reinforcement fine-tuning；多任务表格展示 RLVR 相对 SFT 的优势。
- VLM-R1：R1-style VLM pipeline、SFT/RL 随训练 step 对比、reward hacking 与 aha moment 分析。
- Reason-RFT：两阶段 SFT+GRPO overview；ID/OOD、2B/7B、ANS-SFT/CoT-SFT/RFT-Zero 的主表；reasoning token redundancy 分析。

### 对我们论文的定位

这些工作说明 VLM RLVR 已经成立，但大多关注 reward design、数据构造或整体 RL recipe。我们的差异是训练信号选择机制：当 sparse reward 把轨迹判为错误时，不立即丢弃或回退 imitation，而是用 teacher recoverability 判断它是否能提供 dense on-policy supervision。

## 6. GKD / OPD and Teacher-Guided Training

来源代表：

- Knowledge Distillation：https://arxiv.org/abs/1503.02531
- Sequence-Level KD：https://arxiv.org/abs/1606.07947
- STaR：https://arxiv.org/abs/2203.14465
- Constitutional AI：https://arxiv.org/abs/2212.08073
- Distilling Step-by-Step：https://arxiv.org/abs/2305.02301
- GKD / On-Policy Distillation：https://arxiv.org/abs/2306.13649
- RFT：https://arxiv.org/abs/2308.01825
- ReST：https://arxiv.org/abs/2308.08998

### 方法谱系

- Generic KD：固定数据上匹配 teacher soft labels，主要解决压缩。
- Sequence-level KD：teacher 生成完整伪序列，student 做 MLE，但仍是 teacher/off-policy distribution。
- Self-distillation：模型自举产生更好轨迹或软标签。
- Teacher feedback/RLAIF：teacher 提供 critique、preference 或 reward。
- Rejection sampling/RFT：采样多候选，只保留 verifier 认为正确的轨迹。
- OPD/GKD：student 当前策略生成轨迹，teacher 在 student contexts 上给 dense feedback。

### 对我们论文的定位

我们的最接近谱系是 OPD/GKD，但关键差异是 teacher signal 被 recoverability gate 控制。不是每条 student trajectory 都蒸馏；只有 `wrong but teacher-verifiable` 的轨迹被转化为 dense guidance。这样可以避免 generic OPD 对不可恢复错误进行过度 teacher forcing，也避免 RFT 只保留最终正确样本而浪费可恢复错误状态。

## 7. 推荐最终样例论文组合

替换 DePlot/ChartQA 后，建议 related review 中保留以下 5 篇作为“优秀论文样例”：

| 样例论文 | 为什么适合 |
| --- | --- |
| VAR | 写法模板：范式重定义、scaling/主表/消融组织优秀 |
| DyME | 直接 small VLM SFT/RL 动机和 baseline |
| DAPO | GRPO 稀疏奖励稳定化与训练 dashboard |
| SFT or RL? | VLM reasoning 中 SFT vs RL 的训练行为对比 |
| Visual-RFT / VLM-R1 / Reason-RFT 组 | VLM RLVR 训练方法主线 |
| GKD/OPD | on-policy distillation 理论来源 |

这组覆盖：优秀写法模板、直接基线、RL 稳定化、VLM RLVR、OPD teacher-guided training。DePlot/ChartQA 后续仅在实验场景、数据描述和 failure case 中出现。

## 8. 对中文初稿的审稿式建议

1. 相关工作必须至少 40+ 引用，且分成三类：小/多模态推理训练；稀疏可验证奖励与在线 RL；OPD 和 teacher-guided training。
2. 方法部分不应有过多 implementation headings。建议保留 3-4 个小节：Problem, Objective, Adaptive Guidance, Discussion。
3. 不要反复写 DyME。DyME 是 baseline，不是本文叙述主角。
4. 不要把 DePlot/ChartQA 写成核心相关工作。它们是 task/evidence context。
5. 实验计划应先证明 overall effectiveness，再分析 mechanism，最后讲 cost/reliability。

## 9. CLRC 最邻近方法边界

中文稿现以 Closed-Loop Recoverability Curriculum 为核心。相关工作不能再只写
“OPD 有 gate”，而应直接比较状态粒度、teacher trust signal 与全局控制闭环。

| Method | Granularity | Teacher trust signal | Multimodal evidence | Learning routes | Global feedback | Teacher budget control |
|---|---|---|---|---|---|---|
| DyME | batch/optimization step | student batch correctness state | optional visual supervision | SFT or RLVR | discrete current-step state selection | no |
| CHORD | global + expert token | training schedule and student probability on expert token | not specific to multimodal evidence | off-policy SFT + on-policy RL | scheduled global coefficient plus token weights | no |
| GKD | student-generated sequence/token | teacher distribution generally assumed available | task-dependent | on-policy KD, optionally mixed with RL | no realized-route controller | no |
| RG-OPD | trajectory/sample | verifier reward estimates teacher reliability | not inherently multimodal | reward-gated OPD | reward-weighted teacher trust | no explicit completion-route feedback budget |
| IW-OPD | token position | accumulated teacher-student discrepancy | not inherently multimodal | position-weighted OPD | no route-level autonomy feedback | no |
| DOPD | token | privileged teacher/student advantage gap and relative probability | includes VLM experiments but relies on privileged branches | dual token-level distillation | dynamic local routing, no realized GRPO coverage controller | no |
| Vision-OPD | multimodal token / sequence | teacher distribution on visual-reasoning trajectories | explicitly aligns visual perception and reasoning | OPD for multimodal reasoning | no completion-route autonomy controller | no |
| REOPOLD | visual-reasoning sequence / teacher choice | empirical teacher capability under OPD | 3B/7B visual-reasoning students | OPD with multiple teacher choices | no realized GRPO route feedback | no |
| ViCuR | visual cue / privileged signal | whether teacher privilege is recoverable from the visual input | 2B/8B VLM experiments | filter privileged guidance then distill | no realized GRPO route feedback | no |
| TA-OPD | token | teacher corrective mass on the student's current top-K support | not inherently multimodal | retain high-teachability OPD positions | no completion-route or curriculum feedback | no |
| PW-OPSD | token position / branch | privileged branch viability and sequence position | not inherently multimodal | position-weighted OPSD | no route-level autonomy feedback | no |
| SFD / Lookahead Group Reward | token branch | next-step teacher confidence under student prefixes | not inherently multimodal | OPD plus lookahead group reward | no teacher-support curriculum | entropy-triggered compute only |
| GateKD | token / representation | teacher confidence | not inherently multimodal | gated soft, hidden-state, and attention distillation | confidence-modulated closed-loop framing, no realized route occupancy | no |
| CLRC | completion + global training state | gold-hidden teacher answer verified by RLVR reference plus quality gates | image + DePlot/visual facts; optional oracle hint only in upper bound | GRPO / OPD / fallback | realized global GRPO completion coverage | no teacher-compute control in current implementation; post-probe OPD exposure only |

该表限定论文 novelty：

1. 不把 dynamic weighting、position weighting、reward gating、privileged routing 或 selective OPD 单独称为首次提出。
2. ViCuR 已使用 recoverable visual privilege，VOLD 已把 cold-start alignment、GRPO 与 OPD 用于 VLM reasoning，REOPOLD 已在 3B/7B visual-reasoning student 上系统比较 OPD teacher；因此不能把 recoverability、visual-reasoning OPD 或 OPD+RLVR 本身称为首次。候选主贡献必须完整限定为 sub-1B all-wrong completion 的 verifier-confirmed 三路状态路由，以及由 realized global GRPO route occupancy 驱动的 OPD-exposure 闭环。
3. oracle hint 配置只作为 upper bound；no-gold 主方法必须以 leakage metric `0.0` 验证。
4. 若实验只证明 controller 稳定而未证明 final accuracy/compute 改善，摘要应将其写成机制贡献而非效果结论。

### Recoverability novelty 风险

“不是所有 teacher signal 都可学习”已经不能作为 CLRC 的独立新颖性主张：TA-OPD 明确定义 token teachability，PW-OPSD 分析 privileged branch viability，SFD 工作分析长前缀下 teacher corrective signal 的衰减。因此正文必须避免使用“首次识别可学习 teacher signal”之类表述。

CLRC 尚可验证的差异是一个跨粒度组合命题：

1. 局部判据作用于完整 multimodal completion 的 outcome recoverability，而不是单个 token 的 support overlap、position 或 teacher entropy；
2. 判据决定 GRPO、OPD、fallback 三种不同优化目标，而不是只对 OPD token 重加权或筛选；
3. 最终互斥 route counts 形成全局 realized-autonomy feedback，并进一步控制下一步 teacher weight 与 per-prompt budget；
4. gold-hidden-teacher 设置不把 reference 放入 teacher prompt，但当前 routing verifier 仍使用 RLVR reference；这必须与 oracle answer hint 分栏披露。

这四点必须通过 local-routing、token-selective OPD、controller-state 和分层 gold-access 消融共同证明。GateKD 也说明“closed-loop distillation”这一名称本身不是贡献；CLRC 必须明确其反馈量是 realized route occupancy，而不是 teacher confidence。若 full CLRC 不能优于 TA-OPD/PW-OPSD 风格的简单 selective OPD baseline，则 recoverability curriculum 的 AAAI 级方法 claim 不成立。

## 10. AAAI 图表叙事

建议主文只保留能直接证明核心论点的图表：

- Figure 1：左侧 completion-level recoverability routing，右侧 global autonomy feedback loop。
- Figure 2：统一 4epoch 下 accuracy 与 teacher generated tokens 的 Pareto 图。
- Figure 3：GRPO/OPD/SFT routes、controller signal/support 和 task zero-loss 的同步曲线。
- Figure 4：改变 epoch/batch/data scale 后，fixed-step 与 CLRC 动作发生时学生自主覆盖率的比较。
- Table 1：带 `Gold access`、teacher evidence、processed count 的主结果。
- Table 2：local routing、controller state、controller actions 三组正交消融。

图表不把 full-CoT 比例本身画成负向性能。应将输出类型与准确率、截断、parse
failure 和模板错误交叉分析，避免把“会推理”误判为污染。

## 11. OPD-first 论文定位（2026-07-13）

用户确认论文核心应聚焦 OPD 的引入、有效性与互补性。经近邻审计后，允许与禁止的定位如下。

禁止：

- “首个将 OPD 引入 VLM reasoning”；VOLD、REOPOLD、Decomposed OPD 与 VA-OPD 已覆盖。
- “首次发现 teacher signal 并非都可学习”；TA-OPD、PW-OPSD、SFD 已覆盖。
- “首次 closed-loop distillation”；GateKD 已使用该表述。

候选主张：

> To our knowledge, this is the first systematic study of on-policy distillation for sub-billion-parameter VLM reasoning under verifiable rewards.

该主张的论文价值不应依赖狭窄的“first”本身，而应由三类证据支撑：

1. **Effectiveness**：统一 4epoch 下，加入 OPD 相对 DyME/no-OPD 显著提升 held-out accuracy；
2. **Complementarity**：OPD 回收 student-generated wrong states，GRPO 强化已发现解，fallback 稳定完全低信号状态；正交消融证明三者不能被任一单一路线替代；
3. **Small-model analysis**：说明 sub-1B 学生为何比 2B/3B/7B VLM 更依赖 dense on-policy feedback，并报告 zero-loss、all-wrong、route occupancy 与 teacher compute。

controller 降为使 OPD loss exposure 随学生自主能力退出的配套机制。当前 post-probe cap
不减少 teacher generation；除非另行实现 pre-probe cap 并完成 compute Pareto，不能声称
controller 节省 teacher compute，也不把 controller 单独写成第一贡献。

## 12. 2026 OPD 近邻补充审计

截至 `2026-07-13`，以下四篇工作进一步压缩了可用 novelty 空间：

| Work | 已覆盖的核心问题 | 对 CLRC 的直接约束 | 必需实验回应 |
|---|---|---|---|
| TrOPD (`2606.01249`) | teacher--student 分布差异过大时，只在可靠 trust region 内进行 OPD，并处理 outlier region | 不能把“过滤不可靠 teacher supervision”作为 completion routing 的宽泛首创 | P0-E6 至少包含 reliability-matched token/region baseline；报告保留 token 比例与 final accuracy |
| TIP (`2604.14084`) | 高 student entropy 与低 entropy/高 teacher-student divergence token 的重要性 | 不能假设 uniform-token OPD 是充分近邻；style token 问题已有更一般的 token-importance 表述 | P0-E6 固定 completion routes，只替换 token weighting，避免把 route 与 token reliability 混为一谈 |
| AOPD (`2605.06387`) | 在非正 advantage 区域使用局部 divergence minimization，在正 advantage 区域保留 RL | “OPD 补 zero-advantage”本身已有直接 token-level 近邻 | P0-E5 必须用 OPD-only/GRPO-only/OPD+GRPO 证明 sub-1B VLM 中的信号互补性，而不能只展示三路实现 |
| SSOPD (`2605.17497`) | mixed group 中用最短正确 completion 条件 teacher 分布纠正最长错误 completion | correct/wrong completion 状态与 dense process supervision 已被直接使用 | 新增 P0-E9；按 mixed/all-wrong 分解结果，证明外部 recoverability 的收益来自 SSOPD 无法覆盖的 all-wrong group |

这四篇近邻使论文核心问题进一步清晰：本文不是一般地提出 selective OPD，也不是一般地
把 OPD 与 RLVR 联合，而是检验 **0.5B VLM 的 all-wrong student states 是否能被外部
privileged teacher 可靠回收，并且这种 completion-level 信号是否在统一预算下提供超出
DyME fallback、mixed-group self-distillation 与 token-selective OPD 的净收益**。如果
P0-E1/P0-E3、P0-E6 和 P0-E9 不能支持这一点，论文应降级为负结果/系统分析，而不能继续
维持 AAAI 主会方法贡献表述。

## 13. 2026-07-14 Late Threat Update

新增两个直接影响论文定位的主源：

1. REOPOLD (`2603.11137`) 已明确研究 visual reasoning OPD，并在 3B/7B student 上
   比较多种 teacher。它进一步否定“首次把 OPD 引入视觉推理”的宽泛表述，但没有覆盖
   0.5B student、all-wrong RLVR state 或 matched OPD/no-OPD signal attribution。
2. Kaur et al. (`2607.05184`) 指出 privileged teacher 与 thinking student 在关键
   reasoning fork token 上可能选择不同策略，naive privileged OPD 因而可能降低能力。
   该结果支持 CLRC 的 teacher correctness/recoverability gate、hard-trajectory isolation
   和 token/style forensic，但也意味着这些设计必须通过 unconditional OPD 与 no-OPD
   matched 消融证明，而不能仅靠直觉宣称可靠。

因此当前最强可守定位仍是 sub-1B VLM verifiable reasoning 中 OPD 的系统净收益与
OPD/GRPO/fallback 互补性研究。若统一 4epoch 实验不能同时证明这两点，论文应转为
“small-model OPD failure modes and design lessons”，而不是依赖狭窄 first claim。
