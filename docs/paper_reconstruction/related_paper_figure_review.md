# 训练方法论文行文、实验与图表设计复盘

更新时间：2026-07-01

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

