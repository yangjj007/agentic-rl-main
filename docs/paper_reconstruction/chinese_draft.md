# 中文论文初稿：面向稀疏可验证推理的可恢复性感知 On-Policy Distillation

工作标题：

**Recoverability-Guided On-Policy Distillation for Small Vision-Language Reasoning**

> 当前稿按“训练方法论文”重写：弱化实现变量和小消融命名，强调宏观算法、公式、训练动力学和论文论点。ChartQA/DePlot 只作为实验场景与证据来源，不作为相关工作主轴。

## 摘要

小规模视觉语言模型在专用场景和边缘部署中具有实际价值，但其推理训练仍然脆弱。监督微调提供稳定的离线目标，却容易让小模型记忆固定的推理模板；基于可验证奖励的强化学习能够鼓励探索，却经常受到稀疏奖励、低组内方差和长序列信用分配的限制。本文提出一种可恢复性感知的 on-policy distillation 训练范式。核心思想是：错误轨迹并不等价。对不可恢复的失败轨迹，离线模仿仍然是稳定的保底信号；对能够被无答案泄漏 teacher 从视觉证据中恢复的错误轨迹，则应转化为稠密的 on-policy 分布监督。基于这一观察，我们将稀疏 outcome reward、teacher recoverability 和 token-level distribution guidance 结合到统一目标中，在低奖励方差阶段自适应增强可恢复错误轨迹的监督强度。该方法旨在把原本对 GRPO 近似无贡献的失败区域转化为有效学习信号，从而提升小模型推理训练的稳定性、样本利用率和最终任务表现。

## 1. 引言

视觉语言模型正在从通用感知系统走向面向专用任务的推理系统。对于图表理解、医学视觉问答、几何推理等任务，模型不仅要识别视觉内容，还要执行数值比较、关系推断和多步计算。大模型可以通过大量指令数据、长链推理示例或大规模在线 RL 获得此类能力；但在小模型上，简单放大这些训练范式通常并不稳定。

一种直接路线是监督微调。它把人工或强模型生成的推理轨迹作为 hard target，使模型快速获得格式和任务模板。然而，小模型容量有限，长 CoT 往往包含大量与视觉证据无关的语言模式；模型可能学会“像在推理”的文本，而不是学会从图像中恢复关键事实。另一种路线是基于可验证奖励的强化学习。它不要求每一步都有人工标注，只要最终答案可验证，就能用 outcome reward 优化策略。但这类奖励通常是稀疏的：当一组采样结果全错、全对或格式不可解析时，组内相对优势信号会变弱，训练更新变成低效甚至不稳定。

本文关注这两个训练信号之间的空白区域：**错误但可恢复的 on-policy 轨迹**。这类轨迹在最终答案上失败，因此从稀疏 reward 看是负样本；但它们并非完全无用。如果一个 teacher 在不访问 gold answer 的条件下，仅依据输入和可用视觉证据即可恢复正确答案，那么该错误轨迹仍携带接近正确推理区域的状态信息。将其直接丢回离线 SFT 会浪费 on-policy 分布；将所有错误都蒸馏又会放大 teacher 噪声。我们因此引入 recoverability-guided OPD：只在 teacher 可恢复时，把错误轨迹转化为 token-level 分布监督；在不可恢复时，保留稳定的 imitation fallback。

方法上，我们把训练信号看作三类互补来源：可验证奖励负责保留探索，teacher recoverability 负责判断失败轨迹是否仍有学习价值，on-policy distillation 负责把可恢复错误转化为逐 token 的稠密指导。奖励越稀疏、组内差异越弱时，方法越倾向于提高这类稠密指导的权重；teacher 也无法恢复的样本则回到稳定的 fallback supervision。这样，我们不是把所有错误都当作 SFT 样本，也不是无条件蒸馏 teacher，而是在当前策略分布上选择性地修复可恢复失败。

本文贡献如下：

1. 提出 recoverability-guided on-policy distillation，将稀疏 outcome reward 中被视为失败的轨迹进一步区分为可恢复与不可恢复区域。
2. 提出一个统一训练目标，将组内相对 RL、teacher-gated dense distillation 和稳定 fallback 结合起来，提高小模型推理训练中的有效更新比例。
3. 给出奖励方差自适应的监督强度调节，使方法在低 reward-diversity 阶段仍能提供有用梯度。
4. 设计整体实验与诊断框架，从主结果、训练动力学、有效信号转化、泛化/预算和 teacher 代价五个层面验证方法，而不是只依赖单个消融设置。

## 2. 相关工作

### 2.1 小模型与多模态推理训练

**视觉指令调优与小模型训练。** LLaVA、InstructBLIP、MiniGPT-4、Qwen-VL、InternVL、LLaVA-NeXT 等工作奠定了视觉编码器、投影模块和语言模型组合的主流 VLM 训练范式 [1-6]。TinyLLaVA、MobileVLM、SmolVLM 进一步面向小模型结构、数据配方和部署效率优化，说明小 VLM 具备实际应用价值 [7-9]。DyME 则指出，小 VLM 推理训练需要在记忆式监督和探索式 RL 之间动态平衡 [10]。这些工作主要解决对齐、效率和训练模式切换问题，而本文关注稀疏可验证奖励下错误轨迹如何被重新利用。

**多模态 CoT 与视觉推理监督。** Multimodal-CoT、LLaVA-CoT、Insight-V、Mulberry、MatCha、TinyChart 等方法通过人工 rationale、结构化阶段、搜索轨迹、图表预训练或 PoT 数据构造提升视觉推理能力 [11-16]。这些方法说明推理轨迹对 VLM 重要，但仍以离线 hard target 为主，容易让小模型学习固定解释模板。近期工作也指出，过强或过长的 CoT imitation 可能诱导 pseudo reasoning，并压缩后续 RL 的探索空间 [17]。

**多模态 RLVR。** Visual-RFT、VLM-R1、Reason-RFT、Visual Aha、LMM-R1、MM-Eureka、Vision-R1、OpenVLThinker 等工作将 rule-based reward 和 GRPO/PPO 类优化引入视觉问答、定位、图表理解和多模态数学推理 [18-25]。与这些方法相比，本文不主要设计新的任务 reward 或更长的冷启动 CoT，而是关注 sparse reward 中被浪费的失败区域：当学生生成错误但 teacher 能在无答案证据下恢复时，该轨迹应成为 dense on-policy supervision。

### 2.2 稀疏可验证奖励与在线强化学习

**在线偏好优化与可验证奖励。** PPO 是 RLHF 中最常用的在线策略优化基础，InstructGPT、Constitutional AI、HHH/RLHF 和 RLAIF 展示了 SFT、奖励模型与在线优化结合的典型路线 [26, 27, 63, 64, 69]。DPO、IPO、KTO、ORPO 则用离线偏好数据降低在线 RL 复杂度 [28-31]。这些方法适合对齐和偏好学习；在数学、代码、图表问答等可自动验证任务中，rule-based outcome reward 更便宜，但也更稀疏。

**GRPO 与 R1-style RL。** GRPO 用同一 prompt 的多条采样构造组内相对 advantage，避免额外 value model。DeepSeekMath 和 DeepSeek-R1 将 GRPO/rule-based RL 推向数学与长链推理训练，Kimi k1.5、Open-Reasoner-Zero、SimpleRL-Zoo 和 Minimalist Reasoning 进一步从不同规模和实现角度分析这一范式 [35-42]。这类方法强化了最终答案可验证任务中的探索能力，但当 group 全错、全对或方差过低时，训练信号会明显变弱。

**稀疏信号稳定化。** DAPO、Dr.GRPO、VinePPO、ReMax、RLOO、REINFORCE++、PRIME 以及过程监督方法指出，长推理中的 outcome reward 会带来长度偏置、无效 group、截断、credit assignment 和 step-level supervision 缺失等问题 [38, 39, 43-49, 33, 34]。本文采取互补路线：不直接丢弃所有低方差失败区域，而是用 teacher recoverability 判断其中哪些错误轨迹仍可转化为稠密 OPD 信号。

**SFT-RL 混合训练。** LUFFY、CHORD、SRFT 和 rejection sampling fine-tuning 等方法试图在离线示范和在线探索之间取得平衡 [48-50, 32, 70]。它们通常从全局 schedule、动态权重或筛选数据角度管理 SFT 与 RL 的比例。本文的区别是：训练信号由局部 recoverability 决定，而不是只由训练阶段、prompt 级过滤或全局权重决定。

### 2.3 On-Policy Distillation 与 teacher-guided training

**知识蒸馏与生成式蒸馏。** KD、sequence-level KD、Born-Again Networks、DistilBERT、TinyBERT、MiniLM、MiniLLM 等工作表明，teacher 的 soft distribution 能提供比 hard label 更丰富的学习信号，并可用于压缩、自蒸馏和生成式训练 [51-57]。传统 KD 常在 reference 或 teacher 序列上训练，容易与测试时学生自己的前缀分布不匹配；GKD 等 on-policy distillation 方法则让学生先生成轨迹，再让 teacher 在 student-generated contexts 上提供 token-level 反馈 [58]。

**自训练与 teacher-guided reasoning。** STaR、Self-Improve、ReST、Self-Rewarding、RFT、RAFT 和 best-of-N rejection sampling 通过采样、验证、过滤和再训练获得更高质量轨迹 [59-62, 32, 66, 70]。RLHF、RLAIF、Constitutional AI、RRHF、DPO、process reward models、Distilling Step-by-Step 和 step verification 则用人类或 AI teacher 提供 preference、critique、ranking、rationale 或过程信号 [27, 69, 63, 65, 28, 68, 67, 33]。这些 teacher 信号通常在离线数据构造阶段被固定，或对所有选中样本统一使用，缺少对当前 student failure state 的在线判别。

**Recoverability-guided OPD。** 本文将 teacher-guided supervision 放入可验证奖励训练循环中。teacher 不是无条件标签生成器，而是 recoverability probe：只有当 teacher 在无答案泄漏证据下能恢复正确答案时，它的分布才被用于指导学生当前错误轨迹。因此，OPD 不再只是 dataset-level compression，而是与 on-policy sampling、verifiable reward 和 reward variance 联动的训练信号选择机制。

## 3. 方法

### 3.1 Training Signal as Recoverability

我们考虑可验证视觉推理任务。输入 \(x\) 包含问题和视觉上下文，学生策略 \(\pi_\theta\) 采样轨迹 \(y\)，验证函数 \(R(x,y)\in[0,1]\) 只在完整输出后给出 outcome reward。传统 RL 只从 \(R\) 中学习，而 SFT 只从离线目标 \(y^\star\) 中学习。本文引入第三个训练信号：recoverability。

Recoverability 衡量的是：在不访问 gold answer 的条件下，一个外部 teacher 是否能从输入证据和当前错误状态中恢复正确答案。记 teacher 分布为 \(q_T\)，可用证据为 \(e(x)\)。对于学生采样轨迹 \(y_i\)，recoverability 定义为：

\[
c_i = \mathbf{1}\{\mathrm{Verify}(\hat{y}^{T}_i)=1,\ 
\hat{y}^{T}_i\sim q_T(\cdot|x,e(x),y_i)\}.
\]

它不是额外的 gold label，而是一个在线可验证判别信号。若 \(r_i=0\) 但 \(c_i=1\)，说明该轨迹虽然最终失败，但处在 teacher 可恢复区域；若 \(c_i=0\)，说明 teacher 也无法可靠修复，继续蒸馏可能放大噪声。

### 3.2 Objective

训练目标由 sparse RL、recoverability-guided distillation 和 fallback imitation 三部分构成：

\[
\mathcal{L}(\theta)
= \mathbb{E}_{x,Y\sim\pi_\theta}
\left[
\mathcal{L}_{\mathrm{grp}}(\theta)
+ \lambda(\sigma_R)\mathcal{L}_{\mathrm{rec}}(\theta)
+ \mu\mathcal{L}_{\mathrm{fb}}(\theta)
\right].
\]

第一项 \(\mathcal{L}_{\mathrm{grp}}\) 是 group-relative policy optimization，用同组奖励构造 advantage：

\[
A_i = \frac{r_i-\mathrm{mean}(R)}{\mathrm{std}(R)+\epsilon}.
\]

第二项只作用于错误但可恢复的轨迹：

\[
\mathcal{L}_{\mathrm{rec}}
= \sum_i (1-r_i)c_i
D_{\mathrm{tok}}(q_T,\pi_\theta;y_i),
\]

第三项为不可恢复或训练初期不稳定区域提供保底监督：

\[
\mathcal{L}_{\mathrm{fb}}
= \sum_i (1-r_i)(1-c_i)\cdot
\ell_{\mathrm{imit}}(\theta;x,y^\star).
\]

这三个项的分工是：RL 保留对正确轨迹的探索压力；distillation 将可恢复错误轨迹转化为稠密 token-level 信号；fallback 避免 teacher 失败区域被错误蒸馏。

### 3.3 Adaptive Dense Guidance

稀疏奖励下的主要问题之一是 group reward 方差不足。当 \(\sigma_R\) 很低时，\(\mathcal{L}_{\mathrm{grp}}\) 的有效区分能力下降。我们因此令 distillation 权重依赖 reward diversity：

\[
\lambda(\sigma_R)
=\lambda_0
\left(1+\alpha\cdot
\mathrm{clip}\left(1-\frac{\sigma_R}{\tau},0,1\right)\right).
\]

该设计有两个性质。第一，当 group 已有足够 reward diversity 时，训练主要依赖 RL；第二，当 group 进入低方差区域时，可恢复错误轨迹获得更强的 dense guidance。它不修改 verifier reward，也不改变 teacher 的正确性判定，而是调节不同训练信号在低信号阶段的组合比例。

### 3.4 Discussion

该目标可以看作在 sparse outcome RL 和 hard imitation 之间插入一个 teacher-verifiable 中间信号。与直接 SFT 相比，它不强制学生复制离线 CoT；与普通 RL 相比，它不把所有错误都视为同样无用；与普通蒸馏相比，它只在 teacher 可验证恢复时使用 teacher distribution。宏观上，该方法提升的是 useful-update coverage：更多 on-policy 轨迹在每个训练阶段都能获得非零且可信的学习信号。

## 4. 实验设计

实验应从整体训练方法角度组织，而不是从环境变量或小消融命名出发。

### 4.1 主结果

主表比较 base、SFT、GRPO/RLVR、SFT-RL 混合、generic OPD 与本文方法，在相同模型、数据和训练预算下报告 held-out accuracy、format-valid rate、输出长度和训练开销。若条件允许，应覆盖多个小模型规模和多个可验证视觉推理任务。

### 4.2 训练动力学

训练曲线应展示方法如何改变优化过程：

- answer reward / held-out accuracy over training。
- reward std / nonzero advantage rate。
- recoverable-wrong coverage。
- useful update rate。
- format-valid rate、degeneration rate 和 output length。

核心结论不是某个 checkpoint 偶然更好，而是方法持续提高有效训练信号密度。

### 4.3 机制分析

机制实验回答三个问题：

1. 去掉 recoverability gate 是否导致错误蒸馏或训练不稳定？
2. 去掉 adaptive dense guidance 是否降低低方差阶段的样本利用率？
3. 去掉对低信号失败区域的 teacher recovery 是否让训练退回 sparse RL 或 imitation dominated 状态？

四个 no-VS 4epoch 运行可放在这一节作为局部机制消融，但不应成为整篇论文的实验目标。

### 4.4 成本、可靠性与边界

需要报告 teacher call rate、teacher token cost、wall-clock overhead、no-gold evidence 检查、gold leakage rate、teacher parse success 和 teacher correctness。若 teacher 证据不足或任务需要视觉属性而中间证据丢失，应在失败案例中明确展示。

## 5. 预期论文图表与叙事顺序

参考 `best_paper.pdf` 的写法，图表不应只是补充材料，而应承担论文推进功能。我们的主线建议如下：

第一，Figure 1 应是强 motivation/teaser 图，而不是路由统计图。它同时展示一个直观样例和一组训练现象：SFT 可以给格式但易模板化，RL 可以探索但在低 reward-diversity 区域失效，而本文方法把一部分错误轨迹恢复成有效监督。这张图的作用等价于 VAR 的 Figure 1：让读者在第一页看到“方法改变了什么”。

第二，Figure 2 应是范式对比图。左侧画 hard imitation：所有失败都被离线目标覆盖；中间画 sparse RL：只有最终 reward 提供信号；右侧画 recoverability-guided OPD：错误轨迹先经过 teacher-verifiable recoverability，再转化为 dense on-policy guidance 或 fallback。这张图对应 VAR 的 AR vs VAR 对比图，强调本文是训练范式重组，而不是局部 loss trick。

第三，Table 1 应前置为主结果表，比较 base、SFT、GRPO/RLVR、SFT-RL hybrid、generic OPD 和 ours。随后 Figure 3 再解释训练动力学，Table 2 再验证机制消融。这样的顺序比先讲四个小消融更符合方法论文写法：先证明有效，再证明为什么有效。

| 图表 | 目的 |
| --- | --- |
| Figure 1 | 强 motivation/teaser：展示小模型 sparse RL 中的失败异质性、可恢复错误轨迹和方法效果 |
| Figure 2 | 范式对比：hard imitation vs sparse RL vs recoverability-guided OPD |
| Table 1 | 主结果：base/SFT/RL/SFT-RL/OPD/Ours |
| Figure 3 | 训练动力学：reward、reward std、useful update、输出稳定性 |
| Table 2 | 机制消融：recoverability、adaptive guidance、teacher evidence |
| Figure 4 | useful-signal conversion funnel |
| Figure 5 | 模型规模、训练预算或 teacher compute tradeoff |
| Table 3 | anti-leakage 与 teacher cost |
| Figure 6 | 成功与失败案例 |

## 6. 局限性

方法依赖 teacher 在无答案泄漏证据下的可靠性；如果 teacher 自身无法恢复，distillation 不应被触发。第二，teacher 调用会带来额外计算开销，需要通过 coverage-cost ratio 证明收益合理。第三，若视觉证据缺失关键属性，teacher recoverability 会低估可学习样本。第四，当前主要面向可验证推理任务；对于开放式生成或主观评价任务，需要将 verifier 替换为更可靠的评价器。

## 7. 结论草稿

本文提出 recoverability-guided on-policy distillation，用 teacher-verifiable recoverability 弥合 hard imitation 与 sparse RL 之间的训练信号空白。该方法不把所有错误轨迹统一视为失败，也不无条件蒸馏 teacher，而是只在错误轨迹可被无答案 teacher 恢复时提供稠密分布监督。通过与 reward diversity 联动的自适应权重，方法在低信号阶段提高 useful-update coverage，为小视觉语言模型的可验证推理训练提供了一种更稳定、更高效的训练范式。

## 参考文献候选表

下表用于后续整理 BibTeX。正文相关工作使用数字引用；`cite_key` 仅作为内部整理键名，便于后续转成正式 BibTeX。当前保持 2.1/2.2/2.3 三类，数量超过 40 篇，均为训练方法或训练机制相关工作；ChartQA/DePlot 不作为核心相关工作列入。对应的 seed BibTeX 已生成到 `docs/paper_reconstruction/references_seed.bib`，后续投稿前需要按目标会议格式补齐 authors/year/venue。

| # | cite_key | 论文 | 类别 | 链接 |
| ---: | --- | --- | --- | --- |
| 1 | Liu2023LLaVA | Visual Instruction Tuning | VLM instruction tuning | https://arxiv.org/abs/2304.08485 |
| 2 | Dai2023InstructBLIP | InstructBLIP | VLM instruction tuning | https://arxiv.org/abs/2305.06500 |
| 3 | Zhu2023MiniGPT4 | MiniGPT-4 | VLM alignment training | https://arxiv.org/abs/2304.10592 |
| 4 | Bai2023QwenVL | Qwen-VL | VLM pretraining/instruction tuning | https://arxiv.org/abs/2308.12966 |
| 5 | Chen2023InternVL | InternVL | VLM training | https://arxiv.org/abs/2312.14238 |
| 6 | Liu2024LLaVANeXT | LLaVA-NeXT / LLaVA-1.6 | VLM instruction tuning | https://llava-vl.github.io/blog/2024-01-30-llava-next/ |
| 7 | Zhou2024TinyLLaVA | TinyLLaVA | small VLM recipe | https://arxiv.org/abs/2402.14289 |
| 8 | Chu2024MobileVLMV2 | MobileVLM V2 | mobile VLM training | https://arxiv.org/abs/2402.03766 |
| 9 | Marafioti2025SmolVLM | SmolVLM | efficient small VLM | https://arxiv.org/abs/2504.05299 |
| 10 | Liu2025DyME | DyME | small VLM SFT/RL training | https://arxiv.org/abs/2506.23061 |
| 11 | Zhang2023MultimodalCoT | Multimodal Chain-of-Thought Reasoning in Language Models | multimodal CoT SFT | https://arxiv.org/abs/2302.00923 |
| 12 | Xu2024LLaVACoT | LLaVA-CoT | structured visual reasoning SFT | https://arxiv.org/abs/2411.10440 |
| 13 | Zhang2024InsightV | Insight-V | long-chain visual reasoning | https://arxiv.org/abs/2411.14432 |
| 14 | Yao2024Mulberry | Mulberry | MCTS-generated multimodal reasoning | https://arxiv.org/abs/2412.18319 |
| 15 | Liu2022MatCha | MatCha | chart/math reasoning pretraining | https://arxiv.org/abs/2212.09662 |
| 16 | Zhang2024TinyChart | TinyChart | small chart VLM training | https://arxiv.org/abs/2404.16635 |
| 17 | Yang2025SFTorRL | SFT or RL? | SFT vs RL for VLM reasoning | https://arxiv.org/abs/2504.11468 |
| 18 | Liu2025VisualRFT | Visual Reinforcement Fine-Tuning | VLM RLVR | https://arxiv.org/abs/2503.01785 |
| 19 | Shen2025VLMR1 | VLM-R1 | R1-style VLM training | https://arxiv.org/abs/2504.07615 |
| 20 | Zhang2025ReasonRFT | Reason-RFT | SFT+GRPO visual reasoning | https://arxiv.org/abs/2503.20752 |
| 21 | Zhao2025VisualAha | R1-Zero's Aha Moment in Visual Reasoning | small VLM RL | https://arxiv.org/abs/2503.05132 |
| 22 | Peng2025LMMR1 | LMM-R1 | multimodal rule-based RL | https://arxiv.org/abs/2503.07536 |
| 23 | Meng2025MMEureka | MM-Eureka | multimodal RLVR | https://arxiv.org/abs/2503.07365 |
| 24 | Chen2025VisionR1 | Vision-R1 | cold-start visual RL | https://arxiv.org/abs/2503.06749 |
| 25 | Wang2025OpenVLThinker | OpenVLThinker | iterative SFT/RL multimodal reasoning | https://arxiv.org/abs/2503.17352 |
| 26 | Schulman2017PPO | Proximal Policy Optimization Algorithms | PPO | https://arxiv.org/abs/1707.06347 |
| 27 | Ouyang2022InstructGPT | Training Language Models to Follow Instructions with Human Feedback | RLHF | https://arxiv.org/abs/2203.02155 |
| 28 | Rafailov2023DPO | Direct Preference Optimization | DPO | https://arxiv.org/abs/2305.18290 |
| 29 | Azar2023IPO | A General Theoretical Paradigm to Understand Learning from Human Preferences | IPO/preference theory | https://arxiv.org/abs/2310.12036 |
| 30 | Ethayarajh2024KTO | KTO | direct alignment | https://arxiv.org/abs/2402.01306 |
| 31 | Hong2024ORPO | ORPO | single-stage preference optimization | https://arxiv.org/abs/2403.07691 |
| 32 | Yuan2023RFTScaling | Scaling Relationship on Learning Mathematical Reasoning with LLMs | rejection sampling fine-tuning | https://arxiv.org/abs/2308.01825 |
| 33 | Lightman2023VerifyStep | Let’s Verify Step by Step | process supervision | https://arxiv.org/abs/2305.20050 |
| 34 | Wang2023MathShepherd | Math-Shepherd | process reward model | https://arxiv.org/abs/2312.08935 |
| 35 | Shao2024DeepSeekMath | DeepSeekMath | GRPO | https://arxiv.org/abs/2402.03300 |
| 36 | Guo2025DeepSeekR1 | DeepSeek-R1 | RLVR reasoning | https://arxiv.org/abs/2501.12948 |
| 37 | Team2025Kimik15 | Kimi k1.5 | long-CoT RL | https://arxiv.org/abs/2501.12599 |
| 38 | Yu2025DAPO | DAPO | GRPO stabilization | https://arxiv.org/abs/2503.14476 |
| 39 | Liu2025DrGRPO | Understanding R1-Zero-Like Training / Dr.GRPO | GRPO bias analysis | https://arxiv.org/abs/2503.20783 |
| 40 | Hu2025OpenReasonerZero | Open-Reasoner-Zero | open RLVR | https://arxiv.org/abs/2503.24290 |
| 41 | Zeng2025SimpleRLZoo | SimpleRL-Zoo | zero-RL analysis | https://arxiv.org/abs/2503.18892 |
| 42 | Gandhi2025Minimalist | A Minimalist Approach to LLM Reasoning | RL/rejection analysis | https://arxiv.org/abs/2504.11343 |
| 43 | Cui2025PRIME | PRIME | implicit process reward | https://arxiv.org/abs/2502.01456 |
| 44 | Kazemnejad2024VinePPO | VinePPO | PPO for long reasoning | https://arxiv.org/abs/2410.01679 |
| 45 | Li2023ReMax | ReMax | critic-free RLHF | https://arxiv.org/abs/2310.10505 |
| 46 | Ahmadian2024RLOO | Back to Basics: Revisiting REINFORCE Style Optimization | RLOO/REINFORCE | https://arxiv.org/abs/2402.14740 |
| 47 | Hu2025REINFORCEPP | REINFORCE++ | critic-free policy optimization | https://arxiv.org/abs/2501.03262 |
| 48 | Yan2025LUFFY | LUFFY | off-policy guidance with RL | https://arxiv.org/abs/2504.14945 |
| 49 | Zhang2025CHORD | CHORD | dynamic SFT-RL weighting | https://arxiv.org/abs/2508.11408 |
| 50 | Chen2025SRFT | SRFT | single-stage SFT+RL | https://arxiv.org/abs/2506.19767 |
| 51 | Hinton2015KD | Distilling the Knowledge in a Neural Network | knowledge distillation | https://arxiv.org/abs/1503.02531 |
| 52 | Kim2016SeqKD | Sequence-Level Knowledge Distillation | seq2seq distillation | https://arxiv.org/abs/1606.07947 |
| 53 | Furlanello2018BornAgain | Born Again Neural Networks | self-distillation | https://proceedings.mlr.press/v80/furlanello18a.html |
| 54 | Sanh2019DistilBERT | DistilBERT | language model distillation | https://arxiv.org/abs/1910.01108 |
| 55 | Jiao2019TinyBERT | TinyBERT | transformer distillation | https://arxiv.org/abs/1909.10351 |
| 56 | Wang2020MiniLM | MiniLM | deep self-attention distillation | https://arxiv.org/abs/2002.10957 |
| 57 | Gu2023MiniLLM | MiniLLM | generative KD | https://arxiv.org/abs/2306.08543 |
| 58 | Agarwal2023GKD | Generalized Knowledge Distillation for Auto-regressive Language Models | GKD / on-policy distillation | https://arxiv.org/abs/2306.13649 |
| 59 | Zelikman2022STaR | STaR | self-training reasoning | https://arxiv.org/abs/2203.14465 |
| 60 | Huang2022SelfImprove | Large Language Models Can Self-Improve | self-distillation | https://arxiv.org/abs/2210.11610 |
| 61 | Gulcehre2023ReST | ReST | self-training with feedback | https://arxiv.org/abs/2308.08998 |
| 62 | Yuan2024SelfRewarding | Self-Rewarding Language Models | self-reward/teacher feedback | https://arxiv.org/abs/2401.10020 |
| 63 | Bai2022ConstitutionalAI | Constitutional AI | RLAIF / critique training | https://arxiv.org/abs/2212.08073 |
| 64 | Bai2022HHH | Training a Helpful and Harmless Assistant with RLHF | RLHF | https://arxiv.org/abs/2204.05862 |
| 65 | Yuan2023RRHF | RRHF | ranking feedback | https://arxiv.org/abs/2304.05302 |
| 66 | Dong2023RAFT | RAFT | reward-ranked filtering | https://arxiv.org/abs/2304.06767 |
| 67 | Hsieh2023DistillStep | Distilling Step-by-Step! | rationale distillation | https://arxiv.org/abs/2305.02301 |
| 68 | Uesato2022ProcessOutcome | Solving Math Word Problems with Process- and Outcome-based Feedback | process/outcome feedback | https://arxiv.org/abs/2211.14275 |
| 69 | Lee2023RLAIFvsRLHF | RLAIF vs. RLHF | AI feedback | https://arxiv.org/abs/2309.00267 |
| 70 | Cobbe2021Verifiers | Training Verifiers to Solve Math Word Problems | verifier / rejection sampling | https://arxiv.org/abs/2110.14168 |
