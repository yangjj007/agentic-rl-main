# On-Policy Distillation for Sub-Billion Vision-Language Reasoning

中文工作标题：**面向十亿参数以下视觉语言推理的在策略蒸馏**

> 论文状态：方法与实现章节可按当前代码写作；效果数字以
> `claim_evidence_matrix.md` 和 `experiment_ledger.md` 为准。当前 oracle CLRC
> 仍在训练，gold-hidden-teacher CLRC 主实验尚未完成，因此摘要不预填 CLRC 最终准确率。
> 术语勘误：当前非 oracle 设置是 teacher prompt 不含 gold、但 routing verifier
> 使用 reference answer；正式稿应称为 gold-hidden-teacher / verifier-available，
> 不能称为“完全无 gold recoverability estimator”。

## 摘要

小型视觉语言模型（small VLM）具有部署成本低、任务适配快等优势，但让不足十亿参数的模型学会可靠推理仍然困难。监督微调（SFT）能够提供稳定目标，却要求学生模仿 teacher 生成的离线推理轨迹，容易形成与视觉证据脱节的固定模板；基于可验证奖励的强化学习（RLVR）直接优化学生自己的输出，但当一组采样全部答错时，GRPO 几乎得不到有效的相对优势。两种训练信号分别稳定却偏离学生状态、贴近学生状态却过于稀疏。

据我们所知，本文首次系统研究 **on-policy distillation（OPD）在 sub-1B VLM 可验证推理中的作用**。OPD 不要求学生复现 teacher 的完整轨迹，而是在学生自己生成的 token states 上提供稠密分布指导，因而能够直接补充小模型最缺少的训练信号：学生当前答错、GRPO 无法形成有效优势，但 teacher 仍能给出可靠纠正的 on-policy states。为可靠地引入 OPD，我们提出 Closed-Loop Recoverability Curriculum（CLRC）：学生已经答对的 completion 用 GRPO 强化；学生答错但 gold-hidden teacher 给出 verifier-confirmed 正确答案的 completion 使用 OPD；其余低信号状态进入受控 fallback。系统再根据实际进入 GRPO 的全局 completion 比例连续调整 teacher support，使 OPD 在学生自主探索稀少时提供帮助，并随自主能力形成而逐步退出。

本文的核心经验问题不是控制器本身是否复杂，而是 **OPD 是否有效，以及它是否提供了 SFT 与 GRPO 均不能替代的互补信号**。我们在 ChartQA 上采用统一 4epoch 协议，对比 no-OPD、unconditional OPD、OPD-only、GRPO-only、fallback-only 与完整联合训练，并分别报告 teacher 是否看到 gold、verifier 是否使用 reference、teacher compute、训练动力学和完整 held-out accuracy。当前实验仍在进行；最终稿仅在有效 `summary.csv` 完成后填入 CLRC 的最终结果。

## 1. 引言

视觉语言模型正在从识别图像内容走向基于视觉证据进行推理。图表理解、几何推理和视觉问答不仅要求模型识别物体或文字，还要求它定位相关证据、执行比较或计算，并给出可验证答案。大规模 VLM 可以通过长链监督数据和强化学习获得这些能力，但实际部署常需要参数量不足十亿的小型 VLM。此类模型推理成本低、适合专用领域，却也更容易受容量、指令遵循和稀疏奖励限制 [Zhou2024TinyLLaVA; Chu2024MobileVLMV2; Marafioti2025SmolVLM; Liu2025DyME]。

现有方法主要依赖 SFT 和 RLVR。多模态 CoT、LLaVA-CoT、Insight-V 和 Mulberry 等工作通过高质量推理轨迹教模型逐步作答 [Zhang2023MultimodalCoT; Xu2024LLaVACoT; Zhang2024InsightV; Yao2024Mulberry]。这种监督对学习输出格式和基本解题步骤很有效，但离线 teacher 轨迹并不一定落在小模型能够实现的分布内。对于容量有限的学生，模仿冗长 rationale 可能牺牲视觉 grounding，并产生看似完整但答案错误的 pseudo reasoning。SFT-or-RL 的系统比较也观察到，SFT 形成的刚性推理模式可能妨碍后续强化学习 [Yang2025SFTorRL]。另一方面，Visual-RFT、VLM-R1、Reason-RFT、LMM-R1 和 R1-VL 表明，可验证奖励能够直接激励多模态推理 [Liu2025VisualRFT; Shen2025VLMR1; Zhang2025ReasonRFT; Peng2025LMMR1; Zhang2025R1VL]；但小模型很难稳定地产生正确且可解析的候选。当同组 completion 全错或奖励相同时，GRPO 的相对优势接近零，训练会频繁浪费更新。

DyME 对这一困境给出了直接答案：当学生尚未找到正确解时使用 SFT 记忆，当学生已经产生正确解时使用 RL 探索 [Liu2025DyME]。这一动态切换显著优于静态两阶段训练，但仍留下一个关键空白。SFT 在 teacher 生成的离线序列上优化，GRPO 则只从少量有奖励差异的学生序列中学习；大量“学生当前答错、但 teacher 能够在该学生状态上提供有用纠正”的 on-policy states 没有得到合适的训练信号。把它们全部送回 SFT 会离开学生真实访问的状态，把它们留给 GRPO 又几乎没有梯度。

OPD 为这个空白提供了自然工具。与序列级蒸馏不同，OPD 让学生先生成输出，再让 teacher 在这些学生生成的前缀上给出下一 token 分布 [Agarwal2023GKD]。因此，它不直接强制学生复现一条 teacher sequence，也不要求学生先偶然采样到正确答案，便能在自己的状态分布上获得稠密指导。需要强调的是，避免 exact-sequence imitation 并不自动消除 teacher 风格迁移；teacher distribution 仍可能优先强化结构标题等低价值 token，因此局部 token reliability 必须被测量和消融。这一性质对 sub-1B VLM 尤其重要：模型容量越小，teacher 离线轨迹与学生可实现轨迹之间的差距越大，而纯 RLVR 找到首个正确解的概率又越低。已有 VOLD、Decomposed OPD 和 Visual-Advantage OPD 已证明 OPD 可用于视觉语言推理 [Bousselham2025VOLD; Yoon2026DecomposedOPD; Liu2026VAOPD]，但它们主要关注更大 VLM 的推理迁移、视觉 grounding 或 token weighting。**据我们所知，本文是首个面向 sub-1B VLM 可验证推理、系统研究 OPD 有效性及其与 RLVR 和监督学习互补性的工作。** 我们不声称首次提出 VLM OPD；我们的新意在于把 OPD 引入此前主要由 SFT/RLVR 切换主导的小模型训练区间，并用统一预算和正交消融回答它何时有效、为何有效、能否被现有信号替代。

直接对所有错误 completion 使用 OPD 也并不可靠。teacher 可能误读视觉证据，长 student prefix 上的纠正能力可能衰减，不同 token 的 teachability 也并不相同 [Fu2026RevisitingOPD; Xie2026IWOPD; Wang2026TeachabilityOPD; Liu2026PWOPSD; Liu2026SFD]。此外，学生对 teacher 的需求会随训练变化：早期需要更多稠密指导，后期若继续保持高强度蒸馏，则可能压制学生自己的探索。于是，真正的问题不是简单地“加一个 OPD loss”，而是确定 **何时使用 OPD、何时保留 GRPO、何时采用 fallback，以及 teacher support 应在何时退出**。

为此，我们提出 CLRC。对每个学生 completion，系统首先使用可验证任务 reward 判断学生答案。正确 completion 进入 GRPO；错误 completion 由不接收 reference answer 的多模态 teacher 重新作答，其答案再由 RLVR verifier 检查，只有 teacher 被验证正确时才进入 OPD；其余 completion 使用受控 fallback。这里的 gold-hidden 只描述 teacher 输入，routing verifier 仍可访问与 RLVR reward 相同的 reference。局部三路路由使三种训练信号承担不同职责：GRPO 强化学生已经发现的解，OPD 在学生错误状态上提供稠密纠正，fallback 处理 teacher 也无法可靠纠正的低信号状态。全局上，CLRC 统计最终实际进入各路线的 completion，而不是把预设 loss weight 当作训练状态；随后依据 realized GRPO coverage 调整下一步 teacher support。

这一设计的目的不是提出一个脱离 OPD 的通用控制器，而是让 OPD 在小模型训练中成为可靠、可测量的独立学习信号。本文不把 full-CoT 或结构化 reasoning 格式视为污染；我们关心的是这些输出是否建立在正确视觉证据上，以及训练信号是否真正改善任务答案。我们也不把 teacher prompt 不含 gold 等同于整个算法 reference-free：所有结果将分别披露 `Teacher sees gold` 与 `Verifier uses reference`。

本文的贡献如下：

1. 据我们所知，我们首次系统研究 OPD 在 sub-1B VLM 可验证推理中的有效性，指出它补充了纯 SFT 的 student-state mismatch 与纯 RLVR 的稀疏、零优势信号。
2. 为使 OPD 在多模态错误状态上可靠工作，我们提出 verifier-confirmed completion routing：GRPO 学习学生已经发现的解，OPD 纠正 teacher 可恢复的学生状态，fallback 处理剩余低信号状态；基于 realized GRPO coverage 的连续控制仅作为调节 OPD 介入强度的配套机制。
3. 我们通过 no-OPD、unconditional OPD、OPD-only、GRPO-only、fallback-only 和完整三路训练的统一预算对照，检验 OPD 的净收益、适用边界及其与现有训练信号的互补性，并报告 accuracy、zero-loss、route occupancy 与 teacher compute。

## 2. 相关工作

### 2.1 小型视觉语言模型与多模态推理监督

视觉指令微调奠定了通用 VLM 的训练范式，代表工作包括 LLaVA、InstructBLIP、MiniGPT-4、Qwen-VL 和 InternVL [Liu2023LLaVA; Dai2023InstructBLIP; Zhu2023MiniGPT4; Bai2023QwenVL; Chen2023InternVL]。TinyLLaVA、MobileVLM V2 和 SmolVLM 进一步研究了更小规模、更低部署成本的视觉语言模型 [Zhou2024TinyLLaVA; Chu2024MobileVLMV2; Marafioti2025SmolVLM]。这些模型具备基本视觉问答能力，但较小容量使长推理监督、视觉 grounding 和严格格式遵循之间的冲突更加突出。

多模态 CoT 通过生成中间 rationale 提升视觉推理 [Zhang2023MultimodalCoT]。LLaVA-CoT 将视觉推理组织为结构化阶段 [Xu2024LLaVACoT]，Insight-V 探索长链视觉推理 [Zhang2024InsightV]，Mulberry 使用搜索构造反思式多模态轨迹 [Yao2024Mulberry]。这类方法说明高质量 rationale 可以提供有效监督，但 rationale 的长度和语言完整性并不保证视觉事实正确。SFT-or-RL 进一步表明，模仿强模型轨迹可能形成 pseudo reasoning 并限制后续 RL [Yang2025SFTorRL]。本文不是反对 CoT，而是研究如何在学生实际生成状态上补充 teacher 信号，降低对完整离线轨迹模仿的依赖。

### 2.2 多模态可验证强化学习

PPO、GRPO 和 RLVR 为无需逐步人工标注的推理训练提供了基础 [Schulman2017PPO; Shao2024DeepSeekMath; Guo2025DeepSeekR1]。DAPO、Dr.GRPO、Open-Reasoner-Zero 和 SimpleRL-Zoo 分析并改进了长链 RL 中的 clipping、采样、长度偏差和训练稳定性 [Yu2025DAPO; Liu2025DrGRPO; Hu2025OpenReasonerZero; Zeng2025SimpleRLZoo]。这些工作主要面向语言推理，但其稀疏奖励和低有效更新率问题在小型 VLM 中更严重。

在多模态领域，Visual-RFT 将可验证奖励用于视觉感知任务 [Liu2025VisualRFT]；VLM-R1、Visual-Aha、LMM-R1、MM-Eureka 和 Vision-R1 探索 R1-style 多模态强化学习 [Shen2025VLMR1; Zhao2025VisualAha; Peng2025LMMR1; Meng2025MMEureka; Chen2025VisionR1]；Reason-RFT、R1-VL 和 OpenVLThinker 分别研究视觉推理 RFT、step-wise GRPO 和迭代 SFT-RL [Zhang2025ReasonRFT; Zhang2025R1VL; Wang2025OpenVLThinker]。这些工作证明 RLVR 能够发现超越直接模仿的推理行为，但通常依赖模型先产生一定比例的正确候选。本文关注正确探索极少的 sub-1B 区域，并用 OPD 为错误 on-policy states 提供稠密信号。

### 2.3 SFT、专家指导与 RL 的动态组合

静态 SFT 后接 RL 容易出现阶段间分布不匹配。DyME 根据当前 batch 是否产生正确答案，在 memorization 和 exploration 之间动态切换，是本文最直接的 small-VLM baseline [Liu2025DyME]。LUFFY 使用 off-policy guidance 支持推理学习 [Yan2025LUFFY]，CHORD 从全局系数和 expert-token 权重两层动态调和 SFT 与 on-policy RL [Zhang2025CHORD]，SRFT 在单阶段内联合 supervised 与 reinforcement fine-tuning [Chen2025SRFT]，KDRL 则统一知识蒸馏和强化学习 [Xu2025KDRL]。

CLRC 与这些工作的共同点是承认 imitation 和 exploration 互补。差异在于 OPD 的作用位置：它不在离线 expert sequence 上提供另一个 SFT loss，而是在 student-generated prefixes 上构成独立学习状态。我们的实验必须通过 no-OPD、OPD-only、GRPO-only、fallback-only 和完整三路训练证明这一差异，而不能仅以“动态加权”作为创新。

### 2.4 知识蒸馏与 On-Policy Distillation

经典知识蒸馏匹配 teacher soft targets [Hinton2015KD]，sequence-level KD 使用 teacher 生成序列训练学生 [Kim2016SeqKD]，DistilBERT、TinyBERT、MiniLM 和 MiniLLM 将蒸馏扩展到预训练语言模型和生成模型 [Sanh2019DistilBERT; Jiao2019TinyBERT; Wang2020MiniLM; Gu2023MiniLLM]。这些方法通常在固定数据或 teacher-generated sequence 上训练，存在 exposure bias 和 student-state mismatch。

GKD 将 teacher feedback 移到 student-generated sequences 上，系统化提出 on-policy distillation [Agarwal2023GKD]。近期研究快速扩展了这一方向：VOLD 将 LLM reasoning 迁移到 VLM [Bousselham2025VOLD]；Decomposed OPD 和 VA-OPD 分别处理视觉 grounding 与视觉优势加权 [Yoon2026DecomposedOPD; Liu2026VAOPD]；IW-OPD、control-variate OPD 和 best-of-N teacher selection研究位置偏差、方差和 teacher rollout 选择 [Xie2026IWOPD; Oh2026vOPD; Zhang2026BRTS]；RG-OPD、DOPD 和 SCOPE 研究 verifier gating、privileged supervision 与自适应权重 [Akhondzadeh2026RGOPD; Yu2026DOPD; Zheng2026SCOPE]；TA-OPD、PW-OPSD 和 SFD 则说明 teacher disagreement、token position 和长 student prefix 上的纠正信号并非同等可靠 [Wang2026TeachabilityOPD; Liu2026PWOPSD; Liu2026SFD]。Prefix distribution matching 与 disagreement-adaptive rewarding 进一步表明，OPD 的收益取决于学生访问状态和 teacher-student disagreement，而不能把所有位置视为均匀有效 [Zheng2026PrefixOnPolicy; Lee2026DEAR]。GateKD 也使用 confidence-gated closed-loop distillation 的表述 [Sermsri2026GateKD]。

因此，本文不声称首次提出 VLM OPD、teacher reliability gating 或 closed-loop distillation。我们的核心定位是：**据我们所知，首次在 sub-1B VLM 可验证推理中系统引入和评估 OPD，并以严格消融证明 OPD 相对 no-OPD 的净收益以及它与 GRPO、fallback supervision 的互补性。** verifier routing 与闭环 teacher support 是为这一核心问题服务的方法设计，而不是与 OPD 并列的独立论文主线。

### 2.5 课程学习、图表推理与实验设置

自动课程学习根据学生进度选择训练任务或环境，说明训练日程应随能力变化，而非只依赖固定 step [Matiisen2017TSCL; Portelas2019ALPGMM]。这一思想也被扩展到可靠 LLM reasoning 和 self-evolving reasoning curriculum [Zhao2024AutoCEI; Chen2025SelfEvolvingCurriculum]。CLRC 的 controller 与课程学习共享能力驱动的思想，但它不选择下一个数据样本，而是观察已经发生的 GRPO/OPD/fallback route occupancy，并调节下一步 teacher support。因此 controller 是 OPD 主线的配套机制，其独立价值必须由 fixed-weight、fixed-progress 和 proxy-state 消融验证。

ChartQA 同时要求视觉感知与逻辑推理，是评估小型 VLM 的合适场景 [Masry2022ChartQA]。MatCha 和 TinyChart 研究图表理解与小型 chart model [Liu2022MatCha; Zhang2024TinyChart]，DePlot 将图表转换为结构化表格，为 teacher 提供额外视觉证据 [Liu2022DePlot]。本文的 gold-hidden setting 不把 reference 放入 teacher prompt，但 routing verifier 与标准 RLVR 一样可使用 reference；oracle setting 额外给 teacher answer hint，只作为上界。两者在所有结果表中分栏。

## 3. 预备知识

### 3.1 Group Relative Policy Optimization

对 prompt `x`，学生策略 `pi_theta` 采样 `K` 个 completion：

```text
Y = {y_1, ..., y_K},  y_i ~ pi_theta(. | x).
```

可验证 reward `r_i` 对最终答案和输出约束评分。组内标准化优势可写为：

```text
A_i = (r_i - mean(r_1, ..., r_K)) / (std(r_1, ..., r_K) + epsilon).
```

当所有 `r_i` 近似相同，`A_i` 接近零。本文把“task accuracy 全零”与“总 reward 优势全零”分开记录，因为格式或思考奖励可能制造总 reward 方差，却不能证明任务学习信号存在。

### 3.2 On-Policy Distribution Guidance

给定 teacher 生成的固定 trajectory `y^T`，sequence-level SFT/KD 优化：

```text
L_traj = - sum_t log pi_theta(y^T_t | x, y^T_<t).
```

训练状态 `(x, y^T_<t)` 来自 teacher 分布。对于容量有限的学生，teacher prefix 可能并非
学生在推理时会访问或能够稳定延续的状态。相比之下，OPD 先从当前学生采样
`y^S ~ pi_theta(.|x)`，再在 student-generated prefix 上匹配 teacher distribution：

```text
L_OPD = E_{y^S ~ pi_theta(.|x)} [
    sum_t D(pi_T(.|x,y^S_<t) || pi_theta(.|x,y^S_<t))
].
```

其中 `D` 可取 JSD、forward KL 或其他分布距离。两种方法都使用 teacher，但监督发生的
状态分布不同：hard trajectory 要求学生复现 teacher sequence，OPD 则纠正学生实际访问的
状态。因此本文的核心比较不是“有无 teacher”，而是 **teacher supervision 是否作用于
student states**。teacher 在某个 student state 是否可信仍需 verifier 或质量门单独估计。

## 4. Reliable On-Policy Distillation for Sub-Billion VLMs

本文方法的核心是把 OPD 引入 sub-1B VLM 的可验证推理训练。Closed-Loop
Recoverability Curriculum（CLRC）是实现这一目标的具体训练框架：verifier-confirmed
routing 决定哪些学生错误状态适合接受 OPD，realized-autonomy controller 决定 OPD
应以多大强度介入。二者都服务于 OPD 的可靠使用，而不是独立于 OPD 的平行创新。

### 4.1 局部可恢复性学习状态

对每个 completion `y_i`，首先用可验证答案 reward 判断学生是否正确。正确 completion 进入 GRPO。对于错误 completion，teacher probe 接收问题、图像和允许的视觉证据，并生成可解析答案。gold-hidden-teacher 设置禁止把 reference answer 放入 probe prompt，但随后使用 RLVR verifier 和 reference 判断 teacher answer 是否正确。

定义 teacher recoverability indicator：

```text
q_i = 1[teacher answer is verifiably correct and passes evidence/quality gates].
```

局部 route 为：

```text
GRPO,      if student completion is correct;
OPD,       if student is wrong and q_i = 1;
fallback,  if student is wrong and q_i = 0.
```

fallback 可以是受控 trajectory/SFT repair，也可以在 teacher 信号不可信时跳过。论文主实验必须固定 fallback 定义，避免把多个机制同时变化。

### 4.2 三路联合目标

令 `M_G`、`M_O`、`M_F` 分别为最终 GRPO、OPD 和 fallback masks。主方法目标写为：

```text
L_t = lambda_G L_GRPO(M_G)
    + lambda_O(t) L_OPD(M_O)
    + lambda_F(t) L_fallback(M_F).
```

`L_fallback` 可以是受控 SFT，也可以为零损失 skip，但在 matched 消融中必须保持定义不变。
teacher hard-trajectory loss `L_traj` 不属于核心 OPD 目标，仅作为序列模仿对照；已有
oracle 实验表明把它与 OPD 直接混合会造成明显 train-eval gap。`lambda_O(t)` 由全局
闭环控制器根据 realized autonomy 调节。实现中最终 route 互斥，避免一个 completion
同时接受相互冲突的 GRPO、OPD 与 fallback 更新。

### 4.3 全局 realized-autonomy feedback

在局部 route 完成后，各 rank 统计：

```text
N_grpo, N_opd, N_sft, N_skip, N_total.
```

通过一次跨 rank sum-reduction 得到全局快照。定义实际自主覆盖率：

```text
a_t = N_grpo / N_total.
```

该值直接表示当前 batch 中最终进入 GRPO 的 completion 比例，而不是 mixed-group proxy 或包含格式奖励的 total-reward variance。

### 4.4 连续控制器

控制器首先平滑自主覆盖率：

```text
z_t = alpha * a_t + (1 - alpha) * z_(t-1).
```

当前主配置使用 `alpha = 0.10`。定义单调 mastery：

```text
m_t = max(m_(t-1), z_t).
```

再映射为 teacher support：

```text
u_t = clip(m_t / tau, 0, 1),
s_t = 1 - smoothstep(u_t),
smoothstep(u) = u^2 (3 - 2u).
```

当前 target `tau = 0.30`。控制器实现可让多个 teacher-support 动作共享同一
snapshot，例如：

```text
lambda_O: 1.5 -> 0.5,
OPD cap per prompt: 8 -> 2.
```

其中 `lambda_T` 表示可选 hard teacher-trajectory loss。它在早期联合实验中曾按
`0.5 -> 0` 衰减，但该实验的 held-out accuracy 仅为 `0.5120`，并出现严重固定模板
污染。因此当前 OPD isolation recipe 将 `lambda_T` 从训练开始固定为 `0`，同时关闭
teacher-SFT repair；闭环只调节 OPD weight 与 OPD completion budget。hard trajectory
仅保留为负对照和消融，不是论文主方法的组成部分。

使用单调 mastery 的目的是避免偶然坏 batch 让 teacher support 反复振荡。其局限是学生发生长期能力回退时不能自动恢复 support，本文将在局限和非单调控制器消融中讨论。

### 4.5 因果顺序与分布式一致性

step `t` 的生成和路由使用 step `t-1` 后保存的控制状态。step `t` 最终 route 完成后，系统才计算 `a_t` 并更新控制器，供 step `t+1` 使用。该一拍延迟避免同一步 route 既是动作结果又是动作输入。

所有 controller rank 读取同一个跨 rank snapshot。早期诊断显示 rank-local mixed/zero-loss 与 global task state 可显著不同，因此论文只使用全局最终 route 作为控制信号，local health metrics 仅作诊断。

### 4.6 算法流程

```text
Input: student policy, teacher, verifiable reward, visual evidence
Initialize EMA z_0=0, mastery m_0=0, full teacher support
For each training step t:
  1. Sample K student completions per prompt.
  2. Compute verifiable rewards and GRPO advantages.
  3. Probe wrong completions with the allowed teacher evidence.
  4. Apply quality/leakage gates and assign mutually exclusive routes.
  5. Optimize GRPO + OPD + the matched fallback objective using current actions;
     keep hard teacher-trajectory loss disabled in the main OPD recipe.
  6. Sum final route counts across all ranks.
  7. Compute a_t, update EMA/mastery, and derive next-step actions.
  8. Log route, task-zero, disagreement, compute and leakage metrics.
```

## 5. 实验设计

### 5.1 研究问题

- RQ1：在 matched 4epoch 预算下，OPD 是否相对 no-OPD 带来稳定的 ChartQA 净收益？
- RQ2：OPD 是否与 GRPO 和 fallback supervision 互补，而不能被任一单独信号替代？
- RQ3：verifier-routed OPD 是否优于 unconditional OPD，证明可靠性选择而非单纯增加 teacher compute 产生收益？
- RQ4：realized-autonomy adaptive support 是否能进一步改善 OPD 的 accuracy/teacher-compute Pareto？
- RQ5：OPD 的收益是否在改变 epoch、batch 或数据规模后仍然成立？

### 5.2 数据、模型与评估

主任务为 ChartQA，student 为 LLaVA-OneVision 0.5B，teacher 为 7B。所有主结果固定 4epoch，并报告有效 batch、训练样本数、视觉证据、gold access、teacher calls、generated tokens、GPU hours 和 eval processed count。

正式主表要求 `2500/2500`；8-GPU 分片 eval 的 `2496/2500` 可用于快速迭代决策，但必须标注 processed count。最终准确率只从 `eval_chartqa/summary.csv` 读取。

### 5.3 主结果设置

至少比较：

1. Base/SFT；
2. DyME official；
3. matched gold-hidden-teacher no-OPD；
4. gold-hidden-teacher unconditional OPD；
5. gold-hidden-teacher verifier-routed OPD without controller；
6. gold-hidden-teacher verifier-routed OPD with adaptive support；
7. oracle official；
8. oracle OPD upper bound。

oracle 行与 gold-hidden-teacher 行必须在 `Teacher sees gold` 列明确分开；所有行另设 `Verifier uses reference`。

### 5.4 消融与分析

第一组核心消融比较 no-OPD、OPD-only、GRPO-only、fallback-only、OPD+GRPO
和完整三路训练，直接检验信号互补性。第二组比较 unconditional OPD、reward gate、
token-selective OPD 和 verifier-confirmed completion routing，检验 OPD 的可靠使用方式。
只有在 OPD 主效应成立后，第三组才比较 fixed weights、fixed step、normalized progress、
mixed/zero-loss proxy 和 global GRPO route，并拆分 OPD weight 与 OPD completion cap。
hard trajectory 作为 supervision-type 负对照单列，不作为主 controller action。

训练动力学统一报告 global GRPO/OPD/SFT、task all-wrong、task zero-loss、total-reward zero-loss、disagreement、accuracy、clip、EOS、degenerate、controller EMA/mastery/support 和 teacher funnel。full-CoT 比例不是单独的负指标；只有它与错误模板、截断、parse failure 或答案错误关联时才作为失败分析。

## 6. 当前结果与实验进度

### 6.1 已验证基线

| Method | Teacher sees gold | Verifier uses reference | Epoch | ChartQA accuracy | Processed |
|---|---|---|---:|---:|---:|
| gold-hidden-teacher PCD aligned | no | yes | 4 | 0.5420 | 2500/2500 |
| oracle route_guard | yes | yes | 4 | 0.5592 | 2500/2500 |
| oracle full-template repair | yes | yes | 4 | 0.5624 | 2500/2500 |
| oracle constrained repair | yes | yes | 4 | 0.5656 | 2500/2500 |
| oracle student_hint_short | yes | yes | 4 | 0.5800 | 2500/2500 |
| oracle official | yes | yes | 4 | 0.5872 | 2500/2500 |
| oracle OPD + full teacher trajectory | yes | yes | 4 | 0.5120 | 2500/2500 |

这些数字来自 `experiment_ledger.md` 中列出的 eval artifacts。它们说明短目标 repair 改善了现有 oracle pipeline，但不证明 gold-hidden-teacher OPD 有效。更重要的是，OPD 与 full teacher-trajectory hard supervision 的直接组合只得到 `0.5120`：尽管该 run 的 last50 train accuracy/global GRPO route 已达到约 `0.445/0.486`，held-out 输出中却有 `2397/2500` 被归为 full-CoT，`Goal:` 出现 2415 次，并伴随大量空 section 与异常 `Answer:`。因此，训练 reward 上升不能证明 teacher trajectory 形成了可迁移推理；hard imitation 可能把格式先验强化为 train-eval mismatch。

这一负结果进一步明确本文的 OPD motivation。我们并不反对 full-CoT，也不把结构化推理本身视为错误；问题是 student 是否被迫复现 teacher 的固定 hard sequence。OPD 的目标恰恰是避免这一点：teacher distribution 应作用于 student-generated states，而不是用整条 teacher trajectory 替换学生状态。后续 matched 实验因此关闭 teacher trajectory 和 teacher-SFT repair，仅保留 verifier-routed OPD。

### 6.2 当前运行实验

`global_grpo_route_full_4epoch_20260712_205549` 已完成，其 `0.5120` 结果作为“OPD + hard trajectory”负对照。

当前运行 `oracle_opd_no_hard_imitation_adaptive_4epoch_20260713_121946`。它只改变一个核心因素：关闭 teacher trajectory 与 teacher-SFT repair，同时保留 verifier-routed OPD、GRPO/fallback、effective sampling 与 realized-autonomy support。训练日志强制报告 hard-imitation invariants，以及 full-template、partial-template、Goal-without-Answer、empty-skeleton 和 malformed-answer 行为指标；持续完整模板塌缩会自动停止任务。partial drift 仅作为早期告警，因为它在被 probe 的错误 completion 上是条件统计，历史 `0.5872` oracle run 也曾短暂达到很高比例，不能单独预测 held-out 失败。该 run 仍使用 oracle hint，只能作为 OPD 隔离实验与 oracle upper bound，不能支持 gold-hidden-teacher 主张。

### 6.3 论文完成门槛

效果主张只有在以下条件满足后进入摘要：

1. matched gold-hidden-teacher no-OPD 与 verifier-routed OPD 均完成 4epoch 和完整 eval；
2. OPD 明确优于统一预算 no-OPD/DyME 对照；
3. leakage metric 为 0；
4. teacher compute 有可比较统计；
5. OPD-only、GRPO-only、fallback-only 与联合训练支持互补性；
6. 若声称超过 60%，必须存在 accuracy `>0.60` 的有效 summary artifact。

## 7. 讨论与局限

OPD 的有效性依赖 teacher 在学生生成状态上的可靠性。如果 DePlot 或 visual facts 错误，teacher-correct gate 可能把错误证据转成高置信监督。gold-hidden teacher 降低直接答案模仿，但 verifier 仍使用 reference，且该设置不自动保证视觉 grounding。未来需要使用更严格的 evidence attribution、reference-free reliability diagnostic 或视觉 token reliability。

单调 mastery 提供稳定课程，但假设学生自主能力总体不回退。若高学习率、数据分布变化或灾难性遗忘导致长期 regression，控制器可能继续保持过低 teacher support。非单调滞回控制器是重要消融方向。

单一 realized-GRPO 信号还存在另一种潜在失效：当 GRPO coverage 很低时，controller
按设计保持最大 OPD support；但若某些 teacher token 主要传递格式偏好而非可迁移答案
信息，高 OPD 又可能延迟有效 GRPO 的出现，从而形成“低 autonomy -> 强 OPD -> 继续低
autonomy”的正反馈。本文不通过向 controller 临时加入第二个健康信号来掩盖该问题，
而把它视为 local OPD reliability 问题：比较统一 token OPD 与 token-selective OPD，
并分别报告结构标题、答案位置和视觉相关 token 的接受率。

当前主任务是 ChartQA，短答案可验证性强。几何、医学或开放式视觉推理中的 verifier noise 和 teacher recoverability 定义可能不同。跨任务实验是证明方法普适性的必要后续，而不是从单任务结果直接外推。

## 8. 结论

本文研究如何把 OPD 引入 sub-1B VLM 的可验证推理训练。OPD 在学生自己生成的错误状态上提供稠密分布指导，填补了离线 SFT 的状态分布偏移与 RLVR 的稀疏、零优势信号之间的空白。为使这一信号可靠工作，我们使用 verifier-confirmed routing 选择可恢复状态，并以学生实际 GRPO coverage 调节 OPD 的介入强度。论文的核心结论将由 matched no-OPD 对照和单信号/联合信号消融决定：OPD 是否有效，以及它是否与 GRPO 和 fallback supervision 互补。

最终版本将以统一 4epoch、明确 gold access、完整 held-out eval 和 teacher compute 对照验证该目标；在证据完成前，不提前宣称 OPD 方法已超过 60% 或已优于 DyME。
