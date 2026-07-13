# On-Policy Distillation for Sub-Billion Vision-Language Reasoning

中文工作标题：**面向十亿参数以下视觉语言推理的在策略蒸馏**

> 论文状态：方法与实现章节可按当前代码写作；效果数字以
> `claim_evidence_matrix.md` 和 `experiment_ledger.md` 为准。最近的 oracle 诊断 run
> 已在 step 60 因健康闸门失败而停止；真正关闭全部 full-hint hard SFT 的 matched run
> `oracle_opd_no_full_hint_hard_sft_adaptive_4epoch_20260713_150545` 在 step 86 因 rank 7
> CUDA 瞬态错误中止，没有 checkpoint 或 final eval。相同配方的 resilient rerun
> `oracle_opd_no_full_hint_hard_sft_adaptive_resilient_4epoch_20260713_181613` 已排队，
> 使用 50-step checkpoint 和自动恢复。截至 `2026-07-14 00:47 CST`，该 run 仍由 GPU
> gate 等待外部任务释放显存；外部 ScaleDivide 训练最新完成 epoch 55，且 GPU 0/4/7
> 仍有间歇性视频生成任务。当前 run 尚未生成 output/checkpoint。gold-hidden-teacher CLRC 主实验仍未完成，因此
> 摘要不预填 CLRC 最终准确率。
> 术语勘误：当前非 oracle 设置是 teacher prompt 不含 gold、但 routing verifier
> 使用 reference answer；正式稿应称为 gold-hidden-teacher / verifier-available，
> 不能称为“完全无 gold recoverability estimator”。

## 摘要

不足十亿参数的视觉语言模型部署成本低，却很难从现有训练信号中学会可靠推理。监督微调（SFT）要求学生模仿离线 teacher 轨迹，训练稳定但容易偏离小模型推理时真正访问的状态；基于可验证奖励的强化学习（RLVR）直接优化学生输出，但当同组采样全部答错时，GRPO 几乎没有相对优势可学。两者之间因此留下一个关键状态：**学生已经访问了一个错误推理前缀，当前策略无法从奖励中改进，但 teacher 能够在该前缀上给出可靠纠正。**

本文研究 on-policy distillation（OPD）能否填补这一缺口。OPD 不要求学生复现完整 teacher 轨迹，而是在 student-generated prefixes 上提供下一 token 分布，因此既保留 on-policy 状态，又能为全错组提供稠密信号。据我们所知，本文是首个系统检验 **OPD 在 sub-1B VLM 可验证推理中的净收益及其与 GRPO、fallback 互补性** 的工作。为可靠使用 OPD，我们提出 CLRC：正确 completion 使用 GRPO；错误但被 gold-hidden teacher 可靠纠正的 completion 使用 OPD；其余 completion 进入受控 fallback。实际进入 GRPO 的全局 completion 比例进一步控制后续 OPD exposure，使 teacher 支持随学生自主能力增长而平滑退出。

我们在 ChartQA 上采用统一 4epoch 协议，以 route-matched no-OPD、无条件 OPD、固定/自适应路由 OPD 以及单信号和联合信号消融，分别回答三个问题：OPD 是否带来独立收益，可靠路由是否必要，以及 OPD 是否能提供 GRPO 与 fallback 无法替代的学习信号。所有实验同时披露 teacher-input gold access、routing-verifier reference access、route occupancy、hard-target exposure、teacher compute 与完整 held-out accuracy。当前主实验仍在运行，最终稿只从有效 eval artifact 填入结果。

## 1. 引言

视觉语言模型正在从感知走向基于视觉证据的推理。图表理解、几何推理和视觉问答不仅要求模型识别内容，还要求它定位证据、执行比较或计算，并给出可验证答案。大模型可以依靠大量长链轨迹和强化学习获得这些能力，但实际部署常需要参数量不足十亿的小型 VLM。此类模型推理成本低，却更容易受到容量、视觉 grounding 和稀疏奖励的共同限制 [Zhou2024TinyLLaVA; Chu2024MobileVLMV2; Marafioti2025SmolVLM; Liu2025DyME]。

现有训练主要依赖 SFT 与 RLVR。多模态 CoT、LLaVA-CoT、Insight-V 和 Mulberry 使用高质量 rationale 教模型逐步作答 [Zhang2023MultimodalCoT; Xu2024LLaVACoT; Zhang2024InsightV; Yao2024Mulberry]，但离线 teacher 轨迹不一定落在小模型能够稳定实现的分布内。对于容量有限的学生，模仿语言上完整的长轨迹可能形成答案错误的 pseudo reasoning，并限制后续探索 [Yang2025SFTorRL]。Visual-RFT、VLM-R1、Reason-RFT、LMM-R1 和 R1-VL 则表明，可验证奖励能够直接激励多模态推理 [Liu2025VisualRFT; Shen2025VLMR1; Zhang2025ReasonRFT; Peng2025LMMR1; Zhang2025R1VL]；然而当同组 completion 全错或奖励相同时，GRPO 的相对优势接近零，更新几乎不包含任务学习信号。

DyME 将这一矛盾概括为 memorization 与 exploration 的动态选择：学生尚未找到正确解时使用 SFT，已经产生正确解时使用 RL [Liu2025DyME]。这一二选一机制显著优于固定两阶段训练，但它把所有“尚未找到正确解”的状态都归入 memorization。实际上，全错组至少包含两类不同失败：一类错误已经落在 teacher 能可靠纠正的 student state 上，另一类连 teacher 也无法从当前视觉证据中恢复。前者若回到离线 `hint + answer` SFT，会丢失学生实际访问的前缀；若只使用 GRPO，又没有非零相对优势。**本文关注的正是这个被 SFT/RL 二分法遗漏的可恢复错误状态。**

OPD 为这一状态提供了直接信号。它先让学生生成 completion，再让 teacher 在 student-generated prefixes 上给出下一 token 分布 [Agarwal2023GKD]。因此，OPD 不要求学生复现完整 teacher sequence，也不要求同组先出现正确答案，便能在学生自己的错误状态上提供稠密纠正。近期梯度几何分析还表明，OPD 的更新结构不能简单还原为 SFT 或 RLVR，这进一步说明它是否提供独立收益应由专门消融检验，而不是被归入已有二选一范式 [Shen2026GeometryOPD]。这一点对 sub-1B VLM 尤其关键：模型越小，离线 teacher trajectory 与学生可实现 trajectory 的差距越大，而纯 RLVR 找到首个正确解的概率越低。据我们所知，本文首次在 sub-1B VLM 可验证推理中系统检验 OPD 的有效性，以及它与 GRPO、fallback 是否构成互补学习信号。我们不声称首次提出 VLM OPD；VOLD、REOPOLD、Decomposed OPD 和 Visual-Advantage OPD 已将 OPD 用于视觉语言推理 [Bousselham2025VOLD; Yu2026REOPOLD; Yoon2026DecomposedOPD; Liu2026VAOPD]。

把 OPD 接入训练并不自动产生收益。我们的统一预算负对照中，OPD 与完整 teacher-trajectory hard supervision 直接叠加仅得到 `51.20%`，低于 `58.72%` 的 oracle official；held-out 输出同时出现大规模固定结构和异常答案段。进一步审计发现，即使关闭 teacher-generated trajectory，legacy online SFT 仍会把 ChartQA 的完整 `Goal/Observation/Reasoning/Conclusion` hint 作为 hard target。这个反例揭示了两个问题。第一，若不隔离 full-hint hard imitation，就无法把收益归因于 OPD 的 student-state guidance。第二，结构化 CoT 本身不是污染；真正的风险是用同一条完整 teacher sequence 对大量 on-policy completion 施加硬替换，导致模型记忆固定表面形式而没有改善视觉推理。

现有近邻进一步限定了本文的问题边界。VOLD 已联合 cold-start alignment、GRPO 与 OPD；REOPOLD 已在 3B/7B 视觉推理模型上比较多种 OPD teacher；ViCuR 使用视觉可恢复线索筛选 privileged supervision；RG-OPD 使用 verifier reward 判断 teacher 是否值得蒸馏；SSOPD 则利用 mixed group 内的正确 completion 纠正错误 completion [Bousselham2025VOLD; Yu2026REOPOLD; Tian2026ViCuR; Akhondzadeh2026RGOPD; Tan2026SSOPD]。因此，VLM reasoning OPD、OPD+RLVR、verifier-gated teacher trust 和 correct-wrong completion 对比都不能单独构成本文的创新。仍缺少的证据是：**在 0.5B VLM 的 all-wrong 状态中，当组内没有 student-correct witness 时，外部 teacher 在 student prefix 上提供的可验证 OPD 是否带来独立收益，并能否与随后出现的 GRPO 信号互补。**

直接对所有错误 completion 使用 OPD 同样不可靠。teacher 可能误读视觉证据，长 prefix 上的纠正能力可能衰减，不同 token 的 teachability 也不同；最新分析还表明，privileged teacher 与 thinking student 在关键分叉 token 上的策略差异可能使 naive privileged OPD 伤害推理能力 [Fu2026RevisitingOPD; Kaur2026PrivilegedOPD; Xie2026IWOPD; Wang2026TeachabilityOPD; Liu2026PWOPSD; Liu2026SFD]。此外，学生对 teacher 支持的需求会随训练变化：早期正确探索稀少，需要更多稠密指导；后期若仍保持高 OPD exposure，则可能压制学生自主探索。由此，关键不再是“是否加入 OPD loss”，而是 **哪些 completion 使用 OPD，哪些保留 GRPO，哪些进入 fallback，以及 OPD 应何时退出。**

为此，我们提出 CLRC。对每个 student completion，正确答案进入 GRPO；错误 completion 交由不接收 reference answer 的多模态 teacher 重新作答，只有 teacher 答案通过 RLVR verifier 时才使用 OPD，其余 completion 进入受控 fallback。随后，系统用所有 data-parallel rank 上实际进入 GRPO 的 completion 比例控制下一步 OPD exposure，而不是依赖固定 epoch 或 step 边界。这里的 gold-hidden 只描述 teacher 输入，routing verifier 仍使用与 RLVR reward 相同的 reference。routing 与 controller 都服务于同一个目的：使 OPD 成为可靠、可归因，并能随学生自主能力形成而退出的第三类学习信号。

本文的贡献如下：

1. 据我们所知，我们首次系统研究 OPD 在 sub-1B VLM 可验证推理中的净收益，聚焦 SFT student-state mismatch 与 RLVR all-wrong zero-advantage 之间的可恢复错误状态。
2. 我们提出面向 OPD 的三路训练机制：verifier-confirmed routing 将 completion 分配给 GRPO、OPD 或 fallback，realized-GRPO 连续控制器再按实际自主覆盖率调节 OPD exposure。
3. 我们建立统一预算、可归因的验证协议，以 route-matched no-OPD、无条件 OPD、固定/自适应路由以及单信号/联合信号消融检验 OPD 的独立收益、可靠性与互补性，并显式披露 gold access、hard-target exposure、route occupancy 与 teacher compute。

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

GKD 将 teacher feedback 移到 student-generated sequences 上，系统化提出 on-policy distillation [Agarwal2023GKD]。OPD 的梯度几何研究进一步指出，on-policy rollout 与 teacher 分布共同产生不同于 SFT 和 RLVR 的更新结构，为将 OPD 作为独立训练信号进行分析提供了理论依据 [Shen2026GeometryOPD]。近期研究快速扩展了这一方向：VOLD 将 LLM reasoning 迁移到 VLM，REOPOLD 在视觉推理上系统比较不同 OPD teacher，Vision-OPD 用 privileged crop teacher 向 full-image policy 自蒸馏，Decomposed OPD 和 VA-OPD 分别处理视觉 grounding 与视觉优势加权 [Bousselham2025VOLD; Yu2026REOPOLD; Xue2026VisionOPD; Yoon2026DecomposedOPD; Liu2026VAOPD]；IW-OPD、control-variate OPD 和 best-of-N teacher selection研究位置偏差、方差和 teacher rollout 选择 [Xie2026IWOPD; Oh2026vOPD; Zhang2026BRTS]；TrOPD、TIP 与 AOPD 分别研究可靠分布区域、token importance 和零/负 advantage 区域的局部蒸馏目标 [Xing2026TrOPD; Xu2026TIP; Jia2026AOPD]；RG-OPD、DOPD 和 SCOPE 研究 verifier gating、privileged supervision 与自适应权重 [Akhondzadeh2026RGOPD; Yu2026DOPD; Zheng2026SCOPE]；privileged-OPD failure analysis、TA-OPD、PW-OPSD、SFD 与 DEAR 则说明 privileged policy mismatch、teacher disagreement、token position、长 student prefix 和 decision-supporting evidence token 上的纠正信号并非同等可靠 [Kaur2026PrivilegedOPD; Wang2026TeachabilityOPD; Liu2026PWOPSD; Liu2026SFD; Xiao2026DEAR]。SSOPD 从 mixed group 的正确与错误 completion 构造无外部解轨迹的 process supervision，但其信号在 all-wrong group 中不可用 [Tan2026SSOPD]。GateKD 也使用 confidence-gated closed-loop distillation 的表述 [Sermsri2026GateKD]。

最接近本文的 VLM OPD 工作仍处在更大的 student 区间。VOLD 以 3B VLM 为 student，
并依赖 cold-start SFT、GRPO 与 OPD 的多阶段 curriculum；VA-OPD 在 2B student 上
估计视觉 token advantage；Decomposed-OPD 在 2B/4B student 上分解视觉证据、推理与
答案区域；DOPD 在 2B VLM 上路由 privileged/student token supervision；ViCuR 在
2B/8B VLM 上以 recoverable visual cue 替代 answer-side privilege；REOPOLD 在
3B/7B student 上研究视觉推理的 teacher 选择；VOLD 则表明
cold-start distribution alignment 对后续联合 GRPO+OPD 的有效迁移很重要
[Bousselham2025VOLD; Liu2026VAOPD; Yoon2026DecomposedOPD; Yu2026DOPD;
Tian2026ViCuR; Yu2026REOPOLD]。这些工作回答了“如何在 VLM 中使用 OPD”、
“如何处理 privileged mismatch”或“如何用 OPD 改善 RLVR 初始化”，但没有直接回答容量
更受限的 0.5B student 在 RLVR 全错状态下是否能从 OPD 获得净收益。这个差异必须由
matched 实验而不是参数规模描述来证明：相同初始化和 4epoch 预算下，比较 DyME/no-OPD、
unconditional OPD、verified wrong-state OPD 和完整联合训练，并报告 full-hint hard-target
exposure、route occupancy 与 held-out accuracy。

| Method | Student scale | OPD selection unit | Hard trajectory role | Primary question |
|---|---:|---|---|---|
| VOLD | 3B | staged global curriculum | cold-start SFT is required | transfer reasoning from LLM to VLM |
| REOPOLD | 3B/7B | teacher/model selection for visual reasoning OPD | task-dependent teacher trajectories | which teacher best supports visual reasoning OPD |
| VA-OPD | 2B | visual-token advantage | retained as part of the recipe | which visual tokens deserve stronger distillation |
| Decomposed-OPD | 2B/4B | evidence/reasoning/answer regions | task recipe dependent | how to decompose visual grounding and reasoning guidance |
| DOPD | 2B VLM | privileged/student token router | privileged branches are central | how to avoid privilege illusion at token level |
| ViCuR | 2B/8B | visual-cue recoverability | privileged signal filtered by visual recoverability | which teacher privilege is recoverable from the image |
| SSOPD | 8B LLM | mixed-group correct-to-wrong self-distillation | no external solution trace | whether a student-correct witness can repair a student-wrong prefix |
| TrOPD / TIP / AOPD | LLM | reliable region / token / local objective | task dependent | where and how token-level OPD remains reliable |
| Ours | 0.5B | verifier-confirmed wrong student completion | measured and isolated from OPD | whether OPD fills the all-wrong RLVR gap beyond DyME fallback |

因此，本文不声称首次提出 VLM OPD、recoverable privilege、OPD-to-RLVR integration、teacher reliability gating 或 closed-loop distillation。我们的核心定位是：**据我们所知，首次在 sub-1B VLM 可验证推理中系统引入和评估 OPD，并以严格消融证明 OPD 相对 no-OPD 的净收益以及它与 GRPO、fallback supervision 的互补性。** 与 ViCuR 的 visual-cue privilege 不同，本文的局部状态是 verifier-confirmed wrong student completion；与 VOLD 强调 cold-start distribution alignment 不同，本文重点检验 all-wrong completion 的 OPD 净收益，并依据最终实际 route occupancy 连续调节 OPD exposure。verifier routing 与闭环 OPD exposure control 是为这一核心问题服务的方法设计，而不是与 OPD 并列的独立论文主线。

### 2.5 课程学习、图表推理与实验设置

自动课程学习根据学生进度选择训练任务或环境，说明训练日程应随能力变化，而非只依赖固定 step [Matiisen2017TSCL; Portelas2019ALPGMM]。这一思想也被扩展到可靠 LLM reasoning 和 self-evolving reasoning curriculum [Zhao2024AutoCEI; Chen2025SelfEvolvingCurriculum]。CLRC 的 controller 与课程学习共享能力驱动的思想，但它不选择下一个数据样本，而是观察已经发生的 GRPO/OPD/fallback route occupancy，并调节下一步 OPD loss weight 与 post-probe route cap。因此 controller 是 OPD 主线的配套机制，其独立价值必须由 fixed-weight、fixed-progress 和 proxy-state 消融验证。

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

其中 `D` 可取 JSD、forward KL 或其他分布距离；本文当前 clean OPD 主配置使用 JSD。
两种方法都使用 teacher，但监督发生的
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

fallback 可以是受控 trajectory/SFT repair，也可以在 teacher 信号不可信时跳过。论文主实验必须固定 fallback 定义，避免把多个机制同时变化。当前 clean OPD isolation 配方不使用任何 hard fallback：teacher probe 失败时，mixed group 中的错误 completion 回到 GRPO，all-wrong group 中的错误 completion 进入 skip。该定义与 teacher trajectory、teacher-SFT repair、legacy online SFT 和 malformed-output forced SFT 全部关闭的约束共同固定。

### 4.2 三路联合目标

令 `M_G`、`M_O`、`M_F` 分别为最终 GRPO、OPD 和 fallback masks。主方法目标写为：

```text
L_t = lambda_G L_GRPO(M_G)
    + lambda_O(t) L_OPD(M_O)
    + lambda_F(t) L_fallback(M_F).
```

`L_fallback` 可以是受控 SFT，也可以为零损失 skip，但在 matched 消融中必须保持定义不变。
对于当前 clean isolation 实例，`M_F` 仅对应 all-wrong probe failure 的 skip，因而
`lambda_F L_fallback = 0`；mixed probe failure 不产生新的 imitation target，而是保留其
原有 GRPO route。这里保留一般形式是为了描述消融空间，不表示主配置仍使用 SFT。
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

再映射为 OPD supervision exposure：

```text
u_t = clip(m_t / tau, 0, 1),
s_t = 1 - smoothstep(u_t),
smoothstep(u) = u^2 (3 - 2u).
```

当前 target `tau = 0.30`。控制器实现可让多个 OPD-supervision 动作共享同一
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

当前 completion cap 在 teacher probe 完成后应用，因此它控制的是进入 OPD loss 的
completion 数量，而不是已经发生的 teacher calls 或 generated tokens。本文不据此声称
降低 teacher compute；只有未来将 cap 前移到 probe generation 之前并完成 matched
compute/accuracy 对照后，才能支持效率主张。

使用单调 mastery 的目的是避免偶然坏 batch 让 OPD exposure 反复振荡。其局限是学生发生长期能力回退时不能自动恢复 exposure，本文将在局限和非单调控制器消融中讨论。

### 4.5 因果顺序与分布式一致性

step `t` 的生成和路由使用 step `t-1` 后保存的控制状态。step `t` 最终 route 完成后，系统才计算 `a_t` 并更新控制器，供 step `t+1` 使用。该一拍延迟避免同一步 route 既是动作结果又是动作输入。

所有 controller rank 读取同一个跨 rank snapshot。早期诊断显示 rank-local mixed/zero-loss 与 global task state 可显著不同，因此论文只使用全局最终 route 作为控制信号，local health metrics 仅作诊断。

### 4.6 算法流程

```text
Input: student policy, teacher, verifiable reward, visual evidence
Initialize EMA z_0=0, mastery m_0=0, full OPD supervision exposure
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
- RQ4：realized-autonomy adaptive exposure 是否能进一步改善 accuracy，同时减少进入 OPD loss 的 completion 比例？
- RQ5：OPD 的收益是否在改变 epoch、batch 或数据规模后仍然成立？

### 5.2 数据、模型与评估

主任务为 ChartQA，student 为 LLaVA-OneVision 0.5B，teacher 为 7B。所有主结果固定 4epoch，并报告有效 batch、训练样本数、视觉证据、gold access、teacher calls、generated tokens、GPU hours 和 eval processed count。

统一 4epoch 协议以实际 `resolved_config.json` 和 `run_env.json` 为准：

| Component | Setting |
|---|---|
| Train / test | ChartQA train `23,171`；test `2,500` |
| Student / teacher | LLaVA-OneVision `0.5B` / `7B`，BF16 |
| Distributed training | 8 GPU，DeepSpeed ZeRO-1 |
| Rollout | 每 prompt `8` completions，max completion length `96`，temperature `0.5`，repetition penalty `1.5` |
| Optimization | per-device batch `2`，gradient accumulation `16`，learning rate `5e-5`，warmup `50` steps，weight decay `0.01`，max grad norm `1.0`，seed `42` |
| GRPO | task/format/context reward weights `1.0/0.5/1.5`，GRPO loss weight `1.0` |
| OPD | JSD，`beta=0.5`；initial/final weight `1.5/0.5`；per-prompt cap `8/2` |
| Controller | realized global GRPO route，EMA `0.10`，target readiness `0.30` |
| Teacher probe | deterministic decoding，max `500` new tokens；completion-level verifier gate |
| Probe failure fallback | mixed group 回到 GRPO；all-wrong group skip；不构造 hard target |
| OPD candidate handling | 不跳过 degenerate wrong completion；仍须通过 teacher correctness/quality gate |
| Hard supervision in clean OPD | teacher trajectory、teacher-SFT repair、legacy online SFT 和 forced SFT 全部关闭 |

oracle diagnostic 使用 `format_only + visual_facts_deplot + oracle_hint`；正式
gold-hidden-teacher 主实验移除 `oracle_hint`，但仍明确披露 routing verifier 使用
reference。除消融指定因素外，模型初始化、数据顺序、generation、优化器、训练 epoch 和
eval 协议保持一致。teacher calls、teacher generated tokens 和 GPU hours 从运行日志汇总，
不能用 candidate 数量替代 teacher compute。

RQ1 使用两个不同的零 OPD 对照。`GRPO-only` 关闭 teacher probe 与 OPD，用于测量纯
RLVR 的 all-wrong zero-signal；`route-matched no-OPD` 保留与 verifier-routed OPD 相同的
teacher calls、candidate selection、completion routes 和 skip 行为，只把 OPD loss weight
设为零。后者是 OPD loss 主效应的直接因果对照，不能与前者合并报告。

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
和完整三路训练，直接检验信号互补性。其中 fallback-only 只有在顶层 base-loss mask
能够证明实际 GRPO contribution 为零后才进入结果表；当前仅有 runner 环境合同，不构成
有效单信号结果。第二组比较 unconditional OPD、reward gate、
token-selective OPD 和 verifier-confirmed completion routing，检验 OPD 的可靠使用方式。
只有在 OPD 主效应成立后，第三组才比较 fixed weights、fixed step、normalized progress、
mixed/zero-loss proxy 和 global GRPO route，并拆分 OPD weight 与 OPD completion cap。
hard trajectory 作为 supervision-type 负对照单列，不作为主 controller action。

训练动力学统一报告 global GRPO/OPD/SFT、task all-wrong、task zero-loss、total-reward zero-loss、disagreement、accuracy、clip、EOS、degenerate、controller EMA/mastery/support 和 teacher funnel。full-CoT 比例不是单独的负指标；只有它与错误模板、截断、parse failure 或答案错误关联时才作为失败分析。

## 6. 当前结果与实验进度

### 6.1 已验证基线

| Evidence regime | Method | Teacher sees gold | Verifier uses reference | Hard-target status | ChartQA acc. | Paper role |
|---|---|---|---|---|---:|---|
| gold-hidden teacher | legacy PCD aligned | no | yes | legacy exposure not fully audited | 0.5420 | historical lower baseline, not clean OPD causal row |
| oracle | route guard | yes | yes | legacy online-SFT recipe | 0.5592 | routing baseline |
| oracle | full-template repair | yes | yes | full template hard repair | 0.5624 | hard-target diagnostic |
| oracle | constrained repair | yes | yes | constrained hard repair | 0.5656 | hard-target diagnostic |
| oracle | student_hint_short | yes | yes | short hard repair | 0.5800 | strongest completed internal recipe |
| oracle | official | yes | yes | official DyME-aligned recipe | 0.5872 | privileged upper baseline |
| oracle | OPD + full teacher trajectory | yes | yes | full trajectory hard supervision | 0.5120 | negative control |

所有行均为 4epoch、`2500/2500` eval，数字来自 `experiment_ledger.md` 中列出的
artifact。该表是诊断账本，不是最终主结果表：除第一行外均使用 oracle evidence，第一行
又缺少当前 clean hard-target exposure 审计，因此没有任何一行可以证明 gold-hidden
verifier-routed OPD 的净收益。它们说明短目标 repair 改善了现有 oracle pipeline，但不证明
gold-hidden-teacher OPD 有效。更重要的是，OPD 与 full teacher-trajectory hard supervision
的直接组合只得到 `0.5120`：尽管该 run 的 last50 train accuracy/global GRPO route 已达到
约 `0.445/0.486`，held-out 输出中却有 `2397/2500` 被归为 full-CoT，`Goal:` 出现 2415 次，
并伴随大量空 section 与异常 `Answer:`。因此，训练 reward 上升不能证明 teacher trajectory
形成了可迁移推理；hard imitation 可能把格式先验强化为 train-eval mismatch。

这一负结果进一步明确本文的 OPD motivation。我们并不反对 full-CoT，也不把结构化推理本身视为错误；问题是 student 是否被迫复现 teacher 的固定 hard sequence。OPD 的目标恰恰是避免这一点：teacher distribution 应作用于 student-generated states，而不是用整条 teacher trajectory 替换学生状态。后续 matched 实验因此关闭 teacher trajectory 和 teacher-SFT repair，仅保留 verifier-routed OPD。

### 6.2 诊断链与当前运行实验

`global_grpo_route_full_4epoch_20260712_205549` 已完成，其 `0.5120` 结果作为“OPD + hard trajectory”负对照。

诊断运行 `oracle_opd_no_hard_imitation_adaptive_4epoch_20260713_121946` 关闭了 teacher trajectory 与 teacher-SFT repair，同时保留 verifier-routed OPD、GRPO/fallback、effective sampling 与 realized-autonomy support。该 run 在 step 60 因恢复闸门失败而停止，不能作为完整效果结果。进一步代码审计发现，它仍保留 legacy online SFT：训练器以 dataset `hint + answer` 构造硬 target，而 ChartQA hint 本身包含完整 `Goal/Observation/Reasoning/Conclusion`。因此该实验只隔离了 teacher-generated hard trajectory，没有隔离所有 full-hint hard supervision；后续实验必须先关闭 online-SFT slots，才能检验纯 OPD 的 style-transfer 风险。该 run 仍使用 oracle hint，只能作为机制诊断与 oracle setting，不能支持 gold-hidden-teacher 主张。

clean run `oracle_opd_no_full_hint_hard_sft_adaptive_4epoch_20260713_150545`
进一步关闭 online-SFT slots、all-wrong SFT fallback 与 malformed-output forced SFT。
4-step 8-GPU smoke 和正式训练的所有已记录窗口中，teacher trajectory、teacher-SFT
repair、legacy online SFT 与 full-hint hard-target rate 均严格为零，因此该 run 首次真正
隔离了 soft OPD 与 full-hint hard imitation。正式训练在 steps 72--81 出现持续恢复：
latest-ten accuracy/GRPO/OPD route 为 `0.0836/0.0906/0.7500`，degenerate/clipped/EOS
为 `0.1406/0.2871/0.6656`，controller mastery 上升到 `0.0842`，OPD weight 从 `1.5`
降至 `1.308`。这些数字只说明训练动力学开始恢复，不能作为 held-out 效果。

该 clean run 在 step 86 的 DeepSpeed backward 中因 rank 7
`CUDA error: unspecified launch failure` 中止。GPU 7 Inforom BBX 的事件时间与崩溃
对齐，ECC 为零且无需 reset；旧 runner 又只在 epoch 边界保存，因此没有 checkpoint
或 final eval。论文将其标为 hardware-transient interrupted，而不是失败分数或完成实验。

相同方法配置的 resilient rerun
`oracle_opd_no_full_hint_hard_sft_adaptive_resilient_4epoch_20260713_181613` 已排队。
唯一变化是运行韧性：每 50 step 保存、保留 3 个 checkpoint，并在 8 卡显存、温度和
利用率连续满足门槛后启动；明确的 CUDA/NCCL 瞬态错误从最新 checkpoint 自动恢复。
训练完成后自动执行 8-GPU ChartQA final eval。在有效 `summary.csv` 产生前，本实验在
主结果表中保持空值。

### 6.3 论文完成门槛

效果主张只有在以下条件满足后进入摘要：

1. matched gold-hidden-teacher no-OPD 与 verifier-routed OPD 均完成 4epoch 和完整 eval；
2. OPD 明确优于统一预算 no-OPD/DyME 对照；
3. leakage metric 为 0；
4. teacher compute 有可比较统计；
5. OPD-only、GRPO-only、fallback-only 与联合训练支持互补性；
6. 若声称超过 60%，必须存在 accuracy `>0.60` 的有效 summary artifact。

这里的 `0.60` 是工程突破线，不是 DyME parity。DyME 原文在同为 LLaVA-OV-S 0.5B、
ChartQA relaxed correctness 的设置下报告 Pure DyME `64.9%`（Medium CoT）和 full
DyME `67.5%`（带 Visual Supervision）[Liu2025DyME]。由于其数据质量、视觉监督与本仓库当前协议并非
自动完全一致，正文不能直接把外部数字当作 matched baseline；但任何 `60.x%` 结果也不能
被描述为“达到 DyME”。论文必须补充统一模型、数据、decode、预算与 eval 的 in-repo
DyME reproduction，并分别比较 Pure DyME 和 full DyME。

此外，sub-1B 定位只有在四项证据同时成立时才构成方法贡献：第一，matched OPD 必须
优于相同预算的 DyME/no-OPD；第二，OPD-only、GRPO-only 与联合训练必须支持信号互补性；
第三，所有 full-hint hard-target exposure 必须可观测并在 clean causal run 中为零；第四，
结果必须给出 teacher compute 与 route occupancy，排除单纯增加 teacher 调用次数的解释。
若只满足模型规模更小而没有效果或机制消融，论文不得把 sub-1B 写成创新结论。

## 7. 讨论与局限

OPD 的有效性依赖 teacher 在学生生成状态上的可靠性。如果 DePlot 或 visual facts 错误，teacher-correct gate 可能把错误证据转成高置信监督。gold-hidden teacher 降低直接答案模仿，但 verifier 仍使用 reference，且该设置不自动保证视觉 grounding。未来需要使用更严格的 evidence attribution、reference-free reliability diagnostic 或视觉 token reliability。

单调 mastery 提供稳定课程，但假设学生自主能力总体不回退。若高学习率、数据分布变化或灾难性遗忘导致长期 regression，控制器可能继续保持过低 OPD exposure。非单调滞回控制器是重要消融方向。

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
