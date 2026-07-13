# OPD 研究定位笔记

日期：2026-06-22

> **历史定位说明（2026-07-13，已被取代）**：本文只保留早期 PCD/OPD 研究过程，
> 不再作为论文 claim、配置或实验结果的事实源。下文部分内容早于已完成的 ChartQA
> held-out eval、JSD clean-OPD recipe、no-full-hint hard-SFT isolation 和 sub-1B OPD
> 文献审计。文中的 `no-gold` 应解释为 **teacher input 不含 gold answer**；当前
> routing verifier 仍使用 reference，因此规范名称是 **gold-hidden teacher /
> verifier-available routing**。当前事实源是 `chinese_draft.md`、
> `claim_evidence_matrix.md`、`experiment_ledger.md` 和 `docs/opd_experiment_plan.md`。
> 当前核心创新是 verifier-routed OPD for sub-1B verifiable VLM reasoning；routing
> 与 realized-route controller 是使 OPD 可靠工作的支撑机制，不再使用“PCD 是主方法”
> 的表述，也不声称首次提出 VLM OPD。

这份笔记整理当前项目中 OPD 的算法理解、和 DyME/相关工作的 novelty 边界、需要补充的 related work、可用的 research question 和 motivation 叙事。主体用中文写，必要的专有名词保留英文或中英并列。

## 1. 当前项目背景

本项目基于 DyME。DyME 的核心问题设定是：Small VLM / SVLM 在推理训练中很难同时处理好 SFT 的“记忆”与 RLVR/GRPO 的“探索”。

DyME 将训练状态分成两类：

- **Memorization / 记忆模式**：当小模型不能生成可验证答案时，回退到 SFT。
- **Exploration / 探索模式**：当同一 prompt 的一组 rollout 中至少有一个正确答案时，使用 RLVR/GRPO。

DyME 对小 VLM 的诊断很重要：

- 长 CoT 的 SFT 容易让小模型学习到 pseudo-thinking traces，即形式上像推理、实际缺少 visual grounding 的伪思维轨迹。
- RLVR/GRPO 在所有 rollout 都错时会缺少有效 advantage signal，容易出现 advantage collapse。

当前 OPD 扩展不应该被写成简单的“DyME 加一个更强 teacher”。更准确的理解是：

- OPD 在 DyME 的 SFT fallback 和 GRPO exploration 之间加入第三种学习状态。
- 这个状态针对 **wrong but recoverable completions / 错误但可恢复的 completion**。
- 冻结的 7B VLM teacher 在没有 gold answer 泄漏的条件下判断该错误轨迹是否可恢复。
- 如果 teacher 能在 no-gold visual context 下解出同一 prompt，就对学生的 on-policy 轨迹做 token-level distillation；如果 teacher 也失败，则回退到 SFT。

关键本地文件：

- `scripts/test/train_opd.sh`：4 epoch fast OPD 启动脚本。
- `scripts/test/config/fast_profile.py`：fast baseline 中 DyME-aligned OPD routing 的定义。
- `config/config_opd_7b_dyme_probe.py`：更完整的 teacher-probe OPD 配置，包括 SRKL OPD 和 teacher-trajectory FKL。
- `opsd_utils/mode_router.py`：prompt/completion 路由逻辑。
- `opsd_utils/opsd_loss.py`：token-level OPD/OPSD loss。
- `opsd_utils/privileged/providers.py`：no-gold teacher context providers。

## 2. 当前 OPD 算法理解

`scripts/test/train_opd.sh` 对应的 fast OPD 路由可以概括为：

1. 每个 prompt 生成 `K` 个 completions。
2. 用 verifiable reward 判断 answer correctness。
3. 如果整组全错，则整组走在线 SFT。
4. 如果某个 completion 答对，则该 completion 走 GRPO。
5. 如果某个 completion 答错，但同组至少有一个 completion 答对，则对该错误 completion 运行 7B teacher probe。
6. 如果 teacher 在 no-gold context 下答对，则该错误 completion 走 OPD。
7. 如果 teacher 也答错，则该 completion 回退到在线 SFT。

fast OPD 的 teacher context 是有意设计成 anti-leakage 的：

- `privileged_providers = ["format_only", "visual_facts_deplot"]`
- `text_include_gold = False`
- teacher 不读取 answer、hint、reference reasoning。
- teacher 只看到格式提示和离线 DePlot visual facts。

更完整的 `config_opd_7b_dyme_probe.py` 还包含：

- `loss_type = "srkl"`，即 skew reverse KL 类型的 OPD loss。
- 可选的 teacher-trajectory FKL：teacher 自己生成的正确 trajectory 也可以被蒸馏。
- 继承自 DyME 的 visual supervision 机制。

因此后续论文和表格里至少要区分两个版本：

- **纯 teacher-probe OPD**：无 Visual Supervision，无 gold context，只使用 DePlot-only visual facts。
- **OPD + VS / teacher-trajectory 版本**：包含 visual checker/refiner 或 teacher trajectory distillation。

如果把这些结果混成一个 OPD 数字，方法会很难解释。

## 3. 当前本地证据与注意事项

本地 fast training logs 显示，OPD 在 4 epoch 设置下比 pure DyME 稳定。

日志层面的现象：

- fast DyME 后期接近 total degeneration：accuracy reward 接近 0，format reward 接近 0，clipped ratio 接近 1。
- fast OPD 后期 format reward 和 accuracy reward 明显更好，degeneration 更低。
- `train_opd_7b_dyme_probe_*` 相关日志更强，但看起来包含 visual supervision activity，因此不能和 clean fast OPD 混为一类。

粗略日志统计如下：

| Run | Accuracy Reward Mean | Last-20 Accuracy | Degenerate Mean | Last-20 Degenerate | OPD Call Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast DyME | 0.0003 | 0.0000 | 0.9921 | 1.0000 | 0.0000 |
| fast OPD | 0.0358 | 0.2691 | 0.8413 | 0.0437 | 0.0035 |
| OPD probe + VS-like run | 0.1926 | 0.2793 | 0.0325 | 0.0031 | 0.0279 |

注意：

- 这些是 training-log signals，不是 held-out evaluation score。
- 当前仓库里我没有找到明确的 held-out eval result 文件。
- 某些 OPD 输出目录的 metadata 显示曾有历史环境变量污染，所以最终论文 claim 必须基于干净的 config snapshot 和正式 eval。

## 4. Novelty 边界

### 4.1 不适合作为主 novelty 的点

下面这些点已经被 DyME 或其他工作覆盖，不建议作为主要创新点：

- “动态切换 SFT 和 RL”：DyME 已经是这个核心。
- “动态混合 SFT 和 RL”：CHORD、LUFFY 等工作已经接近。
- “在 student-generated outputs 上蒸馏 teacher”：GKD / OPD 类方法已有。
- “用更强 VLM teacher 改进 VLM training”：已有多篇 VLM RL/distillation 工作。
- “使用 visual facts、visual checker、visual refiner”：DyME 已经覆盖。

### 4.2 更安全的 novelty 方向

更稳的创新点应围绕 **recoverability-aware OPD for small VLM reasoning / 面向小 VLM 推理的可恢复性感知 OPD** 展开。

核心表述：

> DyME 将失败状态统一映射到 SFT fallback，将成功状态映射到 GRPO exploration。我们的 OPD 进一步细分失败状态：判断错误 completion 是否可由 no-gold visual teacher 恢复；只有可恢复的错误轨迹才接受 dense distribution guidance，不可恢复错误仍回退 SFT。

可强调的差异：

- **Recoverability-aware routing / 可恢复性感知路由**：不是所有 wrong samples 都被蒸馏，只有 teacher-solvable wrong completions 才走 OPD。
- **No-gold teacher probe / 无答案泄漏 teacher 探针**：teacher 必须在不看 answer/hint 的情况下答对，避免把 oracle leakage 当成提升。
- **Per-completion triage / completion 级三路分流**：在同一个 GRPO group 内部区分 correct、wrong-but-recoverable、unrecoverable，而不是只做 prompt-level binary switch。
- **Small-VLM focus / 小模型设定**：目标失败模式是 sub-1B VLM 的早收敛、伪 CoT 和 advantage collapse。
- **Dense guidance for wrong-but-near states / 对近错状态提供稠密指导**：在 reward 太稀疏、SFT 又太 off-policy 的区域，OPD 提供 token-level signal。

## 5. 候选 Research Question

推荐主问题：

> Can small VLMs learn visual reasoning more efficiently if DyME's binary fallback is refined into a recoverability-aware third mode that distills a no-gold visual teacher only on wrong-but-solvable student trajectories?

中文版本：

> 小 VLM 的失败 rollout 是否应被一概 SFT 回退？我们研究一种 recoverability-aware OPD：当学生已经进入可验证任务的局部可解区域、但单个 completion 出错时，用无答案泄漏的视觉 teacher 提供 token-level 分布监督；当任务完全不可恢复时才回退 SFT。

其他可选问题：

> 当 sparse reward 无法区分“有价值的部分推理”和“不可恢复失败”时，小 VLM 应如何从自己的错误 rollout 中学习？

> No-gold teacher recoverability 能否作为一种在线信号，帮助小 VLM 在 imitate、explore、distill 三种学习模式之间做选择？

> 对小 VLM 的早收敛阶段而言，dense teacher distribution guidance 是否比 hard CoT imitation 更有效？

## 6. Motivation 叙事

“OPD 在小 VLM 上比 SFT 收敛更快”这个观察可以保留，但建议写得更机制化。

推荐 motivation：

1. 小 VLM 容量有限，训练中很容易快速收敛到局部模板或退化输出。
2. 长 CoT SFT 提供的是 hard off-policy target，容易让模型记住格式和浅层模板，而不一定提升 visual grounding。
3. GRPO 提供的 reward 很稀疏；当 group 全错或格式崩掉时，advantage signal 很弱。
4. DyME 通过 SFT/GRPO switching 稳定训练，但它对失败状态的处理仍然较粗。
5. 并非所有 wrong rollout 都一样：有些是不可恢复失败，有些则是在视觉证据充分时可以被更强 no-gold teacher 解出的 recoverable mistakes。
6. OPD 利用这个中间区域：正确 rollout 继续 GRPO；不可恢复失败继续 SFT；错误但可恢复的 rollout 用 teacher distribution 做 dense on-policy guidance。

可直接写进论文的版本：

> We observe that the failure states of small VLM RLVR are heterogeneous. Some failures reflect missing task competence and require stable SFT fallback, while others occur in prompts where the visual evidence is sufficient and a stronger no-gold teacher can recover the answer. Treating both cases as SFT wastes on-policy information and reinforces off-policy CoT templates. We therefore introduce recoverability-aware OPD, which turns wrong-but-solvable rollouts into dense token-level supervision while preserving DyME's safe SFT fallback for truly unrecoverable cases.

中文对应：

> 我们观察到，小 VLM 在 RLVR 训练中的失败状态并不均质。有些失败反映模型尚未具备任务能力，需要稳定的 SFT fallback；另一些失败则发生在视觉证据已经足够、强 teacher 能在 no-gold 条件下恢复正确答案的样本上。若将这两类失败都统一映射到 SFT，不仅浪费了学生当前策略分布中的 on-policy 信息，也可能继续强化 off-policy CoT 模板。因此，我们引入 recoverability-aware OPD，将 wrong-but-solvable rollouts 转化为 dense token-level supervision，同时保留 DyME 对真正不可恢复失败的 SFT fallback。

## 7. 可以继续打磨的方法创新点

### 7.1 Recoverability-Aware Learning State

DyME 是两状态：

- all wrong -> SFT；
- any correct -> GRPO。

OPD 可以被定义为三状态：

- correct rollout -> GRPO；
- wrong but teacher-recoverable -> OPD；
- unrecoverable / all-wrong -> SFT。

这比“加一个 distillation loss”更像一个清晰的方法贡献。

### 7.2 No-Gold Teacher Probe

Teacher probe 可以作为核心方法模块来写：

- 它测试 prompt 是否能从视觉证据和格式引导中被解决。
- 它避免 gold-answer leakage 造成虚假的 recoverability。
- 它过滤掉 teacher 自己也解不出的样本，减少错误蒸馏。

这个点很关键，因为很多 VLM distillation 方法使用强 teacher，但不一定明确区分 teacher ability 和 answer leakage。

### 7.3 On-Policy Distribution Guidance 而不是 Hard CoT Imitation

SFT 给的是单一 hard target sequence。OPD 给的是学生当前 trajectory 上的 token-level distribution guidance。

可以强调：

- 比完整 CoT imitation 更不容易过拟合固定模板；
- 比 sparse reward 更稠密；
- 比 offline SFT 更贴近 student current policy。

### 7.4 Error-Aware Teacher Compute

当前路由不是对所有样本都调用 teacher，而是主要在 candidate wrong completions 上调用。

潜在 claim：

> Teacher compute is spent where the student is wrong but the prompt has evidence of solvability, rather than uniformly distilling every token or every prompt.

中文表达：

> Teacher compute 被集中用于学生出错但样本仍可恢复的区域，而不是对所有 prompt 或所有 token 做均匀蒸馏。

这个 claim 需要 route statistics 支撑。

### 7.5 Teacher-Trajectory Distillation 作为可选扩展

完整配置中包含 teacher trajectory FKL。这个部分建议作为：

- ablation/extension：只在 student wrong trajectory 上做 OPD vs OPD + teacher trajectory；
- 或者在实验显示它必要时，作为主算法组件。

不要在论文叙事中把这两个版本混在一起。

## 8. 需要补充的 Related Work

### 8.1 DyME 与小 VLM 推理

- DyME: *Empowering Small VLMs to Think with Dynamic Memorization and Exploration*. 这是直接 base method，应首先讨论。  
  https://arxiv.org/abs/2506.23061

### 8.2 SFT/RL Hybrid 与动态指导

- CHORD: *On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting*. 接近动态 SFT/RL balance。  
  https://arxiv.org/abs/2508.11408
- LUFFY: *Learning to Reason under Off-Policy Guidance*. 和 reasoning RL 中的 off-policy guidance 相关。  
  https://arxiv.org/abs/2504.14945
- *SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training*. 可支撑 SFT-vs-RL motivation。  
  https://arxiv.org/abs/2501.17161
- *SFT or RL? An Early Investigation into Training R1-like Reasoning Large Vision-Language Models*. 和 VLM reasoning post-training 选择相关。  
  https://arxiv.org/abs/2504.11468

### 8.3 VLM RLVR / R1-Style Reasoning

- Visual-RFT: *Visual Reinforcement Fine-Tuning*.  
  https://arxiv.org/abs/2503.01785
- R1-VL: *Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization*.  
  https://arxiv.org/abs/2503.12937
- VLM-R1: *A Stable and Generalizable R1-style Large Vision-Language Model*.  
  https://arxiv.org/abs/2504.07615
- LMM-R1: *Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL*.  
  https://arxiv.org/abs/2503.07536
- R1-V: 低成本 VLM RL reasoning project，可作为 practical RLVR baseline reference。  
  https://github.com/Deep-Agent/R1-V

### 8.4 OPD / On-Policy Distillation

- GKD: *Generalized Knowledge Distillation for Auto-regressive Language Models*. 早期 student-generated / on-policy-style distillation 相关工作。  
  https://arxiv.org/abs/2306.13649
- OPCD: *Learn from Yourself: Improving Language Model Pre-Training via On-Policy Context Distillation*. 与 on-policy contexts 上的 distillation 相关。  
  https://arxiv.org/abs/2602.12275
- OPSD / Self-Distilled Reasoner: reasoning 场景中的 on-policy self-distillation。  
  https://arxiv.org/abs/2601.18734

### 8.5 Multimodal OPD / Visual Distillation

这几篇是 novelty 边界风险最大的工作，必须认真比较：

- VOLD: visual on-policy distillation for aligning VLMs with their own visual perceptions.  
  https://arxiv.org/abs/2510.23497
- VA-OPD: visual-advantage on-policy distillation for data-efficient visual reasoning. 这篇尤其接近，需要作为 primary related work。  
  https://arxiv.org/abs/2605.21924
- ViGOS: applies on-policy self-distillation to VLM grounding.  
  https://arxiv.org/abs/2606.19120
- Visual-OPSD: visual on-policy self-distillation for unified multimodal understanding and generation.  
  https://arxiv.org/abs/2606.18974

### 8.6 OPD 机制分析

- *On the Geometry of On-Policy Distillation in LLMs*. 可用于解释 OPD 的 update geometry。  
  https://arxiv.org/abs/2606.07082
- *Dense Supervision, Sparse Updates: A Geometric Perspective on On-Policy Distillation in Large Language Models*. 可用于说明 dense teacher supervision 不一定意味着 dense parameter updates。  
  https://arxiv.org/abs/2606.13657

## 9. 如何和相近工作区分

### 9.1 Versus DyME

DyME 在 prompt/group level 动态选择 SFT 或 GRPO。我们的 OPD 加入第三状态，先判断错误 rollout 是否 recoverable，再决定学习信号。

可用表述：

> Unlike DyME, which maps all failure groups to SFT, our method separates unrecoverable failures from teacher-recoverable mistakes and applies dense on-policy distillation only to the latter.

中文：

> 不同于 DyME 将失败 group 统一映射到 SFT，我们区分不可恢复失败与 teacher-recoverable mistakes，并且只对后者施加 dense on-policy distillation。

### 9.2 Versus CHORD / LUFFY

CHORD/LUFFY 主要通过 weighting 或 off-policy guidance 管理 SFT/RL balance。OPD 则基于 no-gold teacher recoverability 做离散的 per-completion routing。

可用表述：

> Rather than assigning a global or schedule-driven off-policy weight, we use teacher recoverability as a local state variable for each rollout.

中文：

> 我们不是设定全局或 schedule-driven 的 off-policy weight，而是把 teacher recoverability 作为每个 rollout 的局部状态变量。

### 9.3 Versus GKD / Generic OPD

通用 OPD 在 student outputs 上蒸馏 teacher distributions。我们的区别在于 small VLM RLVR 场景，并且用 verifiable reward + no-gold teacher correctness 判断什么时候 distillation 是安全的。

可用表述：

> The key difference is not merely the use of on-policy distillation, but the recoverability gate that integrates OPD into a verifiable-reward training loop for small VLMs.

中文：

> 关键差异不只是使用 on-policy distillation，而是用 recoverability gate 将 OPD 接入小 VLM 的 verifiable-reward training loop。

### 9.4 Versus VA-OPD / VOLD / Visual-OPSD

这些工作已经涉及 visual/on-policy distillation，因此非常接近。我们的区分点应放在：

- focus on small VLM reasoning，而不是 general VLM alignment 或 grounding；
- 与 DyME 的 SFT/GRPO switch 结合；
- no-gold teacher-probe recoverability gate；
- anti-leakage 设计；
- completion 级 GRPO/OPD/SFT 三路路由。

可用表述：

> Existing visual OPD methods mainly ask how to distill visual or multimodal teacher signals on-policy. We ask when such distillation should replace SFT fallback in small-VLM RLVR, and introduce a no-gold recoverability gate to avoid distilling unrecoverable or teacher-failed rollouts.

中文：

> 现有 visual OPD 方法主要关注如何在 on-policy 条件下蒸馏 visual/multimodal teacher signals。我们关注的是：在小 VLM 的 RLVR 中，什么时候 distillation 应该替代 SFT fallback；为此我们引入 no-gold recoverability gate，避免蒸馏不可恢复或 teacher 自身失败的 rollout。

## 10. 建议论文故事线

可以按以下 narrative 展开：

1. 小 VLM 在部署上有价值，但 reasoning post-training 很脆弱。
2. DyME 证明动态 SFT/GRPO switching 比静态 SFT/RL mixing 更适合小 VLM。
3. 但 DyME 对 failure handling 仍然粗粒度：即使强 no-gold visual teacher 可以恢复，wrong rollouts 也可能被直接路由到 SFT。
4. 我们提出 recoverability-aware OPD，为 wrong-but-solvable student trajectories 引入第三种学习模式。
5. 方法保留 DyME 的安全 fallback：
   - correct rollout -> GRPO；
   - wrong but teacher-recoverable rollout -> OPD；
   - unrecoverable/all-wrong case -> SFT。
6. 这样可以在避免 hard CoT over-imitation 的同时，提供 dense、on-policy、anti-leakage supervision。
7. 实验应验证：更快收敛、更高 final accuracy、更低 degeneration、更好的 reasoning trace grounding。

## 11. 后续需要验证的实验与检查

当前阶段不要求做实验，但为了支撑论文 claim，后面至少需要：

1. 对 SFT、DyME、OPD、OPD+VS 做干净的 held-out evaluation。
2. 表格中拆开：
   - pure DyME；
   - SFT；
   - teacher-probe OPD without VS；
   - OPD + teacher trajectory；
   - OPD + visual supervision。
3. Ablate teacher probe：
   - no probe，所有 wrong completions 都 OPD；
   - teacher probe with gold/context leakage；
   - 当前 no-gold probe。
4. Ablate learning signal：
   - SFT fallback only；
   - OPD on student wrong trajectory only；
   - OPD plus teacher trajectory FKL。
5. 报告 routing statistics：
   - SFT replaced ratio；
   - GRPO-on-correct ratio；
   - teacher-probe candidate rate；
   - teacher-probe correct rate；
   - actual OPD call rate。
6. 报告 anti-leakage diagnostics：
   - `privileged_suffix_has_gold_rate`；
   - visual fact empty rate；
   - teacher probe answer accuracy。
7. 和近邻工作比较或至少在 discussion 中解释为什么 direct implementation/comparison 有困难：
   - DyME；
   - CHORD/LUFFY-style SFT/RL mixing；
   - GKD/OPD；
   - VA-OPD/VOLD/Visual-OPSD。

## 12. 当前最推荐的 Contribution Statement

英文版：

> We introduce recoverability-aware on-policy distillation for small VLM reasoning. Instead of treating every failed rollout as an SFT case, our method uses a no-gold visual teacher probe to identify wrong-but-solvable completions and applies token-level OPD only to those trajectories, while preserving GRPO for correct rollouts and SFT for unrecoverable failures.

中文版：

> 我们提出面向小 VLM 推理的可恢复性感知 OPD。不同于把所有失败 rollout 都回退到 SFT，我们用无答案泄漏的视觉 teacher probe 判断错误 completion 是否可恢复；只有 teacher 能在 no-gold 条件下解出时，才对学生的 on-policy 错误轨迹做 token-level 分布蒸馏，同时保留正确 rollout 的 GRPO 和不可恢复失败的 SFT fallback。
