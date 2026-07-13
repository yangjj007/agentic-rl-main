# Reference Integrity Audit

更新时间：2026-07-14

本文件记录论文引用与 arXiv 当前元数据的机械核验结果。它用于阻止早期研究笔记中的
错误题名、错误 arXiv ID 或不存在的近邻工作进入正式稿。

## 审计范围

- `references_seed.bib`：107 个唯一 BibTeX key；
- 其中 105 条包含可解析的 arXiv ID，2 条来自一手官方网页/PMLR；
- 中文初稿当前引用 73 个 key；
- citation key 缺失：0；
- duplicate BibTeX key：0；
- novelty-critical 2026 条目：24。
- author/title/year 完整条目：107/107。

## 发现并修复的严重错误

| 旧条目 | 问题 | 处理 |
|---|---|---|
| `Wang2026PRISM`, `2607.00456` | 该 ID 实际对应 *Multiplicity for partially ordered sets*；未找到所写 PRISM 论文的 arXiv 主记录 | 删除 BibTeX、正文引用和 PRISM-style 实验命名；直接近邻改为有真实证据的 VOLD-style cold-start + online OPD/GRPO |
| `Wang2026ViCuR`, `2607.06563` | 该 ID 实际对应声学机器人论文 | 替换为 `Tian2026ViCuR`, `2606.05718`, *ViCuR: Visual Cues as Recoverable Privilege for Multimodal On-Policy Distillation* |
| `Lee2026DEAR`, `2605.21942` | 该 ID 实际对应 photon blockade 论文 | 删除虚假 disagreement-adaptive 条目 |
| `Zheng2026PrefixOnPolicy`, `2606.22830` | ID 对应真实 OPD 工作，但题名、作者和方法描述均错误 | 替换为 `Xiao2026DEAR`, *Finding the Evidence: Discovering Decision-Supporting Tokens for On-Policy Reasoning Distillation* |
| `Xue2026VisionOPD`, `2605.18740` | ID 正确，但题名、作者和方法描述错误 | 更新为 arXiv 当前题名、作者与 crop-teacher/full-image-student 自蒸馏描述 |

## 当前 2026 近邻核验

修复后，24/24 个 `26xx.xxxxx` ID 均能从 arXiv 主记录返回；24/24 个 BibTeX title 与
当前 arXiv title 规范化后完全一致。包括本文最关键的：

- VLM OPD：VOLD、REOPOLD、Vision-OPD、Visual-Advantage OPD、Decomposed OPD、DOPD、ViCuR；
- OPD reliability：RG-OPD、SCOPE、TrOPD、TIP、AOPD、TA-OPD、PW-OPSD、SFD、DEAR；
- RLVR/process supervision：SSOPD；
- OPD estimator/selection：IW-OPD、control-variate OPD、best-of-N teacher selection；
- privileged-OPD failure：Kaur et al. 的 thinking-model fork-token analysis。

VOLD (`2510.23497`) 也已单独核验：原论文明确联合 cold-start alignment、GRPO 与
OPD，因此足以否定“首次把 OPD 与 VLM RL 结合”的宽泛主张。它不支持此前虚构的
PRISM-style staged pre-alignment 描述。

此外，105 条 arXiv 引用已从主记录核验 author/title/year/eprint/archivePrefix；
LLaVA-NeXT 使用 LLaVA 官方技术博客提供的作者与发布日期，Born Again Neural Networks
使用 PMLR 页面提供的正式作者、年份和会议信息。刷新保留原 citation key 与研究备注。

## Sub-1B OPD Novelty Audit

截至 2026-07-14，再次使用 arXiv 主记录检索以下组合：

- `on-policy distillation` + `vision-language` / `multimodal` / `reasoning`；
- `on-policy self-distillation` + `VLM`；
- `0.5B` / `sub-billion` / `Qwen2.5-VL-0.5B` + `on-policy distillation`。

检索没有发现一篇直接以 sub-1B VLM 为 student、并在 verifiable reasoning/RLVR
设置中系统比较 OPD 与 matched no-OPD 的论文。已核验的最接近 VLM OPD 工作包括
VOLD、REOPOLD、Vision-OPD、VA-OPD、Decomposed-OPD、DOPD 与 ViCuR；它们足以否定“首个
VLM OPD”“首个 OPD+VLM RL”或“首个 recoverable multimodal privilege”等宽泛
表述，但当前未直接覆盖本文的 0.5B student 问题。

这是一个**检索边界结论**，不是效果证据，也不是排他性证明。正式稿只允许使用：

> To our knowledge, this is the first systematic study of OPD for sub-billion
> VLM reasoning under verifiable rewards.

该句还必须同时满足 matched OPD-vs-no-OPD、OPD/GRPO/fallback 互补性和 sub-1B
failure analysis 三项实验门槛；任一门槛缺失时，摘要与结论只能写成研究目标。

### 2026-07-14 late-search threat check

在当前实验等待 GPU gate 期间，又以 arXiv 主记录检索了 `OPD + RLVR + VLM`、
`sub-billion/0.5B + OPD` 和 `multimodal on-policy distillation`。本轮最直接的威胁项为：

- VOLD (`2510.23497`) 已明确联合 GRPO 与 OPD，并报告 cold-start alignment 对在线
  reasoning transfer 的必要性。因此“首次在 VLM 中联合 OPD 与 RLVR”已经被否定。
- REOPOLD (`2603.11137`) 已明确在 3B/7B student 上研究 visual reasoning OPD 和
  teacher 选择。因此“首次把 OPD 引入视觉推理”也已被直接否定；本文只能保留
  sub-1B、all-wrong RLVR state 和 matched signal attribution 的受限定位。
- ViCuR (`2606.05718`) 已提出 multimodal recoverable privilege，并在 Qwen3-VL 2B/8B
  student 上比较 answer-side privilege、visual cue privilege 与 stronger-teacher OPD。
  因此“首次提出 multimodal recoverability”已经被否定；本文只能研究不同的
  completion-level answer-verifier route 及 sub-1B failure regime。
- Vision-OPD (`2605.18740`) 已在 student-generated rollouts 上把 crop-conditioned teacher
  蒸馏到 full-image student。因此“首次视觉 OPD/视觉自蒸馏”已经被否定。
- SSOPD (`2605.17497`) 已把 mixed-group correct/wrong contrast 转为 RLVR 中的稠密
  process supervision，但其核心实验是 reasoning LLM，而非 sub-1B VLM。它仍是本文
  必须匹配的 process-supervision 近邻，而不是可忽略的背景引用。
- Self-Distilled RLVR (`2604.03128`) 已系统讨论 privileged-teacher distillation 与 RLVR
  的互补和泄漏风险。因此“首次证明 OPD 与 RLVR 互补”也不能脱离 sub-1B VLM 的
  matched evidence 单独主张。
- RG-OPD (`2607.04037`) 的 arXiv 主记录于 2026-07-07 提交，并明确用 verifier
  feedback 估计 teacher 相对 student 的可靠性后再门控 distillation。因此“首个
  verifier/reward-gated OPD”也被否定；它当前覆盖 reasoning/coding LLM，而不是本文的
  0.5B multimodal all-wrong route，故应作为 unconditional-vs-gated OPD 的直接方法近邻。
- Kaur et al. (`2607.05184`) 进一步指出 privileged teacher/student 在 reasoning fork
  token 上可能发生策略不一致，naive privileged OPD 甚至会损害 thinking model。这一
  结果支持本文隔离 hard trajectory 并记录 token/style mismatch，也要求将 teacher-state
  mismatch 列为低分 run 的首要 forensic，而不是默认增加 OPD 权重。

本轮没有发现直接以 sub-1B VLM student、verifiable multimodal reasoning、matched
no-OPD 和 route-specific complementarity 为同一研究对象的论文。可保留的 novelty
不是 OPD、routing 或 recoverability 任一组件本身，而是对这一受限 regime 的首个系统
因果研究；该定位仍以 P0-E1/P0-E3 和 signal-contribution 消融完成为生效条件。

## 写作约束

1. 未通过 ID、题名和摘要三项核验的 2026 工作不得进入 claim matrix。
2. 搜索结果或研究笔记不能单独作为引用事实源；正式条目必须对应 arXiv/API 或论文正文。
3. `to our knowledge` 主张仍必须由 matched experiments 支撑，文献中暂未发现不等于
   方法贡献已成立。
4. 每次增加 novelty-critical 引用后，重新运行 citation-key audit、duplicate-key audit
   与 2026 arXiv title audit。
