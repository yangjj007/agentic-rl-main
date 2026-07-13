# Closed-Loop Recoverability Curriculum Paper Reconstruction Plan

> **Historical execution plan (superseded 2026-07-13):** 已完成的文档重构产物继续
> 保留，但最终论文定位已经收敛到 OPD 核心创新。`no-gold` 仅是 legacy label；规范
> 表述为 gold-hidden teacher / verifier-available routing。当前事实源见
> `docs/paper_reconstruction/experiment_ledger.md` 与 `claim_evidence_matrix.md`。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 OPD 中文初稿与实验文档重构为一套围绕 Closed-Loop Recoverability Curriculum (CLRC) 的、论点与证据严格对应的 AAAI 方法论文材料。

**Architecture:** 文档采用“正文、证据矩阵、实验账本、实验计划、图表计划、事实结果”六层分工。正文只引用 claim-evidence matrix 中已验证的结果；experiment ledger 作为所有 run 的统一事实源；oracle/privileged 与 no-gold 主方法严格分栏。

**Tech Stack:** Markdown、BibTeX、现有训练/eval 日志、`rg`、Python `ast`/CSV 解析脚本。

---

### Task 1: 建立 Claim-Evidence Matrix

**Files:**
- Create: `docs/paper_reconstruction/claim_evidence_matrix.md`
- Read: `docs/superpowers/specs/2026-07-12-closed-loop-recoverability-curriculum-paper-design.md`
- Read: `docs/opd_main_training_eval_results.md`

- [ ] **Step 1: 创建 claim 状态规范**

在文档开头定义四种状态：

```markdown
- `verified`：完整训练与可复现 eval artifact 已直接支持。
- `partial`：实现或训练日志支持机制存在，但尚无完整主结果/消融。
- `running`：实验正在运行，不得在摘要中引用结果。
- `missing`：尚无直接证据，必须安排实验。
```

- [ ] **Step 2: 写入方法层 claims**

至少记录以下条目及其证据路径：

```markdown
| Claim | Status | Evidence | Required next evidence |
|---|---|---|---|
| 跨 rank 最终 route 归约形成一致全局快照 | verified | `tests/test_global_training_signal*.py`; 8-GPU smoke log | none |
| realized GRPO route 单信号控制器滞后一拍更新 | verified | controller/trainer tests; smoke metrics | none |
| 控制器不依赖固定 epoch/step boundary | verified | controller API/tests | 跨 epoch/batch robustness eval |
| no-gold multimodal recoverability 是主方法 | partial | no-gold probe code path exists | gold-rate=0.0 的 4epoch main run |
```

- [ ] **Step 3: 写入效果层 claims**

明确记录：

```markdown
| student_hint_short 达到 0.5800 | verified |
| oracle official 达到 0.5872 | verified |
| CLRC 超过 0.60 | running/missing，直到 summary.csv 证明 |
| CLRC 优于 DyME | running/missing，直到统一预算主表证明 |
| CLRC 降低 teacher compute | missing，直到 calls/tokens 对照完成 |
```

- [ ] **Step 4: 校验无越权 claim**

Run:

```bash
rg -n "0\.60|优于|超过|显著|no-gold|无泄漏" docs/paper_reconstruction/claim_evidence_matrix.md
```

Expected: 每条强主张同一行或相邻行包含状态与证据要求。

### Task 2: 建立统一 Experiment Ledger

**Files:**
- Create: `docs/paper_reconstruction/experiment_ledger.md`
- Read: `outputs/test-fast/pcd-no-visual/**/eval_chartqa/summary.csv`
- Read: `outputs/test-fast/long-runs/**/status`

- [ ] **Step 1: 定义账本列**

使用统一表头：

```markdown
| Run | Method role | Epoch | Student | Teacher | Evidence | Gold access | Controller | Status | Final acc | Processed | Artifact |
```

- [ ] **Step 2: 登记已验证基线**

至少写入：

```markdown
| no-gold PCD aligned | baseline | 4 | 0.5B | 7B | DePlot | no | none | complete | 0.5420 | 2500 | existing summary |
| oracle route_guard | oracle baseline | 4 | 0.5B | 7B | oracle | yes | none | complete | 0.5592 | 2500 | existing summary |
| student_hint_short | strongest completed internal | 4 | 0.5B | 7B | oracle | yes | fixed schedule | complete | 0.5800 | 2500 | summary.csv |
| oracle official | upper baseline | 4 | 0.5B | 7B | oracle | yes | official | complete | 0.5872 | 2500 | existing artifact |
```

- [ ] **Step 3: 登记当前 CLRC run**

写入 run：

```text
global_grpo_route_full_4epoch_20260712_205549
```

标记 `Gold access=yes`、`Method role=oracle CLRC upper bound`、`Status=running`，不得预填 final accuracy。

- [ ] **Step 4: 增加 no-gold 主实验槽位**

加入状态为 `planned` 的：

```text
no_gold_clrc_full_4epoch
no_gold_recoverability_without_controller_4epoch
no_gold_fixed_schedule_4epoch
```

### Task 3: 重构中文论文正文

**Files:**
- Modify: `docs/paper_reconstruction/chinese_draft.md`
- Read: `docs/paper_reconstruction/claim_evidence_matrix.md`
- Read: `docs/paper_reconstruction/experiment_ledger.md`

- [ ] **Step 1: 重写标题、摘要和贡献**

标题采用：

```markdown
# Closed-Loop Recoverability Curriculum for Small Vision-Language Reasoning
```

中文工作标题采用：

```text
面向小型视觉语言推理的闭环可恢复性课程学习
```

摘要必须包含两个失配：错误 completion 异质性与固定课程失配；不得在当前 eval 完成前写入 CLRC 最终 accuracy。

贡献重写为：

1. no-gold multimodal recoverability learning-state estimator；
2. completion-level GRPO/OPD/fallback 三路监督；
3. realized global GRPO coverage 驱动的连续闭环控制器；
4. 训练动力学与效率评估协议。

- [ ] **Step 2: 重写引言的问题链**

按以下顺序组织：

```text
小 VLM 推理需求 -> SFT/RLVR 各自限制 -> 错误状态不均质 -> 固定 step curriculum 语义脆弱 -> CLRC 双时间尺度解决方案 -> 贡献
```

- [ ] **Step 3: 重写方法章节**

必须包含与实现一致的公式：

```text
a_t = N_grpo / N_total
z_t = alpha a_t + (1-alpha) z_(t-1)
m_t = max(m_(t-1), z_t)
s_t = 1 - smoothstep(clip(m_t/tau, 0, 1))
```

并明确 step `t` 快照控制 step `t+1`。

- [ ] **Step 4: 分离 no-gold 主方法与 oracle upper bound**

正文实验设置必须显式写：

```markdown
- No-gold CLRC：`teacher/privileged_suffix_has_gold_rate=0.0`，作为主方法。
- Oracle CLRC：`teacher/privileged_suffix_has_gold_rate=1.0`，只作为分析上界。
```

- [ ] **Step 5: 重写实验与局限章节**

实验章节引用 experiment ledger，不手工复制未核验数字。局限至少讨论：teacher quality、evidence dependence、单调 mastery 可能不适应 catastrophic regression、ChartQA 单任务泛化限制。

- [ ] **Step 6: 扫描旧叙事残留**

Run:

```bash
rg -n "after_step|294|固定步数|OPD =|只.*gate|已经超过 0\.60" docs/paper_reconstruction/chinese_draft.md
```

Expected: 不存在把固定 step schedule 写成主方法或无证据性能结论的段落。

### Task 4: 更新 Related Work 与参考文献

**Files:**
- Modify: `docs/paper_reconstruction/references_seed.bib`
- Modify: `docs/paper_reconstruction/related_paper_figure_review.md`
- Modify: `docs/paper_reconstruction/chinese_draft.md`

- [ ] **Step 1: 核验最邻近工作元数据**

使用论文主页/arXiv 原始页面核验并加入：

```text
DyME
CHORD
RG-OPD
GKD
Visual-RFT
R1-VL / VLM-R1
Position-aware or token-reliability on-policy distillation work
```

- [ ] **Step 2: 写出差异表**

在 related review 中加入列：

```markdown
| Method | Granularity | Teacher trust signal | Multimodal evidence | Routes | Global feedback | Teacher budget control |
```

- [ ] **Step 3: 在正文中避免 novelty 过度声明**

正文只主张组合后的双时间尺度闭环是贡献，不把 dynamic weighting、reward gating 或 OPD 单独称为首次提出。

- [ ] **Step 4: BibTeX 完整性检查**

Run:

```bash
rg -n '^@' docs/paper_reconstruction/references_seed.bib
rg -n 'TBD|TODO|unknown|xxxx' docs/paper_reconstruction/references_seed.bib
```

Expected: 所有正文核心比较方法均有条目，无占位元数据。

### Task 5: 重构实验矩阵和图表计划

**Files:**
- Modify: `docs/opd_experiment_plan.md`
- Modify: `docs/paper_reconstruction/experiment_plot_plan.md`

- [ ] **Step 1: 将实验分为 P0/P1/P2**

P0 必须包含：

```text
E0 no-gold 主结果
E1 recoverability routing 消融
E2 controller state 消融
E3 oracle upper bound 与 leakage disclosure
```

P1 包含 action split、compute efficiency、epoch/batch robustness；P2 包含跨任务泛化和质性分析。

- [ ] **Step 2: 为每个实验写唯一变化量**

每行包含：base config、唯一改变项、4epoch budget、eval checkpoint、健康门槛、论文 claim。

- [ ] **Step 3: 更新主图定义**

图表计划至少包含：

```text
Figure 1: local routing + global feedback method diagram
Figure 2: main accuracy/teacher-compute Pareto
Figure 3: route/controller dynamics
Figure 4: fixed schedule vs CLRC under changed epoch/batch
Table 1: unified main results with gold-access disclosure
Table 2: routing/controller/action ablations
```

- [ ] **Step 4: 设置论文硬门槛**

实验计划明确：若 no-gold full CLRC 未优于统一预算 DyME，则不能把 no-gold effectiveness 写进摘要；若 oracle CLRC 未超过 `0.5872`，先分析控制器而非扩大 claim。

### Task 6: 同步事实结果与当前训练进程

**Files:**
- Modify: `docs/opd_main_training_eval_results.md`
- Modify: `docs/paper_reconstruction/experiment_ledger.md`
- Modify: `docs/paper_reconstruction/claim_evidence_matrix.md`

- [ ] **Step 1: 写入当前 controller 实现验证**

记录 66-test focused suite、4-step 8-GPU smoke 和 global/local signal disagreement 诊断，附 artifact 路径。

- [ ] **Step 2: 写入当前 full run 状态**

记录 run id、variant、592 total steps、tmux、monitor、auto-eval pipeline，以及当前状态；不将中间 reward 解释为 final accuracy。

- [ ] **Step 3: 训练完成后原子更新结果**

只从以下 artifact 读取：

```text
eval_chartqa/summary.csv
eval_final_checkpoint_*.log
training health window summary
teacher candidate funnel
```

同步更新 ledger、claim matrix 和正文结果表。

- [ ] **Step 4: 一致性审计**

Run:

```bash
rg -n "0\.5800|0\.5872|0\.60|global_grpo_route_full_4epoch_20260712_205549" \
  docs/paper_reconstruction docs/opd_main_training_eval_results.md docs/opd_experiment_plan.md
```

Expected: 相同 run/结果在所有文档中的角色、gold access、状态和数字一致。

### Task 7: 最终文档验收

**Files:**
- Verify: `docs/paper_reconstruction/*.md`
- Verify: `docs/opd_experiment_plan.md`
- Verify: `docs/opd_main_training_eval_results.md`

- [ ] **Step 1: 检查链接和文件存在性**

Run:

```bash
rg -o '`[^`]+\.(csv|log|png|md)`' docs/paper_reconstruction docs/opd_*.md
```

人工核对所有作为证据的本地路径存在。

- [ ] **Step 2: 检查 claim 状态**

Run:

```bash
rg -n "verified|partial|running|missing" docs/paper_reconstruction/claim_evidence_matrix.md
```

Expected: 每个摘要/贡献强 claim 都可映射到一条 matrix 记录。

- [ ] **Step 3: 检查泄漏披露**

Run:

```bash
rg -n "gold access|Gold access|privileged_suffix_has_gold_rate|oracle upper" \
  docs/paper_reconstruction docs/opd_experiment_plan.md
```

Expected: 正文、账本、主表设计和实验计划均显式区分 no-gold 与 oracle。

- [ ] **Step 4: 输出重构摘要**

汇总修改文件、当前已验证 claims、仍缺失实验和下一条优先训练命令，不宣布未完成的 `>0.60` 目标成功。
