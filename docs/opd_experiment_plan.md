# OPD 低成本实验规划

日期：2026-06-23  
关联定位笔记：`docs/opd_research_positioning.md`  
Motivation 日志结果：`docs/opd_motivation_log_results.md`

这份文档只保留实验思路、motivation 证据链、创新点验证逻辑和最小执行顺序。具体日志提取数值和 Step 1 结果记录放到 `docs/opd_motivation_log_results.md`，避免规划文件变成流水账。

## 0. 核心研究问题

小 VLM 的失败 rollout 是否应该被一概 SFT 回退？

我们的主张是：在可验证任务中，错误 completion 并不都等价。有些错误来自完全不可恢复的状态，应回退 SFT；有些错误发生在“学生已有正确 rollout、但部分 completion 失败”的局部可解区域，可以由 no-gold teacher probe 判断是否 recoverable，并对 teacher-correct wrong completion 施加 OPD token-level 分布监督。

对应方法定位：

> Recoverability-aware OPD = correct rollout 走 GRPO；all-wrong 或 teacher-wrong 走 SFT；wrong 但 no-gold teacher 可修复的 completion 走 OPD。

## 1. 论文需要支撑的三条证据链

### 1.1 Motivation：小 VLM 在 DyME/GRPO 类训练中容易退化

要证明的问题：

- DyME clean fast run 中直接训练信号较早消失：accuracy reward 与 GRPO route 长期接近 0；
- 训练长期处于 SFT-dominated 状态：SFT route ratio 接近 1；
- OPD 能在相同 fast budget 后段恢复 answer/format reward，并让更多 completion 进入 GRPO route。

最重要图表：

- Figure B：training health curves。
- 建议使用直接训练/算法指标：`rewards/accuracy/mean`、`rewards/format/mean`、`reward`、`loss`、`routing/sft_replaced_ratio`、`routing/grpo_on_correct_rate`、`routing/opd_teacher_call_rate`。
- 横轴按每 10 training steps 分箱取均值，不依赖最后 20 step 摘要。

结果记录位置：`docs/opd_motivation_log_results.md`。

### 1.2 方法创新：OPD 是 recoverability-aware 第三学习状态

要证明的问题：

- OPD 不是“给 DyME 加一个 teacher”；
- 真实训练中存在三路 routing：SFT / GRPO / OPD；
- OPD call 只发生在 teacher probe 判断可恢复的错误 completion 上。

最重要图表：

- Figure C：routing stacked area plot。
- Figure D：method flow diagram。

推荐展示：

```text
student rollouts
  -> correct completion -> GRPO
  -> wrong completion + no-gold teacher correct -> OPD
  -> all-wrong / teacher wrong -> SFT
```

### 1.3 边界与可信度：收益不能来自 gold leakage 或 Visual Supervision 混淆

要证明的问题：

- 主方法 teacher context 不包含 gold answer；
- Visual Supervision 版本必须单独标注，不能混入 clean main claim；
- leaky/gold diagnostic 只能作为反例或上界诊断，不作为正式 baseline。

需要检查的混淆字段：

- `teacher/privileged_suffix_has_gold_rate`
- `visual/ic_ok_rate`
- `routing/teacher_probe_candidate_rate`
- `routing/teacher_probe_correct_rate`
- `routing/teacher_probe_wrong_rate`

## 2. 实验预算原则

### Tier 0：零训练日志分析

优先级最高。复用已有 `outputs/test-fast/logs/` 和 `resolved_config.json`，几乎没有计算成本。

用途：

- 形成 motivation 曲线；
- 检查 DyME collapse 与 OPD 稳定性；
- 统计 route ratio、teacher probe candidate/correct/wrong；
- 排查 gold leakage 和 Visual Supervision 混淆。

### Tier 1：短 run / 500-step ablation

只用于关键 ablation，不做大规模搜索。

优先用于：

- no-gold probe vs no-probe；
- OPD only vs OPD + teacher trajectory；
- OPD without VS vs OPD + VS。

### Tier 2：4 epoch fast baseline

用于 main table 和核心图。保持 `scripts/test/` 的训练规模，不主动扩大。

## 3. 实验矩阵

| ID | 目的 | 方法/对照 | 训练规模 | 优先级 | 论文产物 |
| --- | --- | --- | --- | --- | --- |
| E0 | 主结果对比 | Base / SFT / DyME / OPD | 已有 checkpoint 或 4 epoch fast | P0 | main table + bar chart |
| E1 | motivation：退化与稳定性 | DyME vs OPD training health | 复用日志 | P0 | Figure B |
| E2 | 三路路由机制 | OPD route statistics | 复用日志 | P0 | Figure C/D |
| E3 | teacher probe 必要性 | no-gold probe vs no-probe | 500 steps 起 | P1 | Figure E |
| E4 | loss 组件拆分 | OPD only vs OPD + teacher trajectory | 500 steps 起 | P1 | ablation table |
| E5 | VS 混淆控制 | OPD without VS vs OPD + VS | 复用/短 run | P1 | controlled table |
| E6 | SFT 平台期/模板化 | SFT 1/2/4 epoch | 复用或短 run | P2 | epoch curve |
| E7 | 质性样例 | SFT/DyME/OPD outputs | eval outputs | P2 | qualitative figure |

P0 是最小论文证据链。P1 用于增强 novelty 与排除替代解释。P2 用于 introduction/motivation 补充。

## 4. E0：主结果对比

核心问题：在同等 fast training budget 下，OPD 是否比 SFT 和 DyME 有更高 ChartQA held-out performance？

推荐命令：

```bash
CHECKPOINT_DIR=outputs/test-fast/sft/final_checkpoint EXPERIMENT=sft \
  bash scripts/run_eval_ablation.sh

CHECKPOINT_DIR=outputs/test-fast/dyme/final_checkpoint EXPERIMENT=dyme \
  bash scripts/run_eval_ablation.sh

CHECKPOINT_DIR=outputs/test-fast/opd-7b-ds/final_checkpoint EXPERIMENT=opd \
  bash scripts/run_eval_ablation.sh
```

图表建议：

- Figure A：Base / SFT / DyME / OPD bar chart。
- Table A：checkpoint、epoch、Rel-Corr、训练时间、teacher-probe 额外开销。

当前备注：主结果用户已有，本轮不重复 eval。

## 5. E1：Motivation 训练健康曲线

核心问题：DyME 是否出现 collapse，OPD 是否改善训练健康？

解析命令：

```bash
python3 scripts/analyze_opd_routes.py \
  --compare DyME=outputs/test-fast/logs/train_test_dyme_20260621_112902.log \
            OPD=outputs/test-fast/logs/train_test_opd_20260621_212323.log \
  --step-interval 10 \
  --csv-out docs/figures/opd_motivation_direct_metrics_10step.csv \
  --plot-out docs/figures/opd_motivation_direct_metrics_10step.png
```

推荐图：

- task/format/total reward vs step；
- SFT / GRPO / OPD route ratio vs step；
- training loss vs step。

结果记录位置：`docs/opd_motivation_log_results.md`。

## 6. E2：Recoverability-Aware 三路路由

核心问题：OPD 是否真实形成 GRPO / OPD / SFT 三种学习状态？

关键字段：

- `routing/sft_replaced_ratio`
- `routing/grpo_on_correct_rate`
- `routing/opd_teacher_call_rate`
- `routing/teacher_probe_candidate_rate`
- `routing/teacher_probe_correct_rate`
- `routing/teacher_probe_wrong_rate`

推荐图：

- stacked area：SFT / GRPO / OPD；
- 折线：teacher-probe candidate/correct；
- method flow diagram：解释 route rule。

## 7. E3：No-Gold Teacher Probe Ablation

核心问题：no-gold teacher probe 是否必要？如果关闭 probe，让 wrong completions 直接走 OPD，是否会带来错误蒸馏或不稳定？

最小对照：

1. **no-gold probe OPD**：当前方法。
2. **no-probe OPD**：关闭 teacher probe。
3. **leaky/gold diagnostic**：仅作 leakage diagnostic，不进入主方法 claim。

已准备脚本：

```bash
# 不启动训练，只检查配置、已有日志字段和 dry-run 命令
bash scripts/test/smoke_opd_probe_ablation.sh

# 默认 dry-run，不启动训练
bash scripts/test/run_opd_probe_ablation.sh --dry-run --max-steps 500

# GPU 空闲后再跑
bash scripts/test/run_opd_probe_ablation.sh --run --max-steps 500
```

脚本设计原则：

- 关闭 Visual Supervision；
- 关闭 teacher trajectory；
- 关闭 DePlot；
- 保持 no-gold teacher context；
- 日志必须包含 health、routing、teacher-probe 和 `loss/opsd` 字段。

推荐图：

- Figure E 左图：no-gold probe vs no-probe 的 final 10-step-bin accuracy reward 或 held-out Rel-Corr；
- Figure E 右图：SFT / GRPO / OPD route ratio 与 OPD call rate；
- caption 强调 no-gold probe 是 recoverability gate，而不是 gold answer leakage。

## 8. E4/E5：组件与混淆控制

### E4：OPD loss vs teacher trajectory

命令：

```bash
DYME_TRAIN_MAX_STEPS=500 DYME_TEACHER_TRAJECTORY=0 bash scripts/train_opd_7b_dyme_probe.sh
DYME_TRAIN_MAX_STEPS=500 DYME_TEACHER_TRAJECTORY=1 bash scripts/train_opd_7b_dyme_probe.sh
```

目的：区分收益来自 student wrong trajectory 上的 OPD，还是 teacher-generated trajectory FKL。

### E5：Visual Supervision control

必须单独报告：

- OPD without VS：clean main；
- OPD + VS：diagnostic/control，不混入 clean claim。

表格列必须包含：

- Visual Supervision 是否开启；
- teacher trajectory 是否开启；
- teacher context 是否包含 gold；
- `teacher/privileged_suffix_has_gold_rate`。

### E5b：4epoch DePlot OPD 消融

目标：在固定 4epoch fast budget 下，分离 DePlot evidence、Visual Supervision 和 OPD loss 形式的影响。

统一脚本：

```bash
# 默认 dry-run，不启动训练
bash scripts/test/run_opd_deplot_ablation.sh --dry-run

# 至少 2 step 的冒烟测试，覆盖第 2 个 training step 后的日志字段
bash scripts/test/run_opd_deplot_ablation.sh --smoke --smoke-steps 2 --run-id smoke_check

# GPU 空闲后顺序运行三个 4epoch 变体
bash scripts/test/run_opd_deplot_ablation.sh --run --run-id deplot_4epoch_main
```

三个变体：

| variant | DePlot evidence | Visual Supervision | OPD token loss | 说明 |
| --- | --- | --- | --- | --- |
| `deplot_no_vs_opd` | 开 | 关 | `jsd` | 与已有 no-DePlot 4epoch 权重对照，观察只加 DePlot evidence 的影响 |
| `deplot_vs_opd` | 开 | 开 | `jsd` | 单独观察 Visual Supervision 带来的增益或干扰 |
| `deplot_vs_srkl` | 开 | 开 | `srkl` | 在相同 VS/DePlot 设置下比较普通 OPD loss 与 SRKL |

注意：这里“普通 OPD”指 token-level divergence 使用 `jsd`；GRPO 主训练仍由 `DYME_GRPO_WEIGHT=1.0` 控制，不能把 `DYME_OPSD_LOSS_TYPE` 写成 `grpo`。

关键配置必须固定：

- `DYME_NUM_TRAIN_EPOCHS=4`
- `DYME_OPSD_PROVIDERS=format_only,visual_facts_deplot`
- `DYME_TEACHER_PROBE_PROVIDERS=format_only,visual_facts_deplot`
- `DYME_TEACHER_PROBE_MAX_NEW_TOKENS=96`
- `DYME_TEACHER_TRAJ_MAX_NEW_TOKENS=128`
- `DYME_DEPLOT_ENABLED=0`，只使用已写入数据集的 DePlot 结果，不在训练时重新生成；
- `DYME_OPSD_HANG_DEBUG=0`、`DYME_OPSD_HANG_FORCE=0`。

日志必须包含：

- 启动配置：`[DyME-RUN-CONFIG]`
- 数据状态：`[DyME-DATA]`，需要确认 `deplot_real_rate=1.0` 且 placeholder/missing 为 0；
- `routing/teacher_probe_skipped_no_evidence_rate`
- `routing/teacher_probe_deplot_real_rate`
- `routing/teacher_probe_visual_fact_used_rate`
- `teacher_probe/generated_tokens_mean`
- `teacher_probe/generated_tokens_p95`
- `teacher_probe/clipped_rate`
- `loss/opsd`

## 9. 建议的最小执行顺序

### Step 1：零训练日志图

目标：快速形成 motivation 证据。

输入：

- DyME clean fast log；
- OPD no-gold clean fast log。

产物：

- Figure B：training health curves；
- Figure C：routing stacked area plot；
- `docs/opd_motivation_log_results.md` 中的日志结果表。

### Step 2：确认 4 epoch main baselines

目标：得到主结果表。

当前状态：用户已有主结果，本轮不重复训练或 eval。

产物：

- Figure A；
- Table A。

### Step 3：teacher-probe 最小消融

目标：证明 recoverability gate 必要。

先做：

```bash
bash scripts/test/smoke_opd_probe_ablation.sh
bash scripts/test/run_opd_probe_ablation.sh --dry-run --max-steps 500
```

GPU 空闲后再做：

```bash
bash scripts/test/run_opd_probe_ablation.sh --run --max-steps 500
```

产物：

- Figure E；
- 一张简洁 ablation table。

### Step 3b：4epoch DePlot OPD 控制消融

目标：在固定 4epoch budget 下，检查 DePlot evidence、Visual Supervision 和 `srkl` loss 的影响。

先做：

```bash
bash scripts/test/run_opd_deplot_ablation.sh --dry-run
bash scripts/test/run_opd_deplot_ablation.sh --smoke --smoke-steps 2 --run-id smoke_check
```

GPU 空闲后再做：

```bash
bash scripts/test/run_opd_deplot_ablation.sh --run --run-id deplot_4epoch_main
```

产物：

- Table B：三变体控制消融；
- 简洁折线图：每 10 step 的 `reward`、`rewards/accuracy/mean`、`routing/grpo_on_correct_rate`、`routing/opd_teacher_call_rate`；
- 一张日志诊断小表：DePlot real rate、teacher probe skipped no evidence rate、visual fact used rate。

### Step 4：质性样例

目标：补充 2-3 个直观样例。

展示：

- SFT 模板化但视觉值错误；
- DyME 退化或截断；
- OPD 输出短、格式稳定、答案正确。

## 10. 推荐论文图表

| 图表 | 放置位置 | 目的 | 数据来源 | 必要性 |
| --- | --- | --- | --- | --- |
| Figure A：主结果 bar chart | Experiments | final score | held-out eval | 必须 |
| Figure B：direct training curves | Introduction / Motivation | DyME direct training signal disappears; OPD recovers reward/GRPO signal | training logs | 必须 |
| Figure C：routing stacked area | Method / Analysis | 三路路由真实发生 | OPD logs | 必须 |
| Figure D：route flow diagram | Method | 解释算法 | 手工图 | 必须 |
| Figure E：teacher probe ablation | Ablation | recoverability gate 必要性 | 500-step ablation | 建议 |
| Table A：主结果表 | Experiments | 数字结果 | eval | 必须 |
| Table B：component/control ablation | Ablation | 排除替代解释 | short run/logs | 建议 |
| Figure G：qualitative examples | Analysis | 输出形态改善 | eval predictions | 可选 |

## 11. 写作可用结论模板

如果结果稳定，可以写成：

> Under the same 4-epoch fast-training budget, clean DyME quickly enters an SFT-dominated regime: accuracy reward and GRPO routing remain near zero after the early stage. In contrast, clean no-gold OPD shows a late-stage recovery in accuracy reward, format reward, and GRPO routing, while SFT fallback decreases. Routing statistics further show that OPD is not a uniform teacher-distillation add-on: unrecoverable failures still fall back to SFT, correct rollouts use GRPO, and only teacher-recoverable wrong completions activate OPD.

中文版本：

> 在相同 4 epoch fast-training budget 下，clean DyME 较早进入 SFT-dominated 训练状态：accuracy reward 和 GRPO route 在早期之后长期接近 0。相比之下，clean no-gold OPD 在后段恢复 accuracy reward、format reward 和 GRPO route，同时 SFT fallback 下降。路由统计进一步说明，OPD 不是简单添加 teacher distillation：不可恢复失败仍回退 SFT，正确 rollout 仍使用 GRPO，只有 teacher 可恢复的错误 completion 才触发 OPD。
