# 实验与绘图设计计划

更新时间：2026-07-01

本文档按新的论文定位更新：`bash scripts/test/run_pcd_no_visual_10epoch.sh` 作为主效果实验。它对应论文中的 **Ours / Recoverability-guided OPD**，而不是附属小消融。

核心论文论点：

> 小模型在可验证视觉推理训练中同时面临 off-policy imitation 的模板化风险和 sparse GRPO 的低信号风险。Recoverability-guided OPD 用无答案泄漏 teacher 判断错误轨迹是否可恢复，并把可恢复失败转化为 dense on-policy supervision，从而提升训练稳定性、有效更新比例和最终性能。

## 0. 主实验定位

主实验命令：

```bash
DYME_PCD_RUN_ID=<run_id> bash scripts/test/run_pcd_no_visual_10epoch.sh
```

该脚本实际锚定的设置：

| 项 | 设置 | 论文解释 |
| --- | --- | --- |
| Variant | `deplot_no_vs_opd_pcd` | Ours / Recoverability-guided OPD |
| Training budget | 10 epochs | 主效果预算 |
| Visual supervision | off | 排除额外视觉监督混淆 |
| Evidence | DePlot textual evidence on | teacher 使用 no-gold visual evidence |
| OPD loss | JSD | dense token-level distribution guidance |
| All-wrong rescue | on from step 0 | 低方差/全错 group 也进入 teacher recoverability probe |
| Teacher trajectory | on | teacher-correct 样本进入 OPD/trajectory guidance |
| Variance-adaptive weight | off | 主方法先只验证 PCD routing；VA 放入机制/扩展消融 |

推荐论文命名：

- `Ours`：`deplot_no_vs_opd_pcd`。
- `Ours w/o PCD rescue`：`deplot_no_vs_opd`。
- `Ours + variance-adaptive weight`：`deplot_no_vs_opd_va_pcd`，只作为机制增强，不作为默认主方法。

不要在主表显眼位置使用实现名；实现名只保留在 appendix 或实验复现说明中。

## 1. 主实验能直接支撑哪些图表

一次 10epoch PCD no-visual 主实验可以提供以下证据，但不能单独完成整篇论文的所有比较。

| 图表 | 从主实验提取什么 | 还需要什么对照 |
| --- | --- | --- |
| Table 1: Main results | final checkpoint ChartQA accuracy、format-valid rate、平均长度、训练耗时 | Base、SFT、GRPO/DyME、`deplot_no_vs_opd` |
| Figure 1: Motivation | low reward std、all-wrong group、teacher-recoverable wrong samples | 至少 GRPO/DyME 或 no-PCD anchor |
| Figure 3: Training dynamics | reward、reward std、OPD coverage、fallback/teacher route、输出长度曲线 | SFT/GRPO/no-PCD 的同预算曲线 |
| Figure 4: Useful-signal funnel | generated -> wrong -> all-wrong/low-variance -> teacher-recoverable -> OPD/fallback | no-PCD anchor 用于显示 all-wrong rescue 的增量 |
| Table 3: Cost and reliability | teacher call rate、candidate parse rate、gold suffix rate、DePlot evidence coverage | teacher/evidence sanity controls |
| Qualitative figure | student wrong、teacher recovery、after-training correction | 至少挑 3-5 个成功与失败案例 |

主实验关键产物路径：

```text
outputs/test-fast/pcd-no-visual/<run_id>/deplot_no_vs_opd_pcd/
outputs/test-fast/pcd-no-visual/<run_id>/deplot_no_vs_opd_pcd/final_checkpoint
outputs/test-fast/pcd-no-visual/<run_id>/deplot_no_vs_opd_pcd/teacher_probe_candidates/rank*.jsonl
outputs/test-fast/logs/pcd_no_visual_<run_id>/deplot_no_vs_opd_pcd/
```

final checkpoint eval 可以先用单模型命令：

```bash
CHECKPOINT_DIR=outputs/test-fast/pcd-no-visual/<run_id>/deplot_no_vs_opd_pcd/final_checkpoint \
EXPERIMENT=pcd_no_visual_10epoch \
bash scripts/run_eval_ablation.sh
```

如果要并行评估该 run 下的所有 epoch checkpoint：

```bash
MODEL_DIR=outputs/test-fast/pcd-no-visual/<run_id>/deplot_no_vs_opd_pcd \
OUT_DIR=outputs/eval_chartqa_pcd_no_visual_<run_id> \
GPU_LIST="0 1 2 3 4 5 6" \
bash scripts/test/eval_opd_epochs_parallel.sh
```

## 2. 主实验之外必须补跑的实验

这些实验是完成主图和主表所需的最低集合。已有 checkpoint 的只需要补 eval；没有 checkpoint 的再补训练。

| 优先级 | 实验 | 推荐命名 | 目的 | 产物 |
| --- | --- | --- | --- | --- |
| 必须 | Base / no training eval | `Base` | 证明训练带来的绝对增益 | final eval |
| 必须 | 10epoch SFT | `SFT` | 对照 off-policy imitation | final eval + length/format |
| 必须 | 10epoch GRPO/DyME/RLVR | `GRPO` 或 `DyME` | 对照 sparse verifiable RL | final eval + reward std 曲线 |
| 必须 | 10epoch no-PCD anchor | `Ours w/o PCD rescue` | 隔离 all-wrong rescue 的贡献 | final eval + route/funnel |
| 必须 | Teacher/evidence sanity | `No-gold teacher check` | 支撑无答案泄漏 claim | gold suffix、parse、teacher accuracy |
| 必须 | 主实验 final checkpoint eval | `Ours` | 填 Table 1 主行 | accuracy、format、length |

建议使用总控脚本补齐主实验之外的必须实验：

```bash
bash scripts/test/run_paper_required_10epoch.sh \
  --dry-run \
  --run-id paper_required_10epoch \
  --main-run-id <main_run_id>
```

确认 dry-run 输出后，可按阶段启动：

```bash
bash scripts/test/run_paper_required_10epoch.sh \
  --run \
  --run-id paper_required_10epoch \
  --main-run-id <main_run_id> \
  --stages sft_train,dyme_train,no_pcd_anchor
```

训练完成后再补 eval 和 sanity：

```bash
bash scripts/test/run_paper_required_10epoch.sh \
  --run \
  --run-id paper_required_10epoch \
  --main-run-id <main_run_id> \
  --stages base_eval,sanity,eval_required
```

如果只想直接跑 no-PCD anchor：

```bash
bash scripts/test/run_opd_deplot_ablation.sh \
  --run \
  --epochs 10 \
  --run-id deplot_10epoch_main_anchor \
  --variants deplot_no_vs_opd
```

如果 Base/SFT/GRPO 还没有同预算产物，可用现有 fast baseline 入口统一补：

```bash
DYME_FAST_NUM_TRAIN_EPOCHS=10 \
DYME_FAST_SFT_EPOCHS=10 \
bash scripts/test/run_all_baselines.sh
```

注意：如果 `run_all_baselines.sh` 里的 OPD baseline 和主实验设置不完全一致，论文主表中应把它标为 generic OPD 或 fast OPD baseline，不要和 `Ours w/o PCD rescue` 混用。

## 3. 机制增强实验

这些不是主实验成立的必要条件，但能解释“为什么有效”和“哪些设计贡献最大”。

| 实验 | Variant | 目的 | 推荐位置 |
| --- | --- | --- | --- |
| Variance-adaptive only | `deplot_no_vs_opd_va` | 检查低 reward std 时放大 OPD 权重是否有用 | Table 2 |
| PCD only | `deplot_no_vs_opd_pcd` | 主方法；也用于机制表 | Table 1 + Table 2 |
| VA + PCD | `deplot_no_vs_opd_va_pcd` | 检查 adaptive weight 与 all-wrong rescue 是否互补 | Table 2 |

推荐一次性补跑/复用：

```bash
bash scripts/test/run_opd_deplot_ablation.sh \
  --run \
  --epochs 10 \
  --run-id deplot_10epoch_va_pcd \
  --variants deplot_no_vs_opd_va,deplot_no_vs_opd_pcd,deplot_no_vs_opd_va_pcd
```

如果这三个已经跑完，优先做 final checkpoint eval 和日志统计，不需要重复训练。

## 4. 可选但很加分的补充实验

| 实验 | 目的 | 何时需要 |
| --- | --- | --- |
| Budget curve: 1/2/4/10 epoch 或 epoch checkpoints | 证明不是 final checkpoint 偶然好，而是同预算更高效 | 主表结果差距不大时很重要 |
| Longer continuation | 看方法是否继续提升或过拟合 | 10epoch 曲线仍在上升时 |
| Visual-supervision controls | 证明收益不是来自额外 VS | 审稿人质疑视觉监督混淆时 |
| Teacher trajectory off | 隔离 teacher trajectory 与 OPD routing | 机制表空间足够时 |
| Evidence quality ablation | format-only vs DePlot evidence | 需要证明 DePlot evidence 的必要性时 |
| Second benchmark | A-OKVQA、geometry 或另一个可验证视觉任务 | 投稿强度需要跨任务泛化时 |

如果 10epoch 仍未收敛，longer continuation 可以从主实验继续接：

```bash
DYME_PCD_RUN_ID=<run_id> bash scripts/test/run_pcd_no_visual.sh 20 --resume auto
```

## 5. 图表最终组织

推荐正文顺序：

1. **Figure 1: Failure heterogeneity and recoverability.** 用 GRPO/no-PCD/Ours 日志展示 low-variance groups、wrong-but-recoverable samples 和 all-wrong rescue 需求。
2. **Figure 2: Method overview.** 画宏观训练信号，不画环境变量路由图。
3. **Table 1: Main results.** Base、SFT、GRPO/DyME、Ours w/o PCD、Ours。
4. **Figure 3: Training dynamics.** reward、reward std、useful update coverage、format/length。
5. **Table 2: Mechanism ablations.** no-PCD、VA-only、PCD、VA+PCD、必要时 teacher-trajectory/evidence control。
6. **Figure 4: Useful-signal conversion funnel.** 主实验和 no-PCD anchor 对比，突出 all-wrong rescue 把低信号失败转化为 OPD。
7. **Table 3: Cost and reliability.** teacher calls、parse success、gold suffix rate、DePlot evidence coverage、wall-clock。
8. **Figure 5: Qualitative cases.** 成功恢复、失败回退、teacher 不可恢复三类样例。

## 6. 当前绘图产物如何落地

已有 `scripts/analysis/pcd_paper_artifacts.py` 可以继续用来生成机制图，但需要在 manifest 里把主实验路径指向：

```text
outputs/test-fast/pcd-no-visual/<run_id>/deplot_no_vs_opd_pcd
```

建议分工：

- `fig1_motivation`：改成 Figure 1 的 recoverability/failure heterogeneity 面板来源。
- `fig5_teacher_rescue_funnel`：作为 Figure 4 useful-signal funnel 的核心。
- `fig6_va_vs_pcd_diagnosis`：放在 Table 2 机制消融附近。
- 新增或补齐 main-result table 脚本：从 eval summaries 汇总 Table 1，不要从训练 reward 直接替代 held-out eval。

## 7. 最小完成清单

论文主结果达到可写状态前，至少完成：

- `Ours` 主实验 final checkpoint eval。
- `Base/SFT/GRPO or DyME/Ours w/o PCD/Ours` 的同预算 Table 1。
- `Ours` 与 `Ours w/o PCD` 的训练日志曲线和 useful-signal funnel。
- teacher no-gold/evidence sanity 表。
- 3-5 个 qualitative correction cases。

VA-only 和 VA+PCD 可以作为次级机制消融；如果它们收益不稳定，也不会动摇主方法，因为主 claim 是 recoverability-guided all-wrong rescue 与 dense on-policy supervision。
