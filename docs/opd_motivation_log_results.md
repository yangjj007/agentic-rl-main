# OPD Motivation 日志结果记录

日期：2026-06-23  
关联规划：`docs/opd_experiment_plan.md`

本文档只记录 Step 1 零训练日志分析结果。为避免过度依赖最后 20 step 的摘要，本版使用 **每 10 个 training step 分箱取均值** 的折线图。指标只选直接训练/算法指标，不使用 `degenerate_rate`、`collapse_rate` 等二级健康指标。

## 1. 使用日志

只比较两条 clean fast logs：

| Method | Log Path | 说明 |
| --- | --- | --- |
| DyME | `outputs/test-fast/logs/train_test_dyme_20260621_112902.log` | clean fast DyME |
| OPD | `outputs/test-fast/logs/train_test_opd_20260621_212323.log` | clean no-gold OPD |

## 2. 指标选择

使用直接训练/算法指标：

| Metric Key | 图中名称 | 含义 |
| --- | --- | --- |
| `rewards/accuracy/mean` | Accuracy reward | 任务答案奖励，直接反映训练中可用 answer signal |
| `rewards/format/mean` | Format reward | 格式奖励，反映训练输出是否满足格式约束 |
| `reward` | Total reward | 总 reward |
| `loss` | Training loss | 训练 loss |
| `routing/sft_replaced_ratio` | SFT route ratio | completion 被 SFT 替换/回退的比例 |
| `routing/grpo_on_correct_rate` | GRPO route ratio | 正确 completion 进入 GRPO 的比例 |
| `routing/opd_teacher_call_rate` | OPD route ratio | 触发 OPD teacher call 的比例 |

说明：`completions/degenerate_rate`、`completions/clipped_ratio`、health alert 等不再作为本图主指标，因为它们更像实现侧/诊断侧二级指标。本文档只在文字中谈“训练信号消失/恢复”，不再用这些二级指标定义 collapse。

## 3. 生成命令

```bash
python3 scripts/analyze_opd_routes.py \
  --compare DyME=outputs/test-fast/logs/train_test_dyme_20260621_112902.log \
            OPD=outputs/test-fast/logs/train_test_opd_20260621_212323.log \
  --step-interval 10 \
  --csv-out docs/figures/opd_motivation_direct_metrics_10step.csv \
  --plot-out docs/figures/opd_motivation_direct_metrics_10step.png
```

产物：

- CSV：`docs/figures/opd_motivation_direct_metrics_10step.csv`
- 图：`docs/figures/opd_motivation_direct_metrics_10step.png`

采样方式：每 10 个 metric rows 做一个 non-overlapping bin mean；横轴为该 bin 的结束 step。因此最后一个点是 step 588，对应最后 8 个 rows 的均值。

## 4. 图

![Direct training and routing metrics](figures/opd_motivation_direct_metrics_10step.png)

## 5. 曲线读数

下面是从 10-step CSV 中读取的代表性锚点。它们不是最后 20 step 摘要，而是每 10 step 分箱后的曲线点。

### 5.1 DyME

| Step | Accuracy Reward | Format Reward | Total Reward | Loss | SFT Route | GRPO Route | OPD Route |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.0133 | 0.0461 | 0.1949 | -0.9614 | 0.9625 | 0.0063 | 0.0000 |
| 100 | 0.0000 | 0.0039 | 0.2115 | -0.9984 | 1.0000 | 0.0000 | 0.0000 |
| 200 | 0.0000 | 0.0063 | 0.1963 | -0.9984 | 1.0000 | 0.0000 | 0.0000 |
| 300 | 0.0000 | 0.0031 | 0.1954 | -0.9977 | 0.9938 | 0.0000 | 0.0000 |
| 400 | 0.0000 | 0.0055 | 0.2295 | -0.9945 | 1.0000 | 0.0000 | 0.0000 |
| 500 | 0.0008 | 0.0063 | 0.2810 | -0.9965 | 0.9938 | 0.0000 | 0.0000 |
| 588 | 0.0000 | 0.0010 | 0.2490 | -1.0000 | 1.0000 | 0.0000 | 0.0000 |

### 5.2 OPD

| Step | Accuracy Reward | Format Reward | Total Reward | Loss | SFT Route | GRPO Route | OPD Route |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.0301 | 0.1246 | 0.2243 | -0.9751 | 0.9906 | 0.0094 | 0.0000 |
| 100 | 0.0000 | 0.0016 | 0.3058 | -1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 200 | 0.0000 | 0.0023 | 0.3986 | -1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 300 | 0.0000 | 0.0059 | 0.4434 | -1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 400 | 0.0008 | 0.0043 | 0.4841 | -1.0011 | 0.9969 | 0.0031 | 0.0000 |
| 500 | 0.0121 | 0.2691 | 0.5304 | -1.0043 | 0.9844 | 0.0156 | 0.0000 |
| 588 | 0.2725 | 0.9629 | 0.8609 | -0.8223 | 0.7891 | 0.1992 | 0.0117 |

## 6. 结果解读

### 6.1 DyME 的问题不是只出现在最后 20 步

从 10-step 曲线看，DyME 在 step 100 之后 `rewards/accuracy/mean` 基本长期为 0，`routing/grpo_on_correct_rate` 也长期为 0。也就是说，训练早期之后几乎没有 correct rollout 进入 GRPO，训练主要表现为 SFT route ratio 接近 1。

这比“last-20 很差”的表述更稳：问题不是最后突然坏掉，而是从较早阶段开始就缺少可持续的 answer reward / GRPO signal。

### 6.2 OPD 的改善主要发生在后段训练

OPD 在前 300 step 也没有明显 accuracy reward，但后段开始恢复：

- step 400：accuracy reward 仍接近 0，但 GRPO route 已有轻微出现；
- step 500：accuracy reward 上升到 0.0121，format reward 到 0.2691；
- step 588：accuracy reward 到 0.2725，format reward 到 0.9629，GRPO route 到 0.1992，SFT route 降到 0.7891。

这说明 OPD 的现有 clean run 更像是后段“解锁”了可训练信号：format reward、accuracy reward、GRPO route 一起上升，而 DyME 没有出现类似趋势。

### 6.3 OPD route 本身比例不高，但它改变了后续训练状态

在 clean OPD 中，`routing/opd_teacher_call_rate` 的曲线整体较低，最后一个 10-step bin 约 0.0117。这意味着主张不应写成“OPD 大量替代了训练样本”，而应写成：

> 少量 teacher-recoverable wrong completions 被 OPD 利用后，后段训练逐渐恢复 format/accuracy reward，并让更多 completion 进入 GRPO，而不是一直回退 SFT。

这更符合 recoverability-aware third mode 的叙事，也避免把方法误解释为大规模 teacher distillation。

## 7. 口头报告版本

Step 1 我们重新按每 10 step 分箱画了直接训练指标，不再用 degenerate/collapse 这类二级指标。结果更清楚：DyME 在 step 100 之后 accuracy reward 和 GRPO route 基本长期为 0，SFT route 几乎一直接近 1，说明训练没有持续产生 correct rollout 信号。OPD 前期也不强，但在后段出现明显转折：accuracy reward、format reward、total reward 和 GRPO route 同时上升，SFT route 下降，最后一个 10-step bin 的 accuracy reward 达到 0.2725、format reward 达到 0.9629、GRPO route 达到 0.1992。这个趋势比单看 last-20 更能支撑 motivation：OPD 不是靠大规模 teacher 替代训练，而是通过 recoverability-aware 的少量 OPD 触发，让后段训练重新进入有 reward、有 GRPO signal 的状态。

## 8. 写作建议

更稳妥的 motivation 表述：

> In the clean DyME run, direct training signals vanish early: after roughly 100 optimization steps, accuracy reward and GRPO routing remain near zero while the SFT fallback ratio stays close to one. In contrast, clean no-gold OPD shows a late-stage recovery: accuracy reward, format reward, and GRPO routing increase together, while SFT fallback decreases. This suggests that OPD helps the small VLM move out of an SFT-dominated training regime and recover usable RL signal.

中文版本：

> 在 clean DyME run 中，直接训练信号较早消失：约 100 step 后，accuracy reward 和 GRPO route 长期接近 0，而 SFT fallback ratio 基本接近 1。相比之下，clean no-gold OPD 在后段出现恢复：accuracy reward、format reward 和 GRPO route 同时上升，SFT fallback 下降。这说明 OPD 的作用不是大规模 teacher 替代，而是帮助小 VLM 从 SFT-dominated 训练状态中恢复可用的 RL signal。
