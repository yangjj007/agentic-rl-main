# OPD 10 Epoch 主训练与 Eval 结果记录

更新时间：2026-06-23

## 1. 主训练日志

日志文件：

`outputs/test-fast/logs/train_opd_7b_dyme_probe_20260622_101112.log`

解析到 1470 条训练 metric 记录，对应 10 epoch。

最后一步直接指标：

| 指标 | 数值 |
| --- | ---: |
| epoch | 10.0 |
| rewards/accuracy/mean | 0.5039 |
| rewards/format/mean | 0.9453 |
| reward | 2.0078 |
| loss | -0.5269 |
| routing/opd_teacher_call_rate | 0.0000 |
| routing/sft_replaced_ratio | 0.7813 |
| routing/grpo_on_correct_rate | 0.2188 |

最后 50 step 滑动平均：

| 指标 | 数值 |
| --- | ---: |
| rewards/accuracy/mean | 0.5827 |
| rewards/format/mean | 0.9434 |
| reward | 2.1627 |
| loss | -0.4737 |
| routing/sft_replaced_ratio | 0.3969 |
| routing/grpo_on_correct_rate | 0.5787 |
| routing/opd_teacher_call_rate | 0.0244 |
| routing/teacher_probe_candidate_rate | 0.1062 |
| routing/teacher_probe_correct_rate | 0.0244 |
| routing/teacher_probe_wrong_rate | 0.0819 |
| signal/grpo_zero_loss_rate | 0.3200 |
| loss/opsd | 0.0267 |
| loss/teacher_traj_fkl | 0.0131 |

图表：

- `docs/figures/opd_10epoch_training_direct_metrics_10step.png`
- `docs/figures/opd_10epoch_routing_teacher_metrics_10step.png`
- `docs/figures/opd_10epoch_loss_safety_metrics_10step.png`

## 2. DePlot 数据生成状态

已完成，无需继续等待。

| 项目 | 数值 |
| --- | --- |
| 输出文件 | `/tmp/train_medium_vf_full_deplot.json` |
| cache 文件 | `/tmp/train_medium_vf_full_deplot_cache.json` |
| 日志文件 | `/tmp/train_medium_vf_full_deplot.nohup.log` |
| 总样本数 | 23171 |
| real | 23171 |
| cached | 16458 |
| placeholder | 0 |
| skipped | 0 |
| failed | 0 |

用 `visual_fact_deplot` 字段复查，空值或 DePlot 占位符数量为 0。

## 3. 不同 Epoch 权重 Eval

评测脚本：

`scripts/test/eval_opd_epochs_parallel.sh`

汇总文件：

`outputs/eval_chartqa_opd_epochs/summary.csv`

图表：

- `docs/figures/opd_epoch_eval_chartqa_accuracy.png`
- `docs/figures/opd_epoch_eval_chartqa_accuracy.csv`

所有权重均完成 2500/2500 条 ChartQA eval，退出码为 0。最优权重是 `checkpoint-1176`，accuracy 为 0.5224。

| 权重 | Accuracy | Processed |
| --- | ---: | ---: |
| checkpoint-147 | 0.4372 | 2500/2500 |
| checkpoint-294 | 0.4984 | 2500/2500 |
| checkpoint-441 | 0.4952 | 2500/2500 |
| checkpoint-588 | 0.5092 | 2500/2500 |
| checkpoint-735 | 0.4788 | 2500/2500 |
| checkpoint-882 | 0.4728 | 2500/2500 |
| checkpoint-1029 | 0.5056 | 2500/2500 |
| checkpoint-1176 | **0.5224** | 2500/2500 |
| checkpoint-1323 | 0.5112 | 2500/2500 |
| checkpoint-1470 | 0.5216 | 2500/2500 |
| final_checkpoint | 0.5216 | 2500/2500 |

当前结论：第 8 个 epoch 左右的 `checkpoint-1176` 最好；最终权重与 `checkpoint-1470` 持平，但略低于 `checkpoint-1176`。
