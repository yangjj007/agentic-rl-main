# OPD / CLRC Main5 Paper Plan

Date: 2026-07-14

本文档用于论文主线和 10epoch 全量实验排程。可执行迁移步骤见
`docs/chartqa_10epoch_ablation_migration.md`。

## 1. 论文主张

主张写窄：

```text
CLRC is a verifier-routed OPD curriculum for sub-1B VLM ChartQA reasoning. It uses realized group-level RL signal to decide when teacher distribution matching should help, and it reduces teacher style transfer with answer-, confidence-, and evidence-backed selective OPD.
```

不要写成“OPD 首创”或“通用 VLM RL 方法”。AAAI 主会创新点应落在：

- recoverability routing：只在 teacher 能无 gold 证据恢复答案时使用 OPD；
- realized-GRPO controller：用真实组内 GRPO coverage 调节 OPD exposure；
- selective OPD：answer-anchor、strict confidence、evidence-backed 三种不同训练信号；
- matched decomposition：DyME、CLRC、OPD/GRPO/diagnostic controls 分开报告；
- 负结果审计：解释高 OPD、低 GRPO、teacher style transfer 为什么会失败。

## 2. 当前证据

已完成 4epoch 事实：

```text
gold-hidden legacy PCD aligned: 0.5420
oracle route_guard: 0.5592
oracle full-template repair: 0.5624
oracle constrained repair: 0.5656
oracle student_hint_short: 0.5800
oracle official: 0.5872
oracle OPD + full teacher trajectory: 0.5120
```

核心瓶颈不是 teacher 不会答，而是 teacher-correct 难转成学生自己的 GRPO signal：

- `student_hint_short` last50：`all_wrong=0.545`、`grpo_route=0.2325`、`opd_route=0.6319`、`grpo_zero_loss=0.75`；
- oracle official final `0.5872`，但 last50 仍有 `all_wrong=0.775`、`grpo_route=0.040`、`zero=0.87`；
- full-trajectory OPD final 只有 `0.5120`，但 train accuracy/GRPO 已到约 `0.445/0.486`，说明 held-out/style gap 很大；
- clean no-full-hint OPD 在 step 72--81 有恢复迹象：accuracy/GRPO/OPD `0.0836/0.0906/0.7500`，但无 final eval。

因此主实验要优先回答两件事：

1. OPD 是否能超过 DyME full baseline；
2. 哪一种 selective OPD 能减少 teacher-style transfer，并让 GRPO 接管。

## 3. Main5 全量实验

默认只跑 5 条 10epoch 主全量：

| Label | Role | Unique change | 主 claim |
|---|---|---|---|
| `dyme_full_original` | 强 baseline | 原始 DyME full，统一改为 `python -m accelerate.commands.launch` | 效果目标线 |
| `clrc_full` | 主方法 | routed OPD + realized-GRPO controller | recoverability + controller 是否有效 |
| `answer_anchor_clrc` | selective OPD 1 | 只高权重蒸馏 Answer marker 后与数字/答案 token | 降低 rationale/style transfer |
| `confidence_weighted_clrc` | selective OPD 2 | strict teacher-probe：Answer 标记、可解析、不截断、0 容差 | 提高 teacher-correct precision |
| `evidence_adaptive_clrc` | selective OPD 3 | visual evidence gate + CoT gate + answer-anchor token weighting | 减少错误视觉证据带来的 OPD 噪声 |

推荐主表列：

```text
method, teacher_gold, verifier_reference, epochs, final_acc,
opd_route_last50, grpo_route_last50, all_wrong_last50,
teacher_probe_correct_rate, teacher_probe_strict_rejected_rate,
teacher_probe_evidence_present_rate, output_type_counts
```

胜负判断：

- `dyme_full_original` 是主效果线；
- `clrc_full` 若超过 DyME，论文主方法成立；
- 三条 selective OPD 中任一超过 `clrc_full` 或显著降低 style gap，可作为最终方法；
- 若都未超过 DyME，用 appendix 诊断说明瓶颈，不做强 claim。

## 4. Appendix / 二阶段实验

`--preset all` 保留 18 项，但不默认跑全量。优先级：

```text
gold_hidden_routed_opd_fixed: CLRC 去掉 global controller 的因果控制。
gold_hidden_no_opd: 同 routing/probe 条件但 OPD weight=0，排除 routing confound。
grpo_only_matched / opd_only_matched: 单信号贡献。
gold_hidden_uncond_opd: 证明 verifier routing 不是装饰。
clrc_target020: controller target 敏感性，不是新方法。
oracle_clean_no_full_hint / oracle_official_best_4e: 上界诊断。
fallback_only_matched / mixed_group_shortest_correct_hard_replay: 短跑诊断。
grpo_recovery_boost_clrc: 二阶段候选，只在 main5 出现高 OPD/低 GRPO 失败时跑。
token_reliability_clrc: lexical 旧原型，已被 answer-anchor/evidence-backed 主线替代。
```

## 5. 相关工作边界

Primary sources：

- ChartQA benchmark: <https://aclanthology.org/2022.findings-acl.177/>
- DePlot plot-to-table reasoning: <https://arxiv.org/abs/2212.10505>
- MatCha chart derendering/math pretraining: <https://arxiv.org/abs/2212.09662>
- LLaVA-OneVision model family: <https://arxiv.org/abs/2408.03326>
- DeepSeekMath / GRPO: <https://arxiv.org/abs/2402.03300>
- Distilling step-by-step reasoning: <https://arxiv.org/abs/2305.02301>

定位：

- ChartQA/DePlot/MatCha 说明 chart reasoning 需要结构化视觉证据，支持 `evidence_adaptive_clrc`；
- LLaVA-OneVision 是当前 student/teacher family，实验不能混用其他 backbone 结论；
- DeepSeekMath/GRPO 支持用 group-relative RL 信号，但不解决 teacher distribution 何时介入；
- step-by-step distillation 支持 rationale supervision，但我们的负结果表明 uniform teacher trajectory 会造成 style transfer，因此需要 selective OPD。

论文写法：

```text
Prior VLM chart methods improve perception and structured evidence. Prior RLVR methods improve sparse-reward optimization. Prior distillation methods provide dense reasoning supervision. CLRC addresses a different interface problem: when an on-policy failed rollout should receive teacher distribution guidance, and which teacher tokens are reliable enough to distill.
```

## 6. 全量运行命令

主全量：

```bash
export RUN_ID=chartqa10_main5_$(date +%Y%m%d)
export DYME_MODEL_ROOT="$PWD/models"
export DYME_STUDENT_MODEL="$PWD/models/llava-0.5b-ov"
export DYME_TEACHER_MODEL="$PWD/models/llava-7b-ov"
export DYME_CHARTQA_ABLATION_OUTPUT_ROOT=/path/to/big_disk/chartqa-ablation/checkpoints
export DYME_CHARTQA_ABLATION_LOG_ROOT=/path/to/big_disk/chartqa-ablation/logs
export DYME_CHARTQA_ABLATION_RESULTS_ROOT="$PWD/docs/experiment_results/chartqa-ablation"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/test/run_chartqa_10epoch_ablation_matrix.sh \
  --run \
  --run-id "$RUN_ID" \
  --epochs 10 \
  --preset main5 \
  --shard-index 0 \
  --shard-count 4 \
  --stages train,eval \
  --speed-profile canonical \
  --resume none
```

四台机器分别用 `--shard-index 0/1/2/3`。训练完成后脚本自动 eval，并把轻量结果复制到：

```text
docs/experiment_results/chartqa-ablation/<RUN_ID>/
```

## 7. Smoke 与状态

已跑通：

```text
answer_anchor_clrc: smoke_newdirs_20260714_v1, 1 step
confidence_weighted_clrc: smoke_newdirs_20260714_v1, 1 step
evidence_adaptive_clrc: smoke_main5_evidence_20260714_v1, 1 step
```

本地验证：

```text
64 passed
bash -n passed
py_compile passed
main5 dry-run lists 5 variants
all dry-run lists 18 variants
```

当前 Git 状态：

```text
local commit: see git log, message "Add main5 ChartQA OPD directions"
remote branch target: chartqa-opd-ablation-bab7fea
push status: pending credentials
```
