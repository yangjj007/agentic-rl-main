# PCD Oracle Exploration Notes

更新时间：2026-07-09

> **历史术语说明（2026-07-13）：** 本文件记录早期 oracle 探索，不再作为论文事实源。
> 其中“full-CoT 污染”应解释为**固定模板与空/异常 Answer 共同出现的输出失败**，而不是
> 把有内容的 full-CoT 或 `Goal/Observation/Reasoning/Conclusion` 格式本身视为错误。
> 当前事实口径见 `paper_reconstruction/chinese_draft.md`、`claim_evidence_matrix.md` 和
> `opd_experiment_plan.md`。

## Diagnostic Direction: `student_hint_short`

`deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short` 单独作为主线不够强，但它仍是有用的 teacher-SFT repair target style。

证据：

- final checkpoint ChartQA test accuracy: `0.5800`
- 输出统计：`other=1629`、`answer_flag=821`、`full_cot=50`
- 相比 constrained repair `0.5656` 有提升，但没有超过 oracle official `0.5872`，也没有达到 `0.60` 目标。
- 训练健康指标改善但没有转化成足够 eval 增益：
  - last50 `group_all_wrong_rate=0.545`
  - last50 `routing/grpo_route_rate=0.2325`
  - last50 `signal/grpo_zero_loss_rate=0.75`
  - teacher-correct rate `0.9248`，parse-fail `0`
- 失败形态：完整四段输出大幅下降，但 eval log 出现大量新的短模板失败，例如
  `Reasoning style` 和异常 `Answer:` 空行格式。

结论：

`student_hint_short` 证明“缩短 teacher-SFT repair target”有一定价值：它显著降低完整
teacher 模板复制，但没有单独解决后期 GRPO 有效信号不足，也产生了新的短模板失败。
后续不再把它作为唯一改动冲 `0.60+`；而是把它作为较低 hard-template exposure 的
repair target，与 OPD decay/cap 或动态采样组合。

保留的对照：

- `deplot_no_vs_opd_pcd_oracle_hint`：当前 oracle official，上界/基准，final `0.5872`
- `deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair`：teacher-SFT repair 主机制对照
- `deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_style`：teacher reasoning short target 对照
- `deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_answer_only`：只答案诊断对照
- `deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay`：`student_hint_short` + 后期 OPD decay/cap 主实验

## New Main Direction: OPD Decay / Cap After Repair

### Motivation

现有 evidence 指向机制瓶颈，而不是 teacher prompt：

- oracle official final `0.5872`，但 last50 `all_wrong=0.775`、`grpo_route=0.040`、`zero=0.87`。
- `student_hint_short` final `0.5800`，full-CoT 降到 `2%`，但 last50 仍 `zero=0.75`、`opd_route=0.6319`。
- teacher parse-fail 基本为 `0`，oracle teacher-correct 很高；问题是 teacher-correct 大量转成 OPD，而不是让后期 GRPO 接管。

因此下一版主实验不再继续调 prompt，而是让 OPD 从冷启动辅助逐步退场。

### Implemented Variant

`deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay`

配置：

- 保留 oracle providers：`format_only,visual_facts_deplot,oracle_hint`
- 保留 all-wrong PCD：`DYME_TEACHER_PROBE_ALL_WRONG_AFTER_STEP=0`
- 保留 teacher-SFT repair：`DYME_TEACHER_CORRECT_REPAIR_MODE=traj_sft`
- repair target：`DYME_TEACHER_SFT_TARGET_STYLE=student_hint_short`
- teacher trajectory FKL decay：
  - start `147`
  - end `294`
  - final weight `0.0`
- OPD loss weight decay：
  - start `294`
  - end `441`
  - final weight `0.5`
- OPD route cap：
  - after step `294`
  - max OPD completions per prompt `2`
  - overflow route `SFT`

Expected effect:

- last50 `opd_route_rate` 明显低于 `0.63`
- last50 `grpo_route_rate` 高于 `0.23`
- last50 `grpo_zero_loss_rate` 低于 `0.75`
- eval 保持 `student_hint_short` 的低 full-CoT 污染，同时超过 oracle official `0.5872`

Smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DYME_PCD_RUN_ID=pcd_oracle_student_hint_opd_decay_smoke10 \
DYME_PCD_MAX_STEPS=10 \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay
```

4epoch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DYME_PCD_RUN_ID=pcd_oracle_student_hint_opd_decay_4epoch \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay
```

Checkpoint sweep after training:

```bash
DYME_EVAL_OUT_ROOT=outputs/test-fast/pcd-no-visual/pcd_oracle_student_hint_opd_decay_4epoch \
bash scripts/test/eval_deplot_ablation_checkpoints.sh --force
```

## Bigger Distribution-Level Variant: Effective Sampling

### Motivation

OPD decay/cap 只改变一个 batch 内的 loss/routing 比例，但如果后期 sampler 仍不断抽到 all-wrong 或 all-correct prompt，GRPO 仍然会有大量 zero-signal。DAPO/ReST/STaR 类方法的共同启发是：训练应主动提高有效样本/正轨迹的占比，而不是平均消耗所有 prompt。

### Implemented Variant

`deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling`

它继承 `student_hint_short_opd_decay` 的所有设置，并新增 dynamic effective sampler：

- 每个训练 prompt 保留最近一次 group 状态：
  - `mixed`: 至少一个 correct、至少一个 wrong
  - `all_wrong`
  - `all_correct`
  - `unknown`
- step `<294` 仍按普通随机 sampler。
- step `>=294` 开始按状态加权采样：
  - mixed weight `4.0`
  - all-wrong weight `1.0`
  - all-correct weight `0.7`
  - unknown weight `1.0`
  - reward std bonus `2.0`
- sampler 仍然对每个选中 prompt 重复 `num_generations` 次，保持 GRPO group 结构不变。

Expected effect:

- 后期 batch 中 mixed prompt 占比提高；
- last50 `signal/grpo_zero_loss_rate` 明显低于 `0.75`;
- last50 `routing/grpo_route_rate` 高于 `0.23`;
- 不显著增加 full-CoT / answer-flag 污染。

Smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DYME_PCD_RUN_ID=pcd_oracle_student_hint_opd_decay_sampling_smoke10 \
DYME_PCD_MAX_STEPS=10 \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling
```

4epoch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DYME_PCD_RUN_ID=pcd_oracle_student_hint_opd_decay_sampling_4epoch \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling
```

Primary comparison:

- `student_hint_short_4epoch`: final `0.5800`
- `student_hint_short_opd_decay_4epoch`: tests whether OPD退火/cap helps without changing sampler
- `student_hint_short_opd_decay_sampling_4epoch`: tests whether changing prompt distribution reduces zero-signal

If effective sampling improves train health but hurts eval, next knob should not be prompt engineering; first lower mixed weight from `4.0` to `2.0` or delay `after_step` from `294` to `441`.

## Bigger Data-Level Direction: Positive Replay Buffer

### Motivation

在线 PCD/oracle 训练里，teacher-correct 数量非常高，但模型没有把这些正确轨迹充分转化成后期 GRPO 信号：

- `student_hint_short` run 的 candidate logs 全量统计：
  - candidate rows: `125451`
  - teacher-correct candidates: `116016`
  - parse-fail: `0`
  - dataset match missing: `0`
  - 去重后 unique positive prompts: `4378`
  - all-wrong positive prompts: `3107`
  - mixed-wrong positive prompts: `1271`
- 这说明 teacher 知道大量正确样本，但在线训练把它们重复地放进 OPD/repair slot，并没有稳定变成学生自己的 correct completions。

因此下一步更大的方向不是继续改 prompt，而是把 teacher-correct 样本沉淀成去重后的 positive replay buffer，做离线 SFT warmup 或在线 replay mixing。

### Implemented Export Tool

新增脚本：

```bash
python scripts/analysis/export_positive_replay_buffer.py \
  --candidate-glob 'outputs/test-fast/pcd-no-visual/pcd_oracle_teacher_sft_repair_student_hint_short_4epoch/deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short/teacher_probe_candidates/rank*.jsonl' \
  --dataset data/chartqa/train_medium_vf_full.json \
  --out-dir outputs/test-fast/positive-replay-buffer/student_hint_short_full \
  --target-style student_hint_short
```

设计原则：

- candidate log 只作为 verifier/filter signal；
- 不使用 raw `teacher_output` 作为 replay target，因为 log 里常见截断和 privileged context；
- 回连 dataset 后，用 verified `hint` / `visual_fact_hint` + reference answer 生成 target：
  - `Reasoning: <verified hint reasoning>`
  - `Answer: <reference answer>`
- 按 dataset row + answer 去重，避免同一个 prompt 的 8 个 completion 变成重复 SFT；
- 输出 `replay.jsonl`、`replay_train.json`、`summary.csv`、`by_scope.csv`、`preview.jsonl`。
- `replay_train.json` 专门兼容现有 `prepare_chart_sft_data`：
  - `hint` 只包含无 `Answer:` 的 reasoning 前缀；
  - `answer` 保持短 reference answer；
  - `target` 保留完整 `Reasoning + Answer` 作为审计字段，训练 data collector 不使用它。

当前 full export 结果：

- `replay.jsonl`: `4378` rows
- `student_short_rate=1.0000`
- `exact_reference_answer_line_rate=1.0000`
- `privileged_tag_rate=0.0000`
- scope:
  - `all_wrong`: `3107`
  - `mixed_wrong`: `1271`

### Next Training Integration

推荐先做两个不互相混合的实验：

1. Replay-SFT warmup:
   - 先用 `replay_train.json` 做 0.25-0.5 epoch 轻量 SFT；
   - 再接原 oracle official 或 `student_hint_short_opd_decay_effective_sampling`；
   - 目标是降低初期 all-wrong，让 GRPO 更早进入 mixed 状态。

   已新增 warmup runner：

   ```bash
   # dry-run
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
   DYME_REPLAY_SFT_RUN_ID=positive_replay_warmup_0p5epoch \
   DYME_REPLAY_SFT_EPOCHS=0.5 \
   bash scripts/test/run_positive_replay_sft_warmup.sh --dry-run

   # run
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
   DYME_REPLAY_SFT_RUN_ID=positive_replay_warmup_0p5epoch \
   DYME_REPLAY_SFT_EPOCHS=0.5 \
   bash scripts/test/run_positive_replay_sft_warmup.sh
   ```

   Warmup 输出：

   ```bash
   outputs/test-fast/positive-replay-sft/positive_replay_warmup_0p5epoch/final_checkpoint
   ```

   接 DyME：

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
   DYME_STUDENT_MODEL=outputs/test-fast/positive-replay-sft/positive_replay_warmup_0p5epoch/final_checkpoint \
   DYME_PCD_RUN_ID=pcd_oracle_replay_warmup_student_hint_opd_decay_sampling_4epoch \
   bash scripts/test/run_pcd_no_visual_4epoch.sh \
     --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling
   ```

   更安全的一键两阶段 runner 已新增。它会先跑 replay SFT warmup，再把 warmup `final_checkpoint`
   明确作为第二阶段 `DYME_STUDENT_MODEL` 传给 DyME，避免只设置无效 env 导致 RL 仍从原始
   `llava-0.5b-ov` 起步。

   ```bash
   # dry-run：检查 warmup checkpoint handoff 和第二阶段 env
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
   DYME_CHAIN_RUN_ID=pcd_replay_warmup_rl_transition_4epoch \
   bash scripts/test/run_replay_warmup_then_pcd_4epoch.sh \
     --dry-run

   # run：stage 1 warmup + stage 2 DyME 4epoch
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
   DYME_CHAIN_RUN_ID=pcd_replay_warmup_rl_transition_4epoch \
   DYME_REPLAY_SFT_EPOCHS=0.5 \
   bash scripts/test/run_replay_warmup_then_pcd_4epoch.sh
   ```

   如果 warmup 已经跑完，只想重跑第二阶段：

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
   DYME_CHAIN_RUN_ID=pcd_replay_warmup_rl_transition_4epoch \
   DYME_CHAIN_SKIP_WARMUP=1 \
   bash scripts/test/run_replay_warmup_then_pcd_4epoch.sh
   ```

2. Online positive replay mixing:
   - 每 N step 从 replay buffer 注入少量 positive SFT batch；
   - 注入比例从 `10%-20%` 起；
   - epoch 2 后降低或关闭 replay mixing，避免覆盖 GRPO 主信号。

   已新增在线 replay variant：

   `deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_replay_mix`

   它继承：

   - oracle hint teacher probe
   - `student_hint_short` all-wrong teacher-SFT repair
   - teacher trajectory decay
   - OPD weight decay/cap
   - effective sampling

   并新增：

   - `DYME_POSITIVE_REPLAY=1`
   - `DYME_POSITIVE_REPLAY_DATASET=outputs/test-fast/positive-replay-buffer/student_hint_short_full/replay_train.json`
   - `DYME_POSITIVE_REPLAY_WEIGHT=0.1`
   - `DYME_POSITIVE_REPLAY_BATCH_SIZE=1`
   - `DYME_POSITIVE_REPLAY_AFTER_STEP=0`

   实现口径：

   - replay 不参与 GRPO reward / advantage / routing；
   - replay 作为额外 CE loss 加到 batch loss；
   - target 使用 replay row 的 `target` 字段，避免双 `Answer:`;
   - 记录 `loss/positive_replay`、`loss/positive_replay_weight`、`replay/positive_batch_size`、`replay/positive_skipped_rate`。

   Smoke:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
   DYME_PCD_RUN_ID=pcd_oracle_replay_mix_smoke10 \
   DYME_PCD_MAX_STEPS=10 \
   bash scripts/test/run_pcd_no_visual_4epoch.sh \
     --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_replay_mix
   ```

   4epoch:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
   DYME_PCD_RUN_ID=pcd_oracle_replay_mix_4epoch \
   bash scripts/test/run_pcd_no_visual_4epoch.sh \
     --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_replay_mix
   ```

3. Rollout-level prioritized replay:
   - static positive replay 学 teacher/dataset target；
   - rollout replay 学学生自己生成过、被 verifier 判对且 advantage 为正的 fresh rollout；
   - 用 PPO/GRPO-style clipped PG loss，而不是 CE；
   - 用 max-age 控制 off-policy staleness；
   - 目标是复用稀有正确 student trajectory，让后期 GRPO 信号密度更高。

   已新增 variant：

   `deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay`

   它继承 replay_mix 全部设置，并新增：

   - `DYME_ROLLOUT_REPLAY=1`
   - `DYME_ROLLOUT_REPLAY_WEIGHT=0.05`
   - `DYME_ROLLOUT_REPLAY_CAPACITY=256`
   - `DYME_ROLLOUT_REPLAY_BATCH_SIZE=2`
   - `DYME_ROLLOUT_REPLAY_AFTER_STEP=50`
   - `DYME_ROLLOUT_REPLAY_MAX_AGE_STEPS=64`
   - `DYME_ROLLOUT_REPLAY_MIN_ABS_ADVANTAGE=0.05`
   - `DYME_ROLLOUT_REPLAY_POSITIVE_ONLY=1`

   实现口径：

   - 每个 rank 维护本地 rollout buffer；
   - 只存 `advantage > 0` 且 `acc_reward > 0.5` 的 student rollout；
   - replay loss 使用保存时 old logp 和当前 logp 的 clipped policy-gradient ratio；
   - sample 优先级按 `abs(advantage) ** priority_alpha`；
   - 当前 batch 先 replay 历史，再把当前正 rollout 入库，避免同 step 自 replay。

   Smoke:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
   DYME_PCD_RUN_ID=pcd_oracle_rollout_replay_smoke10 \
   DYME_PCD_MAX_STEPS=10 \
   bash scripts/test/run_pcd_no_visual_4epoch.sh \
     --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay
   ```

   4epoch:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
   DYME_PCD_RUN_ID=pcd_oracle_rollout_replay_4epoch \
   bash scripts/test/run_pcd_no_visual_4epoch.sh \
     --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay
   ```

成功标准：

- early first50 `group_all_wrong_rate` 明显低于 student_hint_short；
- last50 `routing/grpo_route_rate` 高于 `0.2325`;
- last50 `signal/grpo_zero_loss_rate` 低于 `0.75`;
- final 或 best checkpoint 超过 oracle official `0.5872`;
- eval 输出不重新出现完整固定模板与异常 Answer 的联合失败，且 privileged tag rate 为零。

## Next Directions Toward `0.60+`

### Progress-Scheduled Condition-Aware OPD Overflow

Variant:

`deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_grpo_overflow`

This run replaces the fixed `147/294/441` phase boundaries with normalized
optimizer progress. It keeps structured ChartQA reasoning and
`student_hint_short` repair, while changing late OPD overflow routing:

- teacher trajectory weight decays over progress `0.25 -> 0.50`;
- effective sampling and the OPD route cap activate at progress `0.50`;
- OPD weight decays over progress `0.50 -> 0.75`;
- mixed-group OPD overflow returns to GRPO;
- all-wrong OPD overflow is zero-gradient skip rather than full-hint SFT;
- additive eval-format reward and replay are disabled.

The run also logs diagnostic-only dynamic trigger metrics. `sampling_needed`
tracks high zero-loss plus low mixed rate; `rl_ready` tracks low zero-loss plus
high mixed rate. These shadow conditions do not control the current run.

Smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DYME_PCD_RUN_ID=pcd_oracle_progress_grpo_overflow_smoke10 \
DYME_PCD_MAX_STEPS=10 \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_grpo_overflow
```

4epoch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DYME_PCD_RUN_ID=pcd_oracle_progress_grpo_overflow_4epoch \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_effective_sampling_grpo_overflow
```

Primary health checks:

- `phase/training_progress` and phase-active flags follow normalized progress;
- `routing/opd_route_cap_grpo_rate` becomes nonzero for mixed overflow;
- `routing/opd_route_cap_skip_rate` becomes nonzero for all-wrong overflow;
- `routing/sft_route_rate` falls without reducing genuine short repair;
- both `phase/dynamic_sampling_*` and `phase/dynamic_rl_*` metrics are present.

### 1. Eval-Aligned Output Reward

当前 format reward 后期仍高，但 eval log 里有大量 `Reasoning style`、空 `Answer:`、`Goal/Observation` 残留，说明训练 format reward 与 eval parser 目标不完全一致。

建议新 variant：

`deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward`

只改奖励，不改 teacher/routing：

- 加一个小权重 output-clean reward，要求：
  - 最后一条非空行能解析出答案；
  - 至多一个 `Answer:`；
  - 不包含 `[Oracle]`、`[Final Hard Rule]`、`[Verified Hint]`、`[DePlot]`、`Reasoning style`；
  - 不以 `Goal:` / `Observation:` / `Conclusion:` 结尾。
- 先用 `0.1` 权重做 10-step 和 1epoch smoke，再做 4epoch。

预期解决：短模板污染和 answer-line 污染。

实现状态：

- runner variant 已注册；
- config env 已注册：`DYME_EVAL_FORMAT_REWARD`、`DYME_EVAL_FORMAT_REWARD_WEIGHT`;
- trainer 会记录 `reward/eval_format_mean`、`reward/eval_format_weight`;
- 默认关闭，只有新 variant 打开。

推荐命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DYME_PCD_RUN_ID=pcd_oracle_eval_format_reward_smoke10 \
DYME_PCD_MAX_STEPS=10 \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DYME_PCD_RUN_ID=pcd_oracle_eval_format_reward_4epoch \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward
```

10-step gate：

- training exits OK;
- logs include `reward/eval_format_mean`;
- `DYME_EVAL_FORMAT_REWARD=1` appears in dry-run;
- no `teacher_sft_repair` metrics are required for this variant because it is anchored on oracle official, not teacher-SFT repair.

Smoke 检查：

```bash
python scripts/analysis/check_pcd_variant_smoke.py \
  --variant deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward \
  --log-dir outputs/test-fast/logs/pcd_no_visual_pcd_oracle_eval_format_reward_smoke10/deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward
```

### 2. Late-Phase OPD / Teacher-Trajectory Decay

`student_hint_short` 后期 OPD 仍约 `0.632`，GRPO route 约 `0.233`，zero-loss 仍 `0.75`。这说明 teacher repair 已经改善冷启动，但后期 teacher/OPD 仍压着 GRPO 主信号。

建议新 variant：

`deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay`

第一版只加 teacher trajectory FKL weight schedule，不改 OPD routing：

- step `<294`：teacher trajectory FKL weight 保持 `0.5`;
- step `294-441`：线性降到 `0.0`;
- step `>=441`：teacher trajectory FKL weight 保持 `0.0`;
- mixed group 仍保持 DyME 核心：correct -> GRPO，wrong teacher-correct -> OPD。

预期解决：OPD 后期过强、GRPO route 不足、zero-loss 过高。

实现状态：

- runner variant 已注册；
- config env 已注册：
  - `DYME_TEACHER_TRAJ_WEIGHT_DECAY`
  - `DYME_TEACHER_TRAJ_DECAY_START_STEP`
  - `DYME_TEACHER_TRAJ_DECAY_END_STEP`
  - `DYME_TEACHER_TRAJ_FINAL_WEIGHT`
- trainer 记录 `loss/teacher_traj_effective_weight`;
- 默认关闭，只有新 variant 打开。

推荐命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DYME_PCD_RUN_ID=pcd_oracle_late_traj_decay_smoke10 \
DYME_PCD_MAX_STEPS=10 \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DYME_PCD_RUN_ID=pcd_oracle_late_traj_decay_4epoch \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay
```

4epoch gate：

- `loss/teacher_traj_effective_weight` 从 `0.5` 降到 `0.0`;
- eval full-CoT 和 oracle/contract 残留不高于 oracle official;
- last50 `routing/grpo_route_rate` 高于 oracle official;
- final 或 best checkpoint 逼近/超过 `0.60`。

Smoke 检查：

```bash
python scripts/analysis/check_pcd_variant_smoke.py \
  --variant deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay \
  --log-dir outputs/test-fast/logs/pcd_no_visual_pcd_oracle_late_traj_decay_smoke10/deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay
```

如果这个方向有明显训练健康收益但 eval 不够，再进入第二版：all-wrong OPD cap；不要和第一版一次合并。

### 3. Mixed-Prompt Dynamic Resampling

训练日志里 all-wrong 后期仍超过一半。若每步大量 prompt 不能产生 mixed/correct group，GRPO 有效样本效率不够。

建议新 variant：

`deplot_no_vs_opd_pcd_oracle_hint_dynamic_mixed_sampling`

先不改变 loss，只改变 batch 选择：

- 每个 training step 过采样候选 prompt；
- 优先保留 reward std 非零或 group_mixed 的 prompt；
- 保留少量 all-wrong prompt 给 SFT/OPD repair，避免完全丢掉冷启动样本；
- 记录 `sampling/mixed_accept_rate`、`sampling/all_wrong_kept_rate`、`sampling/resample_attempts_mean`。

预期解决：单位训练步里的 GRPO 有效信号太少。

## Recommended Order

0. 先做 positive replay buffer 的训练接入，因为它是数据/训练分布层面的改动，直接针对“teacher-correct 很多但学生转化不足”的主瓶颈。当前导出数据已经生成：

```bash
outputs/test-fast/positive-replay-buffer/student_hint_short_full/replay_train.json
```

推荐第一个训练分支：

- `replay_sft_warmup -> oracle official`
- `replay_sft_warmup -> student_hint_short_opd_decay_effective_sampling`
- `replay_sft_warmup -> student_hint_short_opd_decay_sampling_rollout_replay_effective_filter`

不要和新的 prompt/reward 小补丁同时合并，先看 replay 是否能降低 early all-wrong 和后期 zero-signal。

一键 dry-run / smoke：

```bash
# 默认只打印三条 smoke 命令和检查命令
bash scripts/test/run_pcd_oracle_new_directions_smoke.sh \
  --dry-run \
  --run-id pcd_oracle_new_directions_smoke10

# GPU 空闲后顺序跑三条 10-step smoke，并自动检查关键 metric
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/test/run_pcd_oracle_new_directions_smoke.sh \
  --run \
  --run-id pcd_oracle_new_directions_smoke10
```

1. 若 replay 接入前需要一个低风险训练 baseline，跑 `student_hint_short_opd_decay_effective_sampling`，它是当前已经实现的最大在线训练改动。
2. `eval_format_reward` 和 `late_traj_decay` 降级为辅助方向：只有 replay / effective sampling 的训练健康改善但 eval 仍被格式污染时再跑。
3. 若辅助方向 smoke 都健康，再跑组合 variant：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DYME_PCD_RUN_ID=pcd_oracle_eval_format_late_traj_decay_smoke10 \
DYME_PCD_MAX_STEPS=10 \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DYME_PCD_RUN_ID=pcd_oracle_eval_format_late_traj_decay_4epoch \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay
```

Smoke 检查：

```bash
python scripts/analysis/check_pcd_variant_smoke.py \
  --variant deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay \
  --log-dir outputs/test-fast/logs/pcd_no_visual_pcd_oracle_eval_format_late_traj_decay_smoke10/deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay
```

4. 旧 `dynamic_mixed_sampling` 设想已由 `student_hint_short_opd_decay_effective_sampling` 替代；后续优先比较 effective sampling 与 positive replay 的组合，而不是再做一个独立 sampler。

## Effective Group Filter: Training-Step 内过滤零信号 group

`deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter`

这个 variant 不是再改变 prompt 采样分布，而是在每个已经生成出来的训练 batch 内直接过滤后期零信号 rows：

- 继承 `student_hint_short` teacher-SFT repair target；
- 继承 OPD/teacher trajectory 后期 decay 与 OPD per-prompt cap；
- 继承 effective sampling、positive replay、rollout replay；
- 从 `global_step >= 294` 开始启用 `DYME_EFFECTIVE_GROUP_FILTER=1`；
- all-wrong group 每个 prompt 只保留 `1` 个 repair row，其余 row 的 mask/advantage/OPD/SFT route 清零；
- all-correct group 默认过滤，因为组内 advantage 近似零；
- mixed group 完整保留，让 correct completion 的 GRPO 信号和 mixed wrong 的 DyME/OPD 继续存在。

它要验证的问题更大：不是“teacher 能不能给正确轨迹”，而是“后期训练步是否仍被 all-wrong/zero-signal rows 消耗”。如果这个方向有效，预期现象不是 teacher-correct 继续升，而是：

- `filter/effective_group_filtered_rate` 在 step 294 后变成非零；
- `filter/effective_group_all_wrong_filtered_rate` 明显非零；
- `routing/grpo_route_rate` 不再被 all-wrong rows 稀释；
- `signal/grpo_zero_loss_rate` 相比 `student_hint_short` 后期下降；
- final 或 best checkpoint 至少超过 `student_hint_short` 的 `0.5800`，理想上超过 oracle official `0.5872`。

Dry-run：

```bash
DYME_PCD_RUN_ID=pcd_oracle_effective_filter_dryrun \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --dry-run \
  --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter
```

10-step smoke：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
DYME_PCD_RUN_ID=pcd_oracle_effective_filter_smoke10 \
DYME_PCD_MAX_STEPS=10 \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter
```

4epoch：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
DYME_PCD_RUN_ID=pcd_oracle_effective_filter_4epoch \
bash scripts/test/run_pcd_no_visual_4epoch.sh \
  --variant deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter
```

## RL Transition: Warmup 后切掉 teacher/OPD 依赖

`deplot_no_vs_opd_pcd_oracle_hint_student_hint_short_opd_decay_sampling_rollout_replay_effective_filter_rl_transition`

这是比 `effective_filter` 更激进的主实验。它假设 replay warmup 和前期 teacher repair 已经完成冷启动，后半程不再让 all-wrong repair/OPD 继续吞训练预算：

- `DYME_POSITIVE_REPLAY=0`：DyME 阶段默认关闭 online positive replay；静态 replay 只在 Stage 1 warmup 做，避免第二阶段继续引入 CE 前向显存成本和静态 SFT 信号；
- `DYME_EFFECTIVE_GROUP_FILTER_AFTER_STEP=294`；
- `DYME_EFFECTIVE_GROUP_FILTER_ALL_WRONG_KEEP=0`：step 294 后 all-wrong group 不再保留 repair row；
- `DYME_OPSD_FINAL_WEIGHT=0.0`：OPD 在 decay window 后退到 0；
- mixed group 仍完整保留，正确 completion 继续走 GRPO，mixed wrong 在 decay window 内仍可走 DyME/OPD。

这条线要验证的是：`student_hint_short` 已经把 early all-wrong 从 `~0.77` 拉到 `~0.55`，但后期 GRPO 仍不够主导；如果 replay warmup 能进一步提高 early mixed 比例，那么后期应果断切掉 teacher/SFT 修复，让训练预算集中到 GRPO/mixed rows。

推荐直接用两阶段 runner：

```bash
# dry-run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
DYME_CHAIN_RUN_ID=pcd_replay_warmup_rl_transition_4epoch \
bash scripts/test/run_replay_warmup_then_pcd_4epoch.sh --dry-run

# 4epoch
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
DYME_CHAIN_RUN_ID=pcd_replay_warmup_rl_transition_4epoch \
DYME_REPLAY_SFT_EPOCHS=0.5 \
bash scripts/test/run_replay_warmup_then_pcd_4epoch.sh
```

判断标准：

- Stage 1 warmup：positive replay SFT 能正常产出 warmup checkpoint，并作为 `DYME_STUDENT_MODEL` 传入 DyME；
- DyME step 294 前：teacher repair 能继续冷启动，但 `replay/positive_*` 应为 0 或 unavailable；
- step 294 后：`filter/effective_group_all_wrong_filtered_rate` 升高，`filter/effective_group_kept_all_wrong_rate` 接近 0；
- step 441 后：`loss/opsd_effective_weight` 接近 0；
- last50 `routing/grpo_route_rate` 明显高于 `student_hint_short` 的 `~0.23`；
- final 或 best checkpoint 高于 `0.5800`，理想上超过 oracle official `0.5872`。

不要把三个方向一次性合并。每个方向先跑 10-step、1epoch，再跑 4epoch final + checkpoint sweep。

4epoch 后统一 eval：

```bash
# 默认只打印每个 checkpoint 的 eval 命令
bash scripts/test/eval_pcd_oracle_new_directions.sh \
  --dry-run \
  --run-id pcd_oracle_new_directions_4epoch

# GPU 空闲后评估三条新方向的 checkpoint-147/294/441/588/final_checkpoint
CUDA_VISIBLE_DEVICES=0 \
DYME_EVAL_BATCH_SIZE=1 \
bash scripts/test/eval_pcd_oracle_new_directions.sh \
  --run \
  --run-id pcd_oracle_new_directions_4epoch
```

每个 variant 会写入：

- `eval_chartqa/eval_<checkpoint>_<timestamp>.log`
- `eval_chartqa/summary.csv`
