# CLRC / OPD 主训练与 Eval 事实记录

更新时间：2026-07-13

本文件只记录已完成 artifact 与正在运行任务的事实，不承担论文 claim 推导。统一实验角色见
`docs/paper_reconstruction/experiment_ledger.md`。

## 1. 4epoch ChartQA 已验证结果

| Run role | Teacher sees gold | Verifier uses reference | Accuracy | Processed | Output types | Artifact |
|---|---|---|---:|---:|---|---|
| gold-hidden-teacher PCD aligned | no | yes | 0.5420 | 2500/2500 | other=2285, full_cot=184, answer_flag=31 | legacy run ID `pcd_no_visual_aligned_4epoch/.../eval_chartqa/summary.csv` |
| oracle route_guard | yes | yes | 0.5592 | 2500/2500 | answer_flag=1308, full_cot=1143, other=49 | `route_guard_oracle_hint_4epoch_7gpu/.../eval_chartqa/summary.csv` |
| oracle full-template repair | yes | yes | 0.5624 | 2500/2500 | answer_flag=1375, full_cot=989, other=136 | corresponding `summary.csv` |
| oracle constrained repair | yes | yes | 0.5656 | 2500/2500 | answer_flag=1390, full_cot=1109, other=1 | corresponding `summary.csv` |
| oracle student_hint_short | yes | yes | 0.5800 | 2500/2500 | other=1629, answer_flag=821, full_cot=50 | valid row `eval_final_checkpoint_bsz1_gpu0_20260709_192652` |
| oracle official | yes | yes | 0.5872 | 2500/2500 | see eval log | checkpoint-588 and final rows in official `summary.csv` |

当前事实：`student_hint_short` 比 constrained repair 高 1.44 points，但仍比 oracle official 低 0.72 points。该对比均为 oracle/privileged 设置，不能代表 gold-hidden-teacher CLRC 主方法。

当前 no-hard-imitation OPD run 在 steps 41--50 的 matched window 中，hard trajectory
与 teacher-SFT repair 均保持为零，但 wrong probed completions 仍呈近乎全量的 partial
`Goal`-without-`Answer` drift。该现象说明关闭 hard sequence target 不能自动消除 OPD
soft token matching 带来的 teacher-style transfer。窗口 accuracy/GRPO 为
`0.0102/0.0000`，degenerate/clip/EOS 为 `0.8438/0.5992/0.4094`。steps 51--60
进一步恶化为 accuracy/GRPO `0.0070/0.0156`、degenerate `0.9156`，未达到预设恢复
闸门，训练已停止且不做 final eval。该结果不能作为 4epoch accuracy，但可作为
“只关闭 teacher-generated hard sequence imitation 仍不足”的机制负证据。代码审计显示该
variant 仍保留 legacy online SFT：目标为 dataset `hint + answer`，而 ChartQA hint 是完整
`Goal/Observation/Reasoning/Conclusion`；warmup 默认每组 4/8 个 SFT slots，实测 SFT route
约 `0.49--0.53`。因此当前 collapse 不能单独归因于 OPD，下一轮必须先关闭这条 full-hint
硬监督通路。

## 2. Student-Hint-Short 训练健康

有效 final eval：

```text
outputs/test-fast/pcd-no-visual/
pcd_oracle_teacher_sft_repair_student_hint_short_4epoch/
deplot_no_vs_opd_pcd_oracle_hint_teacher_sft_repair_student_hint_short/
eval_chartqa/summary.csv
```

last50：

- `group_all_wrong_rate = 0.545`；
- `routing/grpo_route_rate = 0.2325`；
- `routing/opd_route_rate = 0.6319`；
- `rewards/accuracy/mean`: first50 `0.0120` -> last50 `0.2712`；
- `completions/degenerate_rate`: `0.5331` -> `0.1006`；
- `clipped_ratio`: `0.7278` -> `0.1669`；
- `signal/grpo_zero_loss_rate = 0.75`；
- `teacher_sft_privileged_tag_rate = 0.0`。

Teacher probe candidate funnel：

- candidates `125451`；
- teacher-correct `92.48%`；
- parse-fail `0`；
- all-wrong teacher-correct `91.99%`；
- mixed-wrong teacher-correct `94.79%`。

## 3. Global Training Signal 与 Controller 验证

已实现并验证：

- 跨 rank 最终 route counts 单次归约；
- task-accuracy zero 与 total-reward zero 分开记录；
- global/local zero-signal disagreement；
- realized global GRPO route 单信号 EMA；
- monotonic mastery；
- step `t` snapshot 控制 step `t+1`；
- OPD weight、trajectory weight 和 cap 使用同一状态。

Focused regression suite：`66 passed`。

4-step 8-GPU smoke：

```text
run: global_grpo_route_smoke_20260712_200239
final checkpoint: outputs/test-fast/pcd-no-visual/
  global_grpo_route_smoke_20260712_200239/
  deplot_no_vs_opd_pcd_oracle_hint_full_cot_adaptive_supervision/final_checkpoint
```

| step | global GRPO rate | signal EMA | supervision | OPD cap |
|---:|---:|---:|---:|---:|
| 1 | 0.003906 | 0.000391 | 0.999995 | 8 |
| 2 | 0.019531 | 0.002305 | 0.999824 | 8 |
| 3 | 0.019531 | 0.004027 | 0.999464 | 8 |
| 4 | 0.003906 | 0.004015 | 0.999464 | 8 |

该 smoke 证明接线与控制动作正确，不证明 final accuracy。

## 4. 当前 Oracle CLRC 4epoch Run

```text
run id: global_grpo_route_full_4epoch_20260712_205549
variant: deplot_no_vs_opd_pcd_oracle_hint_full_cot_adaptive_supervision
total steps: 592
train tmux: dyme_grpo_route_full_205549
monitor tmux: dyme_grpo_route_monitor_205549
```

状态目录：

```text
outputs/test-fast/long-runs/global_grpo_route_full_4epoch_20260712_205549/
```

训练成功后，同一 pipeline 自动执行 8-GPU ChartQA final eval，batch size 为 1，并解析 `eval_chartqa/summary.csv`。

实验角色说明：该 variant 使用 `oracle_hint`，预期
`teacher/privileged_suffix_has_gold_rate=1.0`，因此只作为 oracle CLRC upper bound。

### Early-window 事实

step 11–20：

- global GRPO route `0.0`；
- accuracy `0.0`；
- clipped `0.9996`；
- degenerate `0.9844`；
- controller supervision 约 `0.9988`。

历史 `0.5800` run 在同一阶段也经历严重 clip collapse，并在 step 21–70 逐步恢复。因此早期 collapse 不单独触发停止；预设决策窗口为 step 50/70。

step 41–50 同步对照：

| Metric | Oracle CLRC current | Historical 0.5800 run |
|---|---:|---:|
| accuracy | 0.0051 | 0.0188 |
| clipped | 0.2856 | 0.5258 |
| EOS | 0.7063 | 0.4719 |
| degenerate | 0.4813 | 0.3563 |
| local GRPO route | 0.0031 | 0.0063 |

当前 run 的序列终止恢复更快，但任务 accuracy 与 GRPO coverage 仍弱于历史对照。global GRPO EMA 低，controller 继续保持约 `0.9988` supervision，符合低自主覆盖率下不提前撤走 teacher support 的设计。下一决策点为 step 70。

step 51–70 同步对照：

| Metric | Oracle CLRC current | Historical 0.5800 run |
|---|---:|---:|
| accuracy | 0.0342 | 0.0797 |
| global/local GRPO route | 0.0346 | 0.0563 |
| OPD route | 0.4766 | 0.4813 |
| SFT route | 0.4889 | 0.4625 |
| GRPO zero-loss | 0.8250 | 0.8500 |
| clipped | 0.3762 | 0.3674 |
| degenerate | 0.1734 | 0.1391 |
| EOS | 0.6188 | 0.6516 |

到 step 70，当前 run 已从早期 clip/degenerate collapse 中恢复，因而不满足预设的灾难性早停条件；但 accuracy 仅约为历史最优 run 同窗口的 `43%`，GRPO coverage 也低约 `2.2` 个百分点。controller 的 GRPO EMA 为 `0.0235`、mastery 为 `0.0246`、supervision 约为 `0.9800`，说明闭环没有在学生尚弱时错误撤掉 teacher support。当前判断是继续运行并在 step 100 再比较恢复斜率，而不是把“控制器按设计保留监督”误写成 effectiveness 已得到验证。

### Step 71–90：恢复斜率反转

step 71–80：

| Metric | Oracle CLRC current | Historical 0.5800 run |
|---|---:|---:|
| accuracy | 0.0770 | 0.0707 |
| global/local GRPO route | 0.0770 | 0.0781 |

### Step 101–117：短期低信号回落，暂不触发重启

step 91–100 是当前 run 迄今最强的十步窗口：accuracy 与 global GRPO route
均约为 `0.119`，all-wrong 为 `0.750`，GRPO zero-loss 降至 `0.50`。随后
step 101–117 出现新的困难样本段：

| Metric | Oracle CLRC current | Historical comparison run |
|---|---:|---:|
| accuracy / global GRPO route | 0.0572 | 0.0593 |
| GRPO zero-loss | 1.0000 | 0.9412 |
| group all-wrong | 0.8529 | 0.8529 |
| clipped | 0.2452 | 0.3796 |
| degenerate | 0.0478 | 0.0846 |
| EOS | 0.7629 | 0.6195 |

该窗口的任务信号明显弱于 step 91–100，但 accuracy 尚未低于历史轨迹，且
clipped、degenerate、EOS 三项生成健康指标均显著优于历史比较。因此当前证据更符合
数据顺序引起的短期低信号段，而不是模型或训练流程崩坏。controller 的 mastery
保持约 `0.108`，supervision 保持约 `0.704`；单调 mastery 没有因短期坏 batch
重新提高 teacher support，这是设计预期，也暴露了后续需要验证的恢复性边界。

当前决策是继续 4epoch run。只有后续连续至少 30 steps 同时满足以下条件才触发
停止并研究重启：accuracy/GRPO route 显著低于历史配对窗口、zero-loss 持续接近
`1.0`，并且 clipped/degenerate/EOS 至少两项同步恶化。单个或十余个 all-wrong
密集窗口不单独作为停止依据。
| OPD route | 0.4828 | 0.4719 |
| SFT route | 0.4402 | 0.4500 |
| GRPO zero-loss | 0.9000 | 0.8500 |
| all-wrong | 0.7250 | 0.8250 |
| degenerate | 0.0531 | 0.1969 |
| EOS | 0.8219 | 0.6031 |

step 81–90：

| Metric | Oracle CLRC current | Historical 0.5800 run |
|---|---:|---:|
| accuracy | 0.0824 | 0.0637 |
| global/local GRPO route | 0.0832 | 0.0250 |
| OPD route | 0.4492 | 0.4688 |
| SFT route | 0.4672 | 0.5000 |
| GRPO zero-loss | 0.9500 | 1.0000 |
| all-wrong | 0.8250 | 0.9500 |
| mixed | 0.1750 | 0.0500 |
| clipped | 0.2418 | 0.4543 |
| degenerate | 0.0785 | 0.0188 |
| EOS | 0.7605 | 0.5938 |

当前 run 在 step 51–70 一度明显落后，但 step 71–90 的 accuracy 和 realized GRPO coverage 已追平并超过历史 `0.5800` run 同窗口，all-wrong 和 clip 也更低。controller 的 signal EMA 从 step 71–80 平均 `0.0410` 上升到 step 81–90 的 `0.0759`；mastery 从 `0.0429` 上升到 `0.0773`，supervision 从 `0.9416` 降至 `0.8350`。这说明 controller 已开始响应学生真实进入 GRPO 的 coverage，而非固定 step 触发。

该反转支持继续跑满 4epoch，但仍不证明 final accuracy 会超过 `0.5800/0.5872`：step 81–90 的 zero-loss 仍高达 `0.9500`，且 controller 撤除 support 后是否保持恢复趋势，需要 step 100、后续 epoch 和 final eval 共同判断。

### Step 100 Health Gate

step 91–100：

| Metric | Oracle CLRC current | Historical 0.5800 run |
|---|---:|---:|
| accuracy | 0.1191 | 0.0793 |
| global/local GRPO route | 0.1195 | 0.1156 |
| OPD route | 0.4570 | 0.4813 |
| SFT route | 0.4168 | 0.4000 |
| GRPO zero-loss | 0.5000 | 1.0000 |
| all-wrong | 0.7500 | 0.7500 |
| mixed | 0.2500 | 0.2500 |
| clipped | 0.2000 | 0.3602 |
| degenerate | 0.0344 | 0.0594 |
| EOS | 0.8016 | 0.6844 |

累计 step 71–100：当前 accuracy `0.0928` vs historical `0.0712`，GRPO route `0.0932` vs `0.0729`，zero-loss `0.7833` vs `0.9500`，all-wrong `0.7667` vs `0.8417`。因此从 step 71 开始的恢复不仅持续，而且在任务信号、有效 route coverage 和生成健康上共同优于历史同窗。

controller 在 step 91–100 的平均 signal EMA/mastery/support 为 `0.0990/0.1004/0.7380`。step 101 snapshot 为 signal EMA `0.0985`、mastery `0.1077`、support `0.7061`、OPD weight `1.2061`、trajectory weight `0.3530`、cap `7`。support 撤除后，91–100 accuracy 与 GRPO route 没有下降，说明截至该窗口尚未出现 controller 撤得过快的证据。

Step-100 gate 结论：**继续跑满 4epoch**。该 run 无 OOM/NCCL/NaN，未出现 generation collapse；近期恢复曲线优于历史最强内部 run。剩余主要风险是该优势能否在后续 support 进一步下降时保持，以及 training-window 改善能否转化为 held-out ChartQA final accuracy。

## 5. 历史 10epoch OPD 结果（归档，不作为当前主预算）

历史日志：

```text
outputs/test-fast/logs/train_opd_7b_dyme_probe_20260622_101112.log
```

该 run 有 1470 metric rows。不同 epoch checkpoint 的完整 ChartQA eval 中，最优为 checkpoint-1176：`0.5224`；final 为 `0.5216`。这些结果说明旧 recipe 的更长训练没有自动带来更高 held-out accuracy，因此当前论文统一使用 4epoch 方法比较。

## 6. 结果更新规则

1. final accuracy 只从有效 eval summary 读取。
2. OOM、traceback 或空 accuracy 行不进入比较。
3. oracle 与 gold-hidden-teacher 分栏，并单列 verifier reference access。
4. training reward、teacher candidate accuracy 和 final ChartQA accuracy 不互相替代。
5. 当前 run 完成后，同步更新本文件、experiment ledger 和 claim-evidence matrix。

## 5. Oracle Full-Trajectory Adaptive Run Final Eval

Run `global_grpo_route_full_4epoch_20260712_205549` 已完整训练并由自动 pipeline
完成 8-GPU ChartQA final eval：

- accuracy: `0.5120`;
- processed: `2500/2500`;
- output types: `full_cot=2397`, `answer_flag=103`;
- train last50 accuracy / global GRPO route: approximately `0.445 / 0.486`;
- train last50 clipped `0.0006`, EOS `0.9994`, degenerate `0.0088`.

训练 reward 与 route coverage 持续上升，但 held-out accuracy 反而显著低于 oracle official
`0.5872` 和 student-hint-short `0.5800`。eval 中 `Goal:` 出现 2415 次，并包含大量
空 `Observation/Reasoning/Conclusion` 与异常 `Answer:`。这说明失败不是“模型会写
full-CoT”本身，而是 hard teacher-trajectory/SFT imitation 使固定结构成为近乎强制的
输出先验，并造成 train-eval/style generalization gap。

下一轮只改变一个主因素：关闭 teacher trajectory 与 teacher-SFT repair，保留
verifier-routed OPD、GRPO、fallback、effective sampling 和同一 adaptive controller。
该实验更直接检验论文核心，即 student-state OPD 是否能在不复制 teacher hard
trajectory 的情况下提供净收益。

## 6. OPD Without Hard Teacher Imitation

The failed `0.5120` run revealed that teacher-probe correctness was coupled to the
presence of a stored teacher trajectory. When trajectory supervision was disabled,
correctly verified `MODE_OPSD` completions were mistakenly treated as unconfirmed and
overwritten by SFT. The routing source of truth is now the post-probe completion mode,
independent of optional trajectory payloads.

TDD and smoke evidence:

- routing regression test fails before and passes after the decoupling fix;
- relevant routing/runner suites pass;
- 4-step 8-GPU smoke reaches global OPD route `0.46--0.52`;
- teacher trajectory effective weight and teacher-SFT repair rate remain exactly `0`;
- no OOM, NCCL, or traceback;
- online template metrics and an external rolling-window checker are active;
- the checker rejects either `full-template > 0.8` with `empty-skeleton > 0.2`, or
  `full-template > 0.8` with `malformed-answer > 0.2`, across two consecutive windows;
- substantive full-CoT without either pathology remains allowed and is not treated as collapse.
- candidate JSONL monitoring additionally reports partial-template and
  Goal-without-Answer drift conditioned on wrong probed completions; this is a warning rather
  than a stop condition because the `0.5872` oracle baseline also shows a strong early spike.
- each rolling checker payload also contains accuracy, GRPO/OPD/SFT route occupancy,
  zero-loss, all-wrong, degenerate, clipped and EOS means for the same latest-20-step window.

The active 4epoch run is
`oracle_opd_no_hard_imitation_adaptive_4epoch_20260713_121946`. It isolates the paper's
core OPD signal while retaining matched GRPO, fallback, effective sampling, and adaptive
support. Training success automatically launches the 8-GPU final evaluation.

At step 10, both hard-imitation invariants and all three template-pathology rates remain
exactly `0`. The run has shown the same early clipping/degeneration burst seen in prior
recipes, so the established step-20/30/50 health gates remain in force; this startup
behavior is tracked separately from teacher-template collapse.

Step-20 update: hard trajectory and teacher-SFT repair remain exactly `0`, and the logged
full-template/empty-skeleton/malformed-answer rates remain `0`. However, direct inspection
of wrong probed completions shows a partial drift beginning near step 12: recent candidate
windows are almost entirely Goal-style headings without an Answer section. This statistic
is explicitly conditioned on wrong teacher-probe candidates. It is not by itself a stop
criterion because the completed `0.5872` oracle baseline also exhibited a strong early
candidate-only spike. The stronger health concern at step 20 is that task GRPO zero-loss,
degenerate rate, and clipped rate all remain `1.0`; the run continues into the predefined
steps 21--30 recovery gate.

Steps 21--30 matched comparison:

| Run | Accuracy | GRPO route | Zero-loss | Degenerate | Clipped | EOS |
|---|---:|---:|---:|---:|---:|---:|
| current no-hard-imitation OPD | 0.0004 | 0.0000 | 1.00 | 0.9563 | 0.9773 | 0.0187 |
| full-trajectory `0.5120` | 0.0004 | 0.0000 | 1.00 | 0.9750 | 0.9898 | 0.0094 |
| student-hint-short `0.5800` | 0.0020 | 0.0031 | 0.90 | 0.4188 | 0.8902 | 0.1187 |

The current run is marginally healthier than the failed full-trajectory run but recovers
far more slowly than the `0.5800` run. This is not yet a stop condition: the failed run
itself recovered EOS and generation health during steps 31--50, so those windows are needed
to determine whether removing hard imitation changes the later trajectory or only final
held-out style.

Steps 31--40 matched comparison:

| Run | Accuracy | GRPO route | Zero-loss | Degenerate | Clipped | EOS |
|---|---:|---:|---:|---:|---:|---:|
| current no-hard-imitation OPD | 0.0176 | 0.0187 | 0.70 | 0.6562 | 0.6277 | 0.4250 |
| full-trajectory `0.5120` | 0.0051 | 0.0031 | 0.70 | 0.5281 | 0.7012 | 0.3312 |
| student-hint-short `0.5800` | 0.0094 | 0.0094 | 1.00 | 0.3344 | 0.7887 | 0.1844 |

Removing hard imitation now produces the first clear positive divergence: current task
accuracy and realized GRPO coverage are highest, while clipping is lower and EOS higher
than both historical runs. Degenerate rate remains elevated because many wrong probed
completions still lack a valid Answer section. Continue through steps 41--50 and later
windows; no training change is justified while this recovery is strengthening.

## 7. Clean No-Full-Hint Hard-SFT OPD Run

The step-60 stop triggered a deeper source audit. Disabling teacher trajectory and
teacher-SFT repair was insufficient because ordinary online SFT still used the dataset
`hint + answer` target. The first distributed smoke of the clean variant found one more
legacy path: `_should_force_sft_replace` replaced malformed completions with the same full
hint, yielding a per-rank forced hard-target rate of `0.125`.

This bypass was handled with a RED-GREEN regression test. In the no-full-hint gate,
forced replacement is now disabled together with online-SFT slots, all-wrong online SFT,
and SFT fallback on teacher-probe failure. Legacy configurations retain their prior
behavior. The focused suite passes `164` tests.

The repeated 4-step, 8-GPU smoke completed with:

- `routing/legacy_online_sft_rate_max = 0.0`;
- `routing/full_hint_hard_target_rate_max = 0.0`;
- `routing/teacher_sft_repair_rate_max = 0.0`;
- `loss/teacher_traj_effective_weight_max = 0.0`;
- no OOM, NCCL error, traceback, or hanging rank;
- `951` candidate rows parsed by the external monitor with `status=ok`.

The clean 4epoch run is
`oracle_opd_no_full_hint_hard_sft_adaptive_4epoch_20260713_150545`. It launched at
`2026-07-13 15:05 CST`; accuracy is intentionally left unset until automatic 8-GPU final
evaluation writes a valid `eval_chartqa/summary.csv`.

Step-20 gate: all four hard-target invariants remain exactly zero, and neither training
outputs nor teacher-probe candidates show full/partial CoT template collapse. The latest
ten training rows have accuracy `0.0043`, realized GRPO route `0.0094`, OPD route `0.9906`,
zero-loss `1.0`, degenerate `0.8531`, clipped `0.8340`, and EOS `0.1563`. This indicates a
weak-autonomy, near-OPD-only phase. The run continues to step 40 because the historical
full-trajectory negative control was even less healthy in steps 21--30 and later recovered
generation health. If no recovery appears by the preregistered later gate, the next
single-factor intervention is a short answer anchor with continuous decay, not restoration
of dataset full-hint or teacher-trajectory hard supervision.

Step-40 update: the latest ten rows have accuracy `0.0160`, GRPO route `0.0063`, OPD route
`0.8938`, zero-loss `0.95`, degenerate `0.4250`, clipped `0.7129`, and EOS `0.2500`.
Compared with the historical `0.5800` run in steps 31--40, accuracy and clipping are better,
while realized GRPO coverage and degeneration are worse. The following step briefly reduces
zero-loss to `0.5`. This mixed recovery is sufficient to continue to the registered step-60
joint gate, but it does not yet show that pure OPD can replace a minimal cold-start anchor.

Step-60 recovery gate passes. The latest ten rows reach accuracy `0.0684`, OPD route
`0.8844`, GRPO route `0.0156`, degenerate `0.1219`, clipped `0.3891`, and EOS `0.6281`.
The accuracy is slightly above the historical `0.5800` run's matched `0.0625`, and
generation health has recovered strongly. However, task GRPO zero-loss remains `1.0`, so
the observed correct answers have not yet produced stable relative-advantage updates.
Training continues unchanged to step 100 to test whether realized GRPO coverage rises and
the controller begins to withdraw OPD support.

Steps 72--81 provide the first sustained autonomy recovery. Latest-ten accuracy is
`0.0836`, realized GRPO route `0.0906`, OPD route `0.7500`, degenerate `0.1406`, clipped
`0.2871`, and EOS `0.6656`. Controller mastery rises to `0.0842`, OPD weight falls to
`1.308`, and the route cap decreases from 8 to 7. These task and route values are stronger
than the historical `0.5800` run around steps 81--90, so the clean run remains eligible to
continue through step 100 and, absent later regression, the full 4epoch budget.

The run did not reach step 100. At step 86, rank 7 failed in the DeepSpeed backward
path with `CUDA error: unspecified launch failure`; the remaining ranks were terminated
by the elastic launcher. GPU 7 Inforom BBX records its latest event at
`2026-07-13 17:36:58`, aligned with the failure, while ECC counters remain zero and no GPU
reset is required. Because the old runner saved only at epoch boundaries, no checkpoint
or final evaluation exists. This interruption is classified as a hardware/runtime event,
not an unhealthy-training stop, and the partial trajectory is not an accuracy result.

The matched rerun therefore keeps the recipe fixed and changes only runtime resilience:
the no-full-hint variant saves every 50 steps with `save_total_limit=3`; legacy variants
retain epoch saves. `run_pcd_no_visual_resilient.sh` waits for all eight GPUs to remain
below the configured memory and temperature thresholds, resumes the latest checkpoint
after recognized CUDA/NCCL transient failures, and archives partial output before a clean
restart if a failure occurs before the first checkpoint.
