# Sub-Billion VLM On-Policy Distillation 实验计划

更新时间：2026-07-13

设计文档：`docs/superpowers/specs/2026-07-12-closed-loop-recoverability-curriculum-paper-design.md`

统一账本：`docs/paper_reconstruction/experiment_ledger.md`

## 1. 成功标准

### 训练方法目标

- 统一 4epoch 预算；
- gold-hidden teacher 主设置 `teacher/privileged_suffix_has_gold_rate=0.0`，routing verifier 仍可使用 reference；
- final ChartQA accuracy 目标 `>0.60`；
- 至少优于统一预算 DyME/gold-hidden fixed schedule；
- 报告 teacher calls、generated tokens 和 GPU hours。

### 论文目标

- 第一优先级证明 OPD 相对 no-OPD 的净收益；
- 第二优先级证明 OPD 与 GRPO、fallback supervision 的互补性；
- verifier routing 与 global controller 作为使 OPD 可靠介入和退出的支撑机制分别消融；
- oracle upper bound 与 gold-hidden-teacher 主结果分栏，并分别披露 verifier reference access。

## 2. 当前实验状态

### 正在运行：Oracle CLRC Upper Bound

```text
run id: global_grpo_route_full_4epoch_20260712_205549
variant: deplot_no_vs_opd_pcd_oracle_hint_full_cot_adaptive_supervision
steps: 592
train tmux: dyme_grpo_route_full_205549
monitor tmux: dyme_grpo_route_monitor_205549
```

该 run 使用 `oracle_hint`，不能作为 gold-hidden-teacher 主结果。训练成功后自动执行 8-GPU final eval。

### 正在运行：Oracle OPD Without Hard Imitation

```text
run id: oracle_opd_no_hard_imitation_adaptive_4epoch_20260713_121946
variant: deplot_no_vs_opd_pcd_oracle_hint_opd_no_hard_imitation_adaptive_supervision
train tmux: dyme_opd_no_hard_full_121946
monitor tmux: dyme_opd_no_hard_monitor_121946
```

该 run 是对 `0.5120` full-trajectory 负结果的单因素修正：关闭 teacher trajectory
与 teacher-SFT repair，保留 verifier-routed OPD、GRPO/fallback、effective sampling
和同一 adaptive controller。在线监控完整五段格式、空模板骨架、异常 answer section，
并对 hard-imitation signal 采用零容忍。

## 3. P0：必须完成的证据链

| ID | Experiment | Base config | Unique change | Budget | Health gate | Paper claim |
|---|---|---|---|---|---|---|
| P0-E0 | Oracle OPD upper bound | current oracle full-CoT recipe | verifier-routed OPD + adaptive support | 4epoch | last50 routes、all-wrong、clip/degenerate | OPD 在 privileged evidence 下的性能上界 |
| P0-E1 | Gold-hidden no-OPD | matched base recipe | 禁用 OPD，保留相同 GRPO/fallback、采样和预算 | 4epoch | teacher gold-rate=0 | OPD 净收益的直接对照 |
| P0-E2 | Gold-hidden unconditional OPD | P0-E1 | 所有 eligible wrong completions 使用 OPD，无 verifier routing/controller | 4epoch | OPD coverage、teacher tokens | OPD 本身有效，但无条件使用可能不可靠 |
| P0-E3 | Gold-hidden verifier-routed OPD | P0-E2 | 仅 verifier-confirmed wrong completions 使用 OPD，fixed support | 4epoch | teacher precision/coverage funnel | 可靠路由对 OPD 的贡献 |
| P0-E4 | Gold-hidden full method | P0-E3 | 加入 realized global GRPO adaptive support | 4epoch | gold-rate=0；controller metrics complete | OPD 主方法与最强效果 |
| P0-E5 | Signal complementarity | matched P0-E4 | OPD-only、GRPO-only、fallback-only、OPD+GRPO、完整三路 | 4epoch 或先 matched-budget screening、关键行 full | final acc、zero-loss、route occupancy | OPD 不能被 GRPO 或 fallback 替代 |
| P0-E6 | Token-selective OPD baseline | P0-E3 | completion routes 固定，仅将 uniform-token OPD 替换为 token teachability/position-reliability weighting | 4epoch | accepted token coverage、teacher tokens、final acc | 区分 completion routing 与 token reliability 的贡献 |

### P0-E6 实现候选：Non-Answer Heading-Selective OPD

该候选只在当前 no-hard-imitation run 的 final forensic 证明 uniform-token OPD 与
模板泛化失败相关时启用。它不是对 `Goal/Observation/Reasoning/Conclusion` 输出的
显式惩罚，也不改变 generation reward：

1. 在每个 student-generated prefix 上计算原有 teacher/student divergence；
2. 若 teacher top token 属于非答案 section heading token 集合，则该位置的 OPD
   token weight 设为零或较小值；
3. `Answer` heading、答案内容、普通 reasoning token 与视觉/数值 token 保持原 OPD；
4. GRPO、fallback、teacher correctness gate、global controller 和 completion route
   全部保持不变；
5. 记录 `opd/non_answer_heading_mask_rate`、保留 token 比例与按 token 类别分解的 JSD。

这一设计检验 teacher distribution 中“格式先于答案”的局部可靠性，而不是假设
full-CoT 有害。matched baseline 必须保持相同 completion routes 和 OPD compute；若只
降低整体 OPD weight，不能把差异归因于 token selection。

执行顺序：P0-E0 完成并评估后，先运行 P0-E1 与 P0-E3，建立最关键的
OPD-vs-no-OPD 因果对照；若 OPD 有净收益，再补 P0-E2 和 P0-E5 解释可靠性与
互补性，并以 P0-E4 冲击最高分。P0-E6 是 novelty-critical near-neighbor baseline。
controller state/action 消融只有在 OPD 主效应成立后才占用完整 4epoch 预算。

## 4. P1：OPD 支撑机制、效率与稳健性

| ID | Experiment | Unique change | Budget | Required output |
|---|---|---|---|---|
| P1-E1 | OPD weight only | hard trajectory 固定为零，controller 不改变 cap | 4epoch | final acc + route dynamics |
| P1-E2 | teacher cap only | weights fixed，仅 cap `8->2` | 4epoch | teacher tokens + final acc |
| P1-E3 | joint OPD actions | hard trajectory 固定为零，联合控制 OPD weight 与 cap | 4epoch | Pareto point |
| P1-E4 | changed effective batch | batch/accumulation 改变 | matched 4epoch | action trigger autonomy state |
| P1-E5 | changed data scale | train subset/full scale | matched updates | fixed-step vs CLRC robustness |
| P1-E6 | non-monotonic hysteresis | mastery 可回退但带滞回 | matched diagnostic/full if promising | regression recovery |
| P1-E7 | controller state | mixed/zero proxy vs global GRPO route | matched diagnostic + promising rows full | state-variable contribution |

## 5. P2：泛化和行为分析

- 另一个可验证视觉推理任务；
- teacher evidence source 消融：format only、DePlot、visual facts；
- full reasoning 与 concise reasoning 的 accuracy-conditioned 分析；
- recoverable/unrecoverable qualitative cases。

P2 不应在 ChartQA 主效果和 P0 因果链完成前占用主要训练预算。

## 6. 运行门槛

### Smoke

- 8-GPU 4 steps；
- 所有 rank 正常退出；
- `global_signal/*` 与 `adaptive/signal_*` 存在；
- step `t` snapshot 不影响同一步；
- 无 OOM/NCCL/NaN。

### Early window

早期 clip collapse 不能单独停止，因为历史 `0.5800` run 在 step 8–20 也出现该现象。至少比较：

- steps 21–30；
- steps 31–40；
- steps 41–50；
- steps 51–70。

若到 step 70 仍同时满足 accuracy≈0、global GRPO≈0、degenerate>0.9、clip>0.9，且明显差于历史同窗口，则停止并研究。

### Full-run health

目标窗口：

- last50 global GRPO route `>0.30`；
- last50 SFT route `<0.30`；
- task all-wrong 明显低于早期；
- leakage 与 privileged tags 符合实验角色；
- controller actions 有实际变化但不在低 autonomy 时提前衰减。

这些是诊断目标，不替代 final eval。

## 7. Eval 协议

训练完成后优先 8-GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DYME_EVAL_BATCH_SIZE=1 \
python -m accelerate.commands.launch \
  --config_file scripts/test/accelerate_single_gpu_no_deepspeed.yaml \
  --num_processes 8 \
  -m eval.eval_chartqa \
  --model_path <run>/<variant>/final_checkpoint
```

快速迭代可接受 `2496/2500`，但正式主表优先补齐 `2500/2500`。OOM/traceback 行不得作为结果。

Final eval 同时输出 `Template behavior counts`：full/partial CoT template、
Goal-without-Answer、empty skeleton 与 malformed Answer。`summary.csv` 保留该字段，
用于区分有内容的 full-CoT 与固定模板污染；这些诊断不改变答案抽取或 accuracy scoring。

## 8. 低于 0.60 的迭代规则

1. 先做训练/评估 failure taxonomy，不同时改变多个机制。
2. 若 global GRPO 长期为 0，优先研究 generation/LR/reward task signal，而不是继续衰减 teacher。
3. 若 GRPO 上升但 eval 不升，检查格式模板、answer parse、teacher target 和 train/eval mismatch。
4. 若 oracle CLRC 超过 baseline 而 gold-hidden-teacher 设置不升，优先提升 evidence/recoverability estimator，而不是宣称 controller 已解决问题。
5. 每轮先 TDD、4-step smoke、健康窗口，再放行 4epoch。

### 8.1 下一轮只允许由证据选择一个主干预

| Final forensic | Primary intervention | Changes explicitly held fixed |
|---|---|---|
| global GRPO coverage 低且 teacher-correct 高 | 改善学生探索/任务梯度，例如 generation entropy、LR 或 advantage construction | recoverability probe、target format、controller mapping |
| global GRPO coverage 上升但 eval 不升 | 检查 answer extraction、reasoning/answer target mismatch 和错误模板条件概率 | controller state 与 teacher budget |
| oracle CLRC 有效、gold-hidden teacher precision/coverage 低 | 改进 gold-hidden visual evidence 与 recoverability quality gate | global controller、optimization recipe |
| teacher precision 高但 OPD route 无收益 | 检查 token-position reliability、teacher/student state mismatch 与 OPD divergence | evidence source、GRPO objective |
| partial Goal-without-Answer drift 高、但完整模板尚未塌缩 | 比较 token-selective/answer-bearing-position OPD 与统一 token OPD；优先降低结构 token 权重而非整体关闭 OPD | teacher correctness gate、GRPO/fallback、controller state |
| 低 GRPO 与最大 OPD support 长期共存 | 检查是否为 OPD token reliability 导致的正反馈；保持单一 controller signal，先增加局部 token gate | controller EMA/target、teacher budget、GRPO objective |
| controller 很少产生动作变化 | 调整 target/calibration 或采用带滞回的可逆控制 | local router 与 teacher target |

每次完整 4epoch run 只选择表中一行作为主因果干预；其余变化仅限修复已验证的实现错误。这样才能把超过 `0.60` 的 recipe 转化为可归因的论文证据。

### 8.2 OPD 主张准入门槛

只有同时满足以下条件，论文才能把 OPD 写成已验证的核心创新：

1. matched 4epoch 的 verifier-routed OPD 明确优于 no-OPD；
2. unconditional OPD 不优于或弱于 verifier-routed OPD，证明收益不只是额外 teacher compute；
3. 完整联合训练优于 OPD-only、GRPO-only 与 fallback-only 中的最强者，支持互补性；
4. 效果至少不低于统一预算 DyME，并以有效 ChartQA `summary.csv` 为准；
5. 所有比较披露 teacher input gold access、routing verifier reference access 和 teacher tokens。

## 9. 论文硬约束

- gold-hidden-teacher effectiveness 只能由 teacher prompt gold-rate=0 的完整 run 支持；这不等于 routing verifier 不使用 reference；
- 主表必须分别披露 `Teacher sees gold` 和 `Verifier uses reference`；
- `>0.60` 只能由有效 eval summary 支持；
- full-CoT 不按比例直接惩罚；
- 所有主表行披露 Teacher sees gold、Verifier uses reference、Evidence、Epoch、Processed；
- 同一消融只改变一个因素。
