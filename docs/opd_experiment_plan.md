# Sub-Billion VLM On-Policy Distillation 实验计划

更新时间：2026-07-13

设计文档：`docs/superpowers/specs/2026-07-12-closed-loop-recoverability-curriculum-paper-design.md`

统一账本：`docs/paper_reconstruction/experiment_ledger.md`

## 1. 成功标准

### 训练方法目标

- 统一 4epoch 预算；
- gold-hidden teacher 主设置 `teacher/privileged_suffix_has_gold_rate=0.0`，routing verifier 仍可使用 reference；
- final ChartQA accuracy 的工程突破线为 `>0.60`；
- 外部 DyME 竞争线单独报告：其原文在 LLaVA-OV-S 0.5B、ChartQA relaxed
  correctness 上给出 Pure DyME `0.649`（Medium CoT）与 full DyME `0.675`
  （带 Visual Supervision）。因此 `0.60--0.649` 只能证明超过当前内部基线，不能写成
  “达到 DyME”；论文级 parity 还需要统一协议的 in-repo DyME reproduction；
- 至少优于统一预算 DyME/gold-hidden fixed schedule；
- 报告 teacher calls、generated tokens 和 GPU hours。

### 论文目标

- 第一优先级证明 OPD 相对 no-OPD 的净收益；
- 第二优先级证明 OPD 与 GRPO、fallback supervision 的互补性；
- verifier routing 与 global controller 作为使 OPD 可靠介入和退出的支撑机制分别消融；
- oracle upper bound 与 gold-hidden-teacher 主结果分栏，并分别披露 verifier reference access。

## 2. 当前实验状态

### DyME Comparator Implementation Audit

仓库目前没有可直接进入论文主表的统一协议 DyME artifact：

- `scripts/test/train_dyme.sh` 默认实际读取 `scripts/test/config/config.py`，而不是旧
  README 所写的 `config_dyme_deepspeed.py`；默认配置的 batch `2`、generation `8`、
  gradient accumulation `16` 更接近当前 CLRC，memory-tuned config 的 batch `1`、
  generation `4` 不能冒充 matched row；
- opsd-off 时 trainer 仍保留 DyME 的 dataset-hint SFT fallback，因此该 baseline 是动态
  SFT/GRPO，不是纯 GRPO；`deplot_no_vs_opd` 则仍开启 OPD，也不能作为 DyME；
- 当前脚本使用 shell 中的 bare `accelerate`、输出目录不带唯一 run ID、只按 epoch 保存，
  且没有自动 8-GPU final eval/summary parse；未激活 conda 的审计运行已以 rc `127`
  暴露 launcher 依赖，但没有启动 GPU 进程；
- matched Pure/Full DyME config/runner 已通过 TDD 实现：
  `scripts/test/config/config_dyme_matched.py` 提供统一预算的 Pure DyME，
  `scripts/test/config/config_dyme_full_matched.py` 只在其深拷贝上开启 Visual Checker、
  Visual Refiner 与 IC prefetch；两者共用
  `scripts/test/run_dyme_matched_4epoch.sh --variant pure|full`。runner 固定与 CLRC 相同的
  base model、Medium dataset、decode、batch/generation、optimizer、4epoch 和 50-step
  checkpoint，使用绝对 Python 路径调用 8-GPU Accelerate，并在 final checkpoint 后自动
  运行 ChartQA eval 和 summary parse。配置深比较证明 Full 相对 Pure 的唯一变化是
  Visual Supervision；两种 dry-run、Python compile、shell syntax 与控制器/路由/冻结配置
  组合回归均通过（`135 passed`），且 GPU PID 前后无变化。当前两者均标记为
  `ready/not queued`，没有启动 GPU，论文中必须作为两条独立 comparator 报告。

这项实现不改变当前冻结 gold-hidden CLRC run；按排程只在 gold-hidden CLRC 首次超过 `0.60` 后
进入 GPU 队列。

### 已完成：Oracle CLRC Upper Bound Negative Control

```text
run id: global_grpo_route_full_4epoch_20260712_205549
variant: deplot_no_vs_opd_pcd_oracle_hint_full_cot_adaptive_supervision
steps: 592
train tmux: dyme_grpo_route_full_205549
monitor tmux: dyme_grpo_route_monitor_205549
```

该 run 使用 `oracle_hint`，不能作为 gold-hidden-teacher 主结果。训练和 8-GPU final
eval 已完成，ChartQA accuracy 为 `0.5120`；它作为 OPD 与 full teacher trajectory
联合训练的负对照。

### 已停止：Oracle OPD Without Hard Imitation

```text
run id: oracle_opd_no_hard_imitation_adaptive_4epoch_20260713_121946
variant: deplot_no_vs_opd_pcd_oracle_hint_opd_no_hard_imitation_adaptive_supervision
train tmux: dyme_opd_no_hard_full_121946
monitor tmux: dyme_opd_no_hard_monitor_121946
```

该 run 是对 `0.5120` full-trajectory 负结果的单因素修正：关闭 teacher trajectory
与 teacher-SFT repair，保留 verifier-routed OPD、GRPO/fallback、effective sampling
和同一 adaptive controller。在线监控完整五段格式、空模板骨架、异常 answer section，
并对 hard-imitation signal 采用零容忍。它在 step 60 因恢复 gate 失败而停止；事后
确认仍有 legacy dataset-hint online SFT，因此不构成 clean OPD 结果。

### 硬件瞬态中止，待同配方重跑：Clean Oracle OPD Without Full-Hint Hard SFT

```text
run id: oracle_opd_no_full_hint_hard_sft_adaptive_4epoch_20260713_150545
variant: deplot_no_vs_opd_pcd_oracle_hint_opd_no_full_hint_hard_sft_adaptive_supervision
train tmux: dyme_no_full_hint_full_150545
monitor tmux: dyme_no_full_hint_full_150545_watch
```

该 run 同时关闭 teacher trajectory、teacher-SFT repair、online-SFT slots、all-wrong
SFT fallback 与 malformed-output forced SFT。四类 hard-target invariant 在已完成窗口
均严格为零。训练在 step 86 因 rank 7 `CUDA error: unspecified launch failure` 中止，
没有 checkpoint 和 final eval，因此不构成效果结果。GPU 7 Inforom BBX 的最新事件时间
为 `2026-07-13 17:36:58`，与训练崩溃对齐；ECC 错误为零且无需 reset。该配方保持不变，
等待资源满足门槛后从头重跑。

重跑默认每 50 step 保存一次并保留 3 个 checkpoint。启动器要求 8 卡连续 3 次满足
每卡显存占用不高于 7168 MiB、温度不高于 70 C、利用率不高于 10%；明确的
CUDA/NCCL 瞬态错误从最新 checkpoint 自动恢复，首个 checkpoint 前失败则先归档
不完整输出再干净重启。

`2026-07-13 19:29 CST` 的 gate 审计发现，仅用 `7168 MiB` 阈值会把占用
`5134 MiB` 的外部训练进程误判为空闲，从而重现同一卡双进程和 rank 7 崩溃风险。
resilient runner 现额外要求全局 compute-process 列表为空；测试先复现误判再修复，
相关 runner/gate 回归共 `50 passed`。当前 tmux 已在训练开始前热重载新 gate，方法配置、
run ID、checkpoint 和 eval 流程均未改变。

当前 leakage-fixed gold-hidden 主实验已于 `2026-07-14 23:58 CST` 启动：

```text
run id: clean_visualfact_fix_opd_main_4epoch_20260714_235824
variant: deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision
train tmux: dyme_gold_hidden_clrc_4e
monitor tmux: dyme_gold_hidden_clrc_4e_watch
automatic eval tmux: dyme_gold_hidden_clrc_4e_eval
post-eval forensic tmux: dyme_gold_hidden_clrc_4e_forensics
state: outputs/test-fast/long-runs/clean_visualfact_fix_opd_main_4epoch_20260714_235824
```

no-full-hint hard-SFT OPD spec 已于 `2026-07-13` 由用户批准，当前配方视为冻结基线。
运行时冻结配置检查 25/25 通过；截至 `2026-07-15 00:04 CST`，八卡训练已到 step 3/592，
hard-target、teacher trajectory 和 full/partial-template 暴露均为零。
watch 会话已启用一次性 `gate_20/40/60/100.json` 快照；step 60 只有在 latest-ten
同时满足 degenerate `>0.60`、accuracy `<0.02`、GRPO route `<0.02` 时才自动停止。
hard-target invariant 回流和“高 full-template + 空/异常 Answer”塌缩仍为即时停止条件。
训练/eval 管线生成 `summary.csv` 后，独立 forensic 会话自动运行
`scripts/analysis/pcd_low_score_forensics.py`，输出到 state 目录下的 `final_forensics/`；
若训练会话退出但没有 summary，则在 `status` 记录 `forensics_missing_summary_at`。

adaptive effective sampling 的配置审计已消除旧 `step=294` 泄漏：adaptive variant
显式导出 `after_step=0`、`start_progress=0.0`，trainer 仍以 `always_active` 保证运行时
不受遗留边界影响。legacy variant 的 step/progress schedule 保持不变；相关回归
`70 passed`，无需重启等待中的 resilient tmux，因为主 runner 只在 GPU gate 放行后读取。

## 3. P0：必须完成的证据链

| ID | Experiment | Base config | Unique change | Budget | Health gate | Paper claim |
|---|---|---|---|---|---|---|
| P0-E0 | Oracle OPD upper bound | current oracle full-CoT recipe | verifier-routed OPD + adaptive support | 4epoch | last50 routes、all-wrong、clip/degenerate | OPD 在 privileged evidence 下的性能上界 |
| P0-E1 | Gold-hidden route-matched no-OPD | matched base recipe | 保留 verifier call、completion route、GRPO/fallback、采样和预算，仅将 OPD loss weight 设为 `0` | 4epoch | teacher gold-rate=0；OPD route 非零但 OPD loss 为零 | OPD loss 净收益的直接对照 |
| P0-E2 | Gold-hidden unconditional OPD | P0-E1 | 所有 eligible wrong completions 使用 OPD，无 verifier routing/controller | 4epoch | OPD coverage、teacher tokens | OPD 本身有效，但无条件使用可能不可靠 |
| P0-E3 | Gold-hidden verifier-routed OPD | P0-E2 | 仅 verifier-confirmed wrong completions 使用 OPD，fixed support | 4epoch | teacher precision/coverage funnel | 可靠路由对 OPD 的贡献 |
| P0-E4 | Gold-hidden full method | P0-E3 | 加入 realized global GRPO adaptive OPD exposure | 4epoch | gold-rate=0；controller metrics complete | OPD 主方法与最强效果 |
| P0-E5 | Signal complementarity | matched P0-E4 | OPD-only、GRPO-only、fallback-only、OPD+GRPO、完整三路 | 4epoch 或先 matched-budget screening、关键行 full | final acc、zero-loss、route occupancy | OPD 不能被 GRPO 或 fallback 替代 |
| P0-E6 | Token-selective OPD baseline | P0-E3 | completion routes 固定，仅将 uniform-token OPD 替换为 token teachability/position-reliability weighting | 4epoch | accepted token coverage、teacher tokens、final acc | 区分 completion routing 与 token reliability 的贡献 |
| P0-E7 | VOLD-style cold-start + online OPD/GRPO near-neighbor | matched P0-E4 | 分配 matched cold-start alignment 预算，随后联合 OPD+GRPO；总 updates/teacher tokens 不变 | matched screening；最强近邻 4epoch | cold-start budget、global GRPO、teacher compute | 区分 all-wrong routed OPD 与 distribution-aligned VLM OPD transfer |
| P0-E8 | Visual-recoverability near-neighbor | matched P0-E3 | 用视觉 cue/privilege consistency gate 替换 answer-verifier completion gate | matched screening；若竞争则 4epoch | accepted precision/coverage、gold access、final acc | 区分 completion recoverability 与 ViCuR-style privilege filtering |
| P0-E9 | Mixed-group self-distillation near-neighbor | matched P0-E4 | mixed group 用 shortest-correct→longest-wrong self-distillation；all-wrong group 不调用 privileged teacher | matched screening；若竞争则 4epoch | mixed/all-wrong 分解 accuracy、zero-loss、teacher calls | 区分 all-wrong external recoverability 与 SSOPD-style intra-group supervision |

### P0-E6 实现候选：Answer-Bearing Token Reliability OPD

该候选只在当前 no-hard-imitation run 的 final forensic 证明 uniform-token OPD 与
模板泛化失败相关时启用。它不是对 `Goal/Observation/Reasoning/Conclusion` 输出的
显式惩罚，也不改变 generation reward：

1. 在每个 student-generated prefix 上计算原有 teacher/student divergence；
2. 根据 token 是否携带视觉实体、数值、运算关系或最终答案信息，给出连续 reliability
   weight；section heading 只作为诊断特征之一，不被硬禁止；
3. 普通 reasoning token 保留非零 OPD，答案、视觉和数值 token 获得更高权重；
4. GRPO、fallback、teacher correctness gate、global controller 和 completion route
   全部保持不变；
5. 记录 `opd/non_answer_heading_mask_rate`、保留 token 比例与按 token 类别分解的 JSD。

这一设计检验 teacher distribution 中“格式先于答案”的局部可靠性，而不是假设
full-CoT 有害，也不对 `Goal/Observation/Reasoning/Conclusion` 施加 reward penalty。
matched baseline 必须保持相同 completion routes 和 OPD compute；若只降低整体 OPD
weight，不能把差异归因于 token selection。

当前 lexical prototype（普通 token `0.75`、数字 `2.0`、`answer` token `1.5`）已通过
1-step runtime smoke，但未通过可靠性准入。8 条实际 OPD completion 共 55 个 token，
其中 14 个被判为 numeric；10 个 numeric token 来自单个幻觉长数字 `1045896273`，
该错误 span 独占约 `33.6%` 的总加权质量。因此“见数字即升权”会放大数字幻觉，
该 prototype 不得进入完整训练。后续 P0-E6 必须改用 teacher support/confidence 或
teacher--student distribution agreement 驱动的 span reliability，并报告加权质量而不只
报告 token 数量。

执行顺序：P0-E0 完成并评估后，先运行 P0-E1 与 P0-E3，建立最关键的
OPD-vs-no-OPD 因果对照；若 OPD 有净收益，再补 P0-E2 和 P0-E5 解释可靠性与
互补性，并以 P0-E4 冲击最高分。P0-E6 是 novelty-critical near-neighbor baseline。
controller state/action 消融只有在 OPD 主效应成立后才占用完整 4epoch 预算。

### 3.1 P0 Runner Readiness Audit

截至 `2026-07-13`，论文中的实验槽位并不都已有可直接运行的显式 variant。当前
排队的 oracle clean-OPD 配方保持冻结；以下缺口只进入后续实现队列，不得修改正在
等待 GPU gate 的 run。

| ID | Runner readiness | Current evidence | Required explicit variant before launch |
|---|---|---|---|
| P0-E0 | `ready/running` | oracle no-full-hint hard-SFT variant、resilient runner、自动 eval 均已实现 | 无；等待当前 GPU gate |
| P0-E1 | `ready/not queued; contract audited` | `...gold_hidden_no_opd` 是 route-matched no-OPD：保留 `mode=dyme_teacher_probe_opd`、teacher probe、cap `8` 与最终 completion routes，只设 OPD weight `0`、GRPO weight `1`；1-step smoke 中 OPD route `0.75`、skip `0.25`，但 loss/grad 均为 `0`，hard-target rate `0`。`2026-07-15` canonical dry-run diff 确认它与 P0-E3 除 variant-specific output/log path 外唯一训练字段差异为 `DYME_OPSD_WEIGHT=0.0` 对 `1.5` | 完成 matched 4epoch；环境合同 artifact 位于 `docs/experiment_results/chartqa-ablation/route_matched_contract_20260715/` |
| P0-E2 | `ready/not queued` | `...gold_hidden_uncond_opd_no_full_hint_hard_sft` 保持 clean hard-target gate，关闭 teacher verifier，all-wrong/mixed wrong completion 保留 OPD route，fixed cap `8`、controller 关闭 | 2-step route smoke 必须确认 teacher-probe calls 为零且 wrong OPD route 非零，再进入完整 4epoch |
| P0-E3 | `ready/not queued` | `...gold_hidden_opd_no_full_hint_hard_sft_fixed` 使用 `format_only,visual_facts_deplot`、short-answer profile、fixed OPD weight `1.5`/cap `8`，controller 关闭；effective sampling 与 E4 都从 step 0 开启 | 2-step runtime snapshot 与 P0-E4 唯一变化检查；oracle E0 完成后按排程启动 |
| P0-E4 | `complete negative result; contract audited` | `...gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision` 在 P0-E3 上启用 global-GRPO controller，target `.30`，并把固定 cap `8` 交给 controller 动态控制为 `8->2`；完整 run `clean_visualfact_fix_opd_main_4epoch_20260714_235824` 的 hard-target/template 指标全零，但 final accuracy 仅 `0.2352`。canonical dry-run diff 确认 teacher probe、sampling、routing、base OPD/GRPO weight、optimizer 和 hard-target gates 不变 | final processed `2500/2500`；teacher-correct 仅 `48.78%`、last50 zero-loss `0.94`，按预注册规则进入 gold-hidden teacher recoverability 单因素改进；合同 artifact 位于 `docs/experiment_results/chartqa-ablation/route_matched_contract_20260715/` |
| P0-E5 | `partial; OPD-only fixed, fallback-only invalid` | pure `GRPO-only` 路径已显式关闭 probe/OPD/SFT slot；`2026-07-14` loss mixer 修复使 OPD-only 在无 OPD sample batch 上仍应用 `GRPO_WEIGHT=0`。fallback-only 虽导出零权重，但 `mode=dyme` 不生成 `opsd_mask`，当前 compute_loss 会绕过 mixer | 先实现独立于 `opsd_mask` 的顶层 base-loss weight 或 route-level mask，为 fallback-only 写真实 compute_loss RED test；之后做 2-step smoke并记录 effective GRPO/OPD/SFT loss contribution |
| P0-E6 | `exploratory-ready; not main claim` | `...gold_hidden_token_reliability_clrc` 的 1-step smoke 证明 runtime path 可用且 hard-target/template leakage 为零；已有 token 审计显示 lexical numeric weighting 可能被幻觉长数字劫持 | 可作为 10epoch 探索项排队，但论文主 claim 前必须重做 teacher-supported span reliability、实现加权质量日志并通过长数字反例 audit |
| P0-E7 | `missing` | 当前 runner 只有无额外 cold-start 的 concurrent routed OPD，没有 matched VOLD-style comparator | 明确 cold-start target 与预算；总 optimizer updates、teacher tokens、generation 和 hard-target exposure 与 P0-E4 matched |
| P0-E8 | `missing` | 当前 teacher quality gate 使用 answer verifier，没有 visual-recoverability comparator | 明确 visual consistency score、reference access 与 fallback；teacher calls 和 OPD budget 与 P0-E3 matched |
| P0-E9 | `diagnostic isolated; matched baseline missing` | 可运行路径已重命名为 `mixed_group_shortest_correct_hard_replay`：选择 mixed group 的 shortest-correct completion，并把其 token 序列作为 hard CE target 复制给所有 mixed-wrong completion；all-wrong completion skip。旧 `ssopd_mixed_group` 与伪 `vold_cold_start` runner 标签会明确失败 | 该诊断不得作为 SSOPD matched baseline；若恢复论文 SSOPD 标签，必须实现 longest-wrong selection、错误 prefix 条件 teacher distribution 与 frontier weight，并通过独立 smoke |

最小 GPU 排程不是一次性铺开全部消融。当前 P0-E4 gold-hidden CLRC 完成后：

1. 若 gold-hidden CLRC 未达到 `0.60`，先做 final forensic，并只实现证据选中的一个
   单因素改进；此时不消耗 gold-hidden 全矩阵预算。
2. 若 gold-hidden CLRC 达到 `0.60`，优先成对实现和运行 P0-E1/P0-E3。这一对是
   “OPD 是否有效”的最低论文证据，优先级高于 controller 消融。
   `0.60` 只触发 matched 因果矩阵，不触发“DyME parity”结论。
3. 只有 P0-E3 优于 P0-E1，才运行 P0-E2 检验 verifier routing，并运行 P0-E4 检验
   adaptive OPD exposure 是否改善 accuracy 与后期自主 route coverage。当前 post-probe
   cap 不降低 teacher calls/tokens，不能作为 compute-Pareto 证据。
4. 只有 OPD 主效应成立，才从 P0-E5 先运行最有信息量的 `GRPO-only` 与
   `OPD+GRPO`，再按结果补 fallback-only 和其余组合。
5. AAAI 最接近方法对照至少保留 P0-E7、P0-E8 与 P0-E9 的 matched screening；其中
   最强者必须与完整方法一起补齐 4epoch。DOPD/TrOPD/TIP-style token reliability 可与
   P0-E6 合并设计，但不能只用论文文字声称 completion/global routing 更优。
6. 与 DyME 的效果结论必须来自统一模型、数据、decode、训练预算和 eval 的 in-repo
   reproduction。外部 Pure DyME `0.649` 与 full DyME `0.675` 是两条不同量级线；后者
   含 Visual Supervision，不能与 Pure DyME 混写。

每个新 variant 在 4epoch 前必须完成：配置解析测试、runner dry-run snapshot、直接 route
单测、4-step 8-GPU smoke，以及 hard-target invariant 全零检查。仅把环境变量设为零但
仍保留对应 route label、teacher call 或 fallback side effect，不视为合格消融。

### 3.2 Final 后的预注册单因素决策规则

为避免在一次低分后同时修改 controller、OPD token loss、sampling 和输出格式，本轮
resilient run 的下一步在看到 final 前按以下规则预注册。统计窗口统一使用最后 100 个
有效 optimizer rows；eval 必须处理至少 `2496/2500`。

1. **继续保留当前方法，不因早期 OPD route 高而提前停止。** 中断 clean run 在 steps
   71--80 的 accuracy/GRPO 为 `0.0797/0.0844`，不低于 `0.5800` 基线相同窗口的
   `0.0707/0.0781`；degenerate 为 `0.1375`，也低于基线 `0.1969`。当前差异主要是把
   约 `0.45` 的 full-hint hard-SFT route 替换成 soft OPD，而非已证明的 OPD 过强。
2. **Controller-exit 单因素。** 若 final `<0.58`，且 last100 同时满足
   `routing/opd_route_rate > 0.50`、`routing/grpo_route_rate < 0.25`、
   `signal/grpo_zero_loss_rate > 0.75`，下一跑只把
   `adaptive_target_readiness: 0.30 -> 0.20`；其余 evidence、loss、cap endpoints、sampling
   和 generation 不变。该条件表示 4epoch 结束时 OPD exposure 仍未退出，而不是仅仅
   训练早期 OPD 高。历史 `0.5800` run 早于 global-snapshot logging，只能用 rank-local
   GRPO-route 序列作兼容反事实：target `0.30` 在 zero-based replay index `417` 达到 cap 2，
   target `0.20` 提前到 index `304`。clean interrupted run 的 86 行则全部使用真实 global
   signal；其中 target `0.30` 始终保持 cap 8，而 target `0.20` 在 index `77` 首次降到
   cap `<=6`。因此降 target 是有条件的退出修复，不是默认更优超参。
3. **Token-reliability 单因素。** 若 final `<0.60`，但 last100 已满足 GRPO route
   `>=0.30`、OPD cap 接近 `2`、degenerate `<0.10`，并且 eval 错误主要集中于
   malformed/empty Answer 或错误 heading 后的低答案准确率，则保持 completion routes 与
   controller 不变，只运行 P0-E6 的 answer-bearing token reliability OPD。不得对
   `Goal/Observation/Reasoning/Conclusion` 名称本身施加 reward penalty。
4. **Optimization/retention 单因素。** 若训练中间窗口明显优于 last100，且 final/最后
   checkpoint 同步退化，而 route、格式和 leakage 均健康，则只比较 final 与已保存的
   step-50 checkpoints，并预注册一个 matched learning-rate/early-stop intervention；不能
   用 checkpoint cherry-picking 作为主结果。
5. **方法成功但论文证据不足。** 若 gold-hidden CLRC final `>0.60`，不继续调主配方；立即
   成对实现 P0-E1/P0-E3，先证明 gold-hidden matched OPD 净收益。若 gold-hidden CLRC 在
   `[0.58, 0.60]` 且无上述明确病因，先完成 final forensic，再选择唯一最高证据因素，
   不启动多因素 sweep。

预启动 forensic 已保存于
`outputs/test-fast/long-runs/oracle_opd_no_full_hint_hard_sft_adaptive_resilient_4epoch_20260713_181613/prelaunch_forensics/`。
它显示 clean interrupted run 的 all-wrong teacher-correct rate 为 `0.9294`，因此 teacher
答案质量不是当前首要瓶颈；截至 step 86 的不足主要是训练尚未进入后期自主区间。

对 interrupted clean run 的 21,183 条 wrong teacher-probe candidates 进一步审计
`student_output` 后发现：step 70--79 的任意 reasoning heading rate 为 `44.82%`，但
完整 `Goal/Observation/Reasoning/Conclusion` 四段模板 rate 为 `0.0%`；同一窗口
`Answer:` rate 为 `88.50%`，empty/malformed Answer 均仅 `0.76%`。step 80--86 的
Goal-without-Answer rate 上升到 `8.01%`，仍不构成完整模板塌缩。这个结果把
“teacher candidate 本身总是 full template”与“student 已复制完整 teacher template”
区分开来：当前只支持晚期 partial style transfer，尚不足以提前启用 heading penalty
或 token-selective OPD。完整数字记录在当前 resilient run 的
`prelaunch_forensics/pcd_low_score_forensics.md`。

## 4. P1：OPD 支撑机制、效率与稳健性

| ID | Experiment | Unique change | Budget | Required output |
|---|---|---|---|---|
| P1-E1 | OPD weight only | hard trajectory 固定为零，controller 不改变 cap | 4epoch | final acc + route dynamics |
| P1-E2 | post-probe OPD cap only | weights fixed，仅将进入 OPD loss 的 completion cap `8->2` | 4epoch | OPD route coverage + final acc；不得声称减少 teacher tokens |
| P1-E3 | joint OPD exposure actions | hard trajectory 固定为零，联合控制 OPD weight 与 post-probe cap | 4epoch | accuracy/exposure tradeoff |
| P1-E4 | changed effective batch | batch/accumulation 改变 | matched 4epoch | action trigger autonomy state |
| P1-E5 | changed data scale | train subset/full scale | matched updates | fixed-step vs CLRC robustness |
| P1-E6 | non-monotonic hysteresis | mastery 可回退但带滞回 | matched diagnostic/full if promising | regression recovery |
| P1-E7 | controller state | mixed/zero proxy vs global GRPO route | matched diagnostic + promising rows full | state-variable contribution |
| P1-E8 | pre-probe candidate cap | 在 teacher generation 前由同一 controller 限制 probe candidates | matched screening；若有效则 4epoch | teacher calls/tokens + final acc；唯一可支持 controller compute-Pareto 的 action |

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
/home/deepseek_VG/.conda/envs/dyme/bin/python -m accelerate.commands.launch \
  --config_file default_config_8gpu.yaml \
  --num_processes 8 \
  -m eval.eval_chartqa \
  --model_path <run>/<variant>/final_checkpoint
```

快速迭代可接受 `2496/2500`，但正式主表优先补齐 `2500/2500`。OOM/traceback 行不得作为结果。

Final eval 同时输出 `Template behavior counts`：full/partial CoT template、
Goal-without-Answer、empty skeleton 与 malformed Answer。`summary.csv` 保留该字段，
用于区分有内容的 full-CoT 与固定模板污染；这些诊断不改变答案抽取或 accuracy scoring。

### 7.1 论文级逐样本评估证据

当前 `eval_chartqa.py` 只把 `prediction #### reference #### correct` 打印到日志，未保存
稳定 sample ID。单点 accuracy 足以决定训练迭代，但不足以证明相对 `0.5872` oracle
official 或 matched no-OPD 的提升具有统计可靠性。正式论文 eval 必须额外保存：

- 每 rank 的 `predictions_rank{rank}.jsonl`，包含 dataset index、question/image ID、原始
  generation、parsed answer、reference、correct、output type 与模板诊断；
- 按 dataset index 确定性合并的 `predictions.jsonl`，并验证无重复、无缺失；
- 与 `summary.csv` 一致的 correct count、processed count 和 artifact hash；
- 对 matched runs 使用同一 2,500 条样本做 paired bootstrap 95% CI 与 McNemar test，
  同时报告 `both correct`、`ours only`、`baseline only`、`both wrong`；
- 按 numeric/non-numeric、full/partial template、Answer parse failure 分层的 paired delta。

`>0.60` 仍以有效 `summary.csv` 为训练目标；论文中的“显著优于”只在逐样本配对证据
可用后使用。旧日志若缺少稳定 sample ID，不得通过输出顺序强行对齐。

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
| clean OPD 在 step 60 前无法形成 GRPO，但 teacher precision 高 | 加入短 `Answer: <gold>` 或 student-short answer anchor，并按 realized GRPO 连续衰减；禁止恢复 dataset full hint | OPD route、teacher probe、generation、controller signal |
| global GRPO coverage 上升但 eval 不升 | 检查 answer extraction、reasoning/answer target mismatch 和错误模板条件概率 | controller state 与 teacher probe budget |
| oracle CLRC 有效、gold-hidden teacher precision/coverage 低 | 改进 gold-hidden visual evidence 与 recoverability quality gate | global controller、optimization recipe |
| teacher precision 高但 OPD route 无收益 | 检查 token-position reliability、teacher/student state mismatch 与 OPD divergence | evidence source、GRPO objective |
| partial Goal-without-Answer drift 高、但完整模板尚未塌缩 | 比较 token-selective/answer-bearing-position OPD 与统一 token OPD；优先降低结构 token 权重而非整体关闭 OPD | teacher correctness gate、GRPO/fallback、controller state |
| 低 GRPO 与最大 OPD exposure 长期共存 | 检查是否为 OPD token reliability 导致的正反馈；保持单一 controller signal，先增加局部 token gate | controller EMA/target、teacher probe budget、GRPO objective |
| controller 很少产生动作变化 | 调整 target/calibration 或采用带滞回的可逆控制 | local router 与 teacher target |

每次完整 4epoch run 只选择表中一行作为主因果干预；其余变化仅限修复已验证的实现错误。这样才能把超过 `0.60` 的 recipe 转化为可归因的论文证据。

当前 gold-hidden 主实验的 gate-40 funnel 已触发上表的 evidence/probe-quality 诊断条件：
all-wrong teacher-correct 仅 `48.91%`，mixed-wrong 为 `70.83%`，且 parse-fail 为
`8.45%`。若 step-60 recovery gate 失败，候选单因素应保持 student route、controller、
OPD objective、sampling 与 optimizer 不变，只改善 teacher probe：强化 DePlot 表头/行列
语义和数值运算指令，并用 teacher-response prefix 固定 canonical `Answer:` 生成，随后先做
同一批样本的 paired teacher micro-eval。该 prefix 只约束用于 recoverability routing/OPD
的 teacher 输出，不对 student completion 施加格式 reward 或 hard target。

Checkpoint-100 后新增一个 exploration diagnostic：完整 all-wrong group 的平均唯一
completion 数由早期 `7.42/8` 降到 `3.47/8`，八次完全相同比例升到 `28.83%`。
因此若 final `<0.60` 且 last100 zero-loss 高，同时 unique-completion 指标继续下降，
下一跑只调整 student generation diversity（优先 temperature/top-p 或显式 entropy
calibration），保持 teacher probe、OPD route/controller、loss、sampling weights 与
optimizer 不变。不得同时加入 answer anchor，否则无法区分“探索恢复”和“teacher
监督增强”的因果贡献。

`2026-07-15` 完成的 teacher-only follow-up 进一步排除了“只要加入结构化 prompt 即可
恢复 oracle 精度”的假设。在同一固定种子 128 条困难候选上，旧 short-answer micro-eval、
结构化两阶段 gold-hidden 和 oracle-hint 的 raw correctness 分别为
`0.0469/0.3203/0.9219`。结构化方案通过无-gold draft 再 canonicalize `Answer:`，把
parse-fail 从 `0.8984` 降至 `0`，但仍未达到预注册的 `0.90` teacher gate。因此不为该
prompt 消耗完整 4epoch；下一单因素必须改变 evidence recovery 本身，例如更可靠的表格
重建、图像/DePlot 一致性选择或更强 teacher，而不是继续加强格式指令。Artifact 位于
`outputs/test-fast/teacher-probe-micro-eval/paired128_20260715/`。

随后四次独立 sampled gold-hidden rollout 的 best-of-N feasibility test 也未通过：四个
单次准确率为 `0.3281/0.2656/0.3594/0.3281`，reference verifier 的 best-of-4 并集仅
`0.5000`。`32/64` 个 prompt 在四次 rollout 中全部错误，说明错误高度相关。BRTS-style
teacher resampling 可以作为 comparator，但不能作为下一轮唯一 evidence intervention；
下一设计必须引入不同 evidence view 或显式修复表格/图像冲突。

`2026-07-15` 的下一轮 teacher-only gate 已实现“原始模态固定、辅助证据动态获取”的
ChartQA adapter。所有 production attempt 都保留原始 chart image；`visual_base`、
`visual_deplot` 与按冲突触发的 `visual_recovery` 不再对应 table-only/image-only 的错误
二分。固定 128 条结果为：base `0.6875`、image+DePlot `0.6328`、当前 reference-free
selector `0.6484`、三路 oracle union `0.7656`。`64/128` 的一致样本准确率达到
`0.8906`，但另外 `64/128` 的 recovery 准确率只有 `0.4062`；accepted precision
`0.6917`、false-accept `0.3083`、mean attempts `2.50`。这证明新 prompt 大幅提升了
gold-hidden image-native teacher，并且异构 evidence configuration 确实互补，但当前
selector 反而损失了 base 的正确答案，因此仍不得进入 4epoch。

下一步必须保持 teacher prompts 和三路 rollout 固定，只改 selector calibration：在独立
calibration split 上确定“image-primary、DePlot 仅在可验证的 table-friendly case 覆盖”
的规则或轻量 router，然后在未参与规则选择的 held-out 128 条上报告 accepted precision、
coverage、selected accuracy 与 route regret。当前 128 条上的 post-hoc 最优规则只能用于
生成假设，不能作为最终结果。Artifact:
`outputs/test-fast/teacher-probe-micro-eval/chartqa_recoverable_128_20260715_122656/`。

按 calibration 结果冻结的第一版风险策略只接受两类样本：base/DePlot 初始一致，或冲突
后 recovery 再次确认 image-native base；其余全部 abstain。`seed=29` 的 128 条
near-held-out probe 上，accepted precision 为 `0.8378` (`62/74`)，通过 `>=0.80`；但
false-accept 为 `0.1622`、abstain 为 `0.4219`、mean attempts 为 `2.51`，仍未通过
`<0.10` false-accept 和 `<=2.2` 成本 gate。base/DePlot/selected/oracle-union 分别为
`0.6484/0.5938/0.4844/0.7656`。该集合与 calibration 有 `2/128` 重叠，因此只能视为
近似 held-out；正式确认需在 sampler 中显式排除 calibration IDs。Artifact:
`outputs/test-fast/teacher-probe-micro-eval/chartqa_recoverable_heldout128_seed29_20260715_123550/`。

当前结论不是“multi-view 无效”，而是“高精度 recovery route 已可构造，但 coverage 和
成本仍不足”。下一单因素应优化停止策略：优先只跑 image-native base；仅在 base 的
可部署不确定性/题型特征触发时调用 DePlot，从而把平均调用降到 `<=2.2`，同时用更严格的
程序/legend grounding verifier 将 false-accept 压到 `<0.10`。在这两个 gate 通过前，
不得接入完整 4epoch。

同日 Qwen2.5-VL-7B image-native teacher probe 给出了更强的 cross-family prompt
diagnostic：新增 `qwen25vl` backend 和 `Answer:` prefill control 后，单路
`visual_answer_prefix` 在三组 128 条困难候选上的 corrected teacher accuracy 为
`0.8594/0.9375/0.8594` (`seed=13/29/31`)；三路 image-native agreement 聚合覆盖
`244/384 = 63.5%`，precision `229/244 = 0.9385`。但这不是 OPD-admissible teacher
证据，因为 Qwen 与 LLaVA 学生不共享 tokenizer/模型谱系；它只能说明 prompt harness 和
agreement 思路可能有效，不能直接进入 OPD/SFT teacher target。

随后改回同谱系 `llava-7b-ov` teacher 后，完全相同的三路 agreement 明显降级：
`seed=13/29/31` 的 all3 precision 分别为 `0.7901/0.8101/0.8171`，三 seed 聚合为
`195/242 = 0.8058`，覆盖 `242/384 = 0.6302`；all-wrong precision 只有
`73/98 = 0.7449`，mixed-wrong precision 为 `122/144 = 0.8472`。单路
`visual_answer_prefix`、`visual_short_answer`、`visual_answer_prefix_numeric` 聚合准确率
分别为 `0.6354/0.6667/0.6120`。Artifact:
`outputs/test-fast/teacher-probe-micro-eval/llava7b_same_lineage_agreement_3seed_summary.csv`。

因此当前结论改为：三路 agreement gate 可以保留为默认关闭的 trainer 实验开关和诊断工具，
但不能作为下一次 4epoch 主训练的 teacher gate。下一步应先提升同谱系 LLaVA teacher
可恢复性，例如改进 image-native prompt、加入可验证的 chart program/legend grounding、
或寻找与学生 tokenizer 兼容的更强同谱系 teacher；在同谱系 accepted precision 未达到
`>=0.90`（尤其 all-wrong `>=0.90`）前，不得把该 gate 接入完整 4epoch。

按错误审计实现的第一版小模型 harness 先从 agreement 改为 verifier-backed multi-view
selection，用作诊断而不是最终方法：teacher 仍为同谱系 `llava-7b-ov`，prompt 不看 gold；micro-eval 同时跑
`visual_answer_prefix`、`visual_operation_answer_prefix` 和
`deplot_operation_answer_prefix`，由 ChartQA verifier 选择第一条正确输出。在 seed31 的
128 条 smoke 中，旧三路 oracle union 为 `95/128 = 0.7422`，新 harness 为
`102/128 = 0.7969`，all-wrong 从 `39/64 = 0.6094` 提高到 `46/64 = 0.7188`，
mean attempts `1.64`。提升主要来自 count、difference、average、extreme 等需要 operand
选择或计算的题型。Artifact:
`outputs/test-fast/teacher-probe-micro-eval/llava7b_verifier_harness128_seed31_20260715_153817/`。
这还不足以直接启动 4epoch，但证明“小模型 prompt/harness + verifier selection”比
同模型 agreement 更适合当前设置。

后续方案不再把该路径定位为“recovery selector”。真正可进入下一阶段的是
`chartqa_closed_loop_recovery`：一个 integrated closed-loop evidence recovery
controller。它把 verifier 拒绝转成运行时事件，再由控制器依次触发 image-native
operation recovery、DePlot operation recovery、table-executable DePlot recovery、
reasoned recovery、target grounding、arithmetic recovery 和 scale/unit recovery。也就是说，
可执行 DePlot 不是最后挑答案的旁路，而是闭环轨迹中的 recovery operator；错误候选会被
同一个 verifier 拒绝并继续恢复。加入 table-executable operator 后，三 seed
image-in-the-loop confirmation 达到 `352/384 = 0.9167` (`114/128`, `119/128`,
`119/128`)，相对上一版 closed-loop `330/384 = 0.8594` 提升 `+5.73` 点，其中
`executable_deplot_recovery` 贡献 `26/384` 个 accepted recoveries。

在该 harness 上继续做了两个低风险改进。首先，teacher-probe-only verifier 接受更合理的
ChartQA 表面形式：带单位/上下文的数值答案可回退到 leading numeric，`decreased` 等趋势
同义词可匹配 `decreasing`；这只影响 teacher probe 选择，不改变 student eval。seed31
重算从 `102/128` 提高到 `104/128`。其次，把已有 `reasoned_deplot_only` 作为最后的
verifier-gated fallback，而不是常规全量 view；post-hoc 四路 selected coverage 达到
`108/128 = 0.8438`，all-wrong/mixed-wrong 为 `49/64 = 0.7656` 和 `59/64 = 0.9219`。
四个 fallback 独有正确样本均是 operation-heavy 问题，但该 view 独立准确率低且输出长，
所以只允许在前三个 answer-only 快路径失败后触发。

`verifier_early_stop` 已通过 TDD 接入 micro-eval 调度，真实 seed31 run 复现
`108/128 = 0.8438`，但实际 draft attempts 从 post-hoc 的 `512` 降到 `230`，并且只对
到达 fallback 的 `24` 个样本做 canonicalization。Artifact:
`outputs/test-fast/teacher-probe-micro-eval/llava7b_verifier_early_stop_reasoned128_seed31_20260715_161843/`。
三 seed 确认后，seed31 是偏乐观样本：`seed=13/29/31` 分别为
`97/128 = 0.7578`、`105/128 = 0.8203`、`108/128 = 0.8438`，aggregate 为
`310/384 = 0.8073`，mean attempts `1.88`。实际 draft attempts 为 `723`，而非
post-hoc 全量四路的 `1536`；accepted-by-control 为 `visual_answer_prefix=249`、
`visual_operation_answer_prefix=16`、`deplot_operation_answer_prefix=34`、
`reasoned_deplot_only=11`。剩余 `74` 个 abstain 中 all-wrong 占 `54` 个，主要题型是
difference/percent/count/average。因此该 harness 提供了清晰工程收益和成本下降，但仍低于
预注册 `>=0.90` teacher gate；下一步应改 teacher/evidence 能力本身，例如更强同谱系
teacher、可验证 chart-program/legend grounding、或对复杂运算问题的专门 evidence
recovery，而不是直接接入完整 4epoch。Aggregate artifact:
`outputs/test-fast/teacher-probe-micro-eval/llava7b_verifier_early_stop_reasoned_3seed_20260715/`。

短 answer anchor 的目标是提供 VOLD 所强调的 cold-start alignment，同时避免本项目已经
观察到的 full teacher-trajectory/template contamination。它必须单独记录
`routing/answer_anchor_rate` 与 target token length，并在 last50 衰减到低占比；否则不能
把最终收益归因于 OPD。

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
