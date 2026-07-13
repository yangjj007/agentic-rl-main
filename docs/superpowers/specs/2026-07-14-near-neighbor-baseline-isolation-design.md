# Near-Neighbor Baseline Semantic Isolation Design

日期：2026-07-14

## 目标

阻止实验 runner 和论文台账把尚未 matched 实现的近邻方法写成 `VOLD` 或 `SSOPD`，同时保留当前已经可运行、可复现的 mixed-group hard-replay 诊断路径。

## 决策

1. 当前 shortest-correct completion 复制到 mixed-wrong completion 的 hard CE 路径，统一命名为 `mixed_group_shortest_correct_hard_replay`。
2. 配置键、环境变量和训练指标都使用 `mixed_group_hard_replay`，不再出现 `ssopd`。旧 `DYME_SSOPD_MIXED_GROUP` 不提供静默兼容，避免历史命令继续产生错误标签。
3. `vold_cold_start` 从可执行 matched matrix 移除。直接请求旧标签时应明确失败，提示需要真正的 two-stage cold-start alignment runner。
4. 当前实现只作为机制诊断：mixed group 使用最短正确序列作 hard target；all-wrong group 跳过。它不是 conditional distribution self-distillation，也不是论文 SSOPD reproduction。
5. 不修改冻结 Oracle variant 的任何默认值、环境变量或训练路径。

## 验收标准

- 默认 matrix 不再列出 `vold_cold_start` 或 `ssopd_mixed_group`。
- 新 hard-replay variant 的 dry-run 显式导出 `DYME_MIXED_GROUP_HARD_REPLAY=1`，并关闭 teacher probe、OPD 和 online SFT slots。
- trainer 日志只产生 `mixed_group_hard_replay_rate` 与 `mixed_group_all_wrong_skip_rate`。
- 全仓生产代码和可执行 runner 中不再出现 `SSOPD`/`ssopd`/`vold_cold_start` 误导标签；论文可在历史审计段落中保留这些名称以解释为何禁止。
- 聚焦测试通过，且冻结 Oracle dry-run 仍满足原 25 项配置契约。

