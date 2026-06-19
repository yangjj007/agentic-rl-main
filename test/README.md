# test/ 快速训练 Baseline

在较短时间内跑通三条 ChartQA baseline，用于与全量 OPD 训练（10 epoch）做快速对比。配置文件名与 [`config/`](../config/) 生产流程一致，**使用全量数据集**，通过**减少 epoch** 控制耗时。

> 注意：本目录 `test/` 与单元测试目录 [`tests/`](../tests/) 无关。

## 三条 Baseline

| Baseline | 脚本 | 配置 | 入口 | 说明 |
|----------|------|------|------|------|
| **纯 SFT** | `train_sft.sh` | `config/config_rlsd_chartqa.py` | `main_sft.py` | 离线监督微调，全量数据，4 epoch |
| **DyME** | `train_dyme.sh` | `config/config.py` | `main.py --mode rl` | 纯 GRPO，无 OPSD，4 epoch |
| **OPD** | `train_opd.sh` | `config/config_opd_7b_chartqa_deepspeed.py` | `main.py --mode rl --opsd_enabled` | 7B teacher + 0.5B student，DeepSpeed，4 epoch |

三条 baseline 默认从同一 base 0.5B 出发。OPD **不依赖**先跑离线 SFT；embedded 冷启动计入总步数（见下）。

### OPD 冷启动步数

默认 `num_train_epochs=4`，`sft_cold_start_frac=0.08`。以 4 GPU、约 600 步/epoch 估算（总步数 ~2400）：

- **冷启动（embedded SFT）**：约前 **192** 步 — 不 generate、100% GT 注入、纯 SFT NLL
- **RL/OPD 阶段**：剩余约 **2208** 步 — RLSD 路由 + 7B teacher OPD

实际总步数由 dataloader 长度决定；gate warmup 按估算总步数同比缩放（见 `config/fast_profile.py`）。

## 统一常量

| 常量 | 默认值 | 环境变量 | 全量训练对照 |
|------|--------|----------|-------------|
| 数据集 | 全量 `train_medium_vf_full.json` | — | 同左 |
| RL epoch | **4** | `DYME_FAST_NUM_TRAIN_EPOCHS` | 10 |
| SFT epoch | **4** | `DYME_FAST_SFT_EPOCHS` | 2 |
| 冷启动占比 | 8% | `DYME_FAST_COLD_START_FRAC` | 8% |
| Gate 步数估算 | 600 步/epoch | `DYME_FAST_EST_STEPS_PER_EPOCH` | — |
| 输出根目录 | `outputs/test-fast/` | `DYME_FAST_OUTPUT_ROOT` | `outputs/*` |

## 快速启动

```bash
# 单条 baseline（首次会自动构建 train_medium_vf_full.json）
bash test/train_sft.sh
bash test/train_dyme.sh
bash test/train_opd.sh

# 顺序跑完全部三条
bash test/run_all_baselines.sh
```

### 输出路径

| Baseline | Checkpoint |
|----------|------------|
| SFT | `outputs/test-fast/sft/final_checkpoint` |
| DyME | `outputs/test-fast/dyme/` |
| OPD | `outputs/test-fast/opd-7b-ds/` |

日志：`outputs/test-fast/logs/`

### 可选：SFT 后再跑 OPD

```bash
export DYME_PRETRAINED_MODEL=outputs/test-fast/sft/final_checkpoint
bash test/train_opd.sh
```

### 调参示例

```bash
# 更短：1 epoch
DYME_FAST_NUM_TRAIN_EPOCHS=1 DYME_FAST_SFT_EPOCHS=1 bash test/train_dyme.sh

# 更长：6 epoch
DYME_FAST_NUM_TRAIN_EPOCHS=6 bash test/run_all_baselines.sh

# OPD 内存紧张时换 ZeRO-2
ACCELERATE_CONFIG=default_config_zero2.yaml bash test/train_opd.sh

# 使用 DDP 版 OPD 配置
DYME_CONFIG=test/config/config_opd_7b_chartqa.py \
  ACCELERATE_CONFIG=default_config.yaml \
  bash test/train_opd.sh
```

## 目录结构

```
test/
  config/
    fast_profile.py              # 统一常量与 override 逻辑
    config.py                    # DyME
    config_rlsd_chartqa.py       # SFT
    config_opd_7b_chartqa.py     # OPD (DDP)
    config_opd_7b_chartqa_deepspeed.py  # OPD (DeepSpeed, 默认)
  launch_utils.sh
  train_sft.sh
  train_dyme.sh
  train_opd.sh
  run_all_baselines.sh
  README.md
```

## 与全量训练的对应关系

| 全量 | test/ 快速版 |
|------|-------------|
| `config/config.py` (10 epoch) | `test/config/config.py` (4 epoch) |
| `config/config_rlsd_chartqa.py` (SFT 2 epoch) | `test/config/config_rlsd_chartqa.py` (4 epoch) |
| `config/config_opd_7b_chartqa_deepspeed.py` (10 epoch) | `test/config/config_opd_7b_chartqa_deepspeed.py` (4 epoch) |
| `scripts/train_chartqa_sft.sh` | `test/train_sft.sh` |
| `scripts/train_baselines.sh MODE=dyme` | `test/train_dyme.sh` |
| `scripts/train_opd_7b_chartqa_deepspeed.sh` | `test/train_opd.sh` |

## 预估耗时（4 GPU 参考，全量 ~2.3 万样本）

| 阶段 | 全量 (10 epoch) | test/ (4 epoch) |
|------|----------------|-----------------|
| SFT | ~数小时 | ~4–8 小时 |
| DyME | ~数天 | ~12–16 小时 |
| OPD | ~1.5 天 | ~16–24 小时 |
| 串行三条 | — | ~1.5–2 天 |

若需进一步压缩，可降低 `DYME_FAST_NUM_TRAIN_EPOCHS` / `DYME_FAST_SFT_EPOCHS`（例如设为 1–2）。
