# test/ 快速训练 Baseline

在数小时内跑通三条 ChartQA baseline，用于与全量 OPD 训练（10 epoch / 2.3 万样本）做快速对比。配置文件名与 [`config/`](../config/) 生产流程一致，统一小样本 + `max_steps` 预算。

> 注意：本目录 `test/` 与单元测试目录 [`tests/`](../tests/) 无关。

## 三条 Baseline

| Baseline | 脚本 | 配置 | 入口 | 说明 |
|----------|------|------|------|------|
| **纯 SFT** | `train_sft.sh` | `config/config_rlsd_chartqa.py` | `main_sft.py` | 离线监督微调，1 epoch × 512 样本 |
| **DyME** | `train_dyme.sh` | `config/config.py` | `main.py --mode rl` | 纯 GRPO，无 OPSD |
| **OPD** | `train_opd.sh` | `config/config_opd_7b_chartqa_deepspeed.py` | `main.py --mode rl --opsd_enabled` | 7B teacher + 0.5B student，DeepSpeed |

三条 baseline 默认从同一 base 0.5B 出发。OPD **不依赖**先跑离线 SFT；embedded 冷启动计入总步数（见下）。

### OPD 冷启动步数

默认 `max_steps=500`，`sft_cold_start_frac=0.08`：

- **冷启动（embedded SFT）**：前 **40** 步 — 不 generate、100% GT 注入、纯 SFT NLL
- **RL/OPD 阶段**：剩余 **460** 步 — RLSD 路由 + 7B teacher OPD

Gate warmup 已按 `max_steps` 同比缩放（见 `config/fast_profile.py`），避免 smoke 配置中 warmup 大于总步数的问题。

## 统一常量

| 常量 | 默认值 | 环境变量 |
|------|--------|----------|
| 训练样本数 | 512 | `DYME_FAST_MAX_SAMPLES` |
| RL 总步数 | 500 | `DYME_FAST_MAX_STEPS` |
| SFT epoch | 1 | `DYME_FAST_SFT_EPOCHS` |
| 冷启动占比 | 8% | `DYME_FAST_COLD_START_FRAC` |
| 输出根目录 | `outputs/test-fast/` | `DYME_FAST_OUTPUT_ROOT` |
| 小数据集 | `data/chartqa/train_fast_512.json` | `DYME_FAST_TRAIN_JSON` |

## 快速启动

```bash
# 1) 仅准备小数据集（首次或删文件后）
bash test/prepare_fast_dataset.sh

# 2) 单条 baseline
bash test/train_sft.sh
bash test/train_dyme.sh
bash test/train_opd.sh

# 3) 顺序跑完全部三条
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
# 更短干跑
DYME_FAST_MAX_STEPS=50 DYME_FAST_MAX_SAMPLES=128 bash test/train_dyme.sh

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
  build_fast_dataset.py
  prepare_fast_dataset.sh
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
| `config/config.py` | `test/config/config.py` |
| `config/config_rlsd_chartqa.py` | `test/config/config_rlsd_chartqa.py` |
| `config/config_opd_7b_chartqa_deepspeed.py` | `test/config/config_opd_7b_chartqa_deepspeed.py` |
| `scripts/train_chartqa_sft.sh` | `test/train_sft.sh` |
| `scripts/train_baselines.sh MODE=dyme` | `test/train_dyme.sh` |
| `scripts/train_opd_7b_chartqa_deepspeed.sh` | `test/train_opd.sh` |

## 预估耗时（4 GPU 参考）

- SFT：分钟级
- DyME 500 steps：约 1–2 小时
- OPD 500 steps（含 7B teacher）：约 3–4 小时
- `run_all_baselines.sh` 串行：约 5–7 小时
