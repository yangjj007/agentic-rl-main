# ChartQA 10epoch OPD/CLRC Ablation Migration

Date: 2026-07-14

本文档用于把当前 ChartQA OPD/CLRC 全量实验迁移到新服务器。默认只跑 ChartQA；训练日志和 checkpoint 放大盘，论文/同步 GitHub 的轻量结果单独放到 `docs/experiment_results/chartqa-ablation/`。

## 1. 代码和环境

```bash
git clone <repo-url> agentic-rl-main
cd agentic-rl-main

conda create -n dyme python=3.11 -y
conda activate dyme
pip install -r requirements.txt
pip install deepspeed flash-attn --no-build-isolation

export DYME_PYTHON_BIN="$(which python)"
export WANDB_MODE=disabled
```

已有环境可跳过创建步骤，但必须保证 `accelerate`、`deepspeed`、`torch`、`transformers`、`datasets`、`huggingface_hub` 可用。

## 2. 模型放到项目内

项目默认从 `./models` 读模型；该目录已在 `.gitignore` 中忽略，不会同步权重。

```bash
bash scripts/download_local_models.sh --model-root "$PWD/models"
bash scripts/prepare_local_models.sh \
  "$PWD/models/llava-0.5b-ov" \
  "$PWD/models/llava-7b-ov"

export DYME_MODEL_ROOT="$PWD/models"
export DYME_STUDENT_MODEL="$PWD/models/llava-0.5b-ov"
export DYME_TEACHER_MODEL="$PWD/models/llava-7b-ov"
```

默认下载：

```text
llava-hf/llava-onevision-qwen2-0.5b-ov-hf -> models/llava-0.5b-ov
llava-hf/llava-onevision-qwen2-7b-ov-hf   -> models/llava-7b-ov
```

离线服务器可先在联网机器下载同名目录，再拷贝到项目内 `models/`。离线运行时加：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## 3. 数据准备

训练脚本会检查 ChartQA 训练 JSON。推荐准备以下路径：

```text
data/chartqa/train_medium.json
data/chartqa/train_medium_vf_full.json
data/chartqa/deplot_cache.json
data/images/chartqa/
```

如缺少视觉事实增强文件：

```bash
python scripts/build_visual_facts_chartqa.py \
  --input data/chartqa/train_medium.json \
  --output data/chartqa/train_medium_vf_hint.json \
  --also-set-visual-fact

python scripts/build_visual_facts_chartqa_deplot.py \
  --input data/chartqa/train_medium_vf_hint.json \
  --output data/chartqa/train_medium_vf_full.json \
  --batch-size 8 \
  --cache data/chartqa/deplot_cache.json
```

eval 需要 HuggingFaceM4/ChartQA 数据集缓存。联网机器可先运行一次 eval 或 `datasets.load_dataset("HuggingFaceM4/ChartQA")`，再拷贝 HuggingFace cache。

## 4. 全量实验脚本

主脚本：

```bash
bash scripts/test/run_chartqa_10epoch_ablation_matrix.sh --help
```

默认矩阵 14 个标签：

```text
dyme_pure_original
dyme_full_original
oracle_official_best_4e
gold_hidden_no_opd
gold_hidden_uncond_opd
gold_hidden_routed_opd_fixed
clrc_full
clrc_target020
grpo_only_matched
opd_only_matched
fallback_only_matched
oracle_clean_no_full_hint
token_reliability_clrc
mixed_group_shortest_correct_hard_replay
```

说明：

```text
dyme_pure_original / dyme_full_original: 原始 DyME baseline，已改为 python -m accelerate.commands.launch。
clrc_full: 主方法，routed OPD + realized GRPO controller。
clrc_target020: controller target 从 0.30 降到 0.20。
token_reliability_clrc: 数字/答案 token 加权 OPD，最有希望的新改进方向。
oracle_*: 上限诊断，不作为主 claim。
fallback_only_matched / mixed_group_shortest_correct_hard_replay: 诊断行，论文主表需谨慎标注。
```

默认 `--stages train,eval`，即每个 variant 训练完成后自动启动对应 ChartQA eval，并解析 `summary.csv`。

## 5. 4x8x80G 全量运行

四台机器各跑一个 shard。每台 8 卡：

```bash
export RUN_ID=chartqa10_clrc_$(date +%Y%m%d)

export DYME_MODEL_ROOT="$PWD/models"
export DYME_STUDENT_MODEL="$PWD/models/llava-0.5b-ov"
export DYME_TEACHER_MODEL="$PWD/models/llava-7b-ov"

export DYME_CHARTQA_ABLATION_OUTPUT_ROOT=/path/to/big_disk/chartqa-ablation/checkpoints
export DYME_CHARTQA_ABLATION_LOG_ROOT=/path/to/big_disk/chartqa-ablation/logs
export DYME_CHARTQA_ABLATION_RESULTS_ROOT="$PWD/docs/experiment_results/chartqa-ablation"

export DYME_DYME_NUM_PROCESSES=8
export DYME_DYME_EVAL_NUM_PROCESSES=8
export DYME_CHARTQA_ABLATION_EVAL_NUM_PROCESSES=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/test/run_chartqa_10epoch_ablation_matrix.sh \
  --run \
  --run-id "$RUN_ID" \
  --epochs 10 \
  --shard-index 0 \
  --shard-count 4 \
  --stages train,eval \
  --speed-profile canonical \
  --resume none
```

把 `--shard-index` 分别改成 `1`、`2`、`3` 后在另外三台机器运行。断点续跑：

```bash
bash scripts/test/run_chartqa_10epoch_ablation_matrix.sh \
  --run \
  --run-id "$RUN_ID" \
  --epochs 10 \
  --shard-index 0 \
  --shard-count 4 \
  --stages train,eval \
  --resume auto
```

单机全矩阵 dry-run：

```bash
bash scripts/test/run_chartqa_10epoch_ablation_matrix.sh \
  --dry-run \
  --run-id chartqa10_dryrun \
  --stages train,eval
```

## 6. 结果同步目录

轻量结果写入：

```text
docs/experiment_results/chartqa-ablation/<RUN_ID>/matrix_manifest.csv
docs/experiment_results/chartqa-ablation/<RUN_ID>/<label>/summary.csv
docs/experiment_results/chartqa-ablation/<RUN_ID>/<label>/manifest.csv
```

这些文件适合同步 GitHub。大文件不进 GitHub：

```text
$DYME_CHARTQA_ABLATION_OUTPUT_ROOT/<RUN_ID>/<label>/...
$DYME_CHARTQA_ABLATION_LOG_ROOT/<RUN_ID>/<label>/...
```

`outputs/`、`models/`、`*.safetensors`、`data/images/` 已在 `.gitignore` 中忽略。

## 7. Smoke 测试

新服务器先跑 1 step：

```bash
CUDA_VISIBLE_DEVICES=0 \
DYME_CHARTQA_ABLATION_SMOKE_STEPS=1 \
DYME_CHARTQA_ABLATION_RESULTS_ROOT=outputs/test-fast/results-smoke-matrix \
bash scripts/test/run_chartqa_10epoch_ablation_matrix.sh \
  --run \
  --smoke \
  --run-id smoke_chartqa10 \
  --variants clrc_full,token_reliability_clrc,grpo_only_matched \
  --stages train \
  --speed-profile fast60
```

本机已跑通的 smoke：

```text
smoke_all14_20260714_v4:
  dyme_pure_original, dyme_full_original, oracle_official_best_4e,
  gold_hidden_no_opd, gold_hidden_uncond_opd, gold_hidden_routed_opd_fixed,
  clrc_full, clrc_target020, grpo_only_matched, opd_only_matched,
  fallback_only_matched

smoke_remaining_20260714_v5:
  grpo_only_matched, fallback_only_matched, oracle_clean_no_full_hint,
  token_reliability_clrc, mixed_group_shortest_correct_hard_replay
```

关键 smoke 结论：

```text
DYME_MAX_STEPS/DYME_PCD_MAX_STEPS 生效。
DYME_SKIP_FINAL_SAVE=1 生效，smoke 不再写 final checkpoint。
grpo_only_matched 和 fallback_only_matched 已禁用 7B teacher 加载。
token_reliability_clrc 日志显示 token_weighting.enabled=true。
mixed_group_shortest_correct_hard_replay 日志显示 mixed_group_hard_replay 路由指标存在。
```

## 8. 代码验证

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_chartqa_10epoch_ablation_matrix.py \
  tests/test_dyme_matched_runner.py \
  tests/test_model_paths.py \
  tests/test_pcd_no_visual_runner.py::test_gold_hidden_fixed_routed_opd_variant_is_clean_and_non_adaptive \
  tests/test_pcd_no_visual_runner.py::test_gold_hidden_adaptive_routed_opd_only_adds_controller \
  tests/test_pcd_no_visual_runner.py::test_gold_hidden_no_opd_variant_disables_opd_and_hard_sft \
  tests/test_pcd_no_visual_runner.py::test_gold_hidden_grpo_only_variant_removes_teacher_probe_and_opd \
  tests/test_pcd_no_visual_runner.py::test_gold_hidden_unconditional_opd_skips_teacher_verifier \
  tests/test_pcd_no_visual_runner.py::test_gold_hidden_target020_changes_only_controller_target \
  tests/test_pcd_no_visual_runner.py::test_gold_hidden_opd_only_variant_zeros_grpo_weight \
  tests/test_pcd_no_visual_runner.py::test_gold_hidden_fallback_only_variant_exports_requested_zero_weights \
  tests/test_pcd_no_visual_runner.py::test_gold_hidden_token_reliability_variant_enables_token_weighting \
  tests/test_pcd_no_visual_runner.py::test_gold_hidden_mixed_group_hard_replay_variant_is_honestly_isolated \
  tests/test_pcd_no_visual_runner.py::test_pcd_runner_rejects_retired_near_neighbor_variants \
  tests/test_pcd_no_visual_runner.py::test_empty_teacher_model_env_disables_teacher_loading \
  tests/test_opsd_loss_teacher.py \
  tests/test_mixed_group_hard_replay.py \
  tests/test_health_monitor.py::test_finish_step_returns_mixed_group_hard_replay_metrics
```

本机最近验证：`40 passed`，`bash -n` 和核心 `py_compile` 均通过。

## 9. 论文 claim 边界

主 claim 写法：

```text
CLRC is a verifier-routed OPD curriculum for sub-1B VLM ChartQA reasoning that uses realized group-level RL signal to control when teacher distribution matching is applied.
```

不要宣称通用 OPD 首创。已有相关工作覆盖 OPD/VLM 对齐、自蒸馏和偏好优化。AAAI 主会创新点应落在：

```text
gold-hidden recoverability routing
realized GRPO readiness controller
matched OPD/GRPO/fallback decomposition
token-reliability OPD for numeric/chart-answer tokens
oracle/legacy hard-SFT 负结果审计
```

目标：`clrc_full`、`clrc_target020` 或 `token_reliability_clrc` 在 10epoch 上达到或超过 DyME baseline；如果未超过，则用 oracle 行解释上限，用完整消融定位失败机制。
