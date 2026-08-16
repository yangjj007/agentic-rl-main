# OPD-only ChartQA：8×H GPU 部署

在现有 [`config/config_opd_only_7b_chartqa.yaml`](../config/config_opd_only_7b_chartqa.yaml) 中直接修改：

```yaml
paths:
  # 唯一的项目内路径基目录；project://... 会由 loader 拼接到这里
  project_root: /workspace/agentic-rl-main

model:
  # SFT 完成后的本地完整 Hugging Face checkpoint（必须含 config.json 与模型权重）
  pretrained_model_path: /mnt/models/chartqa-sft/final_checkpoint
  # 本地冻结 7B teacher
  teacher_model_path: /mnt/models/llava-onevision-qwen2-7b-ov
  teacher_device_map: same

training:
  stage: opd_only
  num_gpus: 8
  num_client: 8
  dyme_args:
    output_dir: /mnt/experiments/opd-only-chartqa-8h
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 16

dataset:
  train_dataset: project://data/chartqa/train_new_prerefine_vf_full_real_deplot_fp32_qwen25.json
```

项目内的 `project://outputs/...`、`project://data/...` 会自动使用该基目录，无需逐项修改。若显存不足，设 `per_device_train_batch_size: 1`、`gradient_accumulation_steps: 32`。

## 8 卡启动

```bash
cd /workspace/agentic-rl-main
mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
  --config_file default_config_8gpu.yaml \
  --num_processes 8 \
  --main_process_port 29531 \
  main.py \
  --config config/config_opd_only_7b_chartqa.yaml \
  --mode rl \
  2>&1 | tee logs/opd_only_chartqa_8h.log
```

不要用 `DYME_*` 环境变量覆盖训练配置。

## 运行后检查

```bash
rg -n "training_stage=opd_only|opd_route_count|grpo_route_count|sft_route_count|loss/opsd|teacher_trajectory|teacher_probe|visual" \
  logs/opd_only_chartqa_8h.log
```

应确认：

- `training_stage=opd_only`
- `opd_route_count` 覆盖全部 completion
- `grpo_route_count=0`、`sft_route_count=0`
- `loss/opsd` 有限且非零
- teacher probe、trajectory、visual checker、refiner 都有输出

运行快照位于 `training.dyme_args.output_dir`，包括 `resolved_config.yaml`、`run_env.json` 与 `logs/`。
