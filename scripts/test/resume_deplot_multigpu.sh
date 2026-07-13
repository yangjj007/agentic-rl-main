#!/usr/bin/env bash
set -euo pipefail

cd /dev/shm/deepseek/agentic-rl-main

LOG_PATH=/tmp/train_medium_vf_full_deplot.nohup.log
{
  echo "[start] $(date -Is) resume multi-GPU DePlot"
  echo "[config] devices=cuda:3,cuda:7 batch_size=4 worker_chunk_size=32 cache=/tmp/train_medium_vf_full_deplot_cache.json"
} > "${LOG_PATH}"

PYTHONUNBUFFERED=1 /home/deepseek_VG/.conda/envs/dyme/bin/python \
  scripts/build_visual_facts_chartqa_deplot.py \
  --input data/chartqa/train_medium_vf_full.json \
  --output /tmp/train_medium_vf_full_deplot.json \
  --cache /tmp/train_medium_vf_full_deplot_cache.json \
  --batch-size 4 \
  --worker-chunk-size 32 \
  --max-new-tokens 384 \
  --model-id /home/deepseek_VG/deepseek/models/deplot \
  --devices cuda:3,cuda:7 \
  --dtype float32 \
  --no-progress >> "${LOG_PATH}" 2>&1

status=$?
echo "[exit] $(date -Is) status=${status}" >> "${LOG_PATH}"
exit "${status}"
