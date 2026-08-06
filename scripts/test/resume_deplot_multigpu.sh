#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

DEPLOT_MODEL="${DYME_DEPLOT_MODEL:-models/deplot}"

LOG_PATH=/tmp/train_medium_vf_full_deplot.nohup.log
{
  echo "[start] $(date -Is) resume multi-GPU DePlot"
  echo "[config] devices=cuda:3,cuda:7 batch_size=4 worker_chunk_size=32 cache=/tmp/train_medium_vf_full_deplot_cache.json"
} > "${LOG_PATH}"

PYTHONUNBUFFERED=1 python \
  scripts/build_visual_facts_chartqa_deplot.py \
  --input data/chartqa/train_medium_vf_full.json \
  --output /tmp/train_medium_vf_full_deplot.json \
  --cache /tmp/train_medium_vf_full_deplot_cache.json \
  --batch-size 4 \
  --worker-chunk-size 32 \
  --max-new-tokens 384 \
  --model-id "${DEPLOT_MODEL}" \
  --devices cuda:3,cuda:7 \
  --dtype float32 \
  --no-progress >> "${LOG_PATH}" 2>&1

status=$?
echo "[exit] $(date -Is) status=${status}" >> "${LOG_PATH}"
exit "${status}"
