#!/usr/bin/env bash
# Normalize ModelScope --local_dir downloads for HuggingFace transformers loading.
# Usage: bash scripts/prepare_local_models.sh ~/deepseek/models/llava-0.5b-ov ~/deepseek/models/llava-7b-ov
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_dir> [model_dir ...]" >&2
  exit 1
fi

for raw in "$@"; do
  dir="$(readlink -f "${raw}")"
  if [[ ! -d "${dir}" ]]; then
    echo "ERROR: not a directory: ${raw}" >&2
    exit 1
  fi

  if [[ -d "${dir}/onnx" ]]; then
    echo "Removing unused onnx/: ${dir}/onnx"
    rm -rf "${dir}/onnx"
  fi

  if [[ ! -f "${dir}/config.json" ]]; then
    echo "ERROR: missing config.json in ${dir}" >&2
    exit 1
  fi

  if compgen -G "${dir}/model.safetensors" > /dev/null || compgen -G "${dir}/model-*.safetensors" > /dev/null; then
    ls -lh "${dir}"/model*.safetensors 2>/dev/null | head -3
    echo "OK: ${dir}"
  else
    echo "ERROR: no model.safetensors in ${dir}" >&2
    exit 1
  fi
done

echo "Done. Launch with:"
echo "  export DYME_MODEL_ROOT=<project>/models"
echo "  export DYME_STUDENT_MODEL=<project>/models/llava-0.5b-ov"
echo "  export DYME_TEACHER_MODEL=<project>/models/llava-7b-ov"
echo "  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
echo "  bash scripts/test/run_chartqa_10epoch_ablation_matrix.sh --dry-run --stages train,eval"
