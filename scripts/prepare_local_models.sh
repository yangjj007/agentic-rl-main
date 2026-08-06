#!/usr/bin/env bash
# Normalize ModelScope --local_dir downloads for HuggingFace transformers loading.
# Usage: bash scripts/prepare_local_models.sh models/llava-0.5b-ov models/llava-7b-ov
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
echo "  export DYME_STUDENT_MODEL=<0.5b-dir>"
echo "  export DYME_TEACHER_MODEL=<7b-dir>"
echo "  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
echo "  bash scripts/train_opd_7b_chartqa_deepspeed.sh"
