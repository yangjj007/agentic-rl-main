#!/usr/bin/env bash
# Shared helpers for accelerate launch scripts.

detect_num_gpus() {
  # Explicit override always wins.
  if [[ -n "${NUM_GPUS:-}" ]]; then
    echo "${NUM_GPUS}"
    return
  fi

  # Container / scheduler often expose only a subset via CUDA_VISIBLE_DEVICES.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local count=0
    local d
    IFS=',' read -ra _DEVS <<< "${CUDA_VISIBLE_DEVICES}"
    for d in "${_DEVS[@]}"; do
      d="${d// /}"
      if [[ -n "${d}" ]]; then
        count=$((count + 1))
      fi
    done
    if [[ "${count}" -gt 0 ]]; then
      echo "${count}"
      return
    fi
  fi

  # torch.cuda.device_count() reflects what this process can actually use.
  if command -v python >/dev/null 2>&1; then
    local torch_count
    torch_count="$(python - <<'PY' 2>/dev/null || true
import torch
print(torch.cuda.device_count())
PY
)"
    if [[ "${torch_count}" =~ ^[0-9]+$ ]] && [[ "${torch_count}" -gt 0 ]]; then
      echo "${torch_count}"
      return
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${count}" =~ ^[0-9]+$ ]] && [[ "${count}" -gt 0 ]]; then
      echo "${count}"
      return
    fi
  fi

  echo 1
}

launch_num_processes_flag() {
  local num_gpus
  num_gpus="$(detect_num_gpus)"
  echo "--num_processes ${num_gpus}"
}

resolve_deepspeed_zero0_config() {
  # ZeRO-0: no param sharding — fastest when VRAM is sufficient (e.g. 8× H800).
  local num_gpus
  num_gpus="$(detect_num_gpus)"
  if [[ "${num_gpus}" -ge 8 ]]; then
    echo "default_config_8gpu_deepspeed.yaml"
  else
    echo "default_config_deepspeed.yaml"
  fi
}

resolve_deepspeed_zero1_config() {
  # ZeRO-1: optimizer-state sharding — default when ZeRO-0 OOMs on teacher+student.
  local num_gpus
  num_gpus="$(detect_num_gpus)"
  if [[ "${num_gpus}" -ge 8 ]]; then
    echo "default_config_8gpu_deepspeed_zero1.yaml"
  else
    echo "default_config_deepspeed_zero1.yaml"
  fi
}

resolve_accelerate_config() {
  # Explicit override always wins.
  if [[ -n "${ACCELERATE_CONFIG:-}" ]]; then
    echo "${ACCELERATE_CONFIG}"
    return
  fi

  local num_gpus
  num_gpus="$(detect_num_gpus)"

  # Default: native PyTorch DDP (MULTI_GPU). No DeepSpeed install required.
  # DeepSpeed ZeRO-0 (no sharding, fastest when VRAM allows):
  #   resolve_deepspeed_zero0_config → default_config_8gpu_deepspeed.yaml (8 GPU)
  # DeepSpeed ZeRO-2/3 (student sharding, lower VRAM):
  #   ACCELERATE_CONFIG=default_config_zero2.yaml bash scripts/train_opd_7b_chartqa_deepspeed.sh
  if [[ "${num_gpus}" -ge 8 ]]; then
    echo "default_config_8gpu.yaml"
  else
    echo "default_config.yaml"
  fi
}

print_launch_plan() {
  local num_gpus
  local accel_config
  num_gpus="$(detect_num_gpus)"
  accel_config="$(resolve_accelerate_config)"
  echo "============================================================"
  echo "Launch plan: --num_processes ${num_gpus}"
  echo "accelerate config: ${accel_config} (DDP/MULTI_GPU unless overridden)"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi -L:"
    nvidia-smi -L 2>/dev/null || true
  fi
  if command -v python >/dev/null 2>&1; then
    python - <<'PY' 2>/dev/null || true
import torch
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
PY
  fi
  echo "============================================================"
}

# Build data/chartqa/train_medium_vf_full.json when missing (gitignored on GitHub).
# F1: hint → visual_fact_hint; F2: DePlot or placeholder (--no-enabled when DYME_DEPLOT_ENABLED=0).
ensure_chartqa_vf_full() {
  local chartqa_raw="${DYME_CHARTQA_RAW:-data/chartqa/train_medium.json}"
  local chartqa_vf_full="${DYME_CHARTQA_VF_FULL:-data/chartqa/train_medium_vf_full.json}"
  local chartqa_vf_hint="${DYME_CHARTQA_VF_HINT:-data/chartqa/train_medium_vf_hint.json}"
  local deplot_enabled="${DYME_DEPLOT_ENABLED:-1}"
  local deplot_batch="${DYME_DEPLOT_BATCH_SIZE:-8}"
  local deplot_tokens="${DYME_DEPLOT_MAX_NEW_TOKENS:-384}"
  local deplot_cache="${DYME_DEPLOT_CACHE:-data/chartqa/deplot_cache.json}"

  if [[ -f "${chartqa_vf_full}" ]]; then
    echo "ChartQA dataset ready: ${chartqa_vf_full}"
    return 0
  fi

  echo "Enriched ChartQA dataset not found at ${chartqa_vf_full}; running preprocessing..."
  if [[ ! -f "${chartqa_raw}" ]]; then
    echo "Missing raw dataset: ${chartqa_raw}" >&2
    echo "Ensure data/chartqa/train_medium.json exists (shipped in repo)." >&2
    return 1
  fi

  python scripts/build_visual_facts_chartqa.py \
    --input "${chartqa_raw}" \
    --output "${chartqa_vf_hint}" \
    --also-set-visual-fact

  local deplot_extra=()
  case "${deplot_enabled}" in
    0|false|no|off|FALSE|NO|OFF)
      deplot_extra+=(--no-enabled)
      echo "DePlot disabled (DYME_DEPLOT_ENABLED=0); writing placeholder visual_fact_deplot."
      ;;
  esac

  python scripts/build_visual_facts_chartqa_deplot.py \
    --input "${chartqa_vf_hint}" \
    --output "${chartqa_vf_full}" \
    --batch-size "${deplot_batch}" \
    --max-new-tokens "${deplot_tokens}" \
    --cache "${deplot_cache}" \
    "${deplot_extra[@]}"

  echo "ChartQA dataset ready: ${chartqa_vf_full}"
}

# RewardCalculatorLocal needs en_core_web_sm; install once before multi-GPU launch.
ensure_spacy_model() {
  python - <<'PY'
from reward_utils.spacy_model import ensure_spacy_english_model
ensure_spacy_english_model()
print("[DyME] spaCy model ready: en_core_web_sm")
PY
}

# transformers imports tokenizers at load time (GGUF integration path).
ensure_tokenizers() {
  python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("tokenizers") is None:
    print("[DyME] tokenizers missing; installing (required by transformers)...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "tokenizers>=0.21.0,<0.23.0"],
    )

import tokenizers

print(f"[DyME] tokenizers ready: {tokenizers.__version__}")
PY
}

# Read deplot.enabled from the Python training config (fallback: enabled).
config_deplot_enabled() {
  local cfg="${1:-}"
  if [[ -n "${DYME_DEPLOT_ENABLED:-}" ]]; then
    echo "${DYME_DEPLOT_ENABLED}"
    return
  fi
  if [[ -z "${cfg}" ]]; then
    echo "1"
    return
  fi
  python -c "
from config.loader import load_config
cfg = load_config('${cfg}')
print(1 if cfg.get('deplot', {}).get('enabled', True) else 0)
"
}

prepare_chartqa_training_data() {
  local cfg="${1:-}"
  export WANDB_MODE="${WANDB_MODE:-disabled}"
  export WANDB_DISABLED="${WANDB_DISABLED:-true}"
  export DYME_DEPLOT_ENABLED="${DYME_DEPLOT_ENABLED:-$(config_deplot_enabled "${cfg}")}"
  ensure_chartqa_vf_full
  ensure_tokenizers
  ensure_spacy_model
}

train_log_path() {
  local prefix="${1:-train}"
  local log_dir="${DYME_LOG_DIR:-./outputs/logs}"
  mkdir -p "${log_dir}"
  echo "${log_dir}/${prefix}_$(date +%Y%m%d_%H%M%S).log"
}
