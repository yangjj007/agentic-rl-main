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

print_launch_plan() {
  local num_gpus
  num_gpus="$(detect_num_gpus)"
  echo "============================================================"
  echo "Launch plan: --num_processes ${num_gpus}"
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
