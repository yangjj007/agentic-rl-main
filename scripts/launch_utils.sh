#!/usr/bin/env bash
# Shared helpers for accelerate launch scripts.

detect_num_gpus() {
  if [[ -n "${NUM_GPUS:-}" ]]; then
    echo "${NUM_GPUS}"
    return
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
  echo "Detected ${num_gpus} visible GPU(s); launching with --num_processes ${num_gpus}"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  fi
}
