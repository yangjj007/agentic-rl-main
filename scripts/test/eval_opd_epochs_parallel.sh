#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

MODEL_DIR="${MODEL_DIR:-outputs/opd-7b-dyme-probe}"
OUT_DIR="${OUT_DIR:-outputs/eval_chartqa_opd_epochs}"
mkdir -p "${OUT_DIR}/logs"

CHECKPOINTS=()
while IFS= read -r checkpoint_path; do
  label="$(basename "${checkpoint_path}")"
  CHECKPOINTS+=("${label}:${checkpoint_path}")
done < <(find "${MODEL_DIR}" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)

INCLUDE_FINAL="${INCLUDE_FINAL:-1}"
if [[ "${INCLUDE_FINAL}" == "1" && -d "${MODEL_DIR}/final_checkpoint" ]]; then
  CHECKPOINTS+=("final_checkpoint:${MODEL_DIR}/final_checkpoint")
fi

if (( ${#CHECKPOINTS[@]} == 0 )); then
  echo "[error] no checkpoint-* or final_checkpoint directories under ${MODEL_DIR}" >&2
  exit 2
fi

GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6}"
read -r -a GPUS <<< "${GPU_LIST}"

echo "[start] $(date -Is)" | tee "${OUT_DIR}/run.log"
echo "[config] model_dir=${MODEL_DIR}" | tee -a "${OUT_DIR}/run.log"
echo "[config] out_dir=${OUT_DIR}" | tee -a "${OUT_DIR}/run.log"
echo "[config] gpus=${GPUS[*]} checkpoints=${#CHECKPOINTS[@]}" | tee -a "${OUT_DIR}/run.log"

run_eval() {
  local label="$1"
  local model_path="$2"
  local gpu="$3"
  local log_path="${OUT_DIR}/logs/${label}.log"

  {
    echo "[eval-start] $(date -Is) label=${label} gpu=${gpu} model_path=${model_path}"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONUNBUFFERED=1 \
      HF_DATASETS_OFFLINE=1 \
      HF_HUB_OFFLINE=1 \
      DYME_EVAL_MAX_NEW_TOKENS="${DYME_EVAL_MAX_NEW_TOKENS:-1024}" \
      /home/deepseek_VG/.conda/envs/dyme/bin/accelerate launch \
        --config_file scripts/test/accelerate_single_gpu_no_deepspeed.yaml \
        --num_processes 1 --main_process_port "$((29600 + gpu))" \
        -m eval.eval_chartqa \
        --model_path "${model_path}"
    status=$?
    set -e
    echo "[eval-exit] $(date -Is) label=${label} status=${status}"
    exit "${status}"
  } > "${log_path}" 2>&1
}

declare -A PID_TO_GPU=()
declare -A PID_TO_LABEL=()
next_idx=0

while (( next_idx < ${#CHECKPOINTS[@]} || ${#PID_TO_GPU[@]} > 0 )); do
  for gpu in "${GPUS[@]}"; do
    if (( next_idx >= ${#CHECKPOINTS[@]} )); then
      break
    fi

    gpu_busy=0
    for pid in "${!PID_TO_GPU[@]}"; do
      if [[ "${PID_TO_GPU[$pid]}" == "${gpu}" ]] && kill -0 "${pid}" 2>/dev/null; then
        gpu_busy=1
        break
      fi
    done
    if (( gpu_busy )); then
      continue
    fi

    item="${CHECKPOINTS[$next_idx]}"
    label="${item%%:*}"
    model_path="${item#*:}"
    run_eval "${label}" "${model_path}" "${gpu}" &
    pid=$!
    PID_TO_GPU["${pid}"]="${gpu}"
    PID_TO_LABEL["${pid}"]="${label}"
    echo "[launch] $(date -Is) pid=${pid} gpu=${gpu} label=${label}" | tee -a "${OUT_DIR}/run.log"
    next_idx=$((next_idx + 1))
  done

  for pid in "${!PID_TO_GPU[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" || true
      echo "[done] $(date -Is) pid=${pid} gpu=${PID_TO_GPU[$pid]} label=${PID_TO_LABEL[$pid]}" | tee -a "${OUT_DIR}/run.log"
      unset "PID_TO_GPU[$pid]"
      unset "PID_TO_LABEL[$pid]"
    fi
  done
  sleep 20
done

python3 scripts/test/parse_eval_chartqa_logs.py "${OUT_DIR}/logs" "${OUT_DIR}/summary.csv" | tee -a "${OUT_DIR}/run.log"
echo "[finish] $(date -Is)" | tee -a "${OUT_DIR}/run.log"
