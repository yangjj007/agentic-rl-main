#!/usr/bin/env bash
# Long-running main5 10-epoch campaign.
#
# Runs the restored gold-hidden CLRC OPD main variant and the stronger
# short-SFT repair route, then evaluates all produced checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${DYME_MAIN5_CAMPAIGN_RUN_ID:-main5_10ep_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${DYME_MAIN5_CAMPAIGN_OUT_ROOT:-outputs/test-fast/pcd-no-visual/${RUN_ID}}"
LOG_ROOT="${DYME_MAIN5_CAMPAIGN_LOG_ROOT:-outputs/test-fast/logs/pcd_no_visual_${RUN_ID}}"
CAMPAIGN_DIR="${DYME_MAIN5_CAMPAIGN_DIR:-outputs/test-fast/main5-campaign/${RUN_ID}}"
TARGET_ACCURACY="${DYME_MAIN5_TARGET_ACCURACY:-0.67}"
GPU_WAIT_INTERVAL="${DYME_MAIN5_GPU_WAIT_INTERVAL:-300}"
MODEL_WAIT_INTERVAL="${DYME_MAIN5_MODEL_WAIT_INTERVAL:-600}"
REQUIRED_GPUS="${DYME_MAIN5_REQUIRED_GPUS:-8}"
GPU_MAX_USED_MB="${DYME_MAIN5_GPU_MAX_USED_MB:-1000}"
PYTHON_BIN="${PYTHON_BIN:-/home/deepseek_VG/.conda/envs/dyme/bin/python}"

STUDENT_MODEL="${DYME_STUDENT_MODEL:-/data/deepseek_vg/yjj/models/llava-onevision-qwen2-0.5b-ov-hf}"
TEACHER_MODEL="${DYME_TEACHER_MODEL:-/data/deepseek_vg/yjj/models/llava-onevision-qwen2-7b-ov-hf}"
STUDENT_REPO="${DYME_MAIN5_STUDENT_REPO:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}"
TEACHER_REPO="${DYME_MAIN5_TEACHER_REPO:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"

VARIANTS=(
  "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision"
  "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair"
)

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${CAMPAIGN_DIR}" "$(dirname "${STUDENT_MODEL}")" "$(dirname "${TEACHER_MODEL}")"
STATUS_TSV="${CAMPAIGN_DIR}/campaign_status.tsv"
SUCCESS_FILE="${CAMPAIGN_DIR}/SUCCESS_67_PLUS"
ATTENTION_FILE="${CAMPAIGN_DIR}/NEEDS_RESEARCH_OR_CODE_CHANGE"

timestamp() {
  date "+%F %T"
}

log() {
  echo "[$(timestamp)] $*"
}

model_ready() {
  local dir="$1"
  [[ -f "${dir}/config.json" ]] || return 1
  compgen -G "${dir}/model*.safetensors" >/dev/null || compgen -G "${dir}/pytorch_model*.bin" >/dev/null
}

download_model_if_missing() {
  local repo="$1"
  local dir="$2"
  if model_ready "${dir}"; then
    log "model ready: ${dir}"
    return 0
  fi
  log "model missing; downloading ${repo} -> ${dir}"
  HF_HUB_DISABLE_XET=1 \
  "${PYTHON_BIN}" - "${repo}" "${dir}" <<'PY'
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=repo,
    local_dir=str(out),
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "video_processor/*.json"],
    ignore_patterns=["onnx/*", "*.onnx"],
    max_workers=4,
)
PY
}

wait_for_models() {
  until model_ready "${STUDENT_MODEL}" && model_ready "${TEACHER_MODEL}"; do
    set +e
    download_model_if_missing "${STUDENT_REPO}" "${STUDENT_MODEL}"
    student_rc=$?
    download_model_if_missing "${TEACHER_REPO}" "${TEACHER_MODEL}"
    teacher_rc=$?
    set -e
    if [[ "${student_rc}" -eq 0 && "${teacher_rc}" -eq 0 ]] && model_ready "${STUDENT_MODEL}" && model_ready "${TEACHER_MODEL}"; then
      break
    fi
    log "models not ready; retry in ${MODEL_WAIT_INTERVAL}s"
    sleep "${MODEL_WAIT_INTERVAL}"
  done
}

free_gpu_count() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -v max_used="${GPU_MAX_USED_MB}" '$1 <= max_used {count++} END {print count + 0}'
}

training_active() {
  pgrep -af "main.py --config opd_7b_dyme_probe" >/dev/null 2>&1 && return 0
  pgrep -af "train_opd_7b_dyme_probe.sh" >/dev/null 2>&1 && return 0
  return 1
}

wait_for_gpus() {
  while true; do
    local free_count
    free_count="$(free_gpu_count)"
    if [[ "${free_count}" -ge "${REQUIRED_GPUS}" ]] && ! training_active; then
      log "GPU gate passed: free=${free_count}/${REQUIRED_GPUS}, max_used_mb=${GPU_MAX_USED_MB}"
      return 0
    fi
    log "waiting for GPUs: free=${free_count}/${REQUIRED_GPUS}, active_training=$(training_active && echo 1 || echo 0)"
    sleep "${GPU_WAIT_INTERVAL}"
  done
}

latest_checkpoint() {
  local variant_dir="$1"
  find "${variant_dir}" -mindepth 1 -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1
}

run_variant() {
  local variant="$1"
  local variant_dir="${OUT_ROOT}/${variant}"
  local resume_args=(--resume none)
  if [[ -d "${variant_dir}/final_checkpoint" ]]; then
    log "${variant}: final_checkpoint exists; skip training"
    return 0
  fi
  if [[ -n "$(latest_checkpoint "${variant_dir}")" ]]; then
    resume_args=(--resume auto)
  fi
  wait_for_models
  wait_for_gpus
  log "${variant}: start 10epoch training (${resume_args[*]})"
  echo -e "$(timestamp)\ttrain_start\t${variant}\t${resume_args[*]}" >> "${STATUS_TSV}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  DYME_STUDENT_MODEL="${STUDENT_MODEL}" \
  DYME_TEACHER_MODEL="${TEACHER_MODEL}" \
  DYME_PCD_RUN_ID="${RUN_ID}" \
  DYME_PCD_OUTPUT_ROOT="${OUT_ROOT}" \
  DYME_PCD_LOG_ROOT="${LOG_ROOT}" \
  bash scripts/test/run_pcd_no_visual.sh 10 "${resume_args[@]}" --variant "${variant}"
  echo -e "$(timestamp)\ttrain_done\t${variant}\t0" >> "${STATUS_TSV}"
}

evaluate_all() {
  wait_for_gpus
  log "start eval for ${RUN_ID}"
  echo -e "$(timestamp)\teval_start\tall\t${OUT_ROOT}" >> "${STATUS_TSV}"
  DYME_DEPLOT_ABLATION_OUTPUT_ROOT="${OUT_ROOT}" \
  DYME_EVAL_NUM_PROCESSES="${DYME_EVAL_NUM_PROCESSES:-8}" \
  DYME_EVAL_WAIT_FOR_TRAIN=1 \
  bash scripts/test/eval_deplot_ablation_checkpoints.sh "${RUN_ID}"
  echo -e "$(timestamp)\teval_done\tall\t0" >> "${STATUS_TSV}"
}

best_accuracy() {
  local summary="${OUT_ROOT}/eval_chartqa_summary.csv"
  [[ -f "${summary}" ]] || {
    echo ""
    return 0
  }
  "${PYTHON_BIN}" - "${summary}" <<'PY'
import csv, sys
best = None
for row in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    if row.get("status") != "ok" or not row.get("accuracy"):
        continue
    acc = float(row["accuracy"])
    best = acc if best is None or acc > best else best
print("" if best is None else f"{best:.6f}")
PY
}

{
  log "campaign start: run_id=${RUN_ID}"
  log "out_root=${OUT_ROOT}"
  log "log_root=${LOG_ROOT}"
  log "campaign_dir=${CAMPAIGN_DIR}"
  log "target_accuracy=${TARGET_ACCURACY}"
  printf "timestamp\tevent\tvariant\tdetail\n" > "${STATUS_TSV}"
  for variant in "${VARIANTS[@]}"; do
    run_variant "${variant}"
  done
  evaluate_all
  best="$(best_accuracy)"
  log "best_accuracy=${best:-NA}"
  if [[ -n "${best}" ]] && "${PYTHON_BIN}" - "${best}" "${TARGET_ACCURACY}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
  then
    echo "best_accuracy=${best}" | tee "${SUCCESS_FILE}"
    rm -f "${ATTENTION_FILE}"
  else
    {
      echo "best_accuracy=${best:-NA}"
      echo "target_accuracy=${TARGET_ACCURACY}"
      echo "next_action=inspect health/eval artifacts, research method gap, patch training flow, restart campaign"
    } | tee "${ATTENTION_FILE}"
  fi
  log "campaign finished"
} 2>&1 | tee -a "${CAMPAIGN_DIR}/campaign.log"
