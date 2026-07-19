#!/usr/bin/env bash
# Long-running main5 10-epoch campaign.
#
# Runs the stronger refiner-SFT repair route and evaluates all produced checkpoints.
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
GPU_MAX_USED_MB="${DYME_MAIN5_GPU_MAX_USED_MB:-20000}"
GPU_MAX_UTIL_PCT="${DYME_MAIN5_GPU_MAX_UTIL_PCT:-20}"
TRAIN_MAX_ATTEMPTS="${DYME_MAIN5_TRAIN_MAX_ATTEMPTS:-3}"
TRAIN_RETRY_DELAY="${DYME_MAIN5_TRAIN_RETRY_DELAY:-300}"
PYTHON_BIN="${PYTHON_BIN:-/home/deepseek_VG/.conda/envs/dyme/bin/python}"

STUDENT_MODEL="${DYME_STUDENT_MODEL:-/data/deepseek_vg/yjj/models/llava-onevision-qwen2-0.5b-ov-hf}"
TEACHER_MODEL="${DYME_TEACHER_MODEL:-/data/deepseek_vg/yjj/models/llava-onevision-qwen2-7b-ov-hf}"
STUDENT_REPO="${DYME_MAIN5_STUDENT_REPO:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}"
TEACHER_REPO="${DYME_MAIN5_TEACHER_REPO:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"

VARIANTS=(
  "deplot_no_vs_opd_pcd_gold_hidden_opd_no_full_hint_hard_sft_adaptive_supervision_sft_repair"
)

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${CAMPAIGN_DIR}" "$(dirname "${STUDENT_MODEL}")" "$(dirname "${TEACHER_MODEL}")"
STATUS_TSV="${CAMPAIGN_DIR}/campaign_status.tsv"
SUCCESS_FILE="${CAMPAIGN_DIR}/SUCCESS_67_PLUS"
ATTENTION_FILE="${CAMPAIGN_DIR}/NEEDS_RESEARCH_OR_CODE_CHANGE"
SELECTED_CUDA_VISIBLE_DEVICES=""

timestamp() {
  date "+%F %T"
}

log() {
  echo "[$(timestamp)] $*"
}

model_ready() {
  local dir="$1"
  [[ -f "${dir}/config.json" ]] || return 1
  local safetensors_index="${dir}/model.safetensors.index.json"
  if [[ -f "${safetensors_index}" ]]; then
    "${PYTHON_BIN}" - "${dir}" "${safetensors_index}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
index_path = Path(sys.argv[2])
try:
    index = json.loads(index_path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"invalid model.safetensors.index.json: {exc}", file=sys.stderr)
    raise SystemExit(1)

weight_map = index.get("weight_map") or {}
expected = sorted(set(str(name) for name in weight_map.values()))
missing = [name for name in expected if not (model_dir / name).is_file()]
if missing:
    print("missing model shard: " + ", ".join(missing[:8]), file=sys.stderr)
    raise SystemExit(1)
raise SystemExit(0 if expected else 1)
PY
    return $?
  fi
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
    max_workers=1,
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

ready_gpu_indices() {
  local gpu_csv
  gpu_csv="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)"
  GPU_CSV="${gpu_csv}" "${PYTHON_BIN}" - "${GPU_MAX_USED_MB}" "${GPU_MAX_UTIL_PCT}" <<'PY'
import os
import sys

max_used = float(sys.argv[1])
max_util = float(sys.argv[2])
for line in os.environ.get("GPU_CSV", "").splitlines():
    parts = line.rstrip("\n").split(",")
    if len(parts) != 3:
        continue
    idx, used, util = parts
    try:
        if float(used.strip()) <= max_used and float(util.strip()) <= max_util:
            print(idx.strip())
    except ValueError:
        continue
PY
}

free_gpu_count() {
  ready_gpu_indices | wc -l | tr -d ' '
}

select_ready_gpus() {
  ready_gpu_indices | head -n "${REQUIRED_GPUS}" | paste -sd, -
}

training_active() {
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import sys

TRAIN_SCRIPTS = (
    b"scripts/test/run_pcd_no_visual.sh",
    b"scripts/train_opd_7b_dyme_probe.sh",
)


def _is_script_arg(arg: bytes, suffix: bytes) -> bool:
    return arg == suffix or arg.endswith(b"/" + suffix)


for proc_dir in Path("/proc").iterdir():
    if not proc_dir.name.isdigit():
        continue
    try:
        cmdline = (proc_dir / "cmdline").read_bytes()
    except OSError:
        continue
    parts = [part for part in cmdline.split(b"\0") if part]
    if not parts:
        continue
    exe = os.path.basename(parts[0])
    if (
        exe.startswith(b"python")
        and b"main.py" in parts
        and b"--config" in parts
        and b"opd_7b_dyme_probe" in parts
    ):
        raise SystemExit(0)
    if exe in (b"bash", b"sh") and len(parts) >= 2:
        if any(_is_script_arg(parts[1], script) for script in TRAIN_SCRIPTS):
            raise SystemExit(0)

raise SystemExit(1)
PY
}

wait_for_gpus() {
  while true; do
    local free_count
    free_count="$(free_gpu_count)"
    if [[ "${free_count}" -ge "${REQUIRED_GPUS}" ]] && ! training_active; then
      SELECTED_CUDA_VISIBLE_DEVICES="$(select_ready_gpus)"
      log "GPU gate passed: free=${free_count}/${REQUIRED_GPUS}, max_used_mb=${GPU_MAX_USED_MB}, max_util_pct=${GPU_MAX_UTIL_PCT}, devices=${SELECTED_CUDA_VISIBLE_DEVICES}"
      return 0
    fi
    log "waiting for GPUs: free=${free_count}/${REQUIRED_GPUS}, max_used_mb=${GPU_MAX_USED_MB}, max_util_pct=${GPU_MAX_UTIL_PCT}, active_training=$(training_active && echo 1 || echo 0)"
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
  local attempt=1
  local train_rc=0
  while true; do
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
    local cuda_devices="${SELECTED_CUDA_VISIBLE_DEVICES}"
    log "${variant}: start 10epoch training (${resume_args[*]}, attempt=${attempt}/${TRAIN_MAX_ATTEMPTS}, devices=${cuda_devices})"
    echo -e "$(timestamp)\ttrain_start\t${variant}\t${resume_args[*]} attempt=${attempt} devices=${cuda_devices}" >> "${STATUS_TSV}"
    set +e
    CUDA_VISIBLE_DEVICES="${cuda_devices}" \
    NUM_GPUS="${REQUIRED_GPUS}" \
    DYME_STUDENT_MODEL="${STUDENT_MODEL}" \
    DYME_TEACHER_MODEL="${TEACHER_MODEL}" \
    DYME_PCD_RUN_ID="${RUN_ID}" \
    DYME_PCD_OUTPUT_ROOT="${OUT_ROOT}" \
    DYME_PCD_LOG_ROOT="${LOG_ROOT}" \
    bash scripts/test/run_pcd_no_visual.sh 10 "${resume_args[@]}" --variant "${variant}"
    train_rc=$?
    set -e
    if [[ "${train_rc}" -eq 0 ]]; then
      echo -e "$(timestamp)\ttrain_done\t${variant}\t0" >> "${STATUS_TSV}"
      return 0
    fi
    echo -e "$(timestamp)\ttrain_failed\t${variant}\texit_code=${train_rc} attempt=${attempt}" >> "${STATUS_TSV}"
    {
      echo "variant=${variant}"
      echo "train_exit_code=${train_rc}"
      echo "attempt=${attempt}"
      echo "next_action=wait for idle GPUs and retry training"
    } > "${ATTENTION_FILE}"
    if [[ "${TRAIN_MAX_ATTEMPTS}" != "0" && "${attempt}" -ge "${TRAIN_MAX_ATTEMPTS}" ]]; then
      log "${variant}: training failed after ${attempt} attempt(s)"
      return "${train_rc}"
    fi
    log "${variant}: training failed with exit code ${train_rc}; retry in ${TRAIN_RETRY_DELAY}s"
    sleep "${TRAIN_RETRY_DELAY}"
    attempt=$((attempt + 1))
  done
}

evaluate_all() {
  wait_for_gpus
  local cuda_devices="${SELECTED_CUDA_VISIBLE_DEVICES}"
  log "start eval for ${RUN_ID} (devices=${cuda_devices})"
  echo -e "$(timestamp)\teval_start\tall\t${OUT_ROOT} devices=${cuda_devices}" >> "${STATUS_TSV}"
  CUDA_VISIBLE_DEVICES="${cuda_devices}" \
  DYME_DEPLOT_ABLATION_OUTPUT_ROOT="${OUT_ROOT}" \
  DYME_EVAL_NUM_PROCESSES="${DYME_EVAL_NUM_PROCESSES:-${REQUIRED_GPUS}}" \
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
  if [[ ! -s "${STATUS_TSV}" ]]; then
    printf "timestamp\tevent\tvariant\tdetail\n" > "${STATUS_TSV}"
  fi
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
