#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

EPOCHS="${1:-4}"
if [[ $# -gt 0 ]]; then
  shift
fi
RUNNER_ARGS=("$@")
RUNNER="${DYME_PCD_RUNNER:-${ROOT}/scripts/test/run_pcd_no_visual.sh}"
RUN_ID="${DYME_PCD_RUN_ID:-pcd_no_visual_staged}"
OUT_ROOT="${DYME_PCD_OUTPUT_ROOT:-outputs/test-fast/pcd-no-visual/${RUN_ID}}"
VARIANT="${DYME_PCD_VARIANT:-deplot_no_vs_opd_pcd}"
for ((i = 0; i < ${#RUNNER_ARGS[@]}; i++)); do
  if [[ "${RUNNER_ARGS[$i]}" == "--variant" && $((i + 1)) -lt ${#RUNNER_ARGS[@]} ]]; then
    VARIANT="${RUNNER_ARGS[$((i + 1))]}"
  fi
done
OUT_DIR="${OUT_ROOT}/${VARIANT}"

STATE_DIR="${DYME_RESILIENT_STATE_DIR:-outputs/test-fast/long-runs/${RUN_ID}/resilient}"
MAX_USED_MIB="${DYME_GPU_MAX_USED_MIB:-7168}"
MAX_TEMP_C="${DYME_GPU_MAX_TEMP_C:-70}"
MAX_UTIL_PCT="${DYME_GPU_MAX_UTIL_PCT:-10}"
STABLE_SAMPLES="${DYME_GPU_STABLE_SAMPLES:-3}"
POLL_SECONDS="${DYME_GPU_POLL_SECONDS:-60}"
MAX_RETRIES="${DYME_MAX_RETRIES:-3}"
RETRY_WAIT_SECONDS="${DYME_RETRY_WAIT_SECONDS:-300}"
mkdir -p "${STATE_DIR}"

latest_checkpoint() {
  [[ -d "${OUT_DIR}" ]] || return 0
  find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | sort -V | tail -1
}

gpu_gate_passes() {
  local rows process_rows index used temp util count=0
  if ! rows="$(nvidia-smi --query-gpu=index,memory.used,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    return 1
  fi
  while IFS=',' read -r index used temp util; do
    index="${index//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    temp="${temp//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    [[ "${index}" =~ ^[0-9]+$ && "${used}" =~ ^[0-9]+$ && "${temp}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] || return 1
    if (( used > MAX_USED_MIB || temp > MAX_TEMP_C || util > MAX_UTIL_PCT )); then
      echo "gpu gate blocked: gpu=${index} used_mib=${used}/${MAX_USED_MIB} temp_c=${temp}/${MAX_TEMP_C} util_pct=${util}/${MAX_UTIL_PCT}"
      return 1
    fi
    count=$((count + 1))
  done <<<"${rows}"
  [[ "${count}" -eq 8 ]] || return 1

  if ! process_rows="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "gpu gate blocked: compute process query unavailable"
    return 1
  fi
  if [[ -n "${process_rows//[[:space:]]/}" ]]; then
    local gpu_uuid pid process used_mib
    IFS=',' read -r gpu_uuid pid process used_mib <<<"${process_rows}"
    gpu_uuid="${gpu_uuid##+([[:space:]])}"
    pid="${pid//[[:space:]]/}"
    process="${process##+([[:space:]])}"
    used_mib="${used_mib//[[:space:]]/}"
    echo "gpu gate blocked: compute process pid=${pid:-unknown} used_mib=${used_mib:-unknown} process=${process:-unknown} gpu_uuid=${gpu_uuid:-unknown}"
    return 1
  fi
  return 0
}

wait_for_gpu_gate() {
  local stable=0
  while (( stable < STABLE_SAMPLES )); do
    if gpu_gate_passes; then
      stable=$((stable + 1))
      echo "gpu gate ready sample ${stable}/${STABLE_SAMPLES}"
    else
      stable=0
    fi
    if (( stable < STABLE_SAMPLES )); then
      sleep "${POLL_SECONDS}"
    fi
  done
}

is_transient_failure() {
  local log_file="$1"
  rg -qi \
    'CUDA error: unspecified launch failure|CUDA error: an illegal memory access|NCCL[^[:cntrl:]]*(error|timeout)|DistBackendError|NVRM: Xid|GPU has fallen off the bus' \
    "${log_file}"
}

attempt=0
while (( attempt <= MAX_RETRIES )); do
  attempt=$((attempt + 1))
  wait_for_gpu_gate

  resume_mode="none"
  if [[ -n "$(latest_checkpoint)" ]]; then
    resume_mode="auto"
  fi
  attempt_log="${STATE_DIR}/attempt_${attempt}.log"
  echo "starting attempt=${attempt} resume=${resume_mode} log=${attempt_log}"

  set +e
  bash "${RUNNER}" "${EPOCHS}" --resume "${resume_mode}" "${RUNNER_ARGS[@]}" 2>&1 | tee "${attempt_log}"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    echo "training completed on attempt=${attempt}"
    exit 0
  fi

  if ! is_transient_failure "${attempt_log}"; then
    echo "non-transient failure; refusing automatic retry (rc=${rc})"
    exit "${rc}"
  fi
  if (( attempt > MAX_RETRIES )); then
    echo "transient failure retry budget exhausted (rc=${rc})"
    exit "${rc}"
  fi

  checkpoint="$(latest_checkpoint)"
  if [[ -n "${checkpoint}" ]]; then
    echo "transient failure detected; next attempt will resume from ${checkpoint}"
  elif [[ -d "${OUT_DIR}" ]]; then
    archive="${OUT_DIR}.failed-attempt-${attempt}-$(date +%Y%m%d_%H%M%S)"
    mv "${OUT_DIR}" "${archive}"
    echo "transient failure before first checkpoint; archived partial output at ${archive}"
  else
    echo "transient failure before first checkpoint; next attempt will restart cleanly"
  fi
  sleep "${RETRY_WAIT_SECONDS}"
done
