#!/usr/bin/env bash
# Shared helpers for scripts/test/ fast baseline launch scripts.
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${TEST_DIR}/../.." && pwd)"
cd "${ROOT}"

source "${ROOT}/scripts/launch_utils.sh"

export DYME_DEPLOT_ENABLED="${DYME_DEPLOT_ENABLED:-0}"
export DYME_LOG_DIR="${DYME_LOG_DIR:-${ROOT}/outputs/test-fast/logs}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

prepare_fast_test_data() {
  local cfg="${1:-scripts/test/config/config_rlsd_chartqa.py}"
  prepare_chartqa_training_data "${cfg}"
}

fast_train_log_path() {
  local prefix="${1:-train}"
  train_log_path "${prefix}"
}

# accelerate | tee under set -e: return the training command exit code, not tee's.
run_train_with_log() {
  local log_file="$1"
  shift
  echo "Writing log to: ${log_file}"
  set +o pipefail
  "$@" 2>&1 | tee "${log_file}"
  local train_ec="${PIPESTATUS[0]}"
  set -o pipefail
  if [[ "${train_ec}" -ne 0 ]]; then
    echo "!!! Training exited with code ${train_ec} (log: ${log_file})" >&2
    return "${train_ec}"
  fi
  echo ">>> Training finished OK (log: ${log_file})"
  return 0
}

run_test_baseline() {
  local name="$1"
  local script="$2"
  echo ""
  echo ">>> [BASELINE] ${name} — bash ${script}"
  if bash "${script}"; then
    echo ">>> [BASELINE] ${name} OK"
    return 0
  fi
  local ec=$?
  echo "!!! [BASELINE] ${name} FAILED (exit ${ec})" >&2
  local log_dir="${DYME_LOG_DIR:-${ROOT}/outputs/test-fast/logs}"
  if [[ -d "${log_dir}" ]]; then
    echo "!!! Recent log tails from ${log_dir}:" >&2
    local f
    while IFS= read -r f; do
      [[ -f "${f}" ]] || continue
      echo "----- tail -50 ${f} -----" >&2
      tail -n 50 "${f}" >&2 || true
    done < <(ls -t "${log_dir}"/*.log 2>/dev/null | head -3)
  else
    echo "!!! Log directory not found: ${log_dir}" >&2
  fi
  exit "${ec}"
}

print_fast_plan() {
  local baseline="${1:-}"
  local config_path="${2:-}"
  local rl_epochs="${DYME_FAST_NUM_TRAIN_EPOCHS:-4}"
  local sft_epochs="${DYME_FAST_SFT_EPOCHS:-4}"
  local est_steps_per_epoch="${DYME_FAST_EST_STEPS_PER_EPOCH:-600}"
  local cold_frac="${DYME_FAST_COLD_START_FRAC:-0.08}"
  local est_rl_steps=$((rl_epochs * est_steps_per_epoch))
  local cold_steps
  cold_steps="$("${PYTHON_BIN}" - <<PY
frac = float("${cold_frac}")
steps = int("${est_rl_steps}")
print(max(1, int(steps * frac)) if frac > 0 and steps > 0 else 0)
PY
)"
  echo "============================================================"
  echo "scripts/test/ fast baseline: ${baseline}"
  echo "config: ${config_path}"
  echo "dataset: full train_medium_vf_full.json"
  if [[ "${baseline}" == "sft" ]]; then
    echo "epochs: ${sft_epochs} (offline SFT)"
  else
    echo "epochs: ${rl_epochs} (RL / OPD)"
    echo "estimated RL steps: ~${est_rl_steps} (${rl_epochs} x ${est_steps_per_epoch})"
  fi
  if [[ "${baseline}" == "opd" ]]; then
    echo "OPD cold-start (embedded SFT): ~${cold_steps}/${est_rl_steps} steps (${cold_frac} of total)"
    echo "OPD RL phase after cold start: ~$((est_rl_steps - cold_steps)) steps"
  fi
  echo "output root: outputs/test-fast/"
  print_launch_plan
}
