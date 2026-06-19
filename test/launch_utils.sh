#!/usr/bin/env bash
# Shared helpers for test/ fast baseline launch scripts.
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${TEST_DIR}/.." && pwd)"
cd "${ROOT}"

source "${ROOT}/scripts/launch_utils.sh"

export DYME_DEPLOT_ENABLED="${DYME_DEPLOT_ENABLED:-0}"
export DYME_LOG_DIR="${DYME_LOG_DIR:-${ROOT}/outputs/test-fast/logs}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

prepare_fast_test_data() {
  local cfg="${1:-test/config/config_rlsd_chartqa.py}"
  prepare_chartqa_training_data "${cfg}"
}

fast_train_log_path() {
  local prefix="${1:-train}"
  train_log_path "${prefix}"
}

print_fast_plan() {
  local baseline="${1:-}"
  local config_path="${2:-}"
  local rl_epochs="${DYME_FAST_NUM_TRAIN_EPOCHS:-1}"
  local sft_epochs="${DYME_FAST_SFT_EPOCHS:-1}"
  local est_steps_per_epoch="${DYME_FAST_EST_STEPS_PER_EPOCH:-600}"
  local cold_frac="${DYME_FAST_COLD_START_FRAC:-0.08}"
  local est_rl_steps=$((rl_epochs * est_steps_per_epoch))
  local cold_steps
  cold_steps="$(python - <<PY
frac = float("${cold_frac}")
steps = int("${est_rl_steps}")
print(max(1, int(steps * frac)) if frac > 0 and steps > 0 else 0)
PY
)"
  echo "============================================================"
  echo "test/ fast baseline: ${baseline}"
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
