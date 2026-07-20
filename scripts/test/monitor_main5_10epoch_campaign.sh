#!/usr/bin/env bash
# Periodic health monitor for run_main5_10epoch_campaign.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${DYME_MAIN5_CAMPAIGN_RUN_ID:?set DYME_MAIN5_CAMPAIGN_RUN_ID}"
OUT_ROOT="${DYME_MAIN5_CAMPAIGN_OUT_ROOT:-outputs/test-fast/pcd-no-visual/${RUN_ID}}"
LOG_ROOT="${DYME_MAIN5_CAMPAIGN_LOG_ROOT:-outputs/test-fast/logs/pcd_no_visual_${RUN_ID}}"
CAMPAIGN_DIR="${DYME_MAIN5_CAMPAIGN_DIR:-outputs/test-fast/main5-campaign/${RUN_ID}}"
SLEEP_SECONDS="${DYME_MAIN5_MONITOR_INTERVAL:-1800}"
PYTHON_BIN="${PYTHON_BIN:-/home/deepseek_VG/.conda/envs/dyme/bin/python}"
ATTENTION_FILE="${CAMPAIGN_DIR}/NEEDS_ATTENTION_HEALTH"

mkdir -p "${CAMPAIGN_DIR}"

timestamp() {
  date "+%F %T"
}

latest_train_log() {
  local variant="$1"
  find "${LOG_ROOT}/${variant}" -maxdepth 1 -type f -name "train_opd_7b_dyme_probe_*.log" 2>/dev/null | sort | tail -1
}

checkpoint_count() {
  local variant_dir="$1"
  find "${variant_dir}" -mindepth 1 -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | wc -l
}

log_epoch_state() {
  local log_file="$1"
  "${PYTHON_BIN}" - "${log_file}" <<'PY'
import ast, json, re, sys
path = sys.argv[1]
rows = []
for line in open(path, encoding="utf-8", errors="ignore"):
    m = re.search(r"\{.*\}", line)
    if not m:
        continue
    try:
        row = ast.literal_eval(m.group())
    except Exception:
        continue
    if isinstance(row, dict) and ("epoch" in row or "global_step" in row):
        rows.append(row)
if not rows:
    print(json.dumps({"rows": 0}))
else:
    last = rows[-1]
    keys = [
        "epoch",
        "global_step",
        "global_signal/accuracy_reward_mean",
        "rewards/accuracy/mean",
        "signal/grpo_zero_loss_rate",
        "routing/grpo_route_rate",
        "routing/opd_route_rate",
        "routing/sft_route_rate",
        "routing/teacher_sft_repair_rate",
        "completions/degenerate_rate",
    ]
    print(json.dumps({k: last.get(k) for k in keys if k in last}, sort_keys=True))
PY
}

check_once() {
  local any=0
  while IFS= read -r variant_dir; do
    [[ -d "${variant_dir}" ]] || continue
    any=1
    variant="$(basename "${variant_dir}")"
    train_log="$(latest_train_log "${variant}")"
    ckpts="$(checkpoint_count "${variant_dir}" | tr -d ' ')"
    if [[ -z "${train_log}" ]]; then
      echo "{\"timestamp\":\"$(timestamp)\",\"variant\":\"${variant}\",\"status\":\"no_train_log\",\"checkpoints\":${ckpts}}" \
        | tee -a "${CAMPAIGN_DIR}/health_snapshots.jsonl"
      continue
    fi
    health_args=(
      "${train_log}"
      --window 20
      --candidate-dir "${variant_dir}/teacher_probe_candidates"
    )
    if [[ "${variant}" == *"sft_repair"* ]]; then
      health_args+=(--allow-teacher-sft-repair)
    fi
    health_json="$("${PYTHON_BIN}" scripts/analysis/check_opd_template_health.py \
      "${health_args[@]}" 2>/dev/null || true)"
    epoch_json="$(log_epoch_state "${train_log}")"
    echo "{\"timestamp\":\"$(timestamp)\",\"variant\":\"${variant}\",\"checkpoints\":${ckpts},\"epoch\":${epoch_json},\"health\":${health_json:-null}}" \
      | tee -a "${CAMPAIGN_DIR}/health_snapshots.jsonl"
    if [[ "${health_json}" == *'"status": "template_collapse"'* ||
          "${health_json}" == *'"status": "template_drift"'* ||
          "${health_json}" == *'"status": "clip_sft_collapse"'* ||
          "${health_json}" == *'"status": "mechanism_violation"'* ]]; then
      {
        echo "timestamp=$(timestamp)"
        echo "variant=${variant}"
        echo "train_log=${train_log}"
        echo "health=${health_json}"
      } > "${ATTENTION_FILE}"
    fi
  done < <(find "${OUT_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
  if [[ "${any}" -eq 0 ]]; then
    echo "{\"timestamp\":\"$(timestamp)\",\"status\":\"no_variant_dirs\"}" \
      | tee -a "${CAMPAIGN_DIR}/health_snapshots.jsonl"
  fi
}

while true; do
  if [[ -f "${CAMPAIGN_DIR}/SUCCESS_67_PLUS" ]]; then
    echo "[$(timestamp)] success file present; monitor exits" | tee -a "${CAMPAIGN_DIR}/monitor.log"
    exit 0
  fi
  check_once 2>&1 | tee -a "${CAMPAIGN_DIR}/monitor.log"
  sleep "${SLEEP_SECONDS}"
done
