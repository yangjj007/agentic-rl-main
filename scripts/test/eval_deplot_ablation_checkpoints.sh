#!/usr/bin/env bash
# Evaluate all checkpoints produced by a DePlot OPD ablation run.
#
# The script waits for active DePlot training processes by default so it does not
# compete with ongoing 4-epoch ablations.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${1:-deplot_4epoch_main}"
OUT_ROOT="${DYME_DEPLOT_ABLATION_OUTPUT_ROOT:-outputs/test-fast/opd-deplot-ablation/${RUN_ID}}"
NUM_PROCESSES="${DYME_EVAL_NUM_PROCESSES:-8}"
WAIT_FOR_TRAIN="${DYME_EVAL_WAIT_FOR_TRAIN:-1}"
WAIT_INTERVAL="${DYME_EVAL_WAIT_INTERVAL:-120}"
STABLE_CHECKS="${DYME_EVAL_STABLE_CHECKS:-3}"
FORCE="${DYME_EVAL_FORCE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

timestamp() {
  date "+%F %T"
}

training_active() {
  pgrep -af "main.py --config opd_7b_dyme_probe" >/dev/null 2>&1 && return 0
  pgrep -af "train_opd_7b_dyme_probe.sh" >/dev/null 2>&1 && return 0
  pgrep -af "run_opd_deplot_ablation.sh --run" >/dev/null 2>&1 && return 0
  return 1
}

if [[ "${WAIT_FOR_TRAIN}" == "1" ]]; then
  stable=0
  while [[ "${stable}" -lt "${STABLE_CHECKS}" ]]; do
    if training_active; then
      echo "[$(timestamp)] active DePlot training detected; waiting ${WAIT_INTERVAL}s before eval..."
      stable=0
      sleep "${WAIT_INTERVAL}"
    else
      stable=$((stable + 1))
      echo "[$(timestamp)] no matching training process (${stable}/${STABLE_CHECKS})"
      if [[ "${stable}" -lt "${STABLE_CHECKS}" ]]; then
        sleep "${WAIT_INTERVAL}"
      fi
    fi
  done
fi

GLOBAL_SUMMARY="${OUT_ROOT}/eval_chartqa_summary.csv"
mkdir -p "${OUT_ROOT}"
if [[ ! -f "${GLOBAL_SUMMARY}" || "${FORCE}" == "1" ]]; then
  echo "variant,checkpoint,model_path,accuracy,log_path,status,started_at,finished_at" > "${GLOBAL_SUMMARY}"
fi

mapfile -t VARIANT_DIRS < <(find "${OUT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#VARIANT_DIRS[@]}" -eq 0 ]]; then
  echo "No variant directories found under ${OUT_ROOT}" >&2
  exit 1
fi

for variant_dir in "${VARIANT_DIRS[@]}"; do
  variant="$(basename "${variant_dir}")"
  eval_dir="${variant_dir}/eval_chartqa"
  mkdir -p "${eval_dir}"
  variant_summary="${eval_dir}/summary.csv"
  if [[ ! -f "${variant_summary}" || "${FORCE}" == "1" ]]; then
    echo "variant,checkpoint,model_path,accuracy,log_path,status,started_at,finished_at" > "${variant_summary}"
  fi

  mapfile -t ckpts < <(
    find "${variant_dir}" -mindepth 1 -maxdepth 1 -type d \( -name "checkpoint-*" -o -name "final_checkpoint" \) \
      | sort -V
  )
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "[$(timestamp)] ${variant}: no checkpoints yet; skip"
    continue
  fi

  for ckpt in "${ckpts[@]}"; do
    ckpt_name="$(basename "${ckpt}")"
    latest_existing="$(find "${eval_dir}" -maxdepth 1 -type f -name "eval_${ckpt_name}_*.log" | sort | tail -1 || true)"
    if [[ "${FORCE}" != "1" && -n "${latest_existing}" ]] && grep -q -- "--- Final Report ---" "${latest_existing}"; then
      acc="$(grep -o "Current Global Mean Accuracy: [0-9.]*" "${latest_existing}" | tail -1 | awk '{print $5}')"
      echo "[$(timestamp)] ${variant}/${ckpt_name}: existing eval ${acc:-NA}; skip"
      continue
    fi

    started_at="$(timestamp)"
    log_file="${eval_dir}/eval_${ckpt_name}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(timestamp)] evaluating ${variant}/${ckpt_name} with ${NUM_PROCESSES} processes"
    set +e
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    WANDB_MODE=disabled \
      "${PYTHON_BIN}" -m accelerate.commands.launch \
        --num_processes "${NUM_PROCESSES}" \
        -m eval.eval_chartqa \
        --model_path "${ckpt}" 2>&1 | tee "${log_file}"
    status_code="${PIPESTATUS[0]}"
    set -e
    finished_at="$(timestamp)"
    if [[ "${status_code}" -eq 0 ]] && grep -q "Current Global Mean Accuracy:" "${log_file}"; then
      acc="$(grep -o "Current Global Mean Accuracy: [0-9.]*" "${log_file}" | tail -1 | awk '{print $5}')"
      status="ok"
    else
      acc=""
      status="failed_${status_code}"
    fi
    echo "${variant},${ckpt_name},${ckpt},${acc},${log_file},${status},${started_at},${finished_at}" | tee -a "${variant_summary}" >> "${GLOBAL_SUMMARY}"
  done
done

"${PYTHON_BIN}" - "${GLOBAL_SUMMARY}" "${OUT_ROOT}/eval_chartqa_summary.md" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("accuracy")]
ok_rows.sort(key=lambda r: float(r["accuracy"]), reverse=True)

lines = ["# DePlot OPD Ablation ChartQA Eval Summary", ""]
lines.append(f"- CSV: `{csv_path}`")
if ok_rows:
    best = ok_rows[0]
    lines.append(
        f"- Best: `{best['variant']}/{best['checkpoint']}` accuracy={float(best['accuracy']):.4f}"
    )
else:
    lines.append("- Best: NA")
lines.extend(["", "| variant | checkpoint | accuracy | status | log |", "| --- | --- | ---: | --- | --- |"])
for row in rows:
    acc = row.get("accuracy") or ""
    if acc:
        acc = f"{float(acc):.4f}"
    lines.append(
        f"| {row.get('variant','')} | {row.get('checkpoint','')} | {acc} | {row.get('status','')} | `{row.get('log_path','')}` |"
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {md_path}")
PY

echo "[$(timestamp)] eval queue finished: ${GLOBAL_SUMMARY}"
