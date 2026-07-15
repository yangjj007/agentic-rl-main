#!/usr/bin/env bash
set -u

cd /home/deepseek_VG/deepseek/agentic-rl-main

log_path="outputs/test-fast/teacher-probe-micro-eval/logs/chartqa_closed_loop_v12_executable_deplot_full_20260715.log"
mkdir -p "$(dirname "${log_path}")"
exec > >(tee -a "${log_path}") 2>&1

for seed in 13 29 31; do
  echo "=== seed ${seed} start $(date -Is) ==="
  /home/deepseek_VG/.conda/envs/dyme/bin/python \
    scripts/analysis/teacher_probe_micro_eval.py \
    --harness chartqa_closed_loop_recovery \
    --max-samples 128 \
    --seed "${seed}" \
    --out-dir "outputs/test-fast/teacher-probe-micro-eval/chartqa_closed_loop_v12_executable_deplot128_seed${seed}_20260715_full"
  echo "=== seed ${seed} done $(date -Is) rc=$? ==="
done
