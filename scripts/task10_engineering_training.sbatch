#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  exit 2
fi
task10_learning_root="$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1
exec "$task10_learning_root/.venv-wbc/bin/python" -B "$task10_learning_root/scripts/task10_engineering_training.py" execute --packet "$1"
