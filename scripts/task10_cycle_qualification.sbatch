#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  exit 2
fi
task10_cycle_root="$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1
exec "$task10_cycle_root/.venv-wbc/bin/python" -B "$task10_cycle_root/scripts/task10_cycle_qualification.py" execute --packet "$1"
