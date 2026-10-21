#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo 'Submit the prepared bounded guard session from human-punching.' >&2
  exit 2
fi
task10_guard_root="$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)"
if [[ ! -f "$task10_guard_root/src/human_punching/wbc/guard_training.py" ]]; then
  echo 'The submit directory must be human-punching.' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$task10_guard_root/.venv-wbc/bin/python" -B "$task10_guard_root/scripts/task10_guard_learning.py" execute --packet "$1"
