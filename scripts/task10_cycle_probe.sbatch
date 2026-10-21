#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo 'Submit the prepared cycle diagnostic from human-punching.' >&2
  exit 2
fi
task10_cycle_root="$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)"
if [[ ! -f "$task10_cycle_root/src/human_punching/wbc/native_cycle_probe.py" ]]; then
  echo 'The submit directory must be human-punching.' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$task10_cycle_root/.venv-wbc/bin/python" -B "$task10_cycle_root/scripts/task10_cycle_probe.py" execute --packet "$1"
