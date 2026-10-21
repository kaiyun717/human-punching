#!/usr/bin/env bash
# Resources and log paths are supplied by the reviewed submission command.
set -euo pipefail

if [[ $# != 1 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo 'Usage inside an allocated Slurm job: task10_probe.sbatch PREPARED_PACKET' >&2
  exit 2
fi
task10_root="$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)"
if [[ ! -f "$task10_root/src/human_punching/wbc/native_probe.py" ]]; then
  echo 'Submit from the human-punching repository.' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$task10_root/.venv-wbc/bin/python" -B "$task10_root/scripts/task10_probe.py" execute --packet "$1"
