#!/usr/bin/env bash
# Resource/log declarations are fixed by the prepared packet's submission command.
set -euo pipefail
if [[ $# != 1 || -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo 'Use task10_migration.sbatch PREPARED_PACKET inside an allocated Slurm GPU job.' >&2
  exit 2
fi
# Slurm executes a spooled copy; BASH_SOURCE points into /var/spool/slurmd.
# The prepared submission command is issued from the human repository.
task10_migration_root="$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)"
if [[ ! -f "$task10_migration_root/src/human_punching/jab_learning.py" || ! -f "$task10_migration_root/scripts/task10_migration.py" ]]; then
  echo 'Submit the prepared migration packet from the human-punching repository.' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$task10_migration_root/.venv-wbc/bin/python" -B "$task10_migration_root/scripts/task10_migration.py" execute --packet "$1"
