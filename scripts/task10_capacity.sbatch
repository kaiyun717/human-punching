#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)"
exec "$root/.venv-wbc/bin/python" -B "$root/scripts/task10_capacity.py" execute --packet "$1"
