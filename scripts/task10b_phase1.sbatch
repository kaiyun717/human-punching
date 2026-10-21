#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Task10B Phase1 requires one packet path and an allocated Slurm job.\n' >&2
  exit 2
fi
task10b_phase1_root="$(pwd -P)"
task10b_phase1_packet="$(readlink -f -- "$1")"
if [[ "$task10b_phase1_packet" != "$task10b_phase1_root/outputs/task10b/"* || ! -f "$task10b_phase1_packet" ]]; then
  printf 'Task10B Phase1 packet is missing or outside the resolved Task10B output root.\n' >&2
  exit 2
fi
printf 'Task10B Phase1 resolved_root=%s resolved_packet=%s\n' "$task10b_phase1_root" "$task10b_phase1_packet" >&2
source "$task10b_phase1_root/outputs/task10b/p1prep_20261020_01/runtime.env"
exec "$task10b_phase1_root/.venv-task10b/bin/python" -B "$task10b_phase1_root/scripts/task10b_phase1.py" execute --packet "$task10b_phase1_packet"
