#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Task10B media launcher: one packet path and allocated Slurm job are required.\n' >&2
  exit 2
fi
task10b_media_root="$(pwd -P)"
task10b_media_packet="$(readlink -f -- "$1")"
printf 'Task10B media launcher: root=%s packet=%s\n' "$task10b_media_root" "$task10b_media_packet" >&2
if [[ "$task10b_media_packet" != "$task10b_media_root/outputs/task10b/"* || ! -f "$task10b_media_packet" ]]; then
  printf 'Task10B media launcher: packet missing or outside resolved output root.\n' >&2
  exit 2
fi
source "$task10b_media_root/outputs/task10b/p0prep_20261020_01/runtime.env"
exec "$task10b_media_root/.venv-task10b/bin/python" -B "$task10b_media_root/scripts/task10b_phase0_media.py" execute --packet "$task10b_media_packet"
