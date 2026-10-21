#!/usr/bin/env bash
set -euo pipefail
task10b_grouped_media_root="$(pwd -P)"
if [[ $# != 1 || -z "${SLURM_JOB_ID:-}" ]]; then exit 2; fi
task10b_grouped_media_packet="$(readlink -f -- "$1")"
if [[ "$task10b_grouped_media_packet" != "$task10b_grouped_media_root/outputs/task10b/"* ]]; then exit 2; fi
source "$task10b_grouped_media_root/outputs/task10b/p2prep_20261021_01/runtime.env"
cd -- "$(dirname -- "$task10b_grouped_media_packet")"
exec "$task10b_grouped_media_root/.venv-task10b/bin/python" -B "$task10b_grouped_media_root/scripts/task10b_phase2_grouped_media_v1.py" execute --packet "$task10b_grouped_media_packet"
