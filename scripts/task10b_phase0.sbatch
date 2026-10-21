#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  printf 'Task10B Phase0 launcher: expected one packet path, received %s arguments.\n' "$#" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Task10B Phase0 launcher: an allocated Slurm job is required.\n' >&2
  exit 2
fi
# sbatch --chdir fixes the job cwd; SLURM_SUBMIT_DIR remains the caller's cwd.
task10b_phase0_root="$(pwd -P)"
# Login and compute nodes can expose the same NFS tree through different aliases.
# Compare resolved paths, not the submitted spelling against physical pwd.
if ! task10b_phase0_packet="$(readlink -f -- "$1")"; then
  printf 'Task10B Phase0 launcher: cannot resolve packet path: %s\n' "$1" >&2
  exit 2
fi
printf 'Task10B Phase0 launcher: resolved_root=%s resolved_packet=%s submitted_packet=%s\n' "$task10b_phase0_root" "$task10b_phase0_packet" "$1" >&2
if [[ "$task10b_phase0_packet" != "$task10b_phase0_root/outputs/task10b/"* || ! -f "$task10b_phase0_packet" ]]; then
  printf 'Task10B Phase0 launcher: packet is missing or outside the resolved Task10B output root.\n' >&2
  exit 2
fi
if [[ ! -f "$task10b_phase0_root/scripts/task10b_phase0.py" ]]; then
  printf 'Task10B Phase0 launcher: wrapper is missing beneath resolved root: %s\n' "$task10b_phase0_root" >&2
  exit 2
fi
if [[ ! -r "$task10b_phase0_root/outputs/task10b/p0prep_20261020_01/runtime.env" || ! -x "$task10b_phase0_root/.venv-task10b/bin/python" ]]; then
  printf 'Task10B Phase0 launcher: local runtime configuration or executable is unavailable beneath: %s\n' "$task10b_phase0_root" >&2
  exit 2
fi
source "$task10b_phase0_root/outputs/task10b/p0prep_20261020_01/runtime.env"
exec "$task10b_phase0_root/.venv-task10b/bin/python" -B "$task10b_phase0_root/scripts/task10b_phase0.py" execute --packet "$task10b_phase0_packet"
