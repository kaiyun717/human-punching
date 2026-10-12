#!/usr/bin/env bash
# Default: all four punches on one GPU. --array: four independent one-GPU tasks.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"
array_args=()
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: bash scripts/submit_gpu.sh [--array] [preview CLI options]'
  echo 'Default: all four punches on one Volta GPU. --array: one punch per GPU task.'
  exit 0
fi
if [[ "${1:-}" == --array ]]; then array_args=(--array=0-3%4 --output=outputs/slurm/%x-%A_%a.log); shift; fi
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing project .venv; follow README setup first.' >&2; exit 2
fi
mkdir -p outputs/slurm
exec sbatch --parsable "${array_args[@]}" scripts/render_volta.sbatch "$@"
