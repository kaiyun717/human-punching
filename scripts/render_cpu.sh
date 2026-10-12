#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=osmesa
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$project_root/.cache/xdg"
export TMPDIR="$project_root/.cache/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
exec "$project_root/.venv/bin/python" -B -m human_punching.cli "$@"
