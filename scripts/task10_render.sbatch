#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo 'Submit recorded-state rendering from human-punching with source/output paths.' >&2
  exit 2
fi
task10_render_root="$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)"
if [[ ! -f "$task10_render_root/src/human_punching/wbc/render_trace.py" ]]; then
  echo 'Submit directory must be human-punching.' >&2
  exit 2
fi
task10_render_packet="$(dirname -- "$2")"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$task10_render_packet/cache/xdg"
export CUDA_CACHE_PATH="$task10_render_packet/cache/cuda"
export TMPDIR="$task10_render_packet/cache/tmp"
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TMPDIR"
# The Python renderer verifies NVIDIA EGL/CUDA UUID identity in its own process.
exec "$task10_render_root/.venv-wbc/bin/python" -B -u -m human_punching.wbc.render_trace --source "$1" --output "$2"
