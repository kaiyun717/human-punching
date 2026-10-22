#!/usr/bin/env bash
# Source from the human-punching checkout. All configurable caches are local.
TASK10C_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TASK10C_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$TASK10C_ROOT/.cache/task10c/xdg"
export UV_CACHE_DIR="$TASK10C_ROOT/.cache/task10c/uv"
export PIP_CACHE_DIR="$TASK10C_ROOT/.cache/task10c/pip"
export TMPDIR="$TASK10C_ROOT/.cache/task10c/tmp"
export TORCH_HOME="$TASK10C_ROOT/.cache/task10c/torch"
export TORCH_EXTENSIONS_DIR="$TASK10C_ROOT/.cache/task10c/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$TASK10C_ROOT/.cache/task10c/torchinductor"
export TRITON_CACHE_DIR="$TASK10C_ROOT/.cache/task10c/triton"
export NUMBA_CACHE_DIR="$TASK10C_ROOT/.cache/task10c/numba"
export MPLCONFIGDIR="$TASK10C_ROOT/.cache/task10c/matplotlib"
export CUDA_CACHE_PATH="$TASK10C_ROOT/.cache/task10c/cuda"
export __GL_SHADER_DISK_CACHE_PATH="$TASK10C_ROOT/.cache/task10c/gl"
export WARP_CACHE_PATH="$TASK10C_ROOT/.cache/task10c/warp"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH" "$WARP_CACHE_PATH"
