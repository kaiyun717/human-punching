#!/usr/bin/env bash
#SBATCH --job-name=human-physics-review
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.log
set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from human-punching}"
cd "$project_root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$project_root/src"
export XDG_CACHE_HOME="$project_root/.cache/gpu/$SLURM_JOB_ID/xdg"
export CUDA_CACHE_PATH="$project_root/.cache/gpu/$SLURM_JOB_ID/cuda"
export __GL_SHADER_DISK_CACHE_PATH="$project_root/.cache/gpu/$SLURM_JOB_ID/gl"
export TMPDIR="$project_root/.cache/gpu/$SLURM_JOB_ID/tmp"
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH" "$TMPDIR"
run_dir="$project_root/outputs/physical-$SLURM_JOB_ID"
mkdir -p "$run_dir"
.venv/bin/python -B scripts/gpu_preflight.py --output "$run_dir/gpu_runtime.json" --env-output "$run_dir/egl_device.env"
source "$run_dir/egl_device.env"
.venv/bin/python -B scripts/render_physical_review.py
