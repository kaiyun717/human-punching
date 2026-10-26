#!/usr/bin/env bash
#SBATCH --job-name=g1-duck-dynamics
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/ducking/dynamics/slurm-%j.log
set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from human-punching}"
cd "$project_root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$project_root/src"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
for vendor in /usr/share/glvnd/egl_vendor.d/*nvidia*.json /etc/glvnd/egl_vendor.d/*nvidia*.json; do
  if [[ -r "$vendor" ]]; then export __EGL_VENDOR_LIBRARY_FILENAMES="$vendor"; break; fi
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$project_root/.cache/ducking/xdg"
export CUDA_CACHE_PATH="$project_root/.cache/ducking/cuda"
export __GL_SHADER_DISK_CACHE_PATH="$project_root/.cache/ducking/gl"
export TMPDIR="$project_root/.cache/ducking/tmp"
export MPLCONFIGDIR="$project_root/.cache/ducking/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH" "$TMPDIR" "$MPLCONFIGDIR"
output="$project_root/outputs/ducking/dynamics/run-$SLURM_JOB_ID"
mkdir -p "$output"
.venv-task10c/bin/python -B -m human_punching.ducking_dynamics --output "$output" "$@"
.venv-task10c/bin/python -B scripts/gpu_preflight.py --output "$output/gpu_runtime.json" --env-output "$output/egl_device.env"
source "$output/egl_device.env"
.venv-task10c/bin/python -B scripts/render_ducking_dynamics.py --trajectory "$output/simulation.npz" --output "$output/video"
