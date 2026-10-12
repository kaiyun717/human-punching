#!/usr/bin/env bash
#SBATCH --job-name=human-warp-tracking
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.log
set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from human-punching}"
cd "$project_root"
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU required}"
run_dir="$project_root/outputs/warp-tracking-$SLURM_JOB_ID"
mkdir -p "$run_dir"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$project_root/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$project_root/.cache/physics/xdg"
export MPLCONFIGDIR="$project_root/.cache/physics/matplotlib"
export CUDA_CACHE_PATH="$project_root/.cache/physics/cuda"
export TMPDIR="$project_root/.cache/physics/tmp-$SLURM_JOB_ID"
export WANDB_MODE=disabled
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TMPDIR" "$MPLCONFIGDIR"
.venv-physics/bin/python -B scripts/warp_tracking_probe.py \
 --model outputs/physical-5890875/model/human_physical.xml \
 --reference outputs/physical-5890875/jab/reference.npz --output "$run_dir" "$@"
