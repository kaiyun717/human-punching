#!/usr/bin/env bash
#SBATCH --job-name=human-physics-probe
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.log
set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from human-punching}"
cd "$project_root"
[[ -x .venv-physics/bin/python && -f scripts/physics_preflight.py ]] || exit 2
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU required}"
run_dir="$project_root/outputs/physics-probe-$SLURM_JOB_ID"
mkdir -p "$run_dir"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$project_root/.cache/physics/matplotlib"
export XDG_CACHE_HOME="$project_root/.cache/physics/xdg"
export CUDA_CACHE_PATH="$project_root/.cache/physics/cuda"
export TMPDIR="$project_root/.cache/physics/tmp-$SLURM_JOB_ID"
export WANDB_MODE=disabled
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TMPDIR"
nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total --format=csv > "$run_dir/nvidia-smi.csv"
.venv-physics/bin/python -B scripts/physics_preflight.py --output "$run_dir/runtime.json" \
 --scene outputs/g1-targeting/scene/scene.xml --trajectory outputs/g1-targeting/jab/trajectory.npz "$@"
