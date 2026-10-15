#!/usr/bin/env bash
#SBATCH --job-name=human-jab-ppo
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.log
set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from human-punching}"
cd "$project_root"
: "${CUDA_VISIBLE_DEVICES:?Slurm GPU required}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$project_root/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$project_root/.cache/physics/xdg"
export MPLCONFIGDIR="$project_root/.cache/physics/matplotlib"
export CUDA_CACHE_PATH="$project_root/.cache/physics/cuda"
export TMPDIR="$project_root/.cache/physics/tmp-$SLURM_JOB_ID"
export WANDB_MODE=disabled
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TMPDIR" "$MPLCONFIGDIR"
source_dir="$project_root/.cache/physics/job-sources/$SLURM_JOB_ID"
mkdir -p "$source_dir"
cp -a src/human_punching "$source_dir/human_punching"
export PYTHONPATH="$source_dir:$project_root/src"
.venv-physics/bin/python -B "$source_dir/human_punching/jab_learning.py" \
  --model outputs/scalar-body/human.xml \
  --body-metadata outputs/scalar-body/scalar_model.json \
  --reference outputs/scalar-reference-v3/jab.npz \
  --output "outputs/jab-learning-$SLURM_JOB_ID" "$@"
