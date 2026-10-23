#!/usr/bin/env bash
#SBATCH --job-name=task10d-p1
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from human-punching}"
source scripts/task10d_env.sh
exec .venv-task10c/bin/python -B -u -m human_punching.task10d.phase1 "$@"
