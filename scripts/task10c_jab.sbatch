#!/usr/bin/env bash
# Task 10C Phase 2 only; launch from human-punching with explicit run-local logs.
#SBATCH --job-name=task10c-p2-jab
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@180
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from human-punching}"
source scripts/task10c_env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec .venv-task10c/bin/python -B -m human_punching.task10c.train --phase2 "$@"
