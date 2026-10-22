#!/usr/bin/env bash
#SBATCH --job-name=task10c-checks
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from human-punching}"
source scripts/task10c_env.sh
.venv-task10c/bin/python -B scripts/task10c_render_references.py --families jab left_hook
exec .venv-task10c/bin/python -B -m human_punching.task10c.native_checks --output outputs/task10c/setup/native01
