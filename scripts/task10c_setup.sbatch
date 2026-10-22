#!/usr/bin/env bash
#SBATCH --job-name=task10c-setup
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from human-punching}"
source scripts/task10c_env.sh
nvidia-smi --query-gpu=name,driver_version --format=csv
.venv-task10c/bin/python -B -c 'import torch; torch.cuda.init(); x=torch.ones(1,device="cuda"); torch.cuda.synchronize(); print("CUDA startup check:",torch.cuda.get_device_name(),x.item())'
.venv-task10c/bin/python -B -m human_punching.task10c.family_checks --output outputs/task10c/setup/family_checks.json
exec .venv-task10c/bin/python -B -m human_punching.task10c.train "$@"
