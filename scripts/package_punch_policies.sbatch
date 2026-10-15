#!/usr/bin/env bash
#SBATCH --job-name=human-policy-package
#SBATCH --partition=xeon-p8
#SBATCH --nodelist=d-18-4-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/slurm/%x-%j.log
set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from human-punching}"
cd "$project_root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-physics/bin/python -B scripts/package_punch_policies.py "$@"
