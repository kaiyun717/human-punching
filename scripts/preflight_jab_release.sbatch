#!/usr/bin/env bash
#SBATCH --job-name=human-jab-release-preflight
#SBATCH --partition=xeon-p8
#SBATCH --nodelist=d-18-4-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/slurm/%x-%j.log
set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from human-punching}"
cd "$project_root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-physics/bin/python -B scripts/package_punch_policies.py \
  --checkpoint jab=outputs/punch-5ms-5931736/checkpoint_00100.pt \
  --evaluation jab=outputs/punch-5ms-5931806 \
  --baseline jab=outputs/punch-5ms-5931807 \
  --media-dir jab=outputs/jab-render-5931851 \
  --media-dir jab=outputs/jab-render-5931873 \
  --media-dir jab=outputs/punch-5ms-5931806/independent/nominal-plots \
  --check-only
