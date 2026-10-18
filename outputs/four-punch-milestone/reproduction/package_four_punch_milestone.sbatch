#!/usr/bin/env bash
#SBATCH --job-name=human-four-punch-release
#SBATCH --partition=xeon-p8
#SBATCH --nodelist=d-18-4-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/slurm/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from human-punching}"
bash scripts/package_four_punch_milestone.sh "$@"
