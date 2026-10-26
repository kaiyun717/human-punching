#!/usr/bin/env bash
#SBATCH --job-name=g1-duck-id
#SBATCH --account=mit_general
#SBATCH --partition=mit_quicktest
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=outputs/ducking/dynamics/slurm-%j.log
set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from human-punching}"
cd "$project_root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$project_root/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$project_root/.cache/ducking/xdg"
export TMPDIR="$project_root/.cache/ducking/tmp"
export MPLCONFIGDIR="$project_root/.cache/ducking/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$MPLCONFIGDIR"
exec .venv-task10c/bin/python -B -m human_punching.ducking_dynamics \
  --output "$project_root/outputs/ducking/dynamics/run-$SLURM_JOB_ID" "$@"
