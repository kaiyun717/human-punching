#!/usr/bin/env bash
#SBATCH --job-name=scalar-punch-ik
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --array=0-3
#SBATCH --output=outputs/slurm/%x-%A_%a.log
set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from human-punching}"
cd "$project_root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$project_root/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
families=(jab cross left_hook right_hook)
punch="${families[$SLURM_ARRAY_TASK_ID]}"
run_root="outputs/scalar-four-reference-$SLURM_ARRAY_JOB_ID"
mkdir -p "$run_root/$punch"
cp src/human_punching/scalar_reference.py "$run_root/$punch/scalar_reference_source.py"
.venv/bin/python -B -m human_punching.scalar_reference \
  --punch "$punch" --speed 1.0 --output "$run_root/$punch" "$@"
