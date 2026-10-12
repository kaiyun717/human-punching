#!/usr/bin/env bash
#SBATCH --job-name=human-punch-preview
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm/%x-%j.log
set -euo pipefail
# Submit from human-punching: mkdir -p outputs/slurm; sbatch scripts/render_l40s.sbatch
project_root="${SLURM_SUBMIT_DIR:?Submit from the human-punching project directory}"
cd "$project_root"
[[ -f pyproject.toml && -d src/human_punching ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$project_root/.cache/xdg"
export CUDA_CACHE_PATH="$project_root/.cache/cuda"
export TMPDIR="$project_root/.cache/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
nvidia-smi --query-gpu=name,driver_version --format=csv
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi L40S || {
  echo 'Allocated GPU is not the requested L40S' >&2; exit 2;
}
"$project_root/.venv/bin/python" -B -m human_punching.cli --output "outputs/preview-l40s-${SLURM_JOB_ID}"
