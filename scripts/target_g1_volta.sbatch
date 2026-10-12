#!/usr/bin/env bash
#SBATCH --job-name=human-g1-target
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/slurm/%x-%j.log
set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from human-punching}"
cd "$project_root"
[[ -f src/human_punching/target_cli.py && -x .venv/bin/python ]] || exit 2
: "${SLURM_JOB_ID:?This worker requires Slurm}"
: "${CUDA_VISIBLE_DEVICES:?Slurm did not assign visible GPUs}"
run_dir="$project_root/outputs/g1-gpu-$SLURM_JOB_ID"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$project_root/.cache/gpu/$SLURM_JOB_ID/xdg"
export CUDA_CACHE_PATH="$project_root/.cache/gpu/$SLURM_JOB_ID/cuda"
export __GL_SHADER_DISK_CACHE_PATH="$project_root/.cache/gpu/$SLURM_JOB_ID/gl"
export TMPDIR="$project_root/.cache/gpu/$SLURM_JOB_ID/tmp"
mkdir -p "$run_dir" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH" "$TMPDIR"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total --format=csv > "$run_dir/nvidia-smi.csv"
.venv/bin/python -B scripts/gpu_preflight.py --output "$run_dir/gpu_runtime.json" --env-output "$run_dir/egl_device.env"
source "$run_dir/egl_device.env"
.venv/bin/python -B -m human_punching.target_cli "$@" --backend egl --output "$run_dir/preview"
.venv/bin/python -B - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1])
report=json.loads((run/'preview/manifest.json').read_text())
assert report['backend']=='egl'
assert report['punches'] and all('media' in p for p in report['punches'].values())
(run/'complete.json').write_text(json.dumps({'status':'completed','punches':list(report['punches']),'elapsed_s':report['elapsed_s']},indent=2)+'\n')
PY
