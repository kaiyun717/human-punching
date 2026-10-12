#!/usr/bin/env bash
#SBATCH --job-name=human-punch-volta
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/slurm/%x-%j.log
set -euo pipefail
# Submit with bash scripts/submit_gpu.sh. No Anaconda module is needed: the
# project's tested Python 3.12 venv works on both login and compute nodes.
project_root="${SLURM_SUBMIT_DIR:?Submit from human-punching}"
cd "$project_root"
[[ -f pyproject.toml && -f src/human_punching/cli.py && -x .venv/bin/python ]] || {
  echo 'Submit from the configured human-punching directory.' >&2; exit 2;
}
: "${SLURM_JOB_ID:?This worker requires a Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?Slurm did not assign visible GPUs}"
run_id="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then run_id+="_${SLURM_ARRAY_TASK_ID}"; fi
run_dir="$project_root/outputs/gpu-$run_id"
mkdir -p "$run_dir"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$project_root/.cache/gpu/$run_id/xdg"
export CUDA_CACHE_PATH="$project_root/.cache/gpu/$run_id/cuda"
export __GL_SHADER_DISK_CACHE_PATH="$project_root/.cache/gpu/$run_id/gl"
export TMPDIR="$project_root/.cache/gpu/$run_id/tmp"
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH" "$TMPDIR"
printf 'Job %s on %s; artifacts: %s\n' "$SLURM_JOB_ID" "${SLURMD_NODENAME:-unknown}" "$run_dir"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total --format=csv > "$run_dir/nvidia-smi.csv"
cat "$run_dir/nvidia-smi.csv"
# Check a real hardware GL context and bind EGL to the allocated CUDA GPU.
.venv/bin/python -B scripts/gpu_preflight.py \
  --output "$run_dir/gpu_runtime.json" --env-output "$run_dir/egl_device.env"
source "$run_dir/egl_device.env"
preview_args=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  punches=(jab right_hook left_hook cross)
  [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-3]$ ]] || { echo 'Array task must be 0..3' >&2; exit 2; }
  preview_args=(--punches "${punches[$SLURM_ARRAY_TASK_ID]}")
fi
.venv/bin/python -B -m human_punching.cli "$@" "${preview_args[@]}" \
  --backend egl --output "$run_dir/preview"
.venv/bin/python -B - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1])
manifest=json.loads((run/'preview/manifest.json').read_text())
assert manifest['backend']=='egl'
assert manifest['punches'] and all('media' in p for p in manifest['punches'].values())
(run/'complete.json').write_text(json.dumps({'status':'completed','punches':list(manifest['punches']),'preview_elapsed_s':manifest['elapsed_s'],'runtime':'gpu_runtime.json'},indent=2)+'\n')
PY
printf 'Complete: %s/preview\n' "$run_dir"
