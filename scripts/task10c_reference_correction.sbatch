#!/usr/bin/env bash
#SBATCH --job-name=task10c-ref-fix
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gpus=1
#SBATCH --exclude=node3006
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from human-punching}"
source scripts/task10c_env.sh
CORRECTION_ROOT=outputs/task10c/reference_correction
.venv-task10c/bin/python -B -c 'import torch; torch.cuda.init(); x=torch.ones(1,device="cuda"); torch.cuda.synchronize(); print(torch.cuda.get_device_name())'
.venv-task10c/bin/python -B scripts/task10c_compare_references.py \
  --families jab cross left_hook right_hook \
  --after-dir "$CORRECTION_ROOT/after" --output "$CORRECTION_ROOT/final_comparison" --render
.venv-task10c/bin/python -B -m human_punching.task10c.guard_checks \
  --reference-root "$CORRECTION_ROOT/before" --output "$CORRECTION_ROOT/guard_before" --seconds 2
.venv-task10c/bin/python -B -m human_punching.task10c.guard_checks \
  --reference-root "$CORRECTION_ROOT/after" --output "$CORRECTION_ROOT/guard_after" --seconds 2
.venv-task10c/bin/python -B scripts/task10c_render_references.py \
  --families jab cross left_hook right_hook \
  --reference-dir "$CORRECTION_ROOT/after" --output outputs/task10c/references/previews
