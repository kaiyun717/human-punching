#!/usr/bin/env bash
#SBATCH --job-name=human-bundle-smoke-audit
#SBATCH --partition=xeon-p8
#SBATCH --nodelist=d-18-4-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/slurm/%x-%j.log
set -euo pipefail
project_root="${SLURM_SUBMIT_DIR:?Submit from human-punching}"
cd "$project_root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$project_root/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-physics/bin/python -B - <<'PY'
import json
from human_punching.jab_evaluation import evaluate_rollout
result = evaluate_rollout(
    'outputs/punch-5ms-5931997/model/human_runtime.xml',
    'outputs/punch-5ms-5931997/rollout_00.npz',
    'outputs/four-punch-milestone/right_hook/reference.npz',
    'outputs/punch-5ms-5931997/independent_smoke.json',
    required_physics_timestep_s=.005, required_policy_timestep_s=.02)
print(json.dumps({key: result[key] for key in (
    'pilot_pass', 'failed_checks', 'precision_targets', 'recorded_checkpoint_sha256',
    'evaluated_physics_timestep_s', 'recorded_policy_timestep_s')}, indent=2))
assert result['pilot_pass']
assert result['precision_targets']['active_glove_rms_target_met']
assert result['precision_targets']['contact_timing_target_met']
assert result['reference_tracking']['attack_active_glove_position_rms_m'] <= .02
PY
