#!/usr/bin/env bash
#SBATCH --job-name=human-release-hashes
#SBATCH --partition=xeon-p8
#SBATCH --nodelist=d-18-4-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/slurm/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from human-punching}"
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -B - <<'PY'
import hashlib
import json
import os
from pathlib import Path
bundle = Path('outputs/four-punch-milestone')
manifest_path = bundle/'manifest.json'
manifest = json.loads(manifest_path.read_text())
for relative, metadata in manifest['artifacts'].items():
    artifact = bundle/relative
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == metadata['sha256'], relative
    assert artifact.stat().st_size == metadata['bytes'], relative
names = list(manifest['artifacts'])
result = {
    'bundle': str(bundle),
    'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    'readme_sha256': hashlib.sha256((bundle/'README.md').read_bytes()).hexdigest(),
    'artifact_count': len(names),
    'artifact_bytes': sum(item['bytes'] for item in manifest['artifacts'].values()),
    'learned_rollouts': sum('/evaluation/rollout_' in name and name.endswith('.npz') for name in names),
    'baseline_rollouts': sum('/baseline/rollout_' in name and name.endswith('.npz') for name in names),
    'videos': sum(name.endswith('.mp4') for name in names),
    'all_artifact_hashes_and_sizes_match': True,
    'families': {family: {'accepted_trials': item['accepted_physical_pilot_trials'],
                          'precision_targets': item['precision_target_pass_counts'],
                          'baseline_passes': item['optional_evidence']['zero_residual_baseline']['baseline_pilot_pass_count']}
                 for family, item in manifest['families'].items()},
}
assert result['learned_rollouts'] == result['baseline_rollouts'] == 256
assert result['videos'] == 8
Path(f"outputs/four-punch-release-verification-{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
PY
