#!/usr/bin/env bash
#SBATCH --job-name=human-hook-window-review
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
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$project_root/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$project_root/.cache/matplotlib"
.venv-physics/bin/python -B scripts/review_reference_windows.py \
  --model outputs/scalar-body/human.xml --bank outputs/scalar-hook-retime-v1 \
  --output "outputs/hook-window-review-$SLURM_JOB_ID"
.venv/bin/python -B - <<'PY'
import json
import subprocess
from pathlib import Path
import imageio_ffmpeg
for family in ("left_hook", "right_hook"):
    video = Path(f"outputs/retimed-hook-render-5931841/{family}/{family}-rollout.mp4")
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(video),
                             "-f", "null", "-"], capture_output=True, text=True, check=False)
    record = dict(video=str(video), ffmpeg_returncode=result.returncode,
                  fully_decoded=result.returncode == 0, errors=result.stderr)
    (video.parent / "video-validation.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record))
    result.check_returncode()
PY
