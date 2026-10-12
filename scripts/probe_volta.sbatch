#!/usr/bin/env bash
#SBATCH --job-name=hp-volta-probe
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=outputs/slurm/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from human-punching}"
export PYTHONDONTWRITEBYTECODE=1
hostname
printf 'Job: %s CUDA_VISIBLE_DEVICES: %s SLURM_JOB_GPUS: %s\n' "$SLURM_JOB_ID" "${CUDA_VISIBLE_DEVICES:-unset}" "${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total --format=csv
/usr/bin/python3 --version
.venv/bin/python -B -c 'import sys,mujoco,numpy; print(sys.version); print("MuJoCo",mujoco.__version__,"NumPy",numpy.__version__)'
MUJOCO_GL=egl .venv/bin/python -B - <<'PY'
import mujoco
from OpenGL import GL
model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
data=mujoco.MjData(model)
mujoco.mj_forward(model,data)
with mujoco.Renderer(model,64,64) as renderer:
    renderer.update_scene(data)
    frame=renderer.render()
    print('GL_VENDOR',GL.glGetString(GL.GL_VENDOR))
    print('GL_RENDERER',GL.glGetString(GL.GL_RENDERER))
    print('frame',frame.shape,frame.std())
PY
