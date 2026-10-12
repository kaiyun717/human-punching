# human-punching

Retarget DuoBox's jab, cross, left hook and right hook to a **1.80 m articulated human mesh**, then render them in **MuJoCo 3.11.0**, the native simulator version used by `robust_sav`.

The project now includes both kinematic previews and a physical-human pilot. The pinned mjlab/MuJoCo-Warp physics and camera pipeline has also been tested on allocated V100 GPUs.

The original previews replay motion kinematically; the new physical pilot uses bounded joint torques, gravity, and floor contact with an unactuated pelvis. Reports distinguish reference playback, native MuJoCo dynamics, and actual Warp dynamics. The body is a plain light-blue mannequin with a featureless head, simple feet and boxing gloves, not a licensed SMPL-X reconstruction.

All code, environments, generated assets and videos live in this directory. `../robust_sav`, `../mjlab` and `../DuoBox` are read-only inputs.

## Physical human and contact results

Start with the [physical pilot report and remaining gates](docs/13-physical-pilot-and-next-gates.md). It links the successful cases, failed experiments, runtime evidence, and phased integration work.

- [Contact contract](docs/10-contact-contract.md): either glove at **≤0.14 m** is contact; both at **≥0.16 m** have safe glove clearance. Impact forces are a separate choice.
- [Human-model research](docs/08-physical-human-research.md): primitive rigid bodies and inferred inertias suffice for this pilot; no muscles or SMPL assets are needed.
- [Pinned GPU runtime and controlled Warp replay](docs/09-physics-runtime.md): actual GPU dynamics, cameras, and bounded-control checks.
- [Twelve contact-directed references and videos](docs/11-reference-contact.md), followed by [cycle admission and continuous two-jab physics](docs/12-cycle-bank.md).

[Four physical punches](outputs/physical-review/overview.png) · [Two physically executed jabs](outputs/physical-review/two_jabs/two_jabs.mp4) use actual guard recovery before the second launch, with no intermediate reset. The physical jab also passes at source speed in Warp; the cross/hooks currently pass at 0.3× speed in native MuJoCo. This is a classical-controller proof, not a trained adversary. The G1 is fixed and head contact is evaluated analytically.

```bash
bash scripts/setup_physics.sh
mkdir -p outputs/slurm
sbatch scripts/physics_probe_volta.sbatch
sbatch scripts/warp_tracking_volta.sbatch --speed .6
sbatch scripts/physical_pilot_volta.sbatch \
  --input outputs/contact-targeting --balance --reaim --with-g1 \
  --punches jab --speed 1 --smooth 4 --center-target .10 --gain-scale 1.5
sbatch scripts/physical_sequence_volta.sbatch
```

The native physical pilot uses **2 ms** steps. The tested **5 ms** variants failed their support/tracking gates, so this is not yet a drop-in replacement for the accepted G1 plant. Ball-joint anatomy, self-collision, full-speed all-family tracking, perception integration, and two-player learning remain explicit gaps. No files in `robust_sav` are changed.

## Preview videos

[Overview](outputs/preview/overview.png) · [Jab](outputs/preview/jab/jab.mp4) · [Right hook](outputs/preview/right_hook/right_hook.mp4) · [Left hook](outputs/preview/left_hook/left_hook.mp4) · [Cross](outputs/preview/cross/cross.mp4)

The gloves follow the full recorded hand rotation. Their flattened shape and pale orientation marks make wrist rotation visible; see the [rotation audit](docs/04-glove-rotation-audit.md).

These local generated artifacts are excluded from Git; regenerate them with the command below.

## Punching toward a standing G1

[Four-punch overview](outputs/g1-targeting/overview.png) · [Jab](outputs/g1-targeting/jab/jab.mp4) · [Right hook](outputs/g1-targeting/right_hook/right_hook.mp4) · [Left hook](outputs/g1-targeting/left_hook/left_hook.mp4) · [Cross](outputs/g1-targeting/cross/cross.mp4)

A shared MuJoCo scene holds the G1 in its configured standing pose while the human punches toward its head. Smooth glove-path deformation and arm IK preserve the recorded root, legs, inactive arm, timing, and endpoint guards. The gloves approach the visible head with a small gap; no learned model, balancing policy, or impact simulation is used. See the [targeting experiment report](docs/07-standing-g1-targeting.md) for measured geometry, limitations, and why IK is enough for this preview.

```bash
bash scripts/render_g1_cpu.sh       # Default output: outputs/g1-targeting
bash scripts/submit_g1_gpu.sh       # One Volta GPU; outputs/g1-gpu-JOBID/preview
```

The generated `scene/scene.xml` and copied assets are portable. Select its `standing` keyframe to reproduce the robot pose in a separate viewer. Use the trajectory archives for human animation. This is a native MuJoCo preview, separate from the Task 08 policy/camera pipeline.

## Setup

Python 3.10–3.13, `uv`, and a working OpenGL backend are required. On this host, Mesa's OSMesa library supports headless CPU rendering.

```bash
cd human-punching
export UV_CACHE_DIR="$PWD/.cache/uv"
uv venv .venv --python python3
uv pip install --python .venv/bin/python -e '.[test]'
```

A tested environment lock is recorded in `requirements-preview.lock`. For exact replay, use `uv pip install --python .venv/bin/python -r requirements-preview.lock` before installing `-e .`.

## Generate the previews

```bash
bash scripts/render_cpu.sh
```

The default reads `../DuoBox` and writes `outputs/preview`. Each punch gets a two-view MP4 with real-time and half-speed playback, an annotated still, a compressed trajectory archive and a metrics manifest. The shared model includes MJCF and OBJ meshes. `overview.png` presents all four punches.

```bash
.venv/bin/python -B -m human_punching.cli --help
MUJOCO_GL=osmesa .venv/bin/python -B -m human_punching.cli \
  --dataset ../DuoBox --output outputs/preview --height 1.80
```

For this server's Volta GPUs:

```bash
bash scripts/submit_gpu.sh          # All four punches on one GPU
bash scripts/submit_gpu.sh --array  # Four independent tasks, one punch per GPU
```

See [GPU server guide](docs/05-gpu-server-guide.md) for the tested server setup,
monitoring, output paths, and the eight-GPU allowance. The worker verifies that
MuJoCo actually renders on the Slurm-allocated NVIDIA device with EGL. It uses the
existing Python 3.12 `.venv`; the example script's Anaconda module is unnecessary.

The earlier `scripts/render_l40s.sbatch` is retained for the separate
`mit_normal_gpu` cluster. Use `render_volta.sbatch` on this server. CPU previews
remain in `outputs/preview`; GPU jobs write separate `outputs/gpu-JOBID` directories.

## Checks and scope

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
```

The source files are checked against the SHA256 hashes already recorded by `robust_sav`. The retargeter scales the bind skeleton and body surface together, composes the complete joint hierarchy, aligns the initial heading, and applies a fixed floor offset per clip. Original timing is retained. Output manifests distinguish crop frames, padding, normalization, FK error and floor clearance.

Read the [scope](docs/00-scope-and-plan.md), [dependency and Task 09 assessment](docs/01-compatibility-assessment.md), [DuoBox audit](docs/02-duobox-motion-audit.md), and [implementation/results report](docs/03-implementation-and-results.md).

The dependency audit identifies a real version conflict: current sibling `mjlab` requires PyTorch ≥2.14 and RSL-RL 5.5.1; `robust_sav` uses PyTorch 2.8 and a pinned older mjlab revision with RSL-RL 5.5.0. The preview isolates MuJoCo 3.11.0 from those training dependencies. Future integration should start from `robust_sav`'s exact source pin.

## Later work

See the [revised Task 09 assessment and phased roadmap](docs/06-task09-revised-phased-roadmap.md) for limitations, unnecessary early complexity, concrete software gaps, and acceptance gates. The runtime probe, contacting references, physical tracker pilot, and repeated-jab guard recovery are now implemented; see the newer [physical pilot report](docs/13-physical-pilot-and-next-gates.md) for measured passes and outstanding gates. The saved kinematic trajectories are useful references, not a drop-in trained adversary.
