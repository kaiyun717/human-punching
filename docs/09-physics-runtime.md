# Physical human runtime: isolated GPU validation

Date: 2026-10-11. The GPU runtime probe and an improved full-speed physical jab against fixed G1 both passed on allocated Tesla V100 GPUs. The strongest controlled result is job **5891019**, described in the final section.

## Environment boundary

The new `.venv-physics` is separate from the existing preview `.venv`. The accepted mjlab revision `8ee51fbcf806a7419189f706d9e394cbeb7790fa` was exported using `git archive` into `.cache/deps/mjlab-8ee51fbc`; the sibling repository was only read. Nothing was staged, committed or pushed. All scripts, package locks, job outputs and intended caches belong to `human-punching`.

The core versions match the accepted `robust_sav` simulation stack: Python 3.12.3, Torch 2.8.0+cu128, MuJoCo 3.11.0, MuJoCo-Warp 3.11.0, Warp 1.14.0, RSL-RL 5.5.0 and NumPy 2.5.3. Supporting dependencies were resolved independently and frozen in [requirements-physics.lock](../requirements-physics.lock); this is not a byte-for-byte reconstruction of every historical `robust_sav` dependency. `uv pip check` passed for all 126 installed packages.

Reproduce from this directory:

```bash
bash scripts/setup_physics.sh
mkdir -p outputs/slurm
sbatch --parsable scripts/physics_probe_volta.sbatch \
  --physical-model outputs/physics-pilot/model/human_physical.xml
```

The [setup script](../scripts/setup_physics.sh) reads the accepted git object from sibling `mjlab`, creates the isolated environment, installs the complete lock, and checks compatibility. The project itself is installed editable so new local source is visible. Do not use the sibling mjlab HEAD: its newer Torch requirement is incompatible with this pin.

## What is tested

The [GPU probe](../scripts/physics_preflight.py) records independent outcomes:

1. A Torch MLP performs five CUDA forward/backward/Adam updates with finite gradients and decreasing loss.
2. A standalone Warp CUDA kernel computes a known numerical result.
3. `mjlab.sim.Simulation` steps 16 independent falling-sphere worlds for 1.2 seconds with gravity and floor contact; all must settle near their 0.10 m radius with low velocity.
4. The fixture camera produces RGB, optical-axis depth and typed geometry IDs using `mujoco_warp.create_render_context/refit_bvh/render`, the same underlying API used by Task 08.
5. The existing mannequin/G1 mesh scene is loaded into two Warp worlds and rendered through that same camera API. Besides an observer camera, it uses the actual Task 08 raster (672 × 376), fovy 54°, torso mount (0.08, 0, 0.45) m and 15° downward pitch, adapting only the body prefix to `g1/torso_link`. Glove visibility and correct RGB channel ordering are checked.
6. The actual physical human runs four worlds for 100 steps at 2 ms. One world has zero control, three receive a 0.5 Nm ball-joint motor pulse. The test verifies finite dynamics, a nonzero response to the motor, normalized joint quaternions and RGB/depth/ID rendering. It is a transmission/runtime check, not a standing or tracking criterion.

The worker requests one V100, six CPUs, 24 GiB host RAM and 30 minutes for initial JIT compilation. There is no OpenGL/EGL fallback in this probe: cameras use the Warp renderer. Outputs appear under `outputs/physics-probe-JOBID/`, with structured `runtime.json`, RGB PNGs and RGB/depth/ID NPZ arrays. A stage failure leaves evidence and makes the job fail even if other stages succeed.

## Verified final result

Slurm job **5890856** completed with **`COMPLETED`, exit `0:0`**, on `d-8-4-1`. The [structured result](../outputs/physics-probe-5890856/runtime.json), [Slurm log](../outputs/slurm/human-physics-probe-5890856.log) and [accounting record](../outputs/physics-slurm-accounting.txt) preserve the evidence.

| Check | Measured result |
|---|---|
| Torch CUDA training operations | Five finite-gradient Adam updates; loss 1.0700 → 0.9883 |
| Warp CUDA kernel | All 512 elements match the expected result |
| Batched floor contact | 16 worlds; 240 steps; final sphere height 0.099633 m; max speed 8.74e-6 |
| Actual human/G1 mesh camera | Two worlds; 64 meshes and 98 geoms; RGB, depth and typed IDs present |
| Task 08 camera geometry | 672 × 376 raw RGB/depth/IDs, correct torso mount and fovy; striking glove visible |
| Physical human | Four worlds; 20 ball joints, 60 motors; 100 steps; finite state; quaternion norm error 1.19e-7 |
| Torque transmission | Nonzero velocity difference between zero-control and torque-pulse worlds |

Images: [human and G1](../outputs/physics-probe-5890856/actual-scene-camera.png), [Task 08 view](../outputs/physics-probe-5890856/task08-camera.png), [physical human](../outputs/physics-probe-5890856/physical-human-camera.png). Each corresponding NPZ stores raw depth and typed geometry IDs alongside RGB.

The Task 08 near-contact image is dominated by the glove: **93.0%** of pixels show the striking glove, median raw depth is **0.0204 m**, and only **4.76%** of pixels fall within Task 08's accepted optical-depth range **[0.1, 3] m**. This is a concrete near-field occlusion case, not a renderer failure. The left glove is blue in the preview, whereas the current production detector expects red gloves. Therefore this raw-camera pass does **not** validate HSV classification, delayed visual observations or protagonist performance. [Pixel metrics](../outputs/physics-probe-5890856/task08-camera-metrics.json)

## Initial runs and corrections

Job **5890789** began on `d-8-5-1`, using a Tesla V100-PCIE-32GB. Torch's actual architecture list contains `sm_70`; the MLP passed all five optimization steps (loss 1.0700 → 0.9883). Warp 1.14.0 initialized with CUDA Toolkit 12.9, recognized the same V100 and successfully executed the numerical kernel. The first script revision then used an obsolete metadata accessor, `wp.context`; that bookkeeping error was corrected to the public `wp.get_cuda_toolkit_version()` API. Job 5890808 passed all numeric stages. Visual review then caught swapped R/B channels in the diagnostic unpacker; this was corrected to the renderer’s packed RGB order (bits 16, 8, 0) and a red-object assertion was added. Job 5890851 passed that correction; job 5890856 additionally verified the Task 08 torso camera. Earlier images remain historical diagnostics and should not be used as final color evidence.

These measurements are more specific than broad CUDA-version assumptions. The current official [PyTorch 2.8 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.8.0) describe removal of Maxwell/Pascal from CUDA 12.8/12.9 builds, while this installed Linux build explicitly retains Volta. Warp's runtime reports its own toolkit version, which differs from Torch's; a separate system CUDA toolkit installation is unnecessary for these wheel-based tests.

## How the physical human uses this API

A free root plus 20 ball joints has 87 position coordinates and 66 velocity coordinates. Three torque motors per ball joint provide 60 control channels. This is a different action representation from the scalar-joint `IdealPDActuator` wrappers used by standard mjlab robot tasks.

The low-level mjlab simulation accepts compiled MuJoCo models directly and exposes batched `sim.data.qpos`, `sim.data.qvel` and `sim.data.ctrl`; a quaternion-aware PD/controller can write torques there. This avoids pretending that 4 quaternion coordinates are 3 scalar joint positions. Controllers must preserve Torch/Warp CUDA stream ordering, keep the free root unactuated, respect torque limits, and log falls and assistance explicitly. The direct ball-joint motor/stepping gate has now passed for the physical model; stable closed-loop standing and punching remain separate criteria.

## Limits of the evidence

Passing these tests establishes that the installed training math, batched dynamics and rendering primitives execute on this server. It does not establish a trained human, stable tracking, G1 avoidance performance, collision accuracy for all body pairs, or multi-GPU learning. The original G1 policy checkpoints are still separate artifacts needed for meaningful protagonist evaluation.

## Controlled standing and jab on actual Warp dynamics

The next gate went beyond the motor smoke test. [`warp_tracking_probe.py`](../scripts/warp_tracking_probe.py) ran two worlds through `mjlab.sim.Simulation`: a 5-second standing hold and the conditioned, re-aimed jab reference at **0.6 playback speed**, followed by a 2-second recovery hold. Slurm job **5890905** passed. The model and reference are snapshotted in the output directory; the [report](../outputs/warp-tracking-5890905/report.json) records their hashes and both backend results.

The existing quaternion stable-PD controller with double-support balance feedforward ran on the CPU. At every step it observed the current GPU `qpos/qvel`, called CPU `mj_forward` to obtain its mass matrix and Jacobians, and uploaded **only bounded motor torques**. Warp performed every physical integration/contact solve. GPU positions and velocities were assigned exactly once at reset; no pose replay, root wrench, external root force or later state correction was applied. Foot contact counts and penetration were read from the actual Warp contact buffers. This deliberately synchronous CPU-controller/GPU-physics arrangement is a transparent compatibility experiment, not a high-throughput training implementation.

| Metric | Standing, Warp | Jab, Warp | Jab, native MuJoCo |
|---|---:|---:|---:|
| Duration | 5.000 s | 4.889 s | 4.889 s |
| Remained upright | Yes | Yes | Yes |
| Minimum pelvis height | 0.98475 m | 0.94856 m | 0.94850 m |
| Minimum pelvis upright-axis component | 0.99871 | 0.99325 | 0.99324 |
| Maximum foot horizontal drift | 1.44 mm | 3.65 mm | 3.63 mm |
| Both feet have contact | 99.84% | 99.84% | 99.84% |
| Maximum torque / per-joint norm limit | 0.386 | 0.956 | 0.956 |
| Mean fraction of saturated joints | 0 | 0 | 0 |
| Maximum root actuator force | 0 | 0 | 0 |
| Closest striking-glove/head distance | 0.50815 m | **0.13893 m** | **0.13889 m** |
| Crossed 0.14 m contact threshold | No | **Yes** | **Yes** |

The native comparison used the same model, conditioned reference, controller and timestep. Across the jab, the maximum root-position difference was **0.118 mm**; the largest component of joint rotation error between the two backends was **0.00217 rad**. These are measured differences, not a claim of identical numerical integration. Warp states are float32; native MuJoCo uses float64. GPU output site coordinates were recomputed at each saved state, while the native helper saves the previous step's kinematics, explaining a small timing offset in site-level metrics.

The same upright gate as the native pilot was used: pelvis height above 70% of its initial value and pelvis up-axis component above 0.70. A pass additionally requires zero root actuator force and torque norms within their limits. Glove contact is evaluated separately against the fixed analytic head center; the human-only physical model does not exchange impact impulses with a G1.

Reproduce the controlled experiment:

```bash
sbatch --parsable scripts/warp_tracking_volta.sbatch
```

The worker defaults to the snapshot input generated by native job `5890875`. Override `--model`, `--reference` and `--speed` for new inputs; output always goes to `outputs/warp-tracking-JOBID`. The sampled trajectories preserve both actual and desired joint poses, actual hands/feet, center of mass and measured contact counts. The results support retaining this classical controller baseline before introducing a learned tracker. They cover one nominal planted jab and a standing hold; broad robustness, all punch families, target/TTC variations and continuous sequences remain separate tests.

### Full-speed sensitivity exposes a tracking limit

Job **5890920** repeated the same reference at **1.0 source speed**. Both native MuJoCo and Warp remained upright and within the torque bounds, but the punch **missed**: minimum striking-glove/head center distance was **0.162581 m**, compared with **0.138935 m** at 0.6 speed. Maximum torque reached its norm cap; the mean fraction of saturated joints was 0.000937, and maximum foot drift rose to **8.80 mm**. [Full-speed report](../outputs/warp-tracking-5890920/report.json)

This distinguishes geometric reference contact from executed physical contact. The full-speed job's `status: passed` refers only to the explicit upright/no-root-assistance/torque-bound **control gate**; `contact_goal_achieved: false` records the failed punching objective. The slower job has `contact_goal_achieved: true`. At that stage the accepted physical jab was a **0.6-speed nominal case**; the later full-speed refinement below supersedes that limitation for this one reference. Reference timing/trajectory and controller improvements are the next classical-control experiments; replacing the physics backend does not address this miss.

## Strongest result: improved full-speed jab with fixed G1

**Job 5891019 passed the physical support and isolated-punch gates at full source speed**, using the fixed-G1 scene and improved reference from native experiment `5890952`. The reference was smoothed with a four-frame Gaussian standard deviation, re-aimed to a **0.10 m** glove/head center target, and tracked with PD gain scale **1.5**. Torque limits were retained. The actual executed trajectory crossed the unchanged **0.14 m** task-contact threshold in both simulation backends. [Full report](../outputs/warp-tracking-5891019/report.json)

| Metric | Warp GPU physics | Native MuJoCo |
|---|---:|---:|
| Playback speed | 1.0 | 1.0 |
| Gain scale | 1.5 | 1.5 |
| Closest striking-glove/head center distance | **0.112658289 m** | **0.112658302 m** |
| First interpolated contact time | **0.801474686 s** | **0.801474636 s** |
| Maximum foot horizontal drift | 5.05448 mm | 5.05430 mm |
| Both feet have contact | 99.7859% | 99.7859% |
| Minimum pelvis height | 0.952118 m | 0.952118 m |
| Minimum pelvis upright-axis component | 0.992875 | 0.992875 |
| Mean fraction of saturated joints | 0.14454% | 0.14454% |
| Maximum root actuator force | **0** | **0** |
| Support / isolated-punch gates | **Pass / Pass** | **Pass / Pass** |

Maximum root-position difference over the jab was **0.686 micrometres**; the largest component of joint rotation error was **3.14e-6 rad**. These describe this matched nominal rollout, not a general backend accuracy guarantee. The accompanying standing world also held its pose for five seconds with 1.41 mm foot drift.

The improved probe records **glove centers at every 2 ms physical step** in `substep_time_s` and `substep_glove_centers_w`. Contact minima and first-contact events use those arrays, independently of the 125 Hz visualization trajectory. Both native and GPU trajectories now recompute FK at the current physical state before measuring contact. Earlier sections preserve historical experiments that used the older sampling/helper behavior.

Support acceptance matches the native pilot: remain upright, maximum foot horizontal drift and height each ≤ 3 cm, floor penetration ≤ 1 cm, and both feet contacting for ≥ 95% of recorded samples. Isolated-punch acceptance additionally requires analytic glove/head contact and mean joint saturation ≤ 5%. Root assistance remains absent and torque norm limits remain enforced. This is one deterministic full-speed jab/recovery case, not a robustness evaluation or continuous combination policy.

The G1 is present as **fixed scene geometry with zero dynamic DOFs**. Task contact is analytic; G1 impact response is disabled. In particular, aiming inside the 0.14 m proxy permits the glove to pass through that envelope without receiving a collision impulse. This tests executable targeting and recovery, not physically realistic glove/head impact or learned G1 avoidance.

Reproduce the strongest Warp experiment:

```bash
sbatch --parsable scripts/warp_tracking_volta.sbatch \
  --model outputs/physical-5890952/scene/physical_scene.xml \
  --reference outputs/physical-5890952/jab/reference.npz \
  --punch jab --speed 1.0 --gain-scale 1.5
```

The [output directory](../outputs/warp-tracking-5891019/) includes model/assets, reference, exact arguments, controller/native-CLI/contact-code/probe/worker snapshots, SHA256 provenance, full-step contact arrays, sampled actual/desired trajectories and native-vs-Warp metrics. The input conditioning and original native result are preserved alongside them. `--punch` controls labeling; `--gain-scale` defaults to 1.0 so the higher gains are always an explicit experiment setting.
