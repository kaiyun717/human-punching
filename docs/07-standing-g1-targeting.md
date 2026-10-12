# Head-targeted punches toward a standing G1

Date: 10 October 2026. All implementation and outputs are in `human-punching`; `robust_sav` and `mjlab` were read only.

## Result and answer to the method question

**The experiment works without machine learning.** The jab, cross, left hook and right hook have been adapted toward a stationary G1's head and rendered together with the robot in one native MuJoCo 3.11.0 scene. Each clip includes an oblique and side view, normal-speed and half-speed playback, and initial-guard/closest-approach stills.

[Overview](../outputs/g1-targeting/overview.png) · [Jab video](../outputs/g1-targeting/jab/jab.mp4) · [Cross video](../outputs/g1-targeting/cross/cross.mp4) · [Left hook video](../outputs/g1-targeting/left_hook/left_hook.mp4) · [Right hook video](../outputs/g1-targeting/right_hook/right_hook.mp4)

This is **motion retargeting implemented with smooth trajectory deformation and inverse kinematics (IK)**. These are complementary descriptions of the same work:

| Method | Role here |
|---|---|
| Motion retargeting | The overall task: preserve recognizable recorded punches while changing their spatial goal to G1's head. |
| Forward kinematics (FK) | Compute where the body, wrists and actual glove geometry end up for a candidate set of joint rotations; verify the solved motion. FK alone does not find the required rotations. |
| Inverse kinematics | Adjust the striking arm so its glove follows the deformed target path. Segment lengths remain fixed. |
| Trajectory optimization | A possible next tool for jointly imposing smoothness, joint/velocity limits, foot support and obstacle clearance across time. A large coupled optimizer was unnecessary for this fixed-target preview. |
| Machine learning | Not needed for the current geometry problem. A learned tracker could later execute references under gravity and feedback; that is a different milestone. A learned controller is not mandatory if another controller passes the required physical tests. |

The current solver itself performs small deterministic least-squares optimizations. “No trajectory optimizer needed” therefore means no full sequence/dynamics optimization problem was required, not that the IK is optimization-free.

## Scene and standing robot

The scene uses the existing 1.80 m blue mannequin and the real G1 XML/STL assets from the sibling mjlab repository. The G1 asset subtree matches the exact mjlab revision pinned by `robust_sav`: `8ee51fbcf806a7419189f706d9e394cbeb7790fa`. Files and the repository license are copied into the generated output; source hashes are recorded.

G1 faces the human, with pelvis XY at `(0.85, 0.20)` m and yaw pi. It uses mjlab's configured `KNEES_BENT_KEYFRAME` joint angles, with visible soles aligned to z=0. This is a held standing configuration, not inference from a learned standing policy or the unavailable settled `qpos_star` artifact. Its head-proxy center is `(0.8539635, 0.20, 1.231490745)` m. That lower height explains why the original human punches need to be redirected downward.

The combined model has 123 qpos coordinates and 101 velocity coordinates: human 87/66, G1 36/35. Explicit entity maps preserve the human layout. Every G1 coordinate stays fixed throughout all four clips. Gravity, contacts and actuators are disabled; playback updates recorded/solved poses with `mj_forward`. It does not simulate balance, impact, or the G1 avoidance policy. Native MuJoCo rendering in this scene does not establish compatibility with the Task 08 Warp physics/camera application.

The portable scene is [scene/scene.xml](../outputs/g1-targeting/scene/scene.xml). A separate MuJoCo viewer must select its **`standing` keyframe**; loading default qpos alone does not apply the robot's bent-joint stance. The saved trajectory archives supply the human animation and complete combined-model poses.

## How the adaptation works

1. Load the same hash-verified DuoBox clips, retain the original 120 Hz timing, and generate the existing full-body replay.
2. Find the attack's forward-reach anchor within the annotated attack interval. For a hook, the useful passing/contact region can precede the inherited outward-completion landmark; the right hook is an example.
3. Deform the active glove-center path toward the front of the head for straight punches and toward the corresponding side for hooks. A quintic envelope fades the correction in and out. Hooks use a shorter raised-attack window so their low wind-up is not translated into an unreachable pose.
4. Use a smooth radial clearance adjustment around the fixed head target. The head is captured once; there is no live target tracking during the motion.
5. Solve Jacobian IK for the active shoulder, upper arm, forearm and wrist ball joints. Position tracking is primary, with source glove orientation and source arm posture as soft objectives. The previous frame's correction initializes the next solve.
6. Recompute FK, body states and velocities from the final joint rotations. Verify the actual glove geometry rather than substituting the wrist origin for its center.

MuJoCo supplies the geometric Jacobian and quaternion-aware configuration integration used in this implementation. [MuJoCo 3.11 API](https://mujoco.readthedocs.io/en/3.11.0/APIreference/APIfunctions.html#mj-jac)

The source root, trunk, legs, inactive arm and timing remain exactly unchanged. Initial and final source poses are also unchanged. The gloves remain rigidly attached to their hands. This preserves the current source motion structure, but does not certify anatomical joint limits or mechanically feasible speeds. Maximum individual arm-joint changes relative to the source reach approximately 88 degrees, so these are substantive pose adaptations, not tiny corrections.

## Head geometry and why the preview stops just short

The 0.06 m spherical G1 head proxy is smaller than parts of its visible head mesh. The rounded block gloves also differ from an 0.08 m spherical glove. A small positive sphere-to-sphere clearance therefore did **not** prevent the visible geometry from intersecting in the first trials.

The final CLI uses 0.044 m extra clearance beyond the nominal 0.06 + 0.08 m center envelope. This was chosen against the actual visible geometry and produces a small visible gap. It is a preview-specific geometric setting, not a change to Task 08's head/glove radii or safety margin. The lower-level `aim_clip` function retains its 0.015 m proxy-targeting default; the scene CLI explicitly supplies the larger visible-head setting.

The authoritative non-overlap check is independent of MuJoCo's convex collision distance: for each sampled pose, transform both meshes' actual vertices to world coordinates and project them onto the axis from the head-mesh center to the glove-mesh center. Compute

```text
gap = min(glove_vertices · axis) - max(head_vertices · axis)
```

A positive gap means the meshes occupy disjoint half-spaces at that pose. The table reports the minimum of this conservative separation bound over every frame. This checks the active glove against the visible head; it does not establish full-body clearance or continuous swept collision avoidance between samples.

A supplemental libccd distance check gives closest visible-head gaps of approximately 3.6, 5.0, 6.0 and 6.5 mm for jab, cross, left hook and right hook. The native CCD diagnostic returned inconsistent mesh distances in this experiment, including zero for some clearly separated configurations, so it is not used alone as acceptance evidence. The exported per-frame vertex-projection bound is reproducible without choosing a collision-distance backend.

**These are punches aimed to near-contact with the visible head, not recorded Task 08 spherical-proxy hits.** The final glove centers remain outside the accepted 0.14 m physical-contact and 0.16 m safety envelopes. No requested TTC was imposed, and no hit, safe-arrival, or adversarial-performance claim is made from these videos.

## Measured final results

All distances below are millimetres. “Anchor IK error” measures deviation from the requested deformed glove path at its selected attack anchor; it is not distance to the head. The maximum path error includes all frames. The last column is the maximum world glove-orientation change from the source, in degrees.

| Punch | Anchor IK error (mm) | Maximum path error (mm) | Minimum visible-head separation bound (mm) | Glove orientation change (deg) |
|---|---:|---:|---:|---:|
| jab | 0.265 | 0.599 | 2.967 | 1.54 |
| cross | 0.293 | 0.566 | 4.682 | 1.46 |
| left hook | 0.408 | 0.436 | 5.026 | 0.73 |
| right hook | 0.333 | 8.018 | 5.934 | 2.87 |

Across all 855 poses, G1's head motion and G1 qpos changes were exactly zero. The nonselected human coordinates and endpoint poses match the source exactly. Recomputed combined-scene body FK matches the exported human body arrays exactly. The right hook has a larger transient path error than the other clips, but remains below 1 cm over this experiment and reaches its attack anchor accurately.

All four 1280 × 720, 30 fps MP4s passed full FFmpeg decoding and frame-count checks. Final Slurm job **5884609** completed successfully on one Tesla V100 in 23 seconds of batch time; the application took approximately 21.2 seconds. The existing GPU preflight confirmed actual NVIDIA EGL rendering on the allocated device. Earlier geometry trials remain in separate job output directories.

Evidence: [final manifest](../outputs/g1-targeting/manifest.json), [pose/media validation](../outputs/g1-targeting/validation.json), [GPU runtime](../outputs/g1-gpu-5884609/gpu_runtime.json), [Slurm accounting](../outputs/g1-gpu-5884609/slurm-accounting.txt), and [test results](../outputs/g1-targeting-tests.txt).

The full test suite passed: **31 tests and 4 subtests**, including new scene-isolation/asset-portability checks and real four-clip target-reaching tests. Ruff and shell syntax checks also passed. The final CLI clearance and all final exported poses were additionally checked through the artifact validation above.

## Reproduction and implementation

```bash
cd human-punching
bash scripts/render_g1_cpu.sh
# Or, using the tested server GPU setup:
bash scripts/submit_g1_gpu.sh
```

CPU output defaults to `outputs/g1-targeting`. GPU jobs write `outputs/g1-gpu-JOBID/preview`, with their own runtime and completion records. The final reviewed GPU output has been copied to the stable `outputs/g1-targeting` directory; the original GPU job artifacts remain separate. Generated assets/videos stay excluded from Git.

For geometry-only export or another placement:

```bash
bash scripts/render_g1_cpu.sh --no-render --output outputs/g1-targeting-check
bash scripts/render_g1_cpu.sh --robot-distance 0.85 --robot-lateral 0.20 --clearance 0.044
```

A changed placement/height/clearance requires a new reach and mesh-separation review; the current solver is not a general feasibility-certified planner.

| File | Responsibility |
|---|---|
| [g1_scene.py](../src/human_punching/g1_scene.py) | Copy/prefix assets, construct the shared scene, set standing pose, expose entity maps and head geometry. |
| [targeting.py](../src/human_punching/targeting.py) | Smooth path deformation, active-arm IK, and updated trajectory export. |
| [target_cli.py](../src/human_punching/target_cli.py) | Orchestrate the four punches, hold G1 state, measure geometry, export manifests and trajectories. |
| [target_render.py](../src/human_punching/target_render.py) | Render the shared scene with two views and both playback speeds. |
| [target_g1_volta.sbatch](../scripts/target_g1_volta.sbatch) | Run the project-owned CPU IK/native EGL workflow in a Slurm GPU allocation. |

## Remaining limitations and the next useful refinement

The source's foot hovering/sliding remains visible because this experiment preserves its root and lower body. The arm rig has unrestricted ball joints and placeholder inertias; no full-body/self-collision, joint-limit, torque, balance, or return-to-guard dynamics validation has been performed. Exact endpoint preservation is a geometric property, not evidence of physical recovery. Punch-family annotations retain their existing candidate status.

For more natural motion across a larger target range, the next step is constrained kinematic retargeting: support-foot constraints, appropriate shoulder/elbow/wrist limits, smooth changes across the entire trajectory, and coordinated torso/knee motion where an arm-only solution is insufficient. Offline trajectory optimization is a reasonable tool for that. For physical execution, add dynamics and a feedback tracker separately. Neither requires introducing machine learning merely to aim these four recorded punches at this stationary target.
