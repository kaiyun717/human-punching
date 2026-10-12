# Implementation and results

Date: 2026-10-10. This report covers the isolated motion-retargeting preview, not physical adversary training.

## Delivered implementation

`src/human_punching/motion.py` loads the four exact, hash-pinned raw DuoBox selections inherited from `robust_sav`. It includes both arms, all 21 source joints, the root trajectory and 120 Hz timing. The default preview adds 0.25 seconds of existing recorded context on either side. It does not fabricate guard poses or close the sequence into a loop. Family and outward-completion annotations are inherited machine-selected candidates, still subject to human review.

`src/human_punching/human.py` generates a plain light-blue mannequin from triangle meshes. It has a featureless oval head, smooth torso/limbs and one solid mesh per foot. Facial features, hair, clothing details, muscle overlays and separate visible toes have been removed. The left glove is blue; the right is red. Both are flattened rounded blocks with a pale orientation mark; all glove geometry rotates rigidly with its corresponding Hand body. The source wrist rotations are preserved; see the [glove audit](04-glove-rotation-audit.md). These colors are review aids, not the accepted Task 08 perception appearance contract. The body is a segmented rigid surface, as permitted for the initial mesh milestone in the Task 09 plan. It is not SMPL-X and is not a reconstruction of the capture participant. No external human model or trained weights are required.

The target is scaled so its neutral T-pose crown-to-sole mesh extent is exactly 1.800000 m. This measures actual compiled mesh vertices, including the featureless head and solid feet. A bent-knee boxing posture is naturally shorter. Mesh counts and the updated normalization scale are reported below. The source skeleton still includes its toe joints to preserve the recorded joint schema, but those joints have no visible geometry.

`src/human_punching/retarget.py` transfers the complete motion to the generated MJCF. For each joint, the source parent-local rotation becomes a MuJoCo ball-joint quaternion. Parent-local bone offsets and world root displacement use the same constant scale as the mesh. The first root-forward direction defines one world yaw alignment for the entire clip. A fixed planar translation centers the initial root. A fixed vertical translation places the lowest mesh vertex over the complete clip at 2 mm above the z=0 reference plane. The rendered floor is at z=-3 mm. There is no per-frame floor clamp, root stabilization or hand-only target correction.

This is a faithful scaled replay on a compatible rig, not constrained IK onto a differently proportioned anatomical model. It preserves the source proportions and joint motion. Solving a constrained retargeting problem becomes necessary if those proportions, DOFs, joint limits, contacts or physical goals change.

`src/human_punching/render.py` renders the actual MJCF and poses with native MuJoCo 3.11.0. Each video has an oblique view and a side view, a real-time pass and a half-speed pass. A short yellow trail marks the active wrist. Text labels and the progress line are added after rendering; the body, geometry and camera views come from MuJoCo. No image-generation tool was used. The camera remains fixed within each clip.

## Output contract

`outputs/preview/model/human.xml` references relative OBJ assets under `model/meshes`. It loads directly in MuJoCo 3.11.0. `human_metadata.json` records exact scale, hierarchy, target offsets, mesh counts and morphology limitations.

Each punch directory contains:

- `<punch>.mp4`: 1280 × 720, H.264, 30 fps; real-time then half-speed replay, with a 0.3-second hold after each pass.
- `<punch>.png`: still at the inherited outward-completion source frame.
- `trajectory.npz`: 120 Hz samples with `time_s`, original `source_frames`, `qpos`, `qvel`, `body_names`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w` and `body_ang_vel_w`.
- `manifest.json`: source hashes/crop bounds, orientation/origin/scale/floor transforms, FK errors, bounds, wrist speed and media metadata.

The root `manifest.json` records runtime versions, the full joint layout, and SHA256 hashes for the XML and every referenced OBJ. It also contains the four motion manifests. `overview.png` is the four-video visual index.

The rig has one free root and 20 ball joints: **87 qpos coordinates and 66 velocity coordinates**. Quaternions are scalar-first `wxyz`; body arrays follow the explicit 21-name list and exclude the world body. `qvel` uses MuJoCo's tangent-space convention via `mj_differentiatePos`. Body linear velocities use clip-local finite differences; body angular velocities use relative rotations in the world frame. Quaternion signs remain continuous within each clip.

These arrays are **not** WBC's scalar-hinge `joint_pos` format or an accepted `robust_sav` checkpoint. The next exporter must adapt to the final physical human's DOFs and recompute body arrays from that exact model. No derivatives are taken across different clips.

## Compute and reproducibility

The default `outputs/preview` artifacts were generated using **CPU OSMesa**, with no learned controller. Subsequent native GPU rendering on this server’s Volta partition is documented in the [GPU server guide](05-gpu-server-guide.md), with separate job-specific outputs. `outputs/compute_inventory.txt` records available Slurm partitions; this connected cluster does not expose the requested `mit_normal_gpu` / L40S resources. `scripts/render_l40s.sbatch` preserves the requested target account, partition and GPU selection, checks the actual GPU name, and uses EGL. That script has not been executed on an L40S here.

This is the same native MuJoCo engine version and MJCF asset format used by `robust_sav`. It is not an execution test of the batched mjlab/MuJoCo-Warp physics or raycast camera implementation. Rigid mesh surfaces were chosen in part to avoid depending on native-only deformable skins, but actual GPU visibility/occlusion must still be checked.

Run `bash scripts/render_cpu.sh` from this project to regenerate the outputs. `requirements-preview.lock` records the tested Python 3.12 package environment. The README includes isolated installation and the target-cluster Slurm command. All persistent project files, environments, caches and outputs are inside `human-punching`.

## Validation and limitations

The test suite covers real source hashes and whole-body reconstruction, rejection of corrupt inputs before pickle decoding, recorded-context padding, exact compiled mesh height, the full joint layout, scaled offset preservation, MuJoCo FK for all four families, quaternion continuity, root motion, the constant floor offset and clip-local tangent-space derivatives. Additional glove tests rotate each wrist independently about all three axes and compare the full hand/glove orientation to the recorded source on every frame. The measured source-to-MuJoCo FK errors are reported below. Test output is saved in `outputs/tests.txt`.

The XML deliberately has **no actuators, no gravity and no collision participation**. Its nonzero inertias are structural placeholders. Rendering calls `mj_forward` on recorded configurations, not a physical tracking policy. The result therefore establishes visual and kinematic fidelity only. Segmented surfaces may show seams or overlap, the source feet may slide or hover, and there is no biomechanical joint-limit or collision guarantee. Disablement is explicit in both model metadata and tests.

Future Task 09 work needs validated masses/inertias/joint limits/effort bounds, contact proxies and a physical tracker; a fixed glove attachment consistent with rendered and safety geometry; full-cycle annotations and transition validation; target/TTC adaptation; actual GPU rendering checks; and finally the expanded human/game state and legal boundary-only planner. The compatibility report details why the existing disturbance vector cannot simply be replaced with a punch ID.

## Measured final artifacts

Simplified mesh revision: 29 mesh assets, 9,562 vertices, 19,008 triangles. Neutral standing height: 1.800000 m; uniform scale: 1.00431662.

| Punch | Source frames (with context) | Source duration | MP4 duration | Maximum FK error |
|---|---:|---:|---:|---:|
| jab | 340–548 (209) | 1.733 s | 5.867 s | 2.91e-07 m |
| right hook | 433–650 (218) | 1.808 s | 6.067 s | 2.99e-07 m |
| left hook | 371–593 (223) | 1.850 s | 6.200 s | 3.27e-07 m |
| cross | 193–397 (205) | 1.700 s | 5.767 s | 3.61e-07 m |

Generation took 101.8 seconds. All four MP4s passed full FFmpeg decoding, resolution/frame-count checks, and trajectory checks. All 30 XML/OBJ hashes match. Evidence: `outputs/media_validation.json`.

Glove rotation was checked across both hands in all 855 source frames and with independent 90° rotations about every wrist axis. No hand-to-glove rotation error was found; the flattened glove geometry and pale patch make the existing rotation visible. Detailed results are in `outputs/glove_rotation_audit.json` and [report 04](04-glove-rotation-audit.md).

Validation: 22 passed, 4 subtests passed in 14.34s. Both source repositories remain unchanged.

The earlier detailed mesh/videos are retained under `outputs/archive/preview-v1-detailed`; the default `outputs/preview` now contains only the simplified mannequin.

[Overview](../outputs/preview/overview.png) · [Jab](../outputs/preview/jab/jab.mp4) · [Right hook](../outputs/preview/right_hook/right_hook.mp4) · [Left hook](../outputs/preview/left_hook/left_hook.mp4) · [Cross](../outputs/preview/cross/cross.mp4)
