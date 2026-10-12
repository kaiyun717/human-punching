# Scope and implementation plan

Date: 2026-10-10. All development and persistent generated files belong to `human-punching`. `../robust_sav`, `../mjlab`, the PDFs, and `../DuoBox` are read-only inputs. No protagonist weights or old output datasets are required.

## Deliverables

1. Audit the actual source revisions and dependency locks against the Task 09 plan, including the change from joint-torque disturbance to a human adversary.
2. Read the four existing, hash-pinned DuoBox selections: jab, cross, left hook, right hook. Retain both arms, spine, legs, and root, rather than just the attacking glove.
3. Build a visible articulated human triangle mesh with a measured neutral crown-to-sole height of 1.80 m; preserve source bone proportions and local joint rotations. Keep its morphology/provenance explicit.
4. Export MuJoCo MJCF, meshes, numerical trajectories and provenance; render four videos with two camera views, real-time and slow replay.
5. Verify source integrity, forward kinematics, height, quaternion validity, clip-local derivatives, and output media. Document limitations and the next integration steps.

## Chosen initial implementation

The same native MuJoCo version used by `robust_sav` is 3.11.0. This preview needs neither PyTorch nor a training checkpoint. The installed sibling `mjlab` source is newer than the revision pinned by `robust_sav`; combining their current requirements directly is incompatible. See the dependency report.

The supplied fitted SMPL-X files contain parameters but no mesh topology or neutral model. To make a reproducible preview without requiring a licensed model download, this milestone authors a procedural, segmented human mesh on the raw 21-joint DuoBox rig. This follows the plan's initial articulated-surface approach but substitutes the body asset. It is not an SMPL-X reconstruction and does not claim subject-specific shape fidelity. Exact uniform height scaling preserves the source proportions, relative joint motion, root displacement and timing; it does not solve biomechanical constrained IK.

The preview writes free-root and ball-joint configurations and calls MuJoCo forward kinematics. It is kinematic motion replay, not a learned or dynamically stabilized human controller. Targeting the G1, TTC adaptation, collision calibration, imitation training and the two-player learner remain later milestones.

## Execution environment

The connected Supercloud login hosts expose `xeon-g6-volta`, `xeon-p8`, and related partitions. `mit_normal_gpu` and L40S were not listed by `sinfo -a`. Do not substitute another GPU type silently. Render this lightweight preview with native MuJoCo's OSMesa CPU backend; provide a separate Slurm worker using the requested `mit_general` / `mit_normal_gpu` / `l40s:1` resources for the target cluster.

Sandboxed shared-filesystem directory creation stalled across reconnects; direct authorized execution restored writes. Installation and all generated artifacts stay under this project. A failed temporary runtime installation used `/tmp` and was abandoned when that temporary filesystem filled; it is not part of the project runtime.
