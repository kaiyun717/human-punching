# Physical human pilot: findings, implementation, and remaining gates

11 October 2026. Development and generated artifacts are confined to `human-punching`. The sibling `robust_sav`, `mjlab`, and DuoBox inputs were read only. No Git staging, commit, or push was performed.

## What changed

The project now has a physical human with a free pelvis, bounded torque control, gravity, and foot contact. It also has the exact Task 08 contact contract, contact-directed references, an isolated compatible GPU environment, real Warp dynamics/camera checks, and a cycle-selection interface. These are concrete preliminary components; they do not constitute a trained protagonist–adversary game.

The strongest backend result is a controlled jab at **source playback speed on actual MuJoCo-Warp through pinned mjlab**, with a fixed G1 in the same scene. The closest glove/head center distance was **0.112658 m**, maximum foot drift **5.05 mm**, and double-foot contact **99.786%**. Root actuator force was zero and torque-vector bounds were respected. Native MuJoCo and Warp differed by at most **0.686 micrometres** in root position. See [GPU evidence](09-physics-runtime.md) and [job 5891019](../outputs/warp-tracking-5891019/report.json). The CPU computes feedback torques from actual GPU states; Warp integrates physics. This validates correctness before a fully GPU-resident controller.

All four families now have an accepted isolated native-physics case. Only the jab has passed at source playback speed; the cross and hooks currently require **0.3× speed**. A separate **two-jab sequence** physically recovers its guard before the next launch, without resetting the state.

[Four-case overview](../outputs/physical-review/overview.png) · [Jab](../outputs/physical-review/jab/jab.mp4) · [Cross](../outputs/physical-review/cross/cross.mp4) · [Left hook](../outputs/physical-review/left_hook/left_hook.mp4) · [Right hook](../outputs/physical-review/right_hook/right_hook.mp4) · [Two jabs](../outputs/physical-review/two_jabs/two_jabs.mp4)

| Accepted native case | Source speed | Minimum active glove/head distance | Maximum sole drift | Source job |
|---|---:|---:|---:|---|
| Jab | 1.0× | 0.112658 m | 5.05 mm | 5890952 |
| Cross | 0.3× | 0.137540 m | 7.80 mm | 5890924 |
| Left hook | 0.3× | 0.116591 m | 5.50 mm | 5890924 |
| Right hook | 0.3× | 0.113157 m | 5.13 mm | 5891006 |

The right hook uses a 12 cm forward stance offset, deeper reference aiming, and gain scale 3; the other accepted cases use gain scale 1.5. **Effort limits are unchanged.** These are separate conditioned cases, not a universal controller shown to generalize. Their final reference and executed-pose audits have no cone violations beyond the recorded tolerance; see [review manifest](../outputs/physical-review/manifest.json).


## 1. The 0.14 / 0.16 m definitions transfer directly

For each glove, use its attached center, not its wrist origin. Contact occurs if either center is at most **0.14 m** from the head center. Glove clearance is safe only if both are at least **0.16 m** away. Thus 0.15 m is unsafe without contact. Equality belongs to contact at 0.14 and safe clearance at 0.16. These definitions are already supplied by the Task 08 wrapper in `robust_sav`; older generic collector defaults are not the Task 08 operating values.

The physical human does not change the thresholds. It changes who generates the center trajectory: body/hand forward kinematics now supplies it. Both gloves remain hazards during all phases. The new physical pilot records both centers at every **2 ms physics step**, separately from the video sampling rate.

“Physical contact” here is an analytic sphere-envelope event, not a requirement to exchange collision impulses. The existing Task 08 sensing fists also have impact contacts disabled. Accordingly, the new human has floor contact while the G1 target is fixed and glove/G1 impact response remains off. This preserves the avoidance event without introducing a different impact-driven transition model. Visual meshes may overlap before/after proxy contact because the visible head and gloves are not those spheres.

The prior videos stopped around 0.184 m and therefore never hit this contract. The new kinematic bank has **12 contacting variants**, four families with three approach directions. Contact radii never change when aiming deeper inside the envelope to tolerate tracking error. See [the complete contract](10-contact-contract.md) and [contact reference results](11-reference-contact.md).

## 2. A physical human can be simple

The new model retains the plain blue body and uses primitive capsules, ellipsoids, spheres, and boxes for dynamics. It assigns segment masses summing to **75 kg** and asks MuJoCo to integrate each primitive's inertia. The mass is a chosen scenario, not an estimate of the capture actor. No SMPL assets, muscles, fingers, toes with detailed surfaces, or anatomical scans are required.

The pilot keeps the source's **20 ball joints / 60 torque motors**, plus an unactuated free root. Limits are isotropic quaternion-rotation cones, not independent anatomical elbow/knee limits. Torque is clamped by each joint's three-axis vector norm, with per-axis XML limits as an additional bound. Tiny toe articulations remain for archive compatibility, but their reference rotations are set to identity. The physical model uses two red gloves to match the existing red-only perception assumption; earlier blue/red preview assets are retained as historical outputs.

This is an engineering rigid-body abstraction, **not a biomechanically calibrated human**. Human self-collision and human/G1 body collision are currently disabled. Contact between the human and floor is enabled. Before calling a learned policy “human-like,” replace unrestricted elbow/knee ball freedom with suitable anatomical joints or enforce stronger axis constraints, audit segment properties and effort ranges, and enable/audit selected self-collisions.

If the adversary only needs to choose among bounded, precomputed glove paths, even this physical rig is optional: a kinematic trajectory-bank adversary is a legitimate simpler game. Physics is needed when claims concern executable human motion, balance, momentum, support, or strength. The evidence and primary-source comparisons are in [physical-human research](08-physical-human-research.md).

## 3. What improved the motion, and what failed

- Foot/COM IK creates a deliberately planted double-support version. It fixes the physical sole positions/orientations and brings the projected COM near the support center. It is not recovery of unknown true capture floor contacts.
- Head aiming is repeated after support correction. Planting feet changes root position; assuming the old aimed glove path survived would falsely report hits.
- A quaternion sign-continuous smoothing option suppresses source rotation spikes. Original hook rotations reached roughly 61–68 rad/s. The altered references, speed factors, aiming depth, and stance offsets are recorded explicitly.
- Pose-only PD fell despite a statically feasible initial guard. Joint torques needed to support gravity were missing. Bounded PD plus contact-aware inverse statics and root-error feedback holds the guard. Its estimated support forces are only a calculation; they are never applied as external forces or used to overwrite the pelvis.
- The controller assumes double support. Its friction projection is not a full constrained dynamics optimizer or stability proof. Contact, slip, saturation, and uprightness must be measured in rollout.

The initial unchanged-speed, unchanged-target jab stayed upright in both native and Warp backends but missed at **0.162581 m**. This was a tracking limitation, not a backend incompatibility. A later, smoothed, deeper-aimed native jab succeeded at source playback speed. Slower cross and both hook references also passed. Failed runs are preserved; staying upright alone is not an accepted punch.

A useful diagnosis corrected an earlier reachability suspicion: the final right-hook reference already entered the contact envelope, but the physical glove lagged by up to roughly 13 cm at the important reference pose. Shoulder tracking error and post-IK angular-speed spikes mattered. The shifted stance reduced motion demand; increased feedback gains and revised aiming produced the accepted slowed case without raising strength. Reference smoothing before IK alone does not bound the speed of the final IK output.

The pilot's engineering acceptance gates are deliberately explicit: pelvis height remains above 70% of initial height, root up component above 0.70, sole drift and lift at most 3 cm, floor penetration at most 1 cm, double-foot contact in at least 95% of samples, and mean joint saturation at most 5%, plus an actual 0.14 m contact event. These are diagnostic thresholds, not new definitions of the Robust Safe Arrival safe set. Final recovery must be checked independently.

**Timestep remains a gap:** a 5 ms version diverged, and a second lower-gain 5 ms trial developed large drift/fell. Do not silently replace the accepted G1 plant's timestep or claim direct 200 Hz integration. The demonstrated human controller uses 2 ms physics and a torque update at every step; integration must explicitly choose and revalidate substepping or a controller/model stable at 5 ms.

## 4. GPU and software compatibility

[The isolated physics environment](09-physics-runtime.md) uses accepted mjlab commit `8ee51fbcf806a7419189f706d9e394cbeb7790fa`, Torch 2.8.0+cu128, MuJoCo/MuJoCo-Warp 3.11.0, Warp 1.14.0, and RSL-RL 5.5.0. The current sibling mjlab checkout's newer dependencies are not installed over this stack.

Actual V100 jobs passed CUDA learning operations, standalone Warp kernels, batched floor contacts, 20-ball/60-motor dynamics, combined human/G1 mesh cameras, and the Task 08 torso camera dimensions/mount. L40S/`mit_normal_gpu` is not exposed on this server; jobs use the tested `xeon-g6-volta` allocation described in the [GPU guide](05-gpu-server-guide.md).

Native/Warp physics supporting ball joints does not make stock mjlab joint-position actions and scalar `IdealPD` compatible with this quaternion archive. The current custom controller handles quaternion error and 60 motor channels explicitly. For batched learned tracking, either implement GPU-resident quaternion-aware observations/actions/PD or finish a scalar-joint rig and re-retarget. Do not call the 87-qpos archive a standard scalar motion-tracking motion file.

The camera probe also found extreme near-field occlusion: at a near-contact preview frame the striking glove occupied about 93% of the G1 image, with median depth near 2 cm. Raw RGB/depth/ID rendering works; this does not validate the existing HSV calibration, valid-depth filtering, history/delay handling, encoder, or G1 policy. Two red physical gloves address one color assumption, not the complete perception integration.

The repeated-jab proof simulated 9.79 seconds with two contacts, both guard recoveries, 6.70 mm maximum foot drift, no saturation, and no root force. Its second launch was admitted only after actual guard errors stayed within tolerance for 0.10 seconds. See [sequence evidence](12-cycle-bank.md). Time-to-contact numbers in these pilots start at the controller cycle/clip start; they are **not yet the PDF’s annotated outward-launch TTC**. The stationary head makes the target positions coincide, but the clocks still differ.

## 5. Phases from here

| Phase | Current evidence | Remaining acceptance work |
|---|---|---|
| A — Geometry and runtime contract | Exact radii/equalities, both-glove events, frozen-target distinction, pinned GPU stack tested | Wire the accepted labels and sensor adapter into a new two-entity task without changing old contracts silently |
| B — Physical rig and support | 75 kg primitive rig; unassisted standing; positive mass/inertia and bounded control tests | Anatomical scalar joints or stronger axis constraints, self-collision, property sensitivity, timestep compatibility |
| C — Feasible reference bank | Four contacting kinematic families / 12 directional variants; support-aware reference pipeline | Broader targets/heights, explicit outward-launch annotations and requested TTC bounds, acceleration/torque-feasible timing |
| D — Physical tracking | Native and Warp jab contact; native isolated slower cross and both hooks passed | Full-speed four-family tracking, robust recovery, perturbation tests, fully GPU-resident controller if training needs throughput |
| E — Continuous fixed-target opponent | Boundary-only bank adapter; two physically executed jabs with measured guard recovery passed | Mixed-family continuous sequences, held launch-head snapshot, explicit launch/TTC clocks and empirical reachability masks |
| F — Two-player training | Interface/gap analysis and finite selector contract | Instantiate a second policy, hold its plan across substeps, add plan/phase state to critic/replay, define adversary reward/termination, train/evaluate against fixed G1 first |

Machine learning is **not required** to define the model, aim the motion, or demonstrate the first physical jab. Use motion optimization/time scaling and classical control first. A residual/imitation tracker becomes useful for fast motion and broader robustness after feasibility gates identify what it must learn. A learned tracker will not fix an unreachable target, wrong contact centers, inconsistent clocks, or state resets masquerading as recovery.

`robust_sav` still owns a single-G1 plant with fixed state/observation assumptions. A physical human changes model dimensions, contact masks, observation ownership, camera visibility, timing, resets, and opponent state. A separate scene/adapter is needed before integration. Missing G1 checkpoint/settled-stance artifacts prevent validating a real protagonist here; the baked standing G1 is an explicit fixture, not an inferred trained controller.

## Reproduction and artifacts

From `human-punching`, after the original preview and G1 fixture have been generated:

```bash
bash scripts/setup_physics.sh
mkdir -p outputs/slurm
sbatch scripts/physics_probe_volta.sbatch
sbatch scripts/warp_tracking_volta.sbatch --speed .6
sbatch scripts/physical_pilot_volta.sbatch \
  --input outputs/contact-targeting --balance --reaim --with-g1 \
  --punches jab --speed 1 --smooth 4 --center-target .10 --gain-scale 1.5
```

`physical_pilot_volta.sbatch` runs native MuJoCo physics and GPU EGL rendering. `warp_tracking_volta.sbatch` runs actual GPU physics with a host feedback controller. These are different tests; a GPU allocation alone is not evidence that physical integration ran on the GPU.

Reports and videos retain failed attempts as well as accepted cases. Unit/integration tests cover contact boundaries, attachments, torque signs/norms, freefall, absence of root actuators/forces, fixed-G1 FK/dynamics isolation, reference timing, and guard admission. The complete regression suite passed **71 tests plus 4 subtests**; Ruff passed across source, tests, and scripts. Video payloads and independent reference/rollout audits are recorded with the final review artifacts.

Final media verification: all **five** review MP4s decoded successfully. All report links resolved. The full-speed Warp, continuous-sequence, and final review jobs completed with exit `0:0`; see [validation](../outputs/physical-review/validation.json) and [Slurm accounting](../outputs/physical-review/slurm-accounting.txt).
