<section id="overview" class="hero">
<div class="eyebrow">HUMAN-PUNCHING / RESEARCH &amp; IMPLEMENTATION RECORD</div>
<h1>From motion replay<br>to physical punch policies.</h1>
<p class="hero-lead">The complete progression: what we tried, what broke, what changed, and what the evidence now supports.</p>
<p class="hero-meta">Evidence snapshot · 15 October 2026 · Task 09 preparation</p>
<div class="hero-stats"><div><strong>4</strong><span>accepted learned punch families</span></div><div><strong>48</strong><span>joints in the training body</span></div><div><strong>5 ms</strong><span>actual human physics step</span></div><div><strong>20 ms</strong><span>learned actor update period</span></div></div>
</section>

<section id="reading-guide" markdown="1">
## What this report establishes

We began with four DuoBox recordings and a visual mannequin. We now have a constrained, freely simulated human body and **four learned reference-conditioned punch controllers** that make head-envelope contact and return to supported guard. Between those endpoints, we tested retargeting, GPU rendering, real GPU physics, classical control, contact definitions, sequence recovery, anatomical constraints and independent policy evaluation.

The original learned milestone was one jab at 60% source speed and 2 ms physics. The follow-up now establishes **four learned punches at genuine 5 ms physics and 20 ms actor updates**, each passing **64/64 fresh test trials** with accurate tracking and supported recovery. Jab and cross retain uniform source timing; hooks use explicitly retimed references to fit the unchanged strength limits. The [new evidence section](#five-ms) follows the original progression below. The G1 shown in videos remains a standing fixture; no accepted G1 policy or impact response is running.

{{video|outputs/jab-milestone/jab.mp4|outputs/jab-milestone/jab-contact.png|The first learned milestone: jab and recovery|Historical 2 ms GPU physics, with fixed G1 visualization and the unchanged 0.14 m contact threshold. This older reference runs at 0.6× source speed. The later 5 ms work is documented below.|Historical · learned}}

<div class="callout" markdown="1">
**Read the evidence labels.** A plausible video can mean very different things. Kinematic replay prescribes poses; a physical rollout integrates forces; a learned rollout adds policy feedback. GPU rendering alone is not GPU physical integration. Throughout this report, those distinctions are explicit.
</div>

| Evidence label | What actually runs | What it can establish |
|---|---|---|
| Kinematic | Recorded or optimized poses + forward kinematics | Geometry, appearance, attachments and sampled reach |
| Native physics | MuJoCo CPU integration; some videos rendered on a GPU | Bounded physical execution for the declared controller and case |
| GPU physics | MuJoCo-Warp integration through pinned mjlab | The tested model/controller runs on the target GPU backend |
| Learned | PPO residual controller + PD/feedforward + GPU physics | A learned contribution, assessed against a matched nonlearned baseline |

**Project boundary.** All implementation, environments and artifacts are inside `human-punching`. `robust_sav`, sibling mjlab and DuoBox were read-only inputs. Nothing was staged, committed or pushed. Original reports are preserved as historical snapshots; their earlier “not yet implemented” statements may be superseded by later results. The complete report archive appears at the end of this page.
</section>

<section id="scope" markdown="1">
<div class="section-kicker">01 / Establish the scope</div>
## A human opponent is feasible, but not a drop-in disturbance

The original objective was to inspect four punches in the same simulator family as Robust Safe Arrival, then build toward a human adversary that could be plugged into the accepted G1 system. Vision5 discusses adversarial joint-torque disturbances. A human instead generates structured, temporally committed motions, with its own body, controller and sequence state.

The Task 09 plan has a useful separation: **reference construction → physical human tracking → legal adversarial plan selection**. We retained that separation, while moving runtime and shared-scene checks earlier. We did not start by training a two-player game on top of untested human dynamics.

<div class="flow" aria-label="Project progression"><span>Source motion</span><b>→</b><span>Targeted reference</span><b>→</b><span>Physical tracker</span><b>→</b><span class="future">Committed planner</span><b>→</b><span class="future">G1 game</span></div>

### The first compatibility obstacle: identical package versions, different code

The inspected `robust_sav` snapshot was `f4672e3…`; its accepted mjlab source was **`8ee51fbc…`**. The sibling mjlab checkout was **`033ae22a…`**. Both advertised version 1.6.0, but their dependency requirements differed.

| Component | Accepted Task 08 stack | Sibling/newer or optional path | Decision |
|---|---|---|---|
| PyTorch | 2.8.0+cu128 | Sibling mjlab requests ≥2.14 | Preserve accepted source pin in an isolated environment |
| RSL-RL | 5.5.0 | Sibling mjlab requests 5.5.1 | Do not infer compatibility from mjlab's version label |
| MuJoCo / MuJoCo-Warp | 3.11.0 | Same simulator family alone is insufficient | Verify actual stepping, cameras and controller APIs |
| Warp | 1.14.0 | Tool/API differences encountered in practice | Fix probes against installed APIs |
| Optional WBC | Not required by this pilot | Inspected revision constrained NumPy &lt;2.5; accepted inventory used 2.5.3 | Defer until a concrete benefit justifies reconciliation |
| SMPL-X body assets | Required model files absent | DuoBox fitted arrays do not contain the full body model | Use a procedural preview first; later adopt the planned SOMA fallback |

The final physics environment is project-local and pinned. This preserved Task 08 while allowing actual Warp tests and later PPO. Source identity, runtime semantics and model dimensions matter as much as a dependency installer succeeding.

**Evidence:** [initial scope](00-scope-and-plan.md), [dependency audit](01-compatibility-assessment.md), [revised phases and integration gaps](06-task09-revised-phased-roadmap.md).
</section>

<section id="source-motion" markdown="1">
<div class="section-kicker">02 / Read the motion correctly</div>
## A local-versus-global rotation mistake would have moved joints by metres

We retained the full 21-joint hierarchy: root, torso, both arms and legs. The attacking actor is `ma`; actor ordering cannot be used to identify the attacker. Four inherited candidate crops supplied the initial examples.

| Family | Source recording | Original inclusive crop | Active hand |
|---|---|---:|---|
| Jab | `000_ma_jab_liu_defense_joints.npy` | 370–518 | Left |
| Cross | `003_ma_cross_liu_defense_joints.npy` | 223–367 | Right |
| Left hook | `006_ma_leadhook_liu_shanbi_joints.npy` | 401–563 | Left |
| Right hook | `007_ma_rearhook_liu_shanbi_joints.npy` | 463–620 | Right |

**Failure avoided.** The root rotation is global, but child rotations are parent-local. Interpreting all matrices as global gives maximum joint-position errors of **1.64–1.76 m**. Hierarchical composition reconstructs source positions within **2.87–3.28 × 10⁻⁷ m**; final source-to-MuJoCo preview FK error is at most **3.61 × 10⁻⁷ m**.

**Data-handling fix.** Original NumPy dictionaries require pickle deserialization. The loader verifies exact bytes against the inherited SHA256 allowlist before loading those same bytes. Derived archives use `allow_pickle=False`.

**Remaining limits.** The 120 Hz clock is inherited from the source-code convention, not independent timestamp calibration. Four crops from one attacker are sufficient for debugging and a tracking pilot, not person-disjoint generalization. Candidate family and full-cycle labels still need broader review. Added guard padding does not establish a measured physical guard.

**Evidence:** [DuoBox audit](02-duobox-motion-audit.md), [preview manifest](../outputs/preview/manifest.json).
</section>

<section id="appearance" markdown="1">
<div class="section-kicker">03 / First visual result and user feedback</div>
## Simplify the human, then make wrist rotation visible

The first surface was too realistic for the intended use. We removed facial features, hair, clothing detail, muscle detail and individual toe surfaces. The approved appearance became a featureless light-blue mannequin with solid feet. This changed presentation; it did not improve physical realism by itself.

<div class="media-grid two">
{{image|outputs/archive/preview-v1-detailed/overview.png|Initial detailed surface — retained as historical evidence of the first preview. This was kinematic playback, not a physical human.|Before}}
{{image|outputs/preview/overview.png|Simplified approved surface — four full-body source motions, plain blue appearance and readable hand markers.|After · kinematic}}
</div>

The simplified preview retained a free root and **20 ball joints**, with `nq=87`, `nv=66`. Its 29 mesh assets contained 9,562 vertices and 19,008 triangles. A single uniform scale produced exactly **1.800000 m** neutral crown-to-sole height while retaining source proportions. A fixed whole-clip floor translation avoided per-frame floor clamping. Later, the exact-height requirement was relaxed, enabling the current native 1.7664 m body.

### Glove rotation: investigate the apparent defect before changing the motion

The gloves looked as though they did not rotate. The numerical audit showed that the rigid hand attachment was already correct. Rounded shapes hid the rotation visually. Both hands were checked across **855 padded frames**, plus independent 90° wrist rotations about each axis.

| Rotation audit | Maximum observed error |
|---|---:|
| Source versus simulated hand orientation | 3.88 × 10⁻¹² degrees |
| Glove rotation relative to its hand | 3.1 × 10⁻¹⁴ degrees |
| Glove position relative to its hand | 4.04 × 10⁻¹⁶ m |

The fix was a flatter glove profile and pale back-of-hand orientation patch, rigidly attached to the hand. No artificial wrist motion was added. The early blue-left/red-right colors helped distinguish hands; later physical previews show two red gloves to match the existing detector assumption. For the latest scalar body, the renderer applies this color without changing the frozen physical XML.

<details class="media-details"><summary>Watch all four original simplified previews</summary><div class="media-grid two">
{{video|outputs/preview/jab/jab.mp4|outputs/preview/jab/jab.png|Jab|Original full-body replay; no balancing controller, gravity or contact simulation.|Kinematic}}
{{video|outputs/preview/cross/cross.mp4|outputs/preview/cross/cross.png|Cross|Original timing and full hand orientation retained.|Kinematic}}
{{video|outputs/preview/left_hook/left_hook.mp4|outputs/preview/left_hook/left_hook.png|Left hook|Large wrist/hand orientation changes remain visible in the glove marker.|Kinematic}}
{{video|outputs/preview/right_hook/right_hook.mp4|outputs/preview/right_hook/right_hook.png|Right hook|Whole-body source motion, including source foot motion.|Kinematic}}
</div></details>

**Limitation.** This rig had placeholder inertias, no actuators and disabled gravity/contact. Sliding or hovering feet were compatible with its replay-only purpose. It was never evidence of a balanced or anatomically constrained boxer.

**Evidence:** [implementation report](03-implementation-and-results.md), [glove audit](04-glove-rotation-audit.md), [numerical rotation results](../outputs/glove_rotation_audit.json).
</section>

<section id="gpu-rendering" markdown="1">
<div class="section-kicker">04 / Make the server usable</div>
## GPU rendering worked after resolving the actual device mapping

The requested L40S / `mit_normal_gpu` environment was not exposed on the connected server. The available cluster was `txe1`, with **Tesla V100-PCIE-32GB** GPUs in `xeon-g6-volta`, two per node. The four-node association allowance is consistent with eight GPUs, but does not reserve them or guarantee immediate availability.

The example Slurm script was adapted to project-local environments, caches and output directories. A native rendering preflight checks allocation, device mapping, NVIDIA OpenGL and nonconstant pixels.

**Obstacle.** `CUDA_VISIBLE_DEVICES` can contain a UUID. CUDA logical device 0 does not necessarily correspond to EGL device 0. The preflight maps UUID/PCI identity to the physical EGL device and verifies the active display. A successful `nvidia-smi` query alone would not have proved the renderer used the allocated GPU.

| Experiment | Result |
|---|---|
| Job 5884571 | Native GPU rendering probe passed |
| Job 5884573 | All four previews on one V100; 15.67 s application time |
| Array 5884574, tasks 0–3 | Four independent tasks across two nodes/four GPUs; two tasks needed EGL device 1 despite CUDA logical device 0 |
| CPU/GPU replay comparison | Trajectories matched within 10⁻¹²; all eight generated videos fully decoded |

This established **GPU rendering**, not GPU dynamics. Actual Warp integration required the separate tests in section 07.

**Evidence:** [GPU server guide](05-gpu-server-guide.md), [GPU validation](../outputs/gpu-validation.json), [runtime device identity](../outputs/gpu-5884573/gpu_runtime.json), [Slurm accounting](../outputs/gpu-slurm-accounting.txt).
</section>

<section id="g1-targeting" markdown="1">
<div class="section-kicker">05 / Put a standing G1 in the scene</div>
## Offline trajectory deformation and IK were enough to aim the previews

The real G1 assets were loaded in a fixed knees-bent standing pose. The robot pelvis was placed at XY `(0.85, 0.20)`, yaw π; its head proxy center was approximately **(0.85396, 0.20, 1.23149) m**. The shorter target required substantial downward arm redirection—some individual arm changes approached 88°.

Smooth glove-path deformation plus active-arm Jacobian IK aimed the source punch toward the head. FK verified the result. Root, trunk, legs, inactive arm, source timing and endpoint poses were retained. This stationary geometric problem did **not** need machine learning.

### Why the first apparent contact was misleading

Spherical diagnostic proxies did not match the visible G1 head and glove surfaces. Small additional proxy clearances still allowed visible overlap. Native mesh CCD distances also proved inconsistent for this diagnostic, so an independent world-vertex projection separation bound was used.

| Geometry iteration | Additional proxy clearance | What happened |
|---|---:|---|
| Job 5884595 | 3 mm | Jab centers 0.143786 m apart; glove mesh still overlapped the head proxy by 8.698 mm |
| Job 5884596 | 15 mm | Positive glove-mesh/proxy gap, but the visible robot head exceeded its proxy |
| Job 5884609 | 44 mm | Visually separated; minimum projection separation bounds 2.967–5.934 mm across families |

<div class="media-grid two">
{{image|outputs/g1-gpu-5884595/preview/jab/jab-contact.png|Early aiming iteration: proxy proximity did not prevent visible surface overlap.|Geometry diagnostic}}
{{image|outputs/g1-targeting/jab/jab-contact.png|Final visual preview: the jab stops at about 0.1847 m center separation. This is a near-head preview, not an accepted analytical contact.|Kinematic near-miss}}
</div>

{{video|outputs/g1-targeting/jab/jab.mp4|outputs/g1-targeting/jab/jab-contact.png|First targeted jab beside G1|A deliberately separated visual preview. Despite historical “contact” filenames, this final preview stays outside both 0.14 m contact and 0.16 m safe-clearance boundaries.|Kinematic}}

<details class="media-details"><summary>Inspect the remaining targeted families</summary><div class="media-grid two">
{{video|outputs/g1-targeting/cross/cross.mp4|outputs/g1-targeting/cross/cross-contact.png|Targeted cross|Fixed G1; active-arm retargeting only; no physical execution.|Kinematic near-miss}}
{{video|outputs/g1-targeting/left_hook/left_hook.mp4|outputs/g1-targeting/left_hook/left_hook-contact.png|Targeted left hook|Visible-head clearance was the objective at this historical stage.|Kinematic near-miss}}
{{video|outputs/g1-targeting/right_hook/right_hook.mp4|outputs/g1-targeting/right_hook/right_hook-contact.png|Targeted right hook|Geometric aim does not establish balance, strength or anatomical feasibility.|Kinematic near-miss}}
</div></details>

The final G1 did not move across 855 poses. Attack-anchor IK error was below 0.41 mm across families, but the largest trajectory-wide error was 8.018 mm for the right hook. The original lower-body limitations remained.

**Evidence:** [standing-G1 targeting report](07-standing-g1-targeting.md), [final validation](../outputs/g1-targeting/validation.json).
</section>

<section id="contact" markdown="1">
<div class="section-kicker">06 / Align with the actual safety contract</div>
## Contact is a center-distance event; visible overlap is a separate question

The page-33 definitions transfer directly to the human. The head proxy radius is 0.06 m, the glove proxy radius is 0.08 m, and the additional safety margin is 0.02 m. FK supplies the attached glove center; the wrist origin or visible mesh centroid is not interchangeable with it.

<div class="thresholds"><div class="danger"><strong>d ≤ 0.14 m</strong><span>Contact if either glove meets this condition</span></div><div class="caution"><strong>0.14 &lt; d &lt; 0.16 m</strong><span>Unsafe clearance without analytical contact</span></div><div class="safe"><strong>Both d ≥ 0.16 m</strong><span>Safe glove clearance; other SAV conditions still apply</span></div></div>

Both gloves remain hazards during attack, return, guard and waiting. Equality belongs to contact at 0.14 m and safe glove clearance at 0.16 m. The production Task 08 wrapper already carries these thresholds; older generic defaults do not redefine the operating contract. No change to `robust_sav` was necessary.

**Correction.** The preceding 0.184 m near-head previews did not hit this contract. The next bank aimed near **0.135 m** center separation, creating **12 contacting reference variants**: four families × approach yaw −6°, 0°, +6°. Deeper aiming never changed the contact threshold.

| Base reference | Minimum active-glove distance | First contact from clip start |
|---|---:|---:|
| Jab | 0.135798 m | 0.762704 s |
| Cross | 0.135854 m | 0.627309 s |
| Left hook | 0.134628 m | 0.849237 s |
| Right hook | 0.134932 m | 0.732864 s |

{{image|outputs/contact-videos-5890904/preview/overview.png|The new four-family contacting references. These establish sampled geometric contact, not physical support or torque feasibility.|Kinematic contact bank}}

{{video|outputs/contact-videos-5890904/preview/jab/jab.mp4|outputs/contact-videos-5890904/preview/jab/jab-contact.png|The corrected contact-directed jab reference|The same fixed 0.14 m threshold now registers a contact. This remains pose replay.|Kinematic}}

Analytical contact is not a collision impulse. The human can have genuine floor dynamics while human–G1 impacts remain disabled, preserving the avoidance event without adding a new impact-driven transition model. These clip-start times are also not yet the PDF's reviewed outward-launch TTC.

**Evidence:** [contact contract](10-contact-contract.md), [contact-reference report](11-reference-contact.md), [12-variant manifest](../outputs/contact-targeting/manifest.json).
</section>

<section id="warp-runtime" markdown="1">
<div class="section-kicker">07 / Test the real simulator and camera path</div>
## Dependency compatibility was demonstrated by execution

The isolated `.venv-physics` used the accepted mjlab revision, Torch 2.8.0+cu128, MuJoCo/MuJoCo-Warp 3.11.0, Warp 1.14.0 and RSL-RL 5.5.0. Its complete installed inventory passed dependency checks. The implementation did not install sibling mjlab HEAD over the accepted stack.

Two practical defects appeared before the final probe passed. An obsolete `wp.context` metadata accessor was replaced by `wp.get_cuda_toolkit_version()`. Visual inspection then exposed swapped red/blue channels in packed camera RGB; extraction was corrected to bits 16/8/0 and a red-object assertion added.

| Final probe, job 5890856 | Executed evidence |
|---|---|
| CUDA learning | Five Adam updates; test loss 1.0700 → 0.9883 |
| Warp kernel | Correct 512-element CUDA computation |
| Floor contact | 16 falling-sphere worlds; settled height 0.099633 m |
| Combined scene camera | Two worlds, 64 meshes, 98 geoms; RGB, depth and typed IDs |
| Physical human | Four worlds, 100 steps; finite state and motor response |
| Task 08 camera geometry | 672 × 376, 54° FOV, inherited torso mount and tilt |

<div class="media-grid two">
{{image|outputs/physics-probe-5890856/actual-scene-camera.png|A raw combined human/G1 Warp camera probe. Small diagnostic raster; inspect it as a runtime check, not a presentation render.|GPU camera}}
{{image|outputs/physics-probe-5890856/task08-camera.png|The inherited torso camera near a punch: the glove covers roughly 93% of the image. The near-field occlusion is a real integration limitation, not a broken report image.|Task 08 camera probe}}
</div>

**Remaining limitation.** At that near-contact frame, median glove depth was approximately **0.0204 m**, and only **4.76%** of pixels were within the accepted 0.1–3 m depth range. Raw rendering does not validate HSV calibration, valid-depth filtering, sensor histories/delay, the encoder or G1 policy. The historical blue glove also disagreed with a red-only detector; two red gloves solve the color assumption, not occlusion.

**Evidence:** [runtime report](09-physics-runtime.md), [runtime results](../outputs/physics-probe-5890856/runtime.json), [camera metrics](../outputs/physics-probe-5890856/task08-camera-metrics.json).
</section>

<section id="classical-physics" markdown="1">
<div class="section-kicker">08 / A real body needs support, not just poses</div>
## The first physical rig fell; support-aware control made it useful

Research into CMU-style rigid bodies, DeepMimic, SMPLSim, PHC and later SOMA/ProtoMotions supported a simple conclusion: the simulation needs masses, inertias, joints, contacts and bounded actuation, but it does not need realistic skin or muscles. A detailed mesh alone supplies none of those guarantees.

The first dynamics prototype retained the procedural DuoBox hierarchy: **20 ball joints, 60 torque motors, a free unactuated pelvis and a declared 75 kg mass scenario**. Primitive shapes supplied inertias and foot contact. Quaternion-aware control and per-joint torque-vector caps were implemented. This was an engineering diagnostic body, not the final learned body.

### What failed and what changed

| Limitation discovered | Implemented response | What remained uncertain |
|---|---|---|
| Pose-only PD fell despite a plausible initial guard | Contact-aware inverse statics plus bounded PD and joint-only root-error feedback | Double-support assumptions still require rollout checks |
| Original foot motion did not define planted support | Sole position/orientation constraints and a projected-COM objective | A deliberately planted adaptation, not recovered ground-truth capture contacts |
| Support correction moved the root and spoiled hand aiming | Re-aim gloves after lower-body conditioning | Actual tracking can still miss a reachable reference |
| Hook rotation spikes reached roughly 61–68 rad/s | Quaternion sign continuity, smoothing and time scaling | IK can create new derivative spikes after smoothing |
| Difficult right hook lagged behind a contacting reference | Forward stance offset, deeper aiming and stronger feedback gains | Case-specific tuning; torque caps unchanged |

Estimated support forces were used only in the joint-torque calculation. They were never applied as external pelvis forces. The model still had overly permissive ball elbows/knees, engineering cone limits and disabled self-collision.

<div class="media-grid two">
{{video|outputs/physical-5890835/jab/jab.mp4|outputs/physical-5890835/jab/jab.png|Failure: pose tracking without sufficient support|The early physical controller fell; maximum sole drift reached 23.90 cm. A geometric reference and plausible first frame were insufficient.|Failed · native physics}}
{{video|outputs/physical-review/jab/jab.mp4|outputs/physical-review/jab/jab.png|Fixed case: supported full-speed jab on the old rig|Smoothed, deeper-aimed reference with bounded classical control. This succeeds at source playback speed on the earlier ball-joint body.|Classical · native physics}}
</div>

### Accepted classical cases, with their actual restrictions

| Old-body case | Source speed | Closest active glove | Maximum sole drift | Native source job |
|---|---:|---:|---:|---|
| Jab | 1.0× | 0.112658 m | 5.05 mm | 5890952 |
| Cross | 0.3× | 0.137540 m | 7.80 mm | 5890924 |
| Left hook | 0.3× | 0.116591 m | 5.50 mm | 5890924 |
| Right hook | 0.3× | 0.113157 m | 5.13 mm | 5891006 |

The right hook used a 12 cm forward stance offset and gain scale 3; other accepted cases used gain scale 1.5. Effort limits were unchanged. The reference already reached the head; executed shoulder lag and post-IK angular spikes caused earlier misses. These are four conditioned cases, not one generalized controller.

<details class="media-details"><summary>Watch the other three accepted classical cases</summary><div class="media-grid two">
{{video|outputs/physical-review/cross/cross.mp4|outputs/physical-review/cross/cross.png|Classical cross|0.3× source speed; earlier ball-joint body.|Classical · native physics}}
{{video|outputs/physical-review/left_hook/left_hook.mp4|outputs/physical-review/left_hook/left_hook.png|Classical left hook|0.3× source speed; support and bounded-torque checks passed for this case.|Classical · native physics}}
{{video|outputs/physical-review/right_hook/right_hook.mp4|outputs/physical-review/right_hook/right_hook.png|Classical right hook|0.3× source speed, shifted stance and separately tuned gains.|Classical · native physics}}
</div></details>

### Native versus Warp: identify the real bottleneck

An unchanged-speed, unchanged-target jab stayed upright but missed at **0.162581 m** in the matched backend work. That was a tracking limitation, not evidence of a GPU incompatibility. After conditioning, the full-speed jab also passed on **actual Warp integration**, job 5891019: closest distance 0.112658 m, foot drift 5.05 mm, double-foot contact 99.786%, zero root actuation and bounded joint torque.

For this nominal matched rollout, native and Warp root positions differed by at most **0.686 μm**. The feedback controller still ran on the CPU using actual GPU state; Warp integrated the dynamics. The review videos above use native physics with GPU EGL rendering, so the separate raw Warp report is the backend evidence.

**Evidence:** [physical-model research](08-physical-human-research.md), [physical pilot findings](13-physical-pilot-and-next-gates.md), [review manifest](../outputs/physical-review/manifest.json), [full-speed Warp comparison](../outputs/warp-tracking-5891019/report.json).
</section>

<section id="timestep" markdown="1">
<div class="section-kicker">09 / A shared-scene blocker</div>
## Earlier physical controllers failed at a 5 ms step

The accepted G1 plant uses 5 ms physics. At this earlier stage, successful human controllers used **2 ms**. A 5 ms classical trial diverged; reducing its gains did not solve the issue. Job 5890953 fell with **1.014 m** maximum sole drift and missed at **0.159821 m**. The later [native-PD repair](#five-ms) resolves the numerical obstacle without hidden substeps.

{{video|outputs/physical-5890953/jab/jab.mp4|outputs/physical-5890953/jab/jab.png|Failure: the lower-gain 5 ms attempt|An actual failed native-physics rollout. Reducing gains did not recover the support/tracking gate.|Failed · 5 ms physics}}

Later scalar-body native tests also exposed rapid 5 ms joint oscillation. This diagnoses the tested body/controller/gain/discretization combinations; it does not prove that no controller can work at 5 ms.

**Unresolved integration choice.** One shared physics model has one simulation clock. We must explicitly choose and revalidate common substepping or develop a controller stable at the G1 clock. Silently changing the accepted G1 plant would invalidate the compatibility claim.

**Evidence:** [failed-run metrics](../outputs/physical-5890953/jab/jab_metrics.json), [physical pilot limitations](13-physical-pilot-and-next-gates.md), [later scalar diagnosis](../outputs/scalar-native-5895250/diagnosis.json).
</section>

<section id="sequences" markdown="1">
<div class="section-kicker">10 / A clip ending is not guard recovery</div>
## Reject invalid joins, then demonstrate a continuous two-jab sequence

A finite motion-bank interface was added with immutable variants and a boundary-only state machine. A plan freezes the reference and head snapshot. While executing or waiting for guard, all plan-selection masks are false; there is no live target input that could turn the punch into homing.

<div class="flow" aria-label="Sequence state machine"><span>Boundary / choose</span><b>→</b><span>Committed cycle</span><b>→</b><span>Wait for measured guard</span><b>→</b><span>Next boundary</span></div>

**Unexpected failure.** None of the 16 base-family end→start pairs passed admission, including jab→jab. That join had **42.85 mm** root mismatch, **0.34774 rad** maximum joint offset and **3.88956 rad/s** velocity mismatch. Source endpoints were not interchangeable stationary guards. A quintic interpolation bridge can fix reference continuity, but only execution through physics can establish recovery.

The next proof therefore ran two conditioned jabs continuously on the old physical rig: each punch, a 0.4 s endpoint hold, a 1.5 s target bridge, then a measured guard dwell of at least 0.10 s. The second launch occurred at **4.898 s**, only after support, pose, velocity and safe-clearance checks passed.

{{video|outputs/physical-review/two_jabs/two_jabs.mp4|outputs/physical-review/two_jabs/two_jabs.png|Two physical jabs with measured recovery between them|9.79 s of continuous simulation, one initial reset and zero later state overwrites. The bridge supplies targets; it does not teleport the human.|Classical · native physics}}

| Continuous sequence evidence | Result |
|---|---:|
| Contact minima, first / second jab | 0.110467 / 0.110762 m |
| Maximum sole drift | 6.698 mm |
| Both-foot contact | 99.9796% |
| Second-launch root error | 1.632 mm |
| Mean joint saturation | 0% |
| Root actuation / external applied force | Zero |

This is one deterministic repeated-jab case at 0.6× source speed, not mixed-family coverage, a learned sequence or the full Task 09 transition matrix. The later scalar-body learned jab is a separate controller and does not inherit this sequence result automatically.

**Evidence:** [cycle-bank report](12-cycle-bank.md), [rejected transition audit](../outputs/contact-targeting/cycle-bank-audit.json), [continuous-sequence metrics](../outputs/physical-sequence-5890969/metrics.json).
</section>

<section id="training-body" markdown="1">
<div class="section-kicker">11 / Freeze a body suitable for learning</div>
## Adopt the plan's SOMA fallback and constrain its articulation

The classical experiments proved that some motions could execute in the diagnostic surrogate. They did not validate unrestricted ball-joint elbows/knees or human biomechanics. A learned policy could exploit that extra freedom and disabled self-collision, so the next milestone changed the body before investing in imitation learning.

The PDF's primary path was SMPL-X/SMPLSim, with a proposed 63-hinge reduction. Required model files were absent. **SOMA23 was already an explicit fallback in that plan**, so it was a practical starting point rather than an unrelated replacement. The selected ProtoMotions source was pinned at `7a8417a9f55b1586b0d6e8f46578c5bc9d7ce1f1`, with source, license and notices preserved locally.

| Body stage | Implemented articulation | Intended role |
|---|---|---|
| Procedural preview | Free root + 20 ball joints; no actuators | Source visualization |
| Classical diagnostic body | Same 20 balls + 60 torque motors | Establish support, bounded tracking and backend feasibility |
| Report 08's 34-DOF proposal | Not implemented | An earlier possible simplification |
| Final SOMA23-derived body | **48 scalar hinges**, `nq=55`, `nv=54`, `nu=48` | Frozen body for the learned jab pilot |

The upstream SOMA asset itself needed an audit: broad three-axis forearms/shins, large ranges, passive spring/damping settings and motor gears were not accepted blindly. The derivative uses one-axis knees, separate elbow flexion and forearm rotation, useful wrist motion, fixed toe/head hierarchy links, explicit motor caps, **zero passive springs/dampers**, floor contacts and **13 selected self-collision pairs**.

{{image|outputs/jab-milestone/jab-final-state.png|The constrained SOMA23-derived body after the learned rollout returns to guard. Native height 1.7664 m; declared engineering mass 75 kg; plain blue simulator primitives.|Current training body}}

The user relaxed the exact 1.80 m requirement, so native bone lengths were retained. This is a **SOMA23-derived rigid-body skeleton**, not an imported SOMA-X skinned mesh, individualized scan or calibrated musculoskeletal model. Appearance and physical validity are separate decisions.

**Evidence:** [model-choice research](14-human-model-for-learned-punching.md), [plan reconciliation](15-task09-body-plan-reconciliation.md), [body audit](16-scalar-body.md), [body metadata](../outputs/scalar-body/scalar_model.json), [upstream notices](../assets/vendor/protomotions/NOTICE.md).
</section>

<section id="scalar-reference" markdown="1">
<div class="section-kicker">12 / Retarget again to the exact training body</div>
## Two subtle IK defects had to be fixed before learning

The new articulation invalidated the old qpos archive. The conditioned DuoBox jab was retargeted using both glove positions and orientations, stationary soles, torso/root priors, joint limits and selected self-collision penalties. The source motion runs at **0.6× speed** for this pilot.

| Reference version | Max sole drift | Max glove-target error | Max joint speed | Diagnosis and decision |
|---|---:|---:|---:|---|
| V1 | 49.599 mm | 8.344 mm | 14.323 rad/s | Unconstrained Jacobian updates followed by clipping jammed knees at limits; rejected |
| V2 | 0.254 mm | 0.251 mm | 4.554 rad/s | Bounded least squares fixed support geometry, but ~0.14 rad guard-boundary coordinate jumps remained |
| V3 | **0.250 mm** | **0.291 mm** | **3.043 rad/s** | Removing the initial seed-pose bias eliminated boundary spikes; accepted frozen reference |

The final archive has 300 frames over **4.9889 s**, including a 0.5 s initial guard, a 1 s explicit return blend and a 0.6 s final hold. Final guard equals initial guard. Velocities use MuJoCo tangent-coordinate differentiation, including the free-root quaternion, rather than componentwise quaternion subtraction.

{{video|outputs/jab-render-5895255/jab-rollout.mp4|outputs/jab-render-5895255/jab-closest.png|The accepted V3 reference|A geometrically feasible scalar-body jab with planted soles and explicit guard return. Its closest glove/head distance is 0.101116 m. This is still prescribed-pose replay.|Kinematic reference}}

Reference contact first occurs **1.5844 s after archive start**. The inserted 0.5 s guard provides a declared launch convention, not a fully reviewed outward-launch/TTC annotation. Tiny Cartesian errors alone would have missed the V2 coordinate discontinuity; auditing the final joint-space derivatives mattered.

**Evidence:** [V1 manifest](../outputs/scalar-reference/manifest.json), [V2 manifest](../outputs/scalar-reference-v2/manifest.json), [V3 manifest](../outputs/scalar-reference-v3/manifest.json), [milestone report](19-scalar-jab-milestone.md).
</section>

<section id="learning" markdown="1">
<div class="section-kicker">13 / Add learning where the baseline actually fails</div>
## A learned residual supplies the missing balance and tracking correction

On the scalar body, reference PD with support feedforward still drifted and fell. In the extended quiet-guard test, the 2 ms native variant fell around 3.736 s; the 5 ms variant oscillated rapidly and fell around 1.11 s. Optional classical root-error feedback through joint torques reduced 3 s quiet-guard drift from **180.69 mm to 7.41 mm**, but did not complete the longer punch-and-recovery protocol. Contact during collapse or late return did not count as a successful attack.

The learner uses a small Gaussian PPO actor/critic. It observes physical state, phase, reference errors, future targets, glove errors and previous action. It outputs **bounded joint-angle residuals** around the reference. PD plus offline support feedforward converts targets into capped motor torques. The pelvis is unactuated; estimated support forces are never applied externally.

The primary run uses actual **GPU physics and learning**, with 2 ms physics and 20 ms policy updates. It does not import a pretrained successful boxer. The learned component is residual feedback, not a raw-torque policy trained from scratch.

### The training history includes unsuccessful checkpoints

| Experiment | Finding |
|---|---|
| Job 5895225 | Eight-world synthetic standing / one-update PPO smoke passed; infrastructure only |
| Job 5895242 | Cancelled before learning when V2 guard discontinuities were discovered |
| Job 5895253 | V3 primary run: 128 worlds × 200 updates × 64 steps = **1,638,400 policy transitions** |
| Checkpoint 50, evaluation 5895265 | **All 16 worlds fell**, despite improving reward and some raw contact events |
| Final primary policy | Completed training after 444.3 s, excluding initialization, baseline and evaluation overhead |
| Optional balance-feedback comparison | Same transition budget, 256 worlds × 100 updates; also passed the final bank; primary selection unchanged |

{{image|outputs/jab-milestone/learning-curve.png|Training progress from the selected PPO run. Training can start mid-reference, so reward and episode counters do not measure continuous full-cycle success. The independent guard-start bank below is the acceptance evidence.|Actual training log}}

The primary checkpoint has the optional classical online balance feedback **off**. It retains PD and offline support feedforward; learned residuals provide the successful correction. Its matched zero-residual baseline uses the same remaining controller. Frozen model/reference hashes, effective gains, joint maps and observation/action schema are validated when loading a checkpoint.

**Evidence:** [learning guide](17-learned-jab-pilot.md), [checkpoint-50 failure](../outputs/jab-learning-5895265/metrics.json), [training log](../outputs/jab-milestone/training.jsonl), [frozen controller schema](../outputs/jab-milestone/controller_schema.json).
</section>

<section id="paired-result" markdown="1">
<div class="section-kicker">14 / Compare on identical starting states</div>
## The reference-PD controller falls; learned corrections complete the pilot

Compare the reference in section 12 with the two physical rollouts below. The reference shows the desired motion. The baseline shows what the unlearned controller actually does. The learned rollout shows the selected policy's actual physical motion. Each physical video is a nominal trial, not a montage representing every evaluation case.

<div class="comparison-controls"><button type="button" id="compare-play">Play both physical rollouts from start</button><button type="button" id="compare-pause" class="secondary">Pause comparison</button><span id="compare-status" role="status">Comparison playback is optional; each video also has its own controls.</span></div>
<div class="media-grid two" id="physical-comparison">
{{video|outputs/jab-render-5895275/jab-rollout.mp4|outputs/jab-render-5895275/jab-final-state.png|Reference-PD baseline: failed|Actual physical rollout: collapse and no recovered guard. The poster is correctly labeled final state, not final guard. Same bounded controller, learned residual disabled.|Failed · GPU physics}}
{{video|outputs/jab-milestone/jab.mp4|outputs/jab-milestone/jab-contact.png|Learned residual: completed|Actual physical jab and supported return. G1 is a fixed visual/analytic fixture, without impact response or protagonist policy.|Passed · GPU physics}}
</div>

### The paired bank and acceptance rules

The fixed bank contains one exact initial guard and 31 perturbations: scalar-joint noise σ=0.005 rad clipped to limits, and generalized-velocity noise σ=0.02 in the appropriate linear/angular units. Root position and orientation are not perturbed. Both controllers receive **the same initial arrays**, body, reference, gains and physical clocks.

Independent evaluation verifies hashes, trial IDs, exact starts, one reset and every recorded physics step. It recomputes FK from recorded states rather than trusting cached geometry. Warp's cached contact forces correspond to the preceding 2 ms integration interval, so supported-guard dwell has at most one-step boundary uncertainty.

| Pilot gate | Reference PD | Learned tracker |
|---|---:|---:|
| **Every gate passes** | **0 / 32** | **32 / 32** |
| No fall | 0 / 32 | 32 / 32 |
| Sole drift ≤3 cm | 0 / 32 | 32 / 32 |
| Joint limits, with declared 0.02 rad soft tolerance | 1 / 32 | 32 / 32 |
| Contact during declared attack window | 12 / 32 | 32 / 32 |
| Supported, quiet, safe guard recovered | 0 / 32 | 32 / 32 |
| Glove tracking ≤10 cm RMS pilot threshold | 0 / 32 | 32 / 32 |
| Bounded torque; no external/root actuation | 32 / 32 | 32 / 32 |

Guard acceptance requires both gloves within **3 cm / 15°** of the chest-relative initial guard, low relative velocity, quiet root/joints, both feet supporting and both gloves at least 0.16 m from the head for a **0.2 s dwell**. Contact alone is insufficient. The coarse attack window is 0.5–3.3889 s; it excludes the appended return blend but still includes some source recovery.

### What the 32 accepted learned trials measured

<div class="metric-grid"><div><strong>0.0980–0.12986 m</strong><span>closest attacking glove/head centers</span></div><div><strong>5.89 mm</strong><span>worst sole drift</span></div><div><strong>≥1.800 s</strong><span>recovered-guard dwell</span></div><div><strong>7.54 mm / 2.31°</strong><span>worst final relative glove error</span></div><div><strong>64.72 N·m</strong><span>peak actual torque; each cap respected</span></div><div><strong>Zero</strong><span>root actuation / external applied forces</span></div></div>

Maximum physical joint-limit excursion was 0.009714 rad, within the declared soft-constraint tolerance. The optional balance-feedback PPO comparison also passed 32/32 at the same transition budget. It is a secondary experiment, not a silent replacement of the selected primary policy or a multi-seed ablation.

**Statistical scope.** These are descriptive counts on one small fixed bank around one trained motion. They do not establish arbitrary punch, target, person, initial-state or training-seed robustness.

**Evidence:** [evaluation contract](18-jab-evaluation-contract.md), [matched learned results](../outputs/jab-milestone/learned_summary.json), [matched baseline](../outputs/jab-milestone/baseline_summary.json), [secondary comparison](../outputs/jab-milestone/secondary_balance_summary.json), [checkpoint bundle](../outputs/jab-milestone/README.md).
</section>

<section id="five-ms" markdown="1">
<div class="section-kicker">15 / Genuine 5 ms control and additional punches</div>
## Repair the actuator, then train the physical constraints

Changing the old timestep alone exposed a numerical instability. Externally computed velocity damping was invisible to the implicit integrator; a coupled leg/ankle mode had a local explicit-damping bound near **2.57 ms**. Native affine PD exposes that derivative to `implicitfast`, while retaining the same body, masses, inertias, passive parameters and original output-strength limits.

`drive = kp × target + kd × reference_velocity + joint_feedforward`

`actual_torque = clip(drive − kp × angle − kd × velocity, original strength limits)`

The drive is not itself applied torque. New records audit actual actuator/generalized forces, the simulator's own clock and actor-update indices. There are exactly four 5 ms steps per 20 ms learned residual; no hidden 2 ms integration, root wrench or mid-cycle reset is used.

{{image|outputs/control-5ms-5931597/comparison.png|Matched native diagnostics: native implicit damping removes the rapid 5 ms oscillation, although unlearned balance and punch recovery still fail later.|Controller diagnosis}}

### An export bug caught before acceptance

The first training jobs used correct native GPU actuator arrays, but `mj_saveLastXML` omitted changed enum/limit flags when saving the runtime model. Recompilation would have produced a different controller. Independent actual-force audits established the true training force law; a replacement exporter preserves exact source-body properties, explicitly serializes the actuator/solver settings, recompiles and checks equality. Accepted evaluations use the corrected export and matching hashes. The erroneous historical files remain diagnostic evidence.

### Accuracy, joint stops and guard recovery are separate objectives

| Jab stage | Independent result on 32-trial validation bank | Consequence |
|---|---|---|
| Full-speed checkpoint 100 | 32 hit and recover; 18 pass every gate; contact 34–44 ms late | Genuine 5 ms control established, but not yet an accepted complete policy |
| Full-speed checkpoint 400 | Attack RMS 5.96–7.29 mm; timing within one physics step; all fail wrist-limit tolerance | Accurate tracking alone does not imply constraint compliance |
| Maximum-substep limit penalty | All satisfy limits and precision; first candidate passes 28/32 complete cycles | Four final-guard dwells are too short; later candidates drift in Chest-relative guard pose |
| Explicit Chest-relative guard reward | **32/32 pass all 18 physical gates**; attack RMS **5.20–6.23 mm**; timing error **≤3.82 ms**; guard dwell **≥1.83 s** | Freeze the checkpoint, then evaluate a fresh 64-trial test bank |

The joint-limit penalty observes every 5 ms state, including excursions between actor updates. The guard reward matches the existing Chest-relative acceptance geometry. No joint limit, 3 cm guard tolerance, 0.2 s dwell, contact threshold or reference clock was relaxed. Sub-step interpolated contact errors are reported with approximately one physics-tick resolution.

The selected jab then passes **64/64 fresh test trials**, versus **0/64** for its matched zero-residual baseline. The bank contains the shared nominal anchor plus 63 newly generated small perturbations, created after checkpoint selection. Attack RMS is **4.98–7.32 mm**, contact error **≤4.44 ms**, and supported recovered-guard dwell **≥1.835 s**. The policy and baseline use identical controller schemas; only the learned residual is disabled in the baseline.

{{video|outputs/jab-render-5931851/jab-rollout.mp4|outputs/jab-render-5931851/jab-closest.png|Improved jab: full source speed, genuine 5 ms physics|Actual recorded fresh-test GPU physics, with fixed G1 visualization. The actor runs every 20 ms; the physical simulation advances every 5 ms. This is the selected accepted jab, not reference-pose playback.|Accepted · learned}}

### Why the hook references needed more than IK

All four initial scalar references were geometrically valid, but inverse-dynamics diagnostics found that exact original hook tracking could require more than twice some shoulder strength limits. Their static center-of-mass margins were healthy; fast coupled motion generated the load. Shoulder-posture adjustments alone did not remove it.

{{image|outputs/reference-dynamics-5931727/hook-demand.png|Ideal exact-reference motor demand under a simplified support allocation. Full-speed hooks exceed configured effort; slower playback reduces demand. This guides conditioning and is not an impossibility proof for nearby motions.|Dynamic reference diagnostic}}

The chosen response keeps the body strength fixed: first learn a slower curriculum, then retime the original joint path nonuniformly so demanding portions have more time. The conditioned hooks approximately retain contact and total-cycle timing, but local source rates vary. They are **explicitly retimed motions**, not unchanged uniform 1.0× capture playback. Right-hook ideal demand falls below every cap; the left-hook peak falls to about 1.045 times a cap, leaving a small residual adjustment for physical control.

Cross and both hooks have separate residual PPO networks. Active-hand attack shaping repairs the earlier reward's tendency to hide a missed punch behind a long, well-tracked guard. Conservative PPO updates, joint-stop costs and Chest-relative recovery are then checked against complete physical rollouts. The cross initially failed return-to-guard because its neck remained too far from the initial joint posture. A worst-joint guard reward repairs that failure without changing acceptance tolerances. The same refinement gives the right hook a much longer recovered guard. Extra left-hook refinement reduces joint-stop error but worsens foot drift; the earlier checkpoint is therefore selected on independent validation, rather than assuming the newest weights are best.

### Four selected policies pass fresh physical testing

Each checkpoint is selected on the existing 32-trial bank before generating its fresh 64-trial bank. Every fresh bank contains one shared nominal anchor and 63 new small joint-state perturbations, with the initial root pose fixed. Every learned trial passes all 18 physical gates; all three tighter precision targets also pass. The identical native controller with the residual disabled fails every complete cycle. This establishes the contribution of these learned residuals, not the impossibility of a better classical controller.

| Family | Learned passes | Zero-residual passes | Attack-glove RMS (mm) | Maximum contact-time error (ms) | Minimum guard dwell (s) | Reference timing |
|---|---:|---:|---:|---:|---:|---|
| Jab | 64/64 | 0/64 | 4.98–7.32 | 4.43 | 1.835 | Uniform 1.0× |
| Cross | 64/64 | 0/64 | 6.92–9.06 | 9.87 | 1.735 | Uniform 1.0× |
| Left Hook | 64/64 | 0/64 | 8.53–13.01 | 8.38 | 1.735 | Nonuniform retiming |
| Right Hook | 64/64 | 0/64 | 6.00–10.08 | 0.93 | 1.755 | Nonuniform retiming |

All trials use the original strength limits and no root assistance or mid-cycle reset. Worst left-hook sole drift is 27.20 mm against the unchanged 30 mm allowance, making foot position its tightest observed margin. Timing decimals estimate threshold crossings between recorded 5 ms samples; they do not add sub-step measurements. Test success is scoped to these four fixed references and small perturbations.

<div class="media-grid" markdown="1">
{{video|outputs/jab-render-5931928/cross-rollout.mp4|outputs/jab-render-5931928/cross-closest.png|Learned cross and supported recovery|Selected cross policy in recorded GPU dynamics. Uniform source timing, actual 5 ms integration and 20 ms learned actions. G1 is a fixed visualization; contact is the analytical 0.14 m center-distance event.|Accepted · learned}}
{{video|outputs/jab-render-5931955/left_hook-rollout.mp4|outputs/jab-render-5931955/left_hook-closest.png|Learned left hook and supported recovery|Selected left-hook policy on the explicitly retimed reference. The body, motor limits and physical acceptance gates are unchanged. This is recorded dynamics, not target-pose playback.|Accepted · retimed reference}}
{{video|outputs/jab-render-5931931/right_hook-rollout.mp4|outputs/jab-render-5931931/right_hook-closest.png|Learned right hook and supported recovery|Selected guard-refined right-hook policy, using a declared nonuniform reference clock. Actual 5 ms integration and 20 ms learned actions; no impacts or dynamic G1 response are simulated.|Accepted · retimed reference}}
</div>

The [frozen four-family release](../outputs/four-punch-milestone/README.md) includes all learned/baseline trial records, videos, checkpoints, references and source snapshots. The [manifest](../outputs/four-punch-milestone/manifest.json) records their byte hashes. Both gloves must regain safe clearance during supported recovery, but the current release still executes one family from its own initial guard at a time.

**Evidence:** [controller investigation](20-5ms-controller-investigation.md), [four reference construction](21-four-punch-references.md), [learning experiment log](22-five-ms-learning-progress.md), [independent validation log](23-five-ms-validation-log.md), [controller interface and G1 gaps](24-five-ms-controller-interface.md), [dynamic feasibility and retiming](25-reference-dynamics.md).
</section>

<section id="limits" markdown="1">
<div class="section-kicker">16 / What remains open</div>
## Four isolated punches are accepted; the two-player game remains separate

| Requirement or gap | Current evidence | Next acceptance work |
|---|---|---|
| Fast, accurate tracking | All four meet the 2 cm full-cycle and attack-only targets | Broader recordings, states and targets; hooks currently use conditioned timing |
| Timing / TTC | All four meet the 20 ms contact-error target; reference-specific timing | Arbitrary requested TTC, source-reviewed outward-launch annotations and timing commands remain separate |
| Four learned families | Four specialist networks, each accepted on 64 fresh trials | Family-conditioned tracking or validated switching between distinct guards |
| Sequences | Two classical jabs on old body; isolated learned cycles for four families on new body | Learned repeated and mixed sequences with measured recovery, no hidden resets |
| Body validity | Anatomical constraints, 75 kg scenario, bounded motors and selected collisions | Sensitivity to mass/strength, broader collision audits; no biological calibration claimed |
| Shared physics clock | Human now uses actual **5 ms physics / 20 ms actions** | G1 and human still use different global cone/solver settings; qualify a shared configuration |
| Perception | Raw production-geometry camera output works; severe near-field occlusion observed | Detector/depth/history/encoder checks with actual human geometry and both red gloves |
| G1 integration | Standing visual/analytic fixture; accepted learned bundles/reset artifacts absent | Restore accepted assets and implement independent entity state/reset/action maps |
| Adversarial game | Reference-bank commitment interface and design analysis | New plan/phase/sequence-aware replay, critic and legal decision boundaries |
| Generalization | One training seed/body/source per family and fixed target; 64 small test perturbations each | Recording-level splits, broader targets/states and independent seeds |

### What the original plan got right—and what could wait

**Keep:** full cycles and both wrists; bounded effort and unassisted balance; frozen launch targets; no teleporting or extra attacks; explicit invalid outcomes; measured recovery; a separate low-level tracker and high-level legal selector.

**Simplify or defer:** mandatory SMPL-X/WBC/Mink stacks, a predetermined hinge count, photorealistic skin/face/fingers, multiple body sizes, free footwork, all transitions before one reliable punch, continuous plan deformation, joint-level adversarial fine-tuning, human–G1 impact dynamics and distributed training on all eight GPUs. The implemented bounded least-squares retargeter, constrained SOMA fallback and small PPO pilot were sufficient to expose the immediate engineering risks.

### The integration gaps are semantic, not just model loading

The existing G1 plant assumes whole-model dimensions **36/35/29**. A second dynamic entity needs explicit G1/human coordinate maps, reset ownership, energy/action accounting and independent histories. Existing production boxing learning enforces null disturbance; a punch-plan ID cannot simply occupy the old 29 torque-disturbance channels. Replay and critic state must represent commitment, phase, timing, sequence, validity and the true pre-reset successor.

Commitment must hold in **Bellman targets as well as rollout actions**. Between legal boundaries the adversary's available action is the already committed plan. A selector also needs multiple valid choices; one reference per fixed family/TTC context can leave no actual adversarial decision.

The actor's 948-feature interface might be retained, but new occlusion and appearance still shift its input distribution. Continuous noisy neural simulation does not automatically inherit the abstract finite deterministic game's guarantee. The appropriate claim is robustness to a declared, validated opponent class.

### Recommended next phases

1. **Continuous learned cycles:** repeated jabs first, then mixed families; bridge their distinct guards, wait for measured recovery and prohibit hidden resets. Establish a feasible rate envelope before accepting rate commands.
2. **Broader validated commands:** add targets, initial states and timing variation with recording-level splits and independent training seeds; consider one family-conditioned tracker only after the specialist baselines are stable.
3. **Shared G1 simulation:** retain the accepted G1 solver/cone configuration, qualify the human under it, add independent entity state/reset/action maps, and restore the G1 assets and camera pipeline. The 5 ms clock is already matched.
4. **Legal adversarial selection:** freeze the low-level tracker, provide multiple admitted choices, enforce boundary commitment in replay and Bellman updates, and compare learned selection with matched random plans.

**Evidence:** [original phased assessment](06-task09-revised-phased-roadmap.md), [body-plan reconciliation](15-task09-body-plan-reconciliation.md), [latest milestone and limitations](22-five-ms-learning-progress.md).
</section>

<section id="experiments" markdown="1">
<div class="section-kicker">17 / Searchable experiment record</div>
## Passed, failed and diagnostic runs

This ledger covers the substantive experiments documented in the preserved reports, including rejected references, failed controllers and successful milestones. Routine repeated rendering attempts are retained in `outputs/`; they are not all separate scientific experiments. Outcome labels apply to the question in that row, not to completion of Task 09.

{{ledger}}
</section>

<section id="reproduce" markdown="1">
<div class="section-kicker">18 / Reproduce and inspect</div>
## Frozen evidence, validation and source history

The selected [four-family bundle](../outputs/four-punch-milestone/README.md) contains all four checkpoints, frozen body/runtime models, references, controller schemas, training source snapshots, logs, plots, matched evaluation summaries, all trial rollouts and videos. Its [manifest](../outputs/four-punch-milestone/manifest.json) records SHA256 hashes and source jobs. The [older jab bundle](../outputs/jab-milestone/README.md) remains available as historical evidence.

At this milestone, **133 project tests and four subtests passed**. The Torch-dependent module was skipped in the preview environment; its **nine learning tests passed separately** with the pinned physics environment. Ruff and shell syntax checks passed. The packaging guide records its additional schema checks and real release preflight. Final learned and paired-baseline videos are fully decoded; contact, recovered-guard and baseline-fall stills are visually reviewed. Media validation records accompany the bundle.

The [bundle README](../outputs/four-punch-milestone/README.md) provides exact evaluation commands. New experiments use the current server's tested V100 launcher:

```bash
sbatch scripts/train_punch_5ms_volta.sbatch \
  --reference outputs/scalar-four-reference-v1/jab/jab.npz \
  --num-envs 128 --updates 200 --save-every 25
```

This command starts a new experiment, not a reproduction of the accepted checkpoint's full training lineage. Frozen training arguments and initialization records preserve that lineage.

Independent audit commands are in the bundle README. The [GPU guide](05-gpu-server-guide.md) covers allocation, environment and EGL mapping. Do not run the checkpoint with different gains, body semantics or clocks and call it the same evaluated controller.

The compact HTML uses relative workspace links. A separate **standalone HTML** embeds the selected images and videos for offline viewing; raw artifact and source-code links still require the project workspace. Both editions contain the full text of the 26 historical Markdown reports below. No external fonts, analytics or CDN assets are needed.

{{sources}}
</section>
