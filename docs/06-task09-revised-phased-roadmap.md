# Task 09: limitations, revised phases, and implementation gaps

Assessment date: 10 October 2026. This is a proposed roadmap, not a claim that the later phases have been implemented or validated.

This report preserves the early preview-stage assessment. For current physical results see [report 13](13-physical-pilot-and-next-gates.md); for the original SMPLSim/SOMA body plan, implemented prototype, and revised training milestones see [report 15](15-task09-body-plan-reconciliation.md). Simple blue appearance does not determine the choice of physical rig.

All implementation belongs in `human-punching`. `robust_sav`, `mjlab`, and DuoBox remain read-only inputs. In particular, the Task 09 PDF's suggested `src/boxing/human/` additions should initially become modules under this project's `src/human_punching/`; changing the accepted `robust_sav` requires a separate explicit instruction.

## 1. Recommendation

**The plan is feasible as a staged research project, but it is considerably larger than “plug a human policy into robust_sav.”** Its separation into reference construction, a low-level human tracker, and a constrained adversarial planner is worth keeping. Its named software stack and all its proposed numerical thresholds should not be treated as prerequisites or established results.

My recommended first physical milestone is **one 1.80 m blue mannequin executing one head-targeted jab, at one feasible TTC, returning to guard under bounded actuation in the actual mjlab environment**. Before expensive tracker training, run a cheap combined G1/human scene test to expose camera, entity indexing, runtime, and GPU compatibility problems. Then expand to four families, short sequences, and adversarial selection.

There is also a useful intermediate result: a planner can choose between **committed kinematic human reference motions** and form a two-player game without learning a torque-level human controller. That measures G1 robustness to the declared motion bank. It does not demonstrate human balance, effort feasibility, or physical impact response. Keep this as a debugging baseline and possible narrower research milestone; retain the physical tracker for the full Task 09 objective. A controller is necessary for physical execution, but it need not be learned if a bounded feedback controller actually passes the same tests.

The most important change to the PDF is **ordering**: discover shared-scene and runtime problems early, rather than after imitation training. This follows Vision5's Task 07b matched-state checks and Task 08 P1 interface/pilot work before P3 training. Optional extensions should have their own phase, as the optional ZED SDK work does in Task 08.

## 2. What we have actually established

| Area | Verified state | What it does not establish |
|---|---|---|
| Human visualization | Featureless light-blue 1.800000 m mannequin; four DuoBox punches; full hand/glove rotation checked across 855 frames. | An anatomical or dynamically validated human model. |
| Motion transfer | Full 21-joint source hierarchy; original 120 Hz convention; maximum FK error about 3.61e-7 m. | Constrained retargeting to different proportions, foot support, or achievable motor commands. The rig preserves source proportions. |
| Human dynamics | Free root plus 20 ball joints; 87 qpos and 66 qvel coordinates. | There are **no actuators**, gravity/contact participation are disabled, and inertias are placeholders. |
| Data | Four hash-verified candidate crops, one per family, all using actor `ma`, with recorded context padding. | Reviewed guard-to-guard labels, independent source splits, broad style coverage, or generalization. |
| Rendering/compute | Native MuJoCo 3.11.0 CPU replay and NVIDIA EGL replay; one full GPU job and a four-task array passed. | mjlab/MuJoCo-Warp physics, its camera pipeline, a tracker, or SAV training on this server. |
| G1 integration | Current source, dependency records, and contracts inspected. | An executed combined scene or restored accepted G1 policy. The trained bundles/reset artifacts are not supplied in this checkout. |

Evidence: [motion audit](02-duobox-motion-audit.md), [implementation/results](03-implementation-and-results.md), [glove audit](04-glove-rotation-audit.md), and [GPU guide](05-gpu-server-guide.md). The visual preview portion is complete; the physical asset portion of the PDF's milestone 09A is not.

This review uses `robust_sav` commit `f4672e3ab403308520beb6c6476a4027096764bf`. Its accepted mjlab source is `8ee51fbcf806a7419189f706d9e394cbeb7790fa`, whereas the sibling working-tree HEAD is `033ae22a2c7a30a25a6fa77b16c113ed88dd1b55`. Vision5's final Task 08 closure includes the verified warm continuations; this assessment does not reopen that accepted task. The accepted simulation results still do not constitute hardware validation or a formal safety certificate.

## 3. Main limitations of the Task 09 plan

### 3.1 The physics and tracking problem is the largest research uncertainty

A mesh exporter does not provide a balanced boxer. A person punching toward the shorter G1 must coordinate pelvis, knees, torso, feet, and both arms. Lowering a recorded wrist or speeding up a clip can produce unreachable, slipping, or torque-infeasible motion. The current constant whole-clip floor offset only makes a replay clear the floor; it does not solve contact constraints.

The PDF correctly mentions masses, limits, PD gains, and unassisted tracking, but these require substantial new work. Begin with a free, unactuated root, suitable floor contacts, plausible inertias and bounded joint actuation. A fixed pelvis or external root wrench may help debugging, but cannot count as an accepted freely standing physical boxer. Joint limits and torque bounds define a simulation surrogate, not validated human biomechanics by themselves.

### 3.2 Target adaptation and TTC are coupled to physical feasibility

The proposed 0.50/0.65/0.80 s TTCs are pilot candidates. We have not established that all four families can hit all proposed head locations at these times and then recover. Time compression increases required velocity and acceleration; broad independent sampling of target and TTC will include infeasible combinations.

Build a small **admissible bank conditioned on starting guard, placement, family, and TTC**. Reject unavailable combinations explicitly. At launch freeze the world target and outward plan; the low-level controller may correct its own tracking error, but must not chase a subsequently moving head. Check reference contact time, actual glove contact with the frozen diagnostic head, and contact with the moving G1 as three distinct measurements. The PDF's §4 distinction should be retained.

### 3.3 The initial data is much smaller than the proposed training/evaluation program

The four current events are enough to debug a retargeter and overfit a tracking pilot. They do not support claims about arbitrary punches, people, guards, or style. Their inherited labels still require review. Return to a plausible final frame is not proof of measured physical guard recovery.

Inspect additional full recordings already available in DuoBox, annotate both hands and feet, and split by recording before cropping, retiming, or augmentation. Use person-disjoint tests only where the data permits. Repeating a clip with many simulator seeds provides execution trials, not new held-out source motion.

### 3.4 The proposed first planner can accidentally have no choices

PDF §8 lets the curriculum specify family, TTC, and count. Our current bank has one example per family. With those variables fixed, a selector over that bank is effectively deterministic.

Before calling an experiment adversarial selection, provide multiple legal within-family variants for at least some supported contexts: different reviewed source cycles or validated stance/trajectory variants. Alternatively, explicitly grant the adversary family or TTC choice and document the changed game. Do not silently turn a user-prescribed sequence into a freely chosen one. Record legal-set size and coverage; singleton contexts carry no adversarial choice.

### 3.5 Existing application code is stricter than the plan's plug-in language suggests

The PDF audited an older `robust_sav` revision and cites `vision/plant.py`; the current production path is `perception/`. There are hard model-size, source-provenance, runtime-location, appearance, and checkpoint checks. These are useful preservation rules for Task 08, but a new combined scene cannot satisfy all of them unchanged.

Moreover, the current production boxing learner rejects enabled disturbances and hardcodes zero disturbance in updates. Its replay stores no adversary action. The generic `rsav` library has adversarial-learning components, but those are not already wired into the accepted captured GPU boxing path. Integration therefore includes a new runtime/scene adapter and a versioned learning path, not just a new policy class.

### 3.6 A discrete, occasional decision is different from per-tick torque disturbance

A punch-plan ID is not a 29-dimensional G1 torque vector. The generic continuous adversary gradient cannot optimize a discrete `argmin` unchanged. For a small bank, evaluate the learned Q function for every legal candidate at a boundary and select its minimizer; a separate planner network is optional.

Commitment must apply to **Bellman targets as well as rollout actions**. Between boundaries the admissible planner action is the already committed plan. Minimizing over fresh punches at every target step would train against a different opponent, even if the rendered rollout looks committed. Section 7 below specifies the missing contract.

### 3.7 A human changes perception even if the actor still has 948 inputs

Body occlusion, guard height, glove shape, and placement alter the input distribution. Task 08's active RGB pipeline expects **both gloves red**; the preview's blue-left/red-right colors are review aids. The inherited appearance function also recolors all geometry before the active renderer restores only G1 materials, so the blue human needs an explicit appearance rule.

Preserve the accepted camera mount, raster, clocks, delay, and detector first. Test actual initial-guard visibility and subsequent occlusion in the production camera path. A 1.80 m person's gloves may be hard to see with the downward-facing torso camera. Search admissible placements/guards before changing camera pitch. If a camera change is required, version it and reevaluate the protagonist; it is not a free compatibility fix.

### 3.8 The theory permits a human adversary, but does not automatically certify this simulator

Vision5's abstract adversary in `x_next = F(x, u, d)` need not be joint torque. A legal committed human plan can instantiate `d`. The state must include the human and its controller, current plan/phase, remaining prescribed sequence, timing, and relevant history.

However, the exact theory assumes a finite stationary deterministic game. Continuous dynamics, noisy observations, neural approximation, and a restricted motion bank do not inherit that guarantee. The accepted critic is explicitly partially observed because delay/queue/fault state is absent. Adding human state alone does not repair that. Either expose sufficient privileged state to a new critic or clearly retain an approximation; a recurrent model can help estimation but is not itself proof of full observability. Report robustness to the declared valid opponent class.

## 4. What to keep, simplify, or defer

| Plan item | Recommendation | Reason |
|---|---|---|
| Full-cycle motion, both wrists, rigid glove attachment | **Keep.** | These define the actual attack and recovery. The current glove orientation audit is useful evidence to retain. |
| Immutable launch plan, simulator-owned legality/count, no teleporting, invalid-attack accounting | **Keep.** | Removing these lets the adversary exploit implementation loopholes. |
| SMPL-X/SMPLSim as mandatory first dependencies | **Make optional.** | The approved stylized surface is sufficient. Use another body/exporter only if it solves a demonstrated physical-model problem; restricted model assets otherwise add work. |
| Exactly 63 scalar hinges | **Do not prescribe the count yet.** | Choose controllable anatomical DOFs and explicit axes/limits. Preserve useful torso/shoulder/forearm/wrist motion; neither the current unconstrained ball rig nor a proposed hinge count is automatically suitable. |
| WBC-mjlab as the default | **Try pinned native mjlab tracking first.** | It already has tracking rewards, reference machinery, and a PPO runner. WBC remains an option if a concrete capability offsets its dependency/API adaptation cost. |
| Mink, GMR, PyRoki, SOMA23 | **Treat as alternatives, not a required stack.** | Use one offline solver when constrained retargeting needs it. A pretrained policy from another body/framework is not a verified warm start. |
| Photorealism, seamless skin, face/finger/toe detail | **Omit.** | Unnecessary for this objective and inconsistent with the requested simple appearance. Simple foot contact geometry still matters physically. |
| Multiple heights/shapes and unrestricted footwork | **Defer.** | Each increases reach, dynamics, visibility, and validation coverage. Start with one body and restricted placement. |
| All 16 ordered pairs and arbitrary lists before any useful game | **Stage.** | Start with one punch, then one repeat and one mixed pair. All 16 pairs remain necessary before claiming complete four-family transition support. |
| Continuous plan deformation and joint-level adversarial fine-tuning | **Defer.** | They can break timing, balance, style, and commitment. A frozen tracker plus finite bank is interpretable first. |
| Human–G1 impact dynamics | **Separate later experiment.** | A floor-contacting physical human can threaten G1 through the accepted analytical hazards without introducing collision impulses or body-charge attacks. |
| Three training seeds and a large release matrix during every pilot | **Defer to release comparisons.** | Debug with bounded experiments; preserve independent seeds and uncertainty reporting for substantive claims. |
| CUDA graph optimization, distributed RL, using all eight GPUs | **Defer.** | Establish correct updates and one-GPU throughput first. Independent jobs can use arrays without distributed learning. |

The PDF itself already labels several of these as optional/later. They are not defects in that document; the revision makes those boundaries operational and removes them from the first milestone's critical path.

## 5. Concrete gaps against the current software

Paths below refer to the inspected source snapshot, not proposed implementation locations.

| Component | Current fact / source evidence | Required work in human-punching |
|---|---|---|
| Dependency identity | Both mjlab commits call themselves 1.6.0, but sibling HEAD requires Torch >=2.14/RSL-RL 5.5.1; Task 08 pins Torch 2.8/RSL-RL 5.5.0. [Audit](01-compatibility-assessment.md) | Own a separately locked environment using the exact accepted source revision. Do not install sibling HEAD by default. |
| Optional WBC | The previously inspected `689f0836` declaration requires NumPy <2.5; Task 08's inventory is 2.5.3. WBC APIs against the accepted pin are untested. [Audit and upstream links](01-compatibility-assessment.md#wbc-is-an-optional-later-dependency-with-real-reconciliation-work) | Resolve/test separately or avoid it. A separate training environment still needs a verified checkpoint/inference handoff. |
| Human asset | [human.py](../src/human_punching/human.py) uses ball joints, placeholder inertia, disabled gravity/contact, and no actuator block. | New physical model/configuration, supported DOF map, bounded action law, contact proxies, and a controller. Reuse the simple visible surface where practical. |
| Reference schema | [retarget.py](../src/human_punching/retarget.py) exports `qpos/qvel` and named body arrays. Native tracking expects `joint_pos/joint_vel` and positional body indices. | Export from the final model with named mappings, correct root separation, rotation conventions and clip-local derivatives. |
| Reference clocks/transitions | At the accepted mjlab pin, `src/mjlab/tasks/tracking/mdp/commands.py` advances one stored frame per environment step and writes state on clip resampling. | Custom continuous-time command; direct 120 Hz playback at 50 Hz would be 2.4 times too slow. Restrict state writes to declared reset/RSI, never between executed punches. |
| Runtime ownership | [standing/runtime.py](../../robust_sav/src/boxing/standing/runtime.py) requires Python/mjlab under the robust_sav root; [stance/plant.py](../../robust_sav/src/boxing/stance/plant.py) additionally expects its cache-owned interpreter. | New project-owned runtime/provenance checks. Matching package versions alone will not let an unchanged old factory run from this environment. |
| G1 plant dimensions | [stance/plant.py](../../robust_sav/src/boxing/stance/plant.py), lines 228–229, rejects whole-model dimensions other than 36/35/29. | New adapter with explicit G1 and human qpos/qvel/actuator maps and separate histories. |
| Scene audit | [perception/scene.py](../../robust_sav/src/boxing/perception/scene.py), lines 110–129, permits only two extra massless glove bodies/geoms and unchanged dynamic dimensions. | New combined-model audit preserving G1 entity parameters/invariants; do not bypass the old audit and claim it passed. |
| State/reset/energy | [training/plant.py](../../robust_sav/src/boxing/training/plant.py) operates on whole-model state in G1 energy/reset paths. | G1-only gather/scatter and partial-reset tests; human resets must not overwrite G1 and vice versa. |
| Glove poses/appearance | [renderer.py](../../robust_sav/src/boxing/perception/renderer.py) restores G1 materials and makes both fists red; [depth.py](../../robust_sav/src/boxing/perception/depth.py) uses a red-only foreground. Scheduled fists are currently separate mocap objects. | Actual hand-attached transforms; explicit blue-body materials; sensor-compatible glove appearance and geom-ID mapping. Preserve analytical proxy definitions. |
| Production learner | [training/learner.py](../../robust_sav/src/boxing/training/learner.py), lines 30–75 and 152–153, uses/requires null disturbance. | New game-aware actor/critic updates. Begin with readable uncaptured tensor code; optimize only after semantic tests. |
| Production replay | [training/replay.py](../../robust_sav/src/boxing/training/replay.py), lines 15–25, has `u_applied` but no human action/commitment fields. | New replay schema with plan/decision state, successor legal choices, sequence state and true pre-reset successors. |
| Termination | [training/collector.py](../../robust_sav/src/boxing/training/collector.py) combines safety, punch return, and stance energy. | Sequence-gated arrival, deadlines, first-event preservation, and independent human-invalid outcomes. |
| Critic/checkpoints | [delayed_acquisition.py](../../robust_sav/src/boxing/perception/delayed_acquisition.py) declares partial observability; [runtime_contract.py](../../robust_sav/src/boxing/perception/runtime_contract.py) and [residual_checkpoint.py](../../robust_sav/src/boxing/training/residual_checkpoint.py) validate semantics as well as dimensions. | Version Task 09 schemas; explicit protagonist weights/normalizer migration; new critics rather than pretending the old Q schema fits. |
| Accepted G1 artifacts | Source is present; trained nominal/residual bundles and reset/stance artifacts are absent here. | Restore the exact accepted assets into project-owned locations before policy comparisons. Source-only work and isolated-human pilots can proceed before that. |

## 6. Revised phases

These phases separate interface checks, controlled pilots, training, and optional extensions. Each phase should produce its own short report, configuration/version manifest, and measurable exit evidence. The current preview remains reproducible throughout.

### P0 — Visual reference baseline: completed within its stated scope

**Objective:** inspect four full-body motions on the approved 1.80 m mannequin.

**Delivered:** motion loader, source hashes, articulated meshes, trajectories, videos, glove-rotation audit, and tested native GPU rendering.

**Exit evidence:** correct height, FK, source timing and hand attachment. This closes the preview, not dynamic feasibility, reviewed full-cycle labels, or a playable adversary.

### P1 — Runtime and shared-scene feasibility

**Objective:** establish that the chosen stack and combined scene can run before investing in tracking RL.

1. Create a separate training environment inside this project using the accepted mjlab commit and compatible CUDA 12 builds.
2. On one allocated GPU, exercise Torch forward/backward, Warp kernels, contact stepping, and the actual RGB/depth/ID camera path. Measure memory and throughput at a small batch before scaling.
3. Build an isolated combined-scene harness with explicit entity maps, using a scripted human for this test. Check G1-only state/energy/reset/action handling and version the new scene/runtime audit. Indexing tests can use fixtures; numerical agreement with accepted stance energy requires the archived stance chart/P artifact and remains pending until that asset is restored.
4. Render both gloves and the blue body through the Task 08 camera contract; verify occlusion and detector behavior. Restore G1 model assets if needed for compilation; trained policies are not required for matched-pose checks.

**Deliverables:** candidate dependency lock, GPU evidence, two-entity scene/pose renders, entity mapping, and measured resource baseline.

**Gate:** real Warp stepping and production-style camera output succeed; human state cannot contaminate G1 indexing; missing accepted assets are explicitly identified. A native EGL video alone does not pass. **Risk:** moderate engineering, with a hard hardware/runtime dependency.

### P2 — Final human model and committed target references

**Objective:** define a physically meaningful body and a small reachable head-targeted motion bank.

1. Keep one 1.80 m stylized body. Fix joint axes/order, limits, mass/inertia, floor contacts, actuator bounds and the hand-to-glove transform. Validate basic standing/actuator response with gravity enabled.
2. Review full-cycle source annotations and establish recording-level splits. Start with a single jab to debug the mapping, then cover the four families without claiming generalization from four crops.
3. Retarget to the final model. Use offline constrained IK only where necessary; preserve support feet and whole-body style. Select one feasible head placement and source-like TTC before enlarging the bank.
4. Implement immutable `plan_punch` and continuous-time `sample_ref`, contact-time measurement, phase labels and explicit infeasibility reasons. Exercise them in the kinematic combined-scene harness.

**Deliverables:** physical MJCF/configuration, versioned reference bank, target/TTC feasibility table, and diagnostic videos.

**Gate:** correct joint/body mapping, compatible starts, no hidden root resets, complete recovery reference, coherent derivatives, correct frozen-head contact, and identical reference arrays in paired post-launch G1-motion tests. These are geometric/reference gates; physical tracking still belongs to P3. **Risk:** moderate to high, chiefly target/support feasibility.

### P3 — One physical punch, then four-family tracking

**Objective:** demonstrate a freely simulated human executing and recovering under bounded feedback.

1. Benchmark a bounded PD tracking baseline. If it cannot balance/track reliably, build a motion-conditioned residual controller using the pinned native mjlab tracking facilities; use WBC only after demonstrating a concrete advantage.
2. Replace the default motion command with a time-aware, fixed-world command. Match training and integration observation order, reference lookahead, normalization, action scales, PD law, and clocks.
3. Train one jab at the admitted target/TTC. Use reference-state initialization for learning, but evaluate from actual guard starts. Expand to all four families and the small feasible target/TTC bank only after the jab works.
4. Evaluate with no external root assistance and no state overwrites during execution. Freeze accepted controller weights, normalizers, actuation settings and schema.

**Deliverables:** standalone controller/checkpoint, inference adapter, full configuration bundle, and a physical execution report.

**Gate:** measured guard-to-guard completion, balance, foot slip, joint/effort limits, glove pose and physical TTC errors meet thresholds fixed after pilots and before held-out tests. Count misses/falls/timeouts. A visually plausible animation does not pass. **Risk:** highest research uncertainty and likely the largest training cost.

### P4 — Sequences and a fixed-opponent G1 baseline

**Objective:** run valid physical human attacks continuously against G1 before optimizing adversarial selection.

1. Integrate the frozen tracker into the P1 adapter. Keep human–floor dynamics; initially disable human–G1 collision impulses while using the accepted two-glove analytical hazards. Explicitly forbid unintended body-charge/interpenetration behavior through placement/legality checks.
2. Implement physical guard completion, deadlines and sequence scheduling. Start with N=1, then a compatible repeated pair and a mixed pair. Reject unsupported transitions; do not reset into the next reference pose.
3. Restore the accepted nominal/residual G1 bundles and required snapshot/stance artifacts. Evaluate a frozen G1 against fixed or random valid human plans; then fine-tune the residual if needed, preserving its actor interface and frozen nominal controller.
4. Preserve capture/delay/history semantics and physical first-event timing. Require final sequence completion for arrival. Record human validity separately, including completion rollouts after early G1 failures where needed.

**Deliverables:** integrated fixed-opponent environment, restricted sequence API, matched G1 baseline, and validity/visibility/termination evidence.

**Gate:** no extra attacks, no teleporting, no homing, correct final-sequence arrival, and all attempts accounted for. Expand to all 16 ordered pairs and balanced N=1,2,3 before advertising the PDF's full sequence support. Pairwise success alone does not prove arbitrary long-list success. **Risk:** high integration effort; requires the absent G1 artifacts.

### P5 — Constrained adversarial plan selection

**Objective:** demonstrate whether choosing difficult legal punches improves adversarial training over the matched random baseline.

1. Populate multiple valid alternatives within the allowed context. Keep family/TTC/count curriculum-controlled initially, as in the PDF, and report coverage where alternatives are unavailable.
2. Implement a finite selector over legal plans at guard boundaries. Freeze the tracker; define which information the planner sees and the order of G1/planner decisions at launch.
3. Introduce the new critic, replay and update schema described below. Test held-plan targets against a small enumerated example before long GPU training.
4. Alternate against frozen opponent snapshots. Compare uniform/fixed-random plans, current adversarial selection and historical selectors under matched held-out recipes. Initially reuse the same small sequence subset rather than wait for every optional extension.

**Deliverables:** versioned two-player learner and inference bundle, fixed-random versus adversarial comparison, and complete validity/coverage accounting.

**Gate:** the implemented choices and targets match the declared game; adversarial results are evaluated with uncertainty, matched conditions, and no selective removal of invalid attempts. A valid implementation may find no improvement; that is a reportable scientific result, not a reason to redefine legality after seeing test outcomes. **Risk:** high semantic integration risk, even though finite selection itself is small.

### P6 — Optional coverage and realism extensions

After the initial physical game is interpretable, expand source diversity, all transitions if not already completed, longer sequences, morphology/reach, footwork, continuous deformation, and eventually joint-level adversarial updates or impact dynamics. Revalidate each expansion. Detailed skin/face rendering is not a planned requirement.

**Deliverable/gate:** separately versioned experiments with renewed feasibility, camera, timing, and validity evidence. This phase must not block the first five.

### Relationship to the PDF's 09A–09E milestones

| Original milestone | Revised placement |
|---|---|
| 09A asset/data | P0 supplies the visual subset; P2 supplies reviewed data and the physical asset. |
| 09B references | P2, with early combined-scene diagnostics enabled by P1. |
| 09C imitation | P3, using a selected tracking implementation rather than mandating WBC. |
| 09D sequences/integration | Integration risk discovery moves to P1; validated execution and G1 baselines remain P4. |
| 09E robust game | P5 for a declared restricted bank/sequence scope; broader coverage belongs to P6. |

The narrower kinematic game can reuse P1/P2 and the P4/P5 interface/learning work while omitting physical-tracker claims. It is not evidence that P3 has passed.

## 7. Proposed plug-in and learning contract

Keep the human/reference provider independent of the G1 learner. These are proposed interfaces, not existing APIs.

| Interface | Essential content |
|---|---|
| `HumanModelSpec` | Model/hash, morphology, joint/body/site/actuator names, coordinate maps, collision policy, glove attachment/proxies, action law, physics/control cadence. |
| `MotionBank` / `PunchPlan` | Source split/provenance, family/hand, compatible starting state/guard, frozen target/world transform, reference/time law, phases, TTC, next guard, deadlines, validation, immutable ID/hash. |
| `plan_punch` / `sample_ref` | Select/adapt a legal reference from actual initial state; explicit rejection if unavailable; sample only the committed plan and elapsed time after launch. |
| `tracker_step` / `guard_status` | Actual human feedback and prior applied action; matched observation/normalization; measured body-relative recovery and dwell. No live G1 homing input. |
| `HumanOpponentAdapter` | Reset/inference clocks, independent entity state/history, actual glove poses, legal boundary mask, sequence status, and distinct invalid reasons. |
| `GameTransition` | Realized actor/critic observations, G1 applied action, selected/held plan, decision eligibility, successor legal set or reproducible context, sequence/time state, first-event labels, validity, true pre-reset successor. |

For the first game, retain a **per-control-tick expanded state**, which fits the existing learning cadence more naturally than a new variable-duration representation. At a continuing successor state, the planner's feasible set is all admitted plans if a new decision is permitted; otherwise it is the singleton containing the committed plan. Use this rule in collection, critic targets, and actor optimization. Terminal SAV handling and the accepted off-control-tick event semantics must remain explicit; do not replace them with a generic PPO reward for G1. PPO is a candidate for human imitation, not a replacement for the protagonist's SAV objective.

A finite minimization through learned Q does not require differentiating through a discrete selector or through MuJoCo physics. If a boundary-only/SMDP representation is introduced later, specify duration-dependent discounting and intermediate failure/arrival events; ordinary fixed-step targets are not sufficient.

Define planner information and move order explicitly. Vision5 allows an adversarial response depending on state and protagonist action; deciding the plan before that action is a different information pattern. Freeze that choice in the experiment contract. The family/TTC/count schedule, legal mask, and sequence completion must be environment-owned, not outputs the adversary can redefine.

The new critic will need human mechanical/controller state, plan/time/sequence state, and the relevant existing G1/sensor history. Include latent delay/buffer/fault state if claiming a fully observed simulator critic; declare any retained omission. Compute new dimensions from the final schema. Do not force the plan into the old 29 zero channels. Preserve the actor's 948-feature order where possible, but make checkpoint migration explicit: new runtime identity, checked actor weights and normalizer, newly initialized/retrained critics.

## 8. Validation changes I would make

**Preserve the Task 08 baseline geometry and timing first.** The analytical glove radius is 0.08 m, head radius 0.06 m, and extra safety margin 0.02 m: frozen-head physical-contact TTC uses 0.14 m separation, while the safety boundary is 0.16 m. Keep both gloves as hazards through guard, outward motion, return, and waiting. A decorative glove mesh needs a declared relation to these proxies; an arbitrary center offset or larger visible shape must not silently change labels. See Vision5 Task 08(f) and the [current runtime contract](../../robust_sav/src/boxing/perception/runtime_contract.py).

**Calibrate the PDF's numerical gates rather than promise them.** The proposed 2 cm glove RMSE, 20 ms P95 physical TTC error, 3 cm/15 degree guard tolerances, 0.15 s dwell, 1.5 s return bound, and 95%/99% completion targets are useful pilot hypotheses. Check source variability and controller performance, choose tolerances before final tests, and report any relaxation with its effect on admitted coverage. Do not make a hard case “pass” by silently substituting an easier TTC.

**Separate timing resolution from root-finding tolerance.** A 1 ms offline reference-contact tolerance is possible for a declared interpolant, but the source interval is approximately 8.33 ms and the accepted physics step is 5 ms. It does not establish 1 ms physical accuracy. Record the crossing bracket or an explicitly defined interpolation estimate. Control is 20 ms; physical event checks remain at 5 ms. Misses must remain in completion statistics even when finite TTC-error percentiles are also reported conditional on contact.

**Treat reliability statistically.** A 99% empirical completion target is not a confidence guarantee. For example, with zero failures in n independent trials, the one-sided 95% exact binomial lower bound is `0.05 ** (1/n)`: about 97.05% at n=100, and at least 99% only at n=299. Correlated crops/retimings do not satisfy independence automatically. Use per-family/TTC/sequence counts and uncertainty; the PDF's trial counts are a pilot floor, not certification.

**Keep all attempted attacks visible in the accounting.** Report admission rejections, valid safe arrivals, valid G1 failures, human falls, misses, timing violations, forbidden motion, return timeouts, and censored attempts. Runtime invalidity must not become a protagonist win or a source of adversary reward. Define whether/when transitions are released to training, especially when final return validates an earlier attack only retrospectively. Preserve early G1 first-event labels and separately test remaining human recovery rather than equating an immediate reset with successful return.

**Use matched comparisons.** Compare random and adversarial selection through the same frozen tracker and motion bank, with matched family/TTC/count, source split, placement, G1 starts, camera settings, and exogenous random streams. Report legal-plan coverage alongside outcomes so selective rejection cannot make one method appear stronger. Use bounded one-seed pilots first; independent trained seeds and a frozen held-out evaluation set belong in the release comparison. Camera changes, longer horizons, and different body models should be separately identified experimental factors.

## 9. Dependency and compute decision

Start with the accepted source and package family: **mjlab `8ee51fbc`, Torch `2.8.0+cu128`, MuJoCo/MuJoCo-Warp `3.11.0`, Warp `1.14.0`, RSL-RL `5.5.0`**. Preserve the preview environment and create a separate candidate training lock. The [dependency assessment](01-compatibility-assessment.md) records the WBC/NumPy conflict and why package version alone does not identify mjlab. No new training lock or WBC integration is claimed by this report.

The connected cluster exposes **V100 32 GiB GPUs on `xeon-g6-volta`**, not the initially mentioned L40S partition. The observed four-node allowance and two GPUs per node are consistent with the stated eight-GPU capacity; that is not an immediate reservation. Native rendering was verified on four GPUs across two nodes. See the [tested submission guide](05-gpu-server-guide.md).

V100 is **not inherently excluded by the candidate stack**. PyTorch's official 2.8 packaging announcement includes Volta support for its CUDA 12.8 build. [PyTorch architecture announcement](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128)

Warp 1.14's documentation lists a CUDA 12.9 standard wheel, minimum compute capability 5.2 for CUDA 12 builds, and minimum 7.5 for CUDA 13 builds. The tested V100 is capability 7.0 with driver 580.159.04; this supports trying the CUDA 12 route, not the CUDA 13 alternative. This is a compatibility inference, **not a completed Torch/Warp training test**. [Warp 1.14 compatibility](https://raw.githubusercontent.com/NVIDIA/warp/v1.14.0/docs/user_guide/compatibility.rst), [installation/build variants](https://raw.githubusercontent.com/NVIDIA/warp/v1.14.0/docs/user_guide/installation.rst)

Use one GPU for initial P1/P3 pilots, then arrays for independent seeds, families, or evaluation batches. Eight GPUs do not automatically accelerate one trainer. Measure contact simulation, camera acquisition, replay memory and learning throughput before selecting environment counts or wall-time requests. The combined body and 672 × 376 sensor workload make extrapolation from a short EGL video particularly unreliable.

## 10. Immediate next deliverable

Implement **P1: a pinned-stack GPU probe and a small G1/human scene harness inside human-punching**, together with a short review of complete source cycles. The review can proceed in parallel with infrastructure work. Success would give us actual Warp/camera evidence and a concrete adapter contract before committing to an exporter or imitation framework.

The remaining critical path is then: physical body/reference feasibility → one reliable punch → restricted sequences and fixed-opponent G1 baseline → a finite constrained game. More realistic rendering and a larger software stack do not remove any of those steps.

## Sources and evidence boundaries

- [RobustSafeArrival-Vision5.pdf](../../RobustSafeArrival-Vision5.pdf): finite-game assumptions; Task 07b measurement/closed-loop phases; Task 08 P1/P2/P3 structure, camera/hazard contracts, and final warm-continuation closure.
- [Task09_Human_Adversary_Implementation_Plan.pdf](../../Task09_Human_Adversary_Implementation_Plan.pdf): §§2–3 assets/data, §4 targeting/TTC, §5 tracker, §6 sequences, §§7–8 integration/game, §9 proposed gates, §10 milestones.
- Current local source links in §5 and project reports 01–05 distinguish implemented behavior from proposals. Native tracking behavior was inspected with `git show` at the exact accepted mjlab commit, not inferred from sibling HEAD.
- External GPU compatibility sources were rechecked for this assessment. WBC dependency findings come from report 01's recorded inspection of revision `689f0836`; its extension APIs have not been executed here.
- This task produced documentation only. No new training, physical-human validation, combined-scene execution, or change to `robust_sav` is represented as completed.
