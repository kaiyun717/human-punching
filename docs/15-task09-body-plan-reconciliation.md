# Task 09 training body: original plan versus implemented prototype

12 October 2026. Assessment against the five-page [Task 09 implementation plan](../../Task09_Human_Adversary_Implementation_Plan.pdf), current project source, generated experiment records, and the earlier project reports. Page numbers below are physical PDF pages. This is a documentation update; it does not import a new body, train a policy, or change simulation behavior.

## Direct answer

**We have a physical body for controller and training-development experiments. We do not yet have the validated training body proposed in Task 09, a registered human imitation-learning task, or a trained human policy.**

The original PDF explicitly selects **SMPL-X through SMPLSim as the primary body**, with **ProtoMotions SOMA23 as a fallback**. Those were already in the plan. The current custom mannequin is an intermediate diagnostic implementation, not an SMPL-X or SOMA export. Earlier discussion should have made that distinction explicit.

The architecture remains reasonable. The prototype strengthens the case for completing the plan's body/control audit before substantial training. It does not establish that the proposed standard bodies are unsuitable, or that a custom model is better.

## 1. What the original PDF actually specifies

| PDF location | Specification | Interpretation |
|---|---|---|
| Page 1, sections 1–2 | DuoBox fitted SMPL-X motion; SMPLSim-generated neutral adult near 1.80 m; one morphology initially | A concrete primary asset path, not simply an unspecified human mesh |
| Page 1, section 2 | SMPLSim model export; articulated rigid mesh segments with separate simple physical proxies | The plan already permits simple rendering and collision geometry; seamless skin is deferred |
| Page 1, section 2 | Free unactuated pelvis, root plus 21 articulated joints, proposed 63 scalar hinges; all three wrist rotations retained; fingers fixed | This reduction requires new exporter work. The PDF explicitly says it is not a shipped preset |
| Page 1, section 1 | SOMA23 fallback: 23 bodies, 66 scalar DOFs, wrists | Its pretrained tracker is explicitly not assumed to transfer to mjlab |
| Page 1, section 2 | Audit masses, inertias, anatomical ranges, feet/friction, self-collision, bounded PD, passive forces and simulator rates | A generated model must pass physical checks; the PDF does not claim that a mesh provides these automatically |
| Page 2, sections 3–4 | Constrained reference adaptation, complete cycles, frozen launch target and first-contact TTC | Offline IK/reference preparation is separate from actual physical tracking |
| Page 3, section 5 | WBC-mjlab human extension, motion-conditioned PPO tracker, bounded reference-relative joint commands | Train physical imitation; motion data do not contain motor actions for the new body |
| Pages 3–4, sections 6–8 | Measured guard recovery; continuous sequences; shared G1 scene; frozen tracker with boundary-only adversarial planning | Keep low-level execution and adversarial plan selection separate |
| Page 5, sections 9–10 | 09A assets/data; 09B references; 09C imitation; 09D sequences/G1; 09E game | Proposed timing and success thresholds are acceptance targets, not demonstrated capabilities |

The named software route is therefore SMPLSim → constrained retargeting with Mink → WBC-mjlab imitation → robust-SAV integration. Native mjlab tracking is already offered as a smaller-dependency alternative; GMR and PyRoki are optional. Importing all these frameworks is not a requirement.

## 2. What exists in human-punching now

The [physical model builder](../src/human_punching/physical.py) adds dynamics to the [procedural blue surface](../src/human_punching/human.py). Its 21-body DuoBox hierarchy has 20 ball joints, 60 torque motors, an unactuated free pelvis, and 87 position / 66 velocity coordinates. Its chosen total mass is 75 kg including gloves. Gravity, primitive-derived inertias, floor contact and bounded torque control are active. Appearance remains independent of physical geometry.

This body can support experiments with a learned controller once the environment and policy interfaces are implemented. It is not currently a drop-in mjlab tracking asset: the quaternion joint coordinates need custom action/observation/PD handling, whereas the stock tracking machinery is oriented toward scalar robot joints.

The local inventory contains no implemented SMPLSim or SOMA body, and neither project environment contains SMPLSim, SOMA/ProtoMotions, Mink or WBC. The dependency cache contains the pinned mjlab source. `DuoBox/fbx_fit_smpl` does contain fitted-motion files; those must not be confused with the separate parametric body-model files or an exported physical body. Their compatibility with a future exporter has not been established by our raw-joint motion adapter.

Three designs appear in the documentation and must not be conflated:

| Design | Status |
|---|---|
| PDF: SMPL-X/SMPLSim, proposed 63 scalar hinges; SOMA23 fallback with 66 | Planned candidates; not imported, reduced, or validated here |
| [Model research report](08-physical-human-research.md): proposed 34 actuated scalar DOFs | Alternative reduced anatomy proposal; not implemented or validated |
| Current code: 20 ball joints / 60 actuated rotational DOFs | Implemented and physically exercised diagnostic surrogate |

The 34-DOF proposal would distribute glove orientation across forearm rotation and two wrist axes. It is a change from the PDF's three-axis wrist prescription, not an equivalent mapping already demonstrated by code. Any reduction must explicitly preserve the glove poses needed by all four punch families.

## 3. What the experiments establish

The [physical pilot report](13-physical-pilot-and-next-gates.md) and [review manifest](../outputs/physical-review/manifest.json) provide the measured evidence:

- A jab at source playback speed contacts the analytic head envelope in both native MuJoCo and actual MuJoCo-Warp, with no root actuation. The Warp run records about 5.05 mm maximum foot drift.
- The cross and both hooks have passing isolated native-physics cases at 0.3× source speed. Different conditioning/gains were used; this is not one controller demonstrated across arbitrary requests.
- Two jabs at 0.6× speed execute continuously with measured guard recovery and no intermediate state reset.
- Successful trials use 2 ms physics/control updates. The tested 5 ms variants failed their tracking/support checks. Accepted G1 integration uses 5 ms physics, so combining the systems still requires a deliberate timing solution and reevaluation.
- Glove/head events use the accepted 0.14 m contact and 0.16 m safe-clearance thresholds. The G1 is fixed; impact response is disabled. This validates an avoidance hazard, not punch force transfer.

The body still has extra elbow/knee freedom, engineering rather than calibrated mass/strength parameters, and disabled self-collision. The controller assumes planted double support. These limitations are especially relevant for learning, which can exploit the model's freedoms.

Current contact delays start at the controller cycle/clip start. They are not yet the PDF's annotated outward-launch TTC. None of these demonstrations establishes the PDF's statistical tracker gates, all 16 family transitions, moving-G1 performance, or an accepted adversary.

## 4. Does the plan still make sense?

**Yes as a staged design; no as a ready-to-run recipe or evidence of a finished training body.** Several important safeguards are already in the PDF. Our experiments turn them into specific implementation requirements rather than reasons to discard the design.

### Keep the body candidates, but validate the articulation

SMPLSim remains a reasonable first candidate because it exports SMPL-family MuJoCo bodies and aligns with the PDF's intended data path. This capability is documented by the [official SMPLSim repository](https://github.com/ZhengyiLuo/SMPLSim). Model-file availability, exact fitted-motion schema, exporter reduction and pinned-runtime compatibility still need checking. SOMA23 remains the explicit fallback if that route is impractical.

Do not treat a standard model name or scalar joint count as proof of anatomy. The inspected [SOMA23 MuJoCo asset](https://raw.githubusercontent.com/NVlabs/ProtoMotions/main/protomotions/data/assets/mjcf/soma23_humanoid.xml) gives forearms and shins three hinge axes with broad ±180° ranges. It also includes passive springs/damping and motor gears that require careful torque accounting. It would need a constraint/control audit here. Likewise, the PDF itself warns that SMPLSim's broad exporter limits are not human biomechanics.

Choose joint axes and limits based on the intended human approximation, then validate reach and wrist orientation. Do not require exactly 34, 63 or 66 actuated coordinates before that work. A constrained rigid-body model is sufficient for this avoidance research; muscles, skin mechanics and detailed fingers are unnecessary.

### Preserve the blue appearance independently

Our procedural choice was useful for the early visual/FK and runtime milestones. The user's request for a featureless blue body is not an argument against SMPL-X, SOMA, or anatomical constraints. A standardized physical rig can retain a simple blue surface. Conversely, a more realistic surface would not repair the present joints or controller.

### Prefer the runtime we have actually tested

Start learning integration from accepted mjlab commit `8ee51fbcf806a7419189f706d9e394cbeb7790fa` and the [tested physics environment](09-physics-runtime.md). The current sibling mjlab checkout and the PDF's WBC lock are different dependency combinations. Native mjlab tracking is a sensible first implementation path; use WBC only if its concrete benefits justify porting and dependency work. In either path, continuous reference clocks, immutable launch plans and physical guard transitions need project-specific code.

### Resolve timing before claiming shared-plant compatibility

The current 2 ms human pilot is not compatible with 5 ms G1 physics merely by placing both XML bodies in one scene. Either achieve stable control at the accepted timestep or define a common finer physics clock that preserves declared actor/sensor schedules and then revalidate G1 behavior. Separate entities in one coupled simulation do not have independent physics timesteps. The PDF already requests rate checks; those checks have now exposed a concrete failure.

### Learn earlier than an exhaustive classical-control campaign

The successful classical controller remains a valuable baseline. It is not necessary to make that controller execute every family at full speed before trying imitation learning. After the body, actuation, feasible references and reset/observation contracts pass a small pilot, train one reference-conditioned jab and assess recovery. Broader fast tracking is a legitimate goal for learning. Learning cannot fix invalid anatomy, unreachable targets, arbitrary TTC demands or inconsistent reference coordinates.

The initial 0.50/0.65/0.80 s TTC grid and tight tracking gates should remain proposed targets until launch annotations and measured feasible ranges exist. The current slowed hooks neither meet those gates nor prove that the recorded human motions are impossible.

## 5. Revised next milestones within the original phases

| Phase | Next work and exit condition |
|---|---|
| **09A — Finalize the training body** | Attempt the planned SMPLSim asset route, using SOMA23 if needed; retain the existing body as a diagnostic baseline. Freeze one 1.80 m morphology, anatomical joint maps/limits, masses/inertias, passive forces, effort caps, soles and collision rules. Verify full glove orientation, unassisted guard support and a documented timestep. Integrate one scalar-joint action/observation path on the pinned stack. No extensive multi-body comparison or detailed skin work is needed. |
| **09B — Retarget to that exact body** | Transfer the existing IK/FK and contact-contract work; re-solve poses after changing the skeleton. Start with a complete jab, then four families. Annotate outward launch and guard phases, export consistent derivatives, and measure feasible target/TTC ranges. Preserve source provenance and recording-level splits. |
| **09C — Train physical imitation** | Register a human mjlab task; train one jab from guard with bounded actuation, no external root force and physical return. Then expand families, placements and feasible timing. Freeze observation/action/controller manifests with checkpoints. Evaluate actual execution, not reference playback. |
| **09D — Continuous sequences and G1** | Extend the two-jab proof to mixed-family transitions. Complete shared-entity state/action/reset indexing, the real camera pipeline, and no-homing tests. Evaluate a fixed/random valid opponent before learning adversarial choices. A real G1-policy evaluation needs its accepted artifacts. |
| **09E — Adversarial planning** | Freeze the accepted tracker. Choose among multiple validated plans only at legal boundaries; include committed plan/phase and opponent state in learning interfaces. Preserve the prescribed sequence/count and report invalid attacks. Broader joint-level adversarial fine-tuning remains a later experiment. |

Shared-scene/runtime checks should continue during 09A–09C rather than waiting until all imitation training is finished. Existing contact definitions, glove attachments, event logging, source audit, experiment fixtures and GPU validation are reusable; body-coordinate references and controller gains must be regenerated when the rig changes.

For scope clarity, the PDF's proposed `src/boxing/human/` modules belong in this project's `src/human_punching/` until the user explicitly authorizes changes to `robust_sav`. This assessment changed documentation only, submitted no GPU jobs, and performed no Git staging, commits or pushes.
