# Choosing a physical human for the punching adversary

This research informed the first physical implementation. References below to the existing nonphysical model describe the original preview; see [the implemented pilot and results](13-physical-pilot-and-next-gates.md) for the current physical code.

Date: 11 October 2026. Research and design assessment; implementation measurements belong in the accompanying experiment reports. This document does not claim that the proposed controller or all proposed model reductions have already passed physical tests.

## Recommendation

**Keep the existing blue mannequin and add a simple rigid-body dynamics model underneath it.** Use primitive collision/inertia shapes, explicit segment masses, a free unactuated pelvis, bounded joint torques, and foot–floor contact. Keep rendering meshes separate from physical shapes. There is no immediate reason to acquire SMPL-X assets, simulate muscles, reproduce fingers/toes, or replace our motion data pipeline.

For an immediate diagnostic, retaining the existing 20 ball joints preserves the tested 87-coordinate pose layout and avoids confusing a controller failure with a retargeting failure. Call this a **60-actuated-DOF surrogate human**, not an anatomical human. It is a valid articulated dynamics abstraction, but its knees and elbows have extra freedom. Before large-scale training, either implement explicit quaternion-aware mjlab control/observation adapters or reduce the skeleton to scalar anatomical axes and retarget again. Do not assume the current ball-joint trajectory files fit the stock tracking task.

The simpler alternative is also legitimate: a **kinematic adversary** with a constrained library of feasible-looking punch paths. That is enough to test G1's camera, avoidance, launch commitment, and family/target selection. It cannot establish human balance, torque feasibility, reaction to contact, or physically executable recovery. The choice depends on what the experiment is meant to claim; the kinematic baseline need not be discarded when physical tracking is added.

## What the literature and released models actually do

| Reference | Relevant design | What to borrow / what not to assume |
|---|---|---|
| DeepMind Control Suite CMU humanoid | MuJoCo model using capsules, spheres, simple feet, bounded hinge motors and joint ranges. It includes more spine/hand detail than this punching task needs. | Borrow the separation of a simple physical body from motion-control objectives. Do not copy joint axes or numerical ranges without transforming from its different skeleton frames. [Released MJCF](https://raw.githubusercontent.com/google-deepmind/dm_control/main/dm_control/suite/humanoid_CMU.xml) |
| DeepMimic | Its paper's humanoid is 1.62 m, 45 kg and 34 DOF. Learned control tracks reference motions using joint PD targets; the study demonstrates martial arts as well as locomotion/acrobatic skills. | A simplified body can produce recognizable physical human motion. A neural policy is a controller choice; it does not eliminate the physics model. [Paper](https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf) |
| DeepMimic released assets | Spherical torso/hip/shoulder/ankle joints, hinge knees/elbows, fixed wrists, primitive body geometry, explicit masses and torque limits. | A useful engineering scale reference, but its fixed wrists are a poor fit to our orientation-sensitive gloves. [Character definition](https://raw.githubusercontent.com/xbpeng/DeepMimic/master/data/characters/humanoid3d.txt) |
| SMPLSim | Generates SMPL/SMPL-H/SMPL-X-compatible humanoid simulator models and supports MuJoCo; provides a small training framework. | Useful if shape-controlled bodies or existing SMPL-based trackers become central. It is not required to put dynamics under our procedural mannequin. [Official repository](https://github.com/ZhengyiLuo/SMPLSim) |
| PHC | Reference-driven physical imitation and recovery; published setup uses Isaac Gym and separately obtained SMPL models. Its README explicitly notes that a single primitive can work without the full mixture architecture. | A future learned-tracking reference. Its checkpoint and training stack are not drop-in components for our DuoBox skeleton and pinned mjlab environment. [Official implementation](https://github.com/ZhengyiLuo/PHC) |
| mjlab tracking | The local pinned code already contains a reference-motion command, tracking rewards, terminations and RL infrastructure. | Start from the stack already used by G1, after checking the action/observation representation and GPU runtime. Reusing task concepts is easier than assuming a new human asset is automatically compatible. [Pinned tracking source](https://github.com/mujocolab/mjlab/blob/8ee51fbcf806a7419189f706d9e394cbeb7790fa/src/mjlab/tasks/tracking/mdp/commands.py) |

These examples support simple rigid-body models. They do **not** validate a particular invented mass, stiffness or torque limit as a measurement of a human boxer.

## Which physical parameters are unavoidable, and which can be simplified?

| Quantity | Kinematic playback | Dynamic human | Practical simplification |
|---|---|---|---|
| Bone lengths / joint transforms | Required | Required | Keep the tested DuoBox proportions scaled to 1.80 m. |
| Mass / center of mass / inertia | Do not affect prescribed motion | Required | Use simple volume shapes and explicit segment masses; let MuJoCo compute consistent inertias. |
| Joint limits | Optional for rendering, valuable for credible targets | Required for the intended constrained surrogate | Use conservative engineering limits and audit references; improve anatomical detail later. |
| Collision / friction | Optional for a visual preview | Required for supporting feet and any intended physical contact | Boxes for soles, capsules for limbs, simple torso/head shapes. |
| Actuation | None | Required for active tracking | Bounded torque motors with feedback control; no muscle model necessary. |
| Detailed skin / face / finger mechanics | Unnecessary | Unnecessary | Preserve the simple blue surface and rigid gloves. |
| Learned control | Unnecessary | Optional | Try feedback/inverse-dynamics or trajectory optimization first; add learning if those cannot meet the performance target. |

MuJoCo can infer body inertias from geometry with assigned mass or density. Its compiler can restrict inertia inference to selected geom groups, so decorative meshes need not contribute twice. Ball joints have three rotational DOFs; their built-in range limits only the total rotation angle from the reference pose. A ball-joint limit therefore does not model separate elbow flexion, twist and lateral-bend ranges. [MuJoCo XML reference](https://mujoco.readthedocs.io/en/3.11.0/XMLreference.html)

An XML that merely compiles is insufficient: positive mass/inertia, proper unit scale, plausible mass distribution, contact behavior, torque accounting, and integration stability must be checked. In particular, the original preview's placeholder inertias and zero gravity are not an acceptable dynamics baseline.

## Concrete model construction plan

### A. Immediate physics diagnostic with the existing skeleton

Retain all 21 body names and the 20 quaternion joints. Keep the free pelvis unactuated. Add physical primitives in a separate geom group, remove placeholder inertias, assign one mass budget per body, and verify the compiled total and inertia tensors. Feet need flat support patches; the current decorative ellipsoidal feet are not a good sole-contact definition.

Use one explicit total mass such as **75 kg including gloves** as an engineering scenario, with an audited body-mass table. There is no need to estimate an unknown capture actor's actual mass. Distribute the budget toward the torso and thighs, give hands/gloves nonzero mass, and avoid allocating substantial mass to invisible schema-only toe segments. Do not present volume-derived distribution as a biomechanical calibration. A later sensitivity study can vary total mass, mass distribution, friction and strength once a nominal controller works.

For ball-joint control, use a three-component orientation error in a consistent tangent frame, angular-velocity error, and three orthogonal torque motors per joint. Keep saturation after any feedforward term; log actual actuator torques. MuJoCo ball motors use gear-axis directions in the child frame, and scalar-joint `actuatorfrcrange` is ignored for ball joints. Per-actuator force limits and, if desired, a joint torque-vector norm cap must therefore be explicit. [MuJoCo actuator/joint reference](https://mujoco.readthedocs.io/en/3.11.0/XMLreference.html#actuator-general)

This diagnostic can answer whether the motion is approximately trackable, whether the feet support the body, whether the code is using the right frames, and which joints saturate. It cannot establish anatomical knee/elbow behavior. A root weld, fixed pelvis, prescribed root motion, or stabilizing external wrench must be marked as **assisted** and kept out of unassisted success statistics.

### B. Reduced model if scalar mjlab actions or anatomical constraints are needed

A reasonable proposed target is **34 actuated scalar DOFs**, plus the six-DOF free root:

| Region | Proposed DOFs | Treatment |
|---|---:|---|
| Two trunk segments | 6 | Three rotations per segment. |
| Neck | 2 | Pitch/yaw; head rigidly follows its neck segment. |
| Shoulders | 6 | Three per side; clavicle transforms fixed initially. |
| Elbows | 2 | Flexion hinge per side. |
| Forearm rotation | 2 | Pronation/supination per side. |
| Wrists | 4 | Two per side; forearm rotation plus wrist motion retain full glove-orientation control. |
| Hips | 6 | Three per side. |
| Knees | 2 | Flexion hinge per side. |
| Ankles | 4 | Pitch/roll per side. |
| Fingers/toes | 0 | Rigid fist/foot; schema markers may remain as sites. |
| **Total** | **34** | **41 qpos / 40 qvel**, including the free root. |

This is a project proposal, not a supplied SMPLSim preset. Adding axes is easy; determining their reference frames and validating their limits requires care. Fit the source motion onto the reduced model using constrained IK with hand orientation, foot support and temporal smoothness. Recompute every exported body pose and velocity from the exact final model. Simply converting source axis-angle values into hinge coordinates would be incorrect.

A useful initial range-design workflow is: define anatomical axis directions from the bind skeleton; set conservative independent envelopes; inspect neutral/guard poses and each recorded cycle; identify violations; improve retargeting or record a deliberate model change. Avoid setting each bound only to the maximum of the four training clips and calling that an independently validated anatomical limit. Large rotations around shoulders and forearms are also sensitive to the chosen Euler factorization, so explicit sequence/frame documentation is necessary.

### C. Effort and gain starting points

DeepMimic's released asset/controller files provide the following **reference settings for its own character**:

| Joint group | Torque limit in character file (N m) | Kp / Kd in controller file |
|---|---:|---:|
| Shoulder | 100 | 400 / 40 |
| Elbow | 60 | 300 / 30 |
| Hip | 200 | 500 / 50 |
| Knee | 150 | 500 / 50 |
| Ankle | 90 | 400 / 40 |

Sources: [character limits](https://raw.githubusercontent.com/xbpeng/DeepMimic/master/data/characters/humanoid3d.txt), [PD controller configuration](https://raw.githubusercontent.com/xbpeng/DeepMimic/master/data/controllers/humanoid3d_ctrl.txt). These are useful scale comparisons, **not measured human strength and not validated gains for our 1.80 m mannequin, masses or timestep**. They should not be copied wholesale or used to infer wrist capability from fixed-wrist examples.

Choose conservative pilot values, freeze them in metadata, and tune with isolated step responses and gravity-loaded standing before attempting full-speed punching. With PD control, proportional gains have torque/radian units and derivative gains torque/(radian/second); numerical settings depend on inertia, integration method, step size and actuator implementation. Report torque saturation and tracking error rather than silently increasing strength until the reference can be followed. If a reference is too aggressive, slow or re-optimize it and record the timing change.

### D. Standing initialization and balance diagnosis

Start from a quiet guard sample, with joint velocities set to zero, and ramp the target gradually before advancing the punch. Place the root using the **physical sole geometry**, not the minimum decorative mesh vertex over the full clip. The current whole-clip vertical shift can leave one or both feet hovering at the initial pose. Solve a small pose-adjustment problem for both soles near z=0 and a nearly upright pelvis/trunk; do not correct only root height when the two feet require different corrections.

Measure the compiled subtree center of mass and project it onto the floor. For an initial quasi-static guard, place that projection inside the convex hull of the active sole contact patches with a positive margin. This is a useful necessary static-balance check, not a guarantee during a fast punch: angular momentum, acceleration, contact force feasibility and friction matter dynamically. Keep sufficient knee bend to avoid singular straight-leg corrections, but avoid a deep crouch that consumes the available knee/hip effort before the punch starts. Verify both intended stance feet actually bear load after settling.

Use separate diagnostics: free fall without motors, supported joint step responses, a free-root quiet-guard hold, then a slow jab. A small timestep (for example 1–2 ms as a pilot setting) can expose whether a 5 ms rollout failure is integration stiffness or a controller/reference problem; the final production timestep must be tested explicitly. Do not combine very small schema-only segment inertias with large derivative gains. Remove/fuse unused toe dynamics in a future reduced model; if preserving them for compatibility, document their small positive inertia and appropriately weak control.

Gravity or inverse-dynamics feedforward can improve posture tracking, but dropping the root entries of a full inverse-dynamics force vector does not make the remaining torques dynamically feasible. A contact-consistent whole-body controller must solve for allowed joint effort and admissible contact forces with zero root actuation. If plain PD tips over, first inspect COM/support, reference foot motion, momentum and torque saturation; a larger gain or hidden root force is not evidence that the problem is solved.

## The actual software gap

Our current human model has a free root and 20 ball joints, no actuators, zero gravity and disabled contacts. The targeting solver preserves original lower-body motion and does not constrain support, anatomical joint limits or effort. These are local implementation facts, not missing capabilities of MuJoCo itself; see [preview report](03-implementation-and-results.md) and [standing-G1 experiment](07-standing-g1-targeting.md).

The mjlab revision pinned by `robust_sav` expands ball-joint qpos and qvel widths in `EntityIndexing`, but the stock joint targets, per-joint default state, encoder bias, position limits and PD/action machinery are designed around scalar robot joints. A native MuJoCo or MuJoCo-Warp step of the ball-joint model is not evidence that stock mjlab tracking works unchanged. The motion loader expects `joint_pos` and `joint_vel`; our preview stores full `qpos` and tangent `qvel` with a different layout. Reference: [pinned Entity implementation](https://github.com/mujocolab/mjlab/blob/8ee51fbcf806a7419189f706d9e394cbeb7790fa/src/mjlab/entity/entity.py), [Entity data](https://github.com/mujocolab/mjlab/blob/8ee51fbcf806a7419189f706d9e394cbeb7790fa/src/mjlab/entity/data.py).

The native rendering jobs already passed on the available V100 GPUs. Training-stack import, a batched dynamics/contact rollout, and the production camera path remain separate acceptance checks; the GPU experiment report should record their actual results. Neither SMPLSim nor a different human mesh resolves a CUDA/runtime incompatibility.

## What should count as progress

1. **Model checks:** compile; inspect total mass and principal inertias; verify neutral height and all joint/actuator maps; verify feet collide with the floor and decorative meshes do not add mass/contact unexpectedly.
2. **Static standing:** compare assisted and free-root runs; measure pelvis/head height, foot slip, contact forces and torque saturation for several seconds. A supported standing success is not free standing.
3. **One slow jab:** initialize consistently, follow a full guard–strike–return reference, and measure glove error and recovery under gravity. Repeat at nominal timing only after the slow case is understood.
4. **Constrained references:** improve foot support and joint envelopes, then retest without changing acceptance definitions. Inspect actual hand orientation as well as glove center.
5. **Four families and sequences:** preserve continuous state between punches, and evaluate return-to-guard rather than resetting the body at every boundary.
6. **Controller decision:** retain model-based tracking if it meets the gate; train a motion-conditioned residual or imitation controller if balance/recovery/generalization remains inadequate. Learning should solve a measured deficiency.

For G1 avoidance the accepted 0.14 m glove/head center contact condition and 0.16 m safety threshold remain the task geometry. A larger visible mesh does not authorize changing those thresholds. Human–floor dynamics and task head-contact detection are separate contracts; if impact response is not being studied, terminate on the task event and avoid claiming that a visible mesh collision defines the event. The detailed PDF/code audit should resolve equality and termination semantics independently of this model choice.

## Asset and dependency decision

No third-party body-model asset is needed for the proposed pilot. Our procedural surface and existing raw DuoBox skeleton already supply the geometry and reference motion; this report researched external implementations without incorporating their code or body assets.

The CMU humanoid source is in the Apache-2.0-licensed dm_control repository, and DeepMimic is MIT-licensed. SMPLSim's code is BSD-3-Clause; its source-code license does not grant rights to separate SMPL/SMPL-X model files. PHC's setup explicitly directs users to obtain those separately. If any external code/assets are adopted later, retain their notices and record exact revisions. [dm_control license](https://raw.githubusercontent.com/google-deepmind/dm_control/main/LICENSE), [DeepMimic license](https://raw.githubusercontent.com/xbpeng/DeepMimic/master/LICENSE), [SMPLSim license](https://raw.githubusercontent.com/ZhengyiLuo/SMPLSim/master/LICENSE), [SMPL-X project and model-download instructions](https://github.com/vchoutas/smplx).

All implementation stays in `human-punching`. The sibling `robust_sav` and `mjlab` trees were read only. No Git add, commit or push is part of this work.
