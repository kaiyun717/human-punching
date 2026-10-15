# Choosing the human model before learning a punching policy

12 October 2026. This assessment answers why the pilot used a procedural mannequin, what its successful motions establish, and what should change before training. No model, reference, controller, or sibling repository was changed for this assessment.

## Why the first model was procedural

The first requested milestone was a 1.80 m human executing four DuoBox punches in the same simulator as G1. The procedural mannequin preserves the existing capture skeleton and quaternion layout, needs no external body assets, and makes glove attachments and source FK easy to verify. A second stage added rigid-body physics underneath that surface to isolate contact, balance, control, and runtime questions.

This was a prototype choice. It was not a conclusion that a custom body is more suitable than SOMA or SMPL for the eventual learned human policy. The original Task 09 PDF already specified SMPL-X through SMPLSim as its primary body and SOMA23 as a fallback. Our later model survey in report 08 did not evaluate SOMA specifically; that omission must not be confused with the original plan. Neither planned standard body has been implemented here. See the [plan-versus-implementation reconciliation](15-task09-body-plan-reconciliation.md). The existing 20-ball-joint rig should remain a diagnostic baseline rather than be treated as a finished anatomical training body.

The user-requested plain blue appearance can be retained with any suitable skeleton or physical model. Removing facial detail does not require removing anatomical joint constraints.

## Three separate properties

| Property | What it supplies | What our pilot establishes |
|---|---|---|
| Visual mesh | Surface appearance and its attachment/deformation with the skeleton | A featureless procedural surface follows the capture-derived skeleton and glove attachments |
| Physical rig | Joint articulation, mass/inertia, contact, friction, passive forces, actuation and limits | A coherent 75 kg articulated surrogate, free unactuated root, bounded torque control and real floor support |
| Executed motion | Actual state trajectory produced by the controller in that physical rig | A full-speed jab and slower other punches pass the pilot's contact/support gates; two jabs recover guard without reset |

None of these alone establishes biomechanical validity. A detailed mesh can be attached to unrealistic joints; a simple visual surface can cover a reasonable articulated body. Mesh realism and physical realism must be assessed separately.

## How physically reasonable is the current model?

Its dynamics are real: positive masses and inertias, gravity, bounded torque motors, an unactuated floating pelvis, and foot contacts. The accepted physical trajectories are produced by integration rather than prescribed pose playback. The native/Warp comparison and no-reset sequence provide useful implementation evidence. See [the physical pilot report](13-physical-pilot-and-next-gates.md).

However, its human validity is limited:

- All 20 articulated joints are ball joints. Elbows and knees have extra rotation axes compared with a constrained human approximation. A spherical rotation cone does not impose independent flexion, twist and lateral-bend limits.
- Mass distribution and torque limits are engineering choices. They are not measurements of the DuoBox actor or a calibrated boxer.
- Human self-collision is disabled. Successful support/contact metrics do not rule out body interpenetration.
- The pilot mainly uses planted double support and a controller assuming that support mode. It does not validate weight transfer, stepping, recovery from pushes, or broader athletic motion.
- The cross and hooks currently pass at 0.3 of source playback speed. Full-speed failures identify limitations of the current reference/controller/model combination; they do not show that a human could not throw the original punch.
- Head/glove contact is an analytic 0.14 m center event with no G1 impact response. It validates the accepted avoidance hazard, not striking mechanics or force transfer.

Accordingly, the accurate claim is: **the accepted motions are executable in the current surrogate simulation; anatomical feasibility for a real human remains unestablished**. Captured human motion is a useful reference, but reconstruction noise, skeleton changes, IK, floor conditioning and timing changes can introduce poses or derivatives that the actor did not perform.

## What SOMA would and would not give us

Assuming the intended SOMA is NVIDIA's current SOMA body representation, [SOMA-X](https://github.com/NVlabs/SOMA-X) supplies a standard parametric body/hand representation, shared topology and rig conventions, skinning, fitting, and conversion tools. Those features would help with body proportions, consistent motion representation and asset interoperability. A parametric surface alone does not define calibrated strength, mass distribution or anatomical physics constraints.

There is also a concrete simulation starting point: NVIDIA's [ProtoMotions SOMA23 configuration](https://github.com/NVlabs/ProtoMotions/blob/main/protomotions/robot_configs/soma23.py) and [MuJoCo asset](https://github.com/NVlabs/ProtoMotions/blob/main/protomotions/data/assets/mjcf/soma23_humanoid.xml). The inspected configuration contains 23 bodies and 66 articulated scalar DOFs. The XML includes primitive physical shapes, motors, joint damping/stiffness, and selected self-collision exclusions. Scalar coordinates could simplify integration with the pinned mjlab action/tracking machinery, although its skeleton and state layout differ from our current DuoBox archive.

**It is not automatically anatomically constrained.** The inspected default XML gives both forearms and shins three hinge axes with broad ±180° ranges. Thus adopting SOMA23 without changes would retain extra elbow/knee freedom in a different coordinate representation. Its passive stiffness/damping, motor gear interpretation, torque bounds and collision exclusions also need auditing under our controller; numerical defaults from another training stack are not proof of human biomechanics.

ProtoMotions documents [physical motion tracking and multiple simulation backends](https://nvlabs.github.io/ProtoMotions/), including a Newton path based on MuJoCo Warp. This is relevant prior work, not a tested drop-in dependency for our pinned mjlab/Torch/Warp environment. We have not imported the SOMA asset, installed ProtoMotions, or validated its GPU/runtime compatibility in this project.

[SMPLSim](https://github.com/ZhengyiLuo/SMPLSim) is another route to a standardized body-derived MuJoCo model. It still requires retargeting, property/constraint audits, and attention to separate body-model asset requirements. [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/) demonstrates learned physics control of simplified humanoids, including martial arts; realistic skin detail is not a prerequisite for that approach.

These are sources inspected on 12 October 2026. Repository `main` branches may change; any implementation should pin the chosen source and asset revisions.

## Recommended next development phase

Before substantial policy training, choose and freeze the physical training rig. A useful first comparison is a standard SOMA/SMPL-derived simulation asset against the current diagnostic rig, judged by control compatibility and physical constraints rather than appearance. Avoid paying for detailed skin or fingers unless the task needs them.

1. **Constrain articulation.** Use knee flexion hinges; constrained elbow flexion plus separately represented forearm pronation/supination; bounded shoulder/hip/trunk and ankle axes; and enough wrist freedom to preserve glove orientation. Retain a free unactuated root and simple rigid fists/feet.
2. **Audit physical properties and forces.** Freeze body scale, mass distribution, inertias, motor strength, passive forces, friction and collision filters. Account for every force source, including joint springs/damping, rather than reporting only capped motor torques.
3. **Retarget and certify references on that exact rig.** Include hand orientation, foot support, temporal regularity, joint envelopes and self-collision. Measure final FK and derivatives after all IK edits. Preserve the 0.08 m glove / 0.06 m head / 0.02 m safety margin contract.
4. **Train motion imitation first.** Learn a reference-conditioned controller for balanced full-cycle execution and guard recovery. Observations should include actual proprioception and reference/phase/target information; actions should pass through a documented bounded actuator controller. Reference imitation and physical penalties should discourage nonhuman exploits.
5. **Add goal-conditioned punching.** Vary feasible head targets and timing, validate all four families at useful speeds, and assess recovery/generalization. Only then add adversarial hit objectives against G1 and the two-player integration.

An unconstrained hit reward could exploit the current extra joint freedoms, disabled self-collision or unrealistic strength assumptions. Learning optimizes the model it is given; it does not infer missing anatomy automatically. Conversely, failing our current PD controller is not evidence that imitation learning cannot execute a properly constrained reference.

A physically constrained rigid-body human with a plain blue surface is sufficient for an initial learned boxing policy. A biomechanically predictive study of injury, muscle activation or human force capacity would require substantially different validation; those claims are not needed for the current G1 avoidance objective.
