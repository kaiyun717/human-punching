# Independent final runtime review

Read-only review of the four selected learned/baseline pairs:

| Family | Learned evaluation | Zero-residual evaluation |
|---|---:|---:|
| Jab | 5931806 | 5931807 |
| Cross | 5931915 | 5931916 |
| Left hook | 5931934 | 5931935 |
| Right hook | 5931918 | 5931919 |

The independent acceptance evaluator covers all 64 worlds in each run. This
additional audit checks nominal world 0 and final bank world 63 independently;
it does not substitute a smaller acceptance bank.

All four pairs pass these checks:

- Recompiled runtime body properties exactly match the source: 75 kg total
  mass, inertias and COM frames, joint ranges, passive damping, armature,
  collision geometry and contact settings. Original torque caps are unchanged.
- Actuators use fixed unit gain, affine position/velocity bias, no activation
  dynamics, no input clipping, and native output force limits.
- The actual simulator clock advances by 5 ms and recorded actor boundaries
  occur every four steps. No external or root assistance appears.
- Actual force agrees with `clip(ctrl - kp*q_previous - kd*v_previous, caps)`
  to within **0.0000301 Nm**; actuator-to-joint transmission is exact.
- The zero-residual drive reconstructed independently from reference pose,
  velocity and support feedforward matches the recorded baseline input
  **bit-for-bit** in all eight checked baseline worlds. Baselines use identical
  controller schemas, runtime models and frozen initial-state banks.
- Learned rollouts' checkpoint byte hashes match their selected checkpoints.

Force samples following an integration step are compared with the preceding
state, because that is where MuJoCo computes the interval's cached forces. The
initial forward-only sample is excluded from this interval check.

Code review also confirmed one Warp step per integration call, four substeps
per policy update, and no integration during FK refresh. A packaging mismatch
was found: nonuniform timing metadata lives inside `reference_timing`, whereas
the first packager read it at the top level. The packaging owner corrected the
lookup and added realistic regression fixtures before publication. This was a
preflight rejection, not an accepted physics artifact with incorrect timing.

The scope remains isolated human policies. Shared human/G1 physics still needs
one agreed global solver/friction-cone configuration and independent validation,
as described in [report 24](../../docs/24-five-ms-controller-interface.md).

[Machine-readable evidence](report.json) · [Audit source](audit.py).
