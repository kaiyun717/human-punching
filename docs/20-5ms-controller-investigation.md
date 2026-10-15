# Genuine 5 ms human physics: controller investigation

Date: 2026-10-15. This report concerns numerical control and is separate from
learned-policy acceptance. The body, 75 kg mass, segment inertias, 48 scalar
joints, joint limits and original strength limits are held fixed.

## Why the old controller struggled at 5 ms

The accepted early jab checkpoint used 2 ms physics and 20 ms policy updates.
Its proportional/derivative (PD) calculation happened in Python/Torch, and the
result was sent to plain torque motors. Although the simulator selected
`implicitfast`, it could not see the velocity derivative of this externally
computed force. The PD damping therefore behaved explicitly.

The accepted initial guard has a largest generalized damping eigenvalue
`lambda_max(M^-1 D) = 778.5152409958 /s`, where `M` is the compiled full mass
matrix and `D` contains the existing joint derivative gains, with zero free-root
entries. For the frozen, unconstrained pure-damping linearization, explicit
integration requires `h * lambda < 2`:

| Physics step | `h * lambda_max` | Local explicit-damping result |
|---|---:|---|
| 2 ms | 1.5570 | Below the bound |
| 5 ms | 3.8926 | Above the bound |

The corresponding bound is about **2.569 ms**. This is a diagnosis of an
unstable local numerical mode, not a proof that every 2 ms motion is stable or
that every 5 ms motion must fall; contact, stiffness and torque saturation alter
the full system. It explains why simply changing the old physics clock is not
a credible solution.

## Native, bounded PD actuation

The new helper [scalar_control.py](../src/human_punching/scalar_control.py)
expresses exactly the same continuous PD force using a native MuJoCo affine
actuator:

```text
drive = kp * desired_joint_angle + kd * desired_joint_velocity + feedforward
actual_torque = clip(drive - kp * joint_angle - kd * joint_velocity,
                     original_minimum_torque, original_maximum_torque)
```

The native actuator has `gainprm[0] = 1`, `biasprm[:3] = [0, -kp, -kd]`, unit
gear and no activation dynamics. Its output is limited with `forcerange`, after
all feedback and feedforward terms. Its input drive is no longer itself an
applied joint torque, so checking the input against the old motor torque range
would be incorrect. Independent evaluation must check the actual actuator
force and generalized joint force against the unchanged output limits.

This exposes the damping derivative to `implicitfast`. It does **not** add
passive damping, increase armature, relax strength or joint limits, actuate the
root, add external support, reset a falling state, or use hidden 2 ms substeps.
A 20 ms action hold contains exactly four 5 ms physics steps.

This approach is already supported by the pinned software: mjlab's
`BuiltinPdActuator` documents native position/velocity actuation for stiff PD,
and the installed MuJoCo-Warp derivative kernel includes affine actuator
velocity bias. Warp suppresses that derivative when the output force is
saturated and clamps the actual output with `forcerange`.
MuJoCo's [force generation and numerical integration documentation](https://mujoco.readthedocs.io/en/stable/computation/)
describes these native force laws and implicit velocity integration.

The conversion creates a distinct runtime model; the earlier body XML and
checkpoints remain immutable. Old checkpoints must not silently be evaluated
under the changed controller semantics.

## Focused verification

[test_scalar_control.py](../tests/test_scalar_control.py) checks:

1. Native output torque equals the original clipped PD law at representative
   positive/negative states, including both saturation directions; root
   actuator forces remain zero.
2. An isolated low-inertia hinge converges with native damping at a real 5 ms
   timestep, while explicit damping at the same gains and timestep diverges.
3. Exporting the full human changes no masses, inertias, body positions, joint
   ranges, damping, armature, dimensions or original output strength limits.
4. An actuator without a provable output-force cap is rejected.

All **four tests passed**. These tests establish implementation properties and
numerical mechanism; they do not substitute for free-standing, learned GPU
rollout evaluation.

Two additional runtime-export round-trip tests were subsequently added after
the serialization issue below: **six controller tests now pass**, covering
both explicit motors and native PD.

## Native full-body probe

[control_5ms_probe.py](../scripts/control_5ms_probe.py) compares explicit versus
native PD at 2 ms and 5 ms, with the previously available bounded joint balance
feedback both disabled and enabled. Each deterministic trial starts once, holds
guard for three seconds, executes the entire accepted jab reference, then holds
guard for one additional second. The probe saves every physics state, actuator
force, generalized actuator force and applied external force. It checks the
MuJoCo clock on every step to reject resets or an unexpected timestep.

The first CPU submission, Slurm **5931585**, failed during Torch import because
the worker could not resolve `libnccl.so.2`; no simulation ran. It ran on the
same worker `d-19-14-4` that produced filesystem errors in another concurrent
task. The retry, **5931597**, ran on `d-6-4-4` and explicitly added the already
installed project-local NCCL directory to the dynamic-library search path.
It completed all eight cases. Avoid the problematic worker for follow-up CPU
jobs; the evidence does not isolate the first failure to a package deficiency.

The [saved numerical results](../outputs/control-5ms-5931597/summary.json)
show the following at **5 ms**, without learning:

| Actuator | Joint balance feedback | First fall | Three-second guard: peak joint speed | Guard: maximum root drift | Attack-window glove/head minimum |
|---|---|---:|---:|---:|---:|
| Explicit PD | Off | 1.225 s | 31.67 rad/s | 1.419 m | 1.373 m |
| Explicit PD | On | 1.505 s | 26.90 rad/s | 1.765 m | 1.081 m |
| Native PD | Off | 3.245 s | 1.32 rad/s | 0.471 m | 1.643 m |
| Native PD | On | 7.785 s | 1.32 rad/s | **0.00641 m** | 0.1735 m |

The native controller with joint balance feedback also limits initial sole
drift to **3.15 mm**. The numerical oscillation is removed, but the jab still
misses the 0.14 m contact criterion during its attack window. It later reaches
0.1265 m on recovery and then falls; that late contact is not a successful
attack. Native PD by itself does not solve free-standing balance. This makes
the distinction between a numerically usable actuator and a learned punching
controller concrete.

Across **all eight cases**, actual output torque respects every original
strength cap; applied external forces and root actuator forces are exactly
zero. The 5 ms records contain 1,798 steps over the full quiet-guard, jab and
final-hold sequence. These long deterministic probes deliberately continue
after a fall to show what happens; there are no hidden recovery resets.

![Explicit versus native PD at genuine 5 ms physics](../outputs/control-5ms-5931597/comparison.png)

The initial gray interval is the three-second guard hold. Native damping removes
the fast oscillation, while the later loss of balance shows why numerical
stability alone is insufficient. [Vector figure](../outputs/control-5ms-5931597/comparison.svg).

## A runtime XML export bug found by independent audit

The first GPU smoke run, **5931602**, and training jobs **5931617/5931618** used
the correct native actuator arrays in memory. Their initial runtime XML export,
however, used `mj_saveLastXML` after mutating a loaded `MjModel`. That export
copied the numeric `biasprm` and output force range but retained the original
`ctrllimited=true` and omitted `biastype=affine`. Recompiling the saved file
therefore reconstructed a different actuator: a plain motor that also clipped
the drive input. A downstream evaluator using that file would audit the wrong
controller. The automatic save also rounded body mass/inertia and joint limits;
the recompiled total mass was 75.000012 kg instead of the original 75 kg.

The [forensic runtime audit](../outputs/control-5ms-5931597/runtime_audit.json)
records this discrepancy without rewriting those historical files or their
hashes. It is a serialization failure, not evidence of unbounded forces in the
actual training simulation. An independent [GPU force audit](../outputs/control-5ms-5931597/gpu_force_audit.json)
checked all nine saved smoke-run rollouts, each containing 1,199 samples. The
recorded actual forces satisfy the native clipped PD law to **3.03e-5 Nm**
maximum numerical error. The recorded generalized hinge force equals the
actuator force exactly, every original torque cap is respected, and root
actuation/applied external forces are exactly zero. Reconstruction correctly
uses the preceding physics state, since these cached force values describe the
interval that produced the newly recorded state.

The fix is `export_runtime_model(source, destination, model)` in
[scalar_control.py](../src/human_punching/scalar_control.py). It copies the exact
source XML body, explicitly writes the actual actuator enum/boolean fields and
all gains/biases/limits, and saves the actual timestep, solver settings, gravity
and flags. It then recompiles the result and requires exact equality of relevant
body, actuator and solver arrays before accepting the export. It refuses a
changed body it cannot faithfully serialize. This avoids both missing actuator
semantics and unnecessary mass/inertia rounding.

The ongoing training weights remain usable because the actual GPU dynamics
were correct. Accepted results must come from new evaluation jobs with the
correct export and matching runtime hash; the old exports remain diagnostic
evidence. New evaluations should also record the simulator's own per-world
clock, in addition to the nominal sampling timestamps, to provide direct
evidence of the physical step count.

## Early full-speed jab precision diagnostic

The independent checkpoint-100 evaluation **5931645** exposed two specific
remaining problems before further training. These are intermediate findings,
not the final status of the subsequent checkpoints.

The worst joint-limit excursion was **0.022247 rad** at `LeftHand_z` at
1.230 s, against the unchanged 0.020 rad gate. The reference reaches the
0.610865 rad upper limit while rapidly decelerating the wrist: its velocity
drops from roughly 3.8 rad/s near 1.16 s to zero by 1.24 s. The actual wrist
lags this braking and briefly reaches 0.633112 rad. Its actuator is already
applying a braking torque at the peak. Clamping the desired angle to the joint
limit therefore does not guarantee that inertia and a compliant physical stop
cannot produce a small transient excursion.

If later checkpoints do not resolve it, a direct joint-limit penalty is a
meaningful first change: penalize the worst actual excursion across joints and
all four 5 ms steps, rather than averaging it over 48 joints or checking only
20 ms action boundaries. Here the worst sample lies between action boundaries.
A small declared inset of commanded target angles is another legitimate
controller option, but it changes the controller contract and requires new
training/evaluation. Relaxing the physical joint limits or the acceptance gate
would not address the problem.

The nominal first contact was **40 ms late**. Full-cycle active-glove RMS error
was only 1.494 cm, but attack-window active-glove RMS was 2.952 cm. At reference
contact, the actual glove was approximately 2.51 cm behind in x, 2.37 cm offset
in y and 5.59 cm low in z. A diagnostic temporal alignment over 0.9–1.4 s finds
roughly 20 ms lag, but still leaves 2.78 cm position RMS. Consequently the
40 ms threshold-crossing delay is not simply an actuator latency that can be
cancelled by advancing the reference 40 ms.

The current precision reward averages both glove errors before applying its
exponential reward and also scores the long guard interval. If further tuning
is needed, separate active-hand tracking during the attack from inactive-hand
guard tracking, and give the attack position/velocity errors direct weight.
Keep the intended reference clock and timing evaluation fixed. The policy
already observes future reference states, so it can learn physical anticipation.
More complete acceleration feedforward is a later option: the current offline
feedforward compensates gravity and velocity effects but omits the reference
mass-times-acceleration term. That requires careful reference-acceleration and
support-force validation and is a larger change than targeted reward tuning.

## Acceptance still required

The relevant downstream milestone is a learned actor validated on MuJoCo-Warp
at **5 ms physics / 20 ms actions**. Native CPU diagnostics alone cannot establish
that. Evaluation must use the saved runtime actuator model, actual force caps,
one initial reset, all physics samples, no external/root assistance, and the
same independent contact, balance and supported-guard checks used for the
earlier milestone. Punch speed and tracking accuracy must then be improved at
that exact clock, followed by the remaining punch families.
