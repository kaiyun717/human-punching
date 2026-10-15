# Human controller interface at 5 ms

15 October 2026. This is the implementation contract for the new isolated human
trackers. It describes what must be preserved when integrating them downstream;
it does not claim that a shared dynamic human/G1 scene is already validated.
Per-checkpoint acceptance results are in [report 22](22-five-ms-learning-progress.md).

## Clock and action contract

| Quantity | Required behavior |
|---|---|
| Physics integration | One actual MuJoCo-Warp step of **0.005 s** |
| Actor inference | Every fourth physics step: **0.020 s / 50 Hz** |
| Held actor output | 48 joint-angle residual commands; same output held for four steps |
| Reference and PD drive | Recomputed every **5 ms**, using the current reference phase and actual state |
| Residual bounds | Clip raw action to `[-1,1]`, multiply by checkpoint scale, currently **0.25 rad**, then clamp the resulting target to joint limits |
| Root | Free and unactuated; no applied root wrench or pose correction |
| Phase | Elapsed time since the single guard reset; advance by 5 ms per integrated step |

The residual is held for 20 ms; the reference and resulting drive/torque are
**not** held constant for that whole period. A 20 ms actor period also does not
add an artificial 20 ms sensor or actuator delay. Learned contact-timing error
is measured separately.

At an actor boundary the current implementation refreshes derived kinematics,
builds the observation, and obtains a new residual. It then performs:

```text
repeat four times:
    sample reference pose, velocity and feedforward at the current phase
    build the bounded joint-position target using the held actor residual
    write the native actuator drive
    integrate exactly one 5 ms physics step
    advance the reference clock by 5 ms
```

`Simulation.forward()` refreshes FK, contacts and forces; it does not integrate
time or replace qpos/qvel. New checkpoints use this refreshed observation
convention. The original 2 ms checkpoint used a different derived-state timing
convention and must not be silently reinterpreted.

Evaluation records nominal timestamps, the simulator's own per-world clock,
physics-step count and actor-update indices. Acceptance checks require
monotonic 5 ms simulator increments and actor updates every four steps, allowing
the documented floating-point clock tolerance. There are no hidden 2 ms steps.

## Actuator input is a drive, not an applied torque

The new runtime uses one native, unit-gear affine actuator for each hinge:

```text
drive = kp * target_angle + kd * reference_velocity + joint_feedforward
actual_torque = clip(drive - kp * actual_angle - kd * actual_velocity,
                     original_minimum_torque, original_maximum_torque)
```

`ctrl` contains the drive. It must not be clipped to the old direct-motor torque
range: the large `kp * target_angle` term is cancelled inside the native force
law. The physical output is bounded by `actuator_forcerange`, with the unchanged
original strength limits. Audit `actuator_force` and the corresponding
`qfrc_actuator` entries, rather than interpreting `ctrl` as torque.

The native damping derivative is included in `implicitfast`; the installed Warp
implementation suppresses that derivative when the output is already
saturated. No additional passive damping, armature, strength or external
support was introduced. Reference support feedforward is an offline joint
torque calculation, not an applied support force. The optional joint balance
feedback setting is part of the checkpoint contract; the primary new trackers
use it disabled.

Cached forces recorded immediately after a physics step describe the preceding
integration interval, while qpos/qvel are the new state. Recomputing force
equivalence must respect that pairing. Independent geometric contact metrics
recompute FK from the recorded state; foot-support evidence uses the recorded
actual contact forces. These recordings do not establish biological muscle or
impact-force calibration.

## Artifact and state compatibility

Keep the checkpoint together with its exact reference, body metadata,
`controller_schema.json`, source-body XML and exported runtime XML. The source
body has 55 qpos coordinates, 54 velocity coordinates and 48 actuators. Map
named joints and their qpos/DOF addresses explicitly when placing it inside a
larger scene; local array offsets are not universal scene offsets.

The runtime XML must preserve native affine bias, force-limit flags and solver
settings. Use `export_runtime_model(...)` from
[scalar_control.py](../src/human_punching/scalar_control.py), which recompiles and
checks the export. The earlier `mj_saveLastXML` serialization failure is
documented in [report 20](20-5ms-controller-investigation.md).

Checkpoint validation covers the model/reference hashes, joint order and
addresses, gains, strength limits, action scale, observation layout, physics and
actor periods, playback speed, hold duration, feedforward, balance-feedback
setting, controller mode and FK-refresh convention. Changed timing, actuators
or observations require explicit transfer/retraining and new evaluation. A
successful tensor load alone is not compatibility evidence.

Each current network is specialized to one family and its reference. The four
reference clips have different initial guards and stances. Accepted isolated
evaluation means one initial reset followed by an uninterrupted attack and
recovery. It does not authorize resetting between punches inside a continuous
game or switching networks from an arbitrary posture. A transition/command
interface still needs to be implemented and validated for continuous play.

## What the G1 integration still needs

Read-only inspection confirms that `robust_sav`'s standing, stance and snapshot
contracts use **5 ms physics, decimation 4 and 20 ms control**. See
`robust_sav/src/boxing/snapshots/artifacts.py` and
`robust_sav/src/boxing/snapshots/restoration.py`. The new human clock matches
that required cadence.

Matching clocks does not make the other global simulator settings identical:

| Global setting | Current human tracker | Recorded G1 native recipe |
|---|---|---|
| Integrator | implicitfast | implicitfast |
| Friction cone | **Elliptic** | **Pyramidal** |
| Solver | Newton | Newton |
| Solver iterations | 30 | 10 |
| Line-search iterations | 15 | 20 |
| Tolerance | 1e-7 | 1e-8 |

The G1 values come from the checked-in
`robust_sav/configs/boxing/standing/native_recipe.json`; the exact accepted G1
artifact/configuration must be verified at integration time. A shared MuJoCo
model has one global cone and solver configuration. Preserve the accepted G1
configuration first, then qualify or adapt the human under that shared
configuration. The current human support-force decoder assumes elliptic
constraints and also needs adjustment if using a pyramidal cone. The present
5 ms claim does not cover that unperformed solver-configuration comparison.

Integration also requires entity-local observations/actions, independent
state/reset ownership, suitable contact/constraint capacity, and tests showing
that adding the human does not alter the accepted G1 action/observation
contract. Current human references target one fixed head center; moving-target
conditioning, perception/latency, punch transitions and adversarial choices
remain separate work. Analytic glove/head distance checks and actual impact
response must be distinguished when configuring inter-body collisions.

Before accepting a packaged tracker or shared-scene adapter, recompile its
runtime XML and compare body mass/inertia, geometry, joint ranges, armature,
passive damping and actuator output limits with the frozen source. Also verify
controller/reference hashes, actual simulator-clock records, force caps,
zero root/external assistance and complete guard-start evaluation results.
These checks preserve the demonstrated physical controller while making the
remaining integration changes explicit.

The [final independent runtime review](../outputs/five-ms-release-independent-audit/README.md)
checks the selected four learned/baseline pairs separately from their full
64-world acceptance audits. Recompiled body properties and original strength
caps match exactly. Actual force satisfies the recorded native law to within
0.0000301 Nm, and the reconstructed zero-residual baseline drive matches
bit-for-bit in the nominal and last bank worlds for every family. These checks
support the isolated controller contract; they do not remove the shared-scene
solver and contact-integration work described above.
