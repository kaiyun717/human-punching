# Finite reference bank and full-cycle commitment

Date: 11 October 2026. This report covers an isolated `human-punching` reference adapter and, in the final section, a separate continuous physical two-jab proof. Neither runs the G1 policy or trains an adversary.

## Implemented contract

[cycle_bank.py](../src/human_punching/cycle_bank.py) loads the 12 geometrically accepted references described in [11-reference-contact.md](11-reference-contact.md). It checks the model XML hash, each human-trajectory hash, joint/state dimensions, normalized quaternions, monotonic clocks, finite states, and accepted contact metadata. Each bank entry exposes family, source duration, frozen target center, payload hash, and measured diagnostic contact time.

A `BoundaryCyclePlayer` enforces the following sequence:

| State | Allowed action | Result |
|---|---|---|
| `boundary` | Select a legal variant and provide matching current position/velocity | Freeze the variant and launch head center; enter `active`. |
| `active` | Advance the committed reference at the configured control timestep | Sample the same reference throughout the cycle. New plans are rejected. |
| `waiting_guard` | Use an external transition/controller; acknowledge a measured guard matching the next reference | Return to `boundary`. No automatic jump or next-punch launch occurs. |

The launch plan is an immutable dataclass. Its head snapshot is a tuple copied from the caller, so subsequently changing the caller's head array cannot redirect the punch. The `advance` method has no live head-target argument. A completed **reference clock** is explicitly different from physical punch completion: the terminal sample carries `requires_guard_controller=True`.

At a boundary, plan masks can filter family, maximum source duration, and a window of measured diagnostic contact times. They also check that the launch head matches the pretargeted reference head. At present all references target the same standing G1 placement. A different head location produces an empty mask and requires retargeting; the adapter does not silently shift a precomputed trajectory or begin homing. All masks are false while active or waiting for guard acknowledgment.

The timing-window filter does not synthesize a requested TTC. It filters the existing first-contact measurements from clip start. Source durations and timing are retained, including guard padding whose phase annotation remains provisional.

## Control-clock sampling

The default control interval is 0.02 s. Positions interpolate between saved poses using MuJoCo's tangent-space differentiation and integration, including shortest-arc quaternion rotation. Exported reference velocities are linearly interpolated as feedforward values; they are approximate desired velocities, not an exact derivative certificate for the pose interpolant.

At the first control tick at or after the source endpoint, the adapter returns the final source pose and marks the cycle complete. World time remains on the control clock; the reference elapsed time is capped at the clip duration:

| Family | Source duration | Completion control time at 50 Hz |
|---|---:|---:|
| Jab | 1.733333 s | 1.74 s |
| Cross | 1.700000 s | 1.70 s |
| Left hook | 1.850000 s | 1.86 s |
| Right hook | 1.808333 s | 1.82 s |

No final pose is teleported to another clip. Sampling poses is not physical stepping, and these control-clock results do not establish a tracking controller's recovery time.

## Guard admission uses separate quantities and units

The adapter compares the supplied current state with the selected clip's initial state. `guard_errors()` reports root and joint errors separately; ball-joint angle errors use the norm of their rotation vectors. Every articulated joint participates in the maximum joint error.

| Quantity | Default admission tolerance |
|---|---:|
| Root position error | 0.02 m |
| Root orientation error | 0.02 rad |
| Maximum articulated-joint angle error | 0.02 rad |
| Root linear-velocity error | 0.10 m/s |
| Root angular-velocity error | 0.10 rad/s |
| Maximum articulated-joint angular-velocity error | 0.10 rad/s |
| Slide-joint position/velocity error, if present | 0.02 m / 0.10 m/s |

These are explicit engineering defaults for avoiding discontinuous reference admission, not empirical physiological limits or a physical guard-recovery certificate. The physical controller must additionally establish appropriate foot support, balance, actuator feasibility, and attack eligibility. A caller must supply actual controller state when using the adapter with physics; substituting reference arrays would only demonstrate kinematic playback.

The [guard audit](../outputs/contact-targeting/cycle-bank-audit.json) found that **none of the 16 base-family endpoint-to-start pairs passes admission**, including repetition of the same family. Example same-family discontinuities are:

| Repeated reference | Root-position mismatch | Root-orientation mismatch | Maximum joint-angle mismatch | Maximum joint angular-velocity mismatch |
|---|---:|---:|---:|---:|
| Jab → jab | 0.04285 m | 0.12919 rad | 0.34774 rad | 3.88956 rad/s |
| Cross → cross | 0.19620 m | 0.24400 rad | 0.24564 rad | 2.30065 rad/s |
| Left hook → left hook | 0.06869 m | 0.24082 rad | 0.49585 rad | 9.35831 rad/s |
| Right hook → right hook | 0.02906 m | 0.18991 rad | 0.20540 rad | 2.22744 rad/s |

Preserving recorded endpoints during IK therefore did not make the clips stationary, recovered guards or freely concatenable cycles. The source crop annotations and return phases need further refinement before continuous physical sequence acceptance.

## Optional bridge is explicitly kinematic

`kinematic_guard_bridge()` generates a quintic pose transition with normalized quaternion interpolation and zero endpoint velocities. This is useful for inspecting the gap between two guard configurations. It does not constrain support feet, avoid self-collision, respect joint/torque limits, or solve balance. Its zero terminal velocity can also differ from the next source clip's initial velocity; the ordinary guard-admission check still applies.

The bridge returns `requires_physical_guard_controller=True`. This bank adapter and its generic kinematic bridge do not by themselves establish physically continuous execution, and no bridge is automatically executed or admitted by the cycle player. The separate conditioned physical proof below measures execution and guard recovery explicitly.

## Validation and usage

**Eight tests pass** in [test_cycle_bank.py](../tests/test_cycle_bank.py): immutable launch snapshot, mid-cycle plan rejection, legal masks, dimensional guard metrics, position/velocity admission, quaternion interpolation, explicit completion/guard states, payload tamper rejection, and an actual generated-bank cycle. Ruff passes. The all-family control-clock/guard measurements are recorded in [cycle-bank-audit.json](../outputs/contact-targeting/cycle-bank-audit.json).

```python
from human_punching.cycle_bank import CycleBank, BoundaryCyclePlayer

bank = CycleBank("outputs/contact-targeting/manifest.json")
player = BoundaryCyclePlayer(bank, control_dt_s=0.02)
mask = player.valid_plan_mask(measured_head_center, family="jab")
# Choose one admitted variant at the boundary, then pass actual controller state.
reference = player.start(
    chosen_variant, measured_head_center, launch_time_s,
    current_qpos=measured_human_qpos, current_qvel=measured_human_qvel,
)
# Each control tick: reference = player.advance(); send it to a separate tracker.
# On completion, run a guard controller before player.acknowledge_guard(...).
```

The adapter is a foundation for fixed/random finite selection and later boundary-only adversarial choice. It does not implement the RSAV learner/replay changes, Bellman commitment masks, physical sequence controller, or adversarial training.


## Continuous physical two-jab proof

A separate run now demonstrates **two consecutive physically executed jabs and return to guard**, with one initial reset and no state overwrites between punches. It reuses the successful conditioned jab and physical scene from `outputs/physical-5890912`, rather than directly concatenating the unconditioned source bank discussed above.

[Watch the continuous two-jab video](../outputs/physical-sequence-5890969/two_jabs.mp4) · [Still](../outputs/physical-sequence-5890969/two_jabs.png) · [Metrics](../outputs/physical-sequence-5890969/metrics.json)

The human has a free, unactuated root. The same bounded quaternion PD plus contact-aware balance feedforward controller runs at 2 ms physics steps, with gain scale 1.5 and reference playback speed 0.6. G1 is fixed in the scene, the head snapshot stays fixed in world coordinates, and glove/head entry is an analytic event with no impact response. No learned policy is involved.

The sequence runs:

1. Execute the full jab reference at 0.6 speed.
2. Hold its final target for approximately 0.4 s.
3. Track a 1.5 s quintic generalized-coordinate return to the initial guard.
4. Hold the initial guard until actual measured pose, velocities, both-foot contacts, glove clearance and support limits pass continuously for at least 0.10 s. Reject the next launch after a 3 s timeout.
5. Launch the second jab only after that measured admission, then repeat the return and guard check.

Only controller targets interpolate during the bridge. The actual state evolves through `mj_step`; it is not assigned to either endpoint. Guard comparisons use the separate dimensional tolerances listed above, with a stationary target guard velocity of zero. This is a dedicated same-jab sequence proof; it does not establish arbitrary family transitions or all 12 bank variants.

### Measured results

Slurm job **5890969** completed with exit code `0:0` in 11 seconds on a Tesla V100 (`d-8-4-1`). It simulated **9.79 s / 4,895 physics steps**, including both final guard recovery and the second launch at **4.898 s**.

| Metric | Result |
|---|---:|
| First jab: minimum active glove/head distance | 0.110467 m |
| Second jab: minimum active glove/head distance | 0.110762 m |
| First jab: measured time to contact from launch | 1.143171 s |
| Second jab: measured time to contact from launch | 1.144021 s |
| Both feet in contact | 99.9796% of sampled physics states |
| Maximum sole horizontal drift | 0.006698 m |
| Maximum floor penetration | 0.001795 m |
| Mean fraction of joints at torque saturation | 0% |
| Maximum joint torque norm / permitted norm | 0.683108 |
| Root actuator force and externally applied force | Exactly zero |

Contact was detected during both punch phases; no contact occurred during endpoint holds, return bridges, or guard waiting. The inactive glove remained outside the 0.14 m envelope. The complete sequence passed the explicit support, contact, torque, and recovered-guard gates.

At second launch, measured guard errors were **1.632 mm root position**, **0.001577 rad root orientation**, **0.002856 rad maximum joint angle**, **0.000617 m/s root linear velocity**, **0.002723 rad/s root angular velocity**, and **0.008295 rad/s maximum joint angular velocity**. The complete actual `qpos`/`qvel` snapshot is retained in the launch record. Across the immediately adjacent 2 ms samples at that launch, the glove centers moved only 9.29 and 9.72 micrometres; no reference-pose jump was inserted.

Evidence: [full rollout](../outputs/physical-sequence-5890969/two_jabs_rollout.npz), [independent validation](../outputs/physical-sequence-5890969/validation.json), [GPU runtime](../outputs/physical-sequence-5890969/gpu_runtime.json), [Slurm accounting](../outputs/physical-sequence-5890969/slurm-accounting.txt). The video passed full FFmpeg decoding, all saved states are finite, and both launch/recovery snapshots independently pass the declared guard thresholds.

Implementation: [sequence_cli.py](../src/human_punching/sequence_cli.py), with the existing physical controller and renderer reused. [test_sequence.py](../tests/test_sequence.py) additionally verifies a deliberately insufficient guard timeout prevents a second launch. That regression passes; Ruff and shell syntax checks pass.

```bash
# Run from human-punching. Uses the successful conditioned physical jab artifact.
sbatch scripts/physical_sequence_volta.sbatch

# CPU physics without rendering:
.venv/bin/python -m human_punching.sequence_cli --output outputs/physical-sequence-local
```

The 1.143–1.144 s contact delay is measured from **controller cycle start**, including guard padding. It is a result of the slowed reference, not the PDF’s separately annotated outward-launch TTC or a requested TTC guarantee. This proof has one morphology, one stationary target, one punch family, and one deterministic initialization. It establishes a useful classical-control baseline, not robustness to disturbances, broader sequence feasibility, learned adversarial selection, or a G1 avoidance-policy evaluation.
