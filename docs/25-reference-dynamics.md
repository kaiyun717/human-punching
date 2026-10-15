# Reference dynamics: why the hooks need conditioning

15 October 2026. This investigation follows the numerically stable 5 ms
controller in [report 20](20-5ms-controller-investigation.md) and the geometric
reference bank in [report 21](21-four-punch-references.md). It helps explain
why successful retargeting and planted soles do not guarantee physical tracking.
The body, joint ranges and original torque limits remain unchanged.

## Diagnostic method and its limits

[reference_dynamics_probe.py](../scripts/reference_dynamics_probe.py) evaluates
the ideal generalized force needed to track each sampled reference exactly:

`required_force = M(q) * reference_acceleration + bias(q, reference_velocity)`

Acceleration is a finite difference of the saved generalized velocities. A
linear program allocates ground forces at eight sole corners to match the six
unactuated root-force/moment equations. It minimizes the largest fraction of
the original actuator torque limits needed by the remaining 48 joint equations.
Normal forces must be nonnegative. Tangential forces obey
`abs(Fx) + abs(Fy) <= 0.9 * Fz`, a conservative diamond inside the circular
friction cone. This is an offline calculation; no forces are applied to a
simulation and no model parameters change.

This is a **conditional necessary-condition diagnostic**, not a proof that a
punch is feasible or impossible. It assumes exact reference motion and the
specified simultaneous foot support. It omits contact compliance, consistency
of foot accelerations, passive joint-stop/self-contact reactions and force-rate
limits. The friction approximation can reject support that a different cone
would allow. Conversely a feasible allocation does not show that a feedback
controller can realize it. A learned policy can change the body trajectory
while preserving the punch objective.

An allocation failure means that the assumed ground forces cannot realize the
reference root wrench at that sample. A torque fraction above 1 means the
assumed exact reference needs more motor strength than available even under
the best feasible force allocation. For upper-body joints, the sole Jacobian
columns are zero: ground-force redistribution cannot directly reduce their
required generalized torque. The updated diagnostic reports this demand even
at samples where the root-wrench allocation fails. Other passive reactions are
still outside that calculation.

## Original reference bank

Job **5931716** sampled all four families every third reference frame. Job
**5931727** repeated both hooks at every reference frame. The clock multiplier
0.6 scales reference velocities by 0.6 and accelerations by 0.36 at the same
poses; it also stretches inserted recovery in this diagnostic. It is not a
claim that an unchanged-duration, uniformly slower cycle exists.

| Family | Reference clock multiplier | Peak minimum torque fraction on support-feasible samples | Root-support allocation failures | Sampling |
|---|---:|---:|---:|---|
| Jab | 1.0 | 0.838 | 4 / 78 | Every third frame |
| Jab | 0.6 | 0.449 | 0 / 78 | Every third frame |
| Cross | 1.0 | 0.555 | 8 / 77 | Every third frame |
| Cross | 0.6 | 0.561 | 2 / 77 | Every third frame |
| Left hook | 1.0 | **2.783** | 14 / 238 | Every frame |
| Left hook | 0.6 | **1.181** | 0 / 238 | Every frame |
| Right hook | 1.0 | **2.207** | 10 / 236 | Every frame |
| Right hook | 0.6 | **0.988** | 3 / 236 | Every frame |

Low torque fractions in this table must not be read independently of the
support failures. For example, the cross's feasible samples need modest motor
effort, yet parts of its prescribed root motion still fail the support test.

All four references have approximately **9.6–10.9 cm minimum static COM margin**
inside the combined sole polygon. Thus the main problem is not simply the COM
moving outside the feet. Dynamic acceleration and angular-momentum demand can
be inconsistent with foot support even when this static margin looks healthy.

![Hook demand at original and slower reference clocks](../outputs/reference-dynamics-5931727/hook-demand.png)

Both curves use reference phase on the horizontal axis so that the same poses
align; their real elapsed times differ. Gaps and crosses identify samples with
no feasible assumed ground-force allocation. [Vector figure](../outputs/reference-dynamics-5931727/hook-demand.svg).

## The overloaded motion is broader than one bad IK sample

The original left hook peaks at **97.4 Nm** on `LeftShoulder_y` near 1.133 s,
against its 35 Nm limit. The right hook reaches approximately **77.3 Nm** on
`RightShoulder_z` near 1.164 s, also against 35 Nm. Ground-independent
upper-body overload occurs at 21 left-hook frames and 17 right-hook frames,
roughly 0.35 s and 0.28 s of sampled time. These are sustained parts of the
swing, not isolated single-frame glitches.

At a representative left-shoulder-y peak near 1.15 s, the required torque is
90.9 Nm in magnitude. Gravity alone contributes only 9.9 Nm; gravity plus
velocity-dependent bias contributes 74.8 Nm, with the remaining 16.1 Nm coming
from acceleration. The right-shoulder-y peak has the same pattern. In contrast,
the right-shoulder-z peak is mainly acceleration-driven. Fast coupled arm,
torso and root motion therefore matters; reducing only one shoulder angle's
acceleration cannot remove all of the transmitted demand.

These particular shoulder-y peaks lie inside the anatomical joint ranges, so
passive stop reaction does not directly explain them. Detailed decomposition
is saved in [overload_decomposition.json](../outputs/reference-dynamics-5931716/overload_decomposition.json).

## Conditioning attempts without increasing strength

The reference agent tested additional posture regularization while preserving
the original bank. All comparisons below use every reference frame at the
candidate's original clock:

| Candidate | Change | Left-hook peak fraction | Right-hook peak fraction | Finding |
|---|---|---:|---:|---|
| Original v1 | Existing 25 ms smoothing and sole reprojection | 2.783 | 2.209* | High dynamic demand |
| 5931729 | Shoulder guard-posture prior 0.3 | 2.373 | 2.273 | Helps left modestly; slightly worsens right |
| 5931733 | Shoulder guard-posture prior 1.0 | 2.366 | 2.269 | Stronger prior provides little further benefit |
| 5931749 | Root guard-rotation prior 3.0, shoulder prior 0.3, relaxed wrist-orientation weight 0.05 | 2.384 | 2.303 | Does not resolve the overload |

*The 2.209 value includes the largest ground-independent upper-body demand at
a root-support-infeasible frame; the corresponding support-feasible LP maximum
is 2.207. This distinction is preserved in the raw results.

The shoulder-prior candidates become much closer to feasible demand at a 0.6
clock multiplier: about 1.05 times the left-hook cap and 1.03 times the right,
with no root-support allocation failures. This supports a slower training
curriculum, while still requiring actual rollout validation.

A further candidate uses **declared nonuniform retiming**: shorten some initial
preparation and spend more time on the heavily loaded swing while retaining
the overall cycle duration. Its local phase rate differs from the source; it
must not be labelled untouched uniform 1.0× playback. Reference velocities,
accelerations, attack annotations and contact times must all be recomputed on
the new clock, and the added phase acceleration must be included in this
diagnostic before recommending training.

The first retimed bank, `outputs/scalar-hook-retime-v1`, borrows 0.25 s from the
initial quiet interval. Its local source phase rates vary approximately
0.47–1.38 for the left hook and 0.49–1.37 for the right. All poses, velocities,
accelerations and phase/contact annotations are resampled on the new clock.
The source path and complete-cycle duration are retained, with small
interpolation-induced sole movement recorded by the reference audit.

Full-frame diagnostic **5931826** shows a substantial improvement:

| Retimed candidate | Peak required strength fraction | Root-support failures | Interpretation |
|---|---:|---:|---|
| Right hook | **0.7622** | **0 / 236** | All sampled ideal allocations satisfy the original strength bounds; suitable for a physical learning trial |
| Left hook | **1.0448** | **3 / 238** | Much closer, but three shoulder-y samples still exceed 35 Nm and three separate samples lack the assumed root support |

![Original and retimed hook demand](../outputs/reference-dynamics-5931826/retiming-demand.png)

The comparison uses actual elapsed time, so the peaks move as well as shrink.
The total cycle duration remains unchanged. Crosses mark support-allocation
failures, and gaps are not zero motor demand.
[Vector figure](../outputs/reference-dynamics-5931826/retiming-demand.svg).

The remaining left shoulder-y excess occurs at 0.883–0.917 s and peaks at
36.57 Nm; support failures occur at 0.25, 1.417 and 1.8 s. These intervals guide
a smaller follow-up timing adjustment rather than another broad change to the
body. The right-hook result is a useful necessary-condition check, **not yet
evidence that its learned controller succeeds**. Actual 5 ms training and
independent uninterrupted-cycle evaluation remain required.

## Reproducibility and evidence

Inside a CPU Slurm allocation, from the project directory:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B scripts/reference_dynamics_probe.py \
  --output outputs/my-reference-dynamics \
  --families left_hook right_hook --stride 1 --speeds 1 .6
```

The script saves its own source, reference hashes, per-frame allocations,
summary counts and explicit assumptions. Three tests verify asymmetric torque
limits and a known impossible-friction case; all pass. The relevant evidence is:

- [All-family diagnostic](../outputs/reference-dynamics-5931716/summary.json)
  and [method notes](../outputs/reference-dynamics-5931716/README.md).
- [Full-frame original hooks](../outputs/reference-dynamics-5931727/summary.json).
- [Shoulder prior 0.3](../outputs/reference-dynamics-5931744-5931729/summary.json)
  and [shoulder prior 1.0](../outputs/reference-dynamics-5931744-5931733/summary.json).
- [Root/wrist conditioning](../outputs/reference-dynamics-5931780/summary.json).
- [Nonuniform retiming, every frame](../outputs/reference-dynamics-5931826/summary.json).
- [Upper-body demand including support-infeasible frames](../outputs/reference-dynamics-5931716/upperbody_demand_candidates.json).

The outcome is guidance for reference conditioning and learning, not permission
to relax strength, joint-limit, balance or contact gates. Final acceptance
remains a complete learned rollout at 5 ms physics with the original bounds.

## Subsequent physical validation

The retimed candidates were then trained with the unchanged bounded 5 ms
controller. Early checkpoint failures were specific: neck/shoulder return
posture delayed guard recovery, and the left hook briefly hyperextended the
right knee. Their [recorded-rollout diagnosis](../outputs/diagnostic-hooks-5931865-66/README.md)
shows why targeted posture/limit learning was appropriate; neither higher
strength nor relaxed acceptance gates was needed.

The selected retimed hooks subsequently passed all physical gates and the
stricter tracking/timing targets in fresh 64-trial banks: [left hook](../outputs/punch-5ms-5931934/independent/summary.json)
and [right hook](../outputs/punch-5ms-5931918/independent/summary.json).
Matched zero-residual controllers passed 0/64 complete trials for both. The
small left-reference overload therefore did not prevent a successful learned
motion that could depart slightly from that exact trajectory, illustrating
the diagnostic's stated limits. The reference timing remains explicitly
nonuniform. These tests cover small initial-state perturbations of the same
fixed-target clips; they do not establish general motion or moving-target
robustness.
