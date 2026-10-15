# Scalar-body references for the 5 ms learning milestone

15 October 2026. Work remains confined to `human-punching`; source repositories and the accepted historical artifacts are read only.

## Purpose and scope

The earlier learned milestone used one scalar-body jab at 0.6× source speed. This stage extends the same bounded full-body retargeting pipeline to the jab, cross, left hook and right hook, and initially attempts each at 1.0× source speed. These are **offline geometric targets**, not evidence that a policy can physically execute them. The independent 5 ms rollout evaluations determine physical acceptance.

The original approximately 1.77 m, 75 kg, 48-hinge SOMA23-derived body supplies exact forward kinematics and joint limits. Changing actuator implementation for 5 ms integration does not require changing the reference coordinates when articulation, geometry and inertial properties remain identical. The reference manifest records its original XML hash; the training record must separately identify the actual simulation XML and controller.

## Sources and declared conditioning

| Family | Source run | Active hand | Existing source modifications |
|---|---|---|---|
| Jab | `physical-5890952` | Left | Smoothed, planted soles, root/leg support IK, re-aimed at the head |
| Cross | `physical-5890924` | Right | Smoothed, planted soles, support IK, re-aimed |
| Left hook | `physical-5890924` | Left | Smoothed, planted soles, support IK, re-aimed |
| Right hook | `physical-5891006` | Right | Same conditioning plus a 12 cm forward stance offset |

The prior cross/hook controllers used 0.3× playback **at runtime**. Their saved source timestamps retain the source clock, so `--speed 1` here does not perpetuate that slowdown. These are already conditioned source motions; “source speed” describes the timestamp multiplier and does not mean untouched motion capture or independently verified boxing realism. Per-reference manifests preserve the conditioning fields and the original capture-candidate identity.

The four families currently have different initial guards and stance positions, inherited from their source clips. They are isolated training tasks, not a sequence that can switch freely without a transition controller. No claim of continuous mixed-family combinations follows from this reference bank.

## Retargeting and timing

[scalar_reference.py](../src/human_punching/scalar_reference.py) solves bounded least squares using compiled MuJoCo position/orientation Jacobians. It preserves both glove targets, plants both soles, applies torso/root and elbow priors, softly centers the COM, and discourages penetration of selected self-collision pairs. The initial guard receives a longer solve; quiet holds reuse the exact same solved pose rather than allowing null-space drift.

The default complete cycle consists of a 0.5 s initial guard, the source crop at the requested speed, a 1.0 s quintic return blend, and a 0.6 s final guard. **Increasing punch playback speed does not shorten the inserted recovery or quiet holds.** The final configuration exactly equals the initial configuration. Velocities use MuJoCo tangent-space differences, including the free-root quaternion.

Each archive exports the family, active hand, source speed, explicit phase samples, launch time, return start, guard-hold start, both glove centers, first contact and a declared attack window. The stricter attack window starts at playback launch and ends 150 ms after closest active-glove approach, capped at source playback end. This is a geometry-derived follow-through allowance, **not a reviewed outward-phase label from capture**; `attack_window_reviewed=False` records that limitation. A coarse source-playback window is retained separately for provenance. Recovery-only contact must not satisfy the attack gate.

The manifests audit joint limits, maximum step/velocity/acceleration, guard-boundary steps, endpoint equality, sole drift/height, glove target error, contact and selected self-penetration. These diagnostics expose geometric solver discontinuities; they are not acceleration/torque feasibility certificates.

## Execution and reproduction

From the project directory:

```bash
# Four CPU jobs, one per family; native MuJoCo IK, no GPU needed.
sbatch scripts/retarget_scalar_punches.sbatch

# A single reference with a different playback factor and fixed recovery time.
.venv/bin/python -B -m human_punching.scalar_reference \
  --punch cross --speed 1 --return-duration 1 \
  --output outputs/my-cross-reference

.venv/bin/python -B -m pytest tests/test_scalar_reference.py -q
```

The first CPU array, `5931576`, failed before Python because node `d-19-14-4` returned `No such device` while creating a shared output directory. Array `5931593` completed its IK work but exposed a metadata bug: the shared first-contact API requires two glove channels; computing active-only contact must supply the active channel twice. That annotation was corrected without changing the physical contact contract. A later array, `5931623`, stalled similarly on node `d-19-13-3` and was deliberately cancelled. Successful replacement arrays ran on `d-18-4-4`. These are infrastructure/annotation failures, not evidence of dynamic infeasibility.

## Conditioning failures and fixes

The unmodified scalar transfer, array `5931598`, produced a clean source-speed jab. Cross and hook transfers had larger angular speed and acceleration demands. The right hook also overlapped its own gloves by 6.96 mm during its preparatory motion. The source body had not enforced this hand/hand collision constraint.

A stronger leg-posture prior was tried in array `5931614`. Although it reduced some support motion, it worsened the cross's peak joint velocity from 13.64 to 26.41 rad/s by changing the redundant IK solution. That candidate was rejected. Increasing a regularization weight is not automatically a smoother trajectory.

The accepted right-hook correction changes only the **inactive glove target**, pushing it smoothly away from the active glove when the two targets approach within 0.17 m. Its largest target change is 28.63 mm. The active punch target, model radii, strength and joint limits are unchanged. The 0.17 m self-clearance target is distinct from the glove/G1 head contact contract of 0.14 m.

Cross and both hooks then received Gaussian pose smoothing with a **25 ms standard deviation**, without changing timestamps. Scalar angles are averaged within their existing bounds; root rotation is smoothed in a local rotation-vector chart and reconstructed as a unit quaternion. The inserted quiet guards remain exactly fixed. This reduced angular accelerations, but direct coordinate smoothing introduced 8–13 mm of sole motion because forward kinematics is nonlinear. Array `5931630` records that intermediate result.

The final pass, array `5931641`, re-solves the sole constraints while retaining the **smoothed glove path** and a strong local posture prior. Sole drift returned below 0.25 mm while preserving the acceleration improvement. This does not pull the hands back to the original unsmoothed path. Maximum glove-path changes from the earlier conditioned targets are 25.60 mm for cross, 49.24 mm for left hook, and 36.29 mm for right hook. These changes are declared motion conditioning, not tracking error by the learned policy.

## Accepted geometric bank

The immutable bank is [scalar-four-reference-v1](../outputs/scalar-four-reference-v1/manifest.json). The jab is copied byte-for-byte from `5931598`; the other families are copied from `5931641`. Each includes the exact reference-generator snapshot, source hashes, conditioning parameters and metadata. All four use **1.0× source playback**, a 0.5 s initial quiet guard, a 1.0 s inserted return and a 0.6 s final quiet guard.

| Family | Complete reference duration | First contact after reset | Contact after playback launch | Closest active glove/head centers | Peak joint speed | Peak joint acceleration | Maximum sole drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jab | 3.833 s | 1.1506 s | 0.6506 s | 0.10118 m | 5.07 rad/s | 102.1 rad/s² | 0.250 mm |
| Cross | 3.800 s | 1.0953 s | 0.5953 s | 0.12763 m | 10.48 rad/s | 168.6 rad/s² | 0.084 mm |
| Left hook | 3.950 s | 1.3527 s | 0.8527 s | 0.10537 m | 13.20 rad/s | 236.2 rad/s² | 0.069 mm |
| Right hook | 3.908 s | 1.2239 s | 0.7239 s | 0.07121 m | 10.19 rad/s | 293.7 rad/s² | 0.210 mm |

All four pass finite-state/unit-quaternion checks, scalar joint limits, selected self-collision checks, exact initial/final configuration equality, zero endpoint velocity, planted-sole checks, and sampled active-glove contact within the declared attack window. The inactive glove remains outside the head contact envelope. The cross has the shallowest contact margin and may require especially accurate physical tracking.

Compared with unsmoothed transfer, peak acceleration decreases from 403.7 to 168.6 rad/s² for cross, 480.2 to 236.2 for left hook, and 464.6 to 293.7 for right hook. These remain kinematic derivative measurements, not evidence that the chosen strength bounds can realize those accelerations. Four targeted regression tests pass, including tangent-space free-root derivatives, limit-aware IK recovery, timing independence of return duration, and quaternion/bounds/quiet-guard preservation during smoothing. Ruff passes for the changed source and test files.

No physical execution is asserted by this reference-generation report. Separate 5 ms learned rollouts must pass balance, effort, active attack contact, final supported guard, tracking and timing checks. The existing 2 ms jab checkpoint and these references alone do not satisfy that requirement.

## Why the source-speed hooks needed another timing pass

Subsequent learning attempts and an independent inverse-dynamics audit exposed a limitation that planted feet and COM geometry had not detected. The source-speed hook trajectories asked the weak proximal shoulder joints to transmit large inertial loads. The full-frame audit found peak upper-body torque demands of approximately **2.78× the declared limit for left hook and 2.21× for right hook**. Static COM projections remained approximately 10 cm inside the support polygon. Static balance was therefore not the principal issue.

The largest loads were not simply spikes in the shoulder joint's own acceleration. For the left hook's shoulder-Y overload, gravity alone contributed roughly 9.9 Nm, whereas gravity plus velocity-dependent terms contributed roughly 74.8 Nm before acceleration raised total demand to approximately 90.9 Nm. The existing shoulder cap is 35 Nm. Changing the redundant shoulder angle can shift some demand but cannot eliminate the transmitted momentum of the entire moving arm. These numbers are model-based reference requirements, not measured forces from a trained policy.

Two stronger shoulder-posture regularizers were tried in arrays `5931729` and `5931733`. They reduced the left-hook peak by roughly 15% but still required about 2.37× the limit; the right hook remained about 2.27×. Array `5931749` additionally anchored pelvis orientation near guard and relaxed source wrist-orientation tracking. Its peaks remained about 2.38×/2.30×. These candidates were not adopted. All model masses, inertias, torque caps and joint limits remained unchanged.

The next candidate changes **when the same configuration path is traversed**. [retime_punch_reference.py](../scripts/retime_punch_reference.py) applies an explicit monotone PCHIP phase warp. It borrows 250 ms from the inserted initial guard, reducing that quiet hold from 0.5 to 0.25 s, and spends more time on the high-load swing. Complete cycle durations, contact poses and the inserted final recovery/guard interval are retained. The first contact after reset remains near the prior time. Velocities and accelerations are recomputed on the new clock, including the timing transition effects.

This is **not uniform source-speed 1.0× playback**. The NPZ deliberately omits the scalar `source_speed`, records the nominal pre-warp multiplier separately, and exposes the complete phase-warp knots and local speed range. Runtime `--speed 1` means playing this modified archive at its own recorded clock. It does not restore the original capture timing.

The immutable [retimed hook bank](../outputs/scalar-hook-retime-v1/manifest.json) contains:

| Reference | Local source-rate range during playback | First contact after reset | Maximum joint speed / acceleration | Sole drift | Inverse-dynamics peak fraction of original torque cap |
|---|---:|---:|---:|---:|---:|
| Left hook | 0.471–1.377× | 1.3544 s | 8.17 rad/s / 102.6 rad/s² | 0.80 mm | 1.045× |
| Right hook | 0.491–1.374× | 1.2014 s | 5.56 rad/s / 85.1 rad/s² | 1.25 mm | 0.762× |

The [full-frame inverse-dynamics screen](../outputs/reference-dynamics-5931826/summary.json) finds all 236 right-hook reference frames compatible with its ideal foot-force constraints and original torque limits. Left hook is substantially improved but retains three frames exceeding the shoulder cap by at most 4.48%, plus three frames for which the exact prescribed root wrench is infeasible in the screening contact model. A learned controller may alter the pose and timing slightly, but it must demonstrate success under the unchanged physical gates; the screen is not a reason to relax those gates.

The retimed references retain exact final/initial guard equality, legal joint coordinates, selected self-collision clearance and head contact. Small extra sole drift comes from configuration-space interpolation between samples and is reported explicitly. [Left-hook reference preview](../outputs/retimed-hook-render-5931841/left_hook/left_hook-rollout.mp4) and [right-hook reference preview](../outputs/retimed-hook-render-5931841/right_hook/right_hook-rollout.mp4) are kinematic previews, not learned-motion evidence. Independent learned evaluations are reported separately.

Generation jobs were `5931790` and `5931791`; diagnostic job `5931826` screened the resulting references. Focused reference, timing and release-provenance tests cover the retained changes. [Release packaging instructions](four-punch-packaging.md) distinguish physical pilot acceptance, precision targets and explicitly admitted nonuniform timing.

## Independent review of the retimed attack windows

The frozen archives retain `attack_window_reviewed=false`: their windows were derived from launch and closest approach, not manually annotated against the original capture. A separate [derived-window review](../outputs/hook-window-review-5931886/summary.json) now checks their intended geometric meaning. It recomputes forward kinematics at 5 ms sample spacing along the frozen configuration trajectory and examines active/inactive glove distances, radial velocity, all contact episodes and the later inserted return. These are kinematic samples, not a physics simulation.

| Reference | Derived attack window | Only active contact episode | Inbound radial speed at first contact | Minimum active distance during inserted return |
|---|---:|---:|---:|---:|
| Left hook | 0.2500–1.5333 s | 1.35436–1.40043 s | −2.35 m/s | 0.564 m |
| Right hook | 0.2500–1.3807 s | 1.20140–1.26517 s | −4.03 m/s | 0.490 m |

Both derived windows pass this scoped review. Each contains a single approach/contact/withdrawal episode by the declared active hand; the inactive glove never enters the 0.14 m contact envelope. By the window end, the active glove is withdrawing, and it remains beyond the 0.16 m safety distance for the rest of the cycle. Thus the accepted reference contact belongs to the intended strike, not a later guard-recovery pass. Dense versus archived first-contact calculations differ by only 0.068 ms for left hook and 0.010 ms for right hook.

![Left-hook distance and radial-speed review](../outputs/hook-window-review-5931886/left_hook/distance-and-radial-speed.png)

![Right-hook distance and radial-speed review](../outputs/hook-window-review-5931886/right_hook/distance-and-radial-speed.png)

Blue shading denotes the derived attack window; gray shading denotes the inserted return to guard. The red vertical line marks first contact. These plots and [the review script](../scripts/review_reference_windows.py) support `derived_window_review_pass=true` in a separate report without changing the frozen reference hashes or claiming source-capture annotation review. They do not establish learned rollout acceptance, impact-response physics or the tighter learned contact-timing target.

The two linked kinematic preview videos were also completely decoded by FFmpeg without errors; the per-video `video-validation.json` files preserve those results. Closest-approach stills were visually inspected. Review job `5931877` initially stopped because the rendering environment lacked Matplotlib; replacement CPU job `5931886` used the existing physics environment and completed the numerical review and video decoding.
