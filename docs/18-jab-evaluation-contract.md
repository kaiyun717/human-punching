# Scalar jab pilot: independent evaluation and rendering

The first learned-controller milestone needs a recorded **guard → jab → recovered guard** physics trial. A plausible video or a successful offline reference is insufficient. `src/human_punching/jab_evaluation.py` independently recomputes the geometric metrics from saved simulator states and checks recorded actuation, applied forces, contact support, and reset provenance.

This is an engineering pilot acceptance contract. Its numerical tolerances are declared here and in `PilotThresholds`; they are not measured human population limits, an injury criterion, or a claim of 95% policy success.

## Required evidence

An actual rollout NPZ contains:

- `time_s` (aliases `time` or `t`), `qpos`, `qvel`, with one initial state and every subsequent physics step;
- `ctrl`, `qfrc_actuator`, `qfrc_applied`, and `xfrc_applied` at the same cadence;
- `support_contact`, two booleans per frame from actual left/right foot-floor support forces, rather than an FK proximity test;
- `state_reset_count=1`, `physics_step_count=number_of_saved_intervals`, and `physics_timestep_s` if the simulator overrides the XML timestep;
- `evaluation_kind`, such as `reference_pd` or `learned_policy`; model/reference/checkpoint provenance belongs in the run manifest and rollout where available;
- `reference_speed` when the controller changes playback timing relative to its saved reference.

The model is loaded from that run's immutable model snapshot. Joint names, joint addresses, actuator joint transmissions and gear ratios come from the compiled XML, rather than a fixed DOF order. A model has one unactuated free root and scalar hinge joints. Glove sites are `LeftHand_center` and `RightHand_center`; sole sites are `LeftFoot_sole` and `RightFoot_sole`.

Missing evidence produces a failed or unverified gate. It is not filled in using assumptions. The replay itself never steps physics and cannot independently prove that a producer did not overwrite states; the reset counter must be checked against the rollout implementation. A strict all-pass result therefore means that the recorded trial and its declared provenance satisfy the checks, not that an arbitrary NPZ can prove physical execution on its own.

## Pilot gates

| Check | Criterion |
|---|---|
| Physics provenance | One initial reset; saved interval count equals physics-step count; every recorded interval matches the explicit runtime timestep |
| Starts at reference guard | Root position ≤ 0.03 m, root orientation ≤ 0.10 rad, every scalar joint ≤ 0.12 rad from the reference initial state |
| No fall | Root height always ≥ 75% of reference initial height; pelvis tilt from initial upright ≤ 0.60 rad |
| Foot motion | Each sole's maximum 3D displacement from its initial position ≤ 0.03 m |
| Joint limits | Maximum excursion beyond a compiled scalar joint limit ≤ 0.02 rad, allowing a small soft-limit tolerance |
| Bounded control and torque | Recorded motor controls and actual generalized joint torques stay within compiled motor bounds, with 1e-6 numerical tolerance |
| No external assistance | Every recorded generalized applied force and body applied wrench is zero within 1e-9 |
| Unactuated root | Every recorded root actuator-force component is zero within 1e-9 |
| Attack-window contact | At least one actual saved glove center reaches head-center distance ≤ 0.14 m during the declared attack window |
| Reference tracking | Each glove's world-position RMS tracking error ≤ 0.10 m |
| Guard recovery | After contact, the trial ends with ≥ 0.20 s continuous satisfaction of every return condition below |

The return conditions are root position ≤ 0.03 m, root orientation ≤ 0.10 rad, scalar joint maximum error ≤ 0.12 rad, root linear speed ≤ 0.05 m/s, root angular speed ≤ 0.12 rad/s, every scalar joint speed ≤ 0.50 rad/s, both gloves within 0.03 m and 15° of their initial Chest-relative guard pose, both Chest-relative glove linear speeds ≤ 0.15 m/s, both gloves ≥ 0.16 m from the head, and recorded support at both feet. These are evaluated against the initial reference guard. Merely passing through guard once, remaining in initial guard, or ending with moving arms does not qualify. Chest-relative velocity is computed with body/site Jacobians as `R_chest.T @ (v_glove - v_chest - omega_chest × (p_glove - p_chest))`, so whole-torso rotation and translation are removed correctly. This pilot returns to one fixed initial guard; tracking a moving or newly selected guard is later work. The explicit glove gates were added before inspecting PPO results, not adjusted to admit a resulting controller.

Reference alignment uses linear interpolation for scalar positions and SLERP for the free-root orientation. The declared playback speed is applied. The final reference is held after its end. Glove orientation error and root tracking error are reported separately even though the initial strict pilot gates do not impose additional RMS thresholds on them.

Contact uses the established 0.06 m head / 0.08 m glove proxies. The first-contact clock is explicitly **elapsed time from guard reset**, not the PDF's outward-launch TTC. For this pilot, contact acceptance is restricted to reference-clock playback launch at 0.5 s through explicit return-blend start at 3.3888888888888893 s, divided by the controller playback speed. Raw any-cycle contact is reported separately, so a late return contact cannot substitute for an attack. This is a coarse window that still includes recovery present in the source clip; reviewed outward/follow-through annotations remain future work. Missing phase metadata fails the attack gate. Supply `--attack-window 0.5 3.3888888888888893` for copied references whose original manifest is not adjacent. The target is fixed and has no physical impact response. The 0.16 m glove-clearance predicate is only one part of robust-SAV safety.

The current MuJoCo-Warp step saves post-integration qpos/qvel, while cached forces and contacts describe the forward pass for the preceding integration interval. Therefore all geometric contact/pose metrics are recomputed by independent FK from saved qpos. Support flags are interpreted as actual positive foot-floor forces over the preceding interval. A counted recovery interval requires recovered geometric/velocity state at both saved endpoints and positive support at its right endpoint; an isolated recovered sample contributes zero dwell. Dwell-boundary resolution is one physics step (2 ms in these trials). No extra forward pass is inserted into learned-policy observations or stepping.

A single passing nominal trial is not a perturbation benchmark. Report each recorded perturbed trial separately and state the number tested before summarizing success fractions.

## Commands

From `human-punching`, evaluate without rendering:

```bash
.venv/bin/python -m human_punching.jab_evaluation \
  --model outputs/RUN/model/human.xml \
  --rollout outputs/RUN/learned/rollout_00.npz \
  --reference outputs/RUN/reference.npz \
  --output outputs/RUN/independent-evaluation \
  --attack-window 0.5 3.3888888888888893
```

If an older rollout omits its actual simulation timestep, add `--physics-timestep 0.002` after verifying that value against the run's controller schema. Do not infer success from the original XML's default timestep when the simulator overrides it.

Render a kinematic reference diagnostic:

```bash
sbatch scripts/render_jab_volta.sbatch \
  --rollout outputs/scalar-reference-v2/jab.npz --kind reference
```

Evaluate and render an actual learned trial:

```bash
sbatch scripts/render_jab_volta.sbatch \
  --model outputs/RUN/model/human.xml \
  --rollout outputs/RUN/learned/rollout_00.npz \
  --reference outputs/RUN/reference.npz --kind learned \
  --attack-window 0.5 3.3888888888888893
```

`--kind classical` labels a reference-PD baseline. `--scene-model` can supply a scene containing the same human and a baked fixed G1 for rendering; the original model still controls evaluation. The scene must preserve the exact human qpos/qvel order. `physical_scene.attach_static_g1` verifies that order while baking the fixture.

The video has oblique and side views, full-time and half-time replays, actual center distances, and explicit kinematic / learned / reference-PD labels. It replays recorded states without interpolation or state alteration. Renderer-only overrides enable physical geom group 3, make both gloves red against the light-blue body, and brighten the headlight. They do not modify the source XML, its hash, mass, inertia, contact geometry, or dynamics.

## Initial diagnostic, 12 October 2026

The corrected V2 reference independently passed its geometric checks: minimum left-glove distance 0.1011 m, sole drift 0.131 mm / 0.254 mm, no scalar joint-limit excursion, and exact final guard return. `outputs/scalar-reference-v2/independent-fk-audit/evaluation.json` correctly reports `pilot_pass=false`: this input is a kinematic reference, with no recorded actuator forces, physical foot support, or one-reset physics provenance. Its zero tracking error is a self-comparison and provides no controller evidence.

GPU previews `outputs/jab-render-5895244` and `outputs/jab-render-5895248` show V2 with a target sphere and with the fixed G1, respectively. These are early kinematic diagnostics. Later reference/controller artifacts and their independent evaluations supersede them; this section does not claim a learned result.

## First reference-PD baseline: run 5895253

The independent evaluator applied the strengthened Chest-relative guard gates before inspecting a PPO result. The nominal unlearned baseline `outputs/jab-learning-5895253/baseline/rollout_00.npz` fails the pilot:

| Measured result | Value |
|---|---:|
| Minimum left-glove / head-center distance | 0.16016 m (miss) |
| Minimum root height | 0.07499 m (fall) |
| Maximum sole drift, left / right | 0.2501 m / 0.2383 m |
| Final recovered guard dwell | 0 s |
| Maximum compiled joint-limit excursion | 0.01841 rad |
| Maximum actual joint torque | 138.03 N·m |
| Root actuation / externally applied forces | Exactly zero in recorded arrays |

The single nominal trajectory passes the recording, bounded-actuation and declared joint-limit gates but fails balance, foot drift, target contact, reference glove tracking and guard return. Small joint-space tracking errors alone do not prevent whole-body collapse; the independent root/world and Chest-relative checks make that distinction explicit. See `outputs/jab-learning-5895253/baseline/independent/evaluation.json`. This is an unlearned reference-PD baseline and does not establish the outcome of the learned controller.


## Paired evaluation bank

`outputs/scalar-evaluation-bank/guard_trials.npz` contains 32 fixed trials generated by `scripts/make_jab_evaluation_bank.py`, using seed 20261012 and NumPy PCG64. Trial 0 is exactly the V3 first guard pose with zero generalized velocity. Trials 1–31 add independent Gaussian hinge-position noise with 0.005 rad standard deviation, clipped to the compiled limits, and Gaussian generalized-velocity noise with standard deviation 0.02 (m/s for root translation, rad/s for root angular and joint velocities). Root positions and quaternions remain unchanged.

Random draws are interleaved per trial: joint position vector, then complete velocity vector. The first eight trials regenerated as an eight-trial bank exactly match the first eight of the 32-trial bank, making ordering independent of requested batch size. The bank records model and reference hashes; its manifest records the attack window without rewriting the frozen reference.

Both baseline and learned controllers must start from these exact arrays and record every trial. `scripts/audit_jab_bank.py` checks bank hash, model/reference hashes, all 32 unique trial IDs, and initial states within 3e-7 for float32 transfer before computing per-trial acceptance and aggregate counts. This is a paired small-perturbation test on the **same jab motion**, not held-out motion, target, body, or opponent generalization.

```bash
.venv/bin/python scripts/audit_jab_bank.py \
  --model outputs/EVALUATION/model/human.xml \
  --reference outputs/EVALUATION/reference.npz \
  --rollouts outputs/EVALUATION \
  --bank outputs/scalar-evaluation-bank/guard_trials.npz \
  --output outputs/EVALUATION/independent --workers 4
```


## Paired learned result: 32 trials

The immutable 32-trial bank was evaluated by baseline job **5895279** and learned-policy job **5895280**. Independent replay verified exact initial states (allowing float32 transfer tolerance), model/reference hashes, bank hashes and all 32 unique trial IDs before scoring. The learned controller came from PPO run **5895253**, checkpoint after 200 updates. The comparison keeps the same reference-PD and offline support feedforward; the learned controller supplies policy residuals.

| Independent gate | Reference-PD baseline | Learned policy |
|---|---:|---:|
| Every pilot gate together | 0/32 | **32/32** |
| No fall | 0/32 | 32/32 |
| Sole drift ≤ 3 cm | 0/32 | 32/32 |
| Compiled joint limits, with declared 0.02 rad tolerance | 1/32 | 32/32 |
| Attack-window contact | 12/32 | 32/32 |
| Recovered guard with recorded support | 0/32 | 32/32 |
| Reference glove tracking | 0/32 | 32/32 |
| Bounded controls and actual joint torques | 32/32 | 32/32 |
| Zero external applied forces / root actuation | 32/32 | 32/32 |
| One-reset, every-step recording provenance | 32/32 | 32/32 |

The per-trial JSON files and complete summaries are in `outputs/jab-learning-5895279/independent/` and `outputs/jab-learning-5895280/independent/`. These results establish a small-perturbation pilot on one fixed jab and one fixed head target. They do not establish performance on another punch, target, morphology, moving G1, or large disturbance distribution.

A separately recorded nominal training-run evaluation also passes all gates: minimum attack distance 0.11399 m, first sampled attack contact 1.750 s after reset, minimum root height 0.96347 m, maximum sole drift 4.95 mm, and recovered guard dwell 1.804 s. Final Chest-relative glove errors are 6.35 / 4.80 mm, 2.14° / 1.41°, and 0.00419 / 0.01206 m/s. Its video is `outputs/jab-render-5895283/jab-rollout.mp4`, with independent metrics in `outputs/jab-learning-5895253/learned/independent/evaluation.json`. The video shows actual learned-controller state replay beside the fixed G1; G1 is a visual target with no impact response.


Across all 32 accepted primary-policy trials, the closest attack distance ranged from 0.0980 to 0.12986 m and final recovered dwell from 1.800 to 1.824 s. Maximum final Chest-relative glove error was 7.54 mm / 2.31°, and relative speed was at most 0.0290 m/s. Maximum sole drift was 5.89 mm; peak joint torque was 64.72 N·m; the largest soft joint-limit excursion was 0.009714 rad, within the declared 0.02 rad tolerance. Root actuation and external applied forces remained exactly zero in every trial. Full ranges are saved in `outputs/jab-learning-5895280/independent/range_summary.json`.

The final reference preview (`jab-render-5895255`), actual learned rollout (`jab-render-5895283`), and corrected baseline diagnostic (`jab-render-5895275`) each passed a complete FFmpeg decode with exit code zero and no errors. Each render directory contains `video-validation.json`.

An exploratory variant with explicit online balance feedback plus learned residuals also passed all frozen gates in 32/32 trials (`outputs/jab-learning-5895286/independent/summary.json`). That evaluation uses checkpoint 100 of the separate 256-world training run, with the same 1,638,400 training transitions as the primary 128-world/200-update run. It is a secondary comparison; the selected milestone controller remains the original policy without that extra online balance-feedback helper. This small shared bank does not establish that either controller is generally superior.

## Five-millisecond, four-family extension: acceptance contract

The October 15 extension preserves the existing physical pilot thresholds and adds explicit timing and actuator evidence. Earlier results above remain 2 ms historical experiments; adding an evaluator option does not convert those checkpoints to 5 ms.

The proposed native implicit-PD actuator uses `gain=1`, affine bias `[0, -kp, -kd]`, and the original strength bounds as compiled `forcerange`. Its drive input is `kp*q_target + kd*v_reference + feedforward`. That input is **not torque** and can exceed the strength limit without violating it. The independent evaluator consequently requires every-step `actuator_force`, verifies each force against its compiled cap, transmits those forces through the XML gear/joint mapping, and compares the result with recorded `qfrc_actuator`. It still checks actual joint torques, zero root actuation, and zero externally applied forces. Unlimited drive is accepted only for the recognized force-limited direct-hinge affine actuator. Historical direct-torque motors retain their original control-range checks.

New releases must explicitly request the 5 ms physics and 20 ms action gates. Physics acceptance requires agreement among the frozen runtime XML, recorded timestep, every recorded state interval, and the **actual simulator time** saved as `simulation_time_s`. The latter must advance by 5 ms and agree with elapsed nominal time, allowing only dtype-specific floating-point accumulation error. Synthetic timestamps alone cannot pass. Action acceptance requires a recorded 20 ms interval and actual update indices `[0, 4, 8, ...]` before their corresponding physics intervals. Recording at 5 ms while using a different compiled timestep cannot pass. The end frame has no following control interval and is excluded from the update list.

```bash
.venv/bin/python scripts/audit_jab_bank.py \
  --model outputs/RUN/model/human.xml \
  --runtime-model outputs/RUN/model/human_runtime.xml \
  --reference outputs/RUN/reference.npz \
  --rollouts outputs/RUN \
  --bank outputs/BANK/guard_trials.npz \
  --require-physics-timestep 0.005 --require-policy-timestep 0.020 \
  --output outputs/RUN/independent --workers 4
```

`--model` remains the bank's immutable source-body XML for provenance. `--runtime-model` is the frozen compiled-controller XML; its hash must match every rollout. Exact runtime-file names follow the run manifest. These requirements make model/controller changes visible without rewriting the previous body or bank.

The first 5 ms GPU smoke run (5931602) exposed a serialization error: `mj_saveLastXML` preserved numerical affine coefficients but omitted changed actuator-type/limit flags. The saved XML therefore recompiled as a capped torque motor even though the live GPU controller used affine PD. Independent auditing caught this mismatch. Those files remain diagnostic artifacts, and a schema-reconstructed analysis explicitly fails runtime-file provenance. New accepted evaluations must use a correctly exported, recompiled runtime XML. The evaluator now checks the recorded controller mode against compiled actuator semantics, as well as the runtime-file hash; a matching hash alone cannot prove that a serializer captured the live controller correctly.

Annotated references identify the punch family, active hand, absolute source-speed factor, and attack-window semantics. The evaluator additionally requires contact by the designated active hand. It reports full-cycle and attack-window glove RMS, actual and reference first-contact times, their signed timing error, and linear threshold-crossing estimates at the available physics resolution. Positive timing error means the actual punch is late. Interpolation estimates a crossing between measured distance samples; it does not create substep measurements. An unreviewed or coarse window stays explicitly marked as such.

The plan's **2 cm active-glove RMS** and **20 ms contact-timing error** are reported separately as precision targets; they do not silently replace or relax the existing pilot gates. A passing physical pilot can still miss those tighter goals. Each family's fixed bank tests small guard perturbations of that reference only, not unseen motions, moving targets, continuous combinations, or biological realism.
