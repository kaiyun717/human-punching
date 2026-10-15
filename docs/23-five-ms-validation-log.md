# Independent validation of the 5 ms learned punches

This log records candidate evaluations as they are completed. An evaluated checkpoint is not automatically a released policy. Physical acceptance requires all declared gates on every recorded trial; the tracking/timing targets are reported separately. See [the evaluation contract](18-jab-evaluation-contract.md), [controller investigation](20-5ms-controller-investigation.md), [four-family references](21-four-punch-references.md), and [training progress](22-five-ms-learning-progress.md).

## Fixed validation banks

Every family uses 32 deterministic initial states generated with NumPy PCG64 seed 20261015: an exact guard trial and 31 trials with 0.005 rad hinge-position Gaussian perturbations and 0.02 generalized-velocity Gaussian perturbations. Root positions/orientations are unchanged. These exact arrays are not fed to gradient-update training. They are **validation banks**, because their results inform checkpoint/controller selection; they are not untouched final tests or unseen-motion generalization.

| Family | Active hand | Reference attack window (s) | Bank directory |
|---|---|---|---|
| Jab | Left | 0.500–1.38333 | `outputs/five-ms-evaluation-banks/jab-source1/` |
| Cross | Right | 0.500–1.30000 | `outputs/five-ms-evaluation-banks/cross-source1/` |
| Left hook | Left | 0.500–1.51667 | `outputs/five-ms-evaluation-banks/left_hook-source1/` |
| Right hook | Right | 0.500–1.39734 | `outputs/five-ms-evaluation-banks/right_hook-source1/` |

These initial four references use uniform 1× source speed; later hook retiming has separate banks described below. Windows are derived from playback launch through closest active-glove approach plus 150 ms; they remain explicitly marked **not capture-reviewed**. Bank manifests contain complete reference/model hashes and generator details. Every audit verifies the exact initial state, all unique trial IDs, hashes, every physics sample, actual simulator-time increments, and the `[0,4,8,...]` policy-update indices.

## Early export fault: diagnostic only

Smoke job 5931602 used actual native affine PD at 5 ms, but `mj_saveLastXML` incorrectly serialized its actuator type/limit flags. The resulting runtime file could not reproduce the live actuator. Independent auditing exposed the problem; those files were retained, not repaired silently. A diagnostic reconstruction from the saved body and controller schema confirms bounded recorded motor output and zero root/external assistance, but deliberately fails the runtime-file provenance gate. Both the unlearned controller and the one-update PPO smoke controller fell. No accepted policy claim comes from this run.

The corrected exporter writes explicit actuator attributes and checks the recompiled model's geometry, dynamics, actuator fields, and solver settings against the live model. Subsequent validation jobs use that verified runtime XML and record its hash. See `outputs/punch-5ms-5931602/schema-reconstruction-diagnostic.json` and the controller report for details.

## Full-speed jab: precise tracking exposed a wrist-limit problem

Both candidates below come from training job 5931617. They use the same body, 1× jab reference, 5 ms physics, 20 ms actor interval, and frozen 32-trial bank. Both passed **all actual timing, no-fall, sole-drift, active-hand contact, supported guard recovery, bounded-force, and no-assistance gates in 32/32 trials**. Neither passed the complete pilot, because of a soft joint-limit excursion.

| Candidate | Evaluation job | All physical gates | Full-cycle active-glove RMS | Attack-window active-glove RMS | Contact delay versus reference | Maximum joint excursion |
|---|---:|---:|---:|---:|---:|---:|
| Update 100 | 5931645 | 18/32 | 14.37–15.99 mm | 27.74–31.41 mm | 33.97–43.95 ms | 0.02225 rad |
| Update 400 | 5931667 | 0/32 | 4.03–4.73 mm | 5.96–7.29 mm | 1.14–3.61 ms | 0.03210 rad |

The timing estimates linearly interpolate the 0.14 m distance crossing between 5 ms FK samples. The second row is therefore best read as **within approximately one physics tick**, not as sub-millisecond ground truth. The 2 cm full-cycle RMS and 20 ms timing targets pass in all 32 update-400 trials; the attack-window RMS also lies well below 2 cm.

The failed joint is consistently `LeftHand_z`: it reaches the upper 35° wrist limit and overshoots during contact. At update 400, the nominal trial reaches 0.64127 rad at t = 1.210 s, exceeding the 0.610865 rad limit by 0.03040 rad; the declared tolerance remains 0.02 rad. At that time, the reference angle is already 0.60724 rad but its velocity reference is still +0.972 rad/s. Positive velocity feedforward near the stop is one plausible contributor. The remedy is being investigated through controller/training changes; **the acceptance threshold is not relaxed**.

This distinction matters: a high imitation reward and excellent glove tracking can coexist with a failed physical constraint. Update 600 later had worse online imitation reward than update 400, so training duration alone does not determine checkpoint selection.

Evidence:

- [Update 100 independent summary](../outputs/punch-5ms-5931645/independent/summary.json), [nominal trace plot](../outputs/punch-5ms-5931645/independent/nominal-plots/trial-evidence.png), and [joint-limit diagnosis](../outputs/punch-5ms-5931645/independent/joint-limit-diagnostic.json).
- [Update 400 independent summary](../outputs/punch-5ms-5931667/independent/summary.json), [nominal trace plot](../outputs/punch-5ms-5931667/independent/nominal-plots/trial-evidence.png), and [joint-limit diagnosis](../outputs/punch-5ms-5931667/independent/joint-limit-diagnostic.json).

## Joint-limit penalty fine tuning

Run 5931681 fine-tunes the accurate update-400 jab policy with a penalty on actual joint-limit excursions sampled across the four physics steps per actor update. Its update-100 checkpoint was evaluated in job 5931694 with verified checkpoint, body, reference, and runtime hashes.

The result improves to **28/32 full pilot passes**, with **32/32 joint-limit passes**. Maximum joint excursion is 0.00621 rad; active-glove RMS is 3.55–4.21 mm over the complete cycle and 4.85–6.15 mm during the attack window. Interpolated contact error is within 3.31 ms in every trial. All trials remain upright, contact with the active hand, respect force bounds, and meet the actual 5 ms / 20 ms clocks.

The remaining four failures are specific: Chest-relative glove position enters and stays within the 3 cm return tolerance only 165–190 ms before the recording ends, short of the unchanged 200 ms dwell requirement. They are not falls, missed punches, torque violations, or an unquiet root. The evaluator now reports dwell for each guard component so this issue is visible independently of the aggregate gate. See [the independent summary](../outputs/punch-5ms-5931694/independent/summary.json) and [guard-component diagnosis](../outputs/punch-5ms-5931694/independent/guard-dwell-diagnostic.json).

The update-200 checkpoint (evaluation 5931711) retains excellent attack precision and joint-limit compliance, but **0/32 complete guard recoveries**. In the nominal trial, final Chest-relative glove errors are 30.44 / 30.94 mm, just beyond the 30 mm gate; the root itself returns within 3.23 mm and 0.020 rad. Full-cycle glove RMS remains only 3.17–3.91 mm across trials. This exposes a distinct reward gap: matching world-space glove positions does not ensure that the hands return to the correct position relative to the chest. The controller can redistribute torso and arm posture while preserving the fist's world position. An explicit guard-return objective is being considered; the gate and recording duration remain unchanged. See [the update-200 summary](../outputs/punch-5ms-5931711/independent/summary.json).

## Accepted jab and fresh post-selection test

Run 5931736 added explicit Chest-relative glove recovery to the training objective while keeping the physical model, guard thresholds, episode duration, and 5 ms / 20 ms clocks fixed. Its update-100 checkpoint passed **all 18 gates and both precision targets in 32/32 validation trials** ([evaluation 5931779](../outputs/punch-5ms-5931779/independent/summary.json)). It was selected and hashed before generating a new test bank.

The selected checkpoint is `outputs/punch-5ms-5931736/checkpoint_00100.pt`, SHA-256 `bafa080cb9025e8b70eeb7697e7efb4e4eb9981b104cdd25044a5b58c28978c0`. The [selection record](../outputs/five-ms-test-banks/jab-source1/selection.json) precedes the new 64-trial bank, seed 20261016. Trial 0 remains the shared nominal guard anchor; the other 63 initial perturbations are new draws and were not used in gradient updates or checkpoint selection. This is fixed-motion, fixed-target evidence with small initial-state disturbances, not a claim of generalization to unseen punches or moving targets.

| Fresh 64-trial result | Learned residual PPO | Same controller, zero residual |
|---|---:|---:|
| All 18 pilot gates | **64/64** | 0/64 |
| No fall | 64/64 | 0/64 |
| Active-hand contact inside attack window | 64/64 | 27/64 |
| Supported guard recovery | 64/64 | 0/64 |
| Both precision targets | 64/64 | 0/64 |
| Actual 5 ms physics / 20 ms action cadence | 64/64 | 64/64 |
| Actual force bounds and no root/external assistance | 64/64 | 64/64 |

For the learned policy, full-cycle active-glove RMS is **3.81–5.78 mm**, attack-window RMS is **4.98–7.32 mm**, and interpolated first-contact error is **1.11–4.44 ms early**. The latter means contact is within one recorded physics tick; interpolation does not provide a finer physical measurement. Guard dwell is at least **1.835 s**, maximum sole drift is **11.48 mm**, and maximum joint-limit excursion is **0.00387 rad**, comfortably inside the unchanged 0.02 rad tolerance.

The learned and zero-residual runs have **identical complete controller schemas**, including reference feedforward, gains, physical timestep, actor interval, torque caps, source playback rate, and disabled online balance feedback. The only behavioral difference is the learned residual. The failed baseline demonstrates that this feedforward-plus-PD setup alone is insufficient; it does not establish that all classical control approaches would fail. See [paired comparison](../outputs/five-ms-test-banks/jab-source1/paired-results.json), [learned audit](../outputs/punch-5ms-5931806/independent/summary.json), [baseline audit](../outputs/punch-5ms-5931807/independent/summary.json), and [nominal numerical traces](../outputs/punch-5ms-5931806/independent/nominal-plots/trial-evidence.png).

The accepted nominal trial is rendered beside the fixed G1 in the [learned jab video](../outputs/jab-render-5931851/jab-rollout.mp4), with [closest-approach](../outputs/jab-render-5931851/jab-closest.png) and [final-guard](../outputs/jab-render-5931851/jab-final-state.png) images. The overlay states 5 ms physics and 20 ms actions. The complete video decoded successfully, and both stills were visually inspected. These are recorded simulated states; the G1 remains a visual and analytic target without collision response.

The [matched zero-residual baseline video](../outputs/jab-render-5931873/jab-rollout.mp4) shows the nominal failure under the same model and controller parameters; its [final image](../outputs/jab-render-5931873/jab-final-state.png) confirms the fall. This video also passed a complete decode. It is labeled as a reference-PD baseline, not a learned policy.

## Other families: initial update-200 candidates

The initial full-speed cross and hooks did not immediately inherit the jab's performance. All recorded runs below pass actual 5 ms physics, 20 ms action updates, force bounds, runtime/checkpoint provenance, and zero root/external assistance. Their motion failures are kept separate from those implementation checks.

| Family / training run | Evaluation job | Full pilot | No fall | Active attack contact | Attack-window active-glove RMS |
|---|---:|---:|---:|---:|---:|
| Cross / 5931664 | 5931688 | 0/32 | 32/32 | 0/32 | 18.69–19.13 cm |
| Left hook / 5931665 | 5931689 | 0/32 | 0/32 | 0/32 | 22.47–23.03 cm |
| Right hook / 5931666 | 5931690 | 0/32 | 16/32 | 0/32 | 33.11–33.76 cm |

The cross demonstrates why average imitation reward is insufficient: an online reward of about 0.927 coexists with a missed attack in every trial. The nominal active glove remains approximately 0.296 m from the head. The hook failures include substantial foot drift and late falls; incidental contacts during failed recovery do not satisfy the attack-window gate. All three candidates also violate joint-limit criteria.

These results motivated attack-focused rewards, more conservative PPO updates, and a slower initial curriculum for the hooks while retaining 5 ms physics. They do not establish that the reference motions are physically impossible. Complete summaries are stored in `outputs/punch-5ms-5931688/independent/`, `outputs/punch-5ms-5931689/independent/`, and `outputs/punch-5ms-5931690/independent/`.

## Cross and hook refinement

Attack-focused cross training 5931706 reached contact and precise tracking in all 32 trials at update 400, but overshot joint limits by 0.0969–0.1042 rad; one trial also exceeded the 3 cm sole-drift gate. Constraint refinement 5931781 then restored joint-limit and foot compliance. Its update-100 and update-300 candidates pass **17 of 18 gates in every trial**, with the remaining failure localized to joint-position return during final guard. The nominal largest final joint error is 0.1366 and 0.1452 rad, respectively, beyond the unchanged 0.12 rad return tolerance. Chest-relative glove recovery, support, root quietness, contact, and force bounds all pass. Update 300 has 4.77–5.72 mm full-cycle active-glove RMS, 6.64–7.72 mm attack RMS, and contact error at most 5.93 ms. See [cross update-100 audit](../outputs/punch-5ms-5931852/independent/summary.json) and [update-300 audit](../outputs/punch-5ms-5931853/independent/summary.json).

The slower right-hook curriculum (5931708, uniform 0.6× playback) reached active-hand contact in 32/32 trials at update 400, with full-cycle active-glove RMS 18.15–19.34 mm and contact error at most 1.83 ms. It still fails sole drift, joint limits, and guard recovery, so it remains a transfer checkpoint rather than an accepted policy. See [evaluation 5931785](../outputs/punch-5ms-5931785/independent/summary.json).

The next hook references preserve the spatial path but use nonuniform time warps to reduce peak load. Their 32-trial banks are `outputs/five-ms-evaluation-banks/left_hook-retime-v1/` and `right_hook-retime-v1/`, separate from the earlier uniform-speed banks. Both explicitly record `source_speed: null`, `timing_kind: nonuniform_pchip_phase_warp`, and local source-speed ranges (left 0.471–1.377, right 0.491–1.374). The total reference cycle remains unchanged while 250 ms is borrowed from the initial guard to lengthen the high-load swing. Runtime `--speed 1` means normal playback of this conditioned archive, **not uniform 1× source motion**. The attack windows remain explicitly marked as derived and not capture-reviewed; contact is required within those windows. Physical feasibility evidence and its limitations are detailed in [the reference-dynamics report](25-reference-dynamics.md).

At update 100 after transfer to the retimed references, **both hooks remain upright and contact within the active attack window in 32/32 trials**, with compliant foot drift and actual force bounds. The right hook passes 17/18 gates; only joint-position guard dwell is too short (0–145 ms). The left hook passes 16/18 gates: joint-limit excursion is 0.0258–0.0290 rad and joint-position guard dwell is 0–35 ms. Full-cycle active-glove RMS is 15.38–17.04 mm for the left hook and 17.18–17.90 mm for the right; attack-window RMS is 18.78–22.40 mm and 8.31–11.63 mm, respectively. First-contact error is at most 16.12 ms for left and 3.94 ms for right. Both secondary precision targets pass in 32/32, but neither policy passes the full pilot yet. See [left-hook evaluation 5931865](../outputs/punch-5ms-5931865/independent/summary.json) and [right-hook evaluation 5931866](../outputs/punch-5ms-5931866/independent/summary.json).

A separate [derived-window review](../outputs/hook-window-review-5931886/summary.json) checks the frozen references with dense 5 ms FK. Each hook has exactly one active-hand contact episode inside its window: left 1.35436–1.40043 s and right 1.20140–1.26517 s. Contact begins while the fist moves toward the target, and the fist is withdrawing by the end of the window. Neither inactive hand contacts, and neither inserted recovery produces contact. This validates the window's numerical strike/recovery separation; it does not upgrade the immutable archive's source-annotation flag to capture-reviewed.

Both retimed hook policies reach **32/32 full pilot passes and both precision targets at update 300**. This is validation evidence; final checkpoint selection and fresh tests are separate.

| Retimed hook / checkpoint | Evaluation | Full-cycle active RMS | Attack active RMS | Maximum contact-time error | Minimum guard dwell | Maximum sole drift | Maximum joint excursion |
|---|---:|---:|---:|---:|---:|---:|---:|
| Left / 5931834 update 300 | [5931889](../outputs/punch-5ms-5931889/independent/summary.json) | 5.91–8.86 mm | 8.71–14.46 mm | 7.73 ms | 1.735 s | 26.06 mm | 0.01336 rad |
| Right / 5931833 update 300 | [5931874](../outputs/punch-5ms-5931874/independent/summary.json) | 6.08–6.76 mm | 9.79–11.09 mm | 0.99 ms | 0.230 s | 13.41 mm | 0.00940 rad |

The right hook's guard dwell has only a 30 ms margin over the 200 ms gate, motivating a comparison with the already-running guard-joint refinement before final selection. A provisional right-hook 64-trial bank was generated and jobs 5931892/5931893 were submitted just before this coordination update; both jobs were immediately cancelled without inspecting outcomes. That bank and superseded selection record are retained for provenance. The final selected right-hook policy will use a new bank and seed, generated after selection.

## Cross and right hook: selected policies and fresh tests

The explicit guard-joint objective fixes the cross's remaining neck-posture drift. Cross checkpoint `5931870/checkpoint_00100.pt` passes all 18 gates in 32/32 validation trials and was frozen before a new 64-trial bank (seed 20261016). The right-hook guard refinement `5931884/checkpoint_00100.pt` raises validation guard dwell from at least 0.230 s to at least 1.755 s, improves glove tracking, and passes all 18 gates in 32/32. It was frozen before a new final bank with seed 20261017; the earlier provisional right-hook bank remains separate and unused for selection.

Both selected policies pass **all 18 gates and both precision targets in 64/64 fresh trials**. Their paired zero-residual baselines pass 0/64, fall in every trial, and make no attack-window contact. Complete controller-schema equality was checked for each pair.

| Fresh test | Full-cycle active RMS | Attack active RMS | Maximum absolute interpolated contact error | Minimum guard dwell | Learned / baseline full passes |
|---|---:|---:|---:|---:|---:|
| Cross / [5931915](../outputs/punch-5ms-5931915/independent/summary.json) | 4.77–5.99 mm | 6.92–9.06 mm | 9.88 ms | 1.735 s | 64/64 / 0/64 |
| Right hook / [5931918](../outputs/punch-5ms-5931918/independent/summary.json) | 5.35–7.21 mm | 6.00–10.08 mm | 0.94 ms | 1.755 s | 64/64 / 0/64 |

These timing values are linearly interpolated from 5 ms samples. Right-hook contact is therefore within one physics tick, rather than evidence of sub-millisecond physical accuracy. See [cross paired results](../outputs/five-ms-test-banks/cross-source1/paired-results.json), [right-hook paired results](../outputs/five-ms-test-banks/right_hook-retime-final/paired-results.json), and their selection records for the exact hashes and sampling protocol.

## Left-hook selection and complete four-family result

The stronger left-hook guard/limit refinement was checked before selecting a final policy. Update 100 (5931885) still passes 32/32, but reduces foot-drift margin compared with the original retimed update 300. Update 200 fails the 3 cm sole gate in all 32 trials (33.38–38.29 mm), despite strong guard recovery and low joint-limit excursion. We therefore retain `5931834/checkpoint_00300.pt`; the physical acceptance thresholds remain unchanged. See [the rejected refinement audit](../outputs/punch-5ms-5931922/independent/summary.json).

The selected left hook passes **all 18 gates and both precision targets in 64/64 fresh trials** (seed 20261016), generated after selection. Its active-glove RMS is 5.89–8.01 mm over the cycle and 8.53–13.01 mm during attack; maximum first-contact interpolation error is 8.39 ms. Guard dwell is at least 1.735 s, maximum sole drift is 27.20 mm, and maximum joint excursion is 0.01310 rad. The matched zero-residual controller falls in all 64 trials and makes no attack contact. See [learned audit](../outputs/punch-5ms-5931934/independent/summary.json), [baseline audit](../outputs/punch-5ms-5931935/independent/summary.json), and [paired comparison](../outputs/five-ms-test-banks/left_hook-retime-final/paired-results.json).

All four families now have independently audited **64/64 fresh-test passes at actual 5 ms physics and 20 ms actor updates**, plus a matched zero-residual baseline using identical controller schemas and initial states. These are four separate, reference-conditioned residual policies around an analytic fixed head target. The hook references use disclosed nonuniform timing. This milestone does not validate dynamic G1 impact response, moving-target adaptation, arbitrary-rate commands, policy switching, or long uninterrupted sequences without guard resets.

The [final evidence manifest](../outputs/five-ms-final-validation/manifest.json) links and hashes every selected checkpoint, body/runtime model, reference, selection record, fresh bank, paired comparison, audit summary, video, still, and numerical plot. All eight videos passed complete FFmpeg decoding. Learned contact/final-guard stills and baseline final-fall stills were visually inspected; the manifest distinguishes this limited still review from full manual video review.

| Family | Accepted learned video | Paired zero-residual video | Learned numerical traces |
|---|---|---|---|
| Jab | [Video](../outputs/jab-render-5931851/jab-rollout.mp4) | [Baseline](../outputs/jab-render-5931873/jab-rollout.mp4) | [Plot](../outputs/punch-5ms-5931806/independent/nominal-plots/trial-evidence.png) |
| Cross | [Video](../outputs/jab-render-5931928/cross-rollout.mp4) | [Baseline](../outputs/jab-render-5931947/cross-rollout.mp4) | [Plot](../outputs/punch-5ms-5931915/independent/nominal-plots/trial-evidence.png) |
| Left hook | [Video](../outputs/jab-render-5931955/left_hook-rollout.mp4) | [Baseline](../outputs/jab-render-5931949/left_hook-rollout.mp4) | [Plot](../outputs/punch-5ms-5931934/independent/nominal-plots/trial-evidence.png) |
| Right hook | [Video](../outputs/jab-render-5931931/right_hook-rollout.mp4) | [Baseline](../outputs/jab-render-5931948/right_hook-rollout.mp4) | [Plot](../outputs/punch-5ms-5931918/independent/nominal-plots/trial-evidence.png) |

## Reference visual evidence

Job 5931652 rendered all four final references beside the fixed G1. Each video passed a complete FFmpeg decode; closest-approach two-view stills were visually inspected. These are **kinematic references**, not learned rollouts, and the robot is a fixed visual/analytic target without impact response.

| Family | Video | Closest-approach image |
|---|---|---|
| Jab | [Reference video](../outputs/four-reference-render-5931652/jab/jab-rollout.mp4) | [Image](../outputs/four-reference-render-5931652/jab/jab-closest.png) |
| Cross | [Reference video](../outputs/four-reference-render-5931652/cross/jab-rollout.mp4) | [Image](../outputs/four-reference-render-5931652/cross/jab-closest.png) |
| Left hook | [Reference video](../outputs/four-reference-render-5931652/left_hook/jab-rollout.mp4) | [Image](../outputs/four-reference-render-5931652/left_hook/jab-closest.png) |
| Right hook | [Reference video](../outputs/four-reference-render-5931652/right_hook/jab-rollout.mp4) | [Image](../outputs/four-reference-render-5931652/right_hook/jab-closest.png) |

The legacy video basename is `jab-rollout.mp4` inside each family directory; visible titles correctly identify the family. New renders can use `--artifact-prefix` to name files by family. Validation does not mark attack segmentation reviewed merely because a closest-approach still looks reasonable.

## Audit tooling

`scripts/audit_jab_bank.py` writes per-trial evidence and a summary with gate counts, precision counts, and compact metric ranges. Missing contact times are counted as missing, never filled with zero. `scripts/plot_punch_trial.py` independently recomputes FK and plots reference/actual center distance, glove error, sole drift, and joint-limit excursion without altering or smoothing recorded states. SVG and PNG copies are generated from the same numerical traces.

The evaluator's focused tests cover legacy torque motors, native affine-PD output caps, gear transmission, missing force evidence, actual simulator clocks versus synthetic labels, float32 time accumulation, exact actor-update cadence, active-hand contact, interpolated contact timing, and recovery/provenance checks. Earlier 2 ms evaluation behavior remains available when the new required-clock gates are not requested.
