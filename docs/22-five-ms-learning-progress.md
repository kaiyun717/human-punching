# Five-millisecond physical tracking: experiment log

Started 15 October 2026. Completed objective: establish genuine **5 ms physics** with **20 ms policy updates**, then improve the jab and learn cross, left hook and right hook on the same constrained scalar body. All writes are confined to `human-punching` inside `Azizan_Lab/HumanBox`. Sibling repositories are read-only. No Git staging, commits or pushes.

## Completed milestone: four learned punches at 5 ms

All four selected policies pass **64/64 fresh trials each**, all **18 physical gates**, and all three tighter targets: full-cycle active-glove RMS ≤2 cm, attack-window active-glove RMS ≤2 cm, and interpolated contact-time error ≤20 ms. Each matched zero-residual controller passes **0/64**. These are 256 accepted isolated cycles, not 256 independent recordings or trained policies.

| Family | Learned passes | Zero-residual passes | Attack-glove RMS (mm) | Maximum contact-time error (ms) | Minimum guard dwell (s) | Reference timing |
|---|---:|---:|---:|---:|---:|---|
| Jab | 64/64 | 0/64 | 4.98–7.32 | 4.43 | 1.835 | Uniform 1.0× |
| Cross | 64/64 | 0/64 | 6.92–9.06 | 9.87 | 1.735 | Uniform 1.0× |
| Left Hook | 64/64 | 0/64 | 8.53–13.01 | 8.38 | 1.735 | Nonuniform retiming |
| Right Hook | 64/64 | 0/64 | 6.00–10.08 | 0.93 | 1.755 | Nonuniform retiming |

Each test bank was generated after selecting its checkpoint using the existing 32-trial validation bank. A test bank contains one shared nominal anchor and 63 fresh small perturbations (joint-position σ=0.005 rad, joint-velocity σ=0.02 rad/s; unchanged root pose). Contact-time estimates interpolate distances on a 5 ms clock; their sub-step decimals do not imply sub-millisecond physical measurements. Both gloves must be safely clear during recovered guard, with recorded foot support and quiet posture/velocity. The 0.14 m contact and 0.16 m safe-clearance definitions remain unchanged.

The frozen [four-family bundle](../outputs/four-punch-milestone/README.md) and [hash manifest](../outputs/four-punch-milestone/manifest.json) preserve checkpoints, controller/body/reference snapshots, all learned and paired-baseline rollouts, trial audits, source snapshots, plots and videos. The [validation log](23-five-ms-validation-log.md) records exact job IDs and selection/test-bank provenance; the [controller interface](24-five-ms-controller-interface.md) explains execution and remaining G1 integration differences.

## Required contract

- Exactly 0.005 s per integrated physics step; four steps per 0.02 s policy period. No hidden 2 ms integration or changed reporting clock.
- Original 48-hinge articulation, free unactuated pelvis, chosen 75 kg mass scenario and original motor output-strength limits. No external root assistance, pose replacement or mid-cycle resets.
- Every evaluation starts at guard and records every physics step through attack and supported recovery. Fixed initial-state banks distinguish actual control success from random-phase training diagnostics.
- Existing 12 pilot gates remain, plus explicit physics/policy cadence and actual actuator-force checks. The proposed stronger targets are active-glove RMSE ≤2 cm and reference-contact timing error ≤20 ms; results against these are reported separately and must not be silently substituted for the pilot gate.
- Faster references aim first at 1.0× source playback speed. Quiet guard and return padding are declared separately. This is not a promise of arbitrary requested punch frequency.

## Diagnosis and implementation

The prior explicit PD law computes velocity damping outside MuJoCo. Consequently, implicitfast cannot include that derivative in its implicit damping solve. The accepted guard has a coupled leg/ankle damping mode whose local explicit-integration bound is about 2.57 ms; a 5 ms step crosses that bound even though the continuous torque law is reasonable.

The implemented replacement uses a native affine actuator:

`force = clip(drive - kp*q - kd*v, original torque bounds)`

`drive = kp*q_target + kd*v_reference + support_feedforward`

This exposes damping to implicitfast while preserving the bounded force law, articulation, mass and strength. Drive input is not applied torque: the evaluator must audit actual `actuator_force` and generalized actuator force. Runtime XML and controller schema are frozen, rather than claiming the original source XML fully describes the altered actuator implementation.

The new launcher is `scripts/train_punch_5ms_volta.sbatch`. It enforces 5 ms / 20 ms, snapshots the full local Python package, and records control/evaluation provenance. Learning can explicitly initialize from previous policy weights, but that is documented as transfer with a fresh optimizer, never as an unchanged-checkpoint evaluation or exact resume.

Per-policy-boundary FK refresh computes derived geometry without changing qpos/qvel or advancing time. This removes the old observation's previous-substep geometry lag in the new experiment; old checkpoints retain their original observation convention unless explicitly used for new training.

## Experiments

Controller diagnosis is in [report 20](20-5ms-controller-investigation.md); reference construction is in [report 21](21-four-punch-references.md). This report records the learned experiments and final acceptance evidence.

| Job | Experiment | Result / status |
|---|---|---|
| 5931602 | Eight-world native-PD smoke, one PPO update | Correct actual GPU force law and 5 ms dynamics; untrained controller falls. Diagnostic only. |
| 5931618 | Transfer old jab weights to native PD at 5 ms, same 0.6× clip; 300 updates | Training completed, 2,457,600 transitions; secondary comparison. |
| 5931617 | Full-source-speed jab, precision reward, 600 updates | Completed 600 updates. Checkpoint 400 supplied useful precision behavior; later training regressed. Its historical runtime XML export is not accepted for replay. |
| 5931645 | Corrected-runtime, 32-trial evaluation of full-speed jab checkpoint 100 | 18/32 pass every gate. All 32 hit, remain upright, recover supported guard, obey torque bounds, and prove actual 5 ms / 20 ms clocks. Fourteen narrowly fail the unchanged 0.02 rad joint-limit tolerance. |
| 5931667 | Same bank, jab checkpoint 400 | All 32 meet the precision targets and every physical gate except the wrist-limit tolerance. Attack RMSE 5.96–7.29 mm; interpolated contact error 1.14–3.61 ms, approximately one physics tick. Maximum limit excursion 0.02765–0.03210 rad. |
| 5931664 / 5931665 / 5931666 | Cross / left hook / right hook at 1.0×, native 5 ms; 800 updates each | Explicit weights-only transfer from jab checkpoint 400. Cross/right completed 800 updates; failing left branch stopped after checkpoint 400. No original branch was accepted. |
| 5931681 | Jab checkpoint-400 fine-tuning, maximum substep joint-limit penalty 0.2, learning rate 1e-4 | Completed 400 updates. Improved joint limits; remaining guard mismatch motivated the next reward refinement. |
| 5931683 | Conservative left-hook training, learning rate 1e-4, target KL 0.03, limit penalty 0.1 | Stopped after checkpoint 200: conservative updates did not solve the full-speed reference difficulty. |
| 5931688 / 5931689 / 5931690 | First independent cross / left-hook / right-hook checkpoint-200 tests | None pass complete cycles. Cross remains upright but misses; left hook falls in all 32 trials; right hook falls in 16/32. High average imitation reward concealed poor attack tracking. |
| 5931694 / 5931711 / 5931735 | Jab limit-penalty checkpoints 100 / 200 / 300 | All satisfy joint limits and both precision targets. Checkpoint 100 passes 28/32 complete cycles; later checkpoints fail supported final-guard pose/dwell despite accurate world-space hands. |
| 5931706 | Cross with separate attack-hand tracking reward | Checkpoint 200, evaluated in 5931743, contacts in 32/32 at source speed. Still fails full acceptance due to joint limits, guard and some sole drift. |
| 5931707 / 5931708 | Left / right hook at 0.6× as a training curriculum | Completed 600 updates. Checkpoint-200 tests are upright but miss; later right-hook tracking improves, and final weights initialize the explicitly retimed hooks. |
| 5931736 | Jab with explicit Chest-relative final-guard reward | Fine-tunes the successful limit/precision behavior while correcting the measured guard mismatch. |
| 5931781 | Cross with joint-limit and Chest-relative guard refinement | Transfers the improved attack-tracking checkpoint 400, retaining the attack reward. |
| 5931779 | Jab with Chest-relative guard term, checkpoint 100 | Every physical gate and precision target passes in 32/32 validation trials. Checkpoint selected and frozen before fresh testing. |
| 5931806 / 5931807 | Fresh 64-trial learned jab / matched zero-residual baseline | **64/64 versus 0/64 complete physical passes.** New jab attack RMS 4.98–7.32 mm; contact error ≤4.44 ms; guard dwell ≥1.835 s. |
| 5931852 / 5931853 | Cross refinement checkpoints 100 / 300 | All checks pass except final joint-posture return: neck errors about 0.14 rad exceed the unchanged 0.12 rad guard allowance. |
| 5931833 / 5931834 | Retimed right / left hook, archive playback 1.0 | Both checkpoint-100 evaluations reach contact, remain upright and pass sole-drift checks in all 32 trials. Guard recovery remains incomplete; left also has a small joint-limit excursion. These archives use declared nonuniform timing, not uniform capture playback. |
| 5931870 | Cross with worst-joint guard-return reward | Targets the neck-posture failure without changing physics, gains, allowed joint/guard errors or punch timing. |

### Runtime provenance failure and repair

The first smoke/training jobs had a serialization error: `mj_saveLastXML` did not preserve the in-memory native actuator enum/limit changes. Those saved XML files cannot establish correct downstream replay. Actual GPU forces were independently checked against the native PD equation and respect the original limits; the trained weights remain usable. The corrected exporter writes the actuator semantics explicitly while preserving exact source-body properties, recompiles, and checks equality. New acceptance jobs freeze this corrected runtime XML and verify its hash. Historical diagnostic artifacts are retained unchanged. Report 20 contains the forensic evidence.

### First full-speed independent jab result

[Checkpoint-100 evidence](../outputs/punch-5ms-5931645/independent/summary.json) uses one nominal guard and 31 frozen small perturbations, seed 20261015. All 32 trials have zero applied external/root forces, no fall, active-hand attack contact, and at least 1.845 s of supported recovered guard. Maximum sole drift is 10.85 mm.

Full-cycle active-glove RMSE is 1.44–1.60 cm; attack-only RMSE is 2.77–3.14 cm. Contact is 34–44 ms later than the reference. The wrist transient reaches 0.02225 rad beyond its soft upper stop, compared with the pre-existing 0.02 rad allowance. Thus this checkpoint is strong evidence that genuine 5 ms control works, but it is **not** an accepted complete release and does not meet the stronger contact-timing target. The limit threshold and reference clock are not adjusted to make it pass.

The simulator's own clock is recorded at every step. Independent checks require 5 ms increments, matching nominal timestamps, and policy-update indices every four physics steps. A 20 ms policy period does not imply an extra artificial 20 ms sensor delay.

### Precision improves before joint-stop behavior

The [checkpoint-400 audit](../outputs/punch-5ms-5931667/independent/summary.json) greatly improves glove tracking and contact timing, yet increases the transient wrist-stop excursion. Longer training is not automatically better: the 600-update final checkpoint's online tracking reward later degrades. Checkpoints are selected by independent whole-cycle measurements rather than their update number.

The first corrective experiment adds a cost on the worst actual joint-limit excursion, taking a maximum across all joints **and every one of the four 5 ms steps**. This catches a wrist excursion between policy observations. It does not move the physical joint stop, strengthen the motor, change the allowed excursion, or advance the reference clock. A lower learning rate limits disruption of the already accurate punch.

The initial left-hook transfer also exhibited large policy changes and repeated training falls. An optional PPO target-KL early stop limits further optimizer steps when the sampled divergence grows beyond 1.5 times its configured target. It is a training safeguard, not a physics change or a guarantee that no individual update can exceed the target.

The penalty-only jab illustrates why world-space fist tracking is insufficient for guard acceptance. It retains millimetre-level world-space glove errors while slightly changing torso/arm posture, leaving the gloves about 3.1 cm from their initial **Chest-relative** guard. A new training term therefore targets the same relative geometry as the existing acceptance check. Neither the 3 cm threshold nor the 0.2 s supported dwell is relaxed.

The chosen jab is [checkpoint 100 from run 5931736](../outputs/punch-5ms-5931736/checkpoint_00100.pt), SHA256 `bafa080cb9025e8b70eeb7697e7efb4e4eb9981b104cdd25044a5b58c28978c0`. After the 32-trial selection pass, a fresh seed-20261016 test bank was generated: one shared nominal anchor and 63 previously uninspected perturbations. The learned controller passes all 64 trials and all tighter precision targets; the identical zero-residual controller passes none. This is one trained motion and small initial perturbations, not a general robustness claim.

[Fresh learned evidence](../outputs/punch-5ms-5931806/independent/summary.json) · [Matched baseline](../outputs/punch-5ms-5931807/independent/summary.json) · [Recorded-physics video beside G1](../outputs/jab-render-5931851/jab-rollout.mp4).

### Additional families require attack shaping and dynamic references

Initial cross and hook transfer runs show that high average imitation reward can coexist with a missed attack. The revised reward explicitly weighs the active glove during the geometrically derived attack interval. A rational distance score retains a useful learning signal when a hook initially misses by tens of centimetres, where a sharp exponential tracking reward is nearly zero. This is separate from the inactive-hand and final-guard objectives.

The first full-speed left-hook run (5931665) was intentionally stopped after checkpoint 400; the conservative variant (5931683) was stopped after checkpoint 200. Their retained outputs record why they were superseded. They are failed/diagnostic branches, not successful policies.

Offline inverse-dynamics diagnostics show a further obstacle: exact full-speed hook tracking can require more than twice a configured shoulder effort limit. Static center-of-mass margins are healthy; fast coupled motion is the issue. A slower curriculum followed by explicitly conditioned timing resolves the largest dynamic loads while the original strength limits remain fixed. These calculations assume exact reference tracking and simplified support-force allocation; they guide reference design rather than proving that all nearby punching motions are impossible. See [the diagnostic evidence](../outputs/reference-dynamics-5931716/README.md).

### Learning and scope

Each family currently has its own reference-conditioned residual PPO network. The four networks share the body, actuator mechanism, observation/action layout, 5 ms physics and 20 ms decision period. They are specialized trackers, not yet one network conditioned on a punch-type command. Their guards/stances differ, so selecting a family at a fresh guard reset is distinct from switching families during continuous motion.

Jab and cross preserve 1.0× source playback, with explicit 0.5 s initial guard, 1 s return blend and 0.6 s final guard. Hooks use declared nonuniform phase retiming: 0.25 s of initial quiet guard is reassigned to the demanding swing, approximately preserving contact time and preserving total cycle duration. Their scalar source speed is intentionally null; archive playback is 1.0×. Evaluation adds a 1 s hold for every family. They target the same fixed G1 head center analytically. No learned G1 policy, impact response, arbitrary requested TTC, or variable-frequency sequence has been validated by these isolated-cycle experiments.


### Guard refinement and final checkpoint selection

The cross needed a maximum-joint guard-posture reward in addition to glove geometry: a small neck error failed the unchanged 0.12 rad return allowance. Run 5931870 checkpoint 100 passes all 32 selection trials, then all 64 fresh trials in 5931915. Its matched zero-residual baseline, 5931916, passes none.

Both retimed hooks pass all 32 selection trials at checkpoint 300 (right 5931833, left 5931834). Further right-hook guard-posture training in 5931884 produces a much larger supported recovery margin: checkpoint 100 raises the minimum guard dwell from 0.23 s to 1.755 s. It passes all 64 fresh trials in 5931918; its matched baseline 5931919 passes none.

Extra left-hook refinement in 5931885 improves joint-limit margin but increases foot drift. Checkpoint 100 passes with a smaller foot-drift margin; checkpoint 200 fails the unchanged 3 cm sole-drift gate in all 32 trials (33.38–38.29 mm). The earlier 5931834 checkpoint 300 is retained. Training is intentionally stopped once a selected policy has sufficient evidence; stop records preserve the reason rather than presenting superseded branches as successes.

The first right-hook fresh jobs 5931892/5931893 were cancelled before any test outcomes were inspected, because guard refinement was still being considered. Final right-hook testing uses a new seed-20261017 bank created after selecting 5931884 checkpoint 100. Cancelled jobs are not acceptance evidence.


### Final verification

The complete preview-environment suite passes **133 tests and four subtests**, with the Torch-dependent module skipped there. The pinned physics environment separately passes **nine learning tests**. Ruff and shell syntax checks pass. Packaging has its own evidence-schema tests and real four-family preflight; videos are fully decoded and selected contact/guard/fall stills are reviewed before inclusion. HTML validation is recorded separately by the master-report builder.

All four actors remain specialized reference trackers. This milestone completes the requested clock repair, jab improvement and three additional learned punches. Continuous mixed sequences, requested punch rates, arbitrary targets and the shared dynamic G1 scene remain subsequent work; clock compatibility alone does not resolve the global solver/cone difference.

An independent final audit of nominal and last test worlds per family verifies unchanged body arrays, exact original strength caps, native force-law agreement within 3.01×10⁻⁵ N·m, genuine simulator clock increments and bit-for-bit zero-residual drive reconstruction. See [audit evidence](../outputs/five-ms-release-independent-audit/README.md). Final packaging also caught and repaired a nested retiming-metadata lookup; this affected evidence ingestion, not the accepted dynamics. All four real evaluation banks subsequently passed strict precision preflight.
