# Scalar human and learned jab milestone

12 October 2026. Implementation is confined to `human-punching`; sibling repositories are read-only. No Git staging, commits or pushes are part of this work. **The initial milestone is achieved:** a constrained scalar body, a complete jab reference, and a learned physical tracker that passes all declared pilot gates in 32 matched guard-start trials.

[Watch the learned jab beside G1](../outputs/jab-milestone/jab.mp4) · [Contact still](../outputs/jab-milestone/jab-contact.png) · [Checkpoint](../outputs/jab-milestone/checkpoint.pt) · [Bundle manifest](../outputs/jab-milestone/manifest.json) · [Learning curve](../outputs/jab-milestone/learning-curve.png)

This is one fixed-target jab at 0.6× source speed, with small initial-state perturbations. It is not the full four-family Task 09 tracker or adversarial game.

## Body choice and completed asset work

The original plan's SMPL-X model files were not available in the inspected project inputs. DuoBox fitted-motion arrays are available, but they do not supply those body-model files. We therefore used the plan's explicit fallback: the SOMA23 MuJoCo asset from ProtoMotions, pinned at `7a8417a9f55b1586b0d6e8f46578c5bc9d7ce1f1`. Upstream source, hashes and notices are retained under [assets/vendor/protomotions](../assets/vendor/protomotions/).

The user relaxed the exact-height requirement. The model retains its native bone lengths and measures **1.7664 m** in its neutral standing pose. The physical mass is our declared **75 kg scenario**, not a measurement of a capture actor.

The generated [body](../outputs/scalar-body/human.xml) has **48 actuated scalar hinges**, 55 position coordinates and 54 velocity coordinates, including a free unactuated pelvis. Knees are flexion hinges; elbows separate flexion from forearm rotation; wrist rotation is retained. Head/finger/toe articulation is reduced. Effort limits, gains, coordinate maps and selected collision pairs are explicit in [scalar_model.json](../outputs/scalar-body/scalar_model.json). There are no passive joint springs/dampers. Both sole contact and 13 selected self-collision pairs are enabled.

The visible body uses plain blue primitives derived from the simulator rig. This is a SOMA23-derived physical skeleton, **not an imported SOMA-X skinned mesh or a calibrated musculoskeletal model**. See [the body audit](16-scalar-body.md). The renderer explicitly colors gloves red for inspection without changing physical parameters or the frozen source XML.

## Retargeting the jab

[scalar_reference.py](../src/human_punching/scalar_reference.py) transfers the previously conditioned DuoBox jab onto the exact compiled scalar model. It uses both glove positions/orientations, stationary sole poses, root/torso posture targets, selected self-collision avoidance, and scalar joint limits. It inserts a quiet initial guard and a smooth source-pose return to the initial guard. The nominal source playback speed is **0.6×** for this learning pilot; this does not establish a prescribed fast-punch TTC.

The first solver exposed an important failure: solving an unconstrained Jacobian update and clipping joint angles afterwards left knees stuck at their bounds and produced large sole errors. Bounded least squares fixes that coupling. A second audit caught about 0.14 rad of joint-coordinate change at guard boundaries despite tiny Cartesian errors; removing the initial seed-pose bias eliminated those spikes. Earlier outputs are retained as diagnostics, not accepted references.

The frozen training reference is [outputs/scalar-reference-v3/jab.npz](../outputs/scalar-reference-v3/jab.npz), with [manifest](../outputs/scalar-reference-v3/manifest.json):

| Geometric diagnostic | Result |
|---|---:|
| Duration, including guard/return padding | 4.9889 s |
| Closest attacking glove/head center distance | 0.101116 m |
| Maximum glove-target position error | 0.291 mm |
| Maximum sole drift | 0.250 mm |
| Sole height range | −0.026 to 0.590 mm |
| Penetration among enabled self-collision pairs | None detected |
| Maximum scalar joint speed | 3.043 rad/s |

All reference poses stay within the declared scalar limits. The final held configuration equals the initial guard. Velocities use MuJoCo tangent-coordinate differentiation, including the free-root rotation. The 0.14 m contact and 0.16 m safe-clearance definitions are unchanged.

The reported reference contact is about 1.5844 s after the archive starts, or 1.0844 s after the explicit 0.5 s quiet guard. That launch convention is a declared playback annotation; it is not a reviewed source outward-launch label or a certified requested-TTC contract.

## Training and evaluation implementation

[jab_learning.py](../src/human_punching/jab_learning.py) implements a small reference-conditioned PPO tracker using the already-tested mjlab/MuJoCo-Warp stack. Batched dynamics and policy learning execute on the allocated V100 GPU. The policy adds bounded scalar joint-target residuals to the reference; joint PD plus a documented support feedforward calculation produces capped motor torques. Computed support forces are never applied externally. Observations include proprioception, reference errors, future targets, phase, glove errors and previous action.

Training may reset to different reference phases. **Acceptance evaluation always starts at guard and runs continuously through punch and recovery, with one initial reset.** Checkpoints include the body/reference hashes, controller settings and observation/action contract. Runs snapshot their input body, reference and trainer source.

Independent checks live in [jab_evaluation.py](../src/human_punching/jab_evaluation.py) and [the evaluation contract](18-jab-evaluation-contract.md). They use actual recorded physics states and forces, both-glove contact/clearance, joint/torque limits, sole drift, and measured supported guard recovery. A collision during a fall is a failed attack. A reference replay cannot pass the physical-evidence checks.

The initial controller uses **2 ms physics and 20 ms policy updates**. This remains separate from accepted G1 timing. No compatibility claim is made for coupling this controller to G1's 5 ms plant.

## Experiments and failures

| Job | Purpose | Finding |
|---|---|---|
| 5895225 | Eight-world standing GPU/PPO smoke | Actual batched scalar physics, one PPO update, checkpoint save and continuous evaluation completed. This is an infrastructure check, not a trained jab. |
| 5895250 | Native zero-residual PD; 3 s quiet guard then v2 jab | Failed at both 2 ms and 5 ms. At 2 ms, the pelvis slowly drifts despite planted feet, then falls. At 5 ms, explicit PD develops immediate joint oscillation/saturation. All actuator bounds and zero-external-force conditions remained enforced. |
| 5895253 | PPO on corrected v3 jab, 128 worlds | 200 updates / 1,638,400 policy transitions. Training took 444.3 s after initialization and baseline evaluation. The learned six-second guard-start evaluation had no falls in 128 worlds; full independent acceptance was then checked on the fixed bank below. |
| 5895279 / 5895280 | Matched 32-trial baseline / learned evaluation | Same body, reference, controller gains, physical clocks and exact initial-state bank. Baseline passed 0/32 complete cycles; learned policy passed 32/32. |
| 5895270 / 5895274 | Native optional balance-feedback diagnosis / matched no-feedback control | Joint-only feedback greatly reduces initial guard drift, but neither native test completes the longer quiet-guard-plus-jab protocol. Late return-phase contact is rejected. |
| 5895277 / 5895286 | Secondary PPO with optional joint-only balance feedback / matched evaluation | The saved 100-update checkpoint also passes 32/32 pilot trials. It used 256 worlds and the same 1,638,400-transition budget. Training was intentionally stopped after saving this checkpoint; the selected primary policy remains 5895253. |

The native failure log is [outputs/scalar-native-5895250](../outputs/scalar-native-5895250/). Its head-envelope entry during a fall is explicitly rejected. These tests explain why geometric reachability and bounded torques alone do not establish a balanced punching controller.

## Accepted learned result

The primary checkpoint was trained **without** the optional classical root-error feedback. It uses reference joint PD and the offline support feedforward calculation, with learned residual actions supplying the successful balance/tracking corrections. The zero-residual baseline uses the same remaining controller and fails. This comparison demonstrates a material learned-control contribution.

The independent paired evaluations use one exact guard and 31 fixed perturbations: 0.005 rad standard-deviation scalar joint noise, clipped to limits, and 0.02-unit generalized velocity noise. Root position/orientation are not perturbed. Both rollouts are verified against the exact same bank arrays and hashes; neither resets after time zero. These are descriptive results on a small fixed bank, not a population reliability estimate.

| Independent gate | Reference PD baseline | Learned tracker |
|---|---:|---:|
| Complete pilot: every gate passes | 0/32 | **32/32** |
| No fall throughout cycle | 0/32 | 32/32 |
| Contact during declared attack window | 12/32 | 32/32 |
| Supported, safe, quiet guard recovered | 0/32 | 32/32 |
| Bounded recorded motor torque and no external/root actuation | 32/32 | 32/32 |

Across the 32 learned trials, closest attacking glove/head distance is **0.0980–0.12986 m**, maximum sole drift is **5.89 mm**, and the final recovered-guard dwell is at least **1.800 s**. Final chest-relative glove position error is at most **7.54 mm**, orientation error at most **2.31°**, and relative speed at most **0.029 m/s**. Peak actual joint torque is **64.72 Nm**, with every joint respecting its own configured motor cap. Maximum physical joint-limit excursion is 0.00972 rad, within the declared 0.02 rad soft-constraint tolerance. Root actuator forces and external applied forces are zero.

Guard acceptance requires both gloves within 3 cm / 15° of the chest-relative initial guard, low relative velocity, quiet root/joints, both feet supporting the body, and both gloves at least 0.16 m from the head for a 0.2 s dwell. Contact alone is insufficient. The attack window is the coarse declared reference interval 0.5–3.3889 s, not yet a manually reviewed outward/follow-through interval.

The video replays **actual recorded GPU physics states**, rather than the reference poses. It adds the existing standing G1 as a fixed visual fixture with unchanged human coordinates and places the same analytic head target at G1's proxy center. It does not show a learned G1 policy or impact response. Reference, failed-baseline, and learned videos all decoded completely with FFmpeg.

Authoritative geometry/contact metrics come from independently recomputed FK at every recorded 2 ms state. Warp's cached body geometry and contact/force arrays describe the preceding integration interval; policy evaluation preserves that trained observation convention. Guard dwell checks use the recorded positive foot forces and consecutive recovered states, with at most one-step boundary uncertainty. This timing detail is documented rather than changed only at evaluation.

## Limits of this milestone

- This is one training seed, one morphology, one jab source, one fixed target and small guard perturbations. Other punches, broader initial states, different targets, speed requests and repeated sequences require further training/evaluation.
- The reference runs at 0.6× source speed. The primary nominal attacking-glove tracking RMSE is about **4.14 cm**, and worst per-trial glove RMSE is 4.41 cm. These pass the declared preliminary 10 cm pilot gate but **do not meet the PDF's proposed 2 cm tracker gate**.
- Nominal physical first contact is 1.750 s after guard reset versus reference contact at 1.5844 s, about **166 ms later**. This is not the PDF's proposed 20 ms TTC accuracy. Full outward-launch annotation and timing training remain open.
- The accepted controller uses 2 ms physics; tested 5 ms classical variants are unstable. G1 integration must resolve the shared physics clock explicitly. The current checkpoint must not be silently run under different gains or timing.
- The model has engineering joint limits, masses, strengths and selected self-collision pairs. This is a more constrained training surrogate than the original all-ball rig, not a calibrated prediction of human biomechanics.

The next useful experiment is faster and more accurate jab tracking with broader guard perturbations, followed by additional punch families on this same frozen body. Mixed sequences, camera/perception integration and adversarial plan selection remain later milestones.

## Reproduction

From `human-punching`, with the existing pinned physics environment and conditioned source jab available:

```bash
# Export the fixed scalar body (do not overwrite assets used by an active job).
.venv/bin/python -B -m human_punching.scalar_body --output outputs/scalar-body

# CPU reference preparation; run substantial batches on an allocated CPU node.
.venv/bin/python -B -m human_punching.scalar_reference \
  --source-dir outputs/physical-5890952 --output outputs/scalar-reference-v3

# One allocated GPU; source snapshots and run artifacts stay under outputs/.
sbatch scripts/train_jab_volta.sbatch \
  --reference outputs/scalar-reference-v3/jab.npz \
  --num-envs 128 --updates 200 --save-every 25
```

The source jab is a derived artifact from the earlier physical pilot, rather than a new independent motion recording. Its path and hash are saved with the reference. Regenerating that source follows [the earlier pilot commands](13-physical-pilot-and-next-gates.md); exact retargeting results depend on those conditioning settings.

The [review bundle](../outputs/jab-milestone/README.md) contains the selected checkpoint, reference, body, controller schema, training log/curve, matched evaluation summaries, nominal actual rollout, video, and upstream source notices. Its file hashes and source jobs are recorded in the manifest. The bundle README provides evaluation commands using only its frozen controller assets and the installed project environment. See [the training guide](17-learned-jab-pilot.md) and [evaluation contract](18-jab-evaluation-contract.md) for detailed commands.

## Final verification

The full preview-environment suite passed **94 tests and four subtests**; one Torch-dependent module was skipped there. Its **seven learning tests passed separately** using Torch from the pinned physics environment and pytest from the preview environment. Ruff passed across `src`, `tests` and `scripts`. All three comparison videos passed complete FFmpeg decoding. The bundle's 29 files were checked against their SHA256 manifest entries, and all local links in the README and milestone reports resolve.

The primary training, matched evaluations, final rendering, and secondary evaluation jobs completed. The secondary training job was deliberately cancelled after its saved 100-update checkpoint reached the comparison budget. No GPU jobs remain running. [validation.json](../outputs/jab-milestone/validation.json) retains the test/media evidence. Nothing was staged, committed or pushed.
