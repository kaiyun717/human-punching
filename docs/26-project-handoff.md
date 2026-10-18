# Human-punching handoff: completed work and next milestones

Evidence snapshot: **15 October 2026**. This is the entry point for a new developer or a new server. It consolidates implementation and subsequent architectural discussion; proposed capabilities are explicitly separated from measured results. Earlier numbered reports are historical snapshots and may contain statements superseded here.

Related documents: [server migration and Git/artifact recommendations](27-server-migration.md), [HTML evidence report](master-report.html), [four-policy release and replay instructions](../outputs/four-punch-milestone/README.md), [controller interface](24-five-ms-controller-interface.md), and [Task 09 figure/upload guide](task09-latex-figures.md). The master research log is [main_mjlab.tex](../../main_mjlab.tex), outside the `human-punching` Git repository but inside `HumanBox`.

## 1. What is complete

We progressed from DuoBox motion playback to **four physically executed, learned residual PPO trackers**: jab, cross, left hook and right hook. Each runs on the same constrained SOMA23-derived body in the pinned MuJoCo-Warp/mjlab stack, with **genuine 5 ms physics and 20 ms actor updates**. Each selected checkpoint passes all 18 physical/controller/provenance gates and all three tighter precision targets in **64/64 fresh trials**. The matching reference controller with the learned residual disabled passes **0/64** for each family.

This is a completed low-level motion milestone, **not a completed human adversary or two-player game**. The four networks have separate weights and fixed references. They have not been shown to aim at new targets, accept requested TTCs, switch between their different guards, execute continuous learned combinations, or interact with an executing G1 protagonist.

The user permits autonomous development and GPU jobs but requires edits to remain inside `Azizan_Lab/HumanBox`. Normal development belongs in `human-punching`; `robust_sav`, sibling `mjlab` and `DuoBox` are read-only inputs unless specifically authorized otherwise. The explicit request to edit `main_mjlab.tex` is a document-specific exception to the narrower project boundary. Do not stage, commit or push anything on the user's behalf. No Git index operations or uploads were performed for this handoff.

## 2. Progression, decisions and failed approaches

| Stage | What was implemented or learned | Evidence |
|---|---|---|
| Dependency and Task 09 assessment | Matching package names were insufficient: accepted mjlab source is `8ee51fbcf806a7419189f706d9e394cbeb7790fa`; sibling HEAD had incompatible newer training requirements. Built isolated project environments around the accepted pin. | [01](01-compatibility-assessment.md), [09](09-physics-runtime.md) |
| DuoBox interpretation and visual retargeting | Verified the joint hierarchy, parent-local rotations, source clips and scaling. Early procedural mannequin was exactly 1.80 m. Simplified the overly realistic appearance to featureless light blue and repaired glove orientation attachment. These were kinematic previews. | [02–04](03-implementation-and-results.md) |
| Standing-G1 targeting | Added a fixed G1 visualization and deformed glove paths with arm IK. Early approximately 0.185 m near-head passes were visual near-misses, not accepted contact. | [07](07-standing-g1-targeting.md) |
| Contact-directed references | Preserved the PDF's center-distance thresholds: contact at at most 0.14 m; safe glove clearance at least 0.16 m. Built contacting variants and complete attack/withdrawal/guard cycles. | [10–12](10-contact-contract.md) |
| Classical physical pilot | Added a free root, engineering masses/inertias, bounded effort and floor contact. Established native and actual Warp dynamics, contacting physical punches and two continuous classical jabs on the older body. Successful cases used 2 ms; early 5 ms attempts failed. | [09](09-physics-runtime.md), [13](13-physical-pilot-and-next-gates.md) |
| Body revision | The early ball-joint mannequin had unsuitable elbow/knee freedoms and missing self-collision. SMPL-family model files were not available for the intended body route. Adopted the plan's SOMA fallback and constrained it into 48 scalar hinges, with useful wrist rotation and selected self-collision pairs. Exact 1.80 m height was no longer required. | [14–16](16-scalar-body.md) |
| First learned jab | PPO learned joint-target residuals around a reference/PD/feedforward controller. The original accepted jab was at 0.6× source speed and 2 ms physics; it passed 32 paired physical trials but missed the tighter precision/timing targets. | [17–19](19-scalar-jab-milestone.md) |
| Genuine 5 ms repair | Diagnosed explicit velocity damping outside the implicit solver. A coupled mode had an estimated local timestep bound near 2.57 ms. Moved the same bounded PD force law into native affine actuators so implicitfast sees its damping derivative. Preserved the body and original output limits. | [20](20-5ms-controller-investigation.md), [24](24-five-ms-controller-interface.md) |
| Export/provenance repair | Early runtime XML export omitted modified native-actuator enum/limit flags. Actual GPU forces were correct, but those XMLs were not accepted replay artifacts. Replaced the exporter with explicit serialization, recompilation and equivalence checks. | [20](20-5ms-controller-investigation.md) |
| Jab refinement | Accurate tracking initially overshot wrist limits. A maximum-substep limit cost repaired that; Chest-relative guard rewards then repaired final-guard drift. Independent checkpoint selection beat simply using the last update. | [22](22-five-ms-learning-progress.md), [23](23-five-ms-validation-log.md) |
| Additional families | Initial full-speed transfers missed or fell despite high average imitation reward. Added active-hand attack rewards, conservative updates, slower hook curricula and explicit guard objectives. Cross needed neck/joint-posture recovery. | [22](22-five-ms-learning-progress.md) |
| Hook dynamics and retiming | Exact original hook references demanded more than twice some shoulder caps in simplified inverse-dynamics diagnostics. Retimed their joint paths nonuniformly rather than increasing strength. The right-hook ideal demand fell below all caps; left retained a small residual excess in the diagnostic and was physically executed by the learned controller. | [21](21-four-punch-references.md), [25](25-reference-dynamics.md) |
| Selection and release | Extra left-hook refinement reduced joint excursion but worsened foot drift and was rejected. Final fresh tests passed for all four selected policies. Packaged all learned and paired-baseline trials with media, source snapshots and hashes; fixed a nested timing-metadata packaging mismatch before publication. | [23](23-five-ms-validation-log.md), [packaging guide](four-punch-packaging.md) |
| Documentation and handoff | Maintained Markdown reports, experiment ledgers and an HTML report with successful and failed evidence. Added the Task 09 research-log section and this migration-oriented handoff. | [HTML guide](master-report-guide.md), [migration guide](27-server-migration.md) |

The inverse-dynamics calculations assumed exact reference tracking and simplified support-force allocation. They guided conditioning; they are not a proof that all nearby full-speed motions are impossible.

## 3. Accepted policy release

**Authoritative bundle:** `outputs/four-punch-milestone/`. Preserve it unchanged and write new experiments elsewhere. Its manifest SHA256 is:

```text
16eebb942f71f8350bf3d886f8a083eac549a5bc3486fe9d5304b8cde29b0686
```

The release has **1,264 hashed artifacts totaling 493,177,707 bytes**, excluding the small manifest/README overhead: four checkpoints, exact source/runtime models, references, metadata, schemas, training/evaluation source snapshots, **256 learned and 256 baseline rollouts**, detailed audits, eight videos and eight numerical plot sets. Historical origin paths inside copied JSON are provenance, not proof that every original directory must exist for inference. The bundle README's evaluation command uses bundled input paths.

| Family | Training job / selected update | Fresh learned / baseline jobs | Attack-glove RMS range | Max. interpolated contact error | Min. recovered guard dwell | Source timing |
|---|---|---|---:|---:|---:|---|
| Jab | `5931736 / 100` | `5931806 / 5931807` | 4.98–7.32 mm | 4.43 ms | 1.835 s | Uniform 1.0× |
| Cross | `5931870 / 100` | `5931915 / 5931916` | 6.92–9.06 mm | 9.87 ms | 1.735 s | Uniform 1.0× |
| Left hook | `5931834 / 300` | `5931934 / 5931935` | 8.53–13.01 mm | 8.38 ms | 1.735 s | Explicit nonuniform retiming |
| Right hook | `5931884 / 100` | `5931918 / 5931919` | 6.00–10.08 mm | 0.93 ms | 1.755 s | Explicit nonuniform retiming |

All families pass **64/64**, versus **0/64** matched baselines. All pass full-cycle and attack-window active-glove RMS ≤2 cm and contact-schedule error ≤20 ms. Those precision targets are additional requirements for this release, distinct from the historical pilot gates. Interpolated timing decimals come from 5 ms samples and do not provide finer temporal measurements.

Checkpoints were selected using 32-trial validation banks before generating fresh 64-trial test banks. Each final bank has one shared nominal anchor plus 63 fresh perturbations: hinge-position standard deviation 0.005 rad and generalized-velocity standard deviation 0.02 in the corresponding SI units. Initial root position/orientation is unchanged. This is one source recording and one trained lineage per family, not population-level robustness or independent training-seed replication.

Final verification included **133 project tests and four subtests**, with the Torch-dependent module skipped in that environment and **nine learning tests** passing separately in the pinned physics environment. All eight final videos fully decoded; selected contact/recovery/fall stills were visually inspected. Both HTML editions passed browser/static checks. All 1,264 release hashes and sizes were verified after publication. A right-hook evaluation loading bundled inputs also passed every physical and precision gate. See [release verification](../outputs/four-punch-release-verification-5932008.json) and [bundled replay audit](../outputs/punch-5ms-5931997/independent_smoke.json).

## 4. Execution contract that must survive future changes

- Body: constrained SOMA23-derived primitive geometry, native height **1.7664 m**, declared **75 kg** scenario, **48 scalar motors**, `nq=55`, `nv=54`, free unactuated pelvis, selected self-collision pairs. This is an engineering model, not calibrated human biomechanics or an SMPL-X skin model.
- Runtime: actual **0.005 s** MuJoCo-Warp integration; exactly four physics steps per **0.020 s** learned action. No hidden 2 ms steps. A 20 ms action period is not an added 20 ms sensor delay.
- Action: bounded reference-relative joint-angle residuals, currently ±0.25 rad before joint-target clamping. Reference and PD drive update each physics step; the actor residual is held for four steps.
- Native actuator: `drive = kp*q_target + kd*v_reference + joint_feedforward`; physical torque is `clip(drive - kp*q - kd*v, original torque limits)`. A large drive is not a large applied torque; audit actual actuator and generalized forces.
- State: feedback is the human state, reference errors and phase, including future reference samples and previous action. There is no live G1 head-position or requested-TTC command in the accepted actor observation.
- Reference: each checkpoint is paired with an exact hashed archive. The hooks' scalar source speed is null because timing is nonuniform; runtime playback 1.0 means playback of that modified archive. Do not relabel this as unchanged capture speed.
- Contact: either glove center within **0.14 m** of the head center is analytic contact; both at least **0.16 m** away have safe glove clearance. Guard acceptance additionally checks support, posture and low velocities. These thresholds alone are not a certificate of full-body safety.
- Reset: accepted evaluation resets once at the initial guard, then executes the complete physical cycle without state replacement, root assistance or external applied forces. It does not validate resets or network swaps inside a continuous match.

The source-body SHA256 is `c5c1e602bed82c5ee8f1385eefa012a0eb95ed16a2253fa263c3f824522277ad`; the accepted runtime XML SHA256 is `e98fe0b17d3ccc235ac9e88e8b8d1c5b28d848e14a3db0255175436f12393bca`. Detailed interface and integration checks are in [report 24](24-five-ms-controller-interface.md).

## 5. Limits that a new developer must not infer away

1. **Four specialists, not one general puncher.** Different weights and references share the body/interface. Each returns to its own guard. Direct inspection found starting root placements differing by up to about 13.5 cm, alongside substantial arm-pose differences; current handoffs are not plug-and-play.
2. **Small guard noise is not varied-guard competence.** Training also samples phases of the same reference, but this does not establish arbitrary stances, displaced feet, moving-start combinations or broader disturbances.
3. **Fixed target is not target generalization.** Every accepted reference uses the same head center. Moving the head marker does not redirect a reference-conditioned actor. New targets require reference construction and broader tracker training/evaluation.
4. **Timing accuracy is not TTC control.** The measured error concerns one preset reference schedule after reset. Outward-launch annotations and a validated launch-to-contact command range remain work. Arbitrary speed multipliers can make a previously feasible punch dynamically infeasible.
5. **G1 is currently a visual/analytic fixture.** No accepted G1 policy executes in these human evaluations, and no human–G1 impact response is qualified. Physical human balance/floor dynamics and analytic head contact are real evaluated components; coupled robot avoidance is not.
6. **Matching clocks does not match the whole simulator.** Human uses elliptic friction/Newton 30 iterations/15 line-search iterations/tolerance 1e-7; the recorded G1 recipe uses pyramidal/Newton 10/20/1e-8. A combined scene has one global configuration. Preserve the accepted G1 recipe and qualify/adapt the human under it; current support-force decoding assumes elliptic constraints.
7. **There is no trained tactical adversary.** No learned punch selector, requested-rate scheduler, adversarial replay/critic change, dynamic G1 perception integration or two-player result is included in the release.
8. **This is a bounded physical model and opponent class.** No biological calibration, free footwork, population/morphology coverage or robustness theorem for unrestricted human attacks has been established. The left hook's worst observed foot drift, 27.20 mm against a 30 mm allowance, is a useful sensitivity to watch.

## 6. Recommended architecture and next phases

These are recommendations agreed in the latest assessment, **not implemented functionality**. Keep the four specialists for now. A common network alone would not supply missing guard, target or timing coverage. Aim for a stable low-level interface that can later be served by four specialists or a consolidated tracker.

The high-level adversary should choose `wait` or `launch(punch_family, feasible_nominal_TTC)` from admissible human states. At launch, snapshot the G1 head in world coordinates, construct a feasible reference and freeze it. Continue physical feedback corrections during execution, but do not use updated G1 state to redirect the committed strike. A later punch may snapshot a new head location. A dodge can cause no actual contact even when the human accurately tracks its frozen plan.

Keep waiting time, nominal launch-to-contact duration and actual contact time with a moving head separate. Actual contact time may be absent. Initially expose a small set of validated target/timing options rather than an unrestricted continuous command space. A state-dependent DNN scheduler can be a meaningful adversary over that set; its sophistication cannot compensate for invalid motion primitives.

| Phase | Work | Exit evidence |
|---|---|---|
| **N0 — Reproduce on the destination** | Transfer the release and exact source pin, rebuild environments, adapt allocation/EGL mapping, and replay frozen policies before training. Before expensive new training, also run a bounded human-only comparison under the recorded G1 solver settings; full shared-scene qualification remains N3. | Hash agreement, native-clock/force checks, nominal and matched-bank evaluations; report any hardware-dependent changes rather than promising bitwise trajectory identity. |
| **N1 — Common guard and continuous execution** | Retarget/retrain toward a shared feasible guard region and compatible stance. Define recovery using pose, velocity, safe clearance and measured support. Implement persistent per-entity state and explicit phase/previous-action handling. | Repeated jabs, then all **16 ordered family pairs**, then longer mixed sequences without resets, pose snapping or accumulating foot drift. |
| **N2 — Target and timing coverage** | Derive the relevant head-position range from G1 states. Generate dynamically checked references with IK/trajectory optimization, broaden guard states and train each family across feasible targets/durations. Define outward launch explicitly. | Held-out targets, timing choices and starting states; counterfactual frozen-target accuracy and physical recovery, with explicit infeasible-command handling. |
| **N3 — Shared moving-G1 simulation** | Resolve global solver/cone settings, entity-local indices, resets and action ownership. Restore exact G1 policy/normalizer/snapshot artifacts and exercise the camera pipeline. Keep human launch plans frozen. | Preserved G1 interface behavior, no-homing checks, physical continuous human execution, matched fixed/scripted/random opponent evaluations. |
| **N4 — Tactical adversarial learning** | Freeze accepted trackers and learn a scheduler from G1 state/motion and human readiness. Admit decisions only at legal boundaries. Represent plan, phase and readiness in replay/critic state; enforce commitment in Bellman targets as well as rollout actions. | Stronger valid attacks than matched random/heuristic schedules under the same timing/opportunity budget, with held-out sequences and independent seeds. |
| **N5 — Rapid combinations and optional consolidation** | Add transitions from nonzero-velocity follow-through states. If helpful, distill specialists into a reference/punch-conditioned tracker and fine-tune on continuous execution. | Tight combinations and long rollouts; no reliance on a full settled-guard reset between strikes; no regression in isolated physical gates. |

A full guard return after every strike produces separated attacks, not natural rapid combinations. N1 is an intentional first sequence milestone. Consolidation is optional engineering work; a single scheduler can control a validated skill library. The specialist-to-generalist route has research precedent in [InterMimic](https://sirui-xu.github.io/InterMimic/), but transfer to this model still requires our own experiments.

## 7. Where to resume in the code

| File or directory | Responsibility |
|---|---|
| `src/human_punching/scalar_body.py` | Constrained training body construction and body audits |
| `src/human_punching/scalar_reference.py` | Scalar-body reference construction/retargeting |
| `scripts/retime_punch_reference.py` | Explicit nonuniform timing transformations |
| `src/human_punching/scalar_control.py` | PD/feedforward and correct runtime-model export |
| `src/human_punching/jab_learning.py` | Generic single-reference specialist trainer/evaluator, despite the historical jab name |
| `src/human_punching/jab_evaluation.py` | Independent FK, contact, recovery, cadence and force/provenance gates |
| `scripts/make_jab_evaluation_bank.py`, `scripts/audit_jab_bank.py` | Frozen paired initial states and independent bank evaluation |
| `scripts/train_punch_5ms_volta.sbatch` | Current one-V100 training/evaluation entry point; freezes source snapshots |
| `scripts/render_jab.py`, `scripts/plot_punch_trial.py` | Recorded-physics videos and numerical evidence, not training |
| `scripts/package_punch_policies.py` | Strict immutable release packaging; explicit checkpoint selection |
| `assets/vendor/protomotions/` | Tracked upstream SOMA source/config, version record and notices |
| `docs/20` through `docs/25` | Latest control, reference, learning, validation, interface and dynamics investigations |

Historical preview/classical files remain useful but are not the accepted scalar tracker. In particular, do not use an old procedural model, old runtime XML, mutable latest checkpoint or convenient timing flag in place of the frozen release and call it the same validated controller.

## 8. Immediate handoff checklist

1. Read this file and the migration guide; transfer the approximately 493 MB release before discarding any server-local artifacts.
2. Preserve the exact mjlab source revision and the original four DuoBox inputs needed for future reference work. Keep the research-log sources and figure inventory separately from the code repository where necessary.
3. Follow the bundle's hash and evaluation instructions with a rebuilt pinned environment. Current tested hardware is V100 on `xeon-g6-volta`; the old L40S script belongs to a different cluster. Requalify the new allocation/GPU instead of copying environment directories.
4. Start N1 with measured guard compatibility and repeated execution; do not assume the current policies can be hot-swapped.
5. Store new references, checkpoints, test banks and release versions separately. Freeze selections before fresh testing, and distinguish validation used for tuning from final evidence.

See [27-server-migration.md](27-server-migration.md) for the exact distinction between Git-tracked source, ignored but essential experiment artifacts, reconstructible caches, and optional historical archives. The existing HTML report's media links require their original output paths; copying only its HTML is not equivalent to copying its evidence. Its standalone edition embeds selected media for offline reading, while the full release preserves numerical and policy artifacts.
