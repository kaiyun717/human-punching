# Task 09 compatibility and implementation assessment

Audit date: 10 October 2026. All inspection of `../../robust_sav` and `../../mjlab` was read-only. This project must keep all new environments, code, downloads, caches, outputs and reports under `human-punching`.

## Decision

The proposed human opponent is feasible as a staged research implementation. Motion retargeting and MuJoCo rendering are a sensible first deliverable. The complete Task 09 plan is **not a drop-in extension** of the accepted Task 08 application: it needs a physical human tracker, a two-entity simulation adapter, a constrained planner interface, and a new training/replay schema. No existing checkpoint is needed to construct and inspect the isolated human preview.

Use MuJoCo **3.11.0**, the same engine version recorded by Task 08. For subsequent mjlab integration, use the **Task 08 source commit**, not the sibling mjlab checkout's current HEAD. Keep the preview environment small; adding SMPLSim, Mink and WBC is unnecessary to render an initial articulated human reference. A kinematic replay is a reference/debugging artifact, not evidence that a torque-controlled person can balance, execute the punches or satisfy adversarial timing constraints.

The user has narrowed this iteration to four source punch families on a 1.80 m human mesh, without G1 targeting. That is a subset of milestone 09A and the kinematic portion of 09B. The full target/TTC, physical imitation and adversarial training stages remain future work.

## Audited local sources

- [RobustSafeArrival-Vision5.pdf](../../RobustSafeArrival-Vision5.pdf): finite-game theory, Task 08 implementation/closure, and accepted simulator and observation contracts.
- [Task09_Human_Adversary_Implementation_Plan.pdf](../../Task09_Human_Adversary_Implementation_Plan.pdf): proposed architecture and gates, dated 10 October 2026.
- `robust_sav` HEAD: `f4672e3ab403308520beb6c6476a4027096764bf`.
- `mjlab` working-tree HEAD: `033ae22a2c7a30a25a6fa77b16c113ed88dd1b55`.
- Task 08 mjlab pin, independently repeated in `src/boxing/standing/schemas.py`, `src/boxing/stance/plant.py`, `src/boxing/snapshots/restoration.py` and `configs/boxing/external_inputs.json`: `8ee51fbcf806a7419189f706d9e394cbeb7790fa`.

The pinned mjlab commit is present in the sibling repository's Git object database. Its files can be read or exported without checking out or changing that repository. The version string `mjlab==1.6.0` alone does **not** identify the accepted source: both inspected commits declare that version with different dependencies.

## Dependency comparison

| Component | Accepted robust_sav environment | Pinned mjlab `8ee51fbc` | Sibling HEAD `033ae22a` | Consequence |
|---|---|---|---|---|
| Python | `>=3.10,<3.14` | `>=3.10,<3.14` | `>=3.10,<3.15` | Python 3.11 is in the common range. |
| PyTorch | Package requires `2.8.0`; lock records `2.8.0+cu128` | `>=2.7.0`; CUDA 12.8 extra | `>=2.14.0`; CUDA 13.0 extra | Current HEAD conflicts with robust_sav's exact Torch pin. |
| MuJoCo | `3.11.0` | `~=3.11.0` | `~=3.11.0` | Same engine family; pin the patch version for reproduction. |
| MuJoCo-Warp | `3.11.0` | `~=3.11.0` | `~=3.11.0` | Suitable eventual shared GPU backend. |
| Warp | `1.14.0` | `>=1.14.0` | `>=1.14.0` | Preserve `1.14.0` for initial integration. |
| RSL-RL | `5.5.0` | `==5.5.0` | `==5.5.1` | Another reason to avoid installing sibling HEAD. |
| NumPy | `2.5.3` in standing lock | Unbounded declaration | Unbounded declaration | WBC introduces a conflicting upper bound; see below. |
| OpenCV | `opencv-python-headless==4.11.0.86` | Not a core dependency | Not a core dependency | Added by the perception overlay; not needed for basic replay. |

Evidence: [robust_sav package](../../robust_sav/pyproject.toml), [standing environment lock](../../robust_sav/requirements-standing.lock), [perception overlay](../../robust_sav/requirements-perception.lock), [external source pin](../../robust_sav/configs/boxing/external_inputs.json), and `git -C ../../mjlab show 8ee51fbcf806a7419189f706d9e394cbeb7790fa:pyproject.toml`.

The standing lock is an installed-environment inventory, rather than a universal platform-independent solver guarantee. Its scientific packages and historical CUDA build should not be silently updated in the accepted project. This assessment did not recreate or execute Task 08 training; it compares the checked-in source and dependency records.

### WBC is an optional later dependency, with real reconciliation work

The plan names WBC-mjlab revision `689f0836`. Its [official package declaration](https://github.com/wbc-mjlab/wbc-mjlab/blob/689f0836/pyproject.toml), retrieved during this audit, requires `mjlab>=1.6.0` and **`numpy<2.5`**. That conflicts with the accepted robust_sav lock's NumPy `2.5.3`. The comment attributes the bound to a mediapy problem; this audit has not independently reproduced that problem.

Its [official lock](https://github.com/wbc-mjlab/wbc-mjlab/blob/689f0836/uv.lock) resolves mjlab `1.6.0`, MuJoCo/MuJoCo-Warp `3.11.0`, Warp `1.14.0`, RSL-RL `5.4.2`, and platform/extra-dependent Torch entries including `2.11.0+cu128` and `2.12.1`. These are not the accepted Task 08 dependency versions. The RSL-RL version is a **lock resolution**, not an explicit direct WBC requirement.

Therefore do not apply WBC's lock to robust_sav or assume a common package version proves API compatibility. A future isolated tracker environment should constrain the exact accepted mjlab source, Torch `2.8.0+cu128`, MuJoCo/MuJoCo-Warp `3.11.0`, Warp `1.14.0` and RSL-RL `5.5.0`, then resolve a suitable NumPy below `2.5` and test WBC's extension and training APIs against that combination. This is a candidate compatibility route, **not a tested dependency lock**. Training and integration may initially use separate environments with versioned motion/checkpoint handoff. Native mjlab tracking remains the smaller-dependency fallback.

SMPLSim/Mink are offline construction tools, not requirements that must enter the accepted G1 runtime. Their additional dependencies and model licenses can stay in separate conversion environments if needed. SMPL-X model weights and source motion access are separate from open-source exporter code; an open exporter does not imply that the body-model files are present or freely redistributable.

## What transfers from the plan, and what needs adjustment

The plan's separation into a reference builder, a physical tracker and a boundary-only planner is sound. It prevents the high-level adversary from modifying an outward punch after commitment and gives physical tracking a separately testable contract. It also correctly identifies full-cycle source motion, both wrists, torso, legs and recovery as necessary data. An outward-only glove curve cannot reconstruct this information.

The first mesh need not be a seamless SMPL-X skin. The plan itself proposes articulated rigid mesh segments first. A procedural 1.80 m articulated mesh can exercise the motion-to-MJCF mapping and produce inspectable videos without restricted body-model assets. It must be described as that fallback, not as an SMPL-X reconstruction. Standing height refers to the neutral model; a bent-knee boxing posture naturally has a lower head.

Native MuJoCo skins are visualization-only, with collision handled separately. A successful native render does not establish support in the MuJoCo-Warp camera path. Prefer ordinary articulated mesh geoms with simple collision proxies for the first shared-scene model; test geometry occlusion and red-glove masks in the actual Task 08 GPU renderer before adopting any seamless skin. See the [MuJoCo skin specification](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-skin) and [MuJoCo-Warp project](https://github.com/google-deepmind/mujoco_warp).

SMPL axis-angle rotations, source skeleton joint orientations and MuJoCo scalar hinge positions are different representations. The final exporter must explicitly declare joint ordering, axes, rest frames and rotation conventions, and validate forward kinematics. A proposed 63-hinge human is not automatically supplied by importing SMPL-X or SMPLSim. Reference velocities must be derived within each clip; quaternion interpolation/sign handling cannot be replaced by component-wise interpolation.

For this preview, the important acceptance evidence is source provenance, declared timing/units/frame conversion, the fixed 1.80 m target morphology, four visible and correctly handed punch motions, and successful MJCF compilation/rendering. It does not require optimizing reach to a G1 head, freezing a head target, matching a requested TTC or demonstrating physical guard recovery. Source data availability and actual delivered replay fidelity are documented separately in the implementation report.

## The current Task 08 application is not already a human-adversary game

The Task 09 PDF cites several older paths. In this checkout the rendering integration lives under `src/boxing/perception/`, not the proposed `src/boxing/vision/plant.py`.

| Current source | Relevant finding | Future integration requirement |
|---|---|---|
| `src/boxing/stance/plant.py` | Pins mjlab source; physics `0.005 s`, decimation 4, control `0.02 s`; the application requires allocated CUDA. | Preserve those rates and the accepted G1 plant contract. |
| `src/boxing/training/plant.py` | Stance-energy and state operations use whole-model qpos/qvel; previous action comes from the environment action manager. | Gather G1 state/actions by entity mapping and scatter reset state only to the intended entity. |
| `src/boxing/perception/plant.py` | `sync_fists` synchronizes scheduled fist centers with renderer geometry. | Use hand-attached physical glove poses consistently for rendering, contacts and safety labels. |
| `src/boxing/perception/rendered_plant.py` | Builds the Task 08 camera extension and verifies `0.08 m` glove radii. | Include human-body occlusion and explicit collision filtering while preserving camera/latency semantics initially. |
| `src/boxing/training/residual_learning.py` | Rejects enabled disturbances with `Residual learning requires null disturbance`; critics require two 29-vectors. | New human planner/action/replay adapter; do not repurpose the 29 zero torque-disturbance channels without a new schema. |
| `src/rsav/learner.py` | Generic learner supports configurable bounded adversary modules and an adversary optimizer. | Reuse the SAV objective/update ideas, with correct legal-action and decision-boundary handling. |

The accepted actor has 948 inputs, critic observation 2,011, and Q input 2,069 = 2,011 + 29 + 29. The final 29-vector is currently a zero disturbance. A human plan index is neither that vector nor a joint torque disturbance. Retaining the protagonist actor's input schema is useful for warm starting, but the new opponent's appearance, occlusion and sequence horizon still require reevaluation and likely fine-tuning.

The accepted document also explicitly says delay/pending packets are simulator state absent from its critic, leaving the retained critic partially observed. A new centralized critic should include enough human state, committed plan, phase, sequence remainder, time, reference and sensor-buffer state to justify any Markov claim. Human and G1 action histories and reset procedures must be separate.

### Two additional game-design details

1. **A discrete planner does not fit the existing continuous adversary gradient unchanged.** `Learner.update_adversary` differentiates Q through a bounded adversary output. An integer clip selector/argmin breaks that path. For the first small finite action bank, explicitly evaluate/minimize over legal plans at boundaries, or introduce a documented categorical/planner learner. Continuous bounded plan deformation can use a differentiable actor later, but requires renewed feasibility checks.
2. **Decision masking must apply to targets as well as rollout actions.** Between legal boundaries the transition uses the committed plan, with no new planner choice. Replay and target evaluation must retain that held plan rather than compute a fresh minimizer at every physics/control tick. A per-tick expanded state with commitment/phase is the simplest starting point. A boundary-only transition representation would instead need correct duration-dependent discounting and intermediate first-event handling.

The general theory is not intrinsically restricted to torque disturbances: the manuscript formulates `x_next = F(x, u, d)` with an admissible adversary action set. A human planner can instantiate `d` if the expanded state/transition and legality rules are specified. However, its exact finite stationary deterministic-game assumptions do not directly certify a continuous, noisy, partially observed neural-policy simulation. Restricted legal plans define the robustness class being measured; invalid attacks must be accounted for separately rather than counted as protagonist wins or losses without a declared rule.

## Simulator and compute scope

A native MuJoCo 3.11.0 kinematic video uses the accepted engine and MJCF representation, but **does not run the accepted mjlab/MuJoCo-Warp batched physics and camera pipeline**. That distinction must remain explicit in the preview README and evidence. The current Task 08 application's CUDA requirement is not a reason to change its source or claim CPU preview equivalence. GPU import, physical stepping and camera checks belong to the later integration validation.

Follow the existing Slurm pattern: run GPU work inside an explicit allocation, honor scheduler-assigned `CUDA_VISIBLE_DEVICES`, set EGL for rendering, and keep caches/output project-local. The requested target is `mit_normal_gpu` with one L40S. The root-session cluster discovery for this run reported a different exposed Slurm installation without that partition/GPU type; no successful L40S execution should be claimed unless an actual allocation and device inventory verify it. A job script can preserve the requested target while local CPU rendering produces preliminary visual evidence.

## Proposed next stages after the preview

1. **Reference quality:** acquire/verify complete source recordings, annotate full cycles and splits, compare source/target wrists and feet, then decide whether SMPL-X morphology and Mink IK improve the preview enough to justify their extra assets/dependencies.
2. **Physical human:** define biomechanically plausible limits/inertias/effort bounds and collision proxies; implement one clip tracker without root assistance; measure balance, feet, wrist orientation and full recovery before scaling to four families.
3. **Target/TTC and sequences:** add validated immutable plans, continuous-time sampling, physically measured moving-guard completion and all ordered family-pair transitions. The plan's numeric gates are proposed targets, not achieved results.
4. **Two entities:** implement the adapter in this project first, preserve G1-specific state/action indexing, replace scheduled fists with actual gloves, and evaluate a frozen/random valid human-plan distribution.
5. **Adversary:** version the critic/replay/action schemas, implement the legal boundary-only selector and matched fixed-random comparison, then consider expanded planner freedom.

No changes to `robust_sav` are authorized by this assessment. The eventual plug-in contract can be developed in `human-punching` and proposed as a concrete integration patch only when the user requests that integration.
