# Contact-directed G1 punch references

Date: 11 October 2026. This experiment corrects the objective of the earlier visual near-contact preview while preserving the accepted Task 08 radii. No changes were made to `robust_sav`.

## Result

All four recorded punch families now enter the **0.14 m glove/head proxy-contact envelope** against the standing G1. The target center separation is 0.135 m, giving the IK solver 5 mm of nominal overlap tolerance. This does not change the 0.08 m glove radius, 0.06 m head radius, or 0.16 m safety boundary. Both gloves are measured throughout every frame.

| Reference | Closest active-glove center distance | First contact from clip start | Sampled frames in contact | Maximum IK path error |
|---|---:|---:|---:|---:|
| Jab | 0.135798 m | 0.762704 s | 13 | 0.565 mm |
| Cross | 0.135854 m | 0.627309 s | 25 | 0.564 mm |
| Left hook | 0.134628 m | 0.849237 s | 4 | 0.437 mm |
| Right hook | 0.134932 m | 0.732864 s | 3 | 10.227 mm |

The inactive glove remains outside the contact envelope in all four base references; its closest distances are approximately 0.564–0.619 m. All active contacts also violate the 0.16 m clearance requirement, as expected. These labels describe the glove/head geometry; they do not certify the G1's other safe-arrival conditions.

The contact time uses a head frozen at clip start and first sphere entry along piecewise-linear interpolation of the exported center path. **Clip start is a diagnostic commitment time, not a newly validated launch-phase annotation.** The source timing is retained; no requested TTC is imposed. The saved 120 Hz samples also contain actual proxy-contact frames, so acceptance does not depend solely on inter-frame interpolation.

## Finite geometric variants

The bank contains **12 references**: the four families, each with approach-direction yaw offsets of −6°, 0°, and +6°. The yaw rotates the target normal around the frozen head while keeping its target distance at 0.135 m. It does not rotate the robot or change its placement. All 12 variants pass sampled proxy-contact checks and have finite exported states.

This provides a small within-family choice for later experiments. “Accepted” here means **geometrically reaches the contact envelope**, not dynamically feasible, anatomically validated, or approved for a trained adversary. These references retain source lower-body motion and unrestricted arm ball joints. Physical feasibility, support-foot constraints, bounded actuation, and recovery are separate acceptance gates.

## Implementation and preserved behavior

[aim_clip](../src/human_punching/targeting.py) now accepts `target_mode="proxy_contact"`, `contact_center_distance_m=0.135`, and an optional approach yaw. Its existing `visual_clearance` default remains unchanged, so the older near-contact rendering commands retain their previous behavior. Smooth path deformation, source-orientation regularization, exact guard-endpoint preservation, and unchanged nonselected joints remain in effect.

The solver still allows the original source root, legs and torso to move as recorded. It changes only the active arm's four ball joints. The right hook has a larger transient path error than the other families; its closest-approach contact still passes. This stage should not be represented as a full-body constrained trajectory optimizer.

[contact_target_cli.py](../src/human_punching/contact_target_cli.py) reuses the existing blue-human model, source trajectories, G1 scene, and frozen target; it does not require regenerating assets or installing learning libraries. It exports:

- `human_trajectory.npz`: the human's 87-coordinate reference, corresponding velocities/body states, both glove centers, distances and contact/safety labels.
- `trajectory.npz`: the same reference scattered into the combined G1/human scene, with every G1 coordinate held fixed.
- `manifest.json`: geometry identity, measured first contact, both-glove minima, attachment offsets, solver errors, and hashes.

The shared geometry implementation and exact equality conventions are documented in [10-contact-contract.md](10-contact-contract.md).

## Evidence and reproduction

Generated bank: [outputs/contact-targeting](../outputs/contact-targeting/manifest.json). Independent combined-scene checks: [validation.json](../outputs/contact-targeting/validation.json). Each reference's compiled combined-scene glove centers agree with the human-only export to better than `1e-10 m`; all G1 coordinates are unchanged and all numeric state arrays are finite.

From `human-punching`:

```bash
# Export four base references plus two yaw variants per family.
.venv/bin/python -m human_punching.contact_target_cli --approach-yaws-deg -6 0 6

# Render the four base references; use EGL inside an allocated GPU job.
.venv/bin/python -m human_punching.contact_target_cli \
  --render --backend egl --output outputs/contact-targeting-rendered

# CPU rendering is also available with --backend osmesa.
.venv/bin/python -m pytest tests/test_contact.py tests/test_targeting.py -q
```

The contact/targeting suite passes **20 tests**, including all previous targeting tests, four real-clip contact-mode checks, invalid targets, and the contact contract regressions. Ruff passes for all modified source and test files.

The rendered block glove and original visible head can visibly overlap at proxy contact because they do not share the analytic spheres' boundaries. No impact force or balance claim follows from these kinematic references. The subsequent physical-model experiments should retain the explicit proxy centers/radii, use physical foot/body dynamics where needed, and keep analytic contact-event detection separate from any optional impact-response experiment.

## Completed GPU render

Slurm job **5890904** completed with exit code `0:0` in 15 seconds on a Tesla V100 (`d-8-4-1`). The preflight confirmed native NVIDIA EGL on the allocated GPU; all four videos fully decoded without errors.

[Overview](../outputs/contact-videos-5890904/preview/overview.png) · [Jab](../outputs/contact-videos-5890904/preview/jab/jab.mp4) · [Cross](../outputs/contact-videos-5890904/preview/cross/cross.mp4) · [Left hook](../outputs/contact-videos-5890904/preview/left_hook/left_hook.mp4) · [Right hook](../outputs/contact-videos-5890904/preview/right_hook/right_hook.mp4)

Evidence: [GPU runtime](../outputs/contact-videos-5890904/gpu_runtime.json), [media validation](../outputs/contact-videos-5890904/media-validation.json), and [Slurm accounting](../outputs/contact-videos-5890904/slurm-accounting.txt). Submit from `human-punching` with `sbatch scripts/contact_targets_volta.sbatch`. These videos deliberately retain the original kinematic lower-body motion; visible hovering remains and is not evidence of floor-supported execution.
