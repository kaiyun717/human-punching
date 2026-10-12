# Task 08 contact and safety contract for the articulated human

Date: 11 October 2026. Source: [RobustSafeArrival-Vision5.pdf, physical page 33, §f](../../RobustSafeArrival-Vision5.pdf#page=33). All changes are in `human-punching`; `robust_sav` was read only.

## The user's interpretation is correct

The accepted geometry is a **0.08 m radius glove** and **0.06 m radius head**, with an additional **0.02 m safety margin**. Let `d_g = ||p_glove,g - p_head||` in world coordinates.

| Condition | Accepted definition | Equality |
|---|---|---|
| Physical proxy contact | At least one glove has `d_g <= 0.14 m` | Contact at exactly 0.14 m |
| Safe glove clearance | Both gloves have `d_g >= 0.16 m` | Safe clearance at exactly 0.16 m |
| Signed safety clearance | `min(d_left, d_right) - 0.16 m` | Zero is on the safe boundary |
| Safety violation without contact | At least one glove is closer than 0.16 m, but neither is within 0.14 m | The open 0.14–0.16 m band is unsafe without proxy contact |

This last band is intentional. Contact and safe clearance are not complements. A glove at 0.15 m has violated the required safety margin without touching the physical proxy. “Safe clearance” also is not the full safe set: robot height, joint limits, self-collision, and finite state remain independent requirements. Arrival additionally requires actual punch return and the accepted stance-energy condition.

The PDF requires **both gloves** to remain hazards throughout approach, retraction, guard motion, and waiting. The inactive hand must not be excluded.

## This already exists in `robust_sav`

The current canonical source is [boxing/punches/glove_geometry.py](../../robust_sav/src/boxing/punches/glove_geometry.py). It declares the exact radii, margin, thresholds, and version `task08_head06_gloves08_margin02_v1`. Its wrapper supplies 0.16 m clearance and 0.14 m contact to the shared label implementation. [ResidualCollector](../../robust_sav/src/boxing/training/residual_collector.py) selects this wrapper and evaluates/latches events at 5 ms physics substeps. The [runtime contract](../../robust_sav/src/boxing/perception/runtime_contract.py) also includes the geometry identity.

The shared [older collector defaults](../../robust_sav/src/boxing/training/collector.py) are 0.14 m safety and 0.12 m contact because earlier tasks used 0.06 m gloves. Those are **not the Task 08 operating thresholds**: the Task 08 wrapper explicitly overrides them. No change to `robust_sav` is needed to correct the user's values.

The head center comes from the actual `robot/head_collision` geometry. Its radius is checked to be 0.06 m in the [training plant](../../robust_sav/src/boxing/training/plant.py); the native G1 XML places that sphere at `(0, 0, 0.43)` m in `torso_link`. It is not the robot pelvis, a visual mesh centroid, or an estimated camera detection center.

## Why the earlier targeted videos did not register hits

The previous [standing-G1 experiment](07-standing-g1-targeting.md) deliberately kept the visible rounded glove meshes separate from the larger visible G1 head surface. It used an extra 0.044 m preview clearance above the 0.14 m proxy envelope:

| Preview | Minimum glove/head center distance | Task 08 proxy contact? |
|---|---:|---|
| Jab | 0.184738 m | No |
| Cross | 0.184736 m | No |
| Left hook | 0.183637 m | No |
| Right hook | 0.183755 m | No |

These results were appropriate as near-contact visual previews, but they did not exercise the accepted hit boundary. The visible head is larger than the 0.06 m sphere in some directions, and the block-shaped glove is not an 0.08 m sphere. Consequently, visible mesh overlap and proxy contact can disagree. The correct response is to **keep the accepted 0.14/0.16 m contract** and make the role of each geometry explicit; it is not necessary to redefine contact using the visible head outline.

For the next reference/controller experiments, use 0.08 m spherical glove proxies centered on explicit hand attachments and the original 0.06 m G1 head proxy. Aim the reference into the physical envelope, allowing a small deliberate penetration of that proxy in a kinematic diagnostic so solver tolerance cannot convert a planned hit into a miss. Report the measured distance and first entry, not merely the requested target. If the original visible head mesh is retained, document its mismatch with the proxy or expose an optional diagnostic overlay. Proxy correctness is not evidence that an arbitrary visual mesh is collision free.

## Center attachment and wrist rotation

An articulated human replaces a scheduler-owned glove center with a rigidly attached center:

```text
p_glove_world = p_hand_world + R_hand_world @ p_glove_in_hand
```

The wrist origin is **not** the glove center. MuJoCo's compiled mesh frame also need not equal the input mesh's authoring origin. For the existing 1.80 m preview model, the compiled glove centers relative to the hand body are approximately:

```text
Left:  (+0.06677617, -0.00430814, +0.01077035) m
Right: (-0.06677617, -0.00430814, +0.01077035) m
```

These numbers describe the current generated model, not universal anatomical constants. A revised body model should store its own local attachment offsets in its manifest. Use `data.site_xpos` for a named attachment site or `data.geom_xpos` for its spherical proxy after FK; drive targeting, labels, observations, and any sensing geometry from that same center. Hand rotation must rotate the offset as well as the glove's appearance.

## Three time/contact questions must stay separate

1. **Requested/reference TTC:** freeze the head center at outward launch and find the first reference-glove entry into the 0.14 m envelope. The safety margin does not enter this calculation. Recompute TTC after changing the arm path, glove attachment, morphology, or timing.
2. **Physically executed TTC against that same frozen target:** use the actual tracked glove trajectory against the launch-frozen head. This measures execution and timing fidelity even if G1 subsequently moves.
3. **Observed contact against G1:** compare the actual glove trajectory with the actual moving head. Successful avoidance can make this a miss even when the first two TTCs are finite.

Do not continuously redirect a committed outward punch toward G1's moving head. For a standing head these definitions can coincide only if the physical tracker follows the reference; that coincidence must be measured rather than assumed.

The word “physical” in the Task 08 contact definition describes geometric envelope entry. It does **not** mean the existing simulator applies impact forces: [the sensing scene](../../robust_sav/src/boxing/perception/scene.py) explicitly uses `contype=0` and `conaffinity=0` on the two fist spheres, and its audit rejects added punch contacts. The human can have physical dynamics and foot contacts while glove/head contact remains an analytic event. Simulating striking forces is a separate optional experiment that changes the robot/human transition dynamics; it is unnecessary for preserving the current punch-avoidance objective.

## Implementation in `human-punching`

[contact.py](../src/human_punching/contact.py) provides the geometry without importing `robust_sav` or Torch:

- `evaluate_clearance(glove_centers, head_center)`: NumPy broadcasting over batches/frames, both gloves, per-glove signed margins, any-contact and all-gloves-clear labels. Nonfinite geometry is rejected.
- `world_attached_centers(body_positions, body_rotations, local_offsets)`: the explicit rigid attachment transform.
- `first_contact_time(time_s, glove_centers, head_center)`: first entry along piecewise-linear relative-center segments, including crossings between samples; accepts a fixed launch head or a moving head trajectory and returns infinity on a miss. The output uses the supplied absolute clock; subtract launch time for TTC.

The last routine is an offline diagnostic. Interpolated centers approximate the inter-frame motion; they are not a continuous articulated-trajectory certificate or a replacement for actual substep event evaluation. The geometry utility does not itself alter the earlier targeting exports, add collision response, or implement the remaining safe-arrival predicates.

Validation: **10 tests passed** in [test_contact.py](../tests/test_contact.py), covering both equality boundaries, the unsafe/non-contact band, inactive-hand hazards, moving versus frozen targets, rotated attachment offsets, inter-frame sphere crossings, tangent contact, initial contact, misses, and invalid input. Run from `human-punching`:

```bash
.venv/bin/python -m pytest tests/test_contact.py -q
```
