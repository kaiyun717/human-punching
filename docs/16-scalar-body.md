# Scalar-joint training body: constrained SOMA23 derivative

12 October 2026. Body asset milestone for Task 09A. This report describes the asset and compilation checks; successful learned tracking and physical guard recovery are separate acceptance gates.

## Choice and source

The Task 09 PDF chooses SMPL-X through SMPLSim first and explicitly allows SOMA23 as a fallback. We checked the local workspace for standard SMPL/SMPL-X body-model filenames; none was found. DuoBox's fitted pose/joint files do not supply the separate parametric body-model assets required by the SMPLSim exporter. We therefore used the planned SOMA23 fallback instead of making those assets a prerequisite for the learning pilot.

The source is [ProtoMotions SOMA23](https://github.com/NVlabs/ProtoMotions/blob/7a8417a9f55b1586b0d6e8f46578c5bc9d7ce1f1/protomotions/data/assets/mjcf/soma23_humanoid.xml), pinned to revision `7a8417a9f55b1586b0d6e8f46578c5bc9d7ce1f1`. The original XML, reference config, Apache-2.0 license, third-party notices and file hashes are retained in [assets/vendor/protomotions](../assets/vendor/protomotions/manifest.json). No ProtoMotions runtime or pretrained controller is imported. The source config identifies the body as a 23-body reduction of SOMASkeleton30; this is not an SMPL-X export and does not include SOMA's detailed skinned surface.

Per the user's updated preference, native bone lengths were kept. The modified neutral body measures **1.7664 m** from the lower sole to the top of the simple head envelope. There is no forced rescaling to exactly 1.80 m. The chosen total mass is **75 kg**, including gloves. This mass is an experiment parameter, not an estimate of the recorded actor's weight.

## What was changed

The [builder](../src/human_punching/scalar_body.py) retains all 23 named bodies but reduces the source's 66 hinges to **48 scalar hinges**. There is one unactuated free pelvis: `nq=55`, `nv=54`, `nu=48`.

| Region | Controlled coordinates | Rationale |
|---|---|---|
| Spine1, Spine2, Chest | Three per link, 9 total | Bounded flexion, lateral bend and twist |
| Neck1 | Three | Neck2 and Head are fixed relative links; one neck joint supplies head motion |
| Shoulders and upper arms | Three per link, 12 total | Preserve shoulder-girdle motion and arm orientation |
| Forearms | Elbow flexion followed by axial pronation, 4 total | Remove nonphysical sideways elbow hinge; keep forearm roll |
| Wrists | Three per hand, 6 total | Preserve glove orientation, with a small wrist axial range |
| Hips | Three per leg, 6 total | Hip motion |
| Knees | One flexion hinge per leg, 2 total | Remove sideways bending and free axial knee rotation |
| Ankles | Three per foot, 6 total | Bounded foot orientation; toes are fixed hierarchy landmarks |

Canonical body axes are +Z up, −Y forward and +X left. Unlike the original asset's mirrored left X axes, all exported hinge axes use positive XYZ directions. The neutral free-root quaternion rotates the body 90 degrees around Z, producing **world +X forward and +Y left**. The forearm joint order is elbow Z then pronation X, so pronation occurs about the bent forearm instead of changing its flexion plane. Right elbow flexion is positive Z; left is negative Z. Both knee flexions are positive X. Tests verify the resulting FK directions, not just joint labels.

The model uses simple light-blue ellipsoids/capsules and whole-foot boxes, with no eyes, mouth, hair or detailed toes. Red gloves follow the complete hand transform. Each glove is a radius-0.08 m sphere centered on the corresponding `LeftHand_center` or `RightHand_center` site; the small dark cuff makes wrist rotation visible. These exact sites, not wrist origins, define the glove centers for the 0.14 m head-contact contract. `LeftFoot_sole` and `RightFoot_sole`, plus four sole-corner sites per foot, expose support geometry.

The original primitive envelopes and uniform-density masses were replaced. Per-segment masses are explicitly assigned and normalized to 75 kg; MuJoCo integrates the inertia from the selected physical primitives. The builder exports all masses/inertias through the compiled MJCF. The two neutral soles differ by 0.3 mm because the original skeleton is slightly asymmetric.

## Actuation and collision contract

Every motor has gear 1, symmetric torque control and force limits. The XML has **zero passive spring stiffness, zero passive damping, and zero friction loss**. All requested restoring/damping torque belongs to the explicit bounded controller. Joint armature is 0.02 kg m². This avoids hidden source springs adding hundreds of Nm outside the controller's declared effort budget.

The initial effort caps are 100–120 Nm per trunk hinge, 20 Nm neck, 35 Nm clavicle, 100 Nm upper arm, 60 Nm elbow/pronation, 15 Nm wrist, 200 Nm hip, 150 Nm knee and 90 Nm ankle. These are per-coordinate engineering limits, not a measured strength envelope or a vector-norm cap for each anatomical joint. The gain/effort arrays, ordered joint names, coordinate addresses and angle limits are saved in [scalar_model.json](../outputs/scalar-body/scalar_model.json). Model-based reference preparation and training must use that same ordering and body hash.

All human primitives collide with the floor. Thirteen explicit self-collision pairs cover each glove against head, chest, lower trunk, pelvis and opposite forearm; both gloves; both shins; and both feet. Adjacent anatomical envelopes are not tested against one another because they intentionally overlap. This is selected self-collision coverage, not a complete self-collision model. There is no G1 impact body in this standalone asset.

The builder defaults to 5 ms for compatibility experiments and accepts a finer timestep such as 2 ms. Compilation and 0.15-second bounded-control stepping passed at both. **That short check does not establish balanced standing, jab tracking or recovery at either timestep.** Those must be measured using the retargeted reference and learned/controller execution.

## Reproduction and validation

From `human-punching`:

```bash
.venv/bin/python -m human_punching.scalar_body --output outputs/scalar-body --timestep 0.005
.venv/bin/python -m pytest tests/test_scalar_body.py -q
```

The generated [human.xml](../outputs/scalar-body/human.xml) is self-contained; no mesh files or body-model package is needed to load it. The API `build_scalar_human(output, timestep=.005, mass_kg=75.)` returns the same metadata written to `scalar_model.json`, including the standing T-pose `neutral_qpos`, body/site maps and controller arrays.

Six tests passed. They verify pinned-file hashes, native height and mass, positive-definite full mass matrix, positive moving-link inertias and inertia triangle inequalities, one-to-one scalar-joint/motor ordering, actual torque clipping, zero root actuation and external force, knee/elbow FK directions, hand-attached glove centers/orientations, selected self-collision declarations, floor contact and finite short physical integration at both timesteps. Ruff passed on the builder and tests.

This makes the body usable for the next learning experiment, while keeping the remaining assumptions visible. Anatomical ranges, mass distribution and strengths are approximate; shoulder anatomy is simplified; rigid soles cannot reproduce toe mechanics; only selected self collisions are enabled. A learned controller can exploit these approximations, so accepted rollout metrics and visual review remain necessary. A realistic skin surface would not resolve these physical limitations.
