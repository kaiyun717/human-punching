# Glove rotation audit and simplified appearance

Date: 2026-10-10

The glove transforms already followed the recorded hand rotations correctly. The previous rounded gloves made that rotation difficult to read visually. The revised model uses a plain light-blue mannequin and flatter, rounded gloves with a pale back-of-hand patch to make their orientation visible. The recorded wrist rotations are preserved.

## What was checked

The audit replayed every saved pose from all four preview clips, including the existing 0.25 s source padding at each end: **855 frames and both hands**. It compared actual MuJoCo `data.xmat` hand rotations to the full parent-composed DuoBox rotations after the same initial heading alignment. It then measured each rendered glove component's rotation and position relative to its hand. Checking joint positions alone would miss a dropped terminal wrist rotation; these checks inspect orientation explicitly.

The original previews gave:

| Clip | Padded frames | Active glove maximum rotation from first frame | Active wrist local rotation from first frame |
| --- | ---: | ---: | ---: |
| Jab | 209 | 88.00° | 31.02° |
| Cross | 205 | 92.25° | 16.59° |
| Left hook | 223 | 178.96° | 23.33° |
| Right hook | 218 | 159.36° | 28.71° |

These angles describe the largest orientation difference from the first frame, not accumulated rotation or just forearm pronation. The glove and hand have identical orientation changes. The raw local wrist rotations also vary; they are not identity placeholders.

Across both hands and every clip in the original previews:

- Maximum hand orientation disagreement with the composed source: **3.9 × 10⁻¹² degrees**.
- Maximum change in any glove component's orientation relative to its hand: **3.1 × 10⁻¹⁴ degrees**.
- Maximum change in any glove component's position relative to its hand: **4.5 × 10⁻¹⁶ m**.

These are floating-point noise. The meshes are attached directly to `LeftHand` and `RightHand`, and the hand ball-joint quaternions are populated from the corresponding source local rotations. MuJoCo recenters and rotates imported mesh assets internally, so the audit compares each geom's fixed compiled transform relative to the hand, rather than incorrectly requiring the geom and hand matrices to be identical.

## Appearance changes

The updated mannequin removes eyes, hair, nose, mouth, ears, individual chest details, clothing, and separate toe surfaces. Its head is smooth and its feet are simple solid shapes. The internal source skeleton retains its original 21 joints, including toe joints, to preserve the motion/export contract; there are no visible separate toe meshes.

Each glove now has a flatter rounded shape, a cuff, and a pale patch on its back. The patch and glove share the same hand body, making roll visible without adding artificial wrist motion. Left and right gloves retain their blue and red colors for tracking the two hands.

This checks agreement with the available recorded wrist frames. The dataset does not include a detailed finger skeleton or measured glove surface, so this is not an independent calibration of a particular physical glove's orientation relative to the actor's anatomy.

## Reproduction and regression coverage

From `human-punching`:

```bash
.venv/bin/python -B scripts/audit_glove_rotation.py
.venv/bin/python -B -m pytest tests/test_gloves.py -q
```

The audit reads existing preview MJCF, saved trajectories and the hash-verified DuoBox crops. It prints per-hand numerical results and fails if source orientation or rigid glove attachment exceeds the tolerances. It includes every named glove component, including the new orientation patch.

The 10 regression cases cover:

- Independent 90° rotations about all three wrist axes, for each hand, while the forearm remains fixed. Actual transformed mesh vertices must follow the expected rigid rotation around the wrist.
- Every padded source frame in all four clips, for both hands, including the rendered glove orientation and center after MuJoCo's mesh compilation transforms.

The six synthetic wrist tests do not require DuoBox. The four source integration tests skip when that dataset is unavailable. All tests write temporary models under the project's configured pytest cache.

All development and generated files are confined to `human-punching`. `robust_sav` is a read-only input.

## Regenerated mannequin verification

All 22 project tests passed (including the 10 glove cases). All four revised videos passed complete decoding and asset-hash verification. The saved post-render audit is `outputs/glove_rotation_audit.json`.

Maximum hand/source rotation error: 3.88e-12 degrees. Maximum hand-relative glove orientation drift: 3.1e-14 degrees. Maximum hand-relative glove position drift: 4.04e-16 m.
