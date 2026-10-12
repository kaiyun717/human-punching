# DuoBox motion source and retargeting audit

Date: 2026-10-10. This report concerns the input motion adapter. All source reads are from the sibling `DuoBox` and `robust_sav` trees; all development is inside `human-punching`.

## Available source and selected events

The supplied `DuoBox/fbx_export_joints` recordings contain scalar NumPy dictionaries. For the selected recordings, `ma_position` and `liu_position` are arrays shaped `(frames, 21, 3)`; the corresponding rotation arrays are `(frames, 21, 3, 3)`. Both actors are present in each file. The attacking actor for these four selected events is `ma`; pairing order must not determine actor identity.

The selections are inherited exactly from [`robust_sav/configs/boxing/punch_crops.json`](../../robust_sav/configs/boxing/punch_crops.json), including file hashes, inclusive source frame bounds and anatomical hand. That configuration explicitly marks these as machine-selected candidates pending visual and human label review. This project preserves that status rather than promoting the labels to validated ground truth.

| Preview name | Source filename under `fbx_export_joints` | Inclusive frames | Samples | Elapsed motion | Anatomical hand / wrist joint |
| --- | --- | --- | --- | --- | --- |
| jab | `000_ma_jab_liu_defense_joints.npy` | 370–518 | 149 | 1.2333 s | left / 8 |
| cross | `003_ma_cross_liu_defense_joints.npy` | 223–367 | 145 | 1.2000 s | right / 12 |
| left_hook | `006_ma_leadhook_liu_shanbi_joints.npy` | 401–563 | 163 | 1.3500 s | left / 8 |
| right_hook | `007_ma_rearhook_liu_shanbi_joints.npy` | 463–620 | 158 | 1.3083 s | right / 12 |

The frame convention is 120 Hz, inherited from the existing `robust_sav` source audit. No timestamps or per-file rate metadata are stored in these NumPy dictionaries, so this is a source-code convention, not an independently calibrated clock. Source positions are interpreted in metres based on the inherited convention and measured human-scale limbs.

The crops include the machine-annotated outward motion and recovery to a candidate guard reacquisition. Optional `padding_s=0.25` adds 30 real recorded frames at each end, producing 209, 205, 223 and 218 samples respectively. Padding never invents a pose; it is clipped to recording bounds. The final pose is not forcibly matched to the initial pose, so concatenating the clip into an endless seamless loop is not claimed.

## Coordinate and rotation checks

`Hips` is joint 0. Positions use a right-handed world with +z up. In the skeleton's neutral local frame, +y is up, +x is anatomical left and +z is forward. The hierarchy is:

```text
Hips
├── Spine → Spine1 → Neck → Head
│           ├── LeftShoulder → LeftArm → LeftForeArm → LeftHand
│           └── RightShoulder → RightArm → RightForeArm → RightHand
├── LeftUpLeg → LeftLeg → LeftFoot → LeftToeBase
└── RightUpLeg → RightLeg → RightFoot → RightToeBase
```

Only the root's rotation is directly a world rotation. Every child's stored rotation is parent-relative. Column-vector composition is therefore

```text
R_world[root] = R_source[root]
R_world[j] = R_world[parent[j]] @ R_source[j]
p_world[j] = p_world[parent[j]] + R_world[parent[j]] @ rest_offset[j]
```

The adapter recovers each constant rest offset as the component-wise median of `R_world[parent].T @ (p_child - p_parent)` across the selected frames. Treating the child matrices as world rotations instead produces maximum reconstruction errors of 1.64–1.76 m on these crops. Correct composition reconstructs the supplied positions to submicrometre precision:

| Punch | Maximum reconstructed joint error | Maximum rest-offset deviation |
| --- | --- | --- |
| jab | 2.865 × 10⁻⁷ m | 3.151 × 10⁻⁷ m |
| cross | 2.902 × 10⁻⁷ m | 3.500 × 10⁻⁷ m |
| left_hook | 2.913 × 10⁻⁷ m | 3.260 × 10⁻⁷ m |
| right_hook | 3.281 × 10⁻⁷ m | 2.911 × 10⁻⁷ m |

Stored matrices have small export noise: maximum entrywise orthogonality error is approximately 1.36 × 10⁻⁶. Conversion to normalized quaternions projects this noise onto proper rotations. The source adapter retains the recorded matrices and exposes numerical diagnostics so later replay tolerances can distinguish export noise from frame-convention errors.

For actor `ma`, measured neutral segment lengths are approximately 0.500378 m thigh, 0.421891 m shin, 0.303919 m upper arm and 0.251426 m forearm. Summing neutral offsets places the head joint at local y=0.623897 m relative to the pelvis, feet at y=−0.922269 m and toes at y=−0.987408 m. These are joint locations, **not scalp/sole vertices**; exact 1.80 m body height must be measured from the generated body surface. The mesh builder applies one uniform scale to the body, source offsets and root displacement after accounting for scalp and sole geometry. It does not independently shrink arms, lower a fist, or aim at the G1.

## Data contract and integrity

[`motion.py`](../src/human_punching/motion.py) exports `JOINT_NAMES`, `PARENTS`, `FPS`, `CROP_SPECS`, `MotionClip`, `load_crop`, `compose_global_rotations` and `forward_kinematics`. `MotionClip` exposes original positions, local and global rotation matrices, recovered rest offsets, source frame numbers, relative sample times and provenance metadata. `active_hand_index` is the **21-joint skeleton index** (8 or 12), whereas the inherited configuration also has a separate two-hand-array index (0 or 1).

The raw files are pickle-backed, so the loader verifies the SHA256 of the exact source bytes against the inherited allowlist before enabling NumPy pickle loading. Loading uses those same verified bytes; replacement files fail before deserialization. Derived numeric archives can be loaded with `allow_pickle=False`. The adapter does not import `robust_sav`, Torch, SMPL or any dataset preprocessing/training stack.

| Punch | Verified SHA256 |
| --- | --- |
| jab | `7ed040b3b727baa108b5090872fab864d684ff1152031a746da07b6b260e17a3` |
| cross | `4295973d6348916cd820d8e3de1ff091af6eaf25b40b5d4c81ad4f2e6282f39f` |
| left_hook | `790fb445dd9dd002756f9902f29730e69a40de8259a8c6add7bf4bd3f48de23b` |
| right_hook | `86e378bfbd31779fa65ceac9d4d26d3d9bc0c3aedca74eebe0b027f9b4dfcfb3` |

## Validation performed

Six tests in [`tests/test_motion.py`](../tests/test_motion.py) passed using the project environment. They verify nontrivial parent rotation/child translation behavior, hierarchy and hand indices, input validation, rejection of modified input before pickle loading, reconstruction of all four real source crops, and exact preservation of the original frames under optional context padding. Source-dependent tests skip clearly when the external DuoBox directory is unavailable; they do not claim validation on absent data.

```bash
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_motion.py -v
```

This validates a motion source and its geometric replay contract. It does not validate punch style labels, biomechanical joint limits, contact forces, dynamically stable tracking, perception realism, target adaptation or adversarial policy learning. Those remain separate integration stages.
