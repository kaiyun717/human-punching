"""Load the four provenance-checked DuoBox crops without legacy model dependencies.

The input is a raw OptiTrack skeleton: positions in world z-up metres and
parent-relative rotation matrices. The root rotation maps its local frame to
world. No SMPL model, Torch import, inverse kinematics, or target aiming is used.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FPS = 120.0
JOINT_NAMES = (
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
)
PARENTS = np.array([-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])

# Source hashes, anatomical hands, and inclusive frame bounds inherited from
# robust_sav/configs/boxing/punch_crops.json (selected_source_crops_v1).
# These remain machine-selected candidates pending human label review.
CROP_SPECS: dict[str, dict[str, Any]] = {
    "jab": {
        "filename": "000_ma_jab_liu_defense_joints.npy",
        "actor": "ma",
        "first_frame": 370,
        "last_frame": 518,
        "active_hand_index": 8,
        "anatomical_hand": "left",
        "original_named_drill": "jab",
        "source_sha256": "7ed040b3b727baa108b5090872fab864d684ff1152031a746da07b6b260e17a3",
        "candidate_id": "jb_000_ma_00370_00518",
        "outward_completion_frame": 432,
    },
    "cross": {
        "filename": "003_ma_cross_liu_defense_joints.npy",
        "actor": "ma",
        "first_frame": 223,
        "last_frame": 367,
        "active_hand_index": 12,
        "anatomical_hand": "right",
        "original_named_drill": "cross",
        "source_sha256": "4295973d6348916cd820d8e3de1ff091af6eaf25b40b5d4c81ad4f2e6282f39f",
        "candidate_id": "cr_003_ma_00223_00367",
        "outward_completion_frame": 292,
    },
    "left_hook": {
        "filename": "006_ma_leadhook_liu_shanbi_joints.npy",
        "actor": "ma",
        "first_frame": 401,
        "last_frame": 563,
        "active_hand_index": 8,
        "anatomical_hand": "left",
        "original_named_drill": "leadhook",
        "source_sha256": "790fb445dd9dd002756f9902f29730e69a40de8259a8c6add7bf4bd3f48de23b",
        "candidate_id": "lh_006_ma_00401_00563",
        "outward_completion_frame": 472,
    },
    "right_hook": {
        "filename": "007_ma_rearhook_liu_shanbi_joints.npy",
        "actor": "ma",
        "first_frame": 463,
        "last_frame": 620,
        "active_hand_index": 12,
        "anatomical_hand": "right",
        "original_named_drill": "rearhook",
        "source_sha256": "86e378bfbd31779fa65ceac9d4d26d3d9bc0c3aedca74eebe0b027f9b4dfcfb3",
        "candidate_id": "rh_007_ma_00463_00620",
        "outward_completion_frame": 528,
    },
}


@dataclass(frozen=True)
class MotionClip:
    punch: str
    positions: np.ndarray
    local_rotations: np.ndarray
    global_rotations: np.ndarray
    rest_offsets: np.ndarray
    time_s: np.ndarray
    source_frames: np.ndarray
    metadata: dict[str, Any]

    @property
    def fps(self) -> float:
        return float(self.metadata.get("source_rate_hz", FPS))

    @property
    def duration_s(self) -> float:
        """Elapsed time from the first sample to the last, excluding frame hold."""
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def active_hand_index(self) -> int:
        """Joint index (8=left wrist, 12=right wrist), not a two-hand array index."""
        return int(self.metadata["active_hand_joint_index"])


def compose_global_rotations(local: np.ndarray) -> np.ndarray:
    """Compose column-vector rotations: R_world,j = R_world,parent @ R_parent,j."""
    local = np.asarray(local, dtype=np.float64)
    if local.shape[-3:] != (len(JOINT_NAMES), 3, 3):
        raise ValueError("Expected (..., 21, 3, 3) local rotations")
    result = np.empty_like(local)
    for joint, parent in enumerate(PARENTS):
        result[..., joint, :, :] = (
            local[..., joint, :, :]
            if parent < 0
            else (result[..., parent, :, :] @ local[..., joint, :, :])
        )
    return result


def forward_kinematics(
    root_positions: np.ndarray, global_rotations: np.ndarray, rest_offsets: np.ndarray
) -> np.ndarray:
    """Reconstruct positions using each parent orientation and child rest offset."""
    root_positions = np.asarray(root_positions, dtype=np.float64)
    global_rotations = np.asarray(global_rotations, dtype=np.float64)
    rest_offsets = np.asarray(rest_offsets, dtype=np.float64)
    if global_rotations.shape[-3:] != (21, 3, 3) or rest_offsets.shape != (21, 3):
        raise ValueError("Expected rotations (...,21,3,3) and offsets (21,3)")
    if root_positions.shape != global_rotations.shape[:-3] + (3,):
        raise ValueError("Root positions and rotations must have matching frame dimensions")
    result = np.empty(global_rotations.shape[:-1], dtype=np.float64)
    result[..., 0, :] = root_positions
    for joint, parent in enumerate(PARENTS[1:], start=1):
        result[..., joint, :] = result[..., parent, :] + np.einsum(
            "...ab,b->...a", global_rotations[..., parent, :, :], rest_offsets[joint]
        )
    return result


def _verified_dictionary(path: Path, expected_sha256: str) -> tuple[dict, str]:
    # Raw NPY scalar dictionaries are pickle-backed. Verify the exact known
    # bytes *before* allowing pickle, and load those same bytes to avoid a race.
    import io

    source_bytes = path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Source SHA256 mismatch for {path.name}: {digest}")
    value = np.load(io.BytesIO(source_bytes), allow_pickle=True)
    if value.shape != () or not isinstance(value.item(), dict):
        raise ValueError("Expected a scalar DuoBox raw dictionary")
    return value.item(), digest


def load_crop(dataset_root: str | Path, punch: str, padding_s: float = 0.0) -> MotionClip:
    """Read a selected source crop, optionally including recorded guard context.

    ``dataset_root`` is DuoBox itself, containing ``fbx_export_joints``. Bounds
    are inclusive. Padding adds existing frames and clips at recording edges;
    it does not synthesize poses or blend the endpoints into a loop.
    """
    if punch not in CROP_SPECS:
        raise ValueError(f"Unknown punch {punch!r}; choose one of {tuple(CROP_SPECS)}")
    if not np.isfinite(padding_s) or padding_s < 0:
        raise ValueError("padding_s must be finite and nonnegative")
    spec = CROP_SPECS[punch]
    path = Path(dataset_root).expanduser().resolve() / "fbx_export_joints" / spec["filename"]
    data, digest = _verified_dictionary(path, spec["source_sha256"])
    actor = spec["actor"]
    positions = np.asarray(data[f"{actor}_position"], dtype=np.float64)
    local = np.asarray(data[f"{actor}_rotation"], dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (21, 3):
        raise ValueError("Expected source positions (frames,21,3)")
    if local.shape != positions.shape[:2] + (3, 3):
        raise ValueError("Expected source rotations (frames,21,3,3)")
    if not np.isfinite(positions).all() or not np.isfinite(local).all():
        raise ValueError("Source positions/rotations contain nonfinite values")
    padding = int(round(float(padding_s) * FPS))
    first = max(0, spec["first_frame"] - padding)
    last = min(len(positions) - 1, spec["last_frame"] + padding)
    if spec["last_frame"] >= len(positions):
        raise ValueError("Source recording does not contain selected crop")
    positions = positions[first : last + 1].copy()
    local = local[first : last + 1].copy()
    global_rotations = compose_global_rotations(local)
    local_offsets = np.zeros_like(positions)
    for joint, parent in enumerate(PARENTS[1:], start=1):
        local_offsets[:, joint] = np.einsum(
            "tba,tb->ta", global_rotations[:, parent], positions[:, joint] - positions[:, parent]
        )
    rest_offsets = np.median(local_offsets, axis=0)
    reconstructed = forward_kinematics(positions[:, 0], global_rotations, rest_offsets)
    fk_error = np.linalg.norm(reconstructed - positions, axis=-1)
    orthogonal_error = np.max(np.abs(np.swapaxes(local, -1, -2) @ local - np.eye(3)))
    determinant_error = np.max(np.abs(np.linalg.det(local) - 1))
    if orthogonal_error > 1e-5 or determinant_error > 1e-5:
        raise ValueError("Source rotations are not proper orthonormal matrices")
    if float(fk_error.max()) > 1e-4:
        raise ValueError("Source skeleton does not fit constant parent-local rest offsets")
    frames = np.arange(first, last + 1, dtype=np.int64)
    metadata = {
        "schema": "human_punching_duobox_crop_v1",
        "punch": punch,
        "candidate_id": spec["candidate_id"],
        "actor": actor,
        "source_path": str(path),
        "source_sha256": digest,
        "source_rate_hz": FPS,
        "source_frame_range_inclusive": [first, last],
        "selected_frame_range_inclusive": [spec["first_frame"], spec["last_frame"]],
        "outward_completion_frame": spec["outward_completion_frame"],
        "padding_requested_s": float(padding_s),
        "anatomical_hand": spec["anatomical_hand"],
        "active_hand_joint_index": spec["active_hand_index"],
        "original_named_drill": spec["original_named_drill"],
        "coordinate_system": "world z-up, metres; local bind +y up, +x anatomical left, +z forward",
        "rotation_convention": "column vectors; global child = global parent @ local child",
        "label_status": "inherited machine candidate; human label review pending",
        "retargeting": "none; source skeleton only",
        "fk_max_error_m": float(fk_error.max()),
        "fk_rms_error_m": float(np.sqrt(np.mean(fk_error**2))),
        "rest_offset_max_deviation_m": float(
            np.linalg.norm(local_offsets - rest_offsets, axis=-1).max()
        ),
        "rotation_orthogonality_max_error": float(orthogonal_error),
        "rotation_determinant_max_error": float(determinant_error),
    }
    return MotionClip(
        punch,
        positions,
        local,
        global_rotations,
        rest_offsets,
        np.arange(len(frames), dtype=np.float64) / FPS,
        frames,
        metadata,
    )
