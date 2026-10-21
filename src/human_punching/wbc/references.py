"""Auditable, CPU-only reference conversion; no dynamics or training qualification.

The 200 Hz master owns the physical reference clock. The WBC file is its every
fourth sample, with entity-local body ordering (world excluded) and wxyz quats.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .provenance import writable_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FAMILIES = ("jab", "cross", "left_hook", "right_hook")
PHYSICS_DT = 0.005
POLICY_DT = 0.020


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_output(path: Path) -> Path:
    """Resolve symlinks before any write; never accept a sibling destination."""
    return writable_path(path)


def _array(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    if not np.isfinite(result).all():
        raise ValueError("Reference arrays must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReferenceClip:
    family: str
    model_path: Path
    time_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    target_w: np.ndarray
    launch_time_s: float
    guard_time_s: float
    source_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError("Unknown punch family")
        object.__setattr__(self, "model_path", Path(self.model_path).resolve())
        for name in ("time_s", "qpos", "qvel", "target_w"):
            object.__setattr__(self, name, _array(getattr(self, name)))
        t = self.time_s
        if t.ndim != 1 or len(t) < 2 or t[0] != 0 or np.any(np.diff(t) <= 0):
            raise ValueError("Reference timestamps must increase from zero")
        if self.qpos.shape != (len(t), 55) or self.qvel.shape != (len(t), 54):
            raise ValueError("Expected constrained human nq=55, nv=54")
        if self.target_w.shape != (3,):
            raise ValueError("Target must be a world-frame point")
        if not 0 <= self.launch_time_s < self.guard_time_s <= t[-1] + 1e-9:
            raise ValueError("Invalid launch/guard ordering")
        if not np.allclose(np.linalg.norm(self.qpos[:, 3:7], axis=1), 1, atol=1e-5):
            raise ValueError("Root quaternions must be normalized wxyz")
        object.__setattr__(self, "metadata", dict(self.metadata))


def load_release_reference(release_root: Path, family: str) -> ReferenceClip:
    """Read only present, hash-verified release payloads; no historical defaults."""
    if family not in FAMILIES:
        raise ValueError("Unknown punch family")
    root = Path(release_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for name in ("reference.npz", "model/human.xml", "body_metadata.json",
                 "controller_schema.json", "reference_manifest.json"):
        relative = f"{family}/{name}"
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Release path escapes its root")
        expected = manifest["artifacts"][relative]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"Release hash mismatch: {relative}")
    directory = root / family
    with np.load(directory / "reference.npz", allow_pickle=False) as source:
        times = source["time_s"].copy()
        qpos, qvel = source["qpos"].copy(), source["qvel"].copy()
        target = source["head_center"].copy()
        launch = float(source["launch_time"])
        guard = float(source["guard_hold_start"])
    body = json.loads((directory / "body_metadata.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(directory / "model/human.xml"))
    validate_model_order(model, body)
    return ReferenceClip(family, directory / "model/human.xml", times, qpos, qvel,
                         target, launch, guard, sha256(directory / "reference.npz"),
                         {"origin": "task09_frozen_reference", "requested_ttc_s": None,
                          "g1_snapshot_present": False, "qualified_for_training": False,
                          "body_metadata": body})


def validate_model_order(model: mujoco.MjModel, body: dict[str, Any]) -> None:
    if (model.nq, model.nv, model.nu) != (55, 54, 48):
        raise ValueError("Expected the constrained 48-hinge physical human")
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Human root must remain free")
    if np.any(model.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Only scalar hinge limbs are supported")
    names = [model.joint(i).name for i in range(1, model.njnt)]
    if names != body["joint_names"]:
        raise ValueError("Joint order differs from body metadata")
    for key, actual in (("qpos_addresses", model.jnt_qposadr[1:]),
                        ("dof_addresses", model.jnt_dofadr[1:])):
        if not np.array_equal(actual, body[key]):
            raise ValueError(f"Incorrect {key}")
    if not np.array_equal(model.actuator_trnid[:, 0], np.arange(1, model.njnt)):
        raise ValueError("Actuator order must equal scalar joint order")
    if not np.isclose(model.body_mass.sum(), body["mass_kg"], atol=1e-8):
        raise ValueError("Body mass differs from metadata")
    if any(model.body(name).id != index for name, index in body["body_map"].items()):
        raise ValueError("Body order differs from metadata")


def interpolate_poses(times: np.ndarray, poses: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Scalar interpolation and shortest-arc SLERP, with constant end holds."""
    query = np.clip(np.asarray(query), times[0], times[-1])
    result = np.column_stack([np.interp(query, times, poses[:, j])
                              for j in range(poses.shape[1])])
    rotations = Rotation.from_quat(poses[:, [4, 5, 6, 3]])
    result[:, 3:7] = Slerp(times, rotations)(query).as_quat()[:, [3, 0, 1, 2]]
    # Consistent signs prevent interpolation/export consumers seeing artificial flips.
    for i in range(1, len(result)):
        if result[i - 1, 3:7] @ result[i, 3:7] < 0:
            result[i, 3:7] *= -1
    return result


def tangent_velocities(model: mujoco.MjModel, times: np.ndarray,
                       poses: np.ndarray) -> np.ndarray:
    """Differentiate within one clip; free-root angular velocity is local tangent."""
    velocities = np.empty((len(times), model.nv))
    for i in range(len(times)):
        left, right = max(0, i - 1), min(len(times) - 1, i + 1)
        mujoco.mj_differentiatePos(model, velocities[i], float(times[right] - times[left]),
                                 poses[left], poses[right])
    return velocities


def resample_reference(clip: ReferenceClip, dt: float = PHYSICS_DT) -> ReferenceClip:
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Sampling interval must be positive")
    # Pad with terminal guard rather than changing the physical motion duration.
    count = int(np.ceil((clip.time_s[-1] - 1e-10) / dt)) + 1
    times = np.arange(count, dtype=np.float64) * dt
    poses = interpolate_poses(clip.time_s, clip.qpos, times)
    model = mujoco.MjModel.from_xml_path(str(clip.model_path))
    velocity = tangent_velocities(model, times, poses)
    metadata = {**clip.metadata, "source_duration_s": float(clip.time_s[-1]),
                "sample_dt_s": dt, "terminal_padding_s": float(times[-1]-clip.time_s[-1]),
                "velocity_method": "per-clip MuJoCo tangent central/one-sided differences"}
    return ReferenceClip(clip.family, clip.model_path, times, poses, velocity,
                         clip.target_w, clip.launch_time_s, clip.guard_time_s,
                         clip.source_sha256, metadata)


def kinematic_arrays(clip: ReferenceClip) -> dict[str, np.ndarray]:
    """CPU FK/Jacobians only; body velocities refer to link origins, in world."""
    model = mujoco.MjModel.from_xml_path(str(clip.model_path))
    data = mujoco.MjData(model)
    ids = np.arange(1, model.nbody)
    n = len(clip.time_s)
    positions = np.empty((n, len(ids), 3))
    quaternions = np.empty((n, len(ids), 4))
    linear, angular = np.empty_like(positions), np.empty_like(positions)
    gloves = np.empty((n, 2, 3))
    soles = np.empty((n, 2, 3))
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    hand_ids = [model.site(f"{side}Hand_center").id for side in ("Left", "Right")]
    sole_ids = [model.site(f"{side}Foot_sole").id for side in ("Left", "Right")]
    for i, (pose, velocity) in enumerate(zip(clip.qpos, clip.qvel, strict=True)):
        data.qpos[:], data.qvel[:] = pose, velocity
        mujoco.mj_forward(model, data)
        positions[i], quaternions[i] = data.xpos[ids], data.xquat[ids]
        gloves[i], soles[i] = data.site_xpos[hand_ids], data.site_xpos[sole_ids]
        for j, body_id in enumerate(ids):
            mujoco.mj_jacBody(model, data, jp, jr, int(body_id))
            linear[i, j], angular[i, j] = jp @ velocity, jr @ velocity
    return {"time_s": clip.time_s, "qpos": clip.qpos, "qvel": clip.qvel,
            "joint_pos": clip.qpos[:, model.jnt_qposadr[1:]],
            "joint_vel": clip.qvel[:, model.jnt_dofadr[1:]],
            "body_pos_w": positions, "body_quat_w": quaternions,
            "body_lin_vel_w": linear, "body_ang_vel_w": angular,
            "glove_pos_w": gloves, "sole_pos_w": soles,
            "joint_feedforward": np.zeros((n, model.nu)),
            "joint_names": np.asarray([model.joint(j).name for j in range(1, model.njnt)]),
            "body_names": np.asarray([model.body(int(j)).name for j in ids])}


def reference_contact_time(clip: ReferenceClip, gloves: np.ndarray) -> float | None:
    active = 0 if clip.family in ("jab", "left_hook") else 1
    distance = np.linalg.norm(gloves[:, active] - clip.target_w, axis=1)
    indices = np.flatnonzero((clip.time_s >= clip.launch_time_s) &
                            (clip.time_s < clip.guard_time_s) & (distance <= .14))
    return None if not len(indices) else float(clip.time_s[indices[0]])


def write_reference(clip: ReferenceClip, output_root: Path,
                    reference_id: str | None = None) -> dict[str, Any]:
    """Write an unqualified pair. Existing artifacts are never overwritten."""
    root = project_output(output_root)
    name = reference_id or clip.family
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Reference id must be a simple filename component")
    # A shared .020s endpoint makes the lower-rate view exactly master[::4].
    duration = np.ceil((clip.time_s[-1] - 1e-10) / POLICY_DT) * POLICY_DT
    base = resample_reference(clip)
    if base.time_s[-1] < duration - 1e-9:
        times = np.arange(round(duration / PHYSICS_DT) + 1) * PHYSICS_DT
        poses = interpolate_poses(base.time_s, base.qpos, times)
        model = mujoco.MjModel.from_xml_path(str(base.model_path))
        base = ReferenceClip(base.family, base.model_path, times, poses,
                             tangent_velocities(model, times, poses), base.target_w,
                             base.launch_time_s, base.guard_time_s, base.source_sha256,
                             base.metadata)
    master = kinematic_arrays(base)
    master_path, motion_path = root / "masters" / f"{name}.npz", root / "motions/npz" / f"{name}.npz"
    for path in (master_path, motion_path):
        project_output(path)
        if path.exists():
            raise FileExistsError(path)
    for path in (master_path, motion_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(master_path, **master)
    keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w")
    np.savez_compressed(motion_path, fps=np.asarray(50.0),
                        **{key: master[key][::4] for key in keys},
                        joint_names=master["joint_names"], body_names=master["body_names"])
    contact = reference_contact_time(base, master["glove_pos_w"])
    return {"reference_id": name, "family": clip.family,
            "motion_path": motion_path.relative_to(root).as_posix(),
            "master_path": master_path.relative_to(root).as_posix(),
            "motion_sha256": sha256(motion_path), "master_sha256": sha256(master_path),
            "source_sha256": clip.source_sha256, "launch_time_s": clip.launch_time_s,
            "guard_time_s": clip.guard_time_s, "duration_s": float(base.time_s[-1]),
            "requested_ttc_s": clip.metadata.get("requested_ttc_s"),
            "reference_contact_delay_s": None if contact is None else contact-clip.launch_time_s,
            "target_w": clip.target_w.tolist(), "physics_dt_s": PHYSICS_DT,
            "policy_dt_s": POLICY_DT, "joint_names": master["joint_names"].tolist(),
            "body_names": master["body_names"].tolist(), "quaternion_order": "wxyz",
            "feedforward_mode": "disabled_zero",
            "g1_snapshot_present": clip.metadata.get("g1_snapshot_present", False),
            "launch_features": clip.metadata.get("launch_features", []),
            "launch_feature_layout": clip.metadata.get("launch_feature_layout", []),
            "guard_id": clip.metadata.get("guard_id", f"task09_{clip.family}_guard"),
            "guard_source": clip.metadata.get("guard_source", "task09_family_guard"),
            "common_guard_validated": False,
            "snapshot_fingerprint": clip.metadata.get("snapshot_fingerprint"),
            "launch_snapshot": clip.metadata.get("launch_snapshot"),
            "qualified_for_training": False}


def convert_release(release_root: Path, output_root: Path) -> dict[str, Any]:
    """Prepare all four source clips for WBC without asserting new competence."""
    release_root, output_root = Path(release_root).resolve(), project_output(output_root)
    if output_root.exists():
        raise FileExistsError("Choose a new reference output directory")
    clips = [load_release_reference(release_root, family) for family in FAMILIES]
    # Detect inconsistent models before making any output.
    model_hashes = {sha256(clip.model_path) for clip in clips}
    if len(model_hashes) != 1:
        raise ValueError("A shared tracker requires the same physical body in every clip")
    assets = output_root / "assets"
    assets.mkdir(parents=True)
    for source, destination in ((clips[0].model_path, "human.xml"),
                                (release_root / "jab/body_metadata.json", "body_metadata.json"),
                                (release_root / "jab/controller_schema.json", "controller_schema.json")):
        shutil.copyfile(source, assets / destination)
    rows = [write_reference(clip, output_root) for clip in clips]
    manifest = {"schema": "human_wbc_references_v1", "qualified_for_training": False,
                "reason": "CPU conversion only; no common-guard/domain/physics qualification",
                "model_path": "assets/human.xml", "body_metadata_path": "assets/body_metadata.json",
                "controller_schema_path": "assets/controller_schema.json",
                "asset_sha256": {p.name: sha256(p) for p in sorted(assets.iterdir())},
                "release_manifest_sha256": sha256(release_root / "manifest.json"),
                "physics_dt_s": PHYSICS_DT, "policy_dt_s": POLICY_DT, "decimation": 4,
                "body_frame_convention": "world +Z up; canonical human +X forward,+Y left",
                "quaternion_order": "wxyz", "feedforward_mode": "disabled_zero",
                "launch_feature_layout": [], "clips": rows}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
