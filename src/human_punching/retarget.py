"""Rotation-preserving, uniformly scaled retargeting to the preview MJCF."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .motion import JOINT_NAMES, MotionClip


def wxyz(matrix: np.ndarray) -> np.ndarray:
    """MuJoCo uses scalar-first quaternions; scipy defaults to scalar-last."""
    return np.roll(Rotation.from_matrix(matrix).as_quat(), 1, axis=-1)


def surface_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """World bounds of actual mesh vertices (excluding the static scene)."""
    bounds = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        m = model.geom_dataid[g]
        start, count = model.mesh_vertadr[m], model.mesh_vertnum[m]
        vertices = model.mesh_vert[start : start + count]
        world = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        bounds.extend([world.min(axis=0), world.max(axis=0)])
    if not bounds:
        raise ValueError("Human model has no articulated triangle mesh geoms")
    return np.min(bounds, axis=0), np.max(bounds, axis=0)


def retarget_clip(clip: MotionClip, model: mujoco.MjModel, scale: float) -> dict:
    """Preserve local joint rotations and uniformly scale all root/bone translations.

    One initial heading transform and one constant floor translation apply to the
    entire clip. There is no target aiming, dynamic control, or per-frame foot IK.
    """
    count = len(clip.time_s)
    if count < 3:
        raise ValueError("A clip needs at least three frames for derivatives")
    forward = clip.global_rotations[0, 0, :, 2].copy()
    heading = float(np.arctan2(forward[1], forward[0]))
    align = Rotation.from_euler("z", -heading).as_matrix()
    qpos = np.tile(model.qpos0, (count, 1))
    origin = clip.positions[0, 0].copy()
    origin[2] = 0
    qpos[:, :3] = ((clip.positions[:, 0] - origin) * scale) @ align.T
    qpos[:, 3:7] = wxyz(align @ clip.global_rotations[:, 0])
    for index, name in enumerate(JOINT_NAMES[1:], 1):
        joint = model.joint(f"{name}_joint")
        address = int(joint.qposadr[0])
        qpos[:, address : address + 4] = wxyz(clip.local_rotations[:, index])
    # Select a continuous quaternion representative before differentiating/exporting.
    quaternion_addresses = [3] + [
        int(model.joint(f"{n}_joint").qposadr[0]) for n in JOINT_NAMES[1:]
    ]
    for address in quaternion_addresses:
        for frame in range(1, count):
            if (
                np.dot(qpos[frame - 1, address : address + 4], qpos[frame, address : address + 4])
                < 0
            ):
                qpos[frame, address : address + 4] *= -1
    data = mujoco.MjData(model)
    floor_min = np.inf
    for pose in qpos:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        floor_min = min(floor_min, float(surface_bounds(model, data)[0][2]))
    floor_shift = 0.002 - floor_min
    qpos[:, 2] += floor_shift
    body_ids = np.array([model.body(name).id for name in JOINT_NAMES])
    body_pos = np.empty((count, len(body_ids), 3))
    body_quat = np.empty((count, len(body_ids), 4))
    body_rot = np.empty((count, len(body_ids), 3, 3))
    qvel = np.empty((count, model.nv))
    mins, maxs = [], []
    for frame in range(count):
        before, after = max(0, frame - 1), min(count - 1, frame + 1)
        mujoco.mj_differentiatePos(
            model,
            qvel[frame],
            float(clip.time_s[after] - clip.time_s[before]),
            qpos[before],
            qpos[after],
        )
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids]
        body_rot[frame] = data.xmat[body_ids].reshape(-1, 3, 3)
        low, high = surface_bounds(model, data)
        mins.append(low)
        maxs.append(high)
    for frame in range(1, count):
        flip = (body_quat[frame - 1] * body_quat[frame]).sum(axis=-1) < 0
        body_quat[frame, flip] *= -1
    angular = np.empty((count, len(body_ids), 3))
    for frame in range(count):
        before, after = max(0, frame - 1), min(count - 1, frame + 1)
        delta = body_rot[after] @ np.swapaxes(body_rot[before], -1, -2)
        angular[frame] = Rotation.from_matrix(delta).as_rotvec() / (
            clip.time_s[after] - clip.time_s[before]
        )
    expected = ((clip.positions - origin) * scale) @ align.T
    expected[:, :, 2] += floor_shift
    error = np.linalg.norm(body_pos - expected, axis=-1)
    trajectory = {
        "qpos": qpos,
        "qvel": qvel,
        "time_s": clip.time_s,
        "fps": np.array(clip.fps),
        "source_frames": clip.source_frames,
        "body_names": np.array(JOINT_NAMES),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": np.gradient(body_pos, clip.time_s, axis=0, edge_order=2),
        "body_ang_vel_w": angular,
    }
    if not all(np.isfinite(v).all() for v in trajectory.values() if v.dtype.kind != "U"):
        raise ValueError("Nonfinite trajectory")
    if error.max() > 1e-4:
        raise ValueError(f"Retargeting FK error {error.max():.6g} m exceeds 0.1 mm")
    metrics = {
        "frames": count,
        "source_fps": clip.fps,
        "duration_s": float(clip.time_s[-1] - clip.time_s[0]),
        "scale": float(scale),
        "heading_rotation_rad": -heading,
        "source_origin_xy_m": origin[:2].tolist(),
        "floor_shift_m": floor_shift,
        "floor_policy": "one constant translation; lowest mesh point over clip at 2 mm",
        "mesh_bounds_min_m": np.min(mins, axis=0).tolist(),
        "mesh_bounds_max_m": np.max(maxs, axis=0).tolist(),
        "joint_fk_max_error_m": float(error.max()),
        "joint_fk_rms_error_m": float(np.sqrt(np.mean(error**2))),
        "active_wrist_peak_speed_m_s": float(
            np.max(np.linalg.norm(trajectory["body_lin_vel_w"][:, clip.active_hand_index], axis=-1))
        ),
        "qpos_dimension": model.nq,
        "qvel_dimension": model.nv,
        "quaternion_order": "wxyz",
        "world_axes": "z up; initial root forward aligned to +x",
        "execution": "kinematic qpos playback with mj_forward; no physical tracker",
        "source": clip.metadata,
    }
    return {"trajectory": trajectory, "metrics": metrics}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
