"""Independent FK audits of the final physical reference, after every edit."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .contact import (
    CONTACT_RADIUS_M,
    GEOMETRY_VERSION,
    GLOVE_RADIUS_M,
    HEAD_RADIUS_M,
    SAFE_CLEARANCE_RADIUS_M,
    evaluate_clearance,
    first_contact_time,
)


def _support_margin(points_xy, center_xy):
    """Signed minimum hull halfspace margin: positive inside, negative outside."""
    try:
        hull = ConvexHull(points_xy)
    except QhullError:
        return None
    normals, offsets = hull.equations[:, :2], hull.equations[:, 2]
    distances = (normals @ center_xy + offsets) / np.linalg.norm(normals, axis=1)
    return float(-distances.max())


def audit_reference(model, qpos, time_s, head_center) -> dict:
    """Measure final reference support, cone limits, speed and head targeting.

    Foot support is the projected convex hull of the actual physical sole box
    corners. This is a geometric double-support assumption, not a proof of
    ground reaction or dynamically feasible balance. All contact labels use the
    accepted two-glove center geometry, independent of visible mesh overlap.
    """
    poses = np.asarray(qpos, dtype=float)
    times = np.asarray(time_s, dtype=float)
    head = np.asarray(head_center, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != model.nq or not len(poses):
        raise ValueError("qpos must have nonempty shape (frames, model.nq)")
    if not np.isfinite(poses).all():
        raise ValueError("qpos must be finite")
    if times.shape != (len(poses),) or not np.isfinite(times).all():
        raise ValueError("time_s must be a finite vector matching qpos frames")
    if np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be strictly increasing")
    if head.shape != (3,) or not np.isfinite(head).all():
        raise ValueError("head_center must be a finite world 3-vector")
    foot_ids = [model.geom(f"physical/{side}Foot").id for side in ("Left", "Right")]
    sole_ids = [model.site(f"{side}Foot_sole").id for side in ("Left", "Right")]
    glove_ids = [model.geom(f"physical/{side}Hand").id for side in ("Left", "Right")]
    for foot in foot_ids:
        if model.geom_type[foot] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Support audit requires physical foot boxes")
    for glove in glove_ids:
        if model.geom_type[glove] != mujoco.mjtGeom.mjGEOM_SPHERE:
            raise ValueError("Contact audit requires spherical physical glove proxies")
        if not np.isclose(model.geom_size[glove, 0], GLOVE_RADIUS_M, atol=1e-12, rtol=0):
            raise ValueError("Physical glove radius differs from accepted contact geometry")
    ball_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_BALL)
    cone_rows = {}
    for joint in ball_ids:
        address = int(model.jnt_qposadr[joint])
        quaternion = poses[:, address:address + 4]
        norms = np.linalg.norm(quaternion, axis=1)
        if np.any(abs(norms - 1) > 1e-4):
            raise ValueError("Ball joint quaternions must be normalized")
        angles = 2 * np.arctan2(np.linalg.norm(quaternion[:, 1:], axis=1), abs(quaternion[:, 0]))
        limited = bool(model.jnt_limited[joint])
        limit = float(model.jnt_range[joint, 1]) if limited else None
        violations = np.maximum(angles - limit, 0) if limited else np.zeros(len(poses))
        maximum = int(np.argmax(violations))
        cone_rows[model.joint(int(joint)).name] = {
            "limited": limited,
            "limit_rad": limit,
            "maximum_angle_rad": float(angles.max()),
            "maximum_violation_rad": float(violations.max()),
            "violating_frames": int(np.count_nonzero(violations > 1e-9)),
            "maximum_violation_frame": maximum if violations.max() > 1e-9 else None,
        }
    speed_rows = {name: 0. for name in cone_rows}
    velocity = np.empty(model.nv)
    root_linear_speed = 0.
    root_angular_speed = 0.
    for frame in range(len(poses) - 1):
        mujoco.mj_differentiatePos(
            model, velocity, float(times[frame + 1] - times[frame]), poses[frame], poses[frame + 1]
        )
        for joint in ball_ids:
            address = int(model.jnt_dofadr[joint])
            name = model.joint(int(joint)).name
            speed_rows[name] = max(speed_rows[name], float(np.linalg.norm(velocity[address:address + 3])))
        # This model has one free human root; keep meters/s separate from rad/s.
        for joint in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            address = int(model.jnt_dofadr[joint])
            root_linear_speed = max(root_linear_speed, float(np.linalg.norm(velocity[address:address + 3])))
            root_angular_speed = max(root_angular_speed, float(np.linalg.norm(velocity[address + 3:address + 6])))
    data = mujoco.MjData(model)
    soles, corners, gloves, com, margins, tilts = [], [], [], [], [], []
    human_root = model.body("Hips").id
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        frame_corners, frame_tilts = [], []
        for geom in foot_ids:
            sx, sy, sz = model.geom_size[geom]
            local = np.array([[-sx, -sy, -sz], [-sx, -sy, sz], [sx, -sy, sz], [sx, -sy, -sz]])
            rotation = data.geom_xmat[geom].reshape(3, 3)
            frame_corners.append(local @ rotation.T + data.geom_xpos[geom])
            frame_tilts.append(float(np.arccos(np.clip(rotation[2, 1], -1, 1))))
        frame_corners = np.asarray(frame_corners)
        center = data.subtree_com[human_root].copy()
        soles.append(data.site_xpos[sole_ids].copy())
        corners.append(frame_corners)
        gloves.append(data.geom_xpos[glove_ids].copy())
        com.append(center)
        margins.append(_support_margin(frame_corners[:, :, :2].reshape(-1, 2), center[:2]))
        tilts.append(frame_tilts)
    soles, corners, gloves, com, tilts = map(np.asarray, (soles, corners, gloves, com, tilts))
    valid_margins = np.array([margin for margin in margins if margin is not None])
    clearance = evaluate_clearance(gloves, head)
    event = first_contact_time(times, gloves, head)
    first_sample = np.flatnonzero(clearance.any_contact)
    worst_joint = max(cone_rows, key=lambda name: cone_rows[name]["maximum_violation_rad"], default=None)
    worst_violation = cone_rows[worst_joint]["maximum_violation_rad"] if worst_joint else 0.
    fastest_joint = max(speed_rows, key=speed_rows.get, default=None)
    return {
        "audit_kind": "FK of final supplied reference; no physical stepping",
        "frames": len(poses),
        "duration_s": float(times[-1] - times[0]),
        "foot_order": ["Left", "Right"],
        "sole_horizontal_drift_max_m": np.linalg.norm(soles[:, :, :2] - soles[0, :, :2], axis=2).max(axis=0).tolist(),
        "sole_position_drift_max_m": np.linalg.norm(soles - soles[0], axis=2).max(axis=0).tolist(),
        "sole_center_height_min_m": soles[:, :, 2].min(axis=0).tolist(),
        "sole_center_height_max_m": soles[:, :, 2].max(axis=0).tolist(),
        "sole_corner_height_min_m": corners[:, :, :, 2].min(axis=(0, 2)).tolist(),
        "sole_corner_height_max_m": corners[:, :, :, 2].max(axis=(0, 2)).tolist(),
        "sole_tilt_max_rad": tilts.max(axis=0).tolist(),
        "support_assumption": "Projected hull of both actual foot-box soles; no contact-force feasibility claim",
        "support_hull_valid_frames": len(valid_margins),
        "com_support_halfspace_margin_min_m": float(valid_margins.min()) if len(valid_margins) else None,
        "com_inside_support_fraction": float(np.mean(valid_margins >= 0)) if len(valid_margins) else None,
        "com_height_min_m": float(com[:, 2].min()),
        "com_height_max_m": float(com[:, 2].max()),
        "cone_limits": cone_rows,
        "cone_violation_max_rad": worst_violation,
        "cone_worst_joint": worst_joint if worst_violation > 1e-9 else None,
        "cone_violating_joints": [name for name, row in cone_rows.items() if row["violating_frames"]],
        "joint_angular_speed_max_rad_s": speed_rows,
        "angular_speed_max_rad_s": speed_rows[fastest_joint] if fastest_joint else 0.,
        "angular_speed_max_joint": fastest_joint,
        "speed_method": "Adjacent-frame shortest quaternion arc divided by supplied sample time",
        "root_linear_speed_max_m_s": root_linear_speed,
        "root_angular_speed_max_rad_s": root_angular_speed,
        "head_center_w": head.tolist(),
        "geometry_version": GEOMETRY_VERSION,
        "head_radius_m": HEAD_RADIUS_M,
        "glove_radii_m": [float(model.geom_size[glove, 0]) for glove in glove_ids],
        "contact_center_distance_m": CONTACT_RADIUS_M,
        "safe_center_distance_m": SAFE_CLEARANCE_RADIUS_M,
        "glove_order": ["Left", "Right"],
        "glove_head_distance_min_m": clearance.distances_m.min(axis=0).tolist(),
        "any_sampled_proxy_contact": bool(np.any(clearance.any_contact)),
        "any_sampled_unsafe_clearance": bool(np.any(~clearance.fist_clearance_safe)),
        "first_sampled_contact_time_s": float(times[first_sample[0]]) if len(first_sample) else None,
        "first_linear_center_contact_time_s": float(event) if np.isfinite(event) else None,
        "contact_time_limitation": "Offline sampled / linearly interpolated centers; physical substep events are separate",
    }
