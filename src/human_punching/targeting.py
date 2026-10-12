"""Deterministic glove-path deformation and arm IK for a stationary head target.

This is kinematic retargeting, not a dynamics controller or anatomical feasibility
certificate. Only the selected arm's four source ball joints are modified. The
source pelvis, legs, trunk, other arm, clip timing, and endpoint poses are kept.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .motion import CROP_SPECS


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def _glove_path(model, qpos, geom_id):
    data = mujoco.MjData(model)
    positions, rotations = [], []
    for pose in qpos:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        positions.append(data.geom_xpos[geom_id].copy())
        rotations.append(data.geom_xmat[geom_id].reshape(3, 3).copy())
    return np.array(positions), np.array(rotations)


def _export_trajectory(model, source, qpos):
    """Recompute all exported state arrays from the actual solved MuJoCo poses."""
    result = {key: np.array(value, copy=True) for key, value in source.items()}
    time = result["time_s"]
    names = result["body_names"]
    body_ids = np.array([model.body(str(name)).id for name in names])
    count = len(qpos)
    data = mujoco.MjData(model)
    qvel = np.empty((count, model.nv))
    positions = np.empty((count, len(body_ids), 3))
    quaternions = np.empty((count, len(body_ids), 4))
    rotations = np.empty((count, len(body_ids), 3, 3))
    angular = np.empty((count, len(body_ids), 3))
    for frame in range(count):
        before, after = max(0, frame - 1), min(count - 1, frame + 1)
        mujoco.mj_differentiatePos(
            model, qvel[frame], float(time[after] - time[before]), qpos[before], qpos[after]
        )
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        positions[frame] = data.xpos[body_ids]
        quaternions[frame] = data.xquat[body_ids]
        rotations[frame] = data.xmat[body_ids].reshape(-1, 3, 3)
    for frame in range(1, count):
        flip = (quaternions[frame - 1] * quaternions[frame]).sum(axis=-1) < 0
        quaternions[frame, flip] *= -1
    for frame in range(count):
        before, after = max(0, frame - 1), min(count - 1, frame + 1)
        delta = rotations[after] @ rotations[before].swapaxes(-1, -2)
        angular[frame] = Rotation.from_matrix(delta).as_rotvec() / (time[after] - time[before])
    result.update(
        qpos=qpos,
        qvel=qvel,
        body_pos_w=positions,
        body_quat_w=quaternions,
        body_lin_vel_w=np.gradient(positions, time, axis=0, edge_order=2),
        body_ang_vel_w=angular,
    )
    return result


def aim_clip(
    model: mujoco.MjModel,
    trajectory: dict,
    punch: str,
    head_center: np.ndarray,
    *,
    head_radius_m: float = 0.06,
    glove_radius_m: float = 0.08,
    clearance_m: float = 0.015,
    anchor_frame: int | None = None,
    target_mode: str = "visual_clearance",
    contact_center_distance_m: float = 0.135,
    approach_yaw_offset_rad: float = 0.0,
) -> dict:
    """Aim one recorded full punch cycle at a fixed spherical head proxy.

    A quintic envelope translates the source glove path near the attack. Hooks
    retain their source lateral arc and target the corresponding head side;
    straight punches target its front. A smooth radial projection keeps the
    requested glove-center path outside the summed spherical radii. Damped
    Jacobian IK tracks it while softly retaining source glove orientation and
    source arm posture. This proxy is a visualization clearance, not a mesh
    collision guarantee; the output is not intended for physical stepping.
    The default 15 mm extra proxy clearance compensates for rounded glove-box
    corners in the default scene; actual mesh/head distance must still be checked.

    ``target_mode="proxy_contact"`` instead directs the glove center into the
    accepted physical proxy envelope (default 0.135 m versus 0.14 m contact).
    This changes the reference objective, not the head/glove radii. Visual mesh
    overlap is possible and impact response remains disabled. The optional
    approach yaw offset supplies finite geometric attack variants.

    ``anchor_frame`` is an optional *array index*, not a DuoBox source frame.
    By default the anchor is maximal forward glove reach within the annotated
    attack region, since hook contact can precede the outward-completion label.
    """
    if punch not in CROP_SPECS:
        raise ValueError(f"Unknown punch: {punch}")
    head_center = np.asarray(head_center, dtype=float)
    if head_center.shape != (3,) or not np.isfinite(head_center).all():
        raise ValueError("head_center must be a finite world-coordinate 3-vector")
    if min(head_radius_m, glove_radius_m) <= 0 or clearance_m < 0:
        raise ValueError("Proxy radii must be positive and clearance nonnegative")
    if target_mode not in ("visual_clearance", "proxy_contact"):
        raise ValueError("target_mode must be visual_clearance or proxy_contact")
    if not np.isfinite(approach_yaw_offset_rad):
        raise ValueError("approach_yaw_offset_rad must be finite")
    if target_mode == "proxy_contact" and not (
        np.isfinite(contact_center_distance_m)
        and 0 < contact_center_distance_m < head_radius_m + glove_radius_m
    ):
        raise ValueError("Contact center target must lie strictly inside the physical envelope")
    source = np.asarray(trajectory["qpos"], dtype=float)
    if source.ndim != 2 or source.shape[1] != model.nq or len(source) < 5:
        raise ValueError("Expected at least five human-model qpos frames")
    if not np.isfinite(source).all():
        raise ValueError("Source trajectory contains nonfinite qpos")
    spec = CROP_SPECS[punch]
    side = str(spec["anatomical_hand"]).title()
    geom_id = model.geom(f"{side}_boxing_glove").id
    body_id = int(model.geom_bodyid[geom_id])
    joint_ids = [
        model.joint(f"{side}{part}_joint").id for part in ("Shoulder", "Arm", "ForeArm", "Hand")
    ]
    if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_BALL for j in joint_ids):
        raise ValueError("Targeting expects the source mannequin's four arm ball joints")
    dofs = np.concatenate(
        [np.arange(model.jnt_dofadr[j], model.jnt_dofadr[j] + 3) for j in joint_ids]
    )
    qindices = np.concatenate(
        [np.arange(model.jnt_qposadr[j], model.jnt_qposadr[j] + 4) for j in joint_ids]
    )
    positions, rotations = _glove_path(model, source, geom_id)
    frames = np.asarray(trajectory["source_frames"])
    nominal = int(np.argmin(abs(frames - spec["outward_completion_frame"])))
    if anchor_frame is None:
        # Limit the search to this attack: padding may contain another punch.
        first = max(1, nominal - 25)
        last = min(len(source) - 2, nominal + 9)
        anchor_frame = first + int(np.argmax(positions[first : last + 1, 0]))
    anchor = int(anchor_frame)
    if not 1 <= anchor < len(source) - 1:
        raise ValueError("anchor_frame must be strictly inside the clip")
    indices = np.arange(len(source))
    # Hook wind-ups already reach downward near the arm's extension limit. Keep
    # those intact and lower only the raised attack arc, not the entire cycle.
    start = max(0, anchor - 12) if "hook" in punch else 0
    end = min(len(source) - 1, anchor + 28) if "hook" in punch else len(source) - 1
    envelope = np.where(
        indices <= anchor,
        _smoothstep((indices - start) / (anchor - start)),
        _smoothstep((end - indices) / (end - anchor)),
    )
    normal = np.array([-1.0, 0.0, 0.0])
    if "hook" in punch:
        normal = np.array([-0.65, 0.76 if side == "Left" else -0.76, 0.0])
        normal /= np.linalg.norm(normal)
    normal = Rotation.from_rotvec([0, 0, approach_yaw_offset_rad]).apply(normal)
    radius = float(
        contact_center_distance_m if target_mode == "proxy_contact"
        else head_radius_m + glove_radius_m + clearance_m
    )
    contact = head_center + normal * radius
    desired = positions + envelope[:, None] * (contact - positions[anchor])
    radial = desired - head_center
    distances = np.linalg.norm(radial, axis=1)
    direction = radial / np.maximum(distances[:, None], 1e-12)
    direction[distances < 1e-12] = normal
    # Smooth approximation of max(distance, radius), 1 mm rounding at contact.
    projected_distance = 0.5 * (distances + radius + np.sqrt((distances - radius) ** 2 + 0.002**2))
    desired = head_center + projected_distance[:, None] * direction
    # Exactly retain both guard endpoints, including tiny projection corrections.
    desired[[0, -1]] = positions[[0, -1]]

    qpos = source.copy()
    data = mujoco.MjData(model)
    jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
    difference, update = np.zeros(model.nv), np.zeros(model.nv)
    previous_correction = np.zeros(model.nv)
    regularizer = np.diag(np.repeat([0.022, 0.006, 0.006, 0.012], 3))
    orientation_weight = 0.075
    iterations = []
    for frame in range(1, len(source) - 1):
        pose = source[frame].copy()
        mujoco.mj_integratePos(model, pose, previous_correction, 1.0)
        for _step in range(60):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            mujoco.mj_jac(model, data, jp, jr, data.geom_xpos[geom_id], body_id)
            position_error = desired[frame] - data.geom_xpos[geom_id]
            orientation_error = Rotation.from_matrix(
                rotations[frame] @ data.geom_xmat[geom_id].reshape(3, 3).T
            ).as_rotvec()
            mujoco.mj_differentiatePos(model, difference, 1.0, pose, source[frame])
            matrix = np.vstack((jp[:, dofs], orientation_weight * jr[:, dofs], regularizer))
            residual = np.concatenate(
                (
                    position_error,
                    orientation_weight * orientation_error,
                    regularizer @ difference[dofs],
                )
            )
            dq = np.linalg.lstsq(matrix, residual, rcond=None)[0]
            dq *= min(1.0, 0.12 / max(float(np.max(np.abs(dq))), 1e-12))
            update[:] = 0.0
            update[dofs] = dq
            mujoco.mj_integratePos(model, pose, update, 1.0)
            if np.linalg.norm(dq) < 1e-6:
                break
        qpos[frame, qindices] = pose[qindices]
        mujoco.mj_differentiatePos(model, previous_correction, 1.0, source[frame], qpos[frame])
        previous_correction[np.setdiff1d(np.arange(model.nv), dofs)] = 0.0
        iterations.append(_step + 1)
    # Keep quaternion representatives continuous without altering endpoint values.
    for joint in joint_ids:
        adr = int(model.jnt_qposadr[joint])
        for frame in range(1, len(source) - 1):
            if qpos[frame - 1, adr : adr + 4] @ qpos[frame, adr : adr + 4] < 0:
                qpos[frame, adr : adr + 4] *= -1
    final_positions, final_rotations = _glove_path(model, qpos, geom_id)
    separation = np.linalg.norm(final_positions - head_center, axis=1)
    errors = np.linalg.norm(final_positions - desired, axis=1)
    closest = int(np.argmin(separation))
    correction_angles = []
    for before, after in zip(source, qpos, strict=True):
        mujoco.mj_differentiatePos(model, difference, 1.0, before, after)
        correction_angles.append(np.linalg.norm(difference[dofs].reshape(-1, 3), axis=1))
    orientation_errors = np.linalg.norm(
        Rotation.from_matrix(final_rotations @ rotations.swapaxes(-1, -2)).as_rotvec(), axis=1
    )
    metrics = {
        "method": "smooth source glove-path deformation + damped arm Jacobian IK",
        "execution": "kinematic playback only; no dynamics, ML, or joint-limit certificate",
        "active_hand": side.lower(),
        "modified_joints": [model.joint(j).name for j in joint_ids],
        "head_center_m": head_center.tolist(),
        "head_radius_m": float(head_radius_m),
        "glove_proxy_radius_m": float(glove_radius_m),
        "target_mode": target_mode,
        "requested_clearance_m": float(radius - head_radius_m - glove_radius_m),
        "requested_center_distance_m": radius,
        "approach_yaw_offset_rad": float(approach_yaw_offset_rad),
        "contact_target_m": contact.tolist(),
        "anchor_index": anchor,
        "anchor_source_frame": int(frames[anchor]),
        "deformation_start_index": start,
        "deformation_end_index": end,
        "closest_index": closest,
        "closest_source_frame": int(frames[closest]),
        "closest_time_s": float(trajectory["time_s"][closest]),
        "minimum_center_distance_m": float(separation.min()),
        "minimum_proxy_surface_gap_m": float(separation.min() - head_radius_m - glove_radius_m),
        "anchor_position_error_m": float(errors[anchor]),
        "path_position_error_max_m": float(errors.max()),
        "path_position_error_rms_m": float(np.sqrt(np.mean(errors**2))),
        "maximum_glove_center_correction_m": float(
            np.linalg.norm(final_positions - positions, axis=1).max()
        ),
        "maximum_arm_joint_correction_deg": float(np.rad2deg(correction_angles).max()),
        "maximum_glove_orientation_change_deg": float(np.rad2deg(orientation_errors).max()),
        "maximum_ik_iterations": max(iterations),
        "endpoints_unchanged": bool(np.array_equal(qpos[[0, -1]], source[[0, -1]])),
        "unselected_qpos_unchanged": bool(
            np.array_equal(
                qpos[:, np.setdiff1d(np.arange(model.nq), qindices)],
                source[:, np.setdiff1d(np.arange(model.nq), qindices)],
            )
        ),
        "collision_scope": "active glove and head spherical proxies only; not full mesh collisions",
    }
    result = _export_trajectory(model, trajectory, qpos)
    result["source_active_glove_pos_w"] = positions
    result["target_glove_pos_w"] = desired
    result["actual_glove_pos_w"] = final_positions
    return {"trajectory": result, "metrics": metrics}
