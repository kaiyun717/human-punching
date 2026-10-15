"""Independent checks and video replay of scalar-body jab physics rollouts.

The evaluator never steps or controls the human. It replays recorded states for
FK and uses recorded forces/support for dynamics claims. A passing trial is one
pilot demonstration, not a statistical success-rate or biomechanical validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .contact import CONTACT_RADIUS_M, SAFE_CLEARANCE_RADIUS_M, evaluate_clearance


@dataclass(frozen=True)
class PilotThresholds:
    """Declared engineering pilot criteria; these are not Task 09 paper claims."""

    minimum_root_height_fraction: float = 0.75
    maximum_root_tilt_rad: float = 0.60
    maximum_sole_drift_m: float = 0.03
    maximum_joint_limit_violation_rad: float = 0.02
    maximum_control_limit_violation: float = 1e-6
    maximum_external_force: float = 1e-9
    maximum_root_actuator_force: float = 1e-9
    guard_root_position_m: float = 0.03
    guard_root_orientation_rad: float = 0.10
    guard_joint_position_rad: float = 0.12
    guard_root_linear_speed_m_s: float = 0.05
    guard_root_angular_speed_rad_s: float = 0.12
    guard_joint_speed_rad_s: float = 0.50
    guard_glove_relative_position_m: float = 0.03
    guard_glove_relative_orientation_rad: float = float(np.deg2rad(15.0))
    guard_glove_relative_linear_speed_m_s: float = 0.15
    guard_dwell_s: float = 0.20
    maximum_glove_tracking_rms_m: float = 0.10


def _load_arrays(value):
    if isinstance(value, (str, Path)):
        with np.load(value, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    return {name: np.asarray(array) for name, array in value.items()}


def _times(arrays):
    for key in ("time_s", "time", "t"):
        if key in arrays:
            return np.asarray(arrays[key], dtype=float)
    raise ValueError("Trajectory requires time_s, time or t")


def trailing_dwell(time_s, mask):
    """Duration of the final contiguous true interval; a single sample has zero duration."""
    times, valid = np.asarray(time_s, dtype=float), np.asarray(mask, dtype=bool)
    if times.ndim != 1 or valid.shape != times.shape or not len(times):
        raise ValueError("time_s and mask must be matching nonempty vectors")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be finite and strictly increasing")
    if not valid[-1]:
        return 0.0
    false_indices = np.flatnonzero(~valid)
    start = int(false_indices[-1] + 1) if len(false_indices) else 0
    return float(times[-1] - times[start])


def _quat_angle(a, b):
    return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))


def _structure(model):
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Scalar jab evaluation requires exactly one free root")
    scalar = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    if len(free) + len(scalar) != model.njnt:
        raise ValueError("Expected only a free root and scalar hinge joints")
    root = int(free[0])
    return {
        "root_joint": root,
        "root_body": int(model.jnt_bodyid[root]),
        "chest_body": model.body("Chest").id,
        "root_qpos": int(model.jnt_qposadr[root]),
        "root_dof": int(model.jnt_dofadr[root]),
        "joints": scalar,
        "qadr": model.jnt_qposadr[scalar],
        "vadr": model.jnt_dofadr[scalar],
        "gloves": [model.site(f"{side}Hand_center").id for side in ("Left", "Right")],
        "soles": [model.site(f"{side}Foot_sole").id for side in ("Left", "Right")],
    }


def _validate_trajectory(model, arrays, structure):
    times = _times(arrays)
    poses = np.asarray(arrays["qpos"], dtype=float)
    velocities = np.asarray(arrays["qvel"], dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("At least two finite sample times are required")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Sample times must increase strictly")
    if poses.shape != (len(times), model.nq) or velocities.shape != (len(times), model.nv):
        raise ValueError("qpos/qvel dimensions differ from the supplied model")
    if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
        raise ValueError("Nonfinite recorded states cannot be evaluated as a successful trial")
    address = structure["root_qpos"]
    norms = np.linalg.norm(poses[:, address + 3:address + 7], axis=1)
    if np.any(abs(norms - 1) > 1e-5):
        raise ValueError("Recorded root quaternions must be normalized")
    return times, poses, velocities


def chest_relative_site_state(chest_position, chest_rotation, chest_linear_velocity,
                              chest_angular_velocity, site_positions, site_rotations,
                              site_linear_velocities):
    """Express sites and their velocities in a moving Chest frame.

    Chest arrays have leading batch dimensions plus (3,) or (3,3). Site arrays
    add a site axis before vector/matrix dimensions. The velocity correction
    includes angular motion of the frame, not merely chest translation.
    """
    offset = np.asarray(site_positions) - np.asarray(chest_position)[..., None, :]
    relative_world_velocity = (
        np.asarray(site_linear_velocities)
        - np.asarray(chest_linear_velocity)[..., None, :]
        - np.cross(np.asarray(chest_angular_velocity)[..., None, :], offset)
    )
    return {
        "position": np.einsum("...ji,...kj->...ki", chest_rotation, offset),
        "rotation": np.einsum("...ji,...kjl->...kil", chest_rotation, site_rotations),
        "linear_velocity": np.einsum("...ji,...kj->...ki", chest_rotation, relative_world_velocity),
    }


def _kinematics(model, poses, velocities, structure):
    data = mujoco.MjData(model)
    gloves, rotations, soles = [], [], []
    chest_positions, chest_rotations, chest_linear, chest_angular, glove_linear = [], [], [], [], []
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for pose, velocity in zip(poses, velocities, strict=True):
        data.qpos[:], data.qvel[:] = pose, velocity
        mujoco.mj_forward(model, data)
        gloves.append(data.site_xpos[structure["gloves"]].copy())
        rotations.append(data.site_xmat[structure["gloves"]].reshape(2, 3, 3).copy())
        soles.append(data.site_xpos[structure["soles"]].copy())
        chest = structure["chest_body"]
        chest_positions.append(data.xpos[chest].copy())
        chest_rotations.append(data.xmat[chest].reshape(3, 3).copy())
        mujoco.mj_jacBody(model, data, jacp, jacr, chest)
        chest_linear.append(jacp @ velocity)
        chest_angular.append(jacr @ velocity)
        frame_glove_velocity = []
        for hand in structure["gloves"]:
            mujoco.mj_jacSite(model, data, jacp, jacr, hand)
            frame_glove_velocity.append(jacp @ velocity)
        glove_linear.append(frame_glove_velocity)
    gloves, rotations, soles = np.asarray(gloves), np.asarray(rotations), np.asarray(soles)
    relative = chest_relative_site_state(
        np.asarray(chest_positions), np.asarray(chest_rotations), np.asarray(chest_linear),
        np.asarray(chest_angular), gloves, rotations, np.asarray(glove_linear),
    )
    return {"gloves": gloves, "rotations": rotations, "soles": soles,
            "chest_relative": relative}


def _reference_at(model, reference, times, structure, speed=1.0):
    source_t, source_q, source_v = _validate_trajectory(model, reference, structure)
    # Both clocks are measured from guard reset, independent of saved absolute origin.
    source_t = (source_t - source_t[0]) / speed
    source_v = source_v * speed
    sample_t = np.clip(times - times[0], source_t[0], source_t[-1])
    poses = np.column_stack([np.interp(sample_t, source_t, column) for column in source_q.T])
    velocities = np.column_stack([np.interp(sample_t, source_t, column) for column in source_v.T])
    start = structure["root_qpos"] + 3
    source_rotation = Rotation.from_quat(source_q[:, start:start + 4], scalar_first=True)
    poses[:, start:start + 4] = Slerp(source_t, source_rotation)(sample_t).as_quat(scalar_first=True)
    return poses, velocities


def _recorded_array(arrays, name, shape):
    if name not in arrays:
        return None
    result = np.asarray(arrays[name], dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _actuator_evidence(model, arrays, count, structure, tolerance):
    """Audit actuator output, without mistaking affine-PD drive for joint torque.

    Legacy gear-scaled motors can establish an output bound from their control
    range. Native affine PD instead requires compiled force limits and recorded
    actuator_force: its unlimited drive includes the position/velocity targets.
    """
    ctrl = _recorded_array(arrays, "ctrl", (count, model.nu))
    force = _recorded_array(arrays, "actuator_force", (count, model.nu))
    qact = _recorded_array(arrays, "qfrc_actuator", (count, model.nv))
    hinge = np.all(model.jnt_type[model.actuator_trnid[:, 0]] == mujoco.mjtJoint.mjJNT_HINGE)
    direct = bool(
        model.nu > 0 and hinge
        and np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        and np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
        and np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
    )
    motors = direct and bool(np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE)
                            and np.all(model.actuator_ctrllimited))
    affine = direct and bool(np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
                            and np.all(model.actuator_forcelimited))
    control_violation = None
    if ctrl is not None:
        ranges = model.actuator_ctrlrange
        violations = np.maximum(np.maximum(ranges[:, 0] - ctrl, ctrl - ranges[:, 1]), 0)
        violations[:, ~model.actuator_ctrllimited.astype(bool)] = 0
        control_violation = float(violations.max(initial=0))
    force_violation, transmission_error, torque_violation = None, None, None
    torque_lower, torque_upper = np.zeros(model.nv), np.zeros(model.nv)
    if motors or affine:
        for actuator in range(model.nu):
            if motors:
                limits = np.sort(model.actuator_ctrlrange[actuator] * model.actuator_gainprm[actuator, 0])
                if model.actuator_forcelimited[actuator]:
                    limits = np.clip(limits, *model.actuator_forcerange[actuator])
            else:
                limits = model.actuator_forcerange[actuator]
            limits = np.sort(limits * model.actuator_gear[actuator, 0])
            joint = int(model.actuator_trnid[actuator, 0])
            address = int(model.jnt_dofadr[joint])
            torque_lower[address] += limits[0]
            torque_upper[address] += limits[1]
        if qact is not None:
            vadr = structure["vadr"]
            violation = np.maximum(torque_lower[vadr] - qact[:, vadr], qact[:, vadr] - torque_upper[vadr])
            torque_violation = float(np.maximum(violation, 0).max(initial=0))
    if direct and force is not None:
        transmitted = np.zeros((count, model.nv))
        for actuator in range(model.nu):
            joint = int(model.actuator_trnid[actuator, 0])
            transmitted[:, int(model.jnt_dofadr[joint])] += force[:, actuator] * model.actuator_gear[actuator, 0]
        if qact is not None:
            transmission_error = float(abs(transmitted - qact).max(initial=0))
        ranges = model.actuator_forcerange
        violation = np.maximum(np.maximum(ranges[:, 0] - force, force - ranges[:, 1]), 0)
        violation[:, ~model.actuator_forcelimited.astype(bool)] = 0
        force_violation = float(violation.max(initial=0))
    control_ok = bool(control_violation is not None and control_violation <= tolerance and (
        motors or (affine and force_violation is not None and force_violation <= tolerance)
    ))
    return {
        "kind": "native_affine_pd_force_limited" if affine else "bounded_direct_torque_motor" if motors else "unsupported",
        "control_violation": control_violation,
        "control_ok": control_ok,
        "torque_violation": torque_violation,
        "actuator_force_violation": force_violation,
        "actuator_force_max": float(abs(force).max(initial=0)) if force is not None else None,
        "transmission_error": transmission_error,
        "requires_recorded_actuator_force": affine,
        "qact": qact,
    }


def _reference_metadata(reference, reference_path):
    metadata = {}
    if isinstance(reference_path, (str, Path)):
        sidecar = Path(reference_path).parent / "manifest.json"
        if sidecar.is_file():
            metadata.update(json.loads(sidecar.read_text()))
    for key in ("punch_family", "family", "active_hand", "source_speed", "source_speed_scale",
                "timing_kind", "timing_description", "source_speed_kind",
                "local_source_speed_min", "local_source_speed_max", "retiming_duration_ratio",
                "nominal_source_speed", "source_playback_duration_ratio",
                "attack_window_reference_s", "attack_window_semantics", "attack_window_reviewed",
                "launch_time", "return_start"):
        if reference is not None and key in reference:
            value = np.asarray(reference[key])
            metadata[key] = value.item() if value.ndim == 0 else value.tolist()
    return metadata


def _first_contact_time(times, distances, mask):
    """First sampled geometric contact; no interpolation or invented impact time."""
    indices = np.flatnonzero((np.asarray(distances) <= CONTACT_RADIUS_M) & mask)
    return float(times[indices[0]] - times[0]) if len(indices) else None


def _interpolated_contact_time(times, distances, mask):
    """Linear threshold crossing estimate between two FK distance samples."""
    distances = np.asarray(distances)
    indices = np.flatnonzero((distances <= CONTACT_RADIUS_M) & mask)
    if not len(indices):
        return None
    index = int(indices[0])
    if index == 0 or not mask[index - 1] or distances[index - 1] <= CONTACT_RADIUS_M:
        return float(times[index] - times[0])
    fraction = (distances[index - 1] - CONTACT_RADIUS_M) / (distances[index - 1] - distances[index])
    return float(times[index - 1] + fraction * (times[index] - times[index - 1]) - times[0])


def _simulation_clock_evidence(arrays, times, timestep):
    """Check the simulator clock, allowing only floating-point accumulation error."""
    if "simulation_time_s" not in arrays:
        return {"recorded": False, "verified": False}
    raw = np.asarray(arrays["simulation_time_s"])
    clock = np.asarray(raw, dtype=float)
    if clock.shape != times.shape or not np.isfinite(clock).all():
        raise ValueError("simulation_time_s must be a finite vector matching saved states")
    epsilon = float(np.finfo(raw.dtype).eps if np.issubdtype(raw.dtype, np.floating) else np.finfo(float).eps)
    # A float32 simulator accumulates a small clock drift after many exact dt
    # additions. Bound that rounding separately from the per-step cadence.
    interval_tolerance = max(1e-9, 2 * epsilon * max(float(abs(clock).max()), 1.0))
    accumulated_tolerance = max(1e-8, epsilon * float(np.sum(abs(clock))))
    interval_error = float(abs(np.diff(clock) - timestep).max(initial=0))
    elapsed_error = float(abs(clock - clock[0] - (times - times[0])).max(initial=0))
    return {"recorded": True,
            "verified": bool(np.all(np.diff(clock) > 0)
                             and interval_error <= interval_tolerance
                             and elapsed_error <= accumulated_tolerance),
            "recorded_dtype": str(raw.dtype),
            "maximum_step_error_s": interval_error,
            "maximum_elapsed_clock_error_s": elapsed_error,
            "step_roundoff_tolerance_s": interval_tolerance,
            "accumulated_roundoff_tolerance_s": accumulated_tolerance,
            "method": "Compare actual simulator time increments and elapsed time with saved nominal clock; allow dtype-specific floating-point accumulation only"}


def evaluate_rollout(model_path, rollout_path, reference_path=None, output_path=None,
                     *, head_center=None, thresholds=None, physics_timestep_s=None,
                     attack_window_reference_s=None, required_physics_timestep_s=None,
                     required_policy_timestep_s=None):
    """Evaluate a recorded full guard–jab–return trial without modifying it.

    Input may be NPZ paths or array dictionaries. Model may be an XML path or
    MjModel. Required arrays are time_s (time/t accepted), qpos and qvel. Strict
    acceptance additionally requires recorded ctrl, qfrc_actuator, qfrc_applied,
    xfrc_applied, support_contact (left/right actual floor support), and provenance
    state_reset_count=1 and physics_step_count=number of saved intervals. An NPZ
    sampled less frequently than physics cannot pass the cadence/provenance gate.
    """
    model = model_path if isinstance(model_path, mujoco.MjModel) else mujoco.MjModel.from_xml_path(str(model_path))
    structure = _structure(model)
    arrays = _load_arrays(rollout_path)
    times, poses, velocities = _validate_trajectory(model, arrays, structure)
    count = len(times)
    if physics_timestep_s is None:
        physics_timestep_s = arrays.get("physics_timestep_s", model.opt.timestep)
    actual_timestep = float(np.asarray(physics_timestep_s).item())
    if not np.isfinite(actual_timestep) or actual_timestep <= 0:
        raise ValueError("physics_timestep_s must be finite and positive")
    for name, interval in (("required_physics_timestep_s", required_physics_timestep_s),
                           ("required_policy_timestep_s", required_policy_timestep_s)):
        if interval is not None and (not np.isfinite(interval) or interval <= 0):
            raise ValueError(f"{name} must be finite and positive")
    reference = _load_arrays(reference_path) if reference_path is not None else None
    reference_metadata = _reference_metadata(reference, reference_path)
    timing_kind = str(reference_metadata.get("timing_kind", "uniform_playback"))
    uniform_source_speed = reference_metadata.get("source_speed", reference_metadata.get("source_speed_scale"))
    if "nonuniform" in timing_kind.lower():
        uniform_source_speed = None
    reference_speed = float(np.asarray(arrays.get("reference_speed", 1.0)).item())
    if not np.isfinite(reference_speed) or reference_speed <= 0:
        raise ValueError("reference_speed must be finite and positive")
    if attack_window_reference_s is None and reference is not None:
        if "attack_window_reference_s" in reference_metadata:
            attack_window_reference_s = reference_metadata["attack_window_reference_s"]
        elif "launch_time" in reference and "return_start" in reference:
            attack_window_reference_s = [reference["launch_time"], reference["return_start"]]
        elif isinstance(reference_path, (str, Path)):
            sidecar = Path(reference_path).parent / "manifest.json"
            if sidecar.is_file():
                reference_metadata = json.loads(sidecar.read_text())
                if "launch_time_s" in reference_metadata and "return_start_s" in reference_metadata:
                    attack_window_reference_s = [reference_metadata["launch_time_s"], reference_metadata["return_start_s"]]
    attack_window = None
    if attack_window_reference_s is not None:
        attack_window_reference_s = np.asarray(attack_window_reference_s, dtype=float)
        if attack_window_reference_s.shape != (2,) or not np.isfinite(attack_window_reference_s).all() or attack_window_reference_s[0] < 0 or attack_window_reference_s[1] <= attack_window_reference_s[0]:
            raise ValueError("attack_window_reference_s must be a finite increasing nonnegative pair")
        attack_window = attack_window_reference_s / reference_speed
    if head_center is None:
        for source in (arrays, reference or {}):
            for key in ("head_center_w", "target_head_center_w", "head_center"):
                if key in source:
                    head_center = source[key]
                    break
            if head_center is not None:
                break
    if head_center is None:
        raise ValueError("Supply head_center or save head_center_w in the trajectory/reference")
    head = np.asarray(head_center, dtype=float)
    if head.shape != (3,) or not np.isfinite(head).all():
        raise ValueError("This fixed-target pilot requires a finite head-center 3-vector")
    kinematics = _kinematics(model, poses, velocities, structure)
    clearance = evaluate_clearance(kinematics["gloves"], head)
    qadr, vadr = structure["qadr"], structure["vadr"]
    rq, rv = structure["root_qpos"], structure["root_dof"]
    joint_ids = structure["joints"]
    limits = model.jnt_range[joint_ids]
    violations = np.maximum(np.maximum(limits[:, 0] - poses[:, qadr], poses[:, qadr] - limits[:, 1]), 0)
    violations[:, ~model.jnt_limited[joint_ids].astype(bool)] = 0
    cfg = thresholds or PilotThresholds()
    # A rotating pelvis about vertical remains upright. Rotate its initial local
    # world-up vector rather than assuming that every rig uses local +Z up.
    root_rotation = Rotation.from_quat(poses[:, rq + 3:rq + 7], scalar_first=True)
    local_up = root_rotation[0].inv().apply([0., 0., 1.])
    up = root_rotation.apply(np.broadcast_to(local_up, (count, 3)))
    tilt = np.arccos(np.clip(up[:, 2], -1, 1))
    sole_drift = np.linalg.norm(kinematics["soles"] - kinematics["soles"][0], axis=2)
    support = _recorded_array(arrays, "support_contact", (count, 2))
    if support is not None and not np.isin(support, [0, 1]).all():
        raise ValueError("support_contact must contain boolean/0/1 values")
    support_both = np.all(support > 0, axis=1) if support is not None else np.ones(count, dtype=bool)
    guard_pose = poses[0] if reference is None else np.asarray(reference["qpos"])[0]
    root_error = np.linalg.norm(poses[:, rq:rq + 3] - guard_pose[rq:rq + 3], axis=1)
    rotation_error = _quat_angle(poses[:, rq + 3:rq + 7], guard_pose[rq + 3:rq + 7])
    joint_error = np.max(abs(poses[:, qadr] - guard_pose[qadr]), axis=1)
    root_linear_speed = np.linalg.norm(velocities[:, rv:rv + 3], axis=1)
    root_angular_speed = np.linalg.norm(velocities[:, rv + 3:rv + 6], axis=1)
    joint_speed = np.max(abs(velocities[:, vadr]), axis=1)
    guard_kinematics = _kinematics(model, guard_pose[None, :], np.zeros((1, model.nv)), structure)
    relative = kinematics["chest_relative"]
    guard_relative = guard_kinematics["chest_relative"]
    glove_guard_position = np.linalg.norm(relative["position"] - guard_relative["position"][0], axis=-1)
    glove_relative_rotation = np.einsum("...ji,...jk->...ik", guard_relative["rotation"][0], relative["rotation"])
    glove_guard_orientation = np.arccos(np.clip((np.trace(glove_relative_rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    glove_guard_speed = np.linalg.norm(relative["linear_velocity"], axis=-1)
    contact_indices = np.flatnonzero(clearance.any_contact)
    attack_mask = np.zeros(count, dtype=bool)
    if attack_window is not None:
        elapsed = times - times[0]
        attack_mask = (elapsed >= attack_window[0]) & (elapsed <= attack_window[1])
    attack_contact_indices = np.flatnonzero(clearance.any_contact & attack_mask)
    # Recovery must occur after contact, not merely during initial guard padding.
    after_contact = np.zeros(count, dtype=bool)
    if len(contact_indices):
        after_contact[int(contact_indices[0]) + 1:] = True
    guard_components = {
        "root_position": root_error <= cfg.guard_root_position_m,
        "root_orientation": rotation_error <= cfg.guard_root_orientation_rad,
        "joint_position": joint_error <= cfg.guard_joint_position_rad,
        "root_linear_speed": root_linear_speed <= cfg.guard_root_linear_speed_m_s,
        "root_angular_speed": root_angular_speed <= cfg.guard_root_angular_speed_rad_s,
        "joint_speed": joint_speed <= cfg.guard_joint_speed_rad_s,
        "chest_relative_glove_position": np.all(glove_guard_position <= cfg.guard_glove_relative_position_m, axis=1),
        "chest_relative_glove_orientation": np.all(glove_guard_orientation <= cfg.guard_glove_relative_orientation_rad, axis=1),
        "chest_relative_glove_speed": np.all(glove_guard_speed <= cfg.guard_glove_relative_linear_speed_m_s, axis=1),
        "safe_clearance": clearance.fist_clearance_safe,
        "recorded_double_support": support_both,
        "after_contact": after_contact,
    }
    recovered = np.logical_and.reduce(list(guard_components.values()))
    dwell = trailing_dwell(times, recovered)
    actuator = _actuator_evidence(model, arrays, count, structure, cfg.maximum_control_limit_violation)
    qact = actuator["qact"]
    qapp = _recorded_array(arrays, "qfrc_applied", (count, model.nv))
    xapp = _recorded_array(arrays, "xfrc_applied", (count, model.nbody, 6))
    control_violation = actuator["control_violation"]
    q_external = float(abs(qapp).max(initial=0)) if qapp is not None else None
    x_external = float(abs(xapp).max(initial=0)) if xapp is not None else None
    root_actuator = float(abs(qact[:, rv:rv + 6]).max(initial=0)) if qact is not None else None
    torque_violation = actuator["torque_violation"]
    tracking = None
    active_hand = str(reference_metadata.get("active_hand", "unspecified"))
    active_index = {"left": 0, "right": 1}.get(active_hand.lower())
    contact_timing = None
    if reference is not None:
        ref_q, ref_v = _reference_at(model, reference, times, structure, reference_speed)
        ref_kinematics = _kinematics(model, ref_q, ref_v, structure)
        glove_error = np.linalg.norm(kinematics["gloves"] - ref_kinematics["gloves"], axis=-1)
        relative_rot = np.einsum("...ji,...jk->...ik", ref_kinematics["rotations"], kinematics["rotations"])
        orientation_error = np.arccos(np.clip((np.trace(relative_rot, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
        tracking = {
            "glove_position_rms_m": np.sqrt(np.mean(glove_error**2, axis=0)).tolist(),
            "glove_position_max_m": glove_error.max(axis=0).tolist(),
            "glove_orientation_rms_rad": np.sqrt(np.mean(orientation_error**2, axis=0)).tolist(),
            "root_position_rms_m": float(np.sqrt(np.mean(np.sum((poses[:, rq:rq + 3] - ref_q[:, rq:rq + 3])**2, axis=1)))),
            "root_orientation_rms_rad": float(np.sqrt(np.mean(_quat_angle(poses[:, rq + 3:rq + 7], ref_q[:, rq + 3:rq + 7])**2))),
            "joint_position_rms_rad": float(np.sqrt(np.mean((poses[:, qadr] - ref_q[:, qadr])**2))),
            "reference_speed": reference_speed,
            "method": "Reference scalar positions linear-interpolated, root orientation SLERP; recorded playback speed applied and final reference held after its end",
        }
        tracking["active_hand"] = active_hand
        tracking["active_glove_position_rms_m"] = tracking["glove_position_rms_m"][active_index] if active_index is not None else None
        tracking["attack_glove_position_rms_m"] = np.sqrt(np.mean(glove_error[attack_mask] ** 2, axis=0)).tolist() if np.any(attack_mask) else None
        tracking["attack_active_glove_position_rms_m"] = tracking["attack_glove_position_rms_m"][active_index] if active_index is not None and np.any(attack_mask) else None
        ref_distances = np.linalg.norm(ref_kinematics["gloves"] - head, axis=-1)
        ref_distance = ref_distances[:, active_index] if active_index is not None else ref_distances.min(axis=1)
        actual_distance = clearance.distances_m[:, active_index] if active_index is not None else clearance.distances_m.min(axis=1)
        reference_contact = _first_contact_time(times, ref_distance, attack_mask)
        actual_contact = _first_contact_time(times, actual_distance, attack_mask)
        timing_error = actual_contact - reference_contact if actual_contact is not None and reference_contact is not None else None
        reference_interpolated = _interpolated_contact_time(times, ref_distance, attack_mask)
        actual_interpolated = _interpolated_contact_time(times, actual_distance, attack_mask)
        interpolated_error = actual_interpolated - reference_interpolated if actual_interpolated is not None and reference_interpolated is not None else None
        contact_timing = {
            "hand": active_hand if active_index is not None else "either (active hand unspecified)",
            "reference_first_attack_contact_after_reset_s": reference_contact,
            "actual_first_attack_contact_after_reset_s": actual_contact,
            "signed_first_contact_error_s": timing_error,
            "absolute_first_contact_error_s": abs(timing_error) if timing_error is not None else None,
            "reference_interpolated_first_contact_after_reset_s": reference_interpolated,
            "actual_interpolated_first_contact_after_reset_s": actual_interpolated,
            "signed_interpolated_contact_error_s": interpolated_error,
            "absolute_interpolated_contact_error_s": abs(interpolated_error) if interpolated_error is not None else None,
            "sampling_resolution_s": actual_timestep,
            "method": "Reference and recorded FK on the same physics clock, active hand when annotated; positive error means late. Interpolated fields estimate a linear distance threshold crossing and do not add temporal measurements.",
        }
    reset_count = int(np.asarray(arrays["state_reset_count"]).item()) if "state_reset_count" in arrays else None
    step_count = int(np.asarray(arrays["physics_step_count"]).item()) if "physics_step_count" in arrays else None
    cadence_ok = bool(np.allclose(np.diff(times), actual_timestep, atol=1e-8, rtol=1e-5))
    simulator_clock = _simulation_clock_evidence(arrays, times, actual_timestep)
    root_height = poses[:, rq + 2]
    checks = {
        "one_guard_reset_and_every_physics_step_recorded": bool(reset_count == 1 and step_count == count - 1 and cadence_ok),
        "starts_from_reference_guard": bool(root_error[0] <= cfg.guard_root_position_m and rotation_error[0] <= cfg.guard_root_orientation_rad and joint_error[0] <= cfg.guard_joint_position_rad),
        "no_fall": bool(np.min(root_height) >= cfg.minimum_root_height_fraction * guard_pose[rq + 2] and np.max(tilt) <= cfg.maximum_root_tilt_rad),
        "sole_drift": bool(sole_drift.max() <= cfg.maximum_sole_drift_m),
        "joint_limits": bool(violations.max(initial=0) <= cfg.maximum_joint_limit_violation_rad),
        "bounded_recorded_control": actuator["control_ok"],
        "bounded_recorded_joint_torque": bool(torque_violation is not None and torque_violation <= cfg.maximum_control_limit_violation),
        "no_external_forces": bool(q_external is not None and x_external is not None and max(q_external, x_external) <= cfg.maximum_external_force),
        "unactuated_root": bool(root_actuator is not None and root_actuator <= cfg.maximum_root_actuator_force),
        "attack_window_contact": bool(len(attack_contact_indices)),
        "guard_recovered_with_recorded_support": bool(support is not None and dwell + 1e-10 >= cfg.guard_dwell_s),
        "reference_glove_tracking": bool(tracking is not None and max(tracking["glove_position_rms_m"]) <= cfg.maximum_glove_tracking_rms_m),
    }
    if actuator["requires_recorded_actuator_force"]:
        checks["recorded_actuator_force_matches_joint_torque"] = bool(
            actuator["transmission_error"] is not None
            and actuator["transmission_error"] <= cfg.maximum_control_limit_violation
        )
    controller_mode = str(np.asarray(arrays["controller_mode"]).item()) if "controller_mode" in arrays else None
    if controller_mode is not None:
        expected_kind = {"implicit_pd": "native_affine_pd_force_limited",
                         "explicit_pd": "bounded_direct_torque_motor"}.get(controller_mode)
        checks["recorded_controller_matches_compiled_actuator"] = actuator["kind"] == expected_kind
    runtime_hash_matches = None
    if "runtime_model_sha256" in arrays:
        runtime_hash_matches = bool(
            not isinstance(model_path, mujoco.MjModel)
            and hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
            == str(np.asarray(arrays["runtime_model_sha256"]).item())
        )
        checks["recorded_runtime_model_hash_matches"] = runtime_hash_matches
    if active_index is not None:
        checks["active_hand_attack_contact"] = bool(np.any(
            (clearance.distances_m[:, active_index] <= CONTACT_RADIUS_M) & attack_mask
        ))
    policy_timestep = float(np.asarray(arrays["policy_timestep_s"]).item()) if "policy_timestep_s" in arrays else None
    if policy_timestep is not None and (not np.isfinite(policy_timestep) or policy_timestep <= 0):
        raise ValueError("policy_timestep_s must be finite and positive")
    policy_update_indices = np.asarray(arrays["policy_update_step_indices"]) if "policy_update_step_indices" in arrays else None
    if required_physics_timestep_s is not None:
        checks["required_physics_timestep"] = bool(
            np.isclose(actual_timestep, required_physics_timestep_s, atol=1e-10, rtol=0)
            and np.isclose(model.opt.timestep, required_physics_timestep_s, atol=1e-10, rtol=0)
            and cadence_ok
            and simulator_clock["verified"]
        )
    if required_policy_timestep_s is not None:
        decimation = required_policy_timestep_s / actual_timestep
        valid_interval = bool(np.isfinite(decimation) and decimation >= 1 and abs(decimation - round(decimation)) < 1e-8)
        expected_indices = np.arange(0, count - 1, max(1, round(decimation))) if valid_interval else None
        checks["required_policy_update_interval"] = bool(
            valid_interval and policy_timestep is not None
            and np.isclose(policy_timestep, required_policy_timestep_s, atol=1e-10, rtol=0)
            and policy_update_indices is not None
            and np.array_equal(policy_update_indices, expected_indices)
        )
    result = {
        "evaluation_kind": str(np.asarray(arrays.get("evaluation_kind", "unspecified")).item()),
        "recorded_checkpoint_sha256": str(np.asarray(arrays["checkpoint_sha256"]).item()) if "checkpoint_sha256" in arrays else None,
        "model_xml": str(model_path) if not isinstance(model_path, mujoco.MjModel) else None,
        "recorded_frames": count,
        "recorded_duration_s": float(times[-1] - times[0]),
        "model_xml_timestep_s": float(model.opt.timestep),
        "evaluated_physics_timestep_s": actual_timestep,
        "recorded_policy_timestep_s": policy_timestep,
        "required_physics_timestep_s": required_physics_timestep_s,
        "required_policy_timestep_s": required_policy_timestep_s,
        "policy_update_indices_recorded": policy_update_indices is not None,
        "recording_at_every_model_step": cadence_ok,
        "simulator_clock_evidence": simulator_clock,
        "state_reset_count": reset_count,
        "physics_step_count": step_count,
        "root_body": model.body(structure["root_body"]).name,
        "joint_names": [model.joint(int(j)).name for j in joint_ids],
        "head_center_w": head.tolist(),
        "glove_order": ["Left", "Right"],
        "contact_distance_m": CONTACT_RADIUS_M,
        "safe_distance_m": SAFE_CLEARANCE_RADIUS_M,
        "glove_head_min_distance_m": clearance.distances_m.min(axis=0).tolist(),
        "any_glove_contact": bool(len(contact_indices)),
        "attack_window_reference_s": attack_window_reference_s.tolist() if attack_window is not None else None,
        "attack_window_after_reset_s": attack_window.tolist() if attack_window is not None else None,
        "attack_window_glove_contact": bool(len(attack_contact_indices)),
        "attack_window_glove_min_distance_m": clearance.distances_m[attack_mask].min(axis=0).tolist() if np.any(attack_mask) else None,
        "first_attack_contact_after_reset_s": float(times[attack_contact_indices[0]] - times[0]) if len(attack_contact_indices) else None,
        "punch_family": str(reference_metadata.get("punch_family", reference_metadata.get("family", "unspecified"))),
        "active_hand": active_hand,
        "source_speed": reference_metadata.get("source_speed", reference_metadata.get("source_speed_scale")),
        "effective_source_speed": float(uniform_source_speed) * reference_speed if uniform_source_speed is not None else None,
        "reference_timing": {key: reference_metadata[key] for key in (
            "timing_kind", "timing_description", "source_speed_kind",
            "local_source_speed_min", "local_source_speed_max", "retiming_duration_ratio",
            "nominal_source_speed", "source_playback_duration_ratio")
            if key in reference_metadata},
        "attack_window_limitation": reference_metadata.get("attack_window_semantics", "Coarse playback-launch through explicit return-blend start; includes source recovery and is not reviewed outward/follow-through segmentation"),
        "attack_window_reviewed": bool(reference_metadata.get("attack_window_reviewed", False)),
        "contact_timing": contact_timing,
        "precision_targets": {
            "active_glove_rms_target_m": 0.02,
            "contact_timing_target_s": 0.02,
            "active_glove_rms_target_met": bool(tracking is not None and tracking["active_glove_position_rms_m"] is not None and tracking["active_glove_position_rms_m"] <= 0.02),
            "contact_timing_target_met": bool(contact_timing is not None and contact_timing["absolute_interpolated_contact_error_s"] is not None and contact_timing["absolute_interpolated_contact_error_s"] <= 0.02),
            "affects_pilot_pass": False,
        },
        "first_sampled_contact_after_reset_s": float(times[contact_indices[0]] - times[0]) if len(contact_indices) else None,
        "contact_clock": "Elapsed from guard reset, not annotated outward-launch TTC",
        "final_both_gloves_safe": bool(clearance.fist_clearance_safe[-1]),
        "root_height_min_m": float(root_height.min()),
        "root_tilt_max_rad": float(tilt.max()),
        "sole_position_drift_max_m": sole_drift.max(axis=0).tolist(),
        "recorded_double_support_fraction": float(support_both.mean()) if support is not None else None,
        "support_evidence": "Recorded actual positive foot-floor forces during each preceding physics interval; endpoint-aligned" if support is not None else "Not recorded; FK sole positions alone do not prove support",
        "joint_limit_violation_max_rad": float(violations.max(initial=0)),
        "joint_limit_violations_rad": {model.joint(int(j)).name: float(violations[:, i].max(initial=0)) for i, j in enumerate(joint_ids)},
        "control_limit_violation_max": control_violation,
        "actuator_control_kind": actuator["kind"],
        "recorded_controller_mode": controller_mode,
        "runtime_model_hash_matches_recording": runtime_hash_matches,
        "control_semantics": "Drive includes kp*qtarget + kd*vtarget + feedforward; ctrl is not joint torque. Recorded force output is bounded by compiled actuator forcerange." if actuator["requires_recorded_actuator_force"] else "Gear-scaled motor control with compiled control limits",
        "recorded_actuator_force_max_abs": actuator["actuator_force_max"],
        "recorded_actuator_force_limit_violation_max": actuator["actuator_force_violation"],
        "actuator_force_transmission_error_max": actuator["transmission_error"],
        "actual_joint_torque_max_nm": float(abs(qact[:, vadr]).max(initial=0)) if qact is not None else None,
        "actual_joint_torque_limit_violation_max_nm": torque_violation,
        "generalized_external_force_max_abs": q_external,
        "body_external_wrench_max_abs": x_external,
        "root_actuator_force_max_abs": root_actuator,
        "final_guard": {
            "root_position_error_m": float(root_error[-1]),
            "root_orientation_error_rad": float(rotation_error[-1]),
            "joint_position_error_max_rad": float(joint_error[-1]),
            "root_linear_speed_m_s": float(root_linear_speed[-1]),
            "root_angular_speed_rad_s": float(root_angular_speed[-1]),
            "joint_speed_max_rad_s": float(joint_speed[-1]),
            "trailing_recovered_dwell_s": dwell,
            "component_trailing_dwell_s": {name: trailing_dwell(times, mask) for name, mask in guard_components.items()},
            "final_component_checks": {name: bool(mask[-1]) for name, mask in guard_components.items()},
            "glove_position_in_chest_error_m": glove_guard_position[-1].tolist(),
            "glove_orientation_in_chest_error_rad": glove_guard_orientation[-1].tolist(),
            "glove_relative_linear_speed_m_s": glove_guard_speed[-1].tolist(),
            "relative_frame": "Chest body frame; desired relative guard equals reference first pose",
            "relative_velocity_method": "R_chest.T @ (v_glove - v_chest - omega_chest cross (p_glove - p_chest))",
            "dwell_interval_definition": "Each counted interval has recovered pose/velocity at both saved endpoints and positive support flags at its right endpoint, describing forces during that preceding interval",
            "dwell_boundary_resolution_s": actual_timestep,
        },
        "reference_tracking": tracking,
        "acceptance_thresholds": asdict(cfg),
        "acceptance_checks": checks,
        "pilot_pass": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "limitations": [
            "Single recorded pilot, not a statistical success-rate or human-biomechanics validation.",
            "Fixed target; center contact does not imply simulated head impact forces.",
            "Recorded force/support/provenance arrays are required for dynamics acceptance; replay itself performs FK only.",
            "Initial-reference guard is the return target; safe glove clearance is not the full robust-SAV safety predicate.",
        ],
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def render_rollout(model_path, rollout_path, output_directory, *, head_center=None,
                   label=None, fps=30, playback_speeds=(1.0, 0.5), artifact_prefix="jab"):
    """Render the recorded physics states; clearly label learned vs reference PD."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    from .render import font
    from .target_render import _playback_indices

    if fps <= 0 or any(speed <= 0 for speed in playback_speeds):
        raise ValueError("fps and playback speeds must be positive")
    if artifact_prefix not in ("jab", "cross", "left_hook", "right_hook", "punch"):
        raise ValueError("artifact_prefix must be a punch-family name or 'punch'")
    model = model_path if isinstance(model_path, mujoco.MjModel) else mujoco.MjModel.from_xml_path(str(model_path))
    structure = _structure(model)
    arrays = _load_arrays(rollout_path)
    times, poses, velocities = _validate_trajectory(model, arrays, structure)
    if head_center is None:
        head_center = arrays.get("head_center_w", arrays.get("target_head_center_w", arrays.get("head_center")))
    if head_center is None:
        raise ValueError("Head center required for rendering target and distance overlay")
    head = np.asarray(head_center, dtype=float)
    if head.shape != (3,) or not np.isfinite(head).all():
        raise ValueError("head_center must be a finite 3-vector")
    label = label or str(np.asarray(arrays.get("evaluation_kind", "RECORDED ROLLOUT")).item())
    physical_recording = "physics_step_count" in arrays and "state_reset_count" in arrays
    state_label = "Recorded physics states" if physical_recording else "Reference poses (kinematic)"
    recorded_physics_dt = float(np.asarray(arrays["physics_timestep_s"]).item()) if "physics_timestep_s" in arrays else None
    recorded_policy_dt = float(np.asarray(arrays["policy_timestep_s"]).item()) if "policy_timestep_s" in arrays else None
    clock_label = "Kinematic reference; no simulated control clock"
    if physical_recording:
        clock_parts = []
        if recorded_physics_dt is not None:
            clock_parts.append(f"{1000 * recorded_physics_dt:g} ms physics")
        if recorded_policy_dt is not None:
            clock_parts.append(f"{1000 * recorded_policy_dt:g} ms actions ({1 / recorded_policy_dt:g} Hz)")
        clock_label = " | ".join(clock_parts) or "Control clock not recorded"
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    kinematics = _kinematics(model, poses, velocities, structure)
    clearance = evaluate_clearance(kinematics["gloves"], head)
    closest = int(np.argmin(clearance.distances_m.min(axis=1)))
    root = poses[0, structure["root_qpos"]:structure["root_qpos"] + 3]
    center = (root + head) * 0.5
    actor_xy = np.vstack([poses[:, structure["root_qpos"]:structure["root_qpos"] + 2], head[:2]])
    lower_xy, upper_xy = actor_xy.min(axis=0), actor_xy.max(axis=0)
    center[:2] = (lower_xy + upper_xy) * 0.5
    center[2] = 0.92
    camera_distance = max(3.1, float(np.linalg.norm(upper_xy - lower_xy)) * 1.65 + 1.6)
    cameras = []
    for azimuth in (225, 90):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = center
        camera.distance = camera_distance
        camera.azimuth = azimuth
        camera.elevation = -10
        cameras.append(camera)
    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    options.sitegroup[:] = 0
    # This training asset intentionally uses the physical primitive geoms as
    # its plain blue visible body; MuJoCo hides group 3 by default.
    options.geomgroup[3] = 1
    options.geomgroup[5] = 0
    for geom_id in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[geom_id])).name or ""
        if body_name.startswith("g1/") and model.geom_group[geom_id] == 3:
            model.geom_group[geom_id] = 5
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
    model.vis.headlight.diffuse[:] = [0.65, 0.65, 0.65]
    # Presentation only: distinguish hazard gloves from the light-blue body.
    # This changes neither the saved XML/hash nor any physical parameter.
    for side in ("Left", "Right"):
        model.geom_rgba[model.geom(f"physical/{side}Hand").id] = [0.80, 0.07, 0.045, 1.0]
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 640)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 600)
    renderer = mujoco.Renderer(model, height=600, width=640)
    data = mujoco.MjData(model)
    title_font, small_font = font(23), font(16)
    video = output / f"{artifact_prefix}-rollout.mp4"
    still = output / f"{artifact_prefix}-closest.png"
    end_still = output / f"{artifact_prefix}-final-state.png"
    cache = {}

    def frame(index, speed_label):
        if index not in cache:
            data.qpos[:], data.qvel[:] = poses[index], velocities[index]
            data.time = times[index]
            mujoco.mj_forward(model, data)
            panels = []
            for camera in cameras:
                renderer.update_scene(data, camera=camera, scene_option=options)
                # A fixture proxy is drawn even when the optional G1 visual is absent.
                if renderer.scene.ngeom < renderer.scene.maxgeom:
                    mujoco.mjv_initGeom(renderer.scene.geoms[renderer.scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                       np.array([0.06, 0, 0]), head, np.eye(3).ravel(), np.array([1., .8, .25, .7]))
                    renderer.scene.ngeom += 1
                panels.append(Image.fromarray(renderer.render().copy()))
            cache[index] = panels
        canvas = Image.new("RGB", (1280, 720), (15, 22, 32))
        for number, panel in enumerate(cache[index]):
            canvas.paste(panel, (640 * number, 64))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), label.upper().replace("_", " "), font=title_font, fill=(235, 244, 250))
        draw.text((20, 42), clock_label, font=small_font, fill=(186, 204, 220))
        draw.text((1080, 22), speed_label, font=small_font, fill=(249, 207, 94))
        for number, view_label in enumerate(("OBLIQUE VIEW", "SIDE VIEW")):
            draw.text((640 * number + 15, 74), view_label, font=small_font, fill=(230, 240, 247))
        distance = clearance.distances_m[index]
        contact_label = "CONTACT" if clearance.any_contact[index] else ("SAFE CLEARANCE" if clearance.fist_clearance_safe[index] else "BUFFER BAND")
        draw.text((20, 675), f"{state_label} | fixed target | {contact_label}", font=small_font, fill=(186, 204, 220))
        draw.text((675, 675), f"t={times[index] - times[0]:.3f}s  L={distance[0]:.3f}m  R={distance[1]:.3f}m  contact<=0.14m", font=small_font, fill=(235, 221, 161))
        return canvas

    video_frames = 0
    try:
        with imageio.get_writer(video, fps=fps, codec="libx264", quality=8,
                                ffmpeg_params=["-movflags", "+faststart"], macro_block_size=16) as writer:
            for speed in playback_speeds:
                for index in _playback_indices(times, speed, fps):
                    writer.append_data(np.asarray(frame(int(index), f"{speed:g}x REPLAY")))
                    video_frames += 1
            frame(closest, "CLOSEST").save(still)
            frame(len(times) - 1, "FINAL STATE").save(end_still)
    finally:
        renderer.close()
    result = {"video": str(video), "closest_still": str(still), "final_still": str(end_still),
              "video_frames": video_frames, "fps": fps, "label": label,
              "clock_label": clock_label,
              "recorded_physics_timestep_s": recorded_physics_dt,
              "recorded_policy_timestep_s": recorded_policy_dt,
              "rendering_only": True, "physics_state_interpolation": False,
              "model_xml": str(model_path) if not isinstance(model_path, mujoco.MjModel) else None,
              "playback_speeds": list(playback_speeds),
              "visual_overrides": ["Enable human body geom group 3", "Hide G1 collision overlays", "Red glove RGBA", "Brighter headlight"],
              "source_xml_modified": False}
    (output / "render.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-center", type=float, nargs=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--physics-timestep", type=float)
    parser.add_argument("--require-physics-timestep", type=float)
    parser.add_argument("--require-policy-timestep", type=float)
    parser.add_argument("--attack-window", type=float, nargs=2, help="Reference-clock launch and explicit return-start; playback speed applied")
    parser.add_argument("--label")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_rollout(args.model, args.rollout, args.reference,
                               args.output / "evaluation.json", head_center=args.head_center,
                               physics_timestep_s=args.physics_timestep,
                               attack_window_reference_s=args.attack_window,
                               required_physics_timestep_s=args.require_physics_timestep,
                               required_policy_timestep_s=args.require_policy_timestep)
    if args.render:
        render_rollout(args.model, args.rollout, args.output,
                       head_center=metrics["head_center_w"], label=args.label)
    print(json.dumps({"pilot_pass": metrics["pilot_pass"], "failed_checks": metrics["failed_checks"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
