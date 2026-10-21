"""Sequential active-arm IK candidate helper, separate from the frozen recipe.

Previous envelope-normalized corrections guide the next knot. This encourages
consistent branches but does not certify C2 joint paths, dynamics, or contact.
No catalog dispatch, physics integration, or reference generation runs on import.
"""
from dataclasses import asdict, dataclass
import hashlib

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from .phase2_domain import fingerprint
from .phase2_target_reference import (
    RetargetRejection, RetargetSettings, correction_envelope,
    resolve_joint_mask, _kinematics, _require_standalone_source,
)
from .wbc.references import tangent_velocities


VERSION = "task10b_sequential_active_arm_target_path_v3"
MAXIMUM_CORRECTION_RAD = .205


@dataclass(frozen=True)
class SequentialSettings:
    base: RetargetSettings
    preference_weight: float = .002
    version: str = VERSION

    def __post_init__(self):
        side = self.base.active_side
        names = tuple(side + name for name in (
            "Shoulder_x", "Shoulder_y", "Shoulder_z", "Arm_x", "Arm_y", "Arm_z",
            "ForeArm_z", "ForeArm_x", "Hand_x", "Hand_y", "Hand_z"))
        expected = RetargetSettings(side, names, maximum_iterations=80, maximum_joint_delta_rad=MAXIMUM_CORRECTION_RAD)
        if (self.version != VERSION or self.preference_weight != .002
                or asdict(self.base) != asdict(expected)):
            raise ValueError("Sequential probe keeps the exact11-joint/.205-rad/80-iteration bounded recipe")


def transported_preference(previous, previous_envelope, envelope, low, high, *, maximum_correction_rad=MAXIMUM_CORRECTION_RAD):
    """Transport a bounded correction with the envelope, never the source pose."""
    previous, low, high = (np.asarray(x, float) for x in (previous, low, high))
    if (maximum_correction_rad != MAXIMUM_CORRECTION_RAD or previous.ndim != 1 or previous.shape != low.shape or low.shape != high.shape
            or not all(np.isfinite(x).all() for x in (previous, low, high))
            or np.any(low > high) or not 0 <= previous_envelope <= 1
            or not 0 <= envelope <= 1):
        raise ValueError("Finite correction/envelope bounds required")
    if envelope == 0 or previous_envelope == 0:
        return np.zeros_like(previous)
    normalized = np.clip(previous / previous_envelope, -maximum_correction_rad, maximum_correction_rad)
    return np.clip(envelope * normalized, low, high)


def retarget_sequential(model, source, target_w, *, source_outward_start_s,
                        source_outward_end_s, settings: SequentialSettings, checkpoint=None):
    """Construct a separate kinematic candidate using sequential branch preference.

    .025rad bounds each solver increment, not the difference between adjacent
    reference samples. Derivative and correction-step diagnostics remain explicit.
    Original feedforward/event finalization and offline/native screens are still
    required by the caller; this function applies no forces or actor actions.
    """
    base = settings.base
    t = np.asarray(source["time_s"], float); q = np.asarray(source["qpos"], float)
    target = np.asarray(target_w, float); original = np.asarray(source["head_center"], float)
    active_hand = 0 if base.active_side == "Left" else 1
    if (source.get("active_hand_index") != active_hand or t.ndim != 1 or len(t) < 3
            or t[0] != 0 or q.shape != (len(t), model.nq)
            or not np.allclose(np.diff(t), .005, atol=1e-12, rtol=0)
            or target.shape != (3,) or original.shape != (3,)
            or not all(np.isfinite(x).all() for x in (t, q, target, original))
            or not 0 <= source_outward_start_s < source_outward_end_s
                   <= float(source["guard_hold_start"]) <= t[-1]):
        raise ValueError("Complete finite5ms isolated source and explicit active hand required")
    _require_standalone_source(source, t, q, source_outward_start_s)
    delta_target = target - original
    if np.linalg.norm(delta_target) > base.maximum_target_delta_m:
        raise RetargetRejection("target_offset_outside_recipe", target_delta_m=delta_target.tolist())
    mask = resolve_joint_mask(model, base)
    ids, qa, va = (mask[key] for key in ("joint_ids", "qpos_addresses", "dof_addresses"))
    active, inactive = mask["active_site"], mask["inactive_site"]
    envelope = correction_envelope(t, source_outward_start_s, source_outward_end_s, base.envelope_blend_s)
    result = q.copy(); data = mujoco.MjData(model)
    jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
    previous = np.zeros(len(qa)); previous_e = 0.; records = []
    for frame, (pose, e) in enumerate(zip(q, envelope, strict=True)):
        if checkpoint is not None:
            checkpoint()
        if e == 0:
            transition_step = float(np.max(np.abs(previous)))
            previous = np.zeros(len(qa)); previous_e = 0.
            records.append(dict(iterations=0, active_error_m=0., inactive_error_m=0.,
                inactive_orientation_error_rad=0., correction_step_rad=transition_step, warm_start_used=False))
            continue
        _kinematics(model, data, pose)
        desired = data.site_xpos[active].copy() + e * delta_target
        inactive_position = data.site_xpos[inactive].copy()
        inactive_rotation = data.site_xmat[inactive].reshape(3, 3).copy()
        low = np.maximum(-e * base.maximum_joint_delta_rad, model.jnt_range[ids, 0] - pose[qa])
        high = np.minimum(e * base.maximum_joint_delta_rad, model.jnt_range[ids, 1] - pose[qa])
        if np.any(low > high):
            raise RetargetRejection("source_joint_outside_bounded_search", frame=frame)
        preference = transported_preference(previous, previous_e, float(e), low, high,
            maximum_correction_rad=base.maximum_joint_delta_rad)
        candidate = pose.copy(); candidate[qa] += preference
        residual_history = []; exit_reason = "iteration_cap"; last_step = None
        for iteration in range(base.maximum_iterations + 1):
            if checkpoint is not None:
                checkpoint()
            _kinematics(model, data, candidate)
            active_error = desired - data.site_xpos[active]
            inactive_error = inactive_position - data.site_xpos[inactive]
            rotation_error = Rotation.from_matrix(inactive_rotation @ data.site_xmat[inactive].reshape(3, 3).T).as_rotvec()
            errors = tuple(float(np.linalg.norm(x)) for x in (active_error, inactive_error, rotation_error))
            residual_history.append(errors[0])
            if (errors[0] <= base.active_position_tolerance_m
                    and errors[1] <= base.inactive_position_corridor_m
                    and errors[2] <= base.inactive_orientation_corridor_rad):
                exit_reason = "corridors_satisfied"; break
            if iteration == base.maximum_iterations:
                break
            mujoco.mj_jacSite(model, data, jp, jr, active)
            rows, rhs = [jp[:, va].copy()], [active_error]
            mujoco.mj_jacSite(model, data, jp, jr, inactive)
            correction = candidate[qa] - pose[qa]
            rows.extend((2 * jp[:, va].copy(), .1 * jr[:, va].copy(), settings.preference_weight * np.eye(len(qa))))
            rhs.extend((2 * inactive_error, .1 * rotation_error, settings.preference_weight * (preference - correction)))
            lo = np.maximum(low - correction, -e * base.maximum_iteration_delta_rad)
            hi = np.minimum(high - correction, e * base.maximum_iteration_delta_rad)
            free = hi - lo > 1e-12
            if not free.any():
                exit_reason = "no_free_coordinates"; break
            solution = lsq_linear(np.vstack(rows)[:, free], np.concatenate(rhs),
                bounds=(lo[free], hi[free]), method="bvls", tol=1e-10, max_iter=60)
            if not solution.success or not np.isfinite(solution.x).all():
                exit_reason = "bounded_linear_solver_failure"; break
            candidate[qa[free]] += solution.x
            last_step = float(np.max(np.abs(solution.x)))
        correction = candidate[qa] - pose[qa]
        if exit_reason != "corridors_satisfied":
            at_bound = (np.abs(correction - low) <= 1e-8) | (np.abs(correction - high) <= 1e-8)
            raise RetargetRejection("sequential_upper_body_geometry_unsolved", frame=frame,
                time_s=float(t[frame]), frames_completed=frame, frames_unattempted=len(t)-frame-1,
                iterations=iteration, exit_reason=exit_reason, active_error_m=errors[0],
                inactive_error_m=errors[1], inactive_orientation_error_rad=errors[2],
                active_residual_history_m=residual_history, final_correction_rad=correction.tolist(),
                lower_correction_bound_rad=low.tolist(), upper_correction_bound_rad=high.tolist(),
                saturated_joint_names=[name for name, flag in zip(base.optimized_joint_names, at_bound, strict=True) if flag],
                last_iteration_step_max_rad=last_step, algorithm_version=VERSION)
        result[frame] = candidate
        records.append(dict(iterations=iteration, active_error_m=errors[0], inactive_error_m=errors[1],
            inactive_orientation_error_rad=errors[2], correction_step_rad=float(np.max(np.abs(correction - previous))),
            warm_start_used=previous_e > 0, last_iteration_step_max_rad=last_step))
        previous, previous_e = correction.copy(), float(e)
    fixed = np.ones(model.nq, bool); fixed[qa] = False
    if not np.array_equal(result[:, fixed], q[:, fixed]) or not np.array_equal(result[envelope == 0], q[envelope == 0]):
        raise AssertionError("Protected source coordinates or holds changed")
    out = dict(time_s=t.copy(), t=t.copy(), qpos=result, head_center=target.copy(),
        launch_time=float(source_outward_start_s), return_start=float(source_outward_end_s),
        guard_hold_start=float(source["guard_hold_start"]), active_hand_index=active_hand)
    out["qvel"] = tangent_velocities(model, t, result)
    out["qacc"] = np.gradient(out["qvel"], t, axis=0, edge_order=2)
    correction = result[:, qa] - q[:, qa]
    correction_steps = np.r_[0., np.abs(np.diff(correction, axis=0)).max(axis=1)]
    acceleration = np.abs(out["qacc"][:, va])
    peak_sample, peak_joint = np.unravel_index(np.argmax(acceleration), acceleration.shape)
    corners = []
    for label, center in (("envelope_start", source_outward_start_s),
            ("full_correction_start", source_outward_start_s + base.envelope_blend_s),
            ("full_correction_end", source_outward_end_s - base.envelope_blend_s),
            ("envelope_end", source_outward_end_s)):
        indices = np.flatnonzero(np.abs(t-center) <= .015 + 1e-12)
        corners.append(dict(event=label, declared_time_s=float(center), sample_indices=indices.tolist(),
            maximum_correction_step_rad=float(correction_steps[indices].max()),
            maximum_optimized_speed_rad_s=float(np.abs(out["qvel"][indices][:, va]).max()),
            maximum_optimized_acceleration_rad_s2=float(acceleration[indices].max())))
    metadata = dict(schema="Task10BPhase2SequentialTargetPath/v3", algorithm_version=VERSION,
        recipe=asdict(base), sequential_settings=asdict(settings), recipe_identity=fingerprint(asdict(settings)),
        source_qpos_sha256=hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),
        optimized_joint_names=list(base.optimized_joint_names), optimized_qpos_addresses=qa.tolist(), optimized_dof_addresses=va.tolist(),
        source_outward_start_s=float(source_outward_start_s), source_outward_end_s=float(source_outward_end_s),
        source_target_w=original.tolist(), new_target_w=target.tolist(), target_delta_m=delta_target.tolist(),
        source_clock_and_all_holds_preserved=True, root_and_other_joints_exactly_preserved=True,
        maximum_joint_correction_rad=float(np.abs(correction).max()),
        maximum_5ms_correction_step_rad=float(np.abs(np.diff(correction, axis=0)).max()),
        maximum_5ms_optimized_joint_step_rad=float(np.abs(np.diff(result[:, qa], axis=0)).max()),
        maximum_optimized_acceleration_rad_s2=float(acceleration.max()),
        peak_optimized_acceleration_time_s=float(t[peak_sample]),
        peak_optimized_acceleration_joint=base.optimized_joint_names[peak_joint],
        envelope_transition_neighborhoods=corners, frame_geometry=records,
        continuity_semantics="Previous correction divided by previous envelope then scaled by current envelope, clipped to the separately declared .205rad engineering correction cap and unchanged physical joint ranges. Exact zero-envelope source holds. A sequential branch preference, not a C2 joint-path guarantee; .025rad is a solver increment, not a between-knot limit.",
        corner_diagnostic_semantics="Report full-grid correction and joint first differences and tangent acceleration, including envelope transitions. No derivative threshold or success claim is added.",
        event_semantics="Unchanged standalone event finalization must recompute launch/contact; no parent TTC or native qualification inherited.",
        feedforward_recomputed=False, kinematic_only=True, physical_qualification=False, physics_integrated=False)
    return out, metadata
