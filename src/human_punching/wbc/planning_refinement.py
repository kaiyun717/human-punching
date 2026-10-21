"""Versioned offline common-guard refinement; never a dynamics certificate.

Endpoint task-space corrections and temporal IK replace the original independent
frame solves. Shape-preserving interpolation is followed by a new FK-based contact
measurement and TTC map. No old timing claim survives a trajectory modification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq, lsq_linear
from scipy.spatial.transform import Rotation

from human_punching.scalar_reference import PoseSolver, rotation_error
from .contracts import PunchRequest, quaternion_matrix
from .planning import (LAUNCH_FEATURE_LAYOUT, ReferenceCandidate, _first_crossing,
                       _smoothstep, launch_features, place_reference)
from .references import ReferenceClip, interpolate_poses, kinematic_arrays, tangent_velocities


@dataclass(frozen=True)
class RefinementSettings:
    """Engineering search parameters, not biological or dynamic limits."""
    max_source_joint_speed_rad_s: float = 10.
    temporal_prior_fraction: float = .30
    prior_prediction_fraction: float = .65
    maximum_interpolation_sole_drift_m: float = .015
    smoothing_sigma_s: float = .035
    endpoint_blend_s: float = .12
    recovery_corridor: bool = False
    project_support_after_smoothing: bool = False
    support_projection_iterations: int = 12
    unbiased_support_projection: bool = False
    quiet_boundary_s: float = 0.

    def __post_init__(self):
        if (not np.isfinite(self.max_source_joint_speed_rad_s) or self.max_source_joint_speed_rad_s <= 0
                or not 0 <= self.temporal_prior_fraction <= 1
                or not 0 <= self.prior_prediction_fraction <= 1
                or not np.isfinite(self.maximum_interpolation_sole_drift_m)
                or self.maximum_interpolation_sole_drift_m <= 0
                or not np.isfinite(self.smoothing_sigma_s) or self.smoothing_sigma_s <= 0
                or not np.isfinite(self.endpoint_blend_s) or self.endpoint_blend_s <= 0
                or not np.isfinite(self.quiet_boundary_s) or self.quiet_boundary_s < 0):
            raise ValueError("Invalid offline refinement settings")
        if (type(self.recovery_corridor) is not bool or type(self.project_support_after_smoothing) is not bool
                or type(self.unbiased_support_projection) is not bool
                or type(self.support_projection_iterations) is not int or self.support_projection_iterations < 1):
            raise ValueError("Support/recovery refinement requires explicit flags and iterations")


class _PoseCurve:
    """C1, range-preserving scalar/rotation-chart interpolation with quiet ends."""
    def __init__(self, times, poses):
        self.times = np.asarray(times)
        self.base_rotation = Rotation.from_quat(poses[0, [4, 5, 6, 3]])
        relative = self.base_rotation.inv() * Rotation.from_quat(poses[:, [4, 5, 6, 3]])
        vectors = relative.as_rotvec()
        if np.linalg.norm(vectors, axis=-1).max() >= np.pi-.05:
            raise ValueError("Refined root rotation exceeds the local interpolation chart")
        values = np.column_stack((poses[:, :3], vectors, poses[:, 7:]))
        # Constant samples beyond the clip give PCHIP zero endpoint derivatives;
        # each joint remains inside the range of its neighboring solved samples.
        extended_t = np.r_[times[0]-.02, times, times[-1]+.02]
        extended_values = np.vstack((values[0], values, values[-1]))
        self.curve = PchipInterpolator(extended_t, extended_values, axis=0)

    def __call__(self, times):
        values = self.curve(np.clip(np.atleast_1d(times), self.times[0], self.times[-1]))
        rotations = self.base_rotation * Rotation.from_rotvec(values[:, 3:6])
        quaternions = rotations.as_quat()[:, [3, 0, 1, 2]]
        for i in range(1, len(quaternions)):
            if quaternions[i-1] @ quaternions[i] < 0:
                quaternions[i] *= -1
        return np.column_stack((values[:, :3], quaternions, values[:, 6:]))


def _fk_contact(model, curve, target, active_site, outward_end):
    """Bracket the first crossing on the changed path, then solve its true FK."""
    data = mujoco.MjData(model)
    def distance(time):
        data.qpos[:] = curve(float(time))[0]
        mujoco.mj_forward(model, data)
        return float(np.linalg.norm(data.site_xpos[active_site]-target)-.14)
    times = np.unique(np.r_[np.arange(0., outward_end, .005), outward_end])
    if distance(times[0]) <= 0:
        raise ValueError("Refined trajectory begins in analytic contact")
    for left, right in zip(times[:-1], times[1:], strict=True):
        current = distance(right)
        if current <= 0:
            return float(brentq(distance, left, right, xtol=1e-12))
    raise ValueError("Refined candidate has no outward analytic contact")


def _smooth_guard_curve(times, poses, guard_time, settings):
    """Smooth on a real 5 ms clock, then remeasure all geometry and timing.

    This is an engineering path change, not a guarantee that a filter preserves
    contact. Convex scalar filtering preserves scalar joint ranges. A quintic
    endpoint envelope preserves the common guard and quiet launch/recovery.
    """
    query = np.arange(int(np.ceil(times[-1]/.005))+1)*.005
    sampled = _PoseCurve(times, poses)(query)
    guard = poses[0]
    first = Rotation.from_quat(guard[[4, 5, 6, 3]])
    relative = first.inv()*Rotation.from_quat(sampled[:, [4, 5, 6, 3]])
    values = np.column_stack((sampled[:, :3], relative.as_rotvec(), sampled[:, 7:]))
    filtered = gaussian_filter1d(values, settings.smoothing_sigma_s/.005, axis=0, mode="nearest")
    envelope = (_smoothstep((query-settings.quiet_boundary_s)/settings.endpoint_blend_s)
        *_smoothstep((guard_time-settings.quiet_boundary_s-query)/settings.endpoint_blend_s))
    base = np.r_[guard[:3], np.zeros(3), guard[7:]]
    filtered = base+envelope[:, None]*(filtered-base)
    rotations = first*Rotation.from_rotvec(filtered[:, 3:6])
    result = np.column_stack((filtered[:, :3], rotations.as_quat()[:, [3, 0, 1, 2]], filtered[:, 6:]))
    result[query <= settings.quiet_boundary_s] = guard
    result[query >= guard_time-settings.quiet_boundary_s] = guard
    return _PoseCurve(query, result)


def _unbiased_support_solve(model, prior, temporal, points, orientations, iterations):
    """Project support without an unsolicited COM-centering displacement.

    A pose satisfying its feet/soft targets is an exact fixed point. Each update
    uses native FK/Jacobians and bounded joint steps; no dynamics are integrated.
    The temporal target follows the previous correction, while the original
    smooth pose remains the primary regularizer. Contacts remain soft avoidance
    terms, so the independent serialized-master screen still decides rejection.
    """
    pose=prior.copy()
    data=mujoco.MjData(model)
    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    lower,upper=model.jnt_range[1:].T
    for _ in range(iterations):
        data.qpos[:]=pose
        mujoco.mj_forward(model,data)
        rows,errors=[],[]
        for kind,name,target,weight in points:
            if kind=='site':
                index=model.site(name).id
                current=data.site_xpos[index]
                mujoco.mj_jacSite(model,data,jp,jr,index)
            else:
                index=model.body(name).id
                current=data.xpos[index]
                mujoco.mj_jacBody(model,data,jp,jr,index)
            rows.append(weight*jp.copy())
            errors.append(weight*(target-current))
        for kind,name,target,weight in orientations:
            if kind=='site':
                index=model.site(name).id
                current=data.site_xmat[index].reshape(3,3)
                mujoco.mj_jacSite(model,data,jp,jr,index)
            else:
                index=model.body(name).id
                current=data.xmat[index].reshape(3,3)
                mujoco.mj_jacBody(model,data,jp,jr,index)
            rows.append(weight*jr.copy())
            errors.append(weight*rotation_error(target,current))
        for contact in data.contact:
            g1,g2=int(contact.geom1),int(contact.geom2)
            b1,b2=int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])
            if not b1 or not b2 or contact.dist>=.004:
                continue
            normal=contact.frame.reshape(3,3)[0]
            mujoco.mj_jac(model,data,jp,jr,contact.pos,b1)
            first=jp.copy()
            mujoco.mj_jac(model,data,jp,jr,contact.pos,b2)
            rows.append(5.*(normal@(jp-first))[None])
            errors.append(np.array([5.*(.004-contact.dist)]))
        for target,weight in ((prior,.12),(temporal,.03)):
            displacement=np.zeros(model.nv)
            mujoco.mj_differentiatePos(model,displacement,1.,pose,target)
            rows.append(weight*np.eye(model.nv))
            errors.append(weight*displacement)
        error=np.concatenate(errors)
        if np.max(np.abs(error))<1e-12:
            return pose
        low,high=np.full(model.nv,-.10),np.full(model.nv,.10)
        low[:3],high[:3]=-.02,.02
        low[6:]=np.maximum(low[6:],lower-pose[7:])
        high[6:]=np.minimum(high[6:],upper-pose[7:])
        delta=lsq_linear(np.vstack((np.vstack(rows),.02*np.eye(model.nv))),
            np.r_[error,np.zeros(model.nv)],bounds=(low,high),method='bvls',
            tol=1e-10,max_iter=60).x
        mujoco.mj_integratePos(model,pose,delta,1.)
        if np.any(pose[7:]<lower-1e-10) or np.any(pose[7:]>upper+1e-10):
            raise ValueError('Support projection escaped its joint bounds')
        if np.max(np.abs(delta))<1e-10:
            break
    return pose


def _support_project_curve(model, curve, duration, guard_time, settings):
    """Reproject smoothed paths onto original feet, without a later free filter.

    Hand targets remain soft enough for declared self-collision avoidance; the
    central screen still rejects any remaining intersection. This is offline IK,
    never a physical state replacement or an external support force.
    """
    times = np.arange(int(np.ceil(duration/.005))+1)*.005
    poses = curve(times)
    solver, data = PoseSolver(model), mujoco.MjData(model)
    data.qpos[:] = poses[0]
    mujoco.mj_forward(model,data)
    soles = {side:data.site_xpos[model.site(side+'Foot_sole').id].copy() for side in ('Left','Right')}
    rotations = {side:data.site_xmat[model.site(side+'Foot_sole').id].reshape(3,3).copy()
                 for side in ('Left','Right')}
    output = [poses[0].copy()]
    for i in range(1,len(times)):
        if times[i] <= settings.quiet_boundary_s or times[i] >= guard_time-settings.quiet_boundary_s:
            output.append(poses[0].copy())
            continue
        prior = poses[i]
        data.qpos[:] = prior
        mujoco.mj_forward(model,data)
        points = [('body','Hips',prior[:3].copy(),.8)]
        orient = [('body','Chest',data.xmat[model.body('Chest').id].reshape(3,3).copy(),.35)]
        for side in ('Left','Right'):
            points.extend([('site',side+'Hand_center',data.site_xpos[model.site(side+'Hand_center').id].copy(),3.),
                           ('site',side+'Foot_sole',soles[side],40.)])
            orient.append(('site',side+'Foot_sole',rotations[side],10.))
        delta=np.zeros(model.nv)
        mujoco.mj_differentiatePos(model,delta,1.,poses[i-1],prior)
        seed=output[-1].copy()
        mujoco.mj_integratePos(model,seed,delta,1.)
        seed[solver.address]=np.clip(seed[solver.address],solver.lower,solver.upper)
        if settings.unbiased_support_projection:
            output.append(_unbiased_support_solve(model,prior,seed,points,orient,
                settings.support_projection_iterations))
        else:
            output.append(solver.solve(seed,points,orient,prior=prior,
                iterations=settings.support_projection_iterations,posture_weight=.12,
                shoulder_posture_weight=.15))
    return _PoseCurve(times,np.asarray(output))


def refined_retarget_candidate(source: ReferenceClip, request: PunchRequest, *,
                                guard_qpos_w: np.ndarray,
                                ik_iterations: int = 35,
                                settings: RefinementSettings | None = None) -> ReferenceCandidate:
    """Generate one conservative fixed-foot candidate, recomputing its TTC.

    The temporal prior favors continuation of the previous IK solution while
    retaining the source motion's tangent increments. A per-source-frame joint
    speed bound prevents an IK branch switch from becoming a one-frame jump.
    Neither that bound nor the resulting motion is physically qualified.
    """
    settings = settings or RefinementSettings()
    if type(ik_iterations) is not int or ik_iterations < 1:
        raise ValueError("IK iterations must be a positive integer")
    placed = place_reference(source, request)
    model = mujoco.MjModel.from_xml_path(str(source.model_path))
    data, solver = mujoco.MjData(model), PoseSolver(model)
    physical_lower, physical_upper = solver.lower.copy(), solver.upper.copy()
    guard = np.asarray(guard_qpos_w, dtype=float).copy()
    if guard.shape != (55,) or not np.isfinite(guard).all():
        raise ValueError("Refinement requires a finite complete common guard")
    if (not np.isclose(np.linalg.norm(guard[3:7]), 1., atol=1e-6)
            or np.any(guard[solver.address] < physical_lower)
            or np.any(guard[solver.address] > physical_upper)):
        raise ValueError("Common guard violates quaternion or joint constraints")
    times = np.r_[0., placed.time_s[placed.time_s > placed.launch_time_s]-placed.launch_time_s]
    poses = interpolate_poses(placed.time_s, placed.qpos, times+placed.launch_time_s)
    guard_time = placed.guard_time_s-placed.launch_time_s
    source_clip = ReferenceClip(source.family, source.model_path, times, poses,
        tangent_velocities(model, times, poses), placed.target_w, 0., guard_time, source.source_sha256)
    base = kinematic_arrays(source_clip)
    active = request.family.active_glove_index
    active_site = model.site(("Left" if active == 0 else "Right")+"Hand_center").id
    target = np.asarray(request.snapshot.head_position_w)
    anchor = int(np.argmin(np.where(times < guard_time,
        np.linalg.norm(base["glove_pos_w"][:, active]-placed.target_w, axis=-1), np.inf)))
    peak_time = float(times[anchor])
    if not 0 < peak_time < guard_time:
        raise ValueError("Source lacks an outward strike anchor")
    envelope = np.where(times <= peak_time, _smoothstep(times/peak_time),
                       _smoothstep((guard_time-times)/(guard_time-peak_time)))
    data.qpos[:] = guard
    mujoco.mj_forward(model, data)
    sides = ("Left", "Right")
    soles = {side: data.site_xpos[model.site(side+"Foot_sole").id].copy() for side in sides}
    foot_rotations = {side: data.site_xmat[model.site(side+"Foot_sole").id].reshape(3, 3).copy()
                      for side in sides}
    guard_gloves = np.array([data.site_xpos[model.site(side+"Hand_center").id] for side in sides])
    # The common guard correction is in task space as well as joint/root prior
    # space. This removes the incompatible old-hand/new-posture boundary task.
    glove_offset = guard_gloves-base["glove_pos_w"][0]
    desired_gloves = base["glove_pos_w"]+(1-envelope[:,None,None])*glove_offset
    desired_gloves[:,active] += envelope[:,None]*(target-placed.target_w)
    if settings.recovery_corridor:
        recovery_start = min(guard_time-.2,peak_time+.15)
        anchor_gloves = np.array([[np.interp(recovery_start,times,desired_gloves[:,hand,axis])
                                  for axis in range(3)] for hand in range(2)])
        human_rotation = quaternion_matrix(request.snapshot.human_quat_wxyz)
        for i,time in enumerate(times):
            if time <= recovery_start:
                continue
            fraction = float(_smoothstep((time-recovery_start)/(guard_time-recovery_start)))
            for hand,sign in ((0,1),(1,-1)):
                # Retract each glove through its own outside corridor. The
                # outward punch and follow-through retain their source tasks.
                control = guard_gloves[hand]+human_rotation@np.array([.12,sign*.16,.025])
                desired_gloves[i,hand] = ((1-fraction)**2*anchor_gloves[hand]
                    +2*(1-fraction)*fraction*control+fraction**2*guard_gloves[hand])
    offset = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, offset, 1., poses[0], guard)
    priors = []
    for i, pose in enumerate(poses):
        prior = pose.copy()
        mujoco.mj_integratePos(model, prior, offset, float(1-envelope[i]))
        prior[solver.address] = np.clip(prior[solver.address], physical_lower, physical_upper)
        if times[i] == 0 or times[i] >= guard_time:
            prior = guard.copy()
        priors.append(prior)
    output, max_hand_error, bounded_frames = [guard.copy()], 0., 0
    for i in range(1, len(times)):
        if times[i] >= guard_time:
            output.append(guard.copy())
            continue
        dt = times[i]-times[i-1]
        style = priors[i]
        delta = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, delta, 1., priors[i-1], style)
        predicted = output[-1].copy()
        mujoco.mj_integratePos(model, predicted, delta, settings.prior_prediction_fraction)
        temporal_prior = style.copy()
        mujoco.mj_differentiatePos(model, delta, 1., style, predicted)
        mujoco.mj_integratePos(model, temporal_prior, delta, settings.temporal_prior_fraction)
        step = settings.max_source_joint_speed_rad_s * dt
        solver.lower = np.maximum(physical_lower, output[-1][solver.address]-step)
        solver.upper = np.minimum(physical_upper, output[-1][solver.address]+step)
        predicted[solver.address] = np.clip(predicted[solver.address], solver.lower, solver.upper)
        temporal_prior[solver.address] = np.clip(temporal_prior[solver.address], physical_lower, physical_upper)
        data.qpos[:] = style
        mujoco.mj_forward(model, data)
        points = [("body", "Hips", style[:3].copy(), .8)]
        rotations = [("body", "Chest", data.xmat[model.body("Chest").id].reshape(3, 3).copy(), .35)]
        desired = desired_gloves[i]
        for side_index, side in enumerate(sides):
            points.extend([("site", side+"Hand_center", desired[side_index], 5.),
                           ("site", side+"Foot_sole", soles[side], 12.)])
            rotations.append(("site", side+"Foot_sole", foot_rotations[side], 1.5))
        solved = solver.solve(predicted, points, rotations, prior=temporal_prior,
            iterations=ik_iterations, posture_weight=.08, shoulder_posture_weight=.12)
        bounded_frames += int(np.any(np.abs(solved[solver.address]-output[-1][solver.address]) >= step-1e-6))
        output.append(solved)
        data.qpos[:] = solved
        mujoco.mj_forward(model, data)
        max_hand_error = max(max_hand_error, float(np.linalg.norm(data.site_xpos[active_site]-desired[active])))
    output = np.asarray(output)
    # Root and joint interpolation are changed here. Only now measure contact;
    # the original reference contact time is deliberately discarded.
    curve = _smooth_guard_curve(times, output, guard_time, settings)
    if settings.project_support_after_smoothing:
        curve = _support_project_curve(model,curve,times[-1],guard_time,settings)
    outward_end = min(guard_time, peak_time+.15)
    source_contact = _fk_contact(model, curve, target, active_site, outward_end)
    old_knots = np.array([0., source_contact, guard_time, times[-1]])
    new_knots = np.array([0., request.ttc_s, request.ttc_s+guard_time-source_contact,
                          request.ttc_s+times[-1]-source_contact])
    if np.any(np.diff(old_knots) <= 0) or np.any(np.diff(new_knots) <= 0):
        raise ValueError("Refined reference requires distinct contact, guard and hold boundaries")
    to_source = PchipInterpolator(new_knots, old_knots)
    outward_end_s = float(brentq(lambda t: float(to_source(t))-outward_end,
                                0., new_knots[2], xtol=1e-12))
    duration = float(np.ceil((new_knots[-1]-1e-10)/.02)*.02)
    final_times = np.arange(round(duration/.005)+1)*.005
    sample_source = to_source(np.clip(final_times, 0., new_knots[-1]))
    final_poses = curve(sample_source)
    if settings.quiet_boundary_s:
        final_poses[sample_source <= settings.quiet_boundary_s] = guard
        final_poses[sample_source >= guard_time-settings.quiet_boundary_s] = guard
    final_poses[0] = guard
    final_poses[final_times >= new_knots[2]] = guard
    if np.any(final_poses[:, 7:] < physical_lower-1e-9) or np.any(final_poses[:, 7:] > physical_upper+1e-9):
        raise ValueError("Refined interpolation violates physical joint limits")
    backend = ('refined_v3' if settings.unbiased_support_projection else
        'refined_v2' if settings.project_support_after_smoothing else 'refined_v1')
    metadata = {**source.metadata, "origin": ("offline_common_guard_quiet_support_refinement_v3"
        if settings.unbiased_support_projection else "offline_common_guard_support_refinement_v2"
        if settings.project_support_after_smoothing else "offline_common_guard_continuity_refinement_v1"),
        "requested_ttc_s": request.ttc_s, "g1_snapshot_present": True,
        "snapshot_fingerprint": request.snapshot.fingerprint, "launch_snapshot": asdict(request.snapshot),
        "guard_id": request.guard_id, "launch_features": list(launch_features(request)),
        "guard_source": "explicit_common_guard", "common_guard_validated": False,
        "launch_feature_layout": list(LAUNCH_FEATURE_LAYOUT), "support_motion": "planted_feet",
        "qualified_for_training": False, "refinement_settings": asdict(settings),
        "outward_end_s": outward_end_s,
        "retiming_source_knots_s": old_knots.tolist(),
        "retiming_final_knots_s": new_knots.tolist(),
        "interpolation": "PCHIP, configured Gaussian tangent-chart refinement, quintic guard endpoints; contact re-solved"}
    final = ReferenceClip(source.family, source.model_path, final_times, final_poses,
        tangent_velocities(model, final_times, final_poses), target, 0., float(new_knots[2]),
        source.source_sha256, metadata)
    if settings.quiet_boundary_s:
        acceleration=np.gradient(final.qvel,final_times,axis=0,edge_order=2)
        if (np.max(np.abs(final.qvel[[0,-1]])) > 1e-12
                or np.max(np.abs(acceleration[[0,-1]])) > 1e-10):
            raise ValueError('Retiming compressed quiet boundaries below exact sampled derivative readiness')
    arrays = kinematic_arrays(final)
    contact = _first_crossing(final_times, arrays["glove_pos_w"], target, active, float(new_knots[2]))
    if contact is None or abs(contact-request.ttc_s) > .005+1e-9:
        raise ValueError("Refined sampled FK does not retain the newly solved TTC")
    sole_drift = np.linalg.norm(arrays["sole_pos_w"]-arrays["sole_pos_w"][0], axis=-1)
    if sole_drift.max() > settings.maximum_interpolation_sole_drift_m:
        raise ValueError("Refined trajectory exceeds its declared kinematic foot-drift screening bound")
    digest = hashlib.sha256(final.qpos.tobytes()+final.time_s.tobytes()+request.snapshot.fingerprint.encode()).hexdigest()
    return ReferenceCandidate(final, request, {
        "qualified_for_training": False, "candidate_content_sha256": digest,
        "refinement_backend": backend,
        "refinement_settings": asdict(settings),
        "requested_ttc_s": request.ttc_s, "interpolated_reference_ttc_s": contact,
        "timing_error_s": contact-request.ttc_s, "source_fk_contact_time_s": source_contact,
        "outward_end_s": outward_end_s,
        "contact_remeasured_after_refinement": True, "max_ik_active_glove_error_m": max_hand_error,
        "max_sole_drift_m": float(sole_drift.max()), "source_speed_bounded_frame_count": bounded_frames,
        "external_support_applied": False, "physics_executed": False,
        "dynamic_feasibility": "unqualified", "family_identity": "requires_review",
        "velocity_lead_used": False, "guard_qpos_max_return_error": float(np.max(abs(final_poses[-1]-guard)))})
