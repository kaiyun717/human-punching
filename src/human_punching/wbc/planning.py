"""Deterministic offline target/guard retargeting and explicit TTC candidates.

These functions generate kinematic candidates, never certify a feasible command.
No subsequent G1 observation is accepted. The launch snapshot is immutable and
the plan always targets its initial head position, with no velocity extrapolation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation

from human_punching.scalar_reference import PoseSolver
from .contracts import LaunchPlan, PunchRequest, quaternion_matrix
from .references import (ReferenceClip, interpolate_poses, kinematic_arrays,
                         tangent_velocities)


LAUNCH_FEATURE_LAYOUT = (
    "g1_root_position_h[3]", "g1_root_quat_h_wxyz[4]",
    "g1_root_linear_velocity_h[3]", "g1_root_angular_velocity_h[3]",
    "g1_joint_position[29]", "g1_joint_velocity[29]",
    "head_position_h[3]", "head_velocity_h[3]",
)


def launch_features(request: PunchRequest) -> tuple[float, ...]:
    """Full frozen state plus useful task-frame features; no live target inputs."""
    snapshot = request.snapshot
    human_rotation = quaternion_matrix(snapshot.human_quat_wxyz)
    g1_rotation = quaternion_matrix(snapshot.g1_qpos[3:7])
    relative_quat = Rotation.from_matrix(human_rotation.T @ g1_rotation).as_quat(canonical=True)[[3, 0, 1, 2]]
    # MuJoCo free-joint translation velocities are world-frame, while its
    # rotational qvel is in the free body's local frame. Transform both to H.
    linear = human_rotation.T @ np.asarray(snapshot.g1_qvel[:3])
    angular = human_rotation.T @ g1_rotation @ np.asarray(snapshot.g1_qvel[3:6])
    return (*snapshot.world_to_human_point(snapshot.g1_qpos[:3]), *relative_quat,
            *linear, *angular, *snapshot.g1_qpos[7:], *snapshot.g1_qvel[6:],
            *snapshot.world_to_human_point(snapshot.head_position_w),
            *snapshot.world_to_human_vector(snapshot.head_velocity_w))


def place_reference(clip: ReferenceClip, request: PunchRequest) -> ReferenceClip:
    """Place source pelvis ground projection at the frozen task-frame origin.

    The source's MuJoCo root quaternion already contains SOMA's +90-degree asset
    rotation. Compose the canonical task heading with it, never replace it.
    """
    if clip.family != request.family.value:
        raise ValueError("Punch request and source family differ")
    origin = clip.qpos[0, :3].copy()
    origin[2] = 0
    rotation = quaternion_matrix(request.snapshot.human_quat_wxyz)
    poses = clip.qpos.copy()
    poses[:, :3] = (poses[:, :3] - origin) @ rotation.T + request.snapshot.human_origin_w
    orient = Rotation.from_matrix(rotation) * Rotation.from_quat(poses[:, [4, 5, 6, 3]])
    poses[:, 3:7] = orient.as_quat()[:, [3, 0, 1, 2]]
    target = rotation @ (clip.target_w - origin) + request.snapshot.human_origin_w
    model = mujoco.MjModel.from_xml_path(str(clip.model_path))
    return ReferenceClip(clip.family, clip.model_path, clip.time_s, poses,
                         tangent_velocities(model, clip.time_s, poses), target,
                         clip.launch_time_s, clip.guard_time_s, clip.source_sha256,
                         clip.metadata)


def _smoothstep(x: np.ndarray | float):
    x = np.clip(x, 0, 1)
    return x**3 * (10 + x * (-15 + 6*x))


def _first_crossing(times: np.ndarray, gloves: np.ndarray, target: np.ndarray,
                    active: int, before: float) -> float | None:
    distance = np.linalg.norm(gloves[:, active] - target, axis=1)
    found = np.flatnonzero((distance <= .14) & (times < before))
    if not len(found):
        return None
    i = int(found[0])
    if i == 0:
        return float(times[0])
    fraction = (distance[i-1] - .14) / max(distance[i-1] - distance[i], 1e-12)
    return float(times[i-1] + fraction * (times[i] - times[i-1]))


@dataclass(frozen=True)
class ReferenceCandidate:
    clip: ReferenceClip
    request: PunchRequest
    diagnostics: dict

    @property
    def qualified(self) -> bool:
        # Offline IK/retiming cannot establish unsupported physical execution.
        return False

    def launch_plan(self, reference_id: str, artifact_sha256: str) -> LaunchPlan:
        """Bind a saved candidate identity; admission still requires qualification."""
        return LaunchPlan(self.request, reference_id, artifact_sha256,
                          self.request.snapshot.head_position_w)


@dataclass(frozen=True)
class GuardCandidate:
    qpos_w: tuple[float, ...]
    lateral_width_m: float
    left_lead_m: float
    maximum_sole_error_m: float
    maximum_selected_self_penetration_m: float
    qualified_for_launch: bool = False


def common_guard_candidate(jab_source: ReferenceClip, *,
                            human_origin_w=(0., 0., 0.), human_quat_wxyz=(1., 0., 0., 0.),
                            lateral_width_m: float = .50, left_lead_m: float = .20,
                            iterations: int = 120) -> GuardCandidate:
    """One deterministic orthodox guard seed shared by every punch family.

    Bounds are candidate search bounds, not measured admissibility. Use this same
    returned pose for all four retargets; qualify quiet unsupported balance before
    calling it an admissible launch guard. No learned checkpoint is consumed.
    """
    if jab_source.family != "jab":
        raise ValueError("Use the same jab-derived guard seed for all four families")
    if not .45 <= lateral_width_m <= .55 or not .15 <= left_lead_m <= .25:
        raise ValueError("Guard stance lies outside the declared candidate search box")
    if iterations < 1:
        raise ValueError("Guard IK iterations must be positive")
    origin = np.asarray(human_origin_w, dtype=float)
    rotation = quaternion_matrix(human_quat_wxyz)
    if origin.shape != (3,) or not np.isfinite(origin).all() or abs(origin[2]) > 1e-8:
        raise ValueError("Human task origin must lie on the ground plane")
    if not np.allclose(rotation[:, 2], [0., 0., 1.], atol=1e-8):
        raise ValueError("Human task frame must be level")
    model = mujoco.MjModel.from_xml_path(str(jab_source.model_path))
    solver, data = PoseSolver(model), mujoco.MjData(model)
    pose = jab_source.qpos[0].copy()
    pose[:3] = origin + [0., 0., .95]
    pose[3:7] = (Rotation.from_matrix(rotation) *
                  Rotation.from_euler("z", np.pi/2)).as_quat()[[3, 0, 1, 2]]
    for side in ("Left", "Right"):
        pose[model.joint(f"{side}Shin_x").qposadr[0]] = .25
        pose[model.joint(f"{side}Leg_x").qposadr[0]] = -.12
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    head = data.xpos[model.body("Head").id].copy()
    points, rotations = [("body", "Hips", pose[:3].copy(), .8)], []
    targets = {}
    for side, sign in (("Left", 1), ("Right", -1)):
        target = origin + rotation @ np.array([sign*left_lead_m/2, sign*lateral_width_m/2, .0004])
        targets[side] = target
        points.extend([("site", f"{side}Foot_sole", target, 18.),
                       ("site", f"{side}Hand_center",
                        head + rotation @ np.array([.23, sign*.13, -.12]), 5.)])
        rotations.append(("site", f"{side}Foot_sole",
                           rotation @ Rotation.from_euler("z", np.pi/2).as_matrix(), 1.5))
    rotations.append(("body", "Chest", rotation @ Rotation.from_euler("z", np.pi/2).as_matrix(), .35))
    solved = solver.solve(pose, points, rotations, prior=pose,
                          iterations=iterations, posture_weight=.025)
    # The task origin is defined by the initial pelvis ground projection. Rebase
    # the entire solved stance horizontally, preserving relative foot geometry.
    rebase = np.r_[origin[:2]-solved[:2], 0.]
    solved[:3] += rebase
    targets = {side: target+rebase for side, target in targets.items()}
    data.qpos[:] = solved
    mujoco.mj_forward(model, data)
    sole_error = max(float(np.linalg.norm(data.site_xpos[model.site(f"{side}Foot_sole").id]-target))
                     for side, target in targets.items())
    penetration = max([-float(c.dist) for c in data.contact
                       if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]] + [0.])
    return GuardCandidate(tuple(solved), lateral_width_m, left_lead_m, sole_error, penetration)


def retarget_candidate(source: ReferenceClip, request: PunchRequest, *,
                        guard_qpos_w: np.ndarray | None = None,
                        ik_iterations: int = 35) -> ReferenceCandidate:
    """Generate a whole-body candidate with planted feet and preserved style priors.

    An optional common guard is supplied in world generalized coordinates. Both
    its feet remain fixed during this conservative first retargeter; future
    pivot/heel-lift references require their own explicit support trajectory.
    Whole-body dynamics, effort/support feasibility, and family identity still
    require qualification. Failure is reported, never repaired by moving target
    or silently changing the requested TTC.
    """
    if ik_iterations < 1:
        raise ValueError("IK iterations must be positive")
    placed = place_reference(source, request)
    model = mujoco.MjModel.from_xml_path(str(source.model_path))
    data = mujoco.MjData(model)
    solver = PoseSolver(model)
    # Clip clock zero is the launch boundary, excluding Task09's inserted hold.
    remaining = placed.time_s[placed.time_s > placed.launch_time_s] - placed.launch_time_s
    times = np.r_[0.0, remaining]
    poses = interpolate_poses(placed.time_s, placed.qpos, times + placed.launch_time_s)
    guard_time = placed.guard_time_s - placed.launch_time_s
    guard = poses[0].copy() if guard_qpos_w is None else np.asarray(guard_qpos_w, dtype=float).copy()
    if guard.shape != (model.nq,) or not np.isfinite(guard).all():
        raise ValueError("Guard must be a finite complete human pose")
    if not np.isclose(np.linalg.norm(guard[3:7]), 1, atol=1e-6):
        raise ValueError("Guard root quaternion must be normalized wxyz")
    if np.any(guard[solver.address] < solver.lower) or np.any(guard[solver.address] > solver.upper):
        raise ValueError("Guard violates human joint limits")
    data.qpos[:] = guard
    mujoco.mj_forward(model, data)
    soles = {side: data.site_xpos[model.site(f"{side}Foot_sole").id].copy()
             for side in ("Left", "Right")}
    foot_rotations = {side: data.site_xmat[model.site(f"{side}Foot_sole").id].reshape(3, 3).copy()
                      for side in ("Left", "Right")}
    target = np.asarray(request.snapshot.head_position_w)
    active = request.family.active_glove_index
    active_site = model.site(("Left" if active == 0 else "Right") + "Hand_center").id
    # Contact anchor comes from the existing path, not source playback speed.
    base_clip = ReferenceClip(source.family, source.model_path, times, poses,
                              tangent_velocities(model, times, poses), placed.target_w,
                              0, guard_time, source.source_sha256)
    base_arrays = kinematic_arrays(base_clip)
    closest = int(np.argmin(np.where(times < guard_time,
                 np.linalg.norm(base_arrays["glove_pos_w"][:, active] - placed.target_w, axis=1), np.inf)))
    peak_time = float(times[closest])
    if peak_time <= 0 or guard_time <= peak_time:
        raise ValueError("Source has no valid outward strike anchor")
    shift = target - placed.target_w
    envelope = np.where(times <= peak_time, _smoothstep(times / peak_time),
                         _smoothstep((guard_time - times) / (guard_time - peak_time)))
    # Map source initial guard to supplied common guard in tangent coordinates.
    offset = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, offset, 1., poses[0], guard)
    output = []
    maximum_hand_error = 0.
    for i, pose in enumerate(poses):
        if times[i] == 0 or times[i] >= guard_time:
            output.append(guard.copy())
            continue
        prior = pose.copy()
        mujoco.mj_integratePos(model, prior, offset, float(1-envelope[i]))
        prior[solver.address] = np.clip(prior[solver.address], solver.lower, solver.upper)
        data.qpos[:] = prior
        mujoco.mj_forward(model, data)
        points = [("body", "Hips", prior[:3].copy(), .8)]
        rotations = [("body", "Chest", data.xmat[model.body("Chest").id].reshape(3, 3).copy(), .35)]
        desired_active = base_arrays["glove_pos_w"][i, active] + envelope[i] * shift
        for side_index, side in enumerate(("Left", "Right")):
            hand_id = model.site(f"{side}Hand_center").id
            hand_target = desired_active if side_index == active else data.site_xpos[hand_id].copy()
            points.extend([("site", f"{side}Hand_center", hand_target, 5.),
                           ("site", f"{side}Foot_sole", soles[side], 12.)])
            rotations.append(("site", f"{side}Foot_sole", foot_rotations[side], 1.5))
        solved = solver.solve(prior, points, rotations, prior=prior,
                              iterations=ik_iterations, posture_weight=.08)
        output.append(solved)
        data.qpos[:] = solved
        mujoco.mj_forward(model, data)
        maximum_hand_error = max(maximum_hand_error, float(np.linalg.norm(data.site_xpos[active_site]-desired_active)))
    output = np.asarray(output)
    candidate = ReferenceClip(source.family, source.model_path, times, output,
                              tangent_velocities(model, times, output), target,
                              0., guard_time, source.source_sha256)
    arrays = kinematic_arrays(candidate)
    outward_end = min(guard_time, peak_time + .15)
    contact = _first_crossing(times, arrays["glove_pos_w"], target, active, outward_end)
    if contact is None or contact <= 0:
        raise ValueError("Retargeted candidate does not produce a post-launch outward contact")
    # Retiming scales only launch-to-contact. Contact-to-guard/recovery duration
    # is retained. Every joint/root/body velocity is recomputed on the new clock.
    knots = [0., contact, guard_time]
    new_knots = [0., request.ttc_s, request.ttc_s + guard_time-contact]
    if times[-1] > guard_time + 1e-9:
        knots.append(float(times[-1]))
        new_knots.append(request.ttc_s + float(times[-1])-contact)
    new_times = PchipInterpolator(knots, new_knots)(times)
    if np.any(np.diff(new_times) <= 0):
        raise ValueError("Requested TTC produces a nonmonotone reference clock")
    new_guard = request.ttc_s + guard_time - contact
    metadata = {**source.metadata, "origin": "offline_constrained_target_guard_candidate",
                "requested_ttc_s": request.ttc_s, "g1_snapshot_present": True,
                "snapshot_fingerprint": request.snapshot.fingerprint,
                "launch_snapshot": asdict(request.snapshot),
                "guard_id": request.guard_id, "launch_features": list(launch_features(request)),
                "guard_source": "explicit_common_guard" if guard_qpos_w is not None else "task09_family_guard",
                "common_guard_validated": False,
                "launch_feature_layout": list(LAUNCH_FEATURE_LAYOUT),
                "support_motion": "planted_feet", "qualified_for_training": False}
    final = ReferenceClip(source.family, source.model_path, new_times, output,
                          tangent_velocities(model, new_times, output), target, 0.,
                          new_guard, source.source_sha256, metadata)
    final_arrays = kinematic_arrays(final)
    observed_contact = _first_crossing(new_times, final_arrays["glove_pos_w"], target, active,
                                      float(PchipInterpolator(knots, new_knots)(outward_end)))
    drift = np.linalg.norm(final_arrays["sole_pos_w"]-final_arrays["sole_pos_w"][0], axis=-1)
    digest = hashlib.sha256(final.qpos.tobytes()+final.time_s.tobytes()+request.snapshot.fingerprint.encode()).hexdigest()
    return ReferenceCandidate(final, request,
        {"qualified_for_training": False, "candidate_content_sha256": digest,
         "requested_ttc_s": request.ttc_s, "interpolated_reference_ttc_s": observed_contact,
         "timing_error_s": None if observed_contact is None else observed_contact-request.ttc_s,
         "max_ik_active_glove_error_m": maximum_hand_error,
         "max_sole_drift_m": float(drift.max()), "external_support_applied": False,
         "physics_executed": False, "dynamic_feasibility": "unqualified",
         "family_identity": "requires_review", "velocity_lead_used": False,
         "guard_qpos_max_return_error": float(np.max(np.abs(output[-1]-guard)))})
