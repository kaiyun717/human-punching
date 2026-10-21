"""Task 10B Phase 1 functional guard, independent of historical hit/recovery gates.

This module observes saved states. It never steps a simulator, modifies policy
history, rewrites a reference, or converts recovery after a miss into a hit.
Support row k describes the interval ending at k; pose/velocity predicates are
required at both ends of every counted dwell interval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError


@dataclass(frozen=True)
class FunctionalGuardThresholds:
    version: str = "task10b_functional_guard_pilot_v1"
    glove_position_m: float = .05
    root_linear_speed_m_s: float = .05
    root_angular_speed_rad_s: float = .12
    glove_relative_speed_m_s: float = .15
    dwell_s: float = .20
    minimum_root_height_fraction: float = .75
    maximum_root_tilt_rad: float = .60
    support_normal_force_n: float = 1.e-6
    diagnostic_glove_orientation_rad: float = float(np.deg2rad(20.))
    diagnostic_torso_orientation_rad: float = .15
    diagnostic_root_height_fraction: float = .90
    diagnostic_com_margin_m: float = .02
    diagnostic_joint_speed_rad_s: float = .50
    diagnostic_clearance_m: float = .16
    diagnostic_stability_s: float = .50

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("A predicate version is required")
        for key, value in asdict(self).items():
            if key != "version" and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be finite and positive")
        if self.support_normal_force_n != 1.e-6:
            raise ValueError("Phase 1 retains qualified historical >1e-6 N support decoding")


class FunctionalRecoveryStateMachine:
    """Single uninterrupted cycle; launch and outward completion are explicit.

    Readiness is current, not latched permission to launch later. An explicit
    ``readiness_valid`` supplies the reviewed initial admission predicate;
    without it, the stricter quiet-dwell fixture behavior is used. Completion is
    latched only after outward completion and a new recovery dwell. Contact is
    recorded separately and is never a recovery prerequisite. The caller owns
    actor-boundary launch and the reference's outward endpoint annotation.
    """
    def __init__(self, thresholds: FunctionalGuardThresholds | None = None):
        self.thresholds = thresholds or FunctionalGuardThresholds()
        self.last_time_s: float | None = None
        self.previous_pose_valid = False
        self.ready_dwell_s = 0.
        self.recovery_dwell_s = 0.
        self.readiness_time_s: float | None = None
        self.launch_time_s: float | None = None
        self.outward_completion_time_s: float | None = None
        self.recovery_time_s: float | None = None
        self.first_contact_time_s: float | None = None
        self.complete = False

    def observe(self, time_s: float, pose_valid: bool, supported: bool, *,
                launch: bool = False, outward_complete: bool = False,
                contact: bool = False, readiness_valid: bool | None = None) -> dict[str, Any]:
        t = float(time_s)
        if not np.isfinite(t) or t < 0 or (self.last_time_s is not None and t <= self.last_time_s):
            raise ValueError("Observation times must be finite, nonnegative and strictly increasing")
        dt = 0. if self.last_time_s is None else t - self.last_time_s
        interval_valid = bool(pose_valid and self.previous_pose_valid and supported)
        self.ready_dwell_s = self.ready_dwell_s + dt if interval_valid else 0.
        ready = (self.ready_dwell_s + 1.e-10 >= self.thresholds.dwell_s
                 if readiness_valid is None else bool(readiness_valid))
        if ready and self.readiness_time_s is None:
            self.readiness_time_s = t
        if launch:
            if self.launch_time_s is not None:
                raise ValueError("Only one launch is permitted per uninterrupted cycle")
            if not ready:
                raise ValueError("Launch requires current declared readiness predicate")
            self.launch_time_s = t
        if outward_complete:
            if self.launch_time_s is None or t <= self.launch_time_s:
                raise ValueError("Outward completion must follow an actual launch")
            if self.outward_completion_time_s is not None:
                raise ValueError("Outward completion can only be recorded once")
            self.outward_completion_time_s = t
            self.recovery_dwell_s = 0.
        elif self.outward_completion_time_s is not None:
            self.recovery_dwell_s = self.recovery_dwell_s + dt if interval_valid else 0.
        if (self.outward_completion_time_s is not None and
                self.recovery_dwell_s + 1.e-10 >= self.thresholds.dwell_s and not self.complete):
            self.recovery_time_s = t
            self.complete = True
        if contact and self.first_contact_time_s is None:
            self.first_contact_time_s = t
        self.last_time_s, self.previous_pose_valid = t, bool(pose_valid)
        phase = ("complete" if self.complete else "recovery" if self.outward_completion_time_s is not None
                 else "outward" if self.launch_time_s is not None else "holding")
        return dict(phase=phase, ready=bool(ready), complete=self.complete,
                    ready_dwell_s=float(self.ready_dwell_s), recovery_dwell_s=float(self.recovery_dwell_s),
                    readiness_time_s=self.readiness_time_s, launch_time_s=self.launch_time_s,
                    outward_completion_time_s=self.outward_completion_time_s,
                    recovery_time_s=self.recovery_time_s, first_contact_time_s=self.first_contact_time_s)


def _bindings(model, prefix: str) -> dict[str, Any]:
    roots = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    if len(roots) != 1 or len(joints) + 1 != model.njnt:
        raise ValueError("Phase 1 measures an isolated scalar human with one free root")
    root = int(roots[0])
    result = dict(root_body=int(model.jnt_bodyid[root]), root_qadr=int(model.jnt_qposadr[root]),
                  root_vadr=int(model.jnt_dofadr[root]), qadr=model.jnt_qposadr[joints],
                  vadr=model.jnt_dofadr[joints], chest=model.body(prefix + "Chest").id,
                  head=model.body(prefix + "Head").id,
                  gloves=[model.site(prefix + side + "Hand_center").id for side in ("Left", "Right")],
                  feet=[model.geom(prefix + "physical/" + side + "Foot").id for side in ("Left", "Right")])
    for geom in result["feet"]:
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Qualified support footprint requires physical foot boxes")
    return result


def _rotation_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ji,...jk->...ik", a, b)
    return np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1., 1.))


def _json_measurement(value):
    """Preserve invalid measurements as null; the corresponding finite gate fails."""
    if isinstance(value, list):
        return [_json_measurement(item) for item in value]
    return value if not isinstance(value, float) or np.isfinite(value) else None


def _kinematics(model, qpos, qvel, bindings, sole_local) -> dict[str, np.ndarray]:
    """Pure FK/Jacobians only, with no forward contact/force recomputation."""
    b = bindings
    shapes = dict(glove_position_w=(2, 3), glove_position_chest=(2, 3),
                  glove_rotation_chest=(2, 3, 3), glove_velocity_chest=(2, 3),
                  chest_rotation_w=(3, 3), head_rotation_w=(3, 3),
                  root_rotation_w=(3, 3), root_position_w=(3,),
                  sole_corners_w=(2, 4, 3), com_w=(3,))
    result = {key: np.full((len(qpos), *shape), np.nan) for key, shape in shapes.items()}
    data = mujoco.MjData(model)
    jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
    for i, (q, v) in enumerate(zip(qpos, qvel, strict=True)):
        rq = b["root_qadr"]
        if (not np.isfinite(q).all() or not np.isfinite(v).all() or
                abs(np.linalg.norm(q[rq+3:rq+7])-1.) > 1.e-5):
            continue
        data.qpos[:], data.qvel[:] = q, v
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        rotation = data.xmat[b["chest"]].reshape(3, 3)
        mujoco.mj_jacBody(model, data, jp, jr, b["chest"])
        chest_v, chest_w = jp @ v, jr @ v
        for side, site in enumerate(b["gloves"]):
            offset = data.site_xpos[site] - data.xpos[b["chest"]]
            result["glove_position_w"][i, side] = data.site_xpos[site]
            result["glove_position_chest"][i, side] = rotation.T @ offset
            result["glove_rotation_chest"][i, side] = rotation.T @ data.site_xmat[site].reshape(3, 3)
            mujoco.mj_jacSite(model, data, jp, jr, site)
            result["glove_velocity_chest"][i, side] = rotation.T @ (jp @ v - chest_v - np.cross(chest_w, offset))
        for side, geom in enumerate(b["feet"]):
            result["sole_corners_w"][i, side] = sole_local[side] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        for name, body in (("chest", b["chest"]), ("head", b["head"]), ("root", b["root_body"])):
            result[name + "_rotation_w"][i] = data.xmat[body].reshape(3, 3)
        result["root_position_w"][i] = data.xpos[b["root_body"]]
        result["com_w"][i] = data.subtree_com[b["root_body"]]
    return result


def derive_guard_nominal(model, guard_qpos, *, family: str, prefix: str = "",
                         provenance: dict | None = None) -> dict[str, Any]:
    """Qualified family's initial reference pose defines centers, never trial means.

    For each physical foot box, select the face whose outward normal points most
    downward in the qualified nominal pose. Track its four fixed material corners
    thereafter. The projected hull is a geometric diagnostic; positive measured
    foot-floor forces, not FK or footprint proximity, establish actual support.
    """
    q = np.asarray(guard_qpos, float)
    if q.shape != (model.nq,) or not np.isfinite(q).all():
        raise ValueError("Finite full-model nominal guard pose required")
    b = _bindings(model, prefix)
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    corners, faces = [], []
    for geom in b["feet"]:
        rotation = data.geom_xmat[geom].reshape(3, 3)
        axis = int(np.argmax(abs(rotation[2])))
        sign = -1. if rotation[2, axis] >= 0 else 1.
        other = [index for index in range(3) if index != axis]
        local = np.zeros((4, 3))
        local[:, axis] = sign * model.geom_size[geom, axis]
        local[:, other] = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * model.geom_size[geom, other]
        corners.append(local)
        faces.append(dict(geom=model.geom(geom).name, local_normal_axis=axis, local_normal_sign=sign))
    kin = _kinematics(model, q[None], np.zeros((1, model.nv)), b, np.asarray(corners))
    return dict(schema="task10b_family_nominal_guard_v1", family=family, prefix=prefix,
                qpos=q.tolist(), root_height_m=float(kin["root_position_w"][0, 2]),
                glove_centers_chest_m=kin["glove_position_chest"][0].tolist(),
                glove_rotation_chest=kin["glove_rotation_chest"][0].tolist(),
                chest_rotation_w=kin["chest_rotation_w"][0].tolist(),
                head_rotation_w=kin["head_rotation_w"][0].tolist(),
                root_rotation_w=kin["root_rotation_w"][0].tolist(),
                sole_local_corners_m=np.asarray(corners).tolist(), sole_faces=faces,
                nominal_sole_corners_w=kin["sole_corners_w"][0].tolist(),
                nominal_com_w=kin["com_w"][0].tolist(), provenance=provenance or {},
                support_definition="Both physical foot-floor recorded public normal forces >1e-6 N during preceding interval",
                com_definition="Signed minimum halfspace margin of COM projection in current projected hull of nominal sole material corners; diagnostic, not contact-manifold or dynamic feasibility proof")


def measure_functional_guard(model, qpos, qvel, support_contact, nominal: dict,
                             target_w, *, thresholds: FunctionalGuardThresholds | None = None) -> dict[str, np.ndarray]:
    cfg = thresholds or FunctionalGuardThresholds()
    q, v = np.asarray(qpos, float), np.asarray(qvel, float)
    if q.ndim != 2 or q.shape[1] != model.nq or v.shape != (len(q), model.nv) or not len(q):
        raise ValueError("Matching nonempty full-model state arrays required")
    support = np.asarray(support_contact)
    if support.shape != (len(q), 2) or not np.isin(support, [0, 1]).all():
        raise ValueError("Explicit per-foot support flags are required at every saved sample")
    target = np.asarray(target_w, float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Phase 1 requires a finite isolated frozen target")
    b = _bindings(model, nominal["prefix"])
    kin = _kinematics(model, q, v, b, np.asarray(nominal["sole_local_corners_m"]))
    finite = np.isfinite(q).all(-1) & np.isfinite(v).all(-1)
    finite &= np.isfinite(kin["glove_position_chest"]).all(axis=(1, 2))
    rq, rv = b["root_qadr"], b["root_vadr"]
    # Match the historical physical fall criterion exactly: upright is relative
    # to this trial's initial root orientation, with qualified reference height.
    local_up = kin["root_rotation_w"][0].T @ np.array([0., 0., 1.])
    up = np.einsum("nij,j->ni", kin["root_rotation_w"], local_up)
    tilt = np.arccos(np.clip(up[:, 2], -1, 1))
    height = kin["root_position_w"][:, 2]
    glove_error = np.linalg.norm(kin["glove_position_chest"] - nominal["glove_centers_chest_m"], axis=-1)
    glove_speed = np.linalg.norm(kin["glove_velocity_chest"], axis=-1)
    root_linear = np.linalg.norm(v[:, rv:rv+3], axis=-1)
    root_angular = np.linalg.norm(v[:, rv+3:rv+6], axis=-1)
    upright = (height >= cfg.minimum_root_height_fraction*nominal["root_height_m"]) & (tilt <= cfg.maximum_root_tilt_rad)
    components = dict(finite=finite, upright=upright,
                      hand_position=np.all(glove_error <= cfg.glove_position_m, axis=-1),
                      root_linear_speed=root_linear <= cfg.root_linear_speed_m_s,
                      root_angular_speed=root_angular <= cfg.root_angular_speed_rad_s,
                      hand_relative_speed=np.all(glove_speed <= cfg.glove_relative_speed_m_s, axis=-1))
    margin = np.full(len(q), np.nan)
    for i in np.flatnonzero(finite):
        try:
            hull = ConvexHull(kin["sole_corners_w"][i, :, :, :2].reshape(-1, 2))
            distances = hull.equations[:, :2] @ kin["com_w"][i, :2] + hull.equations[:, 2]
            margin[i] = -np.max(distances / np.linalg.norm(hull.equations[:, :2], axis=1))
        except QhullError:
            pass
    nominal_q = np.asarray(nominal["qpos"])
    result = dict(**kin, **{"operational_"+name: value for name, value in components.items()},
                  operational_pose=np.logical_and.reduce(list(components.values())),
                  support_contact=support.astype(bool), operational_support=np.all(support, axis=1),
                  hand_position_error_m=glove_error, hand_relative_speed_m_s=glove_speed,
                  root_linear_speed_m_s=root_linear, root_angular_speed_rad_s=root_angular,
                  root_height_m=height, root_tilt_rad=tilt,
                  root_height_fraction=height/nominal["root_height_m"],
                  root_position_error_m=np.linalg.norm(q[:, rq:rq+3]-nominal_q[rq:rq+3], axis=-1),
                  root_orientation_error_rad=_rotation_error(np.asarray(nominal["root_rotation_w"]), kin["root_rotation_w"]),
                  joint_return_error_rad=abs(q[:, b["qadr"]]-nominal_q[b["qadr"]]),
                  joint_speed_rad_s=abs(v[:, b["vadr"]]),
                  hand_orientation_error_rad=_rotation_error(np.asarray(nominal["glove_rotation_chest"]), kin["glove_rotation_chest"]),
                  torso_orientation_error_rad=_rotation_error(np.asarray(nominal["chest_rotation_w"]), kin["chest_rotation_w"]),
                  head_orientation_error_rad=_rotation_error(np.asarray(nominal["head_rotation_w"]), kin["head_rotation_w"]),
                  com_support_margin_m=margin,
                  frozen_target_glove_distance_m=np.linalg.norm(kin["glove_position_w"]-target, axis=-1))
    result["operational_guard"] = result["operational_pose"] & result["operational_support"]
    result["readiness_admission"] = (result["operational_finite"] & result["operational_upright"]
                                      & result["operational_hand_position"] & result["operational_support"])
    return result


def evaluate_functional_trace(time_s, measurements: dict, *, launch_time_s: float,
                              outward_complete_time_s: float, active_hand: int,
                              thresholds: FunctionalGuardThresholds | None = None) -> tuple[dict, dict]:
    """Evaluate explicit phase events without mutating recorded state or history."""
    cfg = thresholds or FunctionalGuardThresholds()
    t = np.asarray(time_s, float)
    if (t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
            or active_hand not in (0, 1)):
        raise ValueError("Finite strictly increasing sample times and active hand required")
    if not np.isclose(t[0], 0., atol=1.e-10) or not (0 <= launch_time_s < outward_complete_time_s <= t[-1]):
        raise ValueError("Explicit phase events must lie in the reset-relative trace horizon")
    launch_index = np.flatnonzero(np.isclose(t, launch_time_s, atol=1.e-9, rtol=0))
    outward_index = np.flatnonzero(np.isclose(t, outward_complete_time_s, atol=1.e-9, rtol=0))
    if len(launch_index) != 1 or len(outward_index) != 1:
        raise ValueError("Phase event times must name recorded samples")
    if not np.isclose(launch_time_s/.020, round(launch_time_s/.020), atol=1.e-8):
        raise ValueError("Actual launch must occur at a 20 ms actor boundary")
    if any(np.asarray(measurements[name]).shape != t.shape for name in ("operational_pose", "operational_support")):
        raise ValueError("Operational arrays must match the saved time axis")
    distance = np.asarray(measurements["frozen_target_glove_distance_m"])
    outward_mask = (t >= launch_time_s) & (t <= outward_complete_time_s)
    contact = distance[:, active_hand] <= .14
    machine, states = FunctionalRecoveryStateMachine(cfg), []
    rejected = False
    for i, time in enumerate(t):
        launch = i == launch_index[0]
        try:
            row = machine.observe(time, measurements["operational_pose"][i], measurements["operational_support"][i],
                                  launch=launch, outward_complete=bool(i == outward_index[0] and not rejected),
                                  contact=bool(contact[i] and outward_mask[i]),
                                  readiness_valid=bool(measurements["readiness_admission"][i]))
        except ValueError as error:
            if not launch or "readiness predicate" not in str(error):
                raise
            # Failed launch is a counted disposition. Re-observe this sample with
            # the already computed dwell, without inventing an actual launch.
            rejected = True
            machine.last_time_s, machine.previous_pose_valid = float(time), bool(measurements["operational_pose"][i])
            row = dict(phase="holding", ready=False, complete=False,
                       ready_dwell_s=machine.ready_dwell_s, recovery_dwell_s=0.,
                       readiness_time_s=machine.readiness_time_s, launch_time_s=None,
                       outward_completion_time_s=None, recovery_time_s=None, first_contact_time_s=None)
        states.append(row)
    physical_valid = bool(np.all(measurements["operational_finite"]) and np.all(measurements["operational_upright"]))
    hit_indices = np.flatnonzero(contact & outward_mask)
    any_contact_indices = np.flatnonzero(np.any(distance <= .14, axis=-1))
    inactive_contact_indices = np.flatnonzero(distance[:, 1-active_hand] <= .14)
    inactive_outward_indices = np.flatnonzero((distance[:, 1-active_hand] <= .14) & outward_mask)
    recovery = machine.recovery_time_s
    stability_observed = 0. if recovery is None else min(float(t[-1]-recovery), cfg.diagnostic_stability_s)
    stability_mask = np.zeros(len(t), bool) if recovery is None else (t >= recovery) & (t <= recovery+cfg.diagnostic_stability_s+1.e-10)
    stability_full = bool(recovery is not None and t[-1]+1.e-10 >= recovery+cfg.diagnostic_stability_s)
    stability_pass = (bool(np.all(measurements["operational_guard"][stability_mask])) if stability_full else None)
    diagnostics = {
        "hand_orientation_20deg": np.all(measurements["hand_orientation_error_rad"] <= cfg.diagnostic_glove_orientation_rad, axis=-1),
        "torso_orientation_0p15rad": measurements["torso_orientation_error_rad"] <= cfg.diagnostic_torso_orientation_rad,
        "root_height_90pct": measurements["root_height_fraction"] >= cfg.diagnostic_root_height_fraction,
        "com_margin_2cm": measurements["com_support_margin_m"] >= cfg.diagnostic_com_margin_m,
        "joint_speed_0p50rad_s": np.max(measurements["joint_speed_rad_s"], axis=-1) <= cfg.diagnostic_joint_speed_rad_s,
        "frozen_target_clearance_0p16m": np.all(distance >= cfg.diagnostic_clearance_m, axis=-1),
    }
    events = {key: np.asarray([row[key] for row in states]) for key in
              ("phase", "ready", "complete", "ready_dwell_s", "recovery_dwell_s")}
    events.update(time_s=t, outward_contact=contact & outward_mask, outward_window=outward_mask,
                  stability_observation=stability_mask,
                  **{"diagnostic_"+key: value for key, value in diagnostics.items()})
    summary = dict(predicate=asdict(cfg), initial_completion_false=not states[0]["complete"],
                   readiness_time_s=machine.readiness_time_s, launch_time_s=machine.launch_time_s,
                   requested_launch_time_s=float(launch_time_s), launch_rejected=bool(rejected),
                   readiness_definition="initial_admission_position_support_v1: finite, existing physical upright criterion, both measured feet support, both hands within 0.05m own qualified centers; instantaneous admission; quiet .20s completion dwell is separately required after outward",
                   outward_completion_time_s=machine.outward_completion_time_s,
                   requested_outward_completion_time_s=float(outward_complete_time_s),
                   recovery_time_s=recovery, recovered_after_outward=bool(machine.complete),
                   physical_finite_upright=physical_valid,
                   functional_complete=bool(machine.complete and physical_valid and not rejected),
                   contact_outcome="hit" if len(hit_indices) else "miss",
                   active_hand_outward_contact=bool(len(hit_indices)),
                   first_active_hand_outward_contact_s=float(t[hit_indices[0]]) if len(hit_indices) else None,
                   first_any_hand_contact_s=float(t[any_contact_indices[0]]) if len(any_contact_indices) else None,
                   first_inactive_hand_contact_s=float(t[inactive_contact_indices[0]]) if len(inactive_contact_indices) else None,
                   first_inactive_hand_outward_contact_s=float(t[inactive_outward_indices[0]]) if len(inactive_outward_indices) else None,
                   inactive_hand_first_outward_contact=bool(len(inactive_outward_indices) and
                       (not len(hit_indices) or inactive_outward_indices[0] < hit_indices[0])),
                   final_recovery_dwell_s=float(machine.recovery_dwell_s),
                   diagnostic_stability_observed_s=stability_observed,
                   diagnostic_stability_complete=stability_full, diagnostic_stability_pass=stability_pass,
                   diagnostic_final_checks={key: bool(value[-1]) for key, value in diagnostics.items()},
                   diagnostic_final_values={key: _json_measurement(np.asarray(measurements[key][-1]).tolist()) for key in
                       ("hand_position_error_m", "hand_relative_speed_m_s", "root_linear_speed_m_s",
                        "root_angular_speed_rad_s", "joint_return_error_rad", "joint_speed_rad_s",
                        "hand_orientation_error_rad", "torso_orientation_error_rad", "head_orientation_error_rad",
                        "com_support_margin_m", "root_height_fraction", "frozen_target_glove_distance_m")},
                   operational_failure_sample_counts={key: int(np.count_nonzero(~value)) for key, value in measurements.items()
                                                      if key.startswith("operational_")},
                   recovery_semantics="Post-outward uninterrupted functional dwell; contact is independent; recovered miss remains miss",
                   dwell_semantics="Pose predicates at both interval endpoints; both positive foot-floor force flags at right endpoint")
    return summary, events


def evaluate_functional_panel(model, arrays, inputs, origins, output, *,
                              predicate: FunctionalGuardThresholds | dict | None = None,
                              historical_summary: dict | None = None,
                              launch_time_s: float | None = None,
                              outward_complete_time_s: float | None = None) -> dict:
    """Write new per-trial observations using the original unpadded horizon."""
    from .wbc.provenance import sha256_file, writable_path, write_json
    from .wbc.release_baseline_evaluation import recorded_support

    record = predicate if isinstance(predicate, dict) else None
    cfg = (FunctionalGuardThresholds(**record.get("thresholds", record))
           if record is not None else predicate or FunctionalGuardThresholds())
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    frames, worlds = np.asarray(arrays["time"]).shape
    steps = int(np.ceil(float(inputs["clip"]["legacy_episode_duration_s"])/.005))
    n = steps + 1
    if frames < n:
        raise ValueError("Incomplete original horizon")
    origins = np.asarray(origins)
    if origins.shape != (worlds, 3) or not np.isfinite(origins).all():
        raise ValueError("One finite origin per world required")
    support = recorded_support(model, arrays, worlds)
    clip = inputs["clip"]
    schedule = {}
    if record is not None and "schedule" in record:
        schedules = record["schedule"]
        schedule = (schedules[clip["family"]] if isinstance(schedules, dict)
                    else next(row for row in schedules if row["family"] == clip["family"]))
    if launch_time_s is None:
        launch_time_s = schedule.get("launch_time_s")
    if outward_complete_time_s is None:
        outward_complete_time_s = schedule.get("outward_completion_time_s")
    if launch_time_s is None:
        raise ValueError("Supply reviewed actor-boundary launch from first non-guard reference interval, not rounded geometric annotation")
    launch = float(launch_time_s)
    outward = float(np.ceil(float(clip["outward_end_s"])/.005-1.e-10)*.005) if outward_complete_time_s is None else float(outward_complete_time_s)
    active = 0 if clip["family"] in ("jab", "left_hook") else 1
    trial_ids = np.asarray(inputs.get("trial_ids", np.arange(worlds)), int)
    if trial_ids.shape != (worlds,) or len(np.unique(trial_ids)) != worlds:
        raise ValueError("Unique preserved trial IDs required")
    historical = {int(row["trial_id"]): row for row in (historical_summary or {}).get("trials", [])}
    reports = []
    for world, trial_id in enumerate(trial_ids):
        guard = np.asarray(inputs["legacy"]["qpos"][0], float).copy()
        guard[:3] += origins[world]
        nominal = derive_guard_nominal(model, guard, family=clip["family"], prefix="robot/", provenance=dict(
            reference_sha256=inputs["input_sha256"]["reference"],
            qualification="Accepted Task09 original family reference; Task10B Phase0 accepted 64/64 under each solver profile",
            initial_pose_definition="Original qualified reference row zero, never evaluation trial mean"))
        measured = measure_functional_guard(model, arrays["qpos"][:n, world], arrays["qvel"][:n, world],
                                            support[:n, world], nominal, np.asarray(clip["target_w"])+origins[world], thresholds=cfg)
        report, events = evaluate_functional_trace(np.arange(n)*.005, measured, launch_time_s=launch,
            outward_complete_time_s=outward, active_hand=active, thresholds=cfg)
        report.update(trial_id=int(trial_id), family=clip["family"],
                      historical_exact_pose=historical.get(int(trial_id)),
                      resets_one=bool(np.all(arrays["reset_counts"][:n, world] == 1)))
        report["functional_complete"] &= report["resets_one"]
        prior = historical.get(int(trial_id))
        precision_names = ("full_cycle_active_glove_rms_2cm", "attack_active_glove_rms_2cm", "contact_timing_20ms")
        report["commanded_hit_evidence_available"] = prior is not None
        report["commanded_hit_success"] = bool(prior is not None
            and report["active_hand_outward_contact"]
            and prior.get("acceptance_checks", {}).get("active_hand_attack_contact", False)
            and all(prior.get("precision", {}).get(name, False) for name in precision_names))
        report["nominal_complete_cycle"] = bool(report["functional_complete"] and report["commanded_hit_success"])
        trace = destination/f"functional_{trial_id:02d}.npz"
        with trace.open("xb") as stream:
            np.savez_compressed(stream, **measured, **events)
        write_json(destination/f"nominal_{trial_id:02d}.json", nominal)
        report["measurement_sha256"] = sha256_file(trace)
        write_json(destination/f"trial_{trial_id:02d}.json", report)
        reports.append(report)
    summary = dict(schema="task10b_phase1_functional_panel_v1", family=clip["family"], predicate=asdict(cfg),
                   trial_count=worlds, trial_ids=trial_ids.tolist(),
                   functional_complete_count=sum(row["functional_complete"] for row in reports),
                   commanded_hit_success_count=sum(row["commanded_hit_success"] for row in reports),
                   nominal_complete_cycle_count=sum(row["nominal_complete_cycle"] for row in reports),
                   active_hand_outward_hit_count=sum(row["active_hand_outward_contact"] for row in reports),
                   launch_rejected_count=sum(row["launch_rejected"] for row in reports),
                   recovered_miss_count=sum(row["functional_complete"] and row["contact_outcome"] == "miss" for row in reports),
                   historical_physics_steps=steps, excluded_actor_padding_steps=frames-n,
                   launch_time_s=launch, outward_complete_time_s=outward,
                   reviewed_reference_schedule=schedule,
                   source_launch_annotation_s=float(clip["launch_time_s"]),
                   source_outward_end_annotation_s=float(clip["outward_end_s"]),
                   phase_semantics="Original reset-relative reference clock; original initial hold, live controller/history; outward end is conservative source-playback endpoint and can include source withdrawal",
                   source_inputs=inputs["input_sha256"], measurement_source_sha256=sha256_file(__file__),
                   weights_changed=False, training=False, moving_g1_qualification=False, trials=reports)
    write_json(destination/"summary.json", summary)
    return summary
