"""Offline shared-support candidate; no physical/controller state writes.

Recipe v3 matches all 48 initial joint positions/velocities, including bounded
D3 leg perturbations. Explicit quintic leg preparation reaches the accepted jab
support geometry before the retained source. Root motion remains excluded.
Initial reference acceleration is assumed zero, never measured physical qacc.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib

import mujoco
import numpy as np

from .phase2_phases import PhaseSchedule
from .wbc.references import tangent_velocities
from .phase2_shape_derivatives import project_derivatives

RECIPE_VERSION = "task10b_jab_d3_joint_preparation_shared_support_shape_v3"
DERIVATIVE_VERSION = "task10b_shared_analytic_connector_projected_source_c2_v4"
LOWER_JOINT_NAMES = tuple(side+part for side in ("Right", "Left") for part in
    ("Leg_x", "Leg_y", "Leg_z", "Shin_x", "Foot_x", "Foot_y", "Foot_z"))
DT = .005


@dataclass(frozen=True)
class SharedPreparationBounds:
    """New engineering bounds on the small leg preparation, not accepted gates."""
    maximum_lower_displacement_from_jab_rad: float = .05
    maximum_lower_speed_rad_s: float = .25
    maximum_lower_acceleration_rad_s2: float = 5.

    def __post_init__(self):
        values=asdict(self)
        if (not all(not isinstance(value,bool) and np.isfinite(value) and value>0 for value in values.values())
                or self.maximum_lower_displacement_from_jab_rad>.05
                or self.maximum_lower_speed_rad_s>.25
                or self.maximum_lower_acceleration_rad_s2>5.):
            raise ValueError("Explicit finite bounded leg-preparation recipe required")


class SharedReferenceRejection(ValueError):
    def __init__(self, reason, **details):
        self.disposition = dict(reason=reason, recipe=RECIPE_VERSION,
            physical_qualification=False, **details)
        super().__init__(reason)


def array_digest(*arrays):
    """Canonical float64 array identity binds a supplied support receipt to q/v."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def resolve_shared_joints(model):
    """Resolve names, never assume leg coordinates merely occupy a suffix."""
    if (model.nq, model.nv, model.njnt) != (55, 54, 49) or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Accepted free-root 55/54/49 articulation required")
    if (model.jnt_qposadr[0], model.jnt_dofadr[0]) != (0, 0):
        raise ValueError("Free-root tangent must occupy the accepted leading coordinates")
    if np.any(model.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE) or not np.all(model.jnt_limited[1:]):
        raise ValueError("All forty-eight scalar joints must be bounded hinges")
    try:
        lower = np.array([model.joint(name).id for name in LOWER_JOINT_NAMES], dtype=int)
    except KeyError as error:
        raise ValueError("All fourteen named leg/foot joints are required") from error
    if len(set(lower)) != 14:
        raise ValueError("Named lower-body mask is not one-to-one")
    upper = np.array([joint for joint in range(1, 49) if joint not in set(lower)], dtype=int)
    # Constant root/legs only preserve sole geometry if upper joints are not
    # ancestors of any sole. Verify that assumption against this actual model.
    for side in ("Right", "Left"):
        for corner in range(4):
            site = model.site(side+f"Foot_sole_{corner}")
            body = int(model.site_bodyid[site.id]); ancestors = set()
            while body:
                ancestors.add(body); body = int(model.body_parentid[body])
            if any(int(model.jnt_bodyid[joint]) in ancestors for joint in upper):
                raise ValueError("An upper-body joint changes shared sole geometry")
    return dict(lower_joint_ids=lower, upper_joint_ids=upper,
        lower_qpos=model.jnt_qposadr[lower].copy(), lower_dof=model.jnt_dofadr[lower].copy(),
        upper_qpos=model.jnt_qposadr[upper].copy(), upper_dof=model.jnt_dofadr[upper].copy())


def quintic_coefficients(q0, v0, a0, q1, v1, a1, duration_s):
    """Power coefficients in normalized time u=t/T, with analytic C2 endpoints."""
    values = [np.asarray(value, float) for value in (q0, v0, a0, q1, v1, a1)]
    if (not np.isfinite(duration_s) or duration_s <= 0
            or any(value.shape != values[0].shape or not np.isfinite(value).all() for value in values)):
        raise ValueError("Finite equally shaped endpoints and a positive duration required")
    q0, v0, a0, q1, v1, a1 = values
    c0, c1, c2 = q0, duration_s*v0, .5*duration_s**2*a0
    r0, r1, r2 = q1-c0-c1-c2, duration_s*v1-c1-2*c2, duration_s**2*a1-2*c2
    return np.stack((c0, c1, c2, 10*r0-4*r1+.5*r2, -15*r0+7*r1-r2, 6*r0-3*r1+.5*r2))


def evaluate_quintic(coefficients, times_s, duration_s):
    coefficients = np.asarray(coefficients, float); times = np.asarray(times_s, float)
    if (coefficients.ndim != 2 or coefficients.shape[0] != 6 or times.ndim != 1
            or not np.isfinite(coefficients).all() or not np.isfinite(times).all()
            or not np.isfinite(duration_s) or duration_s <= 0
            or np.any(times < 0) or np.any(times > duration_s+1.e-12)):
        raise ValueError("Finite quintic coefficients and in-interval times required")
    u = times/duration_s
    result = []
    for order in range(3):
        c = coefficients.copy()
        for _ in range(order):
            c = np.arange(1, len(c))[:, None]*c[1:]/duration_s
        result.append((u[:, None]**np.arange(len(c)))@c)
    return tuple(result)


def polynomial_joint_bounds(coefficients):
    """Continuous quintic extrema, including between-sample overshoot."""
    coefficients = np.asarray(coefficients, float)
    if coefficients.ndim != 2 or coefficients.shape[0] != 6 or not np.isfinite(coefficients).all():
        raise ValueError("Finite six-coefficient vector per joint required")
    lower, upper = [], []
    for column in coefficients.T:
        derivative = np.arange(1, 6)*column[1:]
        roots = np.polynomial.polynomial.polyroots(derivative) if np.any(derivative) else []
        points = [0., 1.]+[float(root.real) for root in roots if abs(root.imag) < 1.e-8 and 0 < root.real < 1]
        values = np.polynomial.polynomial.polyval(points, column)
        lower.append(float(min(values))); upper.append(float(max(values)))
    return np.array(lower), np.array(upper)


def polynomial_derivative_bounds(coefficients,duration_s,order):
    """Continuous analytic derivative extrema in normalized time, not samples."""
    coefficients=np.asarray(coefficients,float)
    if (coefficients.ndim!=2 or coefficients.shape[0]!=6 or not np.isfinite(coefficients).all()
            or not np.isfinite(duration_s) or duration_s<=0 or order not in (1,2)):
        raise ValueError("Finite quintic, positive duration and derivative order one/two required")
    values=coefficients.copy()
    for _ in range(order):values=np.arange(1,len(values))[:,None]*values[1:]/duration_s
    lower=[];upper=[]
    for column in values.T:
        derivative=np.arange(1,len(column))*column[1:]
        roots=np.polynomial.polynomial.polyroots(derivative) if np.any(derivative) else []
        points=[0.,1.]+[float(r.real) for r in roots if abs(r.imag)<1.e-8 and 0<r.real<1]
        evaluated=np.polynomial.polynomial.polyval(points,column)
        lower.append(float(min(evaluated)));upper.append(float(max(evaluated)))
    return np.array(lower),np.array(upper)


def _require_limits(coefficients, limits, *, phase, segment=None):
    low, high = polynomial_joint_bounds(coefficients)
    excursion = np.maximum(np.maximum(limits[:, 0]-low, high-limits[:, 1]), 0.)
    if np.max(excursion, initial=0) > 1.e-10:
        raise SharedReferenceRejection("joint_limit_excursion", phase=phase,
            segment=segment, maximum_excursion_rad=float(excursion.max()))
    return low, high


def _glove_positions(model, poses):
    """Pure forward kinematics; no integration, force allocation or support proof."""
    data = mujoco.MjData(model)
    sites = [model.site(side+"Hand_center").id for side in ("Left", "Right")]
    gloves = np.empty((len(poses), 2, 3))
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        gloves[frame] = data.site_xpos[sites]
    return gloves


def _sole_positions(model,poses):
    data=mujoco.MjData(model)
    sites=[[model.site(side+f"Foot_sole_{corner}").id for corner in range(4)] for side in ("Right","Left")]
    positions=np.empty((len(poses),2,4,3))
    for frame,pose in enumerate(poses):
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
        positions[frame]=data.site_xpos[sites]
    return positions


def _validate_start(jab, q0, v0, mask, support):
    if not np.array_equal(q0[:7], jab[:7]):
        raise SharedReferenceRejection("protected_root_pose_differs_from_jab")
    if np.any(v0[:6] != 0.):
        raise SharedReferenceRejection("nonzero_protected_root_velocity")
    for region in ("upper","lower"):
        if np.max(np.abs(q0[mask[region+"_qpos"]]-jab[mask[region+"_qpos"]]), initial=0) > .015+1.e-12:
            raise SharedReferenceRejection(region+"_start_pose_outside_local_0015rad_region")
        if np.max(np.abs(v0[mask[region+"_dof"]]), initial=0) > .06+1.e-12:
            raise SharedReferenceRejection(region+"_start_velocity_outside_local_006_region")
    if not isinstance(support, dict) or not isinstance(support.get("evidence_id"), str) or not support["evidence_id"]:
        raise SharedReferenceRejection("missing_explicit_support_evidence")
    if support.get("state_qpos_qvel_sha256") != array_digest(q0, v0):
        raise SharedReferenceRejection("support_evidence_does_not_bind_start")
    if support.get("evidence_kind")=="provisional_geometry_construction_only":
        if (support.get("native_support_established") is not False
                or support.get("native_execution_permitted") is not False
                or any(key in support for key in ("supported_feet","finite_upright","disallowed_body_floor_contact",
                    "state_time_s","support_interval_end_s"))):
            raise SharedReferenceRejection("provisional_geometry_must_not_fabricate_physical_support")
        return
    feet = support.get("supported_feet")
    if (not isinstance(feet, (list, tuple)) or len(feet) != 2
            or any(type(flag) is not bool or not flag for flag in feet)
            or support.get("finite_upright") is not True
            or support.get("disallowed_body_floor_contact") is not False):
        raise SharedReferenceRejection("unsupported_or_unqualified_start")
    clocks = [support.get(name) for name in ("state_time_s", "support_interval_end_s")]
    if (any(isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0 for value in clocks)
            or abs(clocks[0]-clocks[1]) > 1.e-6):
        raise SharedReferenceRejection("stale_or_invalid_support_evidence")


def compose_shared_reference(model, source, accepted_jab_qpos, start_qpos, start_qvel,
                             *, source_schedule: PhaseSchedule, preparation_s, support_evidence,
                             preparation_bounds=SharedPreparationBounds()):
    """Return in-memory candidate/receipt, or a structured explicit rejection.

    The connector is analytic. Source intervals use a C2 quintic curve through
    unchanged positions, with shared nodal velocity/acceleration projected from
    tangent finite-difference estimates into local Bezier endpoint boxes. The
    resulting derivatives are analytic curve derivatives. Actual physical
    acceleration and readiness still require uninterrupted execution.
    """
    mask = resolve_shared_joints(model)
    t = np.asarray(source["time_s"], float); original = np.asarray(source["qpos"], float)
    jab, q0, v0 = (np.asarray(value, float) for value in (accepted_jab_qpos, start_qpos, start_qvel))
    target = np.asarray(source["head_center"], float)
    if (t.ndim != 1 or len(t) < 4 or t[0] != 0 or not np.allclose(np.diff(t), DT, atol=1.e-12, rtol=0)
            or original.shape != (len(t), 55) or jab.shape != (55,) or q0.shape != (55,)
            or v0.shape != (54,) or target.shape != (3,)
            or any(not np.isfinite(value).all() for value in (t, original, jab, q0, v0, target))):
        raise ValueError("Finite full source on exact 5ms clock and complete human qpos/qvel required")
    if (not isinstance(source_schedule, PhaseSchedule) or source_schedule.reference_end_tick != len(t)-1
            or source_schedule.preparation_complete_tick != 0):
        raise ValueError("Explicit complete original-family phase schedule with no existing connector required")
    if source.get("active_hand_index") not in (0, 1):
        raise ValueError("Explicit active hand required")
    if not all(np.allclose(np.linalg.norm(value[..., 3:7], axis=-1), 1., atol=1.e-7, rtol=0)
               for value in (original, jab, q0)):
        raise ValueError("Normalized wxyz root quaternions required; no silent normalization")
    if not np.isfinite(preparation_s) or not 0 < preparation_s <= 1.:
        raise ValueError("Recipe v2 preparation must be explicitly bounded within (0,1] seconds")
    if not isinstance(preparation_bounds,SharedPreparationBounds):
        raise ValueError("Explicit shared leg-preparation bounds required")
    ticks = int(round(preparation_s/DT))
    if abs(ticks*DT-preparation_s) > 1.e-12 or ticks % 4:
        raise ValueError("Preparation must occupy whole 20ms actor intervals without hidden padding")
    _validate_start(jab, q0, v0, mask, support_evidence)
    qa, va = mask["upper_qpos"], mask["upper_dof"]
    lower_q, lower_v = mask["lower_qpos"], mask["lower_dof"]
    lower_limits = model.jnt_range[mask["lower_joint_ids"]]
    if np.any(jab[lower_q] < lower_limits[:, 0]) or np.any(jab[lower_q] > lower_limits[:, 1]):
        raise SharedReferenceRejection("accepted_jab_lower_pose_outside_model_limits")
    # Retarget reference geometry only. This is never written into physical data.
    q = original.copy(); q[:, :7] = jab[:7]; q[:, lower_q] = jab[lower_q]
    v = tangent_velocities(model, t, q)
    hold_windows = ((0, source_schedule.launch_tick),
                    (source_schedule.guard_hold_tick, source_schedule.reference_end_tick))
    for first, last in hold_windows:
        if not np.array_equal(q[first:last+1, qa], np.tile(q[first, qa], (last-first+1, 1))):
            raise SharedReferenceRejection("declared_original_hold_not_constant", first_tick=first, last_tick=last)
        # Central differences leak neighboring movement into hold endpoints.
        # The declared constant source holds instead have exact zero analytic
        # derivatives on both sides, preserving C2 source/connector joins.
        v[first:last+1] = 0.
    # Source acceleration differentiates the final knot velocities, including
    # the declared zero-velocity boundaries used by adjacent moving intervals.
    a = np.gradient(v, t, axis=0, edge_order=2)
    for first, last in hold_windows:
        a[first:last+1] = 0.
    if np.any(v[:, :6]) or np.any(v[:, lower_v]) or np.any(a[:, :6]) or np.any(a[:, lower_v]):
        raise AssertionError("Constant protected coordinates acquired tangent derivatives")
    limits = model.jnt_range[mask["upper_joint_ids"]]
    source_projection=project_derivatives(q[:,qa],v[:,va],a[:,va],limits,
        joint_names=[model.joint(int(joint)).name for joint in mask["upper_joint_ids"]])
    v[:,va]=source_projection.velocity
    a[:,va]=source_projection.acceleration
    # Every source knot remains unchanged. Its between-knot interpretation is
    # explicitly C2 Hermite using projected analytic v/a, not a C2 claim about
    # the historical source interpolator or physically measured acceleration.
    source_coefficients = np.stack([quintic_coefficients(q[i, qa], v[i, va], a[i, va],
        q[i+1, qa], v[i+1, va], a[i+1, va], DT) for i in range(len(t)-1)])
    for index, coefficients in enumerate(source_coefficients):
        _require_limits(coefficients, limits, phase="retargeted_source", segment=index)
    connector_joint_ids=np.arange(1,49)
    connector_qa=model.jnt_qposadr[connector_joint_ids];connector_va=model.jnt_dofadr[connector_joint_ids]
    initial_reference_acceleration = np.zeros(48)
    coefficients = quintic_coefficients(q0[connector_qa], v0[connector_va], initial_reference_acceleration,
        q[0, connector_qa], v[0, connector_va], a[0, connector_va], preparation_s)
    connector_bounds = _require_limits(coefficients, model.jnt_range[connector_joint_ids], phase="preparation")
    speed_bounds=polynomial_derivative_bounds(coefficients,preparation_s,1)
    acceleration_bounds=polynomial_derivative_bounds(coefficients,preparation_s,2)
    lower_columns=np.array([int(np.flatnonzero(connector_joint_ids==joint)[0]) for joint in mask["lower_joint_ids"]])
    lower_displacement=float(np.max(np.maximum(abs(connector_bounds[0][lower_columns]-jab[lower_q]),
        abs(connector_bounds[1][lower_columns]-jab[lower_q])),initial=0))
    lower_speed=float(np.max(np.maximum(abs(speed_bounds[0][lower_columns]),abs(speed_bounds[1][lower_columns])),initial=0))
    lower_acceleration=float(np.max(np.maximum(abs(acceleration_bounds[0][lower_columns]),abs(acceleration_bounds[1][lower_columns])),initial=0))
    measured=dict(maximum_lower_displacement_from_jab_rad=lower_displacement,
        maximum_lower_speed_rad_s=lower_speed,maximum_lower_acceleration_rad_s2=lower_acceleration)
    if any(measured[key]>value+1.e-10 for key,value in asdict(preparation_bounds).items()):
        raise SharedReferenceRejection("lower_preparation_motion_outside_recipe",measured=measured,bounds=asdict(preparation_bounds))
    prep_time = np.arange(ticks+1)*DT
    pq, pv, pa = evaluate_quintic(coefficients, prep_time, preparation_s)
    # Preserve exact declared endpoint numbers rather than floating cancellation
    # in polynomial evaluation; these are arrays, never a physical state swap.
    pq[0], pv[0], pa[0] = q0[connector_qa], v0[connector_va], initial_reference_acceleration
    pq[-1], pv[-1], pa[-1] = q[0, connector_qa], v[0, connector_va], a[0, connector_va]
    cq = np.tile(jab, (ticks+1, 1)); cv = np.zeros((ticks+1, 54)); ca = cv.copy()
    cq[:, connector_qa], cv[:, connector_va], ca[:, connector_va] = pq, pv, pa
    full_q, full_v, full_a = (np.concatenate((prefix[:-1], suffix)) for prefix, suffix in ((cq, q), (cv, v), (ca, a)))
    full_t = np.arange(len(full_q))*DT
    schedule = PhaseSchedule(ticks, ticks+source_schedule.launch_tick,
        ticks+source_schedule.outward_complete_tick, ticks+source_schedule.guard_hold_tick,
        ticks+source_schedule.reference_end_tick)
    fresh_v = tangent_velocities(model, full_t, full_q)
    fresh_a = np.gradient(fresh_v, full_t, axis=0, edge_order=2)
    original_gloves = _glove_positions(model, original)
    full_gloves = _glove_positions(model, full_q)
    sole_positions=_sole_positions(model,cq)
    sole_movement=np.linalg.norm(sole_positions-sole_positions[0],axis=-1)
    sole_speed=np.linalg.norm(np.diff(sole_positions,axis=0)/DT,axis=-1)
    lower_preparation_moves=bool(np.any(q0[lower_q]!=jab[lower_q]) or np.any(v0[lower_v]))
    candidate = dict(time_s=full_t, t=full_t.copy(), qpos=full_q, qvel=full_v, qacc=full_a,
        derivative_version=DERIVATIVE_VERSION,
        head_center=target.copy(), active_hand_index=int(source["active_hand_index"]),
        glove_pos=full_gloves, source_original_glove_pos=original_gloves,
        source_time_s=t.copy(), source_start_tick=ticks, schedule=schedule,
        launch_time=schedule.launch_tick*DT, return_start=schedule.outward_complete_tick*DT,
        guard_hold_start=schedule.guard_hold_tick*DT,
        connector_coefficients=coefficients,connector_qpos_addresses=connector_qa.copy(),
        connector_dof_addresses=connector_va.copy(),source_segment_coefficients=source_coefficients,
        source_segment_qpos_addresses=qa.copy(),source_segment_dof_addresses=va.copy())
    receipt = dict(schema="task10b_phase2_shared_reference_candidate_shape_v3", recipe=RECIPE_VERSION,
        derivative_version=DERIVATIVE_VERSION, source_derivative_projection=source_projection.receipt, schedule=asdict(schedule),
        source_schedule=asdict(source_schedule), preparation_s=preparation_s,
        phase_denominator_s=schedule.phase_denominator_s, source_start_tick=ticks,
        lower_joint_names=list(LOWER_JOINT_NAMES),
        lower_qpos_addresses=lower_q.tolist(), lower_dof_addresses=lower_v.tolist(),
        upper_qpos_addresses=qa.tolist(), upper_dof_addresses=va.tolist(),
        connector_qpos_addresses=connector_qa.tolist(),connector_dof_addresses=connector_va.tolist(),
        source_qpos_sha256=array_digest(original), original_glove_fk_sha256=array_digest(original_gloves),
        accepted_jab_qpos_sha256=array_digest(jab), start_qpos_qvel_sha256=array_digest(q0, v0),
        support_evidence=deepcopy(support_evidence),
        native_support_established=support_evidence.get("evidence_kind")!="provisional_geometry_construction_only",
        eligibility="Accepted-jab root pose with exactly zero initial root tangent velocity; all48 joint-position deltas <=.015rad and joint-velocity deltas <=.06rad/s. Explicit bounded leg preparation is admitted as a construction candidate. No physical four-start reachability qualification.",
        reference_vs_physical_start="Supplied qpos/qvel are treated as the caller's measured/saved physical start and matched exactly in the reference. All14 leg/foot perturbations and velocities are retained at the connector start and explicitly moved toward nominal jab support by the connector. Root perturbation or root velocity is rejected. No actual physical state, foot pose, previous action or controller history is replaced.",
        source_clock_and_all_hold_knots_preserved=True, source_upper_joint_knots_exactly_preserved=True,
        source_upper_path_between_knots="Explicit quintic Hermite interpolation through unchanged source upper-joint knots with recomputed v/a; historical interpolation is not claimed identical.",
        existing_linear_reference_library_compatible=False,
        runtime_sampling_required="Any future between-node queries must evaluate the stored connector/source polynomial coefficients and derivatives, or explicitly rebuild and requalify another interpolation contract. Audited exact-grid indexing is a separate minimal option below. The existing linear ReferenceLibrary must not consume these arrays unchanged while claiming continuous C2/derivative consistency.",
        future_indexed_grid_sampler_option="A minimal future native adapter may instead audit exact integer 5ms query ticks and directly index qpos/qvel/qacc, including future offsets +20/+50 ticks. These stored analytic curve samples need no between-node evaluation or executed continuous-C2 claim. Preserve historical interpolation for original controls. This option changes reference sampling only, not actor architecture; no adapter or parity qualification is implemented here.",
        root_and_lower_reference_geometry="Root is constant accepted-jab throughout. All14 leg/foot joints follow explicit analytic preparation from their declared perturbed start to nominal jab geometry, then remain constant through source/outward/recovery/holds. No root translation reconciles soles; no fixed-sole claim during preparation.",
        original_world_glove_path_preserved=False,
        target_geometry_status="Original requested target retained unresolved. Subsequent explicit active-arm retargeting must use retained original active-glove FK, then remeasure first contact/TTC.",
        target_w=target.tolist(), source_original_glove_clock="source_time_s; source_start_tick offsets it into the full reference; no original FK is invented for preparation",
        connector_derivatives="Analytic quintic q/v/a on all48hinges, using explicit connector_qpos_addresses/connector_dof_addresses. Exactly matches supplied initial joint q/v and retargeted source initial q/v/a. Initial reference acceleration is zero by engineering assumption, not measured physical qacc. Root qpos is unchanged and root tangent qvel/qacc remain zero.",
        source_derivatives="Start from fresh MuJoCo tangent/finite-difference estimates and exact declared holds, then project only scalar-hinge nodal v/a to local quintic Bezier endpoint boxes. Original q knots are unchanged. Shared analytic v/a makes source intervals C2 and bounds each interval by its unchanged endpoints; acceleration is the analytic curve derivative rather than a new finite difference of projected velocities. Exact repeated joint intervals have zero v/a at both ends. Free-root convention is unchanged, with zero root tangent here. Full local change inventory is retained.",
        constant_source_hold_windows_ticks=[list(window) for window in hold_windows],
        initial_physical_qacc_measured=False, initial_physical_qacc_continuity_qualified=False,
        finite_difference_crosscheck=dict(velocity_max_abs=float(np.max(abs(full_v-fresh_v))),
            acceleration_max_abs=float(np.max(abs(full_a-fresh_a))),
            use="Diagnostic truncation differences only; analytic connector must not be replaced with centered differences that alter measured start velocity.",
            strict_finite_difference_only_audit_applicable=False),
        connector_joint_min=connector_bounds[0].tolist(), connector_joint_max=connector_bounds[1].tolist(),
        preparation_bounds=asdict(preparation_bounds),preparation_motion=measured,
        lower_preparation_moves=lower_preparation_moves,
        sole_preparation_diagnostics=dict(sampling_dt_s=DT,
            maximum_sampled_displacement_from_initial_m=float(np.max(sole_movement,initial=0)),
            maximum_sampled_segment_speed_m_s=float(np.max(sole_speed,initial=0)),
            start_positions_w=sole_positions[0].tolist(),end_positions_w=sole_positions[-1].tolist(),
            minimum_sampled_height_w_m=float(np.min(sole_positions[...,2])),
            maximum_sampled_height_w_m=float(np.max(sole_positions[...,2])),
            scope="Pure5msforward-kinematics diagnostics; no continuous sole extrema, friction/support/dynamics proof or fixed-sole assumption. Any prep movement remains visible for native review."),
        continuous_polynomial_joint_limits_screened=True,
        launch_inferred_from_motion=False, state_or_history_modified=False,
        support_receipt_scope="The supplied support receipt binds qpos/qvel only. Full native/controller/action-history identity and uninterrupted execution remain the separate phase2_phases SavedExecutionState/launch contract.",
        sole_geometry_preserved_by_verified_kinematic_ancestry=not lower_preparation_moves,
        source_sole_geometry_constant_after_preparation_by_verified_kinematic_ancestry=True,
        fixed_sole_during_preparation_claimed=False,
        dynamics_or_support_from_geometry_qualified=False, physics_integrated=False,
        feedforward_recomputed=False, physical_qualification=False,
        required_next_steps=["versioned_target_retargeting", "derivative_contract_audit", "original_joint_only_feedforward_recompute",
            "self_collision_and_floor_geometry", "first_contact_and_requested_TTC", "declared_support_with_unilateral_friction_forces",
            "six_unactuated_base_equations", "effort_caps", "fine_tuning", "uninterrupted_native_execution_and_history_audit"])
    return candidate, receipt
