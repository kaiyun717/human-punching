"""Project post-IK scalar source derivatives; keep the connector and q knots."""
from dataclasses import asdict
import numpy as np

from .phase2_phases import PhaseSchedule
from .phase2_shape_derivatives import project_derivatives,VERSION as PROJECTION_VERSION
from .phase2_shared_reference_shape_v3 import _require_limits

VERSION="task10b_shared_target_analytic_connector_projected_source_c2_v3"


def refine_shared_target_derivatives(model,candidate,metadata):
    schedule=candidate.get("schedule")
    if not isinstance(schedule,PhaseSchedule) or metadata.get("schedule")!=asdict(schedule):
        raise ValueError("Exact explicit shared phase schedule required")
    if (model.nq,model.nv,model.nu)!=(55,54,48):raise ValueError("Accepted SOMA articulation required")
    # The accepted model has one leading free joint and forty-eight scalar
    # hinges. Use its named addresses, not a reinterpretation of a root tangent.
    if (int(model.jnt_type[0])!=0 or np.any(np.asarray(model.jnt_type)[1:]!=3)
            or np.any(np.asarray(model.jnt_limited)[1:]!=1)):
        raise ValueError("Only the accepted bounded scalar-hinge articulation is supported")
    qa=np.asarray(model.jnt_qposadr)[1:];va=np.asarray(model.jnt_dofadr)[1:]
    if (not np.array_equal(qa,np.arange(7,55)) or not np.array_equal(va,np.arange(6,54))):
        raise ValueError("Accepted named scalar-hinge qpos/dof addresses required")
    q=np.asarray(candidate["qpos"]);v=np.asarray(candidate["qvel"]);a=np.asarray(candidate["qacc"])
    t=np.asarray(candidate.get("time_s"));clock=np.asarray(candidate.get("t"))
    prep=schedule.preparation_complete_tick
    if (q.shape!=(schedule.reference_end_tick+1,55) or v.shape!=(len(q),54) or a.shape!=v.shape
            or not 0<prep<=200 or prep%4 or any(x.dtype!=np.float64 for x in (q,v,a))
            or any(not np.isfinite(x).all() for x in (q,v,a))
            or t.shape!=(len(q),) or not np.array_equal(clock,t)
            or not np.allclose(t,np.arange(len(q))*.005,rtol=0,atol=1.e-12)
            or candidate.get("source_start_tick")!=prep
            or not np.array_equal(q[:,:7],np.repeat(q[:1,:7],len(q),axis=0))
            or np.any(v[:,:6]) or np.any(a[:,:6])):
        raise ValueError("Complete fixed-root exact-grid shared candidate with unchanged free-root tangent required")
    names=[model.joint(int(i)).name for i in range(1,49)]
    projected=project_derivatives(q[prep:,qa],v[prep:,va],a[prep:,va],model.jnt_range[1:],joint_names=names)
    # Retain the independently computed polynomial-extrema screen, with the
    # original1e-10rad tolerance. A convex-hull certificate does not replace it.
    for interval,coefficients in enumerate(projected.coefficients):
        _require_limits(coefficients,model.jnt_range[1:],phase="post_ik_projected_source",segment=interval)
    out={k:value.copy() if isinstance(value,np.ndarray) else value for k,value in candidate.items()}
    if (not np.array_equal(projected.velocity[0],v[prep,va])
            or not np.array_equal(projected.acceleration[0],a[prep,va])):
        raise ValueError("Post-IK projection may not change the analytic connector endpoint")
    out["qvel"][prep+1:,va]=projected.velocity[1:];out["qacc"][prep+1:,va]=projected.acceleration[1:]
    if (not np.array_equal(out["qpos"],q) or not np.array_equal(out["qvel"][:prep+1],v[:prep+1])
            or not np.array_equal(out["qacc"][:prep+1],a[:prep+1])):
        raise ValueError("Post-IK projection may not change the analytic connector or its shared endpoint")
    out["source_segment_coefficients"]=projected.coefficients
    out["source_segment_qpos_addresses"]=qa.copy();out["source_segment_dof_addresses"]=va.copy()
    out["derivative_version"]=VERSION
    info={**metadata,"derivative_version":VERSION,"source_segment_coefficients_omitted":False,
        "projection_version":PROJECTION_VERSION,"source_derivative_projection":projected.receipt,
        "source_segment_start_tick":prep,"connector_derivatives_preserved_through_tick":prep,
        "derivative_semantics":"Unchanged post-IK q knots and connector. Projected scalar-hinge nodal v/a defines a C2 quintic source whose Bezier controls stay inside each original endpoint box. Root tangents remain zero. Recompute original joint-only feedforward after this operation.",
        "feedforward_recomputed":False,"continuous_source_endpoint_bounds_certified":True,
        "independent_polynomial_extrema_screen_passed":True,
        "polynomial_extrema_screen_intervals":len(projected.coefficients),
        "polynomial_extrema_screen_tolerance_rad":1.e-10,"physical_qualification":False}
    return out,info
