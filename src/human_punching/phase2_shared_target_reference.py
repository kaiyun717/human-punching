"""Explicit original-glove-path correction after shared-support conversion.

The shared source's world glove path changes when its lower body changes. This
separate recipe blends toward the original source's actual FK corridor plus the
requested target displacement. It preserves the connector, shared guard endpoints
and explicit phase schedule. Nothing here generates files or qualifies execution.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from .phase2_domain import fingerprint
from .phase2_phases import PhaseSchedule
from .phase2_target_reference import (RetargetRejection,RetargetSettings,
    _kinematics,correction_envelope,resolve_joint_mask)
from .wbc.references import tangent_velocities


@dataclass(frozen=True)
class SharedTargetRecipe:
    upper_body: RetargetSettings
    version: str = "task10b_shared_original_glove_corridor_v1"
    maximum_combined_path_displacement_m: float = .18

    def __post_init__(self):
        if (not isinstance(self.upper_body,RetargetSettings)
                or self.version != "task10b_shared_original_glove_corridor_v1"
                or not np.isfinite(self.maximum_combined_path_displacement_m)
                or not 0 < self.maximum_combined_path_displacement_m <= .20):
            raise ValueError("Explicit bounded original-glove-corridor recipe required")


def _digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _source_arrays(model,source):
    t=np.asarray(source["time_s"],float);q=np.asarray(source["qpos"],float)
    if (t.ndim!=1 or len(t)<3 or q.shape!=(len(t),model.nq) or t[0]!=0
            or not np.isfinite(t).all() or not np.isfinite(q).all()
            or not np.allclose(np.diff(t),.005,atol=1e-12,rtol=0)):
        raise ValueError("Complete finite source on the exact 5ms clock required")
    return t,q


def _point_paths(model,q,active,inactive,checkpoint=None):
    data=mujoco.MjData(model);a=[];b=[];r=[]
    for frame,pose in enumerate(q):
        if checkpoint is not None and frame%16==0:checkpoint()
        _kinematics(model,data,pose)
        a.append(data.site_xpos[active].copy());b.append(data.site_xpos[inactive].copy())
        r.append(data.site_xmat[inactive].reshape(3,3).copy())
    return np.array(a),np.array(b),np.array(r)


def desired_shared_active_path(shared_fk,original_fk,*,target_delta,preparation_ticks,envelope):
    """Pure explicit blend: shared FK + e*(original FK + delta - shared FK)."""
    shared=np.asarray(shared_fk,float);original=np.asarray(original_fk,float)
    delta=np.asarray(target_delta,float);e=np.asarray(envelope,float)
    if (type(preparation_ticks) is not int or preparation_ticks<0
            or shared.shape!=(len(e),3) or original.shape!=(len(e)-preparation_ticks,3)
            or delta.shape!=(3,) or not all(np.isfinite(x).all() for x in (shared,original,delta,e))
            or np.any((e<0)|(e>1)) or np.any(e[:preparation_ticks+1])):
        raise ValueError("Aligned source corridor and zero connector envelope required")
    desired=shared.copy()
    desired[preparation_ticks:]+=e[preparation_ticks:,None]*(original+delta-shared[preparation_ticks:])
    return desired


def compose_shared_target(model,shared,original_source,original_active_glove_path,target_w,*,
                          source_schedule:PhaseSchedule,recipe:SharedTargetRecipe,checkpoint=None):
    """In-memory bounded composition, preserving original connector derivatives.

    ``original_active_glove_path`` is bound to independent FK of original_source,
    so a mislabeled path or target cannot silently redefine the command. Shared
    source qvel/qacc inside the connector use its declared analytic polynomial;
    changed source-segment tangent derivatives are freshly recomputed.
    """
    settings=recipe.upper_body;schedule=shared.get("schedule")
    if not isinstance(schedule,PhaseSchedule) or not isinstance(source_schedule,PhaseSchedule):
        raise ValueError("Explicit shared and original source phase schedules required")
    if source_schedule.preparation_complete_tick != 0:
        raise ValueError("The original corridor source must not contain another connector")
    prep=schedule.preparation_complete_tick
    if prep<1 or prep%4:
        raise ValueError("Shared connector must end on an explicit nonzero actor boundary")
    if shared.get("source_start_tick",prep)!=prep:
        raise ValueError("Source mapping differs from preparation endpoint")
    for name in ("launch_tick","outward_complete_tick","guard_hold_tick","reference_end_tick"):
        if getattr(schedule,name) != prep+getattr(source_schedule,name):
            raise ValueError("Shared schedule must retain every source offset after preparation")
    t,q=_source_arrays(model,shared);original_t,original_q=_source_arrays(model,original_source)
    if (len(t)-1 != schedule.reference_end_tick or len(original_t)-1 != source_schedule.reference_end_tick
            or len(t)-prep != len(original_t)
            or not np.allclose(t[prep:]-t[prep],original_t,atol=1e-12,rtol=0)):
        raise ValueError("Original source and shared source clocks do not align")
    v=np.asarray(shared["qvel"],float);acc=np.asarray(shared["qacc"],float)
    if v.shape!=(len(t),model.nv) or acc.shape!=v.shape or not np.isfinite(v).all() or not np.isfinite(acc).all():
        raise ValueError("Complete shared connector velocity/acceleration required")
    if "derivative_version" not in shared or not isinstance(shared["derivative_version"],str):
        raise ValueError("Shared input must explicitly identify its connector derivative version")
    holds=((prep,schedule.launch_tick),(schedule.guard_hold_tick,schedule.reference_end_tick))
    for left,right in holds:
        if (not np.array_equal(q[left:right+1],np.repeat(q[left:left+1],right-left+1,axis=0))
                or np.any(v[left:right+1]) or np.any(acc[left:right+1])):
            raise ValueError("Shared source holds require exact constant knots and zero velocity/acceleration")
    active_hand=0 if settings.active_side=="Left" else 1
    if any(s.get("active_hand_index")!=active_hand for s in (shared,original_source)):
        raise ValueError("Shared source, original source and upper-body active hand must agree")
    target=np.asarray(target_w,float);original_target=np.asarray(original_source["head_center"],float)
    if (target.shape!=(3,) or original_target.shape!=(3,) or not np.isfinite(target).all()
            or not np.isfinite(original_target).all() or not np.array_equal(shared["head_center"],original_target)):
        raise ValueError("Keep the original target identity; no shared-source head_center relabeling")
    target_delta=target-original_target
    if np.linalg.norm(target_delta)>settings.maximum_target_delta_m:
        raise RetargetRejection("target_offset_outside_recipe",target_delta_m=target_delta.tolist())
    mask=resolve_joint_mask(model,settings);ids=mask["joint_ids"];qa=mask["qpos_addresses"];va=mask["dof_addresses"]
    if not np.array_equal(q[prep:,qa],original_q[:,qa]):
        raise ValueError("Initial composition requires unmodified copied source upper-joint knots")
    active,inactive=mask["active_site"],mask["inactive_site"]
    shared_fk,inactive_fk,inactive_rot=_point_paths(model,q,active,inactive,checkpoint)
    actual_original_fk,_,_=_point_paths(model,original_q,active,inactive,checkpoint)
    original_fk=np.asarray(original_active_glove_path,float)
    if original_fk.shape!=actual_original_fk.shape or not np.allclose(original_fk,actual_original_fk,atol=1e-10,rtol=0):
        raise ValueError("Original active-glove path does not match its source reference FK")
    e=correction_envelope(t,schedule.launch_tick*.005,schedule.outward_complete_tick*.005,settings.envelope_blend_s)
    if np.any((e < -1e-12)|(e > 1+1e-12)):
        raise ValueError("Correction envelope escaped its mathematical unit interval")
    # The quintic polynomial can exceed one by a few ulps at the plateau join.
    # Restore only its mathematical bounds; requested targets remain unchanged.
    e=np.clip(e,0.,1.)
    desired=desired_shared_active_path(shared_fk,original_fk,target_delta=target_delta,preparation_ticks=prep,envelope=e)
    unscaled=original_fk+target_delta-shared_fk[prep:]
    moving=e[prep:]>0
    if np.any(np.linalg.norm(unscaled[moving],axis=1)>recipe.maximum_combined_path_displacement_m):
        raise RetargetRejection("shared_geometry_compensation_outside_recipe",
            maximum_combined_path_displacement_m=float(np.linalg.norm(unscaled[moving],axis=1).max()))
    result=q.copy();data=mujoco.MjData(model);jp,jr=np.empty((3,model.nv)),np.empty((3,model.nv));records=[]
    for frame,(pose,amount) in enumerate(zip(q,e,strict=True)):
        if checkpoint is not None:checkpoint()
        if amount==0:continue
        candidate=pose.copy()
        low=np.maximum(-amount*settings.maximum_joint_delta_rad,model.jnt_range[ids,0]-pose[qa])
        high=np.minimum(amount*settings.maximum_joint_delta_rad,model.jnt_range[ids,1]-pose[qa])
        if np.any(low>high):raise RetargetRejection("source_joint_outside_bounded_search",frame=frame)
        for iteration in range(settings.maximum_iterations+1):
            _kinematics(model,data,candidate)
            error=desired[frame]-data.site_xpos[active]
            inactive_error=inactive_fk[frame]-data.site_xpos[inactive]
            orientation=Rotation.from_matrix(inactive_rot[frame]@data.site_xmat[inactive].reshape(3,3).T).as_rotvec()
            norms=tuple(float(np.linalg.norm(x)) for x in (error,inactive_error,orientation))
            if (norms[0]<=settings.active_position_tolerance_m and norms[1]<=settings.inactive_position_corridor_m
                    and norms[2]<=settings.inactive_orientation_corridor_rad):break
            if iteration==settings.maximum_iterations:break
            mujoco.mj_jacSite(model,data,jp,jr,active);rows=[jp[:,va].copy()];rhs=[error]
            mujoco.mj_jacSite(model,data,jp,jr,inactive)
            rows.extend((2*jp[:,va].copy(),.1*jr[:,va].copy(),.002*np.eye(len(qa))))
            rhs.extend((2*inactive_error,.1*orientation,-.002*(candidate[qa]-pose[qa])))
            correction=candidate[qa]-pose[qa]
            lo=np.maximum(low-correction,-amount*settings.maximum_iteration_delta_rad)
            hi=np.minimum(high-correction,amount*settings.maximum_iteration_delta_rad)
            free=hi-lo>1e-12
            if not free.any():break
            solution=lsq_linear(np.vstack(rows)[:,free],np.concatenate(rhs),bounds=(lo[free],hi[free]),method="bvls",tol=1e-10,max_iter=60)
            if not solution.success or not np.isfinite(solution.x).all():break
            candidate[qa[free]]+=solution.x
        if (norms[0]>settings.active_position_tolerance_m or norms[1]>settings.inactive_position_corridor_m
                or norms[2]>settings.inactive_orientation_corridor_rad):
            raise RetargetRejection("shared_original_corridor_geometry_unsolved",frame=frame,time_s=float(t[frame]),
                frames_completed=frame,frames_unattempted=len(t)-frame-1,iterations=iteration,
                active_error_m=norms[0],inactive_error_m=norms[1],inactive_orientation_error_rad=norms[2])
        result[frame]=candidate
        records.append({"frame":frame,"iterations":iteration,"active_error_m":norms[0],
            "inactive_error_m":norms[1],"inactive_orientation_error_rad":norms[2]})
    fixed=np.ones(model.nq,bool);fixed[qa]=False
    assert np.array_equal(result[:,fixed],q[:,fixed]) and np.array_equal(result[e==0],q[e==0])
    velocity=tangent_velocities(model,t,result)
    # Retain the connector's exact polynomial endpoint conditions. Outside that
    # prefix, use the fresh source-segment finite-difference derivative contract.
    velocity[:prep+1]=v[:prep+1]
    for left,right in holds:
        velocity[left:right+1]=0.
    # Differentiate the actual returned velocity, including its boundary
    # conditions, before prescribing analytic connector/constant-hold qacc.
    acceleration=np.gradient(velocity,t,axis=0,edge_order=2)
    acceleration[:prep+1]=acc[:prep+1]
    for left,right in holds:
        acceleration[left:right+1]=0.
    out={"time_s":t.copy(),"t":t.copy(),"qpos":result,"qvel":velocity,"qacc":acceleration,
        "head_center":target.copy(),"active_hand_index":active_hand,"schedule":schedule,
        "launch_time":schedule.launch_tick*.005,"return_start":schedule.outward_complete_tick*.005,
        "guard_hold_start":schedule.guard_hold_tick*.005,
        "derivative_version":"task10b_shared_target_analytic_connector_fd_zero_holds_v2",
        "source_start_tick":prep,"source_time_s":original_t.copy()}
    if "connector_coefficients" in shared:
        out["connector_coefficients"]=np.asarray(shared["connector_coefficients"]).copy()
    address_fields=("connector_qpos_addresses","connector_dof_addresses")
    if any(name in shared for name in address_fields):
        if not all(name in shared for name in (*address_fields,"connector_coefficients")):
            raise ValueError("Connector coefficient addresses must be supplied together")
        cqa,cva=(np.asarray(shared[name]) for name in address_fields)
        if (cqa.ndim!=1 or cva.shape!=cqa.shape or not np.issubdtype(cqa.dtype,np.integer)
                or not np.issubdtype(cva.dtype,np.integer) or len(set(cqa.tolist()))!=len(cqa)
                or len(set(cva.tolist()))!=len(cva) or np.any((cqa<7)|(cqa>=model.nq))
                or np.any((cva<6)|(cva>=model.nv))
                or out["connector_coefficients"].shape!=(6,len(cqa))):
            raise ValueError("Explicit bounded scalar-joint connector coefficient addresses required")
        for name in address_fields:out[name]=np.asarray(shared[name]).copy()
    if "source_original_glove_pos" in shared:
        historical_gloves=np.asarray(shared["source_original_glove_pos"],float)
        if (historical_gloves.shape!=(len(original_t),2,3)
                or not np.isfinite(historical_gloves).all()
                or not np.allclose(historical_gloves[:,active_hand],original_fk,atol=1e-10,rtol=0)):
            raise ValueError("Stored original glove path disagrees with explicitly bound original FK")
        out["source_original_glove_pos"]=historical_gloves.copy()
    metadata={"schema":"Task10BPhase2SharedTargetComposition/v1","recipe":asdict(recipe),
        "recipe_identity":fingerprint(asdict(recipe)),"source_schedule":asdict(source_schedule),"schedule":asdict(schedule),
        "shared_qpos_sha256":_digest(q),"shared_qvel_sha256":_digest(v),"shared_qacc_sha256":_digest(acc),
        "original_source_qpos_sha256":_digest(original_q),"original_source_active_glove_path_sha256":_digest(original_fk),
        "desired_active_glove_path_sha256":_digest(desired),"original_target_w":original_target.tolist(),"target_w":target.tolist(),
        "source_derivative_version":shared["derivative_version"],"derivative_version":out["derivative_version"],
        "connector_derivatives_preserved_through_tick":prep,"phase_denominator_s":schedule.phase_denominator_s,
        "connector_coefficient_addresses_preserved":all(name in out for name in address_fields),
        "optimized_joint_names":list(settings.optimized_joint_names+settings.trunk_joint_names),"frame_geometry":records,
        "maximum_joint_correction_rad":float(abs(result-q).max()),
        "source_segment_coefficients_omitted":True,
        "constant_source_hold_windows_ticks":[list(window) for window in holds],
        "path_semantics":"shared FK + C2 envelope*(original source FK at global time minus preparation duration + new target minus original target - shared FK); connector and guard endpoints unchanged",
        "phase_semantics":"Explicit source launch/outward/guard offsets plus preparation duration, never inferred from connector motion; same family/TTC request requires new first-contact screening and native qualification",
        "derivative_semantics":"Exact input analytic connector qvel/qacc through preparation endpoint; new MuJoCo tangent velocities with connector and constant-hold velocity conditions applied before differentiating for source acceleration. Then retain analytic connector acceleration and exact zero acceleration on validated constant holds including endpoints. All remaining source accelerations differentiate the returned velocities, including adjacent boundary stencils. Stale source Hermite coefficients are omitted; no continuous-polynomial, globally C2 joint-motion, or continuous between-knot limit claim is inherited. Such interpolation needs a new coefficient build and screen.",
        "readiness_semantics":"Geometric connector endpoint and prescribed schedule do not establish current physical readiness or controller-history continuity",
        "feedforward_recomputed":False,"kinematic_only":True,"physical_qualification":False,"physics_integrated":False}
    return out,metadata


def finalize_shared_target_feedforward(model,candidate,metadata):
    """Original joint-only feedforward, retaining explicit shared launch schedule."""
    from .jab_learning import support_feedforward
    from .phase2_reference import glove_positions
    schedule=candidate.get("schedule")
    if not isinstance(schedule,PhaseSchedule) or metadata.get("schedule")!=asdict(schedule):
        raise ValueError("Shared target schedule changed before finalization")
    hinge=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
    if model.nu!=len(hinge) or not np.array_equal(model.actuator_trnid[:,0],hinge):
        raise ValueError("Ordered scalar joint actuators required; no root actuator")
    out={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in candidate.items()}
    out["torque"]=support_feedforward(model,out,model.jnt_dofadr[hinge])
    if out["torque"].shape!=(len(out["time_s"]),model.nu) or not np.isfinite(out["torque"]).all():
        raise RetargetRejection("nonfinite_or_malformed_original_feedforward")
    out["glove_pos"]=glove_positions(model,out["qpos"])
    out["attack_window_reference_s"]=np.array([schedule.launch_tick,schedule.outward_complete_tick])*.005
    info={**metadata,"feedforward_recomputed":True,"feedforward_route":"original joint-only jab_learning.support_feedforward with preserved connector and recomputed source derivatives",
        "first_contact_TTC_not_yet_screened":True,"physical_qualification":False}
    return out,info
