"""Bounded offline target retargeting; no execution or qualification claims.

The default recipe changes explicitly named joints on the active arm only.
Trunk joints require a separately named frozen recipe. Root, legs and all other
coordinates are copied exactly; the full source clock and holds are retained.
No file is generated on import or by this module itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from .phase2_domain import fingerprint
from .wbc.references import tangent_velocities


@dataclass(frozen=True)
class RetargetSettings:
    active_side: str
    optimized_joint_names: tuple[str, ...]
    recipe_version: str = "task10b_active_arm_target_v1"
    trunk_joint_names: tuple[str, ...] = ()
    maximum_iterations: int = 40
    maximum_joint_delta_rad: float = .20
    maximum_iteration_delta_rad: float = .025
    maximum_target_delta_m: float = .12
    active_position_tolerance_m: float = .0005
    inactive_position_corridor_m: float = .002
    inactive_orientation_corridor_rad: float = .005
    envelope_blend_s: float = .12

    def __post_init__(self):
        object.__setattr__(self,"optimized_joint_names",tuple(self.optimized_joint_names))
        object.__setattr__(self,"trunk_joint_names",tuple(self.trunk_joint_names))
        if self.active_side not in ("Left","Right") or not self.optimized_joint_names:
            raise ValueError("Explicit side and named active-arm joint mask required")
        if (len(set(self.optimized_joint_names)) != len(self.optimized_joint_names)
                or len(set(self.trunk_joint_names)) != len(self.trunk_joint_names)
                or set(self.optimized_joint_names)&set(self.trunk_joint_names)):
            raise ValueError("Duplicate or overlapping optimized joint names")
        if self.trunk_joint_names and (self.recipe_version == "task10b_active_arm_target_v1"
                or not self.recipe_version.startswith("task10b_explicit_trunk_target_")):
            raise ValueError("Trunk changes require a separately versioned explicit-trunk recipe")
        if not self.trunk_joint_names and self.recipe_version != "task10b_active_arm_target_v1":
            raise ValueError("The initial active-arm-only recipe has an explicit version")
        if type(self.maximum_iterations) is not int or not 1 <= self.maximum_iterations <= 100:
            raise ValueError("Finite iteration cap of 1..100 required")
        values=(self.maximum_joint_delta_rad,self.maximum_iteration_delta_rad,
                self.maximum_target_delta_m,self.active_position_tolerance_m,
                self.inactive_position_corridor_m,self.inactive_orientation_corridor_rad,self.envelope_blend_s)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError("Positive finite bounds required")
        if self.maximum_joint_delta_rad > .35 or self.maximum_target_delta_m > .15:
            raise ValueError("Recipe exceeds initial bounded offline search")
        if (self.active_position_tolerance_m > .002 or self.inactive_position_corridor_m > .005
                or self.inactive_orientation_corridor_rad > .02):
            raise ValueError("Recipe must retain a tight source glove corridor")


class RetargetRejection(ValueError):
    def __init__(self,reason,**details):
        self.disposition={"reason":reason,"physical_qualification":False,**details}
        super().__init__(reason)


def correction_envelope(times,start_s,end_s,blend_s):
    """Quintic C2 envelope, exactly zero outside the reviewed outward interval."""
    times=np.asarray(times,float)
    if (not np.isfinite(times).all() or not all(map(math.isfinite,(start_s,end_s,blend_s)))
            or not 0 <= start_s < end_s or not 0 < blend_s <= (end_s-start_s)/2):
        raise ValueError("Reviewed interval and nonoverlapping finite blend windows required")
    def smooth(x):
        x=np.clip(x,0,1)
        return x*x*x*(10+x*(-15+6*x))
    result=smooth((times-start_s)/blend_s)*smooth((end_s-times)/blend_s)
    result[(times <= start_s)|(times >= end_s)]=0.
    return result


def _ancestors(model,body):
    result=set()
    while body:
        result.add(body);body=int(model.body_parentid[body])
    return result


def resolve_joint_mask(model,settings):
    """Resolve actual MuJoCo names/addresses and verify branch ownership."""
    active=model.site(settings.active_side+"Hand_center").id
    other="Right" if settings.active_side == "Left" else "Left"
    inactive=model.site(other+"Hand_center").id
    active_chain=_ancestors(model,int(model.site_bodyid[active]))
    inactive_chain=_ancestors(model,int(model.site_bodyid[inactive]))
    protected=set()
    for i in range(model.nsite):
        if "sole" in (model.site(i).name or "").lower():
            protected|=_ancestors(model,int(model.site_bodyid[i]))
    ids=[]
    for name in settings.optimized_joint_names+settings.trunk_joint_names:
        try:joint=model.joint(name)
        except KeyError as error:raise ValueError(f"Missing optimized joint {name}") from error
        body=int(model.jnt_bodyid[joint.id])
        if (model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_HINGE
                or not model.jnt_limited[joint.id] or body in protected or body not in active_chain):
            raise ValueError(f"Joint {name} is not a bounded active upper-body hinge")
        if name in settings.trunk_joint_names:
            if body not in inactive_chain or name.rsplit("_",1)[0] not in ("Spine1","Spine2","Chest"):
                raise ValueError("Only explicitly named shared torso joints may enter the trunk recipe")
        elif (body in inactive_chain or name.rsplit("_",1)[0] not in
                tuple(settings.active_side+x for x in ("Shoulder","Arm","ForeArm","Hand"))):
            raise ValueError("Active-arm mask cannot modify trunk, opposite arm, head or legs")
        ids.append(joint.id)
    return {"joint_ids":np.array(ids,dtype=int),"qpos_addresses":model.jnt_qposadr[ids].copy(),
            "dof_addresses":model.jnt_dofadr[ids].copy(),"active_site":active,"inactive_site":inactive}


def _kinematics(model,data,pose):
    data.qpos[:]=pose
    mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)


def _require_standalone_source(source,t,q,start_s):
    """Motion-onset clocks are valid only for a source without preparation.

    A moving prefix requires the separate explicit shared-phase composition.
    Check again at finalization so callers cannot bypass the constructor guard.
    """
    schedule=source.get("schedule")
    preparation_tick=(getattr(schedule,"preparation_complete_tick",0)
                      if not isinstance(schedule,dict) else schedule.get("preparation_complete_tick",0))
    if preparation_tick != 0 or source.get("source_start_tick",0) != 0:
        raise RetargetRejection("preparation_requires_explicit_shared_target_recipe")
    prefix=np.asarray(t)<float(start_s)-1.e-12
    if np.any(prefix) and not np.allclose(q[prefix],q[0],atol=1.e-12,rtol=0):
        raise RetargetRejection("moving_preparation_requires_explicit_shared_target_recipe")


def retarget_upper_body(model,source,target_w,*,source_outward_start_s,
                        source_outward_end_s,settings: RetargetSettings,checkpoint=None):
    """Produce an in-memory kinematic candidate, or a structured refusal.

    The target shift is applied to the source active-glove corridor through the
    C2 envelope. This is not a contact/TTC guarantee: both must be remeasured and
    independently screened after the path changes. No lower-body IK is used.
    """
    t=np.asarray(source["time_s"],float);q=np.asarray(source["qpos"],float)
    target=np.asarray(target_w,float);original=np.asarray(source["head_center"],float)
    active_hand=0 if settings.active_side == "Left" else 1
    if source.get("active_hand_index") != active_hand:
        raise ValueError("Explicit source active hand must match the retarget recipe")
    if (len(t)<3 or t.ndim != 1 or t[0] != 0 or q.shape != (len(t),model.nq)
            or not np.allclose(np.diff(t),.005,atol=1e-12,rtol=0)
            or not all(np.isfinite(x).all() for x in (t,q,target,original))
            or target.shape != (3,) or original.shape != (3,)
            or not 0 <= source_outward_start_s < source_outward_end_s <= t[-1]):
        raise ValueError("Finite complete source reference on the 5ms master clock required")
    if not source_outward_end_s <= float(source["guard_hold_start"]) <= t[-1]:
        raise ValueError("Full source guard hold must be retained")
    _require_standalone_source(source,t,q,source_outward_start_s)
    delta_target=target-original
    if np.linalg.norm(delta_target)>settings.maximum_target_delta_m:
        raise RetargetRejection("target_offset_outside_recipe",target_delta_m=delta_target.tolist())
    mask=resolve_joint_mask(model,settings);ids=mask["joint_ids"];qa=mask["qpos_addresses"];va=mask["dof_addresses"]
    active,inactive=mask["active_site"],mask["inactive_site"]
    envelope=correction_envelope(t,source_outward_start_s,source_outward_end_s,settings.envelope_blend_s)
    result=q.copy();data=mujoco.MjData(model);jp,jr=np.empty((3,model.nv)),np.empty((3,model.nv))
    records=[]
    for frame,(pose,e) in enumerate(zip(q,envelope,strict=True)):
        if checkpoint is not None:checkpoint()
        if e == 0:
            records.append({"iterations":0,"active_error_m":0.,"inactive_error_m":0.,"inactive_orientation_error_rad":0.});continue
        _kinematics(model,data,pose)
        desired=data.site_xpos[active].copy()+e*delta_target
        inactive_position=data.site_xpos[inactive].copy();inactive_rotation=data.site_xmat[inactive].reshape(3,3).copy()
        candidate=pose.copy()
        low=np.maximum(-e*settings.maximum_joint_delta_rad,model.jnt_range[ids,0]-pose[qa])
        high=np.minimum(e*settings.maximum_joint_delta_rad,model.jnt_range[ids,1]-pose[qa])
        if np.any(low>high):raise RetargetRejection("source_joint_outside_bounded_search",frame=frame)
        for iteration in range(settings.maximum_iterations+1):
            _kinematics(model,data,candidate)
            active_error=desired-data.site_xpos[active]
            inactive_error=inactive_position-data.site_xpos[inactive]
            orientation_error=Rotation.from_matrix(inactive_rotation@data.site_xmat[inactive].reshape(3,3).T).as_rotvec()
            errors=tuple(float(np.linalg.norm(x)) for x in (active_error,inactive_error,orientation_error))
            if (errors[0]<=settings.active_position_tolerance_m and errors[1]<=settings.inactive_position_corridor_m
                    and errors[2]<=settings.inactive_orientation_corridor_rad):break
            if iteration == settings.maximum_iterations:break
            mujoco.mj_jacSite(model,data,jp,jr,active);rows=[jp[:,va].copy()];rhs=[active_error]
            mujoco.mj_jacSite(model,data,jp,jr,inactive)
            rows.extend((2*jp[:,va].copy(),.1*jr[:,va].copy(),.002*np.eye(len(qa))))
            rhs.extend((2*inactive_error,.1*orientation_error,-.002*(candidate[qa]-pose[qa])))
            correction=candidate[qa]-pose[qa]
            lo=np.maximum(low-correction,-e*settings.maximum_iteration_delta_rad)
            hi=np.minimum(high-correction,e*settings.maximum_iteration_delta_rad)
            free=hi-lo>1e-12
            if not free.any():break
            solution=lsq_linear(np.vstack(rows)[:,free],np.concatenate(rhs),bounds=(lo[free],hi[free]),
                               method="bvls",tol=1e-10,max_iter=60)
            if not solution.success or not np.isfinite(solution.x).all():break
            candidate[qa[free]]+=solution.x
        if (errors[0]>settings.active_position_tolerance_m or errors[1]>settings.inactive_position_corridor_m
                or errors[2]>settings.inactive_orientation_corridor_rad):
            raise RetargetRejection("bounded_upper_body_geometry_unsolved",frame=frame,time_s=float(t[frame]),
                frames_completed=frame,frames_unattempted=len(t)-frame-1,iterations=iteration,
                active_error_m=errors[0],inactive_error_m=errors[1],inactive_orientation_error_rad=errors[2])
        result[frame]=candidate
        records.append(dict(iterations=iteration,active_error_m=errors[0],inactive_error_m=errors[1],inactive_orientation_error_rad=errors[2]))
    fixed=np.ones(model.nq,bool);fixed[qa]=False
    if not np.array_equal(result[:,fixed],q[:,fixed]) or not np.array_equal(result[envelope==0],q[envelope==0]):
        raise AssertionError("Protected root/legs/holds changed")
    out={"time_s":t.copy(),"t":t.copy(),"qpos":result,"head_center":target.copy(),
         "launch_time":float(source_outward_start_s),"return_start":float(source_outward_end_s),
         "guard_hold_start":float(source["guard_hold_start"]),"active_hand_index":active_hand}
    out["qvel"]=tangent_velocities(model,t,result);out["qacc"]=np.gradient(out["qvel"],t,axis=0,edge_order=2)
    metadata={"schema":"Task10BPhase2BoundedTargetPath/v1","recipe":asdict(settings),"recipe_identity":fingerprint(asdict(settings)),
        "source_qpos_sha256":hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),
        "optimized_joint_names":list(settings.optimized_joint_names+settings.trunk_joint_names),
        "optimized_qpos_addresses":qa.tolist(),"optimized_dof_addresses":va.tolist(),
        "source_outward_start_s":float(source_outward_start_s),"source_outward_end_s":float(source_outward_end_s),
        "source_target_w":original.tolist(),"new_target_w":target.tolist(),"target_delta_m":delta_target.tolist(),
        "source_clock_and_all_holds_preserved":True,"root_and_other_joints_exactly_preserved":True,
        "maximum_joint_correction_rad":float(abs(result-q).max()),"frame_geometry":records,
        "envelope_semantics":"C2 quintic envelope zero through source outward start and from reviewed source outward end, including captured withdrawal; full source holds retained. C2 envelope is not a guarantee of globally C2 solved joint paths.",
        "event_semantics":"Changed reference requires fresh actor-boundary launch and first-contact measurements. No original TTC or physical qualification is inherited.",
        "derivative_semantics":"Recomputed MuJoCo tangent velocities, including local free-root angular convention, and derivatives of those velocities on complete 5ms clock",
        "screening_required":["first_contact_and_TTC","joint_limits","self_collision","declared_support","friction_unilateral_forces","six_unactuated_base_equations","effort","native_execution"],
        "feedforward_recomputed":False,"kinematic_only":True,"physical_qualification":False,"physics_integrated":False}
    return out,metadata


def finalize_original_feedforward(model,candidate,metadata):
    """Recompute original joint-only feedforward and newly derived event clocks.

    This remains an offline candidate. Inverse-statics forces are not physical
    support certification; dynamic allocation screening and native checks follow.
    """
    from .jab_learning import support_feedforward
    from .phase2_reference import glove_positions,reference_events
    hinge=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
    if model.nu != len(hinge) or not np.array_equal(model.actuator_trnid[:,0],hinge):
        raise ValueError("Original ordered scalar joint actuators required; no root actuator")
    out={key:value.copy() if isinstance(value,np.ndarray) else value for key,value in candidate.items()}
    t=out["time_s"]
    _require_standalone_source(out,t,out["qpos"],metadata["source_outward_start_s"])
    out["qvel"]=tangent_velocities(model,t,out["qpos"])
    out["qacc"]=np.gradient(out["qvel"],t,axis=0,edge_order=2)
    out["torque"]=support_feedforward(model,out,model.jnt_dofadr[hinge])
    if out["torque"].shape != (len(t),model.nu) or not np.isfinite(out["torque"]).all():
        raise RetargetRejection("nonfinite_or_malformed_original_feedforward")
    out["glove_pos"]=glove_positions(model,out["qpos"])
    events=reference_events(t,out["qpos"],out["qvel"],metadata["source_outward_end_s"],1.)
    events["phase_semantics"]="Complete target-corrected source clock and all source holds, plus original 1s terminal hold; fresh launch from pose/velocity at 20ms actor boundary. Reviewed source outward endpoint including withdrawal retained; no actor history or phase reset."
    out["launch_time"]=events["launch_time_s"]
    out["return_start"]=events["outward_completion_time_s"]
    out["attack_window_reference_s"]=np.array([events["launch_time_s"],events["outward_completion_time_s"]])
    info={**metadata,"feedforward_recomputed":True,"feedforward_route":"original jab_learning.support_feedforward; joint-only inverse statics; no acceleration feedforward, optimized force application or root assistance",
          "events":events,"phase_denominator_s":float(t[-1]+1.),"kinematic_only":True,"physical_qualification":False}
    return out,info
