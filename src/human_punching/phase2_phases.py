"""Explicit connector phase, commitment and state-continuity contracts.

These helpers neither construct a connector nor execute physics. Preparation may
move every joint: motion onset is consequently never used to infer outward launch.
Numerical join/derivative checks do not establish dynamic feasibility or readiness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import mujoco
import numpy as np

PHYSICS_DT = .005
ACTOR_STEPS = 4
FAMILIES = ("jab", "cross", "left_hook", "right_hook")


def _tick(value, name):
    if isinstance(value, bool) or not isinstance(value, (int,np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer physics tick")
    return int(value)


def _sha(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("A lower-case SHA256 identity is required")
    return value


@dataclass(frozen=True)
class PhaseSchedule:
    preparation_complete_tick: int
    launch_tick: int
    outward_complete_tick: int
    guard_hold_tick: int
    reference_end_tick: int
    semantics_version: str = "task10b_explicit_connector_phases_v1"

    def __post_init__(self):
        for name in ("preparation_complete_tick","launch_tick","outward_complete_tick","guard_hold_tick","reference_end_tick"):
            object.__setattr__(self,name,_tick(getattr(self,name),name))
        if not (self.preparation_complete_tick <= self.launch_tick < self.outward_complete_tick
                <= self.guard_hold_tick <= self.reference_end_tick):
            raise ValueError("Preparation, launch, outward, guard and reference-end order differs")
        if self.launch_tick%ACTOR_STEPS:
            raise ValueError("Launch must be a 20ms actor boundary")
        if self.semantics_version!="task10b_explicit_connector_phases_v1":
            raise ValueError("Unreviewed phase-semantics version")

    @property
    def phase_denominator_s(self):
        return self.reference_end_tick*PHYSICS_DT+1.

    @property
    def evaluation_end_tick(self):
        return self.reference_end_tick+200

    @property
    def actor_padded_end_tick(self):
        return ((self.evaluation_end_tick+3)//4)*4

    def phase_at(self,tick):
        tick=_tick(tick,"tick")
        if tick<self.preparation_complete_tick:
            return "preparation"
        if tick<self.launch_tick:
            return "ready_hold"
        if tick<self.outward_complete_tick:
            return "outward"
        if tick<self.guard_hold_tick:
            return "recovery"
        return "guard_hold" if tick<=self.evaluation_end_tick else "actor_padding"


@dataclass(frozen=True)
class RequestedCommand:
    family: str
    ttc_s: float
    style: str
    checkpoint_sha256: str

    def __post_init__(self):
        if self.family not in FAMILIES or not np.isfinite(self.ttc_s) or self.ttc_s<.5:
            raise ValueError("Known family and finite TTC>=.5s required")
        if not isinstance(self.style,str) or not self.style.strip():
            raise ValueError("An explicit bounded style identity is required")
        _sha(self.checkpoint_sha256)


def _frozen_array(value,shape=None,dtype=float):
    result=np.asarray(value,dtype=dtype).copy()
    if shape is not None and result.shape!=shape:
        raise ValueError(f"Expected snapshot array shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Snapshot/reference arrays must be finite")
    # Immutable bytes backing prevents callers re-enabling ndarray write flags.
    return np.frombuffer(result.tobytes(),dtype=result.dtype).reshape(result.shape)


@dataclass(frozen=True)
class SavedExecutionState:
    """Model-sized mjSTATE_INTEGRATION plus every declared controller history.

    Native integration state includes engine HISTORY, actuator state, warmstart,
    equality/mocap/user/plugin state, applied forces and ctrl, not merely q/v.
    Callers must explicitly inventory additional adapter state in extras; an
    empty inventory is an assertion, not permission to discard discovered state.
    """
    model_sha256: str
    controller_version: str
    native_state_spec: int
    native_integration_state: np.ndarray
    native_state_size: int
    previous_raw_action: np.ndarray
    held_raw_action: np.ndarray
    reference_id: str
    reference_clock_s: float
    reference_phase: str
    actor_update_index: int
    decimation_phase: int
    support_contact: np.ndarray
    support_interval_end_s: float
    adapter_inventory_complete: bool
    adapter_state: Mapping[str,np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        _sha(self.model_sha256)
        if self.native_state_spec!=int(mujoco.mjtState.mjSTATE_INTEGRATION):
            raise ValueError("A complete mjSTATE_INTEGRATION vector is required; qpos/qvel is insufficient")
        if isinstance(self.native_state_size,bool) or not isinstance(self.native_state_size,(int,np.integer)) or self.native_state_size<1:
            raise ValueError("A model-checked native state size is required")
        object.__setattr__(self,"native_state_size",int(self.native_state_size))
        object.__setattr__(self,"native_state_spec",int(self.native_state_spec))
        if not self.controller_version or not self.reference_id or self.reference_phase not in ("preparation","ready_hold","outward","recovery","guard_hold"):
            raise ValueError("Explicit controller/reference/phase identities required")
        for name in ("reference_clock_s","support_interval_end_s"):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError("Finite nonnegative controller/support clocks required")
        object.__setattr__(self,"actor_update_index",_tick(self.actor_update_index,"actor_update_index"))
        if _tick(self.decimation_phase,"decimation_phase")>=4:
            raise ValueError("Decimation phase must lie in0..3")
        object.__setattr__(self,"decimation_phase",int(self.decimation_phase))
        if self.adapter_inventory_complete is not True:
            raise ValueError("Inventory all persistent adapter/controller state before comparison")
        for name,shape,dtype in (("native_integration_state",(self.native_state_size,),float),
            ("previous_raw_action",(48,),float),("held_raw_action",(48,),float),
            ("support_contact",(2,),bool)):
            if name=="support_contact" and not np.isin(np.asarray(getattr(self,name)),[0,1]).all():
                raise ValueError("Support flags must be boolean")
            object.__setattr__(self,name,_frozen_array(getattr(self,name),shape,dtype))
        frozen={key:_frozen_array(value,dtype=None) for key,value in self.adapter_state.items()}
        if any(not isinstance(key,str) or not key for key in frozen):
            raise ValueError("Additional adapter state needs explicit named ownership")
        object.__setattr__(self,"adapter_state",MappingProxyType(frozen))

    @classmethod
    def capture_cpu_native(cls,model,data,**controller_fields):
        """Read CPU native state only; performs no forward solve or integration."""
        spec=int(mujoco.mjtState.mjSTATE_INTEGRATION)
        size=mujoco.mj_stateSize(model,spec)
        state=np.empty(size)
        mujoco.mj_getState(model,data,state,spec)
        return cls(native_state_spec=spec,native_state_size=size,native_integration_state=state,**controller_fields)

    def validate_model_size(self,model):
        if self.native_state_size!=mujoco.mj_stateSize(model,self.native_state_spec):
            raise ValueError("Native integration vector does not cover this model's complete state")

    def digest(self):
        h=hashlib.sha256()
        for name in self.__dataclass_fields__:
            value=getattr(self,name)
            h.update(name.encode())
            if isinstance(value,np.ndarray):
                h.update(str(value.shape).encode());h.update(value.dtype.str.encode());h.update(value.tobytes())
            elif name=="adapter_state":
                for key,array in sorted(value.items()):
                    h.update(key.encode());h.update(str(array.shape).encode());h.update(array.dtype.str.encode());h.update(array.tobytes())
            else:
                h.update(json.dumps(value,sort_keys=True).encode())
        return h.hexdigest()


def compare_saved_states(left: SavedExecutionState,right: SavedExecutionState):
    """Strict comparison for separate trials restored from one saved start."""
    changed=[]
    for name in left.__dataclass_fields__:
        a,b=getattr(left,name),getattr(right,name)
        if isinstance(a,np.ndarray):
            same=np.array_equal(a,b)
        elif name=="adapter_state":
            same=set(a)==set(b) and all(a[key].dtype==b[key].dtype and np.array_equal(a[key],b[key]) for key in a)
        else:
            same=a==b
        if not same:
            changed.append(name)
    return {"identical":not changed,"changed_fields":changed,"left_digest":left.digest(),"right_digest":right.digest(),
        "scope":"Separate comparison initializations only; no in-cycle reset or arbitrary recovered-state switching claim"}


@dataclass(frozen=True)
class LaunchCommitment:
    command: RequestedCommand
    target_w: tuple[float,float,float]
    reference_sha256: str
    phase_semantics_version: str
    actual_launch_tick: int
    prepared_tick: int
    measured_state_sha256: str
    final_planning_wall_s: float

    def __post_init__(self):
        target=_frozen_array(self.target_w,(3,))
        object.__setattr__(self,"target_w",tuple(float(value) for value in target))
        _sha(self.reference_sha256);_sha(self.measured_state_sha256)
        if _tick(self.actual_launch_tick,"actual_launch_tick")%4:
            raise ValueError("Immutable launch is not at an actor boundary")
        if _tick(self.prepared_tick,"prepared_tick")>self.actual_launch_tick:
            raise ValueError("Immutable launch precedes preparation completion")
        if not np.isfinite(self.final_planning_wall_s) or not 0<=self.final_planning_wall_s<=.020:
            raise ValueError("Immutable launch exceeds final planning wall bound")

    def assert_unchanged(self,command,target_w,reference_sha256):
        if command!=self.command or tuple(np.asarray(target_w,float))!=self.target_w or reference_sha256!=self.reference_sha256:
            raise ValueError("Committed family/TTC/style/checkpoint, frozen target or reference changed")


class LaunchLatch:
    """One request, one explicit launch attempt; rejection cannot become a late hit."""
    def __init__(self,schedule: PhaseSchedule,command: RequestedCommand):
        self._schedule,self._command=schedule,command
        self._commitment=None
        self._disposition=None

    @property
    def schedule(self):
        return self._schedule

    @property
    def command(self):
        return self._command

    @property
    def commitment(self):
        return self._commitment

    @property
    def disposition(self):
        return self._disposition

    def attempt(self,*,current_tick,prepared_tick,current_ready,target_w,reference_sha256,
                state: SavedExecutionState,final_planning_wall_s,model):
        if self.disposition is not None:
            raise ValueError("This request already has an immutable launch/rejection disposition")
        tick=_tick(current_tick,"current_tick")
        prepared=_tick(prepared_tick,"prepared_tick") if prepared_tick is not None else None
        target=_frozen_array(target_w,(3,))
        _sha(reference_sha256)
        state.validate_model_size(model)
        if not isinstance(current_ready,(bool,np.bool_)):
            raise ValueError("Current measured readiness must be an explicit boolean")
        if not np.isfinite(final_planning_wall_s) or final_planning_wall_s<0:
            raise ValueError("Measured finite final-planning wall latency required")
        reason=("not_actor_boundary" if tick%4 or state.decimation_phase else
            "wrong_launch_boundary" if tick!=self.schedule.launch_tick else
            "preparation_missing_or_late" if prepared is None or prepared>tick or prepared>self.schedule.preparation_complete_tick else
            "current_readiness_failed" if not current_ready else
            "phase_not_ready_for_launch" if state.reference_phase!="ready_hold" else
            "support_not_current" if not state.support_contact.all() or abs(state.support_interval_end_s-state.native_integration_state[0])>1e-5 else
            "final_planning_late" if final_planning_wall_s>.020 else
            "reference_clock_discontinuous" if abs(state.reference_clock_s-tick*PHYSICS_DT)>1e-9 else None)
        self._disposition="rejected:"+reason if reason else "committed"
        if reason:
            return {"accepted":False,"reason":reason,"actual_launch_time_s":None}
        self._commitment=LaunchCommitment(self.command,tuple(float(x) for x in target),reference_sha256,
            self.schedule.semantics_version,tick,prepared,state.digest(),float(final_planning_wall_s))
        return {"accepted":True,"reason":None,"actual_launch_time_s":tick*PHYSICS_DT,
            "commitment":self.commitment}


@dataclass(frozen=True)
class NumericalContinuityTolerances:
    configuration_tangent: float = 1e-8
    velocity: float = 1e-7
    acceleration: float = 1e-5

    def __post_init__(self):
        if any(not np.isfinite(value) or value<0 for value in self.__dict__.values()):
            raise ValueError("Finite nonnegative numerical continuity tolerances required")


def validate_phase_join(model,left,right,*,tolerances=NumericalContinuityTolerances()):
    """Compare two declarations of the same boundary; never overwrite either."""
    states=[]
    for endpoint in (left,right):
        states.append(tuple(_frozen_array(endpoint[key],shape) for key,shape in
            (("qpos",(model.nq,)),("qvel",(model.nv,)),("qacc",(model.nv,)))))
    if model.jnt_type[0]!=mujoco.mjtJoint.mjJNT_FREE or any(not np.isclose(np.linalg.norm(state[0][3:7]),1.,atol=1e-7,rtol=0) for state in states):
        raise ValueError("A free root with normalized endpoint quaternions is required")
    displacement=np.empty(model.nv)
    mujoco.mj_differentiatePos(model,displacement,1.,states[0][0],states[1][0])
    errors={"configuration_tangent":float(np.max(np.abs(displacement),initial=0)),
        "velocity":float(np.max(np.abs(states[0][1]-states[1][1]),initial=0)),
        "acceleration":float(np.max(np.abs(states[0][2]-states[1][2]),initial=0))}
    return {"passed":all(value<=getattr(tolerances,key) for key,value in errors.items()),"errors":errors,
        "scope":"Numerical C2 boundary identity only; no physical feasibility, support or effort assertion"}


def validate_reference_derivatives(model,times,qpos,qvel,qacc,schedule: PhaseSchedule,
                                   *,tolerances=NumericalContinuityTolerances()):
    t=_frozen_array(times)
    if t.ndim!=1 or len(t)<3:
        raise ValueError("At least three reference timestamps are required")
    q,v,a=(_frozen_array(value,shape) for value,shape in
        ((qpos,(len(t),model.nq)),(qvel,(len(t),model.nv)),(qacc,(len(t),model.nv))))
    if (t.ndim!=1 or len(t)<3 or t[0]!=0 or len(t)!=schedule.reference_end_tick+1
            or not np.allclose(np.diff(t),PHYSICS_DT,atol=1e-12,rtol=0)):
        raise ValueError("Complete newly assembled reference must use the explicit exact5ms phase grid")
    if model.jnt_type[0]!=mujoco.mjtJoint.mjJNT_FREE or not np.allclose(np.linalg.norm(q[:,3:7],axis=1),1.,atol=1e-7,rtol=0):
        raise ValueError("A free root with normalized reference quaternions is required")
    fresh=np.empty_like(v)
    for i in range(len(t)):
        lo,hi=max(0,i-1),min(len(t)-1,i+1)
        mujoco.mj_differentiatePos(model,fresh[i],t[hi]-t[lo],q[lo],q[hi])
    acceleration=np.gradient(fresh,t,axis=0,edge_order=2)
    errors={"velocity":float(np.max(abs(v-fresh),initial=0)),
            "acceleration":float(np.max(abs(a-acceleration),initial=0))}
    return {"passed":all(value<=getattr(tolerances,key) for key,value in errors.items()),"errors":errors,
        "phase_denominator_s":schedule.phase_denominator_s,
        "launch_tick":schedule.launch_tick,"launch_inferred_from_motion":False,
        "scope":"Finite-difference consistency on assembled free-root tangent path; adjacent-step motion limits, contact, dynamics and native continuity require separate screening"}
