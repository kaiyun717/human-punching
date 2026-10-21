"""Complete SOMA native/controller snapshots for independent comparisons.

The restore entrypoint accepts a fresh comparison environment only. It never
restores inside preparation, outward execution or recovery. All writes use the
existing native arrays, preserving captured CUDA graph pointers. CPU helpers
copy/pack state without a forward solve or physics integration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import time
import zipfile

import mujoco
import numpy as np
import torch

from .phase2_domain import fingerprint
from .phase2_phases import SavedExecutionState
from .wbc.provenance import PROJECT_ROOT, sha256_file, writable_path, write_json
from .wbc.release_baseline_probe import _tensor_view

VERSION="task10b_native_independent_comparison_state_v1"
CONTROLLER="original_native_affine_5ms_20ms_raw_action_v1"
SPEC=int(mujoco.mjtState.mjSTATE_INTEGRATION)
COUNTS=("nq","nv","nu","na","nhistory","neq","nmocap","nuserdata","nplugin","npluginstate","nbody")
OPTIONAL_COUNTS=("na","nhistory","neq","nmocap","nuserdata","nplugin","npluginstate")
NATIVE_FIELDS=("time","qpos","qvel","act","history","qacc_warmstart","ctrl","qfrc_applied",
    "xfrc_applied","eq_active","mocap_pos","mocap_quat","userdata","plugin_state")
WORLD_FIELDS=("ticks","reset_counts","rejected","launched","initial_up","_last_physical_fallen")
COUNTER_FIELDS=("completed","fallen","physics_steps_executed","actor_updates_executed")
STATIC_FIELDS=frozenset(("args","device","n","dt","decimation","model","ids","qadr","vadr","kp","kd",
    "torque_limits","solver","library","low","high","joint_low","joint_high","sim","hands","feet",
    "chest","hips","contacts","active_hand_index","nominal_qpos","nominal","nominal_centers"))
STATE_FIELDS=frozenset((*WORLD_FIELDS,*COUNTER_FIELDS,"ids_ref","last_action","last_support","hard_faults"))
DIAGNOSTIC_FIELDS=frozenset(("execution_deadline","_phase2_comparison_restore_receipt"))
STATIC_CONTROLLER_ARRAYS=("ids","qadr","vadr","kp","kd","torque_limits","low","high","joint_low","joint_high")


def model_inventory(model):
    counts={key:int(getattr(model,key)) for key in COUNTS}
    if (counts["nq"],counts["nv"],counts["nu"],counts["nbody"])!=(55,54,48,24):
        raise ValueError("Only the accepted 55/54/48-coordinate, 24-body SOMA is supported")
    if any(counts[key] for key in OPTIONAL_COUNTS):
        raise ValueError("Nonempty activation/history/equality/mocap/user/plugin state requires a separately implemented adapter")
    if (int(model.jnt_type[0])!=int(mujoco.mjtJoint.mjJNT_FREE) or model.jnt_qposadr[0]!=0
            or model.jnt_dofadr[0]!=0 or np.any(model.actuator_trnid[:,0]==0)
            or abs(float(model.opt.timestep)-.005)>1.e-12):
        raise ValueError("Accepted free unactuated pelvis and 5ms physics required")
    if int(model.opt.enableflags)&int(mujoco.mjtEnableBit.mjENBL_SLEEP):
        raise ValueError("Sleeping has additional persistent state and is unsupported")
    size=int(mujoco.mj_stateSize(model,SPEC))
    if SPEC!=16383 or size!=410:
        raise ValueError("Complete accepted mjSTATE_INTEGRATION schema changed")
    return dict(counts=counts,native_state_spec=SPEC,native_state_size=size,
        zero_sized_fields_checked=True,sleep_enabled=False)


def model_identity(model):
    """Hash the complete compiled model, including current solver and drive gains."""
    model_inventory(model)
    buffer=np.empty(mujoco.mj_sizeModel(model),np.uint8)
    mujoco.mj_saveModel(model,None,buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def _shape(model,name):
    return {"time":(),"qpos":(model.nq,),"qvel":(model.nv,),"act":(model.na,),
        "history":(model.nhistory,),"qacc_warmstart":(model.nv,),"ctrl":(model.nu,),
        "qfrc_applied":(model.nv,),"xfrc_applied":(model.nbody,6),"eq_active":(model.neq,),
        "mocap_pos":(model.nmocap,3),"mocap_quat":(model.nmocap,4),
        "userdata":(model.nuserdata,),"plugin_state":(model.npluginstate,)}[name]


def _native_views(model,data,worlds):
    model_inventory(model)
    if type(worlds) is not int or not 1<=worlds<=16:
        raise ValueError("One to sixteen comparison worlds required")
    views={}
    for name in NATIVE_FIELDS:
        shape=_shape(model,name)
        empty=bool(shape and np.prod(shape)==0)
        value=getattr(data,name,None)
        if empty:
            # Absence is legal only after checking the compiled model count.
            if value is not None and tuple(value.shape)!=(worlds,*shape):
                raise ValueError("Native zero-sized field has unexpected storage: "+name)
            continue
        if value is None:raise ValueError("Missing native integration field: "+name)
        view=_tensor_view(value)
        if tuple(view.shape)!=(worlds,*shape) or view.dtype!=torch.float32:
            raise ValueError("Expected accepted batched float32 native field: "+name)
        views[name]=view
    return views


def copy_native_integration(model,data,worlds):
    """Pack native named arrays using MuJoCo's complete canonical CPU layout."""
    views=_native_views(model,data,worlds)
    copied={name:value.detach().cpu().numpy().copy() for name,value in views.items()}
    if any(not np.isfinite(value).all() for value in copied.values()):
        raise ValueError("Nonfinite native integration state")
    if not np.allclose(np.linalg.norm(copied["qpos"][:,3:7],axis=-1),1.,atol=1.e-6,rtol=0):
        raise ValueError("Normalized native free-root quaternion required")
    result=np.empty((worlds,mujoco.mj_stateSize(model,SPEC)),np.float64)
    cpu=mujoco.MjData(model)
    for world in range(worlds):
        for name,value in copied.items():
            if name=="time":cpu.time=float(value[world])
            else:getattr(cpu,name)[:]=value[world]
        mujoco.mj_getState(model,cpu,result[world],SPEC)
    return result


def write_native_integration(model,data,values):
    """Write a fully prevalidated state in place; no reset/forward/integration."""
    values=np.asarray(values)
    if values.ndim!=2 or values.dtype!=np.float64 or values.shape[1]!=mujoco.mj_stateSize(model,SPEC) or not np.isfinite(values).all():
        raise ValueError("Finite complete float64 integration vectors required")
    views=_native_views(model,data,len(values));cpu=mujoco.MjData(model)
    restored={name:np.empty(tuple(view.shape),np.float32) for name,view in views.items()}
    for world,value in enumerate(values):
        mujoco.mj_setState(model,cpu,value,SPEC)
        for name in views:
            restored[name][world]=cpu.time if name=="time" else getattr(cpu,name)
    # Reject precision loss before mutating even one destination array.
    check=np.empty_like(values)
    for world in range(len(values)):
        for name,value in restored.items():
            if name=="time":cpu.time=float(value[world])
            else:getattr(cpu,name)[:]=value[world]
        mujoco.mj_getState(model,cpu,check[world],SPEC)
    if not np.array_equal(check,values):
        raise ValueError("Snapshot cannot be represented exactly by native float32 storage")
    pointers={name:view.data_ptr() for name,view in views.items()}
    for name,view in views.items():
        view.copy_(torch.as_tensor(restored[name],dtype=view.dtype,device=view.device))
    if pointers!={name:view.data_ptr() for name,view in views.items()}:
        raise RuntimeError("Native storage pointers changed during restore")


def inspect_adapter(env):
    unknown=set(vars(env))-(STATIC_FIELDS|STATE_FIELDS|DIAGNOSTIC_FIELDS)
    if unknown:raise ValueError("Uninventoried persistent native adapter fields: "+", ".join(sorted(unknown)))
    required=(STATIC_FIELDS|STATE_FIELDS)-{"_last_physical_fallen"}
    if required-set(vars(env)):raise ValueError("Incomplete original specialist adapter inventory")
    if env.n not in range(1,17) or env.decimation!=4 or env.dt!=.005 or env.solver!="accepted":
        raise ValueError("Accepted native comparison controller required")
    sim=env.sim
    if (getattr(sim,"_sensor_context",None) is not None
            or getattr(sim,"_expanded_fields",set()) or getattr(sim,"_world_to_variant",{})
            or getattr(sim,"_per_world_default_fields",set())
            or getattr(getattr(sim,"nan_guard",None),"enabled",False)):
        raise ValueError("Sensor, variant, per-world model or stateful NaN-buffer adapters require explicit support")
    if not isinstance(env.hard_faults,list):raise ValueError("Explicit physical fault ledger required")
    static=hashlib.sha256()
    for name in STATIC_CONTROLLER_ARRAYS:
        value=getattr(env,name)
        array=(_tensor_view(value).detach().cpu().numpy() if isinstance(value,torch.Tensor) or hasattr(value,"_tensor")
            else np.asarray(value))
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError("Explicit finite static drive/controller array required: "+name)
        static.update(name.encode());static.update(array.dtype.str.encode());static.update(str(array.shape).encode());static.update(array.tobytes())
    return dict(adapter_fields=sorted(vars(env)),physical_controller_fields=sorted(STATE_FIELDS&set(vars(env))),
        static_fields=sorted(STATIC_FIELDS),excluded_diagnostic_fields=sorted(DIAGNOSTIC_FIELDS&set(vars(env))),
        static_controller_sha256=static.hexdigest(),
        scratch_contact_buffers="Reducer output/decoded contact buffers are overwritten on every sample; saved support flags and interval remain explicit",
        wall_deadline="Session resource clock is independent of physical/controller snapshot and never rewound")


def _copy(value):return _tensor_view(value).detach().cpu().numpy().copy()


def _ordered_comparison(function):
    """Use the same external CUDA stream as native integration and telemetry."""
    @wraps(function)
    def ordered(env,*args,**kwargs):
        if torch.device(env.device).type!="cuda":return function(env,*args,**kwargs)
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
            raise RuntimeError("Native comparison state access requires a counted CUDA allocation")
        from .wbc.telemetry.cycle_qualification import native_stream_scope
        with native_stream_scope(env.device):return function(env,*args,**kwargs)
    return ordered


def _staged_tensor(value,view,name):
    value=np.asarray(value)
    if value.shape!=tuple(view.shape) or not np.isfinite(value).all():
        raise ValueError("Saved controller field shape/values differ: "+name)
    staged=torch.as_tensor(value.copy(),dtype=view.dtype,device=view.device)
    if not np.array_equal(staged.detach().cpu().numpy(),value):
        raise ValueError("Saved controller field cannot be represented exactly: "+name)
    return staged


@dataclass(frozen=True)
class NativeComparisonSnapshot:
    states: tuple[SavedExecutionState,...]
    inventory_json: str
    version: str=VERSION

    def __post_init__(self):
        object.__setattr__(self,"states",tuple(self.states))
        if self.version!=VERSION or not 1<=len(self.states)<=16:
            raise ValueError("Versioned bounded independent comparison snapshot required")
        inventory=json.loads(self.inventory_json)
        if inventory.get("scope")!="independent_comparison_initialization_only":
            raise ValueError("In-cycle state restoration is forbidden")
        if len({state.model_sha256 for state in self.states})!=1 or any(state.controller_version!=CONTROLLER for state in self.states):
            raise ValueError("One identical physical model/controller required")
        if any(state.decimation_phase for state in self.states):
            raise ValueError("Shared comparison initialization must be at a complete actor boundary")

    @property
    def identity(self):
        return fingerprint(dict(version=self.version,inventory=json.loads(self.inventory_json),
            state_identities=[state.digest() for state in self.states]))

    @property
    def physical_controller_identity(self):
        """Exclude only the separately declared family/reference selection."""
        normalized=[replace(state,reference_id="explicit_independent_comparison_reference_selection",
            adapter_state={key:value for key,value in state.adapter_state.items()
                if key not in ("ids_ref","phase_denominator_s")}).digest() for state in self.states]
        return fingerprint(dict(version=VERSION,states=normalized,
            static_controller_sha256=json.loads(self.inventory_json)["static_controller_sha256"],
            excluded_reference_metadata=["reference_id","ids_ref","phase_denominator_s"]))


@_ordered_comparison
def capture_comparison(env,*,held_raw_action,reference_phase,support_interval_end_s):
    """Capture every physical state/history at a complete actor boundary."""
    inventory=inspect_adapter(env);native=copy_native_integration(env.model,env.sim.data,env.n)
    if env.physics_steps_executed%4 or env.actor_updates_executed!=env.physics_steps_executed//4:
        raise ValueError("Capture only after a complete four-step actor interval")
    held=np.asarray(held_raw_action,np.float32);previous=_copy(env.last_action)
    interval=np.asarray(support_interval_end_s,float)
    if (held.shape!=(env.n,48) or not np.isfinite(held).all() or not np.array_equal(held,previous)
            or interval.shape!=(env.n,) or not np.isfinite(interval).all()
            or not np.allclose(interval,native[:,0],atol=2.e-6,rtol=0)):
        raise ValueError("Explicit completed held action and current support-interval clocks required")
    names=[name for name in WORLD_FIELDS if hasattr(env,name)]
    arrays={name:_copy(getattr(env,name)) for name in names}
    ids=_copy(env.ids_ref).astype(int);denominator=_copy(env.duration)
    faults=np.frombuffer(json.dumps(env.hard_faults,sort_keys=True,allow_nan=False).encode(),np.uint8).copy()
    model_sha=model_identity(env.model);states=[]
    for world in range(env.n):
        adapter={name:value[world] for name,value in arrays.items()}
        adapter.update({name:np.array(getattr(env,name),np.int64) for name in COUNTER_FIELDS})
        adapter.update(ids_ref=np.array(ids[world],np.int64),phase_denominator_s=np.array(denominator[world],np.float32),hard_faults_json=faults)
        row=env.library.rows[ids[world]]
        states.append(SavedExecutionState(model_sha,CONTROLLER,SPEC,native[world],native.shape[1],
            previous[world],held[world],row["reference_sha256"],float(arrays["ticks"][world])*.005,
            reference_phase,int(env.actor_updates_executed),0,_copy(env.last_support)[world],float(interval[world]),True,adapter))
    inventory.update(scope="independent_comparison_initialization_only",model=model_inventory(env.model),
        source_reference_identity=[env.library.rows[i]["reference_sha256"] for i in ids],
        future_reference_selection_may_change_only=["reference_id","ids_ref","phase_denominator_s"],
        native_field_order="Canonical mj_getState/mj_setState layout, mjSTATE_INTEGRATION=16383",
        zero_sized_native_fields=[name for name in NATIVE_FIELDS if _shape(env.model,name) and np.prod(_shape(env.model,name))==0])
    return NativeComparisonSnapshot(tuple(states),json.dumps(inventory,sort_keys=True,allow_nan=False))


@_ordered_comparison
def restore_comparison(env,snapshot,*,trial_id,reference_ids):
    """Restore one fresh independent trial; selected reference is explicit.

    This function does not call env.reset, alter a root pose, reconcile soles or
    start another punch from recovery. Reference selection may change at this
    separate-trial boundary while all physical and raw controller history stays
    identical. A forward refresh is followed by exact integration-state reapply.
    """
    destination_inventory=inspect_adapter(env)
    if (not isinstance(snapshot,NativeComparisonSnapshot) or not isinstance(trial_id,str) or not trial_id
            or hasattr(env,"_phase2_comparison_restore_receipt") or env.physics_steps_executed or env.actor_updates_executed
            or env.hard_faults or len(snapshot.states)!=env.n):
        raise ValueError("Restore is permitted once in a fresh independent comparison trial only")
    if model_identity(env.model)!=snapshot.states[0].model_sha256:
        raise ValueError("Compiled physical/solver/drive model differs from saved comparison state")
    if destination_inventory["static_controller_sha256"]!=json.loads(snapshot.inventory_json)["static_controller_sha256"]:
        raise ValueError("Static native drive/controller arrays differ from saved comparison state")
    selected=np.asarray(reference_ids)
    if selected.shape!=(env.n,) or selected.dtype.kind not in "iu" or np.any(selected<0) or np.any(selected>=len(env.library.rows)):
        raise ValueError("One explicit supported reference selection per comparison world required")
    expected_keys=set(snapshot.states[0].adapter_state)
    if any(set(state.adapter_state)!=expected_keys for state in snapshot.states):
        raise ValueError("Saved controller adapter inventory differs across worlds")
    expected=((set(WORLD_FIELDS)-{"_last_physical_fallen"})&set(vars(env)))|set(COUNTER_FIELDS)|{"ids_ref","phase_denominator_s","hard_faults_json"}
    if "_last_physical_fallen" in expected_keys:expected.add("_last_physical_fallen")
    if expected_keys!=expected:raise ValueError("Saved controller inventory differs from destination adapter")
    for name in COUNTER_FIELDS:
        if len({int(state.adapter_state[name]) for state in snapshot.states})!=1:
            raise ValueError("One consistent native batch counter required")
    for state in snapshot.states:
        if (state.actor_update_index!=int(state.adapter_state["actor_updates_executed"])
                or int(state.adapter_state["physics_steps_executed"])!=4*state.actor_update_index
                or abs(state.reference_clock_s-float(state.adapter_state["ticks"])*.005)>1.e-12
                or abs(state.support_interval_end_s-state.native_integration_state[0])>2.e-6
                or not np.array_equal(state.previous_raw_action,state.held_raw_action)):
            raise ValueError("Saved native/controller/history clocks are inconsistent")
    native=np.stack([state.native_integration_state for state in snapshot.states])
    # Pack/representation and all controller-shape checks precede mutation.
    staged={name:np.stack([state.adapter_state[name] for state in snapshot.states]) for name in WORLD_FIELDS if name in expected_keys}
    staged_views={}
    for name,value in staged.items():
        view=(_tensor_view(getattr(env,name)) if hasattr(env,name) else
            torch.zeros(env.n,dtype=torch.bool,device=env.device))
        staged_views[name]=(view,_staged_tensor(value,view,name))
    for name,values in (("last_action",[state.previous_raw_action for state in snapshot.states]),
            ("last_support",[state.support_contact for state in snapshot.states]),("ids_ref",selected)):
        view=_tensor_view(getattr(env,name));staged_views[name]=(view,_staged_tensor(values,view,name))
    faults={state.adapter_state["hard_faults_json"].tobytes() for state in snapshot.states}
    if len(faults)!=1:raise ValueError("Saved physical fault ledgers differ across worlds")
    fault_ledger=json.loads(next(iter(faults)))
    if not isinstance(fault_ledger,list):raise ValueError("Saved physical fault ledger must be explicit list")
    write_native_integration(env.model,env.sim.data,native)
    env.sim.forward()
    # forward recomputes geometry/contacts and may refresh warmstart. Preserve
    # every integration field exactly; prior support remains a saved interval.
    write_native_integration(env.model,env.sim.data,native)
    for name,(view,value) in staged_views.items():
        view.copy_(value)
        if not hasattr(env,name):setattr(env,name,view)
    for name in COUNTER_FIELDS:setattr(env,name,int(snapshot.states[0].adapter_state[name]))
    env.hard_faults=fault_ledger
    actual=copy_native_integration(env.model,env.sim.data,env.n)
    if not np.array_equal(actual,native):raise RuntimeError("Complete native integration state did not restore exactly")
    if any(not torch.equal(view,value) for view,value in staged_views.values()):
        raise RuntimeError("Complete native raw controller history did not restore exactly")
    receipt=dict(schema=VERSION,trial_id=trial_id,source_snapshot_identity=snapshot.identity,
        physical_controller_identity=snapshot.physical_controller_identity,
        complete_native_integration_equal=True,physical_state_history_preserved=True,
        selected_reference_ids=selected.tolist(),selected_reference_sha256=[env.library.rows[i]["reference_sha256"] for i in selected],
        selected_reference_is_explicit=True,held_raw_action=[state.held_raw_action.tolist() for state in snapshot.states],
        support_interval_end_s=[state.support_interval_end_s for state in snapshot.states],
        reference_phase=[state.reference_phase for state in snapshot.states],
        allowed_reference_metadata_changes=["reference_id","ids_ref","phase_denominator_s"],
        scope="Independent comparison initialization only; no within-cycle reset or continuous punch switching",
        native_execution_parity_qualified=False)
    env._phase2_comparison_restore_receipt=receipt
    return receipt


def save_snapshot(snapshot,output):
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    arrays={};worlds=[]
    for index,state in enumerate(snapshot.states):
        row={}
        for item in fields(state):
            value=getattr(state,item.name)
            if isinstance(value,np.ndarray):
                key=f"world_{index:02d}__{item.name}";arrays[key]=value;row[item.name]={"array":key}
            elif item.name=="adapter_state":
                row[item.name]={}
                for name,array in sorted(value.items()):
                    key=f"world_{index:02d}__adapter__{name}";arrays[key]=array;row[item.name][name]=key
            else:row[item.name]=value
        worlds.append(row)
    path=folder/"state.npz"
    with path.open("xb") as stream:np.savez_compressed(stream,**arrays)
    write_json(folder/"state.json",dict(schema=VERSION,identity=snapshot.identity,version=snapshot.version,
        inventory_json=snapshot.inventory_json,archive_path=str(path),archive_sha256=sha256_file(path),worlds=worlds))
    return folder/"state.json"


def load_snapshot(path):
    path=Path(path).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT.resolve()) or path.stat().st_size>1024**2:
        raise ValueError("Bounded state artifact must resolve inside human-punching")
    data=json.loads(path.read_text());archive=Path(data["archive_path"]).resolve(strict=True)
    if (data.get("schema")!=VERSION or archive.parent!=path.parent or sha256_file(archive)!=data["archive_sha256"]
            or archive.stat().st_size>1024**2):raise ValueError("Changed or unbounded comparison-state archive")
    if not 1<=len(data["worlds"])<=16:raise ValueError("Bounded saved world count required")
    expected=set()
    for row in data["worlds"]:
        for name,value in row.items():
            if name=="adapter_state":expected.update(value.values())
            elif isinstance(value,dict) and set(value)=={"array"}:expected.add(value["array"])
    # Check headers before decompression/allocation, including highly compressed
    # invalid payloads. A complete16-world inventory needs272 small arrays.
    total=0
    with zipfile.ZipFile(archive) as zipped:
        members=zipped.infolist()
        if not 1<=len(members)<=512 or len({m.filename for m in members})!=len(members):
            raise ValueError("Bounded unique snapshot arrays required")
        if {m.filename for m in members}!={name+".npy" for name in expected}:
            raise ValueError("Snapshot archive does not match its exact state inventory")
        for member in members:
            if member.flag_bits&1 or "/" in member.filename or "\\" in member.filename or member.file_size>1024**2+16384:
                raise ValueError("Invalid or oversized snapshot array member")
            with zipped.open(member) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=16384)
                elif version==(2,0):shape,_,dtype=np.lib.format.read_array_header_2_0(stream,max_header_size=16384)
                else:raise ValueError("Unsupported snapshot array header")
                size=math.prod(shape)*dtype.itemsize
                if dtype.hasobject or dtype.fields or dtype.kind not in "biuf" or stream.tell()+size!=member.file_size:
                    raise ValueError("Invalid snapshot numeric array schema")
                total+=size
                if total>1024**2:raise ValueError("State arrays exceed bounded snapshot size")
    states=[]
    with np.load(archive,allow_pickle=False) as arrays:
        if sum(value.nbytes for value in arrays.values())>1024**2:raise ValueError("State arrays exceed bounded snapshot size")
        for row in data["worlds"]:
            values={}
            for name,value in row.items():
                values[name]=({key:arrays[array].copy() for key,array in value.items()} if name=="adapter_state" else
                    arrays[value["array"]].copy() if isinstance(value,dict) and set(value)=={"array"} else value)
            states.append(SavedExecutionState(**values))
    snapshot=NativeComparisonSnapshot(tuple(states),data["inventory_json"],data["version"])
    if snapshot.identity!=data["identity"]:raise ValueError("Saved comparison-state identity differs")
    return snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog",type=Path,required=True);parser.add_argument("--family",required=True)
    parser.add_argument("--release",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--wall-seconds",type=float,required=True)
    parser.add_argument("--capture-bank-domain",type=Path)
    parser.add_argument("--capture-bank-split",choices=("train","validation","test"))
    parser.add_argument("--capture-bank-geometry",type=Path)
    parser.add_argument("--pre-test-freeze",type=Path)
    options=parser.parse_args()
    capture=(options.capture_bank_domain,options.capture_bank_split,options.capture_bank_geometry)
    if any(value is not None for value in capture):
        if not all(value is not None for value in capture) or options.family!="jab":
            parser.error("Bank capture requires all three capture-bank arguments and family jab")
        result=capture_shared_start_bank(options.catalog,options.capture_bank_domain,options.capture_bank_split,
            options.capture_bank_geometry,options.release,options.output,options.wall_seconds,
            pre_test_freeze=options.pre_test_freeze)
        return 0 if result["complete_capture"] else 1
    return native_parity(options.catalog,options.family,options.release,options.output,options.wall_seconds)


def validate_test_bank_freeze(path):
    """Verify measured shared train/validation selection before test-state reads.

    This stable recipe/domain/split receipt precedes capture of the as-yet
    unknown native test-state bytes. Later final declarations bind those bytes;
    they do not authorize adjusting starts/admission from capture outcomes.
    """
    def read_bound(item):
        source=Path(item["path"]).resolve(strict=True)
        if not source.is_relative_to(PROJECT_ROOT.resolve()) or sha256_file(source)!=item["sha256"]:
            raise ValueError("Changed pre-test freeze evidence")
        return json.loads(source.read_text())
    path=Path(path).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT.resolve()):raise ValueError("Owned immutable pre-test freeze required")
    route=json.loads(path.read_text())
    if (route.get("schema")!="task10b_phase2_pre_test_freeze_v1" or route.get("state")!="frozen"
            or route.get("selection_splits")!=["train","validation"] or route.get("test_used_for_selection") is not False
            or route.get("start_mode")!="shared_jab_saved_state"
            or route.get("test_bank_policy")!="capture_all_four_without_admission_filtering_or_start_changes"):
        raise ValueError("Measured frozen shared recipe/domain/splits required before test bank capture")
    from .phase2_qualification import PanelDeclaration,TrialBinding,qualify_panel,validate_resolved_audit
    evidence={name:read_bound(route["successful_validation"][name]) for name in ("qualification","trace_audit","launch","training")}
    qualification,audit,launch,training=(evidence[name] for name in ("qualification","trace_audit","launch","training"))
    selected=PanelDeclaration(**{**qualification["declaration"],"bindings":tuple(TrialBinding(**b) for b in qualification["declaration"]["bindings"])})
    recomputed=qualify_panel(selected,[row["source_outcome"] for row in qualification["trials"]])
    from .wbc.splits import assign_trajectory_groups
    groups=route.get("reserved_test_command_groups",[])
    if (selected.split!="validation" or selected.start_mode!="shared_jab_saved_state" or recomputed.get("passed") is not True
            or recomputed["report_identity"]!=qualification.get("report_identity")
            or selected.family!=route.get("family") or selected.trained_timings_s!=tuple(route.get("trained_timings_s",[]))
            or selected.interpolation_timing_s!=route.get("interpolation_timing_s")
            or selected.checkpoint_sha256!=route.get("selected_checkpoint_sha256")
            or training.get("checkpoint",{}).get("sha256")!=selected.checkpoint_sha256 or training.get("weight_changed") is not True
            or training.get("exception") is not None or training.get("hard_faults")
            or audit.get("passed") is not True or audit.get("closed_protocol_launch_audited") is not True
            or audit.get("shared_saved_state_initialization_audited") is not True
            or audit.get("schema")!="task10b_phase2_independent_trace_audit_v1"
            or not validate_resolved_audit(audit,selected.bindings)["passed"]
            or not launch.get("slurm_job_id") or launch.get("native_execution_requested") is not True
            or launch.get("declaration_identity")!=selected.identity
            or selected.training_receipt_sha256!=route["successful_validation"]["training"]["sha256"]
            or len(groups)!=16 or len(set(groups))!=16 or set(assign_trajectory_groups(groups).values())!={"test"}
            or any(launch.get(key)!=route.get(key) or training.get(key)!=route.get(key)
                for key in ("recipe_sha256","domain_sha256","target_geometry_sha256"))):
        raise ValueError("Pre-test freeze lacks the same successful measured shared validation/learned recipe")
    resolutions={row["trial_id"]:row for row in audit["trial_outcome_resolutions"]}
    if set(resolutions)!={b.trial_id for b in selected.bindings}:
        raise ValueError("Pre-test freeze validation audit covers different trials")
    for trial in qualification["trials"]:
        resolution=resolutions[trial["trial_id"]];source=trial["source_outcome"]
        if ((trial["passed"] and resolution["outcome_resolution"]!="audited_complete")
                or (source["status"] in ("timeout","generation_failure","rejected") and source.get("audit_resolution")!=resolution)):
            raise ValueError("Pre-test freeze qualification differs from independent attempt resolutions")
    launch_directory=Path(route["successful_validation"]["launch"]["path"]).resolve().parent
    if Path(audit.get("run","")).resolve()!=launch_directory:
        raise ValueError("Pre-test freeze audit and native launch refer to different validation runs")
    from .phase2_continuation import verified_run_ancestry
    ancestry=(verified_run_ancestry(launch_directory) if (launch_directory/"execution_identity.json").exists()
        else {"run_directories":[str(launch_directory)]})
    for panel in audit["panels"]:
        raw=(Path(panel["path"])/"raw_trace.npz").resolve(strict=True)
        if (not any(raw.is_relative_to(Path(root)) for root in ancestry["run_directories"])
                or sha256_file(raw)!=panel["raw_sha256"]):
            raise ValueError("Pre-test freeze bound validation trace changed")
    return dict(path=str(path),sha256=sha256_file(path),domain_sha256=route["domain_sha256"],
        recipe_sha256=route["recipe_sha256"],reserved_test_command_groups=groups,
        no_new_human_approval_required=True)


def capture_shared_start_bank(catalog,domain_path,split,geometry_path,release,output,wall_seconds,*,pre_test_freeze=None):
    """Counted four-world native capture before any actor or physics interval."""
    if (not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or split not in ("train","validation","test") or not np.isfinite(wall_seconds) or not 120<wall_seconds<=900):
        raise RuntimeError("Native four-start capture requires a counted CUDA allocation and finite <=900s wall")
    started=time.monotonic()
    # This gate deliberately precedes opening domain, geometry or native inputs.
    freeze=validate_test_bank_freeze(pre_test_freeze) if split=="test" else None
    from .phase2_deadline import ExecutionDeadline
    from .phase2_learning import NativeSpecialistEnv,family_arguments
    from .phase2_shared_catalog import validate_start_bank
    from .phase2_cache import _preflight,_owned_path
    from .wbc.provenance import runtime_record
    from .wbc.telemetry.cycle_qualification import native_stream_scope
    deadline=ExecutionDeadline(wall_seconds,serialization_reserve_s=90.,session_started_monotonic=started)
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    env=None;bank=None;error=None;claim=None
    try:
        with deadline.signal_handlers():
            domain_path=Path(domain_path).resolve(strict=True)
            if not domain_path.is_relative_to(PROJECT_ROOT.resolve()):raise ValueError("Owned frozen domain required")
            domain_sha=sha256_file(domain_path)
            if freeze and freeze["domain_sha256"]!=domain_sha:raise ValueError("Test capture domain differs from prior freeze")
            if split=="test":
                claim=writable_path(PROJECT_ROOT/"outputs/task10b/p2_final_test_bank_claims/shared_jab_four_starts_v1")
                claim.mkdir(parents=True,exist_ok=False)
                write_json(claim/"claim.json",dict(run=str(folder),pre_test_freeze=freeze,scope="Before opening test physical-start payloads",automatic_retry=False))
            domain=json.loads(domain_path.read_text())
            if domain.get("schema")!="Task10BPhase2Domain/v1":raise ValueError("Frozen canonical Phase2 domain required")
            if freeze and set(freeze["reserved_test_command_groups"])!={row["snapshot"]["command_id"] for row in domain["selected_command_groups"] if row["split"]=="test"}:
                raise ValueError("Reserved final command groups changed before native bank capture")
            geometry=_owned_path(geometry_path);geometry_sha=sha256_file(geometry)
            _preflight(geometry,geometry_sha,1024**2)
            with np.load(geometry,allow_pickle=False) as archive:q,v=archive["qpos"].copy(),archive["qvel"].copy()
            args,paths=family_arguments(release,"jab")
            with np.load(paths["reference"],allow_pickle=False) as archive:nominal=archive["qpos"][0].copy()
            validate_start_bank(q,v,nominal,json.loads(paths["body_metadata"].read_text()),domain,split)
            rows=[row for row in json.loads(Path(catalog).read_text())["cells"] if row["family"]=="jab" and row.get("kind")=="original_control"]
            if len(rows)!=1 or rows[0].get("reference_sha256")!=sha256_file(paths["reference"]):
                raise ValueError("One accepted original jab control required")
            runtime=runtime_record()
            if not runtime["declared_pins_match"]:raise ValueError("Fresh Phase2 runtime pins differ")
            with native_stream_scope():
                env=NativeSpecialistEnv(args,rows,num_envs=4);deadline.checkpoint("after_four_world_native_initialization")
                env.reset(random_phase=False,perturb=False,reference_ids=np.zeros(4,dtype=int),initial_qpos=q,initial_qvel=v)
                bank=capture_comparison(env,held_raw_action=_copy(env.last_action),reference_phase="preparation",support_interval_end_s=_copy(env.sim.data.time))
                state_path=save_snapshot(bank,folder/"native_bank")
            result=dict(schema="task10b_phase2_native_shared_bank_capture_v1",complete_capture=True,exception=None,
                split=split,domain_sha256=domain_sha,geometry_start_bank=dict(path=str(geometry),sha256=geometry_sha),
                native_start_state_bank=dict(path=str(state_path),sha256=sha256_file(state_path),identity=bank.identity),
                shared_physical_controller_identity=bank.physical_controller_identity,
                initial_support_contact=[state.support_contact.tolist() for state in bank.states],
                source_sha256=sha256_file(__file__),runtime=runtime,slurm_job_id=os.environ["SLURM_JOB_ID"],
                pre_test_freeze=freeze,planned_native_states=4,captured_native_states=4,
                physics_steps_executed=env.physics_steps_executed,actor_updates_executed=env.actor_updates_executed,
                support_semantics="Initial native forward/support classification at timezero; no executed interval",
                no_actor_execution=True,no_g1_actor_execution=True,native_state_parity_qualified=False,
                shared_start_feasibility_qualified=False,session_wall_s=time.monotonic()-started)
    except Exception as exception:
        error=dict(type=type(exception).__name__,message=str(exception))
        result=dict(schema="task10b_phase2_native_shared_bank_capture_v1",complete_capture=False,exception=error,
            split=split,planned_native_states=4,captured_native_states=0 if bank is None else len(bank.states),
            slurm_job_id=os.environ["SLURM_JOB_ID"],source_sha256=sha256_file(__file__),
            physics_steps_executed=getattr(env,"physics_steps_executed",None),actor_updates_executed=getattr(env,"actor_updates_executed",None),
            native_initialization_attempted=env is not None,execution_extent_unknown=env is None,
            session_wall_s=time.monotonic()-started,pre_test_freeze=freeze,shared_start_feasibility_qualified=False)
    write_json(folder/"bank_capture.json",result)
    if claim is not None:write_json(claim/"terminal.json",dict(run=str(folder),**result))
    return result


def native_parity(catalog,family,release,output,wall_seconds):
    """Future counted-GPU command: retained-state and next-interval native parity."""
    if (not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()
            or not np.isfinite(wall_seconds) or not 120<wall_seconds<=900):
        raise RuntimeError("Native parity requires a counted CUDA allocation and finite <=900s wall")
    from .phase2_deadline import ExecutionDeadline
    from .phase2_learning import NativeSpecialistEnv,family_arguments,fresh_policy
    from .wbc.telemetry.cycle_qualification import native_stream_scope
    started=time.monotonic();deadline=ExecutionDeadline(wall_seconds,serialization_reserve_s=90.)
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    rows=[row for row in json.loads(Path(catalog).read_text())["cells"] if row["family"]==family and row.get("kind")=="original_control"]
    if len(rows)!=1:raise ValueError("Exactly one preserved original family control required")
    args,paths=family_arguments(release,family);error=None;receipt={}
    integrations=[]
    def advance(env,action,label):
        before=env.physics_steps_executed;returned=False
        try:
            env.advance_substep(action,stop_on_fall=True);returned=True
        finally:
            delta=env.physics_steps_executed-before
            integrations.append(dict(branch=label,before_counter=before,after_counter=env.physics_steps_executed,
                retained_completed_physics_steps=delta,call_returned=returned,
                execution_extent_unknown=not returned and delta==0))
    try:
        with deadline.signal_handlers(),native_stream_scope(),torch.no_grad():
            left=NativeSpecialistEnv(args,rows,num_envs=1);deadline.checkpoint("after_first_native_init")
            left.reset(random_phase=False,perturb=False,reference_ids=np.array([0]))
            policy=fresh_policy(paths,left.device);action=policy.actor(left.observe())
            for _ in range(4):advance(left,action,"source_prefix")
            left.last_action[:]=action;left.actor_updates_executed+=1
            saved=capture_comparison(left,held_raw_action=_copy(left.last_action),reference_phase="ready_hold",
                support_interval_end_s=_copy(left.sim.data.time))
            state_path=save_snapshot(saved,folder/"saved_start")
            right=NativeSpecialistEnv(args,rows,num_envs=1);deadline.checkpoint("after_second_native_init")
            right.reset(random_phase=False,perturb=False,reference_ids=np.array([0]))
            restore=restore_comparison(right,load_snapshot(state_path),trial_id="native_parity_independent_destination",reference_ids=np.array([0]))
            left.sim.forward();right.sim.forward()
            observations_equal=torch.equal(left.observe(),right.observe())
            action_left=policy.actor(left.observe());action_right=policy.actor(right.observe())
            interval_equal=[]
            for _ in range(4):
                deadline.checkpoint("native_parity_interval")
                advance(left,action_left,"source_continuation");advance(right,action_right,"restored_continuation")
                interval_equal.append(np.array_equal(copy_native_integration(left.model,left.sim.data,1),copy_native_integration(right.model,right.sim.data,1)))
            receipt=dict(snapshot_identity=saved.identity,restore=restore,observations_equal=observations_equal,
                actor_outputs_equal=torch.equal(action_left,action_right),next_four_integration_states_equal=interval_equal)
    except Exception as error_value:error=dict(type=type(error_value).__name__,message=str(error_value))
    intervals=receipt.get("next_four_integration_states_equal",[])
    passed=bool(not error and receipt.get("observations_equal") and receipt.get("actor_outputs_equal")
        and len(intervals)==4 and all(intervals))
    write_json(folder/"native_parity.json",dict(schema="task10b_phase2_native_state_parity_v1",passed=passed,
        exception=error,receipt=receipt,slurm_job_id=os.environ["SLURM_JOB_ID"],session_wall_s=time.monotonic()-started,
        planned_integration_world_steps=12,integration_calls=integrations,
        retained_completed_physics_steps=sum(row["retained_completed_physics_steps"] for row in integrations),
        unattempted_integration_calls=max(0,12-len(integrations)),
        family=family,g1_actor_executed=False,shared_start_feasibility_qualified=False,source_sha256=sha256_file(__file__)))
    return 0 if passed else 1


if __name__=="__main__":raise SystemExit(main())
