"""Continuous shared-jab preparation panels with complete native start restore.

The selected family restores an identical captured physical/controller start
once, before its first recorded actor interval. Preparation, launch, outward
execution and recovery then share one uninterrupted state/history. This is not
switching from arbitrary recovery states or a scheduler experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .phase2_cache import _owned_path,_preflight
from .phase2_domain import fingerprint
from .phase2_expanded_evaluation import (
    ClosedProtocolNativeSpecialistEnv,EvaluationLimits,MAX_REFERENCE_BYTES,
    execute as execute_expanded,owned_file,read_json,validate_trial_row,
    validate_negative_reference_identity,
)
from .phase2_launch import ClosedProtocolLaunchGuard
from .phase2_learning import NativeSpecialistEnv,family_arguments
from .phase2_native_state import (
    NativeComparisonSnapshot,copy_native_integration,load_snapshot,restore_comparison,
)
from .phase2_phases import PhaseSchedule
from .phase2_sampling import IntegerReferencePlayback,VERSION as PLAYBACK_VERSION
from .wbc.provenance import sha256_file,write_json
from .wbc.release_baseline_probe import _copy

VERSION="task10b_shared_jab_initial_state_identity_v2"
MANIFEST="task10b_phase2_shared_expanded_manifest_v2"
from .phase2_initial_state_v2 import (VERSION as INITIAL_IDENTITY_VERSION,CAPTURE,validate_witness,restore_initial_comparison,read_bound)


def validate_shared_row(binding,declaration,row):
    # Reuse family/target/group/horizon checks without inferring preparation
    # from first motion or relabeling the source's original launch tick.
    validate_trial_row(binding,declaration,{**row,"start_mode":"family_guard","preparation_complete_tick":0})
    if (row.get("start_mode")!="shared_jab_saved_state" or row.get("start_index")!=binding.start_index
            or row.get("phase_semantics_version")!="task10b_explicit_connector_phases_v1"):
        raise ValueError("Shared reference must bind its exact start and reviewed phase semantics")
    schedule=PhaseSchedule(row["preparation_complete_tick"],row["launch_tick"],row["outward_end_tick"],
        row["guard_hold_tick"],row["reference_end_tick"],row["phase_semantics_version"])
    if (not 0<schedule.preparation_complete_tick<=200
            or abs(schedule.reference_end_tick*.005-row["duration_s"])>1.e-9
            or schedule.phase_denominator_s!=row["phase_denominator_s"]):
        raise ValueError("Explicit bounded preparation and full shared reference denominator required")
    return schedule


def validate_native_bank(bank,model,accepted_jab_sha256):
    if len(bank.states)!=4:raise ValueError("One complete native snapshot for each of the four shared starts required")
    q=[];v=[];data=mujoco.MjData(model)
    for state in bank.states:
        state.validate_model_size(model)
        if (state.reference_id!=accepted_jab_sha256 or state.reference_clock_s or state.actor_update_index
                or state.decimation_phase or state.reference_phase!="preparation"
                or np.any(state.previous_raw_action) or np.any(state.held_raw_action)
                or state.native_integration_state[0] or state.support_interval_end_s
                or int(state.adapter_state["ticks"]) or int(state.adapter_state["reset_counts"])!=1
                or bool(state.adapter_state["launched"]) or bool(state.adapter_state["rejected"])
                or any(int(state.adapter_state[name]) for name in ("completed","fallen","physics_steps_executed","actor_updates_executed"))
                or json.loads(state.adapter_state["hard_faults_json"].tobytes())!=[]):
            raise ValueError("Initial shared bank must be a genuine fresh jab capture before its first actor/physics interval")
        mujoco.mj_setState(model,data,state.native_integration_state,state.native_state_spec)
        if np.any(data.ctrl) or np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            raise ValueError("Initial shared capture may not contain root assistance, external forces or an invented held drive")
        q.append(data.qpos.astype(np.float32));v.append(data.qvel.astype(np.float32))
    return np.stack(q),np.stack(v)


class SharedNativeBase(IntegerReferencePlayback,NativeSpecialistEnv):
    """No additional mutable instance state until after independent restoration."""


class SharedProtocolEnv:
    """Compose launch history outside the strictly inventoried native base."""
    cache_versions=ClosedProtocolNativeSpecialistEnv.cache_versions
    protocol_snapshot=ClosedProtocolNativeSpecialistEnv.protocol_snapshot
    current_requests=ClosedProtocolNativeSpecialistEnv.current_requests
    policy_versions=staticmethod(ClosedProtocolNativeSpecialistEnv.policy_versions)
    bind_policy=ClosedProtocolNativeSpecialistEnv.bind_policy
    launch_gate=ClosedProtocolNativeSpecialistEnv.launch_gate

    def __getattr__(self,name):return getattr(self._base,name)

    def __setattr__(self,name,value):
        base=self.__dict__.get("_base")
        if base is not None and name in vars(base):setattr(base,name,value)
        else:object.__setattr__(self,name,value)

    def __init__(self,args,row,*,states,trial_ids,deadline,qpos,qvel,requests,target_input):
        self._base=SharedNativeBase(args,[row],num_envs=len(states.states))
        self.execution_deadline=deadline;self._saved=states;self._trial_ids=tuple(trial_ids)
        self._expected_qpos=np.asarray(qpos,np.float32).copy();self._expected_qvel=np.asarray(qvel,np.float32).copy()
        self._requests=json.loads(json.dumps(requests,allow_nan=False));self._target_input=target_input
        self._bound_policy=None;self.closed_launch_records=[];self.closed_boundary_snapshots=[]
        self.initial_native_state=None;self.restore_receipt=None;self.initial_restore_proof=None;self._reset_done=False
        # All worlds in this native batch use one exact shared reference.
        self.protocol=ClosedProtocolLaunchGuard(self._expected_qpos,self._expected_qvel,self._requests,
            _copy(self.library.launch_ticks)[np.zeros(self.n,dtype=int)],
            _copy(self.library.durations)[np.zeros(self.n,dtype=int)],expected_reference_ids=np.zeros(self.n,dtype=int))

    def reset(self,*,random_phase=True,perturb=True,reference_ids=None,initial_qpos=None,initial_qvel=None,**kwargs):
        if (self._reset_done or random_phase or perturb or kwargs
                or not np.array_equal(np.asarray(reference_ids),np.zeros(self.n,dtype=int))
                or not np.array_equal(np.asarray(initial_qpos,np.float32),self._expected_qpos)
                or not np.array_equal(np.asarray(initial_qvel,np.float32),self._expected_qvel)):
            raise ValueError("Shared comparison permits one exact independent initialization and no subsequent reset")
        self._base.reset(random_phase=False,perturb=False,reference_ids=np.zeros(self.n,dtype=int),
            initial_qpos=self._expected_qpos,initial_qvel=self._expected_qvel)
        self.initial_restore_proof=restore_initial_comparison(self._base,self._saved,
            trial_id="shared:"+fingerprint(self._trial_ids),reference_ids=np.zeros(self.n,dtype=int))
        self.restore_receipt=self.initial_restore_proof.receipt
        self.initial_native_state=copy_native_integration(self.model,self.sim.data,self.n)
        # Install only after strict native/controller restoration. The sampler
        # borrows immutable library tensors and changes no physical history.
        self._base.install_integer_reference_playback()
        self._row_identities=tuple(fingerprint(row) for row in self.library.rows)
        self._cache_versions=self.cache_versions()
        self.protocol.initialize(self.protocol_snapshot());self._reset_done=True
        return self.observe()

    def advance_substep(self,action,*,record=None,stop_on_fall=False):
        self.protocol.before_step(self.protocol_snapshot(),_copy(action))
        self._base.advance_substep(action,record=record,stop_on_fall=stop_on_fall)
        self.protocol.after_step(self.protocol_snapshot(),_copy(action))


class SharedEvaluationProtocol:
    start_mode="shared_jab_saved_state"
    version=VERSION

    def preflight(self,declaration,manifest,catalog,*,limits,mask_decisions,release,route):
        if (manifest.get("schema")!=MANIFEST or manifest.get("declaration_identity")!=declaration.identity
                or manifest.get("initialization_version")!=VERSION or declaration.start_mode!=self.start_mode):
            raise ValueError("Full shared native-state manifest must bind this exact declaration")
        descriptor=manifest["native_start_state_bank"]
        bank_path=owned_file(descriptor["path"],descriptor["sha256"]);bank=load_snapshot(bank_path)
        if bank.identity!=descriptor["identity"]:raise ValueError("Shared native start bank identity differs")
        capture_input=manifest["native_start_state_capture"]
        capture=read_json(capture_input["path"],capture_input["sha256"])
        from . import phase2_initial_capture_v2,phase2_native_bank_geometry_v2
        from .phase2_initial_capture_v2 import validate_capture_allocation
        allocation=validate_capture_allocation(manifest["native_capture_allocation"],capture=capture_input)
        from . import phase2_native_state,phase2_initial_state_v2
        witness_input=manifest["native_initial_identity_witness"]
        witness=validate_witness(witness_input)
        _,base_capture=read_bound(capture["base_capture"])
        if (base_capture.get("schema")!=phase2_native_bank_geometry_v2.VERSION
                or base_capture.get("source_sha256")!=sha256_file(phase2_native_bank_geometry_v2.__file__)
                or any(capture.get(key)!=base_capture.get(key) for key in
                    ("complete_capture","exception","split","domain_sha256","native_start_state_bank",
                     "captured_native_states","physics_steps_executed","actor_updates_executed","slurm_job_id"))
                or capture.get("base_capture_source_sha256")!=sha256_file(phase2_native_bank_geometry_v2.__file__)
                or capture.get("parent_native_adapter_sha256")!=sha256_file(phase2_native_state.__file__)
                or capture.get("initial_identity_consumer_sha256")!=sha256_file(phase2_initial_state_v2.__file__)
                or capture.get("initial_identity_version")!=INITIAL_IDENTITY_VERSION
                or capture.get("initial_identity_witness")!=witness):
            raise ValueError("Versioned capture must preserve its actual unchanged four-world receipt and exact initial-state witness")
        if (capture.get("schema")!=CAPTURE
                or capture.get("complete_capture") is not True or capture.get("exception") is not None
                or not capture.get("slurm_job_id") or capture.get("split")!=declaration.split
                or capture.get("domain_sha256")!=manifest["domain_sha256"]
                or capture.get("native_start_state_bank")!=dict(path=str(bank_path),sha256=descriptor["sha256"],identity=bank.identity)
                or capture.get("source_sha256")!=sha256_file(phase2_initial_capture_v2.__file__)
                or capture.get("captured_native_states")!=4 or capture.get("physics_steps_executed")!=0
                or capture.get("actor_updates_executed")!=0 or capture.get("no_actor_execution") is not True):
            raise ValueError("Actual counted four-world native capture provenance is required")
        if declaration.split=="test" and (not route.get("pre_test_freeze_sha256")
                or capture.get("pre_test_freeze",{}).get("sha256")!=route["pre_test_freeze_sha256"]):
            raise ValueError("Test shared bank was not captured after the bound recipe/domain/split freeze")
        # Initial identity is explicitly versioned; old future-bitwise failures
        # remain failures and never become a successful legacy parity receipt.
        if manifest.get("initial_execution_state_identity_version")!=INITIAL_IDENTITY_VERSION:
            raise ValueError("Explicit reviewed v2 initial-state identity semantics required")
        _,self.jab_paths=family_arguments(release,"jab")
        model=mujoco.MjModel.from_xml_path(str(self.jab_paths["model"]))
        q,v=validate_native_bank(bank,model,sha256_file(self.jab_paths["reference"]))
        entries={entry["trial_id"]:entry for entry in manifest["trials"]}
        if len(entries)!=len(manifest["trials"]) or set(entries)!={b.trial_id for b in declaration.bindings}:
            raise ValueError("Shared manifest must retain every declared request")
        rows={row["candidate_id"]:row for row in catalog["cells"]}
        if len(rows)!=len(catalog["cells"]):raise ValueError("Duplicate shared reference identity")
        starts={};headers={};geometry_banks={};total=0
        for binding in declaration.bindings:
            entry=entries[binding.trial_id];row=rows[entry["candidate_id"]]
            schedule=validate_shared_row(binding,declaration,row)
            if (binding.start_state_sha256!=bank.states[binding.start_index].digest()
                    or entry.get("start_row")!=binding.start_index):
                raise ValueError("Shared trial does not bind its exact full native/controller bank row")
            # Catalogs made before native capture remain explicitly offline
            # candidates; this manifest supplies the later exact native binding.
            if row.get("native_saved_state_available") is True and (
                    row.get("saved_state_sha256")!=descriptor["sha256"] or row.get("saved_state_identity")!=bank.identity
                    or row.get("start_state_sha256")!=binding.start_state_sha256):
                raise ValueError("Shared reference was constructed for a different complete native start")
            geometry_path=owned_file(row["start_bank_path"],row["start_bank_sha256"])
            if geometry_path not in geometry_banks:
                _preflight(_owned_path(geometry_path),row["start_bank_sha256"],1024**2)
                with np.load(geometry_path,allow_pickle=False) as archive:
                    geometry_banks[geometry_path]=(archive["qpos"].copy(),archive["qvel"].copy())
            gq,gv=geometry_banks[geometry_path]
            if not np.array_equal(gq,q) or not np.array_equal(gv,v):
                raise ValueError("Reference geometry bank differs from the actual captured native start")
            failed=validate_negative_reference_identity(row)
            admitted=row.get("kinematic_admission") is True and mask_decisions[binding.trial_id]["admitted"]
            if failed and mask_decisions[binding.trial_id]["admitted"]:
                raise ValueError("Failed shared generation cannot receive execution admission")
            if admitted:
                for name in ("reference","legacy_runtime"):
                    path=_owned_path(row[name+"_path"]);expected=row[name+"_sha256"]
                    if path not in headers:
                        headers[path]=_preflight(path,expected,MAX_REFERENCE_BYTES)
                        total+=headers[path].uncompressed_array_bytes
                        if total>MAX_REFERENCE_BYTES:raise ValueError("Shared reference inputs exceed2GiB")
                    elif headers[path].sha256!=expected:raise ValueError("Conflicting shared reference hash")
                    schema={a.name:a for a in headers[path].arrays};n=schema["qpos"].shape[0]
                    if (not 3<=n<=limits.maximum_reference_frames or schema["qpos"].shape!=(n,55)
                            or headers[path].uncompressed_array_bytes*3+n*4096>limits.resident_reference_bytes):
                        raise ValueError("Shared reference frame/resident chunk bound differs")
                    with np.load(path,allow_pickle=False) as archive:
                        clock=np.asarray(archive["t"],float)
                        if (clock.shape!=(n,) or clock[0]!=0 or not np.allclose(np.diff(clock),.005,atol=1.e-12,rtol=0)
                                or abs(clock[-1]-row["duration_s"])>1.e-9
                                or not np.array_equal(archive["head_center"],binding.target_w)
                                or int(archive["active_hand_index"])!=(0 if declaration.family in ("jab","left_hook") else 1)
                                or not np.array_equal(archive["qpos"][0],q[binding.start_index])
                                or not np.array_equal(archive["qvel"][0],v[binding.start_index])
                                or json.loads(str(archive["phase_schedule_json"]))!=asdict(schedule)):
                            raise ValueError("Shared reference start, derivatives, target, hand or phase schedule differs")
            starts[binding.trial_id]=(q[binding.start_index].copy(),v[binding.start_index].copy())
        self.bank=bank;self.entries=entries;self.bindings={b.trial_id:b for b in declaration.bindings}
        self.metadata=dict(native_start_state_bank=dict(path=str(bank_path),sha256=descriptor["sha256"],identity=bank.identity),
            native_start_state_capture=dict(path=str(owned_file(capture_input["path"])),sha256=capture_input["sha256"]),
            native_initial_identity_witness=witness,initial_execution_state_identity_version=INITIAL_IDENTITY_VERSION,
            native_capture_allocation=allocation,
            future_bitwise_parity_required=False,future_solver_drift_diagnostics_mandatory=True,solver_cause="unknown",
            shared_physical_controller_identity=bank.physical_controller_identity,
            playback_version=PLAYBACK_VERSION,preparation_outward_recovery_continuous=True,
            no_within_cycle_restore=True,support_initialization="Native initial forward/support classification at timezero; no executed interval is invented")
        return entries,rows,starts,dict(unique_reference_uncompressed_bytes=total,unique_reference_archives=len(headers),
            resident_reference_limit_bytes=limits.resident_reference_bytes,initialization_version=VERSION,**self.metadata)

    def validate_starts(self,declaration,domain,starts):
        from .phase2_shared_catalog_v2 import validate_start_bank
        with np.load(self.jab_paths["reference"],allow_pickle=False) as archive:nominal=archive["qpos"][0].copy()
        metadata=read_json(self.jab_paths["body_metadata"])
        q=[];v=[]
        for index in range(4):
            binding=next(b for b in declaration.bindings if b.start_index==index)
            pose,velocity=starts[binding.trial_id];q.append(pose);v.append(velocity)
        validate_start_bank(np.stack(q),np.stack(v),nominal,metadata,domain,declaration.split)

    def environment(self,args,row,bindings,*,deadline,qpos,qvel,requests,target_input):
        selected=NativeComparisonSnapshot(tuple(self.bank.states[b.start_index] for b in bindings),self.bank.inventory_json)
        return SharedProtocolEnv(args,row,states=selected,trial_ids=[b.trial_id for b in bindings],deadline=deadline,
            qpos=qpos,qvel=qvel,requests=requests,target_input=target_input)

    def audit_initialization(self,folder,env,bindings,qpos,qvel):
        path=folder/"raw_trace.npz";checks={}
        expected=np.stack([self.bank.states[b.start_index].native_integration_state for b in bindings])
        checks["full_native_integration_equal"]=env.initial_native_state is not None and np.array_equal(env.initial_native_state,expected)
        if path.exists():
            with np.load(path,allow_pickle=False) as raw:
                checks.update(qpos=np.array_equal(raw["qpos"][0],qpos),qvel=np.array_equal(raw["qvel"][0],qvel),
                    previous_raw_action=bool("policy_previous_raw_action" in raw and
                        np.array_equal(raw["policy_previous_raw_action"][0],np.stack([self.bank.states[b.start_index].previous_raw_action for b in bindings]))),
                    no_within_cycle_reset=bool(np.all(raw["reset_count"]==1)),reference_clock_zero=bool(not np.any(raw["reference_tick"][0])),
                    initial_ctrl_zero=bool(not np.any(raw["ctrl"][0])))
        else:checks["raw_trace_present"]=False
        state_path=folder/"restored_native_initialization.npz"
        with state_path.open("xb") as stream:np.savez_compressed(stream,native_integration_state=env.initial_native_state if env.initial_native_state is not None else np.empty((0,410)))
        proof=env.initial_restore_proof
        checks["exact_initial_identity_before_and_after_refresh"]=bool(proof is not None
            and proof.receipt.get("initial_state_identity_passed") is True
            and proof.receipt.get("exact_after_declared_forward_refresh") is True
            and proof.before.physical_controller_identity==env._saved.physical_controller_identity
            and proof.after.physical_controller_identity==env._saved.physical_controller_identity)
        from .phase2_native_state import save_snapshot
        retained={}
        if proof is not None:
            for label,state in (("before_refresh",proof.before),("after_refresh",proof.after)):
                saved=save_snapshot(state,folder/("initial_identity_"+label))
                retained[label]=dict(path=str(saved),sha256=sha256_file(saved),identity=state.identity)
        derived_path=folder/"initial_identity_derived_refresh.npz"
        differences={}
        if proof is not None:
            payload={label+"__"+key:value for label,arrays in
                (("before",proof.derived_before),("after",proof.derived_after)) for key,value in arrays.items()}
            with derived_path.open("xb") as stream:np.savez_compressed(stream,**payload)
            for key in proof.derived_before:
                before,after=proof.derived_before[key],proof.derived_after[key]
                differences[key]=dict(same_shape=before.shape==after.shape,
                    exact_equal=before.shape==after.shape and before.dtype==after.dtype and before.tobytes()==after.tobytes(),
                    maximum_absolute_difference=float(np.max(np.abs(before.astype(float)-after.astype(float)),initial=0)) if before.shape==after.shape else None)
        report=dict(schema="task10b_shared_native_initialization_audit_v2",passed=all(checks.values()),checks=checks,
            initial_identity_version=INITIAL_IDENTITY_VERSION,initial_identity_witness=self.metadata["native_initial_identity_witness"],
            retained_initial_identity_states=retained,future_bitwise_parity_qualified=False,solver_cause="unknown",
            derived_refresh_evidence=(dict(path=str(derived_path),sha256=sha256_file(derived_path)) if proof is not None else None),
            derived_refresh_differences=differences,derived_differences_are_diagnostic=True,
            start_indices=[b.start_index for b in bindings],trial_ids=[b.trial_id for b in bindings],
            start_state_sha256=[b.start_state_sha256 for b in bindings],bank=self.metadata["native_start_state_bank"],
            retained_native_state=dict(path=str(state_path),sha256=sha256_file(state_path)),restore_receipt=env.restore_receipt,
            full_saved_state_restoration=True,independent_trace_audit_required=True)
        write_json(folder/"restored_native_initialization.json",report)
        return report

    def outcome_metadata(self,trial_id):
        binding=self.bindings[trial_id]
        return dict(start_mode=self.start_mode,shared_start_index=binding.start_index,
            shared_start_state_sha256=binding.start_state_sha256,
            shared_physical_controller_identity=self.bank.physical_controller_identity,
            preparation_clock_includes_connector=True,within_cycle_reset=False)


def execute(*args,**kwargs):
    return execute_expanded(*args,**kwargs,protocol=SharedEvaluationProtocol())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("declaration","manifest","route","checkpoint","training","predicate","mask","output","release"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--wall-seconds",type=float,required=True);parser.add_argument("--maximum-worlds",type=int,default=16)
    parser.add_argument("--continuation",type=Path)
    options=parser.parse_args()
    result=execute(options.declaration,options.manifest,options.route,options.checkpoint,options.training,options.predicate,
        options.mask,options.output,release=options.release,limits=EvaluationLimits(options.wall_seconds,maximum_worlds=options.maximum_worlds),
        continuation=options.continuation)
    print(json.dumps(result,sort_keys=True));return 0 if result["status"]=="executed_pending_audit_and_video" else 1


if __name__=="__main__":raise SystemExit(main())
