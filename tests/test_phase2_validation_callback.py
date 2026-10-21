"""Synthetic held-out callback integration; no native rollout or GPU work."""
from contextlib import nullcontext
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from human_punching import phase2_validation_callback as callback_module
from human_punching.phase2_validation_callback import GroupedValidationCallback
from human_punching.phase2_domain import fingerprint
from human_punching.phase2_grouped_learning import validate_callback_panel


def configured_callback(tmp_path,monkeypatch,*,failed=False,mutate=False):
    from human_punching.wbc.telemetry import cycle_qualification
    monkeypatch.setattr(cycle_qualification,"native_stream_scope",nullcontext)
    monkeypatch.setenv("SLURM_JOB_ID","synthetic_cpu_interface_test")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda:None)
    monkeypatch.setattr(callback_module.gc,"collect",lambda:0)
    monkeypatch.setattr(callback_module,"asdict",lambda value:dict(synthetic_physical_request_template=True))
    groups=tuple(range(8));bindings=[]
    for g in groups:
        for timing in (.6,.675,.75):
            for start in range(4):
                trial=f"g{g}_t{timing}_s{start}"
                bindings.append(SimpleNamespace(trial_id=trial,command_id=g,ttc_s=timing,start_index=start))
    descriptor=dict(command_groups=groups,trained_timings_s=[.6,.75],interpolation_timing_s=.675,
        trial_ids=[b.trial_id for b in bindings])
    callback=GroupedValidationCallback();callback.output=tmp_path
    callback.descriptor=descriptor;callback.declaration=SimpleNamespace(bindings=bindings,identity="template",seed=21021)
    callback.entries={};callback.rows={};callback.catalog=dict(cells=[])
    callback.limits=SimpleNamespace(maximum_worlds=16,resident_reference_bytes=256*1024**2)
    callback.starts={b.trial_id:(np.zeros(55,np.float32),np.zeros(54,np.float32)) for b in bindings}
    callback.decisions={b.trial_id:dict(admitted=True) for b in bindings}
    callback.args=SimpleNamespace(model="no_model_execution");callback.paths={};callback.targets={};callback.predicate={}
    callback.source_configuration=dict(path="synthetic",sha256="a"*64)
    callback.training_declaration=SimpleNamespace(recipe_sha256="b"*64,domain_sha256="c"*64,target_geometry_sha256="d"*64)
    callback.manifest=dict(domain_path="synthetic_domain",domain_sha256="c"*64,
        target_geometry_path="synthetic_geometry",target_geometry_sha256="d"*64)
    batches=[(dict(candidate_id=f"candidate_{i}",kinematic_admission=True),tuple(bindings[i:i+4])) for i in range(0,96,4)]
    monkeypatch.setattr(callback_module,"batch_requests",lambda *args:batches)
    monkeypatch.setattr(callback_module,"launch_requests",lambda declaration,bindings,*args:[dict(trial_id=b.trial_id) for b in bindings])
    monkeypatch.setattr(callback_module,"runtime_record",lambda:dict(synthetic=True))
    monkeypatch.setattr(callback_module,"audit_initialization",lambda *args:dict(passed=True))
    monkeypatch.setattr(callback_module,"export_runtime_model",lambda source,path,model:path.write_text("synthetic"))
    monkeypatch.setattr(callback_module,"write_launch_boundaries",lambda *args:dict(synthetic=True))
    bound=[]
    class FakeEnv:
        def __init__(self,*args,requests,**kwargs):
            self.library=SimpleNamespace(bytes=100);self.model=None;self.hard_faults=[]
            self.closed_launch_records=[dict(trial_id=r["trial_id"],accepted=True) for r in requests]
            self.closed_boundary_snapshots=[]
        def bind_policy(self,policy):bound.append(policy)
    monkeypatch.setattr(callback_module,"ClosedProtocolNativeSpecialistEnv",FakeEnv)
    def evaluate(env,policy,folder,*,trial_ids,**kwargs):
        folder.mkdir();np.savez(folder/"raw_trace.npz",dummy=np.zeros(1))
        if mutate:
            with torch.no_grad():next(policy.parameters()).add_(1)
        faults=[dict(faults={"fall":[0]})] if failed else []
        env.hard_faults=faults
        return dict(status="failed" if failed else "executed",hard_faults=faults,
            exception=dict(type="PhysicalStop",message="fall") if failed else None,
            actual_physics_steps=20,actual_policy_updates=5,
            trials=[] if failed else [dict(trial_id=trial,numeric_complete_cycle=True,
                timing=dict(timing_pass=True),outward_active_glove_rms_m=.01) for trial in trial_ids])
    monkeypatch.setattr(callback_module,"evaluate_policy",evaluate)
    deadline=SimpleNamespace(checkpoint=lambda *args,**kwargs:None)
    return callback,deadline,bound


def test_callback_retains_all96_requests_and_actual_policy_identity(tmp_path,monkeypatch):
    callback,deadline,bound=configured_callback(tmp_path,monkeypatch)
    policy=torch.nn.Linear(2,2)
    result=callback(policy,0,deadline=deadline)
    validate_callback_panel(result,callback.descriptor)
    assert result["status"]=="executed" and result["completed_trials"]==96 and len(bound)==24
    assert result["actual_completed_fresh_transitions"]==0 and result["actor_unchanged"]
    assert not result["learned_qualification"] and result["checkpoint_template_is_unbound"]
    assert (tmp_path/"validation_00000/training/checkpoint_in_memory.pt").exists()
    receipt=json.loads((tmp_path/'validation_00000/batch_0000/closed_protocol_launch.json').read_text())
    assert receipt['initialization_version']==callback_module.INITIALIZATION_VERSION
    with pytest.raises(FileExistsError):callback(policy,0,deadline=deadline)


def test_initial_hard_fall_preserves_partial_and_unattempted_denominator(tmp_path,monkeypatch):
    callback,deadline,bound=configured_callback(tmp_path,monkeypatch,failed=True)
    result=callback(torch.nn.Linear(2,2),0,deadline=deadline)
    validate_callback_panel(result,callback.descriptor)
    assert result["status"]=="failed" and result["completed_trials"]==0 and len(bound)==1
    assert len(result["trials"])==96 and result["hard_faults"]
    assert sum(r["disposition"]=="incomplete_native_failure" for r in result["trials"])==4
    assert sum(r["disposition"]=="unattempted" for r in result["trials"])==92


def test_mutated_policy_and_wrong_cadence_cannot_supply_validation_success(tmp_path,monkeypatch):
    callback,deadline,_=configured_callback(tmp_path,monkeypatch,mutate=True)
    policy=torch.nn.Linear(2,2)
    with pytest.raises(ValueError,match="cadence"):callback(policy,1,deadline=deadline)
    result=callback(policy,8,deadline=deadline)
    assert result["status"]=="failed" and result["exception"]["type"]=="PolicyMutation"
    assert not result["actor_unchanged"] and not result["learned_qualification"]


def test_generation_failures_remain_in_physical_panel_denominator(tmp_path,monkeypatch):
    callback,deadline,_=configured_callback(tmp_path,monkeypatch)
    original=callback_module.batch_requests
    batches=original(None,None,None,None)
    batches[0][0]['reference_identity_kind']='generation_failure_receipt'
    for binding in batches[0][1]:callback.decisions[binding.trial_id]=dict(admitted=False,reason='preserved_generation_failure')
    result=callback(torch.nn.Linear(2,2),0,deadline=deadline)
    assert result['status']=='failed' and result['completed_trials']==92 and len(result['trials'])==96
    failures=[r for r in result['trials'] if r['disposition']=='generation_failure']
    assert len(failures)==4 and all(r['physical_attempt'] is False for r in failures)


def test_direct_callback_requires_counted_allocation_before_creating_output(tmp_path,monkeypatch):
    callback,deadline,_=configured_callback(tmp_path,monkeypatch)
    monkeypatch.delenv('SLURM_JOB_ID')
    with pytest.raises(RuntimeError,match='counted Slurm'):
        callback(torch.nn.Linear(2,2),0,deadline=deadline)
    assert not (tmp_path/'validation_00000').exists()


def test_explicit_shared_protocol_strategy_controls_restore_and_initialization_audit(tmp_path,monkeypatch):
    callback,deadline,bound=configured_callback(tmp_path,monkeypatch)
    base_environment=callback.make_environment;calls=[]
    class Protocol:
        version='synthetic_shared_initialization_version'
        metadata=dict(full_saved_state_restore_required=True,synthetic_interface_only=True)
        def environment(self,args,row,bindings,**kwargs):
            calls.append(('restore',len(bindings)))
            return base_environment(row,bindings,**kwargs)
        def audit_initialization(self,folder,env,bindings,qpos,qvel):
            calls.append(('audit',len(bindings)));return dict(passed=True,synthetic_full_state_strategy=True)
    # Use the captured base bound method with its isolated strategy to avoid
    # recursively redispatching through this test protocol.
    base_class=callback_module.ClosedProtocolNativeSpecialistEnv
    def environment(args,row,bindings,**kwargs):
        calls.append(('restore',len(bindings)))
        return base_class(args,[row],num_envs=len(bindings),expected_qpos=kwargs['qpos'],expected_qvel=kwargs['qvel'],
            deadline=kwargs['deadline'],requests=kwargs['requests'],target_input=kwargs['target_input'])
    protocol=Protocol();protocol.environment=environment;callback.validation_protocol=protocol
    result=callback(torch.nn.Linear(2,2),0,deadline=deadline)
    assert result['status']=='executed' and len(bound)==24
    assert sum(kind=='restore' for kind,_ in calls)==24 and sum(kind=='audit' for kind,_ in calls)==24
    receipt=json.loads((tmp_path/'validation_00000/batch_0000/closed_protocol_launch.json').read_text())
    assert receipt['initialization_version']==protocol.version
