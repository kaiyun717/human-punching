"""Saved-data metadata and failure-accounting checks; no native execution."""
from copy import deepcopy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from human_punching import phase2_target_control_v1 as control

ROOT=control.PROJECT_ROOT


@pytest.fixture(scope='module')
def plan():
    return control.create_plan('outputs/task10b/p2sequentialcatalog02/manifest.json',
        'outputs/task10b/p2sequentialaudit02/audit.json')


@pytest.fixture(scope='module')
def driver():
    spec=importlib.util.spec_from_file_location('test_target_control_driver',ROOT/'scripts/task10b_phase2_target_control_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_real_32_exact_training_cells_and_closed_initial_states(plan):
    data=control.validate_plan(plan)
    assert plan['selected_candidate_ids']==control.IDS and plan['planned_trials']==32
    assert len({r['trial_id'] for r in plan['requests']})==32
    assert len({r['request_identity'] for r in plan['requests']})==32
    assert {r['requested_ttc_s'] for r in plan['requests']}=={.6,.75}
    assert len(plan['initial_state_identities'])==8
    assert all(r['split']=='train' and r['command_id'] in (4184,4251) for r in data[3])
    assert plan['no_test_access'] and plan['fresh_transition_cap']==0 and not plan['physical_qualification']
    q,v=control.bank_states(data[7])
    assert q.dtype==np.float32 and v.dtype==np.float32
    with np.load(data[7]['bank'],allow_pickle=False) as bank:
        assert np.array_equal(q,bank['qpos'][:8].astype(np.float32))
        assert np.array_equal(v,bank['qvel'][:8].astype(np.float32))
    for row in data[3]:
        horizon=plan['horizons'][row['candidate_id']]
        assert horizon['launch_tick']%4==0
        assert horizon['planned_physics_steps']*.005>=row['episode_duration_s']-1e-12
        assert horizon['planned_physics_steps']*.005<row['episode_duration_s']+.02+1e-12


@pytest.mark.parametrize('change',[
    lambda p:p.update(planned_trials=31),lambda p:p.update(environments=16),
    lambda p:p.update(child_wall_seconds=241),lambda p:p.update(training=True),
    lambda p:p.update(fresh_transition_cap=8192),lambda p:p.update(no_test_access=False),
    lambda p:p['requests'][0].update(requested_ttc_s=.675),
    lambda p:p['requests'][0]['target_w'].__setitem__(0,99.),
    lambda p:p['requests'][0].update(checkpoint_sha256='0'*64),
    lambda p:p['initial_state_identities'].__setitem__(0,'0'*64),
    lambda p:p['horizons'][control.IDS[0]].update(launch_tick=0),
    lambda p:p['selected_candidate_ids'].reverse(),
])
def test_no_request_domain_checkpoint_or_budget_substitution(plan,change):
    changed=deepcopy(plan);change(changed)
    with pytest.raises(ValueError):control.validate_plan(changed)


def test_snapshot_callback_preserves_fixed_input_not_mutable_caller(plan):
    requests=deepcopy(plan['requests'][:8]);callback=control.FrozenSnapshotInput(requests)
    original=requests[0]['target_w'].copy();requests[0]['target_w'][0]+=10.
    assert callback(100)[0]['target_w']==original
    assert callback(100)[0]['capture_episode_tick']==100
    with pytest.raises(ValueError):callback(101)


def test_first_attempt_and_unknown_constructor_preserve_all32(plan):
    ledger=control.trial_ledger(plan)
    for row in ledger[:8]:row.update(status='execution_extent_unknown',execution_extent='constructor_or_native_attempt_started')
    assert len(ledger)==32 and sum(r['status']=='execution_extent_unknown' for r in ledger)==8
    assert sum(r['status']=='unattempted' for r in ledger)==24
    assert not any(r['execution_complete'] or r['numeric_complete_cycle'] for r in ledger)


@pytest.mark.parametrize('steps,status,ready,init,credit',[
    (12,'executed',True,True,True),(11,'failed',True,True,False),
    (12,'failed',True,True,False),(12,'executed',False,True,False),
    (12,'executed',True,False,False),
])
def test_saved_physical_extent_separate_from_audit_and_launch_credit(plan,steps,status,ready,init,credit):
    row=control.trial_ledger(plan)[0];ledger={row['trial_id']:row}
    panel=dict(status=status,trials=[dict(trial_id=row['trial_id'],disposition='executed',numeric_complete_cycle=True)])
    control.record_panel(ledger,panel,[dict(trial_id=row['trial_id'],accepted=ready)],
        initialization_passed=init,actual_steps=steps,planned_steps=12)
    assert row['execution_complete']==(steps==12) and row['numeric_complete_cycle']==credit
    assert row['actual_physics_steps']==steps


def test_all_failed_and_unattempted_media_remain_visible(plan):
    ledger=control.trial_ledger(plan);selection=control.media_selection(plan,ledger)
    assert len(selection['planned_required_trial_ids'])==32
    for row in ledger:row['numeric_complete_cycle']=True
    assert control.media_selection(plan,ledger)['planned_required_trial_ids']==sorted(plan['video_selection']['presampled_trial_ids'])
    ledger[0]['numeric_complete_cycle']=False
    assert ledger[0]['trial_id'] in control.media_selection(plan,ledger)['planned_required_trial_ids']


def test_exact_driver_wall_and_late_setup_refusal(plan,driver,tmp_path):
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));inputs={'plan':str(path)}
    scope=driver.validate_spec(driver.MODE,inputs,360,300)
    record=dict(mode=driver.MODE,inputs=inputs,output='outputs/task10b/synthetic_not_run',
        allocation_wall_seconds=360,application_wall_seconds=300,**scope)
    assert driver.native_command(record,240)[-1]=='240.0'
    for remaining in (239.99,270.01,float('nan'),True):
        with pytest.raises(RuntimeError):driver.native_command(record,remaining)
    with pytest.raises(ValueError):driver.validate_spec(driver.MODE,inputs,300,240)
    assert scope['target_control_environments']==8 and scope['fresh_transition_cap']==0


def test_projection_and_actual_source_closure_are_exact(plan,driver,tmp_path):
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));closure=control.files(path)
    assert closure['outputs/task10b/p2sequentialcatalog02/manifest.json']==plan['catalog']['sha256']
    assert 'outputs/task10b/p2sq02/source_snapshot/src/human_punching/phase2_target_path_reference_v3.py' in closure
    assert not any('gridsan' in p for p in closure)
    record=dict(mode=driver.MODE,inputs={'plan':str(path)},bundle='outputs/task10/release_baseline_02',
        release='outputs/four-punch-milestone',predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json',
        successful_route_freeze=None,runtime_namespace_proof=plan['namespace_proof'])
    from human_punching.phase2_runtime_namespace import evidence
    record['runtime_namespace_evidence']=evidence(ROOT,ROOT/plan['namespace_proof']['path'])
    original=driver.legacy.semantic_files
    result=driver.required_inputs(record)
    assert driver.legacy.semantic_files is original
    assert all(result[k]==v for k,v in closure.items())
    assert 'src/human_punching/phase2_sequential_projection_v1.py' in result


def test_no_cuda_or_output_on_unallocated_execute(plan,tmp_path,monkeypatch):
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='Counted Slurm'):control.execute(path,tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_constructor_failure_saved_without_native_steps(plan,tmp_path,monkeypatch):
    # Synthetic allocation and startup functions only. The environment fails
    # before construction; no model, CUDA context, physics or actor is created.
    import torch
    import warp
    from human_punching.wbc.telemetry import cycle_qualification
    monkeypatch.setenv('SLURM_JOB_ID','synthetic_unit_test')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(warp,'init',lambda:None)
    monkeypatch.setattr(cycle_qualification,'native_stream_scope',nullcontext)
    monkeypatch.setattr(control,'runtime_record',lambda:dict(declared_pins_match=True))
    def fail(*args,**kwargs):raise RuntimeError('synthetic constructor failure before native integration')
    monkeypatch.setattr(control,'ClosedProtocolNativeSpecialistEnv',fail)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));result=control.execute(path,tmp_path/'run')
    ledger=json.loads((tmp_path/'run/outcomes.json').read_text())
    assert result['status']=='failed' and result['completed_trials']==0 and result['fresh_transitions']==0
    assert result['panels']==[] and len(ledger)==32
    assert sum(r['status']=='execution_extent_unknown' for r in ledger)==8
    assert sum(r['status']=='unattempted' for r in ledger)==24
    assert (tmp_path/'run/attempt_0000.json').exists()
    assert not any(p.name=='raw_trace.npz' for p in (tmp_path/'run').rglob('*'))
