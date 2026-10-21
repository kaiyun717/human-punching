"""CPU declaration/negative-ledger/first-write tests; no native model execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from human_punching import phase2_family_target_control_v1 as c


def fixture():
    entries=[];cells=[];rows=[]
    for group in (4184,4251):
        for timing in (.6,.75):
            cid=f'cross_{group}_{timing}'
            target=dict(command_id=group,candidate_identity=f't{group}',head_target_w=[.8,.2,1.2],
                payload_sha256='a'*64,snapshot_slot_id=group,placement={},head_point_velocity_w=[0.,0.,0.])
            entry=dict(candidate_id=cid,target_candidate_identity=target['candidate_identity'],parent_candidate_id=f'p{timing}')
            parent=dict(requested_ttc_s=timing,episode_duration_s=5.,phase_denominator_s=5.,launch_tick=40,outward_end_tick=400)
            row=dict(**entry,family='cross',split='train',command_id=group,requested_ttc_s=timing,
                target_w=target['head_target_w'],generation_disposition='generated',kinematic_admission=True,
                rejection_reasons=[],reference_sha256='b'*64,reference_identity_kind='reference_content',
                start_mode='family_guard',preparation_complete_tick=0,
                **{k:parent[k] for k in ('episode_duration_s','phase_denominator_s','launch_tick','outward_end_tick')})
            entries.append(entry);cells.append((entry,target,parent,None,None));rows.append(row)
    source=dict(family='cross',cells=entries)
    catalog=dict(cells=rows,unattempted_cells=[],planned_cells=4,executed_cells=4,unpacked_catalog_sha256='c'*64)
    return source,catalog,cells


def plan_for(rows,cells):
    q=np.zeros((8,55),dtype=np.float32);q[:,3]=1;v=np.zeros((8,54),dtype=np.float32)
    requests=[request for row,(_,target,_,_,_) in zip(rows,cells,strict=True)
              for request in c.requests_for(row,target,'d'*64,q,v)]
    return dict(requests=requests,reference_dispositions=[dict(candidate_id=r['candidate_id'],
        generation_disposition=r['generation_disposition'],native_execution_admitted=r['native_execution_admitted'],
        rejection_reasons=r['rejection_reasons']) for r in rows],
        video_selection={'presampled_trial_ids':[requests[0]['trial_id']]})


def test_retained_cross_geometry_failure_is16_zero_execution_trials():
    source,catalog,cells=fixture()
    for row in catalog['cells'][:2]:
        row.update(generation_disposition='failed',kinematic_admission=False,
            rejection_reasons=['target_reference_generation_failure'],reference_identity_kind='generation_failure_receipt')
    rows=c.declared_rows(catalog,source,cells);plan=plan_for(rows,cells);ledger=c.trial_ledger(plan)
    assert len(ledger)==32 and sum(r['native_execution_admitted'] for r in ledger)==16
    assert all(r['status']=='reference_unavailable' and r['actual_physics_steps']==0
               and not r['execution_complete'] and not r['numeric_complete_cycle'] for r in ledger[:16])
    assert all(r['status']=='unattempted' for r in ledger[16:])
    assert len({r['request_identity'] for r in plan['requests']})==32
    assert {r['family'] for r in plan['requests']}=={'cross'}


def test_timeout_and_unattempted_are_retained_without_fake_reference():
    source,catalog,cells=fixture();catalog['cells']=catalog['cells'][:1]
    catalog['cells'][0].update(generation_disposition='timeout',kinematic_admission=False,
        rejection_reasons=['bounded_timeout'],reference_identity_kind='generation_failure_receipt')
    catalog['executed_cells']=1;catalog['unattempted_cells']=source['cells'][1:]
    rows=c.declared_rows(catalog,source,cells);ledger=c.trial_ledger(plan_for(rows,cells))
    assert len(rows)==4 and len(ledger)==32 and not any(r['native_execution_admitted'] for r in ledger)
    assert rows[1]['reference_identity_kind']=='unattempted_declaration' and 'reference_path' not in rows[1]
    assert all(r['actual_physics_steps']==0 for r in ledger)


@pytest.mark.parametrize('mutation',['drop','duplicate','overlap','wrong_target','wrong_family','wrong_bin','missing_reason','fake_admission'])
def test_no_dropped_negative_or_changed_request(mutation):
    source,catalog,cells=fixture()
    if mutation=='drop':catalog['cells'].pop();catalog['executed_cells']=3
    elif mutation=='duplicate':catalog['cells'][1]=deepcopy(catalog['cells'][0])
    elif mutation=='overlap':catalog['unattempted_cells']=[source['cells'][0]]
    elif mutation=='wrong_target':catalog['cells'][0]['target_w']=[0,0,0]
    elif mutation=='wrong_family':catalog['cells'][0]['family']='jab'
    elif mutation=='wrong_bin':catalog['cells'][0]['requested_ttc_s']=.675
    elif mutation=='missing_reason':catalog['cells'][0].update(generation_disposition='failed',kinematic_admission=False)
    else:catalog['cells'][0]['reference_identity_kind']='generation_failure_receipt'
    with pytest.raises(ValueError):c.declared_rows(catalog,source,cells)


def test_kinematic_rejected_generated_reference_cannot_execute():
    source,catalog,cells=fixture();catalog['cells'][0].update(kinematic_admission=False,rejection_reasons=['early_contact'])
    rows=c.declared_rows(catalog,source,cells)
    assert not rows[0]['native_execution_admitted'] and rows[0]['generation_disposition']=='generated'


def test_event_clock_scope_calls_clockv2_once_restores_on_failure(monkeypatch,tmp_path):
    calls=[];writes=[]
    class Metadata:
        def __init__(self,*args):self.ancestry={}
        def before_write(self,path,value):value.update(preparation_complete_time_s=0.);return value
    monkeypatch.setattr(c,'EventMetadata',Metadata)
    def writer(path,value):writes.append((path,value.copy()))
    def evaluator(env,policy,output,**kwargs):
        calls.append(kwargs);c.route.write_json(output/'trial_000.json',{'preparation_complete_time_s':.4});raise RuntimeError('saved evidence failure')
    monkeypatch.setattr(c.route,'write_json',writer);monkeypatch.setattr(c.route,'evaluate_policy',evaluator)
    with pytest.raises(RuntimeError,match='saved evidence'):
        with c.event_clock_scope():
            c.route.evaluate_policy(None,None,tmp_path,reference_ids=[0],trial_ids=['t'])
    assert len(calls)==1 and writes[0][1]['preparation_complete_time_s']==0.
    assert c.route.evaluate_policy is evaluator and c.route.write_json is writer


def test_actual_bank_states_preserve_native_float32_identity(tmp_path):
    q=np.zeros((8,55));q[:,3]=1;q[:,7]=np.arange(8)*.001;v=np.zeros((8,54))
    p=tmp_path/'bank.npz';np.savez(p,qpos=q,qvel=v)
    actual_q,actual_v=c.bank_states({'bank':p})
    assert actual_q.dtype==actual_v.dtype==np.float32
    assert np.array_equal(actual_q,q.astype(np.float32))


def driver():
    p=c.PROJECT_ROOT/'scripts/task10b_phase2_family_target_control_v1.py'
    spec=importlib.util.spec_from_file_location('family_control_test_driver',p)
    d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d);return d


def record():
    return dict(mode='family-target-control',inputs={'plan':'tests/test_phase2_family_target_control_v1.py'},
        application_wall_seconds=300,allocation_wall_seconds=360,target_control_version=c.VERSION,
        target_control_trials=32,target_control_cells=4,target_control_child_wall_seconds=210,
        target_control_environments=8,target_control_native_attempt_capacity=16,fresh_transition_cap=0,physical_qualification=False,output='outputs/task10b/future/run')


@pytest.mark.parametrize('remaining',[None,270,240,210])
def test_exact210_child_never_uses_hoped_for_early_exit(remaining):
    d=driver();cmd=d.native_command(record(),remaining)
    assert cmd[2:4]==['-m','human_punching.phase2_family_target_control_v1'] and cmd[-1]=='210.0'


@pytest.mark.parametrize('remaining',[209.99,True,float('nan'),271])
def test_late_setup_refuses_truncation(remaining):
    with pytest.raises(RuntimeError):driver().native_command(record(),remaining)


def test_wrong_count_cannot_launch():
    r=record();r['target_control_trials']=16
    with pytest.raises(ValueError):driver().native_command(r)


def test_clockv2_fixed_boundary_padding_horizon():
    source,catalog,cells=fixture();catalog['cells'][0]['episode_duration_s']=5.005
    value=c.horizons(catalog['cells'])[catalog['cells'][0]['candidate_id']]
    assert value['planned_physics_steps']==1004 and value['phase_denominator_s']==5.


def test_zero_admitted_gpu_packet_refused():
    r=record();r['target_control_native_attempt_capacity']=0
    with pytest.raises(ValueError):driver().native_command(r)


def test_all_negative_family_materializes32_without_native(monkeypatch,tmp_path):
    import json
    source,catalog,cells=fixture()
    for row in catalog['cells']:
        row.update(generation_disposition='failed',kinematic_admission=False,
            rejection_reasons=['target_reference_generation_failure'],reference_identity_kind='generation_failure_receipt')
    plan=plan_for(c.declared_rows(catalog,source,cells),cells)
    plan.update(native_attempt_capacity=0,admitted_candidate_ids=[],scope='negative fixture')
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    monkeypatch.setattr(c,'validate_plan',lambda p:None)
    monkeypatch.setattr(c,'writable_path',lambda p:Path(p))
    result=c.materialize_unavailable(path,tmp_path/'negative')
    assert result['native_processes_started']==result['physics_steps']==result['gpus']==0
    assert result['reference_unavailable_trials']==32 and result['completed_trials']==0
    saved=json.loads((tmp_path/'negative/outcomes.json').read_text())
    assert len(saved)==32 and not any(r['numeric_complete_cycle'] for r in saved)
    plan['native_attempt_capacity']=8;path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):c.materialize_unavailable(path,tmp_path/'invalid')
