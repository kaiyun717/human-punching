"""Finite learning-curve selection and regression gates; no native models."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest
from human_punching import phase2_timing_pilot_curve_v2 as curve


def panel():
    return {'trials': [dict(candidate_id=c, trial_id=f'{c}_bank_{i:02d}',
                           numeric_complete_cycle=True)
                      for c in curve.CONTROLS for i in range(8)]}


def test_regression_requires_every_matched_control_and_keeps_bad_bin_separate():
    value=panel()
    value['trials'].append(dict(candidate_id='cross_nonuniform_postcontact_0p750',
                               trial_id='cross_nonuniform_postcontact_0p750_bank_00',
                               numeric_complete_cycle=False))
    assert curve.control_regression(value)['passed']
    value['trials'][3]['numeric_complete_cycle']=False
    result=curve.control_regression(value)
    assert not result['passed']
    assert result['checks']['cross_uniform_0p600']
    assert result['new_physical_fault_claim'] is False


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'renamed', 'integer_success', 'absent_success'])
def test_invalid_control_identity_does_not_pass(mutation):
    value=panel()
    if mutation=='missing':value['trials'].pop()
    elif mutation=='duplicate':value['trials'][-1]=value['trials'][-2].copy()
    elif mutation=='renamed':value['trials'][0]['trial_id']='another_bank'
    elif mutation=='integer_success':value['trials'][0]['numeric_complete_cycle']=1
    else:del value['trials'][0]['numeric_complete_cycle']
    assert not curve.control_regression(value)['passed']


@pytest.fixture
def valid_plan(monkeypatch):
    numerical = dict(path='numbers', sha256='f1a8bf7ca8f3ba0effec20a3b4f97a78f1274c4f0fcbb83c536c9eb4ede06465')
    media = [dict(update=u,packet=dict(path=f'media{u}',sha256='mock')) for u in [0,8,16]]
    evidence = {'parent': {'family':'cross', 'training_candidate_ids':curve.SELECTED,
                           'sources':{}, 'stopping_rules':{'validation_updates':[0,8,16]}},
                'numbers': {'media':{'packets':media}}, 'videos': {'all_required_clips_reviewed':True,
                                         'schema':'task10b_phase2_cross30_visual_review_v1',
                                         'numerical_review':numerical, 'media_packets':media,
                                         'movement_exploit_detected':False, 'trials_reviewed':30}}
    monkeypatch.setattr(curve,'bound',lambda item:Path(item['path']))
    monkeypatch.setattr(curve,'owned',lambda p:Path(p))
    monkeypatch.setattr(curve,'read',lambda p:evidence[str(p)])
    monkeypatch.setattr(curve,'binding',lambda p:dict(path=str(p),sha256=(
        'f1a8bf7ca8f3ba0effec20a3b4f97a78f1274c4f0fcbb83c536c9eb4ede06465' if str(p)=='numbers' else 'mock')))
    monkeypatch.setattr(curve.parent,'validate_plan',lambda p:({'retained':'paths'},['retained_rows'],None))
    monkeypatch.setattr(curve,'validate_rendered_media',lambda *args:None)
    plan=curve.create_plan('parent','numbers','videos')
    return plan,evidence


def test_new_budget_is_distinct_from_immutable_parent(valid_plan):
    plan,evidence=valid_plan
    original=deepcopy(evidence['parent'])
    paths,rows,budget=curve.validate_plan(plan)
    assert budget.transition_cap==262144 and budget.maximum_updates==32
    assert budget.maximum_wall_s==360 and budget.shutdown_reserve_s==90
    assert plan['validation_updates']==[0,8,16,24,32]
    assert plan['planned_validation_trials']==120
    assert evidence['parent']==original
    assert paths=={'retained':'paths'} and rows==['retained_rows']
    execution=curve.execution_plan(plan)
    assert execution['stopping_rules']['validation_updates']==[0,8,16,24,32]
    assert evidence['parent']['stopping_rules']['validation_updates']==[0,8,16]


@pytest.mark.parametrize('key,value', [
    ('seed',21022),('family','jab'),('fresh_transition_cap',524288),
    ('validation_updates',[0,16,32]),('application_wall_seconds',361.),
    ('maximum_session_allocation_seconds',481),('wrapper_wall_seconds',420),
    ('final_test_access',True),('actual_route_freeze',True),
    ('conditional_replication_budget_unlocked',True),('policy_promotion',True),
    ('prior_training_resumed',True),('stop_on_any_original_or_060_control_failure',False)])
def test_contract_mutations_rejected(valid_plan,key,value):
    plan,_=valid_plan;plan[key]=value
    with pytest.raises(ValueError,match='contract'):curve.validate_plan(plan)


@pytest.mark.parametrize('field,value', [('all_required_clips_reviewed',False),
    ('movement_exploit_detected',True),('trials_reviewed',29),('trials_reviewed',30.)])
def test_missing_visual_evidence_blocks_plan(valid_plan,field,value):
    plan,evidence=valid_plan;evidence['videos'][field]=value
    with pytest.raises(ValueError,match='All30'):curve.validate_plan(plan)


def test_changed_numerical_selection_is_not_a_generic_budget_unlock(valid_plan):
    plan,_=valid_plan;plan['numerical_review']['sha256']='changed'
    with pytest.raises(ValueError,match='Exact measured'):curve.validate_plan(plan)


@pytest.mark.parametrize('field,value', [('schema','unrelated30clips'),
    ('numerical_review',{'path':'different','sha256':'different'}),('media_packets',[])])
def test_an_unrelated_visual_review_cannot_unlock_training(valid_plan,field,value):
    plan,evidence=valid_plan;evidence['videos'][field]=value
    with pytest.raises(ValueError,match='exact parent'):curve.validate_plan(plan)


def test_changed_training_rows_and_larger_budget_blocked(valid_plan):
    plan,evidence=valid_plan
    evidence['parent']['training_candidate_ids']=['cross_uniform_0p600','cross_uniform_0p900']
    with pytest.raises(ValueError,match='reviewed cross'):curve.validate_plan(plan)
    evidence['parent']['training_candidate_ids']=curve.SELECTED
    plan['budget']['maximum_wall_s']=400
    with pytest.raises(ValueError,match='Exactly'):curve.validate_plan(plan)


SPEC=importlib.util.spec_from_file_location('tested_curve_transport',
    Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_curve_v2.py')
launcher=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(launcher)


def test_transport_keeps_original_reservation_and_watchdog():
    assert launcher.legacy.submit.__module__=='task10b_curve_frozen_transport'
    assert launcher.legacy.execute.__module__=='task10b_curve_frozen_transport'
    assert launcher.legacy.MODES==('timing-pilot',)
    assert launcher.legacy.CONDITIONAL_MODES==()


@pytest.mark.parametrize('mode,wall,app,proof', [('timing-pilot',481,420,None),
    ('timing-pilot',480,421,None),('timing-pilot',480,420,{}),
    ('grouped-finetune',480,420,None),('media',480,420,None)])
def test_transport_rejects_scope_and_budget_changes(mode,wall,app,proof):
    with pytest.raises(ValueError,match='Only exact'):launcher.validate_spec(mode,{'plan':'unused'},wall,app,proof=proof)


@pytest.mark.parametrize('remaining',[359.,float('nan'),391.,True])
def test_child_cannot_exceed_remaining_watchdog(remaining):
    with pytest.raises(ValueError,match='deadline'):
        launcher.native_command({'mode':'timing-pilot','application_wall_seconds':420},remaining)


def test_inventory_composition_restores_dependency_even_on_failure(monkeypatch):
    original=launcher.legacy.semantic_files
    monkeypatch.setattr(launcher,'curve_files',lambda p:{'a':'digest'})
    def fail(record,**kwargs):
        assert record['mode']=='timing-references' and record['inputs']=={}
        assert launcher.legacy.semantic_files([])=={'a':'digest'}
        raise ValueError('inventory failure')
    monkeypatch.setattr(launcher,'_required_inputs',fail)
    record={'mode':'timing-pilot','inputs':{'plan':'mock'},'runtime_namespace_proof':{'exact':True}}
    with pytest.raises(ValueError,match='inventory failure'):launcher.required_inputs(record)
    assert launcher.legacy.semantic_files is original
    assert record['mode']=='timing-pilot' and record['inputs']=={'plan':'mock'}


@pytest.fixture
def rendered_evidence(monkeypatch):
    data={}; expected=[]; actual=[]
    for update in [0,8,16]:
        old=Path(f'/mock/declared{update}/packet.json')
        path=old if update!=16 else Path('/mock/repaired16/packet.json')
        def item(p):return dict(path=str(p),sha256='digest'+str(update))
        packet=dict(mode='media',training=False,family='cross',seed=21021,
            inputs={'run':'same_run','panel':f'panel{update}'},files={'same_source':'same_hash'},
            allocation_wall_seconds=240,application_wall_seconds=180)
        data[str(old)]=deepcopy(packet)
        if update==16:packet.update(allocation_wall_seconds=300,application_wall_seconds=240)
        data[str(path)]=packet
        expected.append(dict(update=update,packet=item(old)))
        summary=path.parent/'run/summary.json';execution=path.parent/'execution.json';terminal=path.parent/'terminal_accounting.json'
        data[str(summary)]=dict(planned=10,rendered=10,failed=0,actor_calls=0)
        data[str(execution)]=dict(status='executed',returncode=0,packet_sha256=item(path)['sha256'],job_id=str(update))
        data[str(terminal)]=dict(state='COMPLETED',packet_sha256=item(path)['sha256'],job_id=str(update))
        actual.append(dict(update=update,packet=item(path),render_summary=item(summary),
                           render_execution=item(execution),terminal_accounting=item(terminal)))
    monkeypatch.setattr(curve,'bound',lambda item:Path(item['path']))
    monkeypatch.setattr(curve,'read',lambda p:data[str(p)])
    return dict(rendered_media_packets=actual),expected,data


def test_explicit_setup_repair_keeps_declared_selection(rendered_evidence):
    video,expected,_=rendered_evidence
    curve.validate_rendered_media(video,expected)
    assert video['rendered_media_packets'][2]['packet']!=expected[2]['packet']


@pytest.mark.parametrize('mutation',['different_trace','changed_source','too_long','missing_clip','failed_job','wrong_packet','wrong_job','foreign_summary'])
def test_setup_repair_cannot_hide_a_changed_or_failed_render(rendered_evidence,mutation):
    video,expected,data=rendered_evidence
    packet=data['/mock/repaired16/packet.json']
    if mutation=='different_trace':packet['inputs']['run']='unrelated'
    elif mutation=='changed_source':packet['files']['same_source']='newmath'
    elif mutation=='too_long':packet['allocation_wall_seconds']=600
    elif mutation=='missing_clip':data['/mock/repaired16/run/summary.json']['rendered']=9
    elif mutation=='failed_job':data['/mock/repaired16/terminal_accounting.json']['state']='FAILED'
    elif mutation=='wrong_packet':data['/mock/repaired16/execution.json']['packet_sha256']='other'
    elif mutation=='wrong_job':data['/mock/repaired16/execution.json']['job_id']='other'
    else:video['rendered_media_packets'][2]['render_summary']['path']='/unrelated/summary.json'
    with pytest.raises(ValueError):curve.validate_rendered_media(video,expected)
