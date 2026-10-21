"""Five-panel saved bookkeeping: no models, physics, learned updates or jobs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest
from human_punching import phase2_curve_saved_v2 as saved


def summary():
    return dict(schema='task10b_phase2_d0_panel_v1',planned_trials=24,hard_faults=[],trials=[
        dict(trial_id=f'{candidate}_bank_{bank:02d}',candidate_id=candidate,disposition='executed',numeric_complete_cycle=True)
        for candidate in ('cross_original',*saved.pilot.SELECTED) for bank in range(8)])


def test_complete120_requires_raw_evidence_even_with_saved_success():
    panels={u:summary() for u in saved.pilot.UPDATES}
    missing=saved.materialize(panels,terminal=True)
    assert len(missing['trials'])==120 and missing['status_counts']=={'execution_extent_unknown':120}
    extent={(u,c):992 for u in saved.pilot.UPDATES for c in ('cross_original',*saved.pilot.SELECTED)}
    full=saved.materialize(panels,raw_extents=extent,terminal=True)
    assert full['status_counts']=={'executed_pending_independent_audit':120}
    assert not any(r['numeric_success_credited'] for r in full['trials'])


def test_thrown_regression_panel_retained_without_training_summary():
    panel=summary();panel['trials'][0]['numeric_complete_cycle']=False
    control=saved.pilot.control_regression(panel)
    assert not control['passed']
    progress={u:dict(disposition='unattempted') for u in saved.pilot.UPDATES[1:]}
    extent={(0,c):900 for c in ('cross_original',*saved.pilot.SELECTED)}
    result=saved.materialize({0:panel},started=(0,),controls={0:control},raw_extents=extent,progress=progress,terminal=True)
    assert result['status_counts']=={'executed_pending_independent_audit':24,'unattempted':96}
    assert all(r['development_control_regression'] for r in result['trials'][:24])
    assert not any(r['control_regression_is_physical_fault'] for r in result['trials'])
    assert result['trials'][0]['reported_numeric_complete_cycle'] is False


def test_watchdog_interruption_never_invents_unattempted():
    progress={u:dict(disposition='unattempted') for u in saved.pilot.UPDATES}
    result=saved.materialize({},started=(0,),progress=progress,terminal=True)
    assert result['status_counts']=={'execution_extent_unknown':24,'unattempted':96}
    killed=saved.materialize({},started=(0,),progress=progress)
    assert killed['status_counts']=={'execution_extent_unknown':120}
    partial=saved.materialize({},started=(0,),raw_extents={(0,'cross_original'):3})
    assert partial['status_counts']=={'execution_extent_unknown':112,'partial_execution_extent_recorded':8}


def test_constructor_zero_execution_requires_terminal_explicit_progress():
    p={u:dict(disposition='unattempted') for u in saved.pilot.UPDATES}
    assert saved.materialize({},progress=p,terminal=True)['status_counts']=={'unattempted':120}
    assert saved.materialize({})['status_counts']=={'execution_extent_unknown':120}
    assert saved.materialize({},pending=True)['status_counts']=={'not_yet_executed_pending':120}


@pytest.mark.parametrize('kwargs',[
    dict(summaries={7:summary()}),dict(summaries={},started=(40,)),
    dict(summaries={},raw_extents={(0,'other'):1}),dict(summaries={},controls={0:{}}),
    dict(summaries={},pending=True,terminal=True),dict(summaries={0:summary()},pending=True),
    dict(summaries={},raw_extents={(0,'cross_original'):-1}),
    dict(summaries={},raw_extents={(0,'cross_original'):True}),
])
def test_false_or_unknown_evidence_rejected(kwargs):
    with pytest.raises(ValueError):saved.materialize(**kwargs)


@pytest.mark.parametrize('mutation',('duplicate','unknown','wrongcandidate','wrongcontrol'))
def test_mismatched_rows_and_control_receipts_rejected(mutation):
    panel=summary();controls={}
    if mutation=='duplicate':panel['trials'].append(panel['trials'][0])
    if mutation=='unknown':panel['trials'][0]['trial_id']='other'
    if mutation=='wrongcandidate':panel['trials'][0]['candidate_id']=saved.pilot.SELECTED[0]
    if mutation=='wrongcontrol':controls[0]={'passed':False}
    with pytest.raises(ValueError):saved.materialize({0:panel},controls=controls)


SPEC=importlib.util.spec_from_file_location('curve_media_test_launcher',Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_curve_media_v2.py')
launcher=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(launcher)


def test_saved_only_dispatch_and_late_setup(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher.legacy.base,'local',lambda p:Path(p))
    r=dict(mode='media',allocation_wall_seconds=300,application_wall_seconds=240,training=False,
        actor_calls=0,planned_source_validation_trials=120,inputs={'run':'run','panel':'panel'},output='outputs/task10b/example/run')
    command=launcher.native_command(r,130)
    assert command[2]==str(launcher.ROOT/'scripts/task10b_phase2_media.py')
    assert command[-2:]==['--wall-seconds','130'] and '-m' not in command
    for value in (120,211,float('nan'),True):
        with pytest.raises(RuntimeError):launcher.native_command(r,value)
    with pytest.raises(ValueError):launcher.native_command(dict(r,training=True),130)


@pytest.mark.parametrize('update',(0,8,16,24,32))
def test_exact_new_schema_and_all_declared_updates(tmp_path,monkeypatch,update):
    run=tmp_path/'run';run.mkdir();panel=run/f'validation_{update:05d}/summary.json';panel.parent.mkdir();panel.write_text('{}')
    launch=dict(schema=saved.pilot.VERSION,mode='timing_learning_curve',family='cross',seed=21021)
    (run/'launch.json').write_text(json.dumps(launch));monkeypatch.setattr(launcher.legacy.base,'local',lambda p:Path(p).resolve())
    result=launcher.validate_spec('media',dict(run=str(run),panel=str(panel)),300,240)
    assert result['saved_curve_update']==update and result['actor_calls']==0
    launch['schema']='task10b_phase2_selected_d0_timing_pilot_v1';(run/'launch.json').write_text(json.dumps(launch))
    with pytest.raises(ValueError):launcher.validate_spec('media',dict(run=str(run),panel=str(panel)),300,240)


def test_inventory_does_not_recursively_traverse_saved_run_and_restores(monkeypatch):
    r=dict(mode='media',inputs={'run':'r','panel':'p'},runtime_namespace_proof={'x':1})
    old=launcher.legacy.semantic_files
    def error(record,**kw):
        assert record['mode']=='timing-references' and record['inputs']=={}
        assert launcher.legacy.semantic_files([])=={}
        return launcher.legacy.semantic_files(['forbidden'])
    monkeypatch.setattr(launcher,'_required_inputs',error)
    with pytest.raises(ValueError,match='inventory seed'):launcher.required_inputs(r)
    assert launcher.legacy.semantic_files is old


def test_same_budget_lock_and_no_automatic_training():
    assert launcher.legacy.CONDITIONAL_MODES==() and launcher.legacy.MODES==('media',)
    assert launcher.legacy.submit.__module__=='task10b_curve_media_frozen_driver'
    assert launcher.legacy.budget_status.__module__=='human_punching.phase2_budget'
    with pytest.raises(ValueError):launcher.prepare('unused','media',{})


def test_saved_ledger_reads_trace_headers_and_keeps_constructor_interruptions(tmp_path,monkeypatch):
    import numpy as np
    root=tmp_path/'human';run=root/'outputs/task10b/probe/run';run.mkdir(parents=True)
    monkeypatch.setattr(saved.pilot,'ROOT',root)
    monkeypatch.setattr(saved.pilot,'validate_plan',lambda value:None)
    monkeypatch.setattr(saved.pilot,'execution_plan',lambda value:{'explicit':'view'})
    def write(relative,value):
        p=run/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    write('plan.json',{'bounded':'plan'});write('execution_plan.json',{'explicit':'view'})
    expected=[f'{c}_bank_{b:02d}' for c in ('cross_original',*saved.pilot.SELECTED) for b in range(8)]
    write('validation_declared.json',dict(trial_ids=expected,updates={str(u):dict(planned_trials=24) for u in saved.pilot.UPDATES}))
    write('validation_00000_started.json',{'update':0})
    folder=run/'validation_00000/cross_original';folder.mkdir(parents=True)
    np.savez(folder/'raw_trace.npz',qpos=np.zeros((4,8,55)))
    result=saved.ledger(run)
    assert result['status_counts']=={'execution_extent_unknown':112,'partial_execution_extent_recorded':8}
    assert all(row['observed_physics_steps_per_world']==3 for row in result['trials'][:8])
    write('execution.json',dict(schema=saved.pilot.VERSION,plan_sha256=saved.pilot.sha(run/'plan.json'),status='failed',
        validation_progress={str(u):dict(disposition='unattempted') for u in saved.pilot.UPDATES}))
    result=saved.ledger(run)
    assert result['status_counts']=={'execution_extent_unknown':16,'partial_execution_extent_recorded':8,'unattempted':96}


def test_audit_wrapper_never_turns_small_retained_pass_into120_completion(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('curve_audit_test_wrapper',Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_curve_audit_v2.py')
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    root=tmp_path/'human';run=root/'outputs/task10b/run';run.mkdir(parents=True)
    monkeypatch.setattr(wrapper,'ROOT',root)
    empty=saved.materialize({})
    monkeypatch.setattr(wrapper,'ledger',lambda *args,**kwargs:empty)
    output=root/'outputs/task10b/audit'
    report=wrapper.review(run,output,ledger_only=True)
    assert report['planned_trials']==120 and report['route_credit'] is False
    assert report['all_declared_trials_independently_audited'] is False
    assert len(json.loads((output/'declared_ledger.json').read_text())['trials'])==120
    with pytest.raises(ValueError,match='Fresh'):wrapper.review(run,output,ledger_only=True)
