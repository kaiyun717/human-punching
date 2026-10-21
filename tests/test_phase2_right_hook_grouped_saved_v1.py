from pathlib import Path
import json
import numpy as np
import pytest
from human_punching import phase2_right_hook_grouped_saved_v1 as saved


def batches():
    return [('a',[dict(trial_id=f'a{i}') for i in range(4)]),('b',[dict(trial_id=f'b{i}') for i in range(4)]),('c',[dict(trial_id=f'c{i}') for i in range(4)])]


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def raw_batch(panel,index=0,*,steps=8,reported=True):
    folder=panel/f'batch_{index:04d}';folder.mkdir(parents=True)
    put(folder/'plan.json',dict(planned_physics_steps=8))
    np.savez(folder/'raw_trace.npz',pre_qpos=np.zeros((steps,4,55)),qpos=np.zeros((steps+1,4,55)),policy_episode_tick=np.arange(steps//4)*4)
    if reported:put(folder/'summary.json',dict(status='executed',trials=[dict(trial_id=f'{"abc"[index]}{i}',numeric_complete_cycle=i!=0) for i in range(4)]))
    return folder


def test_absent_callback_proves_unattempted_before_native_constructor(tmp_path):
    r=saved.panel_ledger(tmp_path/'absent',batches())
    assert len(r['trials'])==12 and {t['status'] for t in r['trials']}=={'unattempted_callback_not_entered'}


def test_absent_pending_callback_is_explicit(tmp_path):
    assert {t['status'] for t in saved.panel_ledger(tmp_path/'absent',batches(),pending=True)['trials']}=={'pending_job'}


def test_constructor_failure_with_callback_unattempted_labels_is_unknown(tmp_path):
    put(tmp_path/'summary.json',dict(status='failed',exception={'type':'ConstructorError'},trials=[]))
    put(tmp_path/'ledger_0000.json',[dict(trial_id='a0',disposition='unattempted')])
    r=saved.panel_ledger(tmp_path,batches())
    assert {t['status'] for t in r['trials'][:4]}=={'execution_extent_unknown'}
    assert {t['status'] for t in r['trials'][4:]}=={'unattempted_after_callback_stop'}


def test_watchdog_after_ledger_can_interrupt_next_constructor(tmp_path):
    raw_batch(tmp_path);put(tmp_path/'ledger_0000.json',[])
    r=saved.panel_ledger(tmp_path,batches())
    assert {t['status'] for t in r['trials'][:4]}=={'completed_native_attempt_pending_independent_audit'}
    assert {t['status'] for t in r['trials'][4:8]}=={'execution_extent_unknown'}
    assert {t['status'] for t in r['trials'][8:]}=={'unattempted_after_callback_stop'}


def test_prefix_extent_retained_and_no_invented_completed_trials(tmp_path):
    raw_batch(tmp_path,steps=4,reported=False)
    r=saved.panel_ledger(tmp_path,batches())
    assert {t['status'] for t in r['trials'][:4]}=={'retained_raw_execution_extent_known'}
    assert r['panels'][0]['raw']['physics_steps']==4
    assert not r['panels'][0]['complete_saved_counts']


def test_all_intervals_without_postprocessed_trials_get_no_completed_credit(tmp_path):
    raw_batch(tmp_path,reported=False)
    r=saved.panel_ledger(tmp_path,batches())
    assert r['panels'][0]['complete_saved_counts']
    assert {t['status'] for t in r['trials'][:4]}=={'retained_raw_execution_extent_known'}


def test_numeric_failure_remains_completed_measurement(tmp_path):
    raw_batch(tmp_path)
    r=saved.panel_ledger(tmp_path,batches())
    assert r['trials'][0]['recorded_numeric_complete_cycle'] is False
    assert r['trials'][0]['status']=='completed_native_attempt_pending_independent_audit'
    assert r['trials'][1]['recorded_numeric_complete_cycle'] is True


def test_successful_all_batches_no_unknown_active(tmp_path):
    for i in range(3):raw_batch(tmp_path,i)
    put(tmp_path/'summary.json',dict(status='executed'))
    r=saved.panel_ledger(tmp_path,batches())
    assert r['active_batch_inferred_from_progress'] is None
    assert {t['status'] for t in r['trials']}=={'completed_native_attempt_pending_independent_audit'}


def test_expected_batches_uses_declared_order_and_exact_four_starts():
    declaration={'bindings':[dict(trial_id=f't{i}') for i in range(96)]}
    manifest={'trials':[dict(trial_id=f't{i}',candidate_id=f'c{i//4}') for i in range(96)]}
    result=saved.expected_batches(declaration,manifest)
    assert len(result)==24 and result[0][0]=='c0' and result[-1][0]=='c23'
    manifest['trials'][0]['candidate_id']='c1'
    with pytest.raises(ValueError):saved.expected_batches(declaration,manifest)


def test_real_grouped_packet_source_input_closure():
    root=Path(__file__).resolve().parents[1]
    ctx,path,packet,configuration,declaration,rows=saved.context(root/'outputs/task10b/p2rhgp01/packet.json',root/'outputs/task10b/p2hostalias01/probe.json')
    assert packet['schema']=='task10b_phase2_right_hook_grouped_packet_v1'
    assert len(rows)==24 and sum(map(lambda x:len(x[1]),rows))==96
    assert configuration['budget']['maximum_updates']==8
    assert declaration['split']=='validation' and declaration['interpolation_timing_s']==1.1


@pytest.mark.parametrize('mutated',[False,True])
def test_metadata_compared_to_independently_recomputed_events(tmp_path,mutated):
    import importlib.util
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('grouped_audit_test',root/'scripts/task10b_phase2_right_hook_grouped_audit_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    independent=dict(trial_id='one',request_s=0.,preparation_complete_s=0.,first_positional_support_readiness_s=.005,
        launch_s=.46,outward_complete_s=2.,first_active_contact_s=1.06,first_inactive_contact_s=None,recovery_time_s=2.4)
    events=dict(request_time_s=0.,preparation_complete_time_s=.46 if mutated else 0.,first_positional_support_readiness_time_s=.005,
        actual_launch_time_s=.46,outward_complete_time_s=2.,first_active_contact_time_s=1.06,
        first_inactive_contact_time_s=None,functional_recovery_time_s=2.4)
    put(tmp_path/'batch_0000/summary.json',dict(trials=[dict(trial_id='one',event_clocks=events)]))
    result=module.independent_event_comparison(dict(panels=[dict(path='/immutable_saved/batch_0000',trials=[independent])]),tmp_path)
    assert result['passed'] is not mutated


def test_two_panel_ledger_preserves_stopped_second_panel(tmp_path,monkeypatch):
    declaration={'bindings':[dict(trial_id=f't{i}') for i in range(96)]}
    manifest={'trials':[dict(trial_id=f't{i}',candidate_id=f'c{i//4}') for i in range(96)]}
    groups=saved.expected_batches(declaration,manifest)
    packet=tmp_path/'packet.json';put(packet,{'stub':True})
    config=tmp_path/'config.json';put(config,{'stub':True})
    class Context:
        def owned(self,path):return Path(path)
    value=dict(output='run',inputs={'configuration':str(config)})
    monkeypatch.setattr(saved,'ROOT',tmp_path)
    monkeypatch.setattr(saved,'context',lambda *args:(Context(),packet,value,{},declaration,groups))
    put(tmp_path/'terminal_accounting.json',dict(packet_sha256=saved.sha(packet),job_id='9'))
    put(tmp_path/'execution.json',dict(packet_sha256=saved.sha(packet),job_id='9'))
    folder=tmp_path/'run/held_out_validation/validation_00000'
    put(folder/'summary.json',dict(status='failed',trials=[]))
    put(folder/'ledger_0000.json',[])
    result=saved.ledger(packet,None)
    assert result['planned_trials']==len(result['trials'])==192
    assert result['counts']=={'execution_extent_unknown':4,'unattempted_after_callback_stop':92,'unattempted_callback_not_entered':96}
    assert result['fresh_transition_cap']==65536 and result['planned_updates']==[0,8]
    assert not result['route_credit']


def receipts(tmp_path,monkeypatch):
    from human_punching import phase2_checkpoint_selection_v1 as selector
    monkeypatch.setattr(selector,'matching_evaluated_checkpoint',lambda *args:None)
    configuration=dict(path='configuration.json',sha256='config')
    final=tmp_path/'training/checkpoint_final.pt';final.parent.mkdir();final.write_bytes(b'checkpoint')
    training=dict(grouped_schema='task10b_phase2_grouped_training_receipt_v1',status='bounded_stop',
        reload_parity=dict(parameters_equal=True,actions_equal=True,maximum_action_error=0.),
        ancestry=dict(fresh_optimizer=True,fresh_simulation=True,seed=21021,sha256='accepted'),
        initial_policy_digest='initial',final_policy_digest='final',weight_changed=True,
        completed_updates=8,fresh_transitions=65536,checkpoint=saved.binding(final),hard_faults=[])
    put(tmp_path/'grouped_training_receipt.json',training)
    parity=dict(schema='task10b_phase2_grouped_entry_integrity_v3',passed=True,configuration=configuration,
        original_receipt=saved.binding(tmp_path/'grouped_training_receipt.json'),checkpoint=saved.binding(final))
    put(tmp_path/'grouped_integrity_v3.json',parity)
    clocks=dict(schema='task10b_phase2_grouped_clock_entry_v2',passed=True,configuration=configuration,
        inherited_integrity=saved.binding(tmp_path/'grouped_integrity_v3.json'))
    put(tmp_path/'grouped_event_clocks_v2.json',clocks)
    return configuration,training,parity,clocks


def test_consumer_checks_accept_exact_new_schemas(tmp_path,monkeypatch):
    config,*_=receipts(tmp_path,monkeypatch)
    r=saved.consumer_receipts(tmp_path,config)
    assert r['passed'] and not r['route_credit']


@pytest.mark.parametrize('mutation',['missing_parity','false_parity','nonzero_error','old_integrity','old_clocks','changed_checkpoint','counter_overrun','stale_training_receipt','stale_inherited_integrity'])
def test_false_or_wrong_consumer_evidence_fails_closed(tmp_path,monkeypatch,mutation):
    config,training,parity,clocks=receipts(tmp_path,monkeypatch)
    if mutation=='missing_parity':training.pop('reload_parity')
    if mutation=='false_parity':training['reload_parity']['actions_equal']=False
    if mutation=='nonzero_error':training['reload_parity']['maximum_action_error']=1e-8
    if mutation=='old_integrity':parity['schema']='task10b_phase2_grouped_entry_integrity_v2'
    if mutation=='old_clocks':clocks['schema']='task10b_phase2_grouped_clock_entry_v1'
    if mutation=='changed_checkpoint':parity['checkpoint']['sha256']='changed'
    if mutation=='counter_overrun':training['completed_updates']=9;training['fresh_transitions']=73728
    if mutation=='stale_training_receipt':parity['original_receipt']['sha256']='stale'
    if mutation=='stale_inherited_integrity':clocks['inherited_integrity']['sha256']='stale'
    put(tmp_path/'grouped_training_receipt.json',training)
    if mutation!='stale_training_receipt':parity['original_receipt']=saved.binding(tmp_path/'grouped_training_receipt.json')
    put(tmp_path/'grouped_integrity_v3.json',parity)
    if mutation!='stale_inherited_integrity':clocks['inherited_integrity']=saved.binding(tmp_path/'grouped_integrity_v3.json')
    put(tmp_path/'grouped_event_clocks_v2.json',clocks)
    assert saved.consumer_receipts(tmp_path,config)['passed'] is False


def test_missing_consumers_remain_missing_after_interruption(tmp_path):
    r=saved.consumer_receipts(tmp_path,{'sha256':'config'})
    assert not r['passed'] and set(r['missing_receipts'])=={'training','integrity','clocks'}


@pytest.mark.parametrize('key,value',[('schema','task10b_phase2_grouped_clock_packet_v1'),('family','jab'),('fresh_transition_cap',262144),('allocation_wall_seconds',840),('planned_validation_updates',[0,8,16]),('grouped_integrity_gate_version','task10b_phase2_grouped_entry_integrity_v2')])
def test_context_rejects_other_experiment_before_reading_sources(tmp_path,monkeypatch,key,value):
    packet=dict(schema='task10b_phase2_right_hook_grouped_packet_v1',mode='grouped-pilot',
        allocation_wall_seconds=960,application_wall_seconds=900,successful_route_freeze=None,
        family='right_hook',seed=21021,fresh_transition_cap=65536,grouped_child_wall_seconds=720,
        planned_validation_updates=[0,8],planned_validation_attempts=192,
        grouped_integrity_gate_version='task10b_phase2_grouped_entry_integrity_v3')
    packet[key]=value;put(tmp_path/'packet.json',packet)
    class Context:
        def __init__(self,*args,**kwargs):pass
        def owned(self,path):return Path(path)
        def bound(self,*args):raise AssertionError('Should reject before file traversal')
    monkeypatch.setattr(saved,'Inputs',Context)
    with pytest.raises(ValueError,match='Exact initial-budget'):
        saved.context(tmp_path/'packet.json',None)
