from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json
import numpy as np
import pytest
from human_punching import phase2_expanded_clocks_v1 as clocks
from human_punching import phase2_expanded_clock_audit_v1 as auditor


@pytest.fixture
def protocol(tmp_path):
    row=dict(candidate_id='jab_validation_exact',family='jab',requested_ttc_s=.6,target_w=[1.,2.,3.],
        reference_sha256='a'*64,legacy_runtime_sha256='a'*64,start_mode='family_guard',
        preparation_complete_tick=0,launch_tick=4,outward_end_tick=8,phase_denominator_s=1.05)
    decision=dict(world=0,trial_id='trial',boundary_snapshot_index=0,accepted=True,reference_tick=4,
        actual_launch_tick=4,current_positional_support_ready=True,
        request={**{k:row[k] for k in ('candidate_id','family','requested_ttc_s','reference_sha256','target_w')},'trial_id':'trial'},
        planning_wall_s=.003,cache_lookup_wall_s=.001,target_callback_wall_s=.001,
        simulation_pause_wall_s=.003,simulated_time_advanced_s=0.)
    env=SimpleNamespace(n=1,library=SimpleNamespace(rows=[row]),closed_launch_records=[decision],closed_boundary_snapshots=[dict(episode_tick=4)])
    return env,tmp_path/'batch'


def report():
    return dict(trial_id='trial',candidate_id='jab_validation_exact',numeric_complete_cycle=False,
        first_active_contact_s=.62,first_inactive_contact_s=None,full_active_glove_rms_m=.023,
        functional=dict(launch_time_s=.02,readiness_time_s=0.,outward_completion_time_s=.04,recovery_time_s=.045),
        request_time_s=0.,preparation_complete_time_s=.02,final_planning_time_s=.02)


def commitment():
    return dict(trial_id='trial',world=0,requested_episode_tick=4,actual_launch_time_s=.02,
        measured_human_state={'qpos':[1,2,3]},previous_raw_action=[.1],outgoing_raw_action=[.2],
        disposition='committed',static_D0_cached_reference_only=True)


def adapter(protocol):
    env,folder=protocol
    return clocks.EventMetadata(env,folder,[0],['trial'])


def test_prep0_separate_from_actual_launch_and_numeric_untouched(protocol):
    m=adapter(protocol);r=report();old=deepcopy(r)
    m.before_write(m.folder/'trial_0000.json',r)
    assert r['preparation_complete_time_s']==0
    assert r['event_clocks']['actual_launch_time_s']==.02
    assert r['event_clocks']['first_positional_support_readiness_time_s']==0
    assert r['event_clocks']['launch_boundary_readiness_time_s']==.02
    assert {k:r[k] for k in old if k!='preparation_complete_time_s'}=={k:v for k,v in old.items() if k!='preparation_complete_time_s'}
    assert r['event_clocks']['simulation_pause_wall_s']==.003


@pytest.mark.parametrize('reason',['current_positional_support_readiness_failed','final_planning_late','launch_target_input_mismatch_or_stale'])
def test_rejected_boundary_planning_remains_distinct_from_missing_launch(protocol,reason):
    env,_=protocol;d=env.closed_launch_records[0];d.update(accepted=False,actual_launch_tick=None,reason=reason)
    d['current_positional_support_ready']=reason!='current_positional_support_readiness_failed'
    r=report();r['functional'].update(launch_time_s=None,recovery_time_s=None)
    m=adapter(protocol);m.before_write(m.folder/'trial_0000.json',r)
    assert r['final_planning_time_s']==.02 and r['preparation_complete_time_s']==0
    assert r['event_clocks']['actual_launch_time_s'] is None
    assert r['event_clocks']['outward_complete_time_s'] is None
    assert r['first_active_contact_s']==.62 and r['numeric_complete_cycle'] is False


def test_unreached_boundary_never_invents_planning(protocol):
    env,_=protocol;env.closed_launch_records=[]
    r=report();r['functional']['launch_time_s']=None
    m=adapter(protocol);m.before_write(m.folder/'trial_0000.json',r)
    assert r['final_planning_time_s'] is None and r['event_clocks']['closed_decision_sha256'] is None


@pytest.mark.parametrize('field,value',[('start_mode','shared'),('preparation_complete_tick',4),('launch_tick',3),('outward_end_tick',4)])
def test_scope_refusal(protocol,field,value):
    protocol[0].library.rows[0][field]=value
    with pytest.raises(ValueError):adapter(protocol)


@pytest.mark.parametrize('field,value',[('candidate_id','substitute'),('reference_sha256','b'*64),('target_w',[2.,2.,3.]),('family','cross'),('requested_ttc_s',.75)])
def test_immutable_exact_decision_binding(protocol,field,value):
    protocol[0].closed_launch_records[0]['request'][field]=value
    with pytest.raises(ValueError):adapter(protocol).clocks(0)


def test_commitment_replaces_only_inherited_metadata(protocol):
    m=adapter(protocol);c=commitment();state=deepcopy(c['measured_human_state'])
    data=dict(schema='task10b_phase2_launch_commitments_v1',commitments=[c])
    m.before_write(m.folder/'launch_commitments.json',data)
    assert 'static_D0_cached_reference_only' not in c
    assert c['closed_protocol_decision']==protocol[0].closed_launch_records[0]
    assert c['measured_human_state']==state and c['outgoing_raw_action']==[.2]
    assert c['closed_protocol_decision_sha256']==clocks.identity(c['closed_protocol_decision'])


def test_no_old_file_rewrite_or_path_escape(protocol,tmp_path):
    m=adapter(protocol)
    with pytest.raises(ValueError):m.before_write(tmp_path/'elsewhere.json',{})
    m.folder.mkdir();p=m.folder/'plan.json';p.write_text('preserved')
    with pytest.raises(FileExistsError):m.before_write(p,{'schema':'task10b_phase2_execution_plan_v1'})
    assert p.read_text()=='preserved'


def test_original_evaluator_once_writer_restored_on_exception(protocol,monkeypatch):
    from human_punching import phase2_learning as original
    saved=original.write_json;calls=[]
    def fake(env,policy,output,**kwargs):
        calls.append(True);Path(output).mkdir();value=dict(schema='task10b_phase2_execution_plan_v1')
        original.write_json(Path(output)/'plan.json',value)
        assert value['event_clock_version']==clocks.VERSION
        raise RuntimeError('retained original exception')
    monkeypatch.setattr(original,'evaluate_policy',fake)
    with pytest.raises(RuntimeError,match='retained original exception'):
        clocks.evaluate_policy(protocol[0],None,protocol[1],reference_ids=[0],trial_ids=['trial'])
    assert calls==[True] and original.write_json is saved


def test_callback_scope_restores_on_exception():
    from human_punching import phase2_validation_callback as callback
    saved=callback.evaluate_policy
    with pytest.raises(RuntimeError):
        with clocks.validation_clock_scope():
            assert callback.evaluate_policy is clocks.evaluate_policy
            raise RuntimeError()
    assert callback.evaluate_policy is saved


def saved_batch(protocol):
    m=adapter(protocol);m.folder.mkdir()
    plan=dict(schema='task10b_phase2_execution_plan_v1',trials=['trial'],planned_physics_steps=8)
    r=report();c=commitment();commitments=dict(schema='task10b_phase2_launch_commitments_v1',commitments=[c])
    summary=dict(schema='task10b_phase2_native_panel_v1',trials=[r])
    for name,value in [('plan.json',plan),('launch_commitments.json',commitments),('trial_0000.json',r),('summary.json',summary)]:
        m.before_write(m.folder/name,value);(m.folder/name).write_text(json.dumps(value))
    (m.folder/'closed_protocol_launch.json').write_text(json.dumps(dict(decisions=protocol[0].closed_launch_records)))
    np.savez(m.folder/'raw_trace.npz',qpos=np.zeros((9,1,55)),launched=np.array([[False],[True]]),policy_episode_tick=np.array([0,4]),policy_readiness=np.array([[False],[True]]))
    return m.folder


def test_independent_saved_metadata_audit(protocol):
    assert auditor.audit_batch(saved_batch(protocol))['passed']


@pytest.mark.parametrize('mutation',['prep','commitment','raw_launch','raw_readiness','missing_trial','outward'])
def test_saved_audit_catches_metadata_or_raw_mismatch(protocol,mutation):
    folder=saved_batch(protocol)
    if mutation in ('raw_launch','raw_readiness'):
        with np.load(folder/'raw_trace.npz') as a:raw={k:a[k].copy() for k in a.files}
        raw['launched' if mutation=='raw_launch' else 'policy_readiness'][-1,0]=False
        np.savez(folder/'raw_trace.npz',**raw)
    else:
        path=folder/('launch_commitments.json' if mutation=='commitment' else 'summary.json');value=json.loads(path.read_text())
        if mutation=='prep':value['trials'][0]['preparation_complete_time_s']=.02
        elif mutation=='outward':value['trials'][0]['event_clocks']['outward_complete_time_s']=.02
        elif mutation=='commitment':value['commitments'][0]['closed_protocol_decision']['accepted']=False
        else:value['trials']=[]
        path.write_text(json.dumps(value))
    assert not auditor.audit_batch(folder)['passed']


def driver():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('clock_driver_test',root/'scripts/task10b_phase2_grouped_clocks_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('mode,wall,app',[('runtime-pack',1800,1680),('grouped-pilot',1801,1680),('grouped-pilot',1800,1700)])
def test_transport_refuses_other_scope(mode,wall,app):
    with pytest.raises(ValueError):driver().validate_spec(mode,{},wall,app)


def test_transport_dispatch_only_entry_changed(monkeypatch):
    d=driver();args=['python','-B','-m','human_punching.phase2_grouped_entry_v2','--configuration','immutable','--configuration-sha256','bound','--output','fresh']
    monkeypatch.setattr(d,'_native',lambda *unused:list(args))
    actual=d.native_command({'mode':'grouped-pilot'},1600)
    assert actual[:3]==args[:3] and actual[4:]==args[4:]
    assert actual[3]=='human_punching.phase2_grouped_clock_entry_v1'


def test_real_configuration_preflight_and_exact_input_closure():
    d=driver();configuration=d.ROOT/'outputs/task10b/p2jgconfig01/configuration.json'
    inputs={'configuration':str(configuration)}
    scope=d.validate_spec('grouped-pilot',inputs,1800,1680)
    assert scope['expanded_event_clock_version']==clocks.VERSION
    provenance={}
    record=json.loads((d.ROOT/'outputs/task10b/p2pk01/packet.json').read_text())
    record.update(mode='grouped-pilot',inputs=inputs,successful_route_freeze=None)
    closed=d.required_inputs(record,provenance_files=provenance)
    for name in ('phase2_expanded_clocks_v1.py','phase2_expanded_clock_audit_v1.py','phase2_grouped_clock_entry_v1.py'):
        assert 'src/human_punching/'+name in closed
    assert 'outputs/task10b/p2jgconfig01/configuration.json' in closed
    assert 'outputs/task10b/p2pk01/run/catalog_00.json' in closed
    assert 'outputs/task10b/p2pk01/run/catalog_01.json' in closed
    assert 'outputs/task10b/p2domain01/all_slot_dispositions.npz' in closed
    assert provenance


@pytest.mark.parametrize('integrity_pass',[True,False])
def test_grouped_entry_keeps_original_parity_failure(protocol,monkeypatch,integrity_pass):
    from human_punching import phase2_grouped_clock_entry_v1 as entry
    env,folder=protocol
    def original(path,digest,output):
        destination=Path(output)/'held_out_validation/validation_00000/batch_0000'
        destination.parent.mkdir(parents=True)
        saved_batch((env,destination))
        return dict(status='bounded_stop' if integrity_pass else 'failed',grouped_integrity_pass=integrity_pass,
            grouped_integrity_v2={'path':'preserved','sha256':'a'*64})
    monkeypatch.setattr(entry.parent,'execute_configuration',original)
    result=entry.execute_configuration('unchanged','b'*64,folder)
    assert result['event_clocks_pass'] is True
    assert result['status']==('bounded_stop' if integrity_pass else 'failed')
    assert result['grouped_integrity_pass'] is integrity_pass


def test_grouped_entry_rejects_absent_batch_evidence(protocol,monkeypatch):
    from human_punching import phase2_grouped_clock_entry_v1 as entry
    def original(path,digest,output):
        Path(output).mkdir();return dict(status='bounded_stop',grouped_integrity_pass=True)
    monkeypatch.setattr(entry.parent,'execute_configuration',original)
    result=entry.execute_configuration('unchanged','b'*64,protocol[1])
    assert result['status']=='failed' and result['event_clocks_pass'] is False
