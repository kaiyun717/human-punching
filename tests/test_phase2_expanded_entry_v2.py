"""Direct routing/clock metadata and immutable CPU input edges, no native run."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from human_punching import phase2_expanded_entry_v2 as entry
from human_punching import phase2_expanded_inputs_v2 as inputs


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return entry.clocks.binding(path)


@pytest.fixture
def artifacts(tmp_path,monkeypatch):
    folder=tmp_path/'inputs';folder.mkdir()
    reference=folder/'ref.npz';reference.write_bytes(b'exact reference bytes not loaded')
    model=folder/'model.xml';model.write_text('<mujoco/>')
    bank=folder/'bank.npz';bank.write_bytes(b'exact starts not loaded')
    ledger=folder/'all_slots.npz';ledger.write_bytes(b'all dispositions')
    row=dict(candidate_id='exact',family='jab',start_mode='family_guard',preparation_complete_tick=0,
        reference_path=str(reference),reference_sha256=entry.sha256_file(reference),
        legacy_runtime_path=str(reference),legacy_runtime_sha256=entry.sha256_file(reference),
        source_model_path=str(model),source_model_sha256=entry.sha256_file(model))
    catalog=save(folder/'catalog.json',dict(schema='task10b_phase2_reference_catalog_v1',cells=[row],
        historical_documentary_path='/nonexistent/old/accepted/body.xml'))
    domain=save(folder/'domain.json',dict(schema='Task10BPhase2Domain/v1',
        all_slot_ledger=dict(path='all_slots.npz',sha256=entry.sha256_file(ledger))))
    target=save(folder/'target.json',dict(schema='Task10BPhase2RegisteredTargets/v1'))
    manifest=save(folder/'manifest.json',dict(schema='task10b_phase2_expanded_manifest_v1',
        catalog_path=catalog['path'],catalog_sha256=catalog['sha256'],domain_path=domain['path'],
        domain_sha256=domain['sha256'],target_geometry_path=target['path'],target_geometry_sha256=target['sha256'],
        trials=[dict(trial_id='trial',candidate_id='exact',start_bank_path=str(bank),start_bank_sha256=entry.sha256_file(bank))]))
    record={key:str(folder/(key+'.json')) for key in inputs.KEYS}
    for key in inputs.KEYS:
        if key!='manifest':save(Path(record[key]),dict(schema='task10b_phase2_evaluation_route_v1') if key=='route' else {})
    monkeypatch.setattr(entry.parent,'load_declaration',lambda p:SimpleNamespace(start_mode='family_guard',bindings=[SimpleNamespace(trial_id='trial')]))
    return folder,record,row,Path(catalog['path'])


def call(artifacts,output):
    _,record,_,_=artifacts
    return entry.execute(*[record[k] for k in inputs.KEYS],output,release='accepted',limits='bounded')


@pytest.mark.parametrize('clock_pass,started,reported,expected',[(True,4,4,True),(False,4,4,False),(True,8,4,False),(True,0,0,True)])
def test_direct_evaluator_once_first_write_and_clock_count_fail_closed(artifacts,tmp_path,monkeypatch,clock_pass,started,reported,expected):
    called=[];original_writer=entry.parent.write_json;original_evaluator=entry.parent.evaluate_policy
    def fake(*args,**kwargs):
        out=Path(args[7]);out.mkdir()
        assert entry.parent.evaluate_policy is entry.clocks.evaluate_policy
        assert entry.parent.execution_identity is entry.clock_execution_identity
        called.append(True)
        entry.parent.write_json(out/'launch.json',dict(schema='task10b_phase2_expanded_native_launch_v1'))
        if started:
            batch=out/'batch_0000';batch.mkdir()
            entry.parent.write_json(batch/'expanded_scope.json',dict(historical_inner_planning_label='D0'))
        result=dict(schema='task10b_phase2_expanded_execution_v1',status='executed_pending_audit_and_video',native_attempts_started=started,qualification_passed=False)
        entry.parent.write_json(out/'execution.json',result)
        return result
    monkeypatch.setattr(entry.parent,'execute',fake)
    monkeypatch.setattr(entry.clock_audit,'audit_batch',lambda p:dict(passed=clock_pass,trials=reported))
    out=tmp_path/'run';result=call(artifacts,out)
    assert called==[True] and result['event_clocks_pass'] is expected
    assert result['status']==('executed_pending_audit_and_video' if expected else 'failed')
    assert entry.parent.evaluate_policy is original_evaluator and entry.parent.write_json is original_writer
    receipt=json.loads((out/entry.SIDECAR).read_text())
    assert receipt['execution']==entry.clocks.binding(out/'execution.json') and receipt['qualification'] is False
    assert json.loads((out/'execution.json').read_text())['status']=='executed_pending_audit_and_video'
    assert json.loads((out/'launch.json').read_text())['direct_consumer_version']==entry.VERSION
    assert json.loads((out/'launch.json').read_text())['schema']==entry.LAUNCH_SCHEMA
    if started:
        scope=json.loads((out/'batch_0000/expanded_scope.json').read_text())
        assert 'historical_inner_planning_label' not in scope and scope['first_write_event_metadata'] is True


def test_all_hooks_restored_when_native_raises_before_return(artifacts,tmp_path,monkeypatch):
    old=(entry.parent.evaluate_policy,entry.parent.write_json,entry.parent.execution_identity,entry.continuation_module.execution_identity)
    def fail(*a,**k):raise RuntimeError('preserved native exception')
    monkeypatch.setattr(entry.parent,'execute',fail)
    with pytest.raises(RuntimeError,match='preserved native'):call(artifacts,tmp_path/'run')
    assert old==(entry.parent.evaluate_policy,entry.parent.write_json,entry.parent.execution_identity,entry.continuation_module.execution_identity)


def test_clock_failure_does_not_modify_original_execution(artifacts,tmp_path,monkeypatch):
    def fake(*args,**kwargs):
        out=Path(args[7]);out.mkdir();(out/'batch_0000').mkdir()
        value=dict(status='failed',native_attempts_started=4)
        entry.parent.write_json(out/'execution.json',value);return value
    monkeypatch.setattr(entry.parent,'execute',fake)
    result=call(artifacts,tmp_path/'run')
    assert result['status']=='failed' and result['event_clocks_pass'] is False
    receipt=json.loads((tmp_path/'run'/entry.SIDECAR).read_text())
    assert receipt['batch_audits'][0]['exception']['type']=='FileNotFoundError'


@pytest.mark.parametrize('change',[{'start_mode':'shared_jab_saved_state'},{'preparation_complete_tick':4},{'preparation_complete_tick':None}])
def test_moving_or_implicit_prep_is_refused_before_original(artifacts,tmp_path,monkeypatch,change):
    _,record,row,path=artifacts;catalog=json.loads(path.read_text());catalog['cells'][0].update(change);save(path,catalog)
    manifest=Path(record['manifest']);value=json.loads(manifest.read_text());value['catalog_sha256']=entry.sha256_file(path);save(manifest,value)
    monkeypatch.setattr(entry.parent,'execute',lambda *a,**k:pytest.fail('must refuse before native'))
    with pytest.raises(ValueError,match='prep0'):call(artifacts,tmp_path/'run')


def test_existing_output_and_protocol_never_rewritten(artifacts,tmp_path):
    out=tmp_path/'run';out.mkdir();(out/'prior').write_text('retained')
    with pytest.raises(FileExistsError):call(artifacts,out)
    assert (out/'prior').read_text()=='retained'
    with pytest.raises(ValueError,match='replacement'):
        entry.execute(*[artifacts[1][k] for k in inputs.KEYS],tmp_path/'fresh',release='accepted',limits=None,protocol=object())


def test_continuation_identity_binds_clock_source_without_paths(monkeypatch):
    monkeypatch.setattr(entry,'_identity',lambda *a,**k:dict(frozen={'same':'physical request'},identity='old'))
    result=entry.clock_execution_identity()
    assert result['frozen']['same']=='physical request'
    assert result['frozen']['event_metadata']['source_sha256']=={k:v['sha256'] for k,v in entry.source_bindings().items()}
    assert result['identity']==entry.fingerprint(result['frozen']) and result['identity']!='old'
    assert 'path' not in json.dumps(result['frozen']['event_metadata'])


def test_original_final_claim_terminal_is_preserved_but_arbitrary_escape_refused(artifacts,tmp_path,monkeypatch):
    monkeypatch.setattr(entry.parent,'PROJECT_ROOT',tmp_path)
    claim=tmp_path/'outputs/task10b/p2_final_test_claims/exact'
    def fake(*args,**kwargs):
        out=Path(args[7]);out.mkdir();claim.mkdir(parents=True)
        save(claim/'claim.json',dict(run=str(out)))
        result=dict(status='executed_pending_audit_and_video',native_attempts_started=0,application_run=str(out))
        entry.parent.write_json(out/'execution.json',result)
        entry.parent.write_json(claim/'terminal.json',result)
        with pytest.raises(ValueError,match='external'):
            entry.parent.write_json(tmp_path/'foreign.json',result)
        return result
    monkeypatch.setattr(entry.parent,'execute',fake)
    assert call(artifacts,tmp_path/'run')['event_clocks_pass'] is True
    assert json.loads((claim/'terminal.json').read_text())['application_run']==str(tmp_path/'run')
    assert not (tmp_path/'foreign.json').exists()


def test_validation_does_not_open_final_clock_evidence(monkeypatch):
    monkeypatch.setattr(entry.parent,'read_json',lambda *a:pytest.fail('validation must not open final selection'))
    assert entry.validate_selected_validation_clocks(SimpleNamespace(split='validation'),'unused') is None


def test_final_access_requires_selected_validation_clock_consumer(tmp_path,monkeypatch):
    from human_punching import phase2_expanded_review_v2 as review
    launch=save(tmp_path/'selected_validation/launch.json',{'same':'selected successful validation'})
    route=tmp_path/'route.json';save(route,dict(successful_validation=dict(launch=launch)))
    seen=[]
    def verify(path):
        seen.append(path)
        raise ValueError('missing or failed clock sidecar')
    monkeypatch.setattr(review,'verify_event_evidence',verify)
    with pytest.raises(ValueError,match='clock sidecar'):
        entry.validate_selected_validation_clocks(SimpleNamespace(split='test'),route)
    assert seen==[tmp_path/'selected_validation']


def test_exact_runtime_edges_domain_relative_and_opaque_ancestry(artifacts,tmp_path):
    folder,record,_,_=artifacts
    closure=inputs.files(record,root=tmp_path)
    assert 'inputs/all_slots.npz' in closure and 'inputs/ref.npz' in closure and 'inputs/bank.npz' in closure
    assert len(closure)==14


def test_input_hash_change_and_wrong_feedforward_fail_closed(artifacts,tmp_path):
    _,record,_,_=artifacts
    (artifacts[0]/'ref.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):inputs.files(record,root=tmp_path)


def driver():
    path=Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_expanded_v2.py'
    spec=importlib.util.spec_from_file_location('direct_expanded_test_driver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_transport_dispatch_changes_only_entry(artifacts,monkeypatch):
    d=driver();original=['python','-B','-m','human_punching.phase2_expanded_evaluation','--declaration','same','--wall-seconds','300']
    monkeypatch.setattr(d,'_native',lambda *a:list(original))
    command=d.native_command(dict(mode='expanded-evaluation',inputs=artifacts[1]),300)
    assert command[:3]==original[:3] and command[4:]==original[4:]
    assert command[3]=='human_punching.phase2_expanded_entry_v2'


@pytest.mark.parametrize('mode',['grouped-pilot','media','target-references'])
def test_transport_cannot_expand_scope(mode):
    with pytest.raises(ValueError):driver().validate_spec(mode,{},300,240)


def test_route_budget_and_conditional_proof_delegated_unchanged(artifacts,monkeypatch):
    d=driver();seen=[];proof={'exact':'measured successful route'}
    def validator(mode,value,wall,app,*,proof):
        seen.append((mode,value,wall,app,proof));assert entry.continuation_module.execution_identity is entry.clock_execution_identity
        return {'fresh_transition_cap':0}
    monkeypatch.setattr(d,'_validate',validator)
    result=d.validate_spec('expanded-evaluation',artifacts[1],600,540,proof=proof)
    assert seen==[('expanded-evaluation',artifacts[1],600,540,proof)]
    assert result['direct_consumer_version']==entry.VERSION and result['fresh_transition_cap']==0


def test_required_source_closure_includes_unchanged_clock_and_restores_resolver(monkeypatch):
    d=driver();old=d.parent.semantic_files
    monkeypatch.setattr(d,'_inputs',lambda *a,**k:{})
    closure=d.required_inputs(dict(mode='expanded-evaluation',inputs={}))
    assert 'src/human_punching/phase2_expanded_clocks_v1.py' in closure
    assert 'src/human_punching/phase2_expanded_clock_audit_v1.py' in closure
    assert d.parent.semantic_files is old
    monkeypatch.setattr(d,'_inputs',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('closure')))
    with pytest.raises(RuntimeError):d.required_inputs(dict(mode='expanded-evaluation',inputs={}))
    assert d.parent.semantic_files is old


def test_new_batch_script_only_submission_argument(monkeypatch):
    d=driver();old=['sbatch',str(d.ROOT/'scripts/task10b_phase2_expand.sbatch'),'samepacket']
    monkeypatch.setattr(d,'_submission',lambda *a:list(old))
    assert d.submission_command({},None)==['sbatch',str(d.ROOT/'scripts/task10b_phase2_expanded_v2.sbatch'),'samepacket']


def test_conditional_freeze_new_schema_view_requires_clock_verification(tmp_path,monkeypatch):
    from human_punching import phase2_expanded_review_v2 as review
    d=driver();p=tmp_path/'launch.json';save(p,dict(schema=entry.LAUNCH_SCHEMA,direct_consumer_version=entry.VERSION))
    called=[];original=d.parent.read
    monkeypatch.setattr(review,'verify_event_evidence',lambda path:called.append(path) or {'passed':True})
    monkeypatch.setattr(d,'_freeze',lambda path:d.parent.read(path))
    result=d.successful_route_freeze(p)
    assert called==[tmp_path] and result['schema']=='task10b_phase2_expanded_native_launch_v1'
    assert json.loads(p.read_text())['schema']==entry.LAUNCH_SCHEMA and d.parent.read is original
    monkeypatch.setattr(review,'verify_event_evidence',lambda path:(_ for _ in ()).throw(ValueError('failed clocks')))
    with pytest.raises(ValueError,match='clocks'):d.successful_route_freeze(p)
    assert d.parent.read is original


def audit_driver():
    path=Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_expanded_audit_v2.py'
    spec=importlib.util.spec_from_file_location('direct_expanded_test_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('clock_pass,raw_pass',[(True,True),(False,True),(True,False)])
def test_raw_audit_preserves_new_native_schema_and_requires_clock_review(tmp_path,monkeypatch,clock_pass,raw_pass):
    audit=audit_driver();monkeypatch.setattr(audit,'ROOT',tmp_path)
    run=tmp_path/'outputs/task10b/run';run.mkdir(parents=True)
    launch=save(run/'launch.json',dict(schema=entry.LAUNCH_SCHEMA,direct_consumer_version=entry.VERSION))
    old_json=audit.parent.json
    def verify(path):
        if not clock_pass:raise ValueError('failed clock evidence retained')
        return dict(passed=True)
    def raw(path,out):
        assert audit.parent.json.loads((path/'launch.json').read_text())['schema']=='task10b_phase2_expanded_native_launch_v1'
        out.mkdir();save(out/'audit.json',dict(passed=raw_pass))
        return dict(schema='task10b_phase2_independent_trace_audit_v1',passed=raw_pass,status='passed' if raw_pass else 'failed',panels_audited=1,trials_audited=4)
    monkeypatch.setattr(audit,'verify_event_evidence',verify);monkeypatch.setattr(audit.parent,'audit_run',raw)
    result=audit.audit_run(run,tmp_path/'outputs/task10b/audit')
    assert result['passed']==(clock_pass and raw_pass) and audit.parent.json is old_json
    assert entry.clocks.binding(run/'launch.json')==launch and result['native_artifacts_rewritten'] is False
    assert (tmp_path/'outputs/task10b/audit/original_raw_audit/audit.json').exists()


def test_raw_audit_refuses_historical_schema_instead_of_relabeling(tmp_path,monkeypatch):
    audit=audit_driver();monkeypatch.setattr(audit,'ROOT',tmp_path)
    run=tmp_path/'outputs/task10b/run';run.mkdir(parents=True)
    save(run/'launch.json',dict(schema='task10b_phase2_expanded_native_launch_v1'))
    with pytest.raises(ValueError,match='historical'):audit.audit_run(run,tmp_path/'outputs/task10b/audit')
    assert not (tmp_path/'outputs/task10b/audit').exists()
