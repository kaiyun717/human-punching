"""Immutable never-attempted continuation and external-stop ledger recovery."""
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from human_punching import phase2_continuation as continuation
from human_punching.phase2_domain import fingerprint
from human_punching.wbc.provenance import sha256_file


def write(path,value):
    path.write_text(json.dumps(value));return {'path':str(path),'sha256':sha256_file(path)}


def parent_fixture(tmp_path,*,terminal_application=True,real_declaration=False):
    packet=tmp_path/'packet';parent=packet/'run';parent.mkdir(parents=True)
    ids=[f'synthetic_{i:03d}' for i in range(96)]
    declaration=SimpleNamespace(identity='a'*64,family='jab',seed=21021,split='test',
        bindings=[SimpleNamespace(trial_id=trial) for trial in ids])
    raw={'bindings':[{'trial_id':trial} for trial in ids]}
    if real_declaration:
        from human_punching.phase2_qualification import PanelDeclaration,TrialBinding
        from human_punching.wbc.splits import assign_trajectory_groups
        groups=tuple(g for g,s in assign_trajectory_groups(range(1000)).items() if s=='validation')[:8]
        bindings=tuple(TrialBinding(ids[index],group,1,timing,start,'a'*64,'b'*64,(.8,.2,1.23))
            for index,(group,timing,start) in enumerate((g,t,s) for g in groups for t in (.6,.675,.75) for s in range(4)))
        declaration=PanelDeclaration('jab','validation',21021,groups,(.6,.75),.675,bindings,(ids[0],),
            '1'*64,'2'*64,'3'*64,'4'*64,'5'*64,8192,'6'*64)
        raw=asdict(declaration)
    declaration_path=parent/'declaration.json';write(declaration_path,raw)
    frozen={'declaration_identity':declaration.identity,'input_sha256':{'declaration':sha256_file(declaration_path)},
        'source_identity':'synthetic_same_source','runtime':'synthetic_same_runtime'}
    identity=dict(schema=continuation.IDENTITY,identity=fingerprint(frozen),frozen=frozen,
        input_paths={'declaration':str(declaration_path)},slurm_job_id='SYNTHETIC_UNIT_TEST_ONLY')
    identity_bound=write(parent/'execution_identity.json',identity)
    outcomes=[dict(trial_id=trial,status='unattempted',physical_execution_attempted=False,reason='not reached') for trial in ids]
    outcomes[0]=dict(trial_id=ids[0],status='executed',execution_complete=True,physical_execution_attempted=True,
        original_nested_result={'retain_this_value':[1,2,3]})
    outcomes[1]=dict(trial_id=ids[1],status='timeout',execution_complete=False,physical_execution_attempted=True)
    outcomes[2]=dict(trial_id=ids[2],status='generation_failure',physical_execution_attempted=False)
    outcomes[3]=dict(trial_id=ids[3],status='rejected',physical_execution_attempted=False)
    outcomes_bound=write(parent/'outcomes.json',outcomes)
    execution=dict(schema='task10b_phase2_expanded_execution_v1',status='failed',planned_trials=96,
        application_run=str(parent),
        execution_identity=identity['identity'],outcomes_sha256=outcomes_bound['sha256'],logical_origin_run=str(parent))
    execution_bound=write(parent/('execution.json' if terminal_application else 'synthetic_nonterminal_fixture.json'),execution)
    config=dict(schema=continuation.CONFIG,parent_run=str(parent),parent_execution_identity=identity_bound,
        parent_execution=execution_bound,parent_outcomes=outcomes_bound,trial_ids=ids[4:])
    config_path=tmp_path/'continuation.json';write(config_path,config)
    write(parent/'native_attempt_intent_0000.json',dict(execution_identity=identity['identity'],trial_ids=ids[:2]))
    return declaration,identity,parent,config_path,config,outcomes


def test_only_exact_never_attempted_remainder_can_continue_and_prior_outcomes_are_unchanged(tmp_path):
    declaration,identity,parent,path,config,outcomes=parent_fixture(tmp_path)
    result=continuation.load_continuation(path,identity,declaration)
    assert len(result['trial_ids'])==92 and result['outcomes']==outcomes
    assert result['inherited_attempted_ids']==[f'synthetic_{i:03d}' for i in range(4)]
    assert result['outcomes'][0]['original_nested_result']=={'retain_this_value':[1,2,3]}


@pytest.mark.parametrize('mutation',['retry_timeout','retry_completed','drop_unattempted','duplicate'])
def test_no_selected_retries_or_silent_denominator_reduction(tmp_path,mutation):
    declaration,identity,parent,path,config,_=parent_fixture(tmp_path)
    if mutation=='retry_timeout':config['trial_ids'].append('synthetic_001')
    if mutation=='retry_completed':config['trial_ids'].append('synthetic_000')
    if mutation=='drop_unattempted':config['trial_ids'].pop()
    if mutation=='duplicate':config['trial_ids'].append(config['trial_ids'][0])
    write(path,config)
    with pytest.raises(ValueError,match='all and only never-attempted'):
        continuation.load_continuation(path,identity,declaration)


def test_durable_attempt_intent_blocks_false_unattempted_relabeling(tmp_path):
    declaration,identity,parent,path,config,_=parent_fixture(tmp_path)
    write(parent/'native_attempt_intent_0001.json',dict(execution_identity=identity['identity'],trial_ids=['synthetic_004']))
    with pytest.raises(ValueError,match='durably intended physical attempt'):
        continuation.load_continuation(path,identity,declaration)


def test_changed_weights_source_runtime_or_parent_bytes_are_rejected(tmp_path):
    declaration,identity,parent,path,config,_=parent_fixture(tmp_path)
    changed={**identity,'identity':'b'*64}
    with pytest.raises(ValueError,match='changes frozen'):continuation.load_continuation(path,changed,declaration)
    (parent/'outcomes.json').write_text('[]')
    with pytest.raises(ValueError,match='parent evidence changed'):continuation.load_continuation(path,identity,declaration)


def test_test_claim_forms_a_single_append_only_child_chain(tmp_path,monkeypatch):
    declaration,identity,parent,path,config,_=parent_fixture(tmp_path)
    monkeypatch.setattr(continuation,'PROJECT_ROOT',tmp_path)
    first=continuation.claim_test_session(parent,declaration,identity)
    context=continuation.load_continuation(path,identity,declaration)
    child=tmp_path/'child';child.mkdir()
    receipt=continuation.claim_test_session(child,declaration,identity,context)
    assert receipt['path']==first['path'] and receipt['session_path']!=first['session_path']
    assert receipt['logical_origin_run']==str(parent)
    competing=tmp_path/'competing';competing.mkdir()
    with pytest.raises(FileExistsError):continuation.claim_test_session(competing,declaration,identity,context)
    with pytest.raises(ValueError,match='already has an exclusive claim'):
        continuation.claim_test_session(competing,declaration,identity)
    terminal_before=json.loads((__import__('pathlib').Path(first['path'])/'claim.json').read_text())
    assert terminal_before['run']==str(parent)


def test_preflight_failure_without_any_attempt_can_continue_without_consuming_outcomes(tmp_path,monkeypatch):
    declaration,identity,parent,path,config,outcomes=parent_fixture(tmp_path)
    # A separate untouched preflight-only fixture has no durable intents.
    clean=tmp_path/'preflight_only';clean.mkdir()
    clean_ids=[dict(trial_id=b.trial_id,status='unattempted',physical_execution_attempted=False) for b in declaration.bindings]
    declared=write(clean/'declaration.json',{'bindings':[{'trial_id':b.trial_id} for b in declaration.bindings]})
    frozen={**identity['frozen'],'input_sha256':{'declaration':declared['sha256']}}
    clean_identity={**identity,'frozen':frozen,'identity':fingerprint(frozen),'input_paths':{'declaration':declared['path']}}
    identity_bound=write(clean/'execution_identity.json',clean_identity);outcomes_bound=write(clean/'outcomes.json',clean_ids)
    execution_bound=write(clean/'execution.json',dict(schema='task10b_phase2_expanded_execution_v1',status='failed',planned_trials=96,
        application_run=str(clean),
        execution_identity=clean_identity['identity'],outcomes_sha256=outcomes_bound['sha256']))
    clean_config=dict(schema=continuation.CONFIG,parent_run=str(clean),parent_execution_identity=identity_bound,
        parent_execution=execution_bound,parent_outcomes=outcomes_bound,trial_ids=[b.trial_id for b in declaration.bindings])
    write(path,clean_config)
    result=continuation.load_continuation(path,clean_identity,declaration)
    assert len(result['trial_ids'])==96 and all(r['status']=='unattempted' for r in result['outcomes'])


def test_external_preemption_recovery_preserves_prior_results_and_marks_lost_attempt_unknown(tmp_path):
    declaration,identity,parent,path,config,outcomes=parent_fixture(tmp_path,terminal_application=False)
    # Last durable full ledger predates the active second request's termination.
    ledger=[dict(trial_id=b.trial_id,status='unattempted',physical_execution_attempted=False) for b in declaration.bindings]
    ledger[0]=outcomes[0]
    write(parent/'ledger_0000.json',ledger)
    packet=parent.parent;packet_bound=write(packet/'packet.json',{'synthetic_packet':True})
    terminal=write(packet/'terminal_accounting.json',dict(schema='task10b_phase2_terminal_accounting_v1',
        state='PREEMPTED',job_id=identity['slurm_job_id'],packet_sha256=packet_bound['sha256'],allocated_gpu_seconds=42))
    output=tmp_path/'recovered';report=continuation.recover_interrupted_parent(parent,terminal['path'],output)
    recovered=json.loads((output/'outcomes.json').read_text())
    assert recovered[0]==outcomes[0]
    assert recovered[1]['status']=='executed' and recovered[1]['execution_extent_unknown']
    assert recovered[1]['physical_execution_attempted'] and not recovered[1]['execution_complete']
    assert all(row['status']=='unattempted' for row in recovered[2:])
    assert report['unattempted_attempts']==94 and report['unknown_or_lost_extent_blocks_qualification']
    next_config={**config,'parent_execution':{'path':str(output/'execution.json'),'sha256':sha256_file(output/'execution.json')},
        'parent_outcomes':{'path':str(output/'outcomes.json'),'sha256':sha256_file(output/'outcomes.json')},
        'trial_ids':[row['trial_id'] for row in recovered if row['status']=='unattempted']}
    write(path,next_config)
    assert len(continuation.load_continuation(path,identity,declaration)['trial_ids'])==94


def test_interrupted_recovery_rejects_nonterminal_or_wrong_accounting(tmp_path):
    declaration,identity,parent,path,config,_=parent_fixture(tmp_path,terminal_application=False)
    packet=parent.parent;packet_bound=write(packet/'packet.json',{'synthetic_packet':True})
    account=packet/'terminal_accounting.json'
    write(account,dict(schema='task10b_phase2_terminal_accounting_v1',state='RUNNING',job_id=identity['slurm_job_id'],packet_sha256=packet_bound['sha256']))
    with pytest.raises(ValueError,match='same terminal accounted packet'):
        continuation.recover_interrupted_parent(parent,account,tmp_path/'not_created')
    assert not (tmp_path/'not_created').exists()


def test_child_killed_before_its_first_ledger_retains_every_ancestor_outcome(tmp_path):
    declaration,identity,parent,path,config,outcomes=parent_fixture(tmp_path)
    context=continuation.load_continuation(path,identity,declaration)
    packet=tmp_path/'child_packet';child=packet/'run';child.mkdir(parents=True)
    child_identity={**identity,'slurm_job_id':'SYNTHETIC_CHILD'}
    write(child/'execution_identity.json',child_identity)
    write(child/'continuation.json',{key:value for key,value in context.items() if key!='outcomes'})
    packet_bound=write(packet/'packet.json',{'synthetic_child_packet':True})
    terminal=write(packet/'terminal_accounting.json',dict(schema='task10b_phase2_terminal_accounting_v1',
        state='PREEMPTED',job_id='SYNTHETIC_CHILD',packet_sha256=packet_bound['sha256']))
    output=tmp_path/'child_recovered';report=continuation.recover_interrupted_parent(child,terminal['path'],output)
    assert json.loads((output/'outcomes.json').read_text())==outcomes
    assert report['logical_origin_run']==str(parent) and report['unattempted_attempts']==92


def test_unclaimed_intermediate_run_cannot_fork_its_last_claimed_ancestor(tmp_path,monkeypatch):
    declaration,identity,parent,path,config,_=parent_fixture(tmp_path)
    monkeypatch.setattr(continuation,'PROJECT_ROOT',tmp_path)
    continuation.claim_test_session(parent,declaration,identity)
    context=continuation.load_continuation(path,identity,declaration)
    unclaimed=tmp_path/'stopped_before_session_claim';unclaimed.mkdir()
    next_child=tmp_path/'next_child';next_child.mkdir()
    descendant={**context,'parent_run':str(unclaimed),'parent_execution_sha256':'c'*64}
    with pytest.raises(ValueError,match='last claimed terminal ancestor'):
        continuation.claim_test_session(next_child,declaration,identity,descendant)


def test_verified_ancestry_allows_only_exact_hash_bound_same_declaration_parent_roots(tmp_path):
    declaration,identity,parent,path,config,_=parent_fixture(tmp_path,real_declaration=True)
    context=continuation.load_continuation(path,identity,declaration)
    child=tmp_path/'child';child.mkdir()
    write(child/'execution_identity.json',identity)
    write(child/'continuation.json',{key:value for key,value in context.items() if key!='outcomes'})
    ancestry=continuation.verified_run_ancestry(child)
    assert ancestry['run_directories']==[str(child),str(parent)] and ancestry['depth']==2
    (parent/'outcomes.json').write_text('[]')
    with pytest.raises(ValueError,match='parent evidence changed'):continuation.verified_run_ancestry(child)
