"""Clock-consumer regressions using saved synthetic arrays, no native runtime."""
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

from human_punching import phase2_expanded_review_v2 as review
from human_punching import phase2_target_catalog
from human_punching.phase2_domain import fingerprint

SPEC=importlib.util.spec_from_file_location("clock_review_saved_fixture",Path(__file__).with_name("test_phase2_expanded_clocks_v1.py"))
fixture=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(fixture)


def encoded(path,value):
    path.write_text(json.dumps(value));return review.binding(path)


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(phase2_target_catalog,"PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(review.entry,"identity_scope",nullcontext)
    run=tmp_path/"run";run.mkdir()
    monkeypatch.setattr(review.parent,"verified_run_ancestry",lambda _:dict(run_directories=[str(run)]))
    sources={k:encoded(tmp_path/(k+".json"),dict(synthetic_source=k)) for k in ("entry","evaluator","clocks","clock_audit")}
    monkeypatch.setattr(review.entry,"source_bindings",lambda:deepcopy(sources))
    protocol=fixture.protocol.__wrapped__(tmp_path)
    batch=fixture.saved_batch((protocol[0],run/"batch_0000"))
    with np.load(batch/'raw_trace.npz') as saved:raw={k:saved[k].copy() for k in saved.files}
    clock=np.zeros((9,1),dtype=np.float32)
    for i in range(1,len(clock)):clock[i]=clock[i-1]+np.float32(.005)
    np.savez(batch/'raw_trace.npz',**raw,time=clock)
    summary=json.loads((batch/'summary.json').read_text());summary['actual_physics_steps']=8
    encoded(batch/'summary.json',summary)
    spec=importlib.util.spec_from_file_location('clock_review_panel_fixture',Path(__file__).with_name('test_phase2_qualification.py'))
    panel_fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(panel_fixture)
    declaration=encoded(run/"declaration.json",asdict(panel_fixture.declaration()))
    manifest=encoded(tmp_path/"manifest.json",dict(synthetic_manifest=True))
    frozen=dict(event_metadata=dict(version=review.entry.VERSION,event_clock_version=review.entry.clocks.VERSION,
        source_sha256={k:v['sha256'] for k,v in sources.items()}),
        input_sha256=dict(declaration=declaration['sha256'],manifest=manifest['sha256']))
    identity=fingerprint(frozen)
    encoded(run/"execution_identity.json",dict(frozen=frozen,identity=identity,
        input_paths=dict(declaration=declaration['path'],manifest=manifest['path'])))
    common=dict(direct_consumer_version=review.entry.VERSION,event_clock_version=review.entry.clocks.VERSION,
        event_clock_review_receipt_required=review.entry.SIDECAR,execution_identity=identity)
    execution=encoded(run/"execution.json",dict(schema='task10b_phase2_expanded_execution_v1',native_attempts_started=1,**common))
    encoded(run/"launch.json",dict(schema=review.LAUNCH,event_metadata_sources=sources,**common))
    side=dict(schema=review.entry.VERSION,event_clock_version=review.entry.clocks.VERSION,passed=True,
        all_native_batches_required=True,sources=sources,execution=execution,declaration=declaration,manifest=manifest,
        native_attempts_started=1,batch_audits=[dict(batch=batch.name,**review.entry.clock_audit.audit_batch(batch))])
    encoded(run/review.entry.SIDECAR,side)
    return run,side


def test_saved_event_evidence_recomputed_not_trusted(evidence):
    run,_=evidence;result=review.verify_event_evidence(run)
    assert result['passed'] and result['runs'][0]['native_attempts_started']==1
    assert not result['qualification'] and result['new_physics_steps']==0


def test_native_reserialization_of_optional_defaults_keeps_semantic_identity(evidence,tmp_path):
    run,side=evidence
    value=json.loads((run/'declaration.json').read_text())
    value.pop('start_mode');value.pop('predicate_version')
    side['declaration']=encoded(tmp_path/'input_with_omitted_defaults.json',value)
    p=run/'execution_identity.json';ident=json.loads(p.read_text())
    ident['input_paths']['declaration']=side['declaration']['path']
    ident['frozen']['input_sha256']['declaration']=side['declaration']['sha256']
    ident['identity']=fingerprint(ident['frozen']);encoded(p,ident)
    for name in ('launch','execution'):
        p=run/(name+'.json');v=json.loads(p.read_text());v['execution_identity']=ident['identity'];b=encoded(p,v)
        if name=='execution':side['execution']=b
    encoded(run/review.entry.SIDECAR,side)
    assert review.verify_event_evidence(run)['passed']


def test_verified_retained_prefix_is_metadata_only_not_cycle_credit(evidence):
    run,side=evidence;batch=run/'batch_0000'
    p=batch/'plan.json';plan=json.loads(p.read_text());plan['planned_physics_steps']=12;encoded(p,plan)
    audit=review.entry.clock_audit.audit_batch(batch)
    assert audit['passed'] and audit['prefix_only'] and not audit['completed_cycle_credit']
    side['batch_audits']=[dict(batch=batch.name,**audit)];encoded(run/review.entry.SIDECAR,side)
    assert review.verify_event_evidence(run)['qualification'] is False
    # Metadata cannot stand in for independent physical attempt resolution.
    from human_punching.phase2_qualification import qualify_panel
    spec=importlib.util.spec_from_file_location('prefix_qualification_fixture',Path(__file__).with_name('test_phase2_qualification.py'))
    qfixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(qfixture)
    panel=qfixture.declaration();rows=qfixture.success_rows(panel)
    rows[0].update(execution_complete=False)
    rows[1].update(status='unattempted')
    result=qualify_panel(panel,rows)
    assert result['successes']==190
    assert not result['trials'][0]['passed'] and not result['trials'][1]['passed']


@pytest.mark.parametrize('mutation',['missing','failed','false_batch','omit_batch','extra_batch','wrong_execution','changed_source','old_launch','old_identity','wrong_manifest','shared'])
def test_false_incomplete_or_wrong_source_sidecar_never_grants_credit(evidence,mutation):
    run,side=evidence
    if mutation=='missing':(run/review.entry.SIDECAR).unlink()
    elif mutation=='failed':side['passed']=False
    elif mutation=='false_batch':side['batch_audits'][0]['checks']['prep0_declared']=False
    elif mutation=='omit_batch':side['batch_audits']=[]
    elif mutation=='extra_batch':side['batch_audits'].append(deepcopy(side['batch_audits'][0]))
    elif mutation=='wrong_execution':side['execution']['sha256']='f'*64
    elif mutation=='changed_source':Path(side['sources']['clocks']['path']).write_text('{}')
    elif mutation=='old_launch':
        p=run/'launch.json';value=json.loads(p.read_text());value.pop('direct_consumer_version');encoded(p,value)
    elif mutation=='old_identity':
        p=run/'execution_identity.json';value=json.loads(p.read_text());value['frozen'].pop('event_metadata');encoded(p,value)
    elif mutation=='wrong_manifest':side['manifest']=encoded(run/'other_manifest.json',dict(synthetic_other=True))
    elif mutation=='shared':
        side['declaration']=encoded(run/'declaration.json',dict(start_mode='shared_jab_saved_state'))
    if mutation!='missing':encoded(run/review.entry.SIDECAR,side)
    with pytest.raises((ValueError,FileNotFoundError)):review.verify_event_evidence(run)


@pytest.mark.parametrize('mutation',['prep','launch_state','readiness'])
def test_passing_saved_claim_cannot_hide_changed_raw_or_event_metadata(evidence,mutation):
    run,_=evidence;batch=run/'batch_0000'
    if mutation=='prep':
        p=batch/'summary.json';value=json.loads(p.read_text());value['trials'][0]['preparation_complete_time_s']=.02;encoded(p,value)
    else:
        p=batch/'raw_trace.npz'
        with np.load(p) as values:raw={k:values[k].copy() for k in values.files}
        raw['launched' if mutation=='launch_state' else 'policy_readiness'][-1,0]=False
        np.savez(p,**raw)
    with pytest.raises(ValueError,match='recomputed'):review.verify_event_evidence(run)


def test_every_continuation_ancestor_requires_clock_evidence(evidence,tmp_path,monkeypatch):
    run,_=evidence;ancestor=tmp_path/'ancestor';ancestor.mkdir()
    monkeypatch.setattr(review.parent,'verified_run_ancestry',lambda _:dict(run_directories=[str(run),str(ancestor)]))
    with pytest.raises(FileNotFoundError):review.verify_event_evidence(run)


def test_original_review_runs_only_after_mandatory_clock_verification(evidence,tmp_path,monkeypatch):
    run,_=evidence;calls=[]
    original=encoded(tmp_path/'original_review.json',dict(run=str(run)))
    config=encoded(tmp_path/'review.json',dict(schema=review.PANEL,original_review_configuration=original,
        event_clock_entry=review.binding(run/review.entry.SIDECAR)))
    monkeypatch.setattr(review,'_typed_adjudicate',lambda p,r:(calls.append(p) or dict(passed=False,successes=0),dict(original=True)))
    report,receipt=review.adjudicate(config['path'])
    assert len(calls)==1 and report['passed'] is False and receipt['qualification_math_changed'] is False
    (run/review.entry.SIDECAR).write_text('{}')
    with pytest.raises(ValueError):review.adjudicate(config['path'])
    assert len(calls)==1


def test_old_review_receipt_is_never_a_substitute(tmp_path,monkeypatch):
    monkeypatch.setattr(phase2_target_catalog,'PROJECT_ROOT',tmp_path)
    old=encoded(tmp_path/'old_review.json',dict(schema=review.parent.RECEIPT,source_sha256=review.parent.sha256_file(review.parent.__file__)))
    with pytest.raises(ValueError,match='old generic'):review.reviewed_panel(old)


def test_original_reviewer_rejects_new_native_schema_and_explicit_typed_view_preserves_bytes(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("direct_review_original_fixture",Path(__file__).with_name("test_phase2_review.py"))
    oldfixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldfixture)
    example=oldfixture.example.__wrapped__(tmp_path,monkeypatch)
    _,config,*_=example;run=Path(config['run']);launch=json.loads((run/'launch.json').read_text())
    launch['schema']=review.LAUNCH;config['inputs']['launch']=encoded(run/'launch.json',launch)
    path=tmp_path/'original_review_plan.json';encoded(path,config)
    before=(run/'launch.json').read_bytes()
    with pytest.raises(ValueError,match='identities differ'):review.parent.adjudicate(path)
    from human_punching import phase2_continuation
    from human_punching.wbc import provenance
    monkeypatch.setattr(provenance,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(phase2_continuation,'verified_run_ancestry',lambda _:dict(run_directories=[str(run)]))
    report,_=review._typed_adjudicate(path,run)
    assert report['passed'] and report['successes']==192
    assert (run/'launch.json').read_bytes()==before
