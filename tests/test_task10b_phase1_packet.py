"""Phase 1 execution cannot bypass pilot review or alter scientific input identities."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('phase1_packet', ROOT/'scripts/task10b_phase1.py')
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def scope(mode='pilot'):
    allocation, application, products, worlds = packet.LIMITS[mode]
    return dict(schema=packet.SCHEMA,task='10B',program_phase=1,mode=mode,training=False,
        learning_updates=0,automatic_followup=False,solver='accepted_g1',feedforward='original',
        allocation_wall_seconds=allocation,application_wall_seconds=application,
        products_and_caches_bytes=products,conditions=4,environments_per_condition=worlds,
        requested_full_cycle_attempts=4*worlds,bank_rows=list(range(worlds)),
        partition='mit_normal_gpu',excluded_nodes=['node3006'])


@pytest.mark.parametrize('mode',['pilot','qualification'])
def test_frozen_scope_and_attempt_counts(mode):
    value=scope(mode)
    packet.validate_scope(value)
    assert value['allocation_wall_seconds']<=3600
    assert value['requested_full_cycle_attempts']==(64 if mode=='pilot' else 256)


@pytest.mark.parametrize('key,value', [('training',True),('learning_updates',1),('program_phase',2),
    ('mode','fine_tuning'),('automatic_followup',True),('solver','historical'),('feedforward','zero'),
    ('excluded_nodes',[]),('excluded_nodes',['node3006','--time=100']),('partition','other'),
    ('requested_full_cycle_attempts',1024),('allocation_wall_seconds',14401)])
def test_scope_cannot_expand(key,value):
    record=scope();record[key]=value
    with pytest.raises(ValueError):packet.validate_scope(record)


@pytest.fixture
def review_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(packet,'ROOT',tmp_path)
    folder=tmp_path/'pilot';folder.mkdir()
    previous=dict(mode='pilot',bundle='bundle',release='release',runtime={'pin':'same'},
        interpreter_path='python',interpreter_sha256='interpreter',predicate='pilot-predicate.json',
        files={'source.py':'source','pilot-predicate.json':'initial-predicate'})
    def save(path,value):path.write_text(json.dumps(value))
    prior=folder/'packet.json';save(prior,previous)
    execution=folder/'execution.json';save(execution,dict(status='passed',packet_sha256=packet.sha256_file(prior)))
    audit=folder/'independent.json';save(audit,dict(status='passed',packet_sha256=packet.sha256_file(prior)))
    review=folder/'review.json'
    save(review,dict(status='reviewed',predicate_sha256='final-predicate',
        pilot_packet_sha256=packet.sha256_file(prior),pilot_execution_sha256=packet.sha256_file(execution),
        pilot_audit_sha256=packet.sha256_file(audit),thresholds_changed=False))
    record=dict(previous,mode='qualification',predicate='final-predicate.json',predicate_sha256='final-predicate',
        files={'source.py':'source','final-predicate.json':'final-predicate'})
    for key,path in dict(pilot_receipt=execution,pilot_packet=prior,pilot_audit=audit,review=review).items():
        record[key]=str(path.relative_to(tmp_path));record[key+'_sha256']=packet.sha256_file(path)
        record['files'][record[key]]=record[key+'_sha256']
    monkeypatch.setattr(packet,'verify',lambda *args,**kwargs:previous)
    return record,review,audit,execution


def test_qualification_requires_exact_reviewed_bindings(review_fixture):
    record,*_=review_fixture
    packet.verify_pilot_review(record)


@pytest.mark.parametrize('key,value',[('status','pending'),('thresholds_changed',True),
    ('predicate_sha256','different'),('pilot_execution_sha256','different')])
def test_review_cannot_relabel_threshold_changes_or_unreviewed_result(review_fixture,key,value):
    record,review,*_=review_fixture
    payload=json.loads(review.read_text());payload[key]=value
    review.write_text(json.dumps(payload));record['review_sha256']=packet.sha256_file(review)
    with pytest.raises(ValueError,match='Predicate review'):packet.verify_pilot_review(record)


def test_review_requires_independent_audit_of_exact_packet(review_fixture):
    record,_,audit,_=review_fixture
    payload=json.loads(audit.read_text());payload['packet_sha256']='different'
    audit.write_text(json.dumps(payload));record['pilot_audit_sha256']=packet.sha256_file(audit)
    with pytest.raises(ValueError,match='Independent pilot audit'):packet.verify_pilot_review(record)


def test_qualification_changed_source_requires_new_pilot(review_fixture):
    record,*_=review_fixture
    record['files']['source.py']='changed'
    with pytest.raises(ValueError,match='scientific source inputs'):packet.verify_pilot_review(record)


@pytest.mark.parametrize('component',['payload','link','metadata'])
def test_runtime_identity_rejects_changed_components(tmp_path,monkeypatch,component):
    monkeypatch.setattr(packet,'ROOT',tmp_path)
    runtime=tmp_path/'runtime.json';monkeypatch.setattr(packet,'RUNTIME',runtime)
    local=tmp_path/'python';local.mkdir()
    member=local/'stdlib.py';member.write_text('original')
    link=local/'alias';link.symlink_to('stdlib.py')
    manifest=tmp_path/'runtime_interpreter_manifest.json'
    manifest.write_text(json.dumps([dict(path='stdlib.py',bytes=8,sha256=packet.sha256_file(member)),dict(path='alias',symlink='stdlib.py')]))
    metadata=tmp_path/'fake.dist-info/METADATA';metadata.parent.mkdir();metadata.write_text('metadata')
    runtime.write_text(json.dumps(dict(local_python_root=str(local),interpreter_manifest_sha256=packet.sha256_file(manifest),
        distribution_metadata_hashes={'fake/METADATA':dict(bytes=8,sha256=packet.sha256_file(metadata))})))
    monkeypatch.setattr(packet.importlib.metadata,'distribution',lambda name:SimpleNamespace(_path=metadata.parent))
    monkeypatch.setattr(packet,'runtime_record',lambda:dict(declared_pins_match=True))
    record=dict(runtime=dict(declared_pins_match=True))
    packet.verify_runtime_identity(record)
    if component=='payload':member.write_text('modified')
    elif component=='link':
        link.unlink();link.symlink_to('./stdlib.py')
    else:metadata.write_text('modified')
    with pytest.raises(ValueError):packet.verify_runtime_identity(record)


@pytest.mark.parametrize('failure',['verification','cuda'])
def test_preexecution_failure_records_zero_executed_counts(tmp_path,monkeypatch,failure):
    monkeypatch.setattr(packet,'ROOT',tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID','test-only');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    path=tmp_path/'outputs/task10b/failure/packet.json';path.parent.mkdir(parents=True);path.write_text('{}')
    monkeypatch.setattr(packet,'write_json',lambda path,value:Path(path).write_text(json.dumps(value)))
    def verify(path):
        if failure=='verification':raise ValueError('source changed')
        return scope()
    monkeypatch.setattr(packet,'verify',verify)
    def health(path):
        value=dict(status='failed');path.write_text(json.dumps(value));return value
    monkeypatch.setattr(packet,'cuda_health_receipt',health)
    def no_process(*args,**kwargs):raise AssertionError('must not execute')
    monkeypatch.setattr(packet.subprocess,'Popen',no_process)
    assert packet.execute(path)==2
    final=json.loads((path.parent/'execution.json').read_text())
    assert final['status']=='failed' and final['all_native_modes_unattempted']
    assert final['actual_world_physics_transitions']==final['actual_actor_updates']==0
    assert final['native_processes_started']==0


def test_shell_resolves_alias_and_sources_only_phase1(tmp_path):
    root=tmp_path/'physical';root.mkdir();alias=tmp_path/'alias';alias.symlink_to(root)
    runtime=root/'outputs/task10b/p1prep_20261020_01/runtime.env';runtime.parent.mkdir(parents=True);runtime.write_text('# fixture')
    interpreter=root/'.venv-task10b/bin/python';interpreter.parent.mkdir(parents=True)
    interpreter.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$TASK10B_CAPTURE"\n');interpreter.chmod(0o755)
    requested=alias/'outputs/task10b/fixture/packet.json';requested.parent.mkdir();requested.write_text('{}')
    capture=root/'capture.txt'
    result=subprocess.run(['bash',str(ROOT/'scripts/task10b_phase1.sbatch'),str(requested)],cwd=root,
        env=dict(os.environ,SLURM_JOB_ID='test-only',SLURM_SUBMIT_DIR=str(tmp_path),TASK10B_CAPTURE=str(capture)),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert capture.read_text().splitlines()==['-B',str(root/'scripts/task10b_phase1.py'),'execute','--packet',str(requested.resolve())]


def test_paths_cannot_escape_human_root(tmp_path,monkeypatch):
    root=tmp_path/'human';root.mkdir();outside=tmp_path/'external';outside.write_text('outside')
    (root/'link').symlink_to(outside);monkeypatch.setattr(packet,'ROOT',root)
    with pytest.raises(ValueError):packet.local_input('link')


def test_native_probe_owns_run_directory_and_success_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(packet,'ROOT',tmp_path)
    monkeypatch.setattr(packet,'writable_path',lambda path:Path(path))
    monkeypatch.setenv('SLURM_JOB_ID','test-only');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    path=tmp_path/'outputs/task10b/success/packet.json';path.parent.mkdir(parents=True);path.write_text('{}')
    record=dict(scope(),output='outputs/task10b/success/run',bundle='bundle',release='release',predicate='predicate.json')
    monkeypatch.setattr(packet,'verify',lambda path:record)
    monkeypatch.setattr(packet,'write_json',lambda path,value:Path(path).write_text(json.dumps(value)))
    def health(path):
        value=dict(status='passed');path.write_text(json.dumps(value));return value
    monkeypatch.setattr(packet,'cuda_health_receipt',health)
    def start(command,**kwargs):
        output=tmp_path/record['output']
        assert not output.exists(), 'Native probe requires exclusive output creation'
        assert kwargs['cwd']==path.parent
        assert command[3:5]==['human_punching.phase1_probe','--mode']
        output.mkdir();(output/'audit.json').write_text(json.dumps(dict(status='passed',actual_world_physics_transitions=62528,unattempted_conditions=0)))
        return SimpleNamespace(poll=lambda:0,wait=lambda **kw:0)
    monkeypatch.setattr(packet.subprocess,'Popen',start)
    assert packet.execute(path)==0
    final=json.loads((path.parent/'execution.json').read_text())
    assert final['status']=='passed' and final['actual_world_physics_transitions']==62528
    assert final['native_processes_started']==1
