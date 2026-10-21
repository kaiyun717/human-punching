import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from human_punching import phase2_grouped_media_inputs_v1 as media


def driver():
    spec=importlib.util.spec_from_file_location('grouped_media_driver_test',Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_grouped_media_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def parent():
    return [dict(schema='task10b_phase2_grouped_clock_packet_v1',mode='grouped-pilot',training=True,gpus=1,fresh_transition_cap=262144),
        dict(packet_sha256='bound',native_processes_started=1,job_id='123'),
        dict(packet_sha256='bound',state='COMPLETED',allocated_gpu_seconds=1000,job_id='123'),dict(slurm_job_id='123'),'bound']


@pytest.mark.parametrize('state',['COMPLETED','FAILED','TIMEOUT','CANCELLED','PREEMPTED'])
def test_failed_training_retains_saved_media_eligibility(state):
    p=parent();p[2]['state']=state;media.validate_parent(*p)


@pytest.mark.parametrize('section,field,value',[(0,'training',False),(0,'fresh_transition_cap',131072),(0,'gpus',0),
    (1,'native_processes_started',0),(1,'job_id','other'),(1,'packet_sha256','other'),
    (2,'state','RUNNING'),(2,'allocated_gpu_seconds',0),(2,'allocated_gpu_seconds',float('nan')),
    (2,'allocated_gpu_seconds',True),(2,'allocated_gpu_seconds',1801),(3,'slurm_job_id','other')])
def test_different_unaccounted_or_unbounded_parent_rejected(section,field,value):
    p=parent();p[section][field]=value
    with pytest.raises(ValueError):media.validate_parent(*p)


@pytest.mark.parametrize('wall,app,inputs',[(240,180,{'run':'x'}),(300,210,{'run':'x'}),(300,240,{'run':'x','panel':'y'})])
def test_exact_media_envelope_only(wall,app,inputs):
    with pytest.raises(ValueError):driver().validate_spec('media',inputs,wall,app)


def test_original_renderer_union_preserves_six_presampled_and_all_failures():
    renderer=media.renderer_module()
    bindings=[SimpleNamespace(trial_id=str(i),command_id=7,ttc_s=.6,start_index=i%4,identity=str(i)) for i in range(96)]
    declaration=SimpleNamespace(bindings=bindings,video_trial_ids=[str(i) for i in range(6)])
    rows=[dict(trial_id=str(i),disposition='completed_native_attempt',numeric_complete_cycle=True) for i in range(96)]
    assert len(renderer.selection(declaration,rows))==6
    rows[50]['numeric_complete_cycle']=False;rows.pop()
    chosen=renderer.selection(declaration,rows)
    assert {x['trial_id'] for x in chosen}==set(map(str,range(6)))|{'50','95'}


def test_closure_binds_every_retained_file_and_refuses_escaped_symlink(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir();(root/'raw.npz').write_bytes(b'retained')
    added=[];ctx=SimpleNamespace(files={'parent/packet.json':'bound'},add=lambda p:added.append(p))
    monkeypatch.setattr(media,'selected',lambda *a:(ctx,{'planned':6}))
    files,selection=media.files(root,'proof')
    assert files=={'parent/packet.json':'bound'} and added==[root/'raw.npz'] and selection['planned']==6
    outside=tmp_path/'outside';outside.write_text('foreign');(root/'escape').symlink_to(outside)
    with pytest.raises(ValueError,match='escaped'):media.files(root,'proof')


def test_original_native_renderer_command_remains_saved_only(tmp_path):
    d=driver();r=dict(mode='media',inputs={'run':str(tmp_path)},output='outputs/task10b/unused/run',application_wall_seconds=240,files={})
    command=d.legacy.native_command(r,150.)
    assert command[2].endswith('task10b_phase2_expanded_media.py')
    assert command[command.index('--wall-seconds')+1]=='150.0'
    assert '--run' in command and '--product-byte-cap' in command and '-m' not in command
    with pytest.raises(ValueError):d.legacy.native_command(r,120.)


def test_driver_restores_semantic_dependency_even_when_closure_fails(tmp_path,monkeypatch):
    d=driver();(tmp_path/'trace.json').write_text('{}')
    sentinel=d.legacy.semantic_files
    monkeypatch.setattr(d.legacy.base,'local',lambda p:tmp_path)
    monkeypatch.setattr(d.legacy,'bound',lambda p:tmp_path/'proof')
    monkeypatch.setattr(d,'files',lambda *a:(_ for _ in ()).throw(ValueError('closed failure')))
    monkeypatch.setattr(d,'_inputs',lambda record,**kw:d.legacy.semantic_files([tmp_path/'trace.json']))
    with pytest.raises(ValueError,match='closed failure'):
        d.required_inputs(dict(mode='media',inputs={'run':'x'},runtime_namespace_proof={'path':'proof'},planned_required_clips=6))
    assert d.legacy.semantic_files is sentinel


def test_exact_outcome_precedence_and_missing_ledger(tmp_path):
    ctx=SimpleNamespace(read=lambda p:json.loads(p.read_text()))
    assert media.outcome_source(tmp_path,ctx)==(None,[])
    (tmp_path/'summary.json').write_text('{"trials":[{"trial_id":"summary"}]}')
    assert media.outcome_source(tmp_path,ctx)[1][0]['trial_id']=='summary'
    (tmp_path/'ledger_0000.json').write_text('[{"trial_id":"ledger"}]')
    assert media.outcome_source(tmp_path,ctx)[1][0]['trial_id']=='ledger'
    (tmp_path/'outcomes.json').write_text('[{"trial_id":"final"}]')
    assert media.outcome_source(tmp_path,ctx)[1][0]['trial_id']=='final'
