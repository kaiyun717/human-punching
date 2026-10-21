import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from human_punching import phase2_family_control_media_v2 as media


def driver():
    path=Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_family_media_v2.py'
    spec=importlib.util.spec_from_file_location('target_media_test_driver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('remaining',[0,120,211,float('nan'),True])
def test_no_render_without_bounded_flush_margin(remaining):
    d=driver()
    packet=dict(family='left_hook',mode='media',inputs={'run':'unused'},allocation_wall_seconds=300,application_wall_seconds=240,
        target_media_version=media.VERSION,source_control_trials=32,training=False,actor_calls=0)
    with pytest.raises(RuntimeError):d.native_command(packet,remaining)


@pytest.mark.parametrize('change',[{'training':True},{'actor_calls':1},{'source_control_trials':31},{'allocation_wall_seconds':360}])
def test_scope_cannot_turn_media_into_execution(change):
    d=driver()
    packet=dict(family='left_hook',mode='media',inputs={'run':'unused'},allocation_wall_seconds=300,application_wall_seconds=240,
        target_media_version=media.VERSION,source_control_trials=32,training=False,actor_calls=0)
    with pytest.raises(ValueError):d.native_command({**packet,**change})


def test_failed_panel_preserves_all_remaining_panels(tmp_path,monkeypatch):
    """A rendering exception must not erase the other three declared panels."""
    run=tmp_path/'source';run.mkdir();planpath=run/'control_plan.json';planpath.write_text('{}')
    panels=[];payloads={}
    for i in range(4):
        p=run/f'validation_{i:05d}/summary.json';p.parent.mkdir();p.write_text('{}');panels.append(p)
        payloads[p]={'selection':[{'trial_id':f'trial{i}'}]}
    payloads[run/'media_selection.json']={'planned_required_trial_ids':[f'trial{i}' for i in range(4)],'unavailable_trial_ids':[]}
    monkeypatch.setattr(media,'selected',lambda value:(SimpleNamespace(read=lambda p:payloads[p]),planpath,{},panels))
    monkeypatch.setattr(media,'writable_path',lambda p:Path(p))
    def failed_render(*a,**k):raise RuntimeError('retained renderer failure')
    fake=SimpleNamespace(selection=lambda p:p['selection'],render=failed_render)
    monkeypatch.setattr(media.importlib.util,'spec_from_file_location',lambda *a:SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None)))
    monkeypatch.setattr(media.importlib.util,'module_from_spec',lambda s:fake)
    monkeypatch.setenv('SLURM_JOB_ID','123')
    result=media.render(run,tmp_path/'output',180.)
    assert result['status']=='incomplete' and result['rendered']==0
    assert result['exception']['message']=='retained renderer failure'
    assert len(result['panels'])==4 and {r['panel'] for r in result['panels']}==set(range(4))
    assert result['panels'][0]['status']=='failed_extent_unknown'
    assert all(r['status']=='unattempted' for r in result['panels'][1:])
    assert result['actor_calls']==result['physics_steps']==0


@pytest.mark.parametrize('short',['c','r'])
def test_old0205_native_receipts_are_not_relabelled_as0210(short):
 root=Path(__file__).resolve().parents[1]
 with pytest.raises(ValueError,match='Exact frozen'):
  media.selected(root/f'outputs/task10b/p2fc01{short}/run')

@pytest.mark.parametrize('wrong',[dict(planned_required_trial_ids=['a','b'],unavailable_trial_ids=[]),dict(planned_required_trial_ids=['a','b'],unavailable_trial_ids=['a']),dict(planned_required_trial_ids=['a','a'],unavailable_trial_ids=['b'])])
def test_no_unavailable_clip_disappears_or_becomes_fake_video(wrong):
    with pytest.raises(ValueError):media.validate_union(['a'],wrong)


@pytest.mark.parametrize('section,change',[
    ('packet',{'mode':'timing-pilot'}),('packet',{'training':True}),
    ('outer',{'packet_sha256':'another-packet'}),('outer',{'native_processes_started':0}),
    ('outer',{'budget_stop':'application_wall'}),('outer',{'returncode':1}),
    ('terminal',{'allocated_gpu_seconds':0}),('terminal',{'exit_code':'1:0'}),
    ('terminal',{'job_id':'another-job'}),('terminal',{'state':'TIMEOUT'})])
def test_other_or_incomplete_parent_cannot_supply_review_media(section,change):
    parent=dict(schema='task10b_phase2_family_target_control_packet_v2',mode='family-target-control-v2',
        training=False,gpus=1,fresh_transition_cap=0,family='left_hook',target_control_reference_engineering_cap_rad=.210,physical_joint_limits_relaxed=False)
    outer=dict(status='executed',returncode=0,native_processes_started=1,budget_stop=None,packet_sha256='bound',job_id='123')
    terminal=dict(state='COMPLETED',exit_code='0:0',allocated_gpu_seconds=155,packet_sha256='bound',job_id='123')
    {'packet':parent,'outer':outer,'terminal':terminal}[section].update(change)
    with pytest.raises(ValueError):media.validate_parent(parent,outer,terminal,{'slurm_job_id':'123'},'bound')


def test_exact0210_parent_is_accepted_and_all16_unavailable_union_retained():
 parent=dict(schema='task10b_phase2_family_target_control_packet_v2',mode='family-target-control-v2',family='left_hook',target_control_reference_engineering_cap_rad=.210,physical_joint_limits_relaxed=False,training=False,gpus=1,fresh_transition_cap=0)
 outer=dict(status='executed',returncode=0,native_processes_started=1,budget_stop=None,packet_sha256='bound',job_id='123')
 terminal=dict(state='COMPLETED',exit_code='0:0',allocated_gpu_seconds=155,packet_sha256='bound',job_id='123')
 media.validate_parent(parent,outer,terminal,{'slurm_job_id':'123'},'bound')
 missing=[f'missing{i}' for i in range(16)];renderable=[f'actual{i}' for i in range(16)]
 saved=dict(planned_required_trial_ids=missing+renderable,unavailable_trial_ids=missing)
 media.validate_union(renderable,saved)
 with pytest.raises(ValueError):media.validate_union(renderable+missing[:1],saved)
 with pytest.raises(ValueError):media.validate_union(renderable[:-1],saved)


@pytest.mark.parametrize('field,value',[('family','cross'),('target_control_reference_engineering_cap_rad',.205),('physical_joint_limits_relaxed',True)])
def test_0210_parent_scope_cannot_change(field,value):
 parent=dict(schema='task10b_phase2_family_target_control_packet_v2',mode='family-target-control-v2',family='left_hook',target_control_reference_engineering_cap_rad=.210,physical_joint_limits_relaxed=False,training=False,gpus=1,fresh_transition_cap=0)
 outer=dict(status='executed',returncode=0,native_processes_started=1,budget_stop=None,packet_sha256='bound',job_id='123')
 terminal=dict(state='COMPLETED',exit_code='0:0',allocated_gpu_seconds=155,packet_sha256='bound',job_id='123')
 parent[field]=value
 with pytest.raises(ValueError):media.validate_parent(parent,outer,terminal,{'slurm_job_id':'123'},'bound')
