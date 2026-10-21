"""Actual immutable reference selection plus synthetic saved-media tests; no GPU."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from human_punching import phase2_nonuniform_control as m

ROOT=m.ROOT
REVIEW=ROOT/'outputs/task10b/p2nonuniformrefaudit01/review.json'


@pytest.fixture(scope='module')
def plans():return {f:m.create_plan(REVIEW,f) for f in m.FAMILIES}


def test_exact_six_new_rows_and_three_sixteen_trial_panels(plans):
    selected=[]
    for family,plan in plans.items():
        _,_,_,_,rows=m.validate_plan(plan)
        assert len(rows)==2 and {r['family'] for r in rows}=={family}
        assert all(r['kind']=='nonuniform_contact_anchored' for r in rows)
        assert plan['planned_trials']==16 and plan['bank_ids']==list(range(8)) and plan['seed']==21021
        assert not plan['training'] and plan['max_updates']==plan['fresh_transition_cap']==0
        assert not plan['original16_execution'] and plan['child_wall_seconds']==360
        selected.extend(plan['selected_candidate_ids'])
    assert len(set(selected))==6 and set(selected)==set(plans['cross']['all_six_candidate_ids'])


def test_identical_control_plan_under_both_proved_root_spellings(plans):
    foreign=Path('/orcd/home/002/kaisyun/Azizan_Lab/HumanBox/human-punching')
    assert m.create_plan(REVIEW,'cross',root=foreign)==plans['cross']
    assert m.validate_plan(plans['cross'],root=foreign)[-1]==m.validate_plan(plans['cross'])[-1]


@pytest.mark.parametrize('field,value',[('seed',21022),('validation_seed',21021),('training',True),('max_updates',1),
    ('fresh_transition_cap',8192),('planned_trials',8),('bank_ids',[0,1,2,3]),('child_wall_seconds',359),
    ('application_wall_seconds',421),('allocation_wall_seconds',540),('original16_execution',True)])
def test_control_request_or_finite_budget_mutations_are_rejected(plans,field,value):
    plan=deepcopy(plans['cross']);plan[field]=value
    with pytest.raises(ValueError):m.validate_plan(plan)


def test_unrequested_family_and_changed_input_inventory_are_rejected(plans):
    with pytest.raises(ValueError,match='Only cross'):m.create_plan(REVIEW,'jab')
    plan=deepcopy(plans['cross']);key=next(iter(plan['input_inventory']));plan['input_inventory'][key]='0'*64
    with pytest.raises(ValueError):m.validate_plan(plan)


@pytest.mark.parametrize('change',['parent_row','new_row','generation_status','generation_mode','assessment','reference_sha','archive_sha'])
def test_actual_ancestry_and_payload_tampering_cannot_become_a_control(plans,monkeypatch,change):
    manifest=ROOT/'outputs/task10b/p2nr01/run/manifest.json';execution=ROOT/'outputs/task10b/p2nr01/execution.json'
    packet=ROOT/'outputs/task10b/p2nr01/packet.json';parent=ROOT/'outputs/task10b/p2timingcatalog01/manifest.json'
    source_plan=json.loads((ROOT/'outputs/task10b/p2nr01/run/plan.json').read_text())
    assessment=Path(source_plan['assessment']['path']);old_read=Path.read_text;old_sha=m.sha256_file
    def read(path,*args,**kwargs):
        text=old_read(path,*args,**kwargs)
        if change=='parent_row' and path==parent:
            data=json.loads(text);data['cells'][0]['requested_ttc_s']=9.;return json.dumps(data)
        if change=='new_row' and path==manifest:
            data=json.loads(text);data['cells'][16]=data['cells'][0];return json.dumps(data)
        if change=='generation_status' and path==execution:
            data=json.loads(text);data['status']='failed';return json.dumps(data)
        if change=='generation_mode' and path==packet:
            data=json.loads(text);data['mode']='timing-references';return json.dumps(data)
        if change=='assessment' and path==assessment:
            data=json.loads(text);data['measured']['failed_cell_ids'].pop();return json.dumps(data)
        return text
    def sha(path):
        path=Path(path)
        if change=='reference_sha' and path.name=='reference.npz' and 'cross_nonuniform' in str(path):return '0'*64
        if change=='archive_sha' and 'source_snapshot' in str(path) and path.name=='phase2_nonuniform.py':return '0'*64
        return old_sha(path)
    monkeypatch.setattr(Path,'read_text',read);monkeypatch.setattr(m,'sha256_file',sha)
    with pytest.raises(ValueError):m.validate_plan(plans['cross'])


def test_inventory_archives_only_requested_new_reference_products(plans):
    files=plans['cross']['input_inventory'];prefix='outputs/task10b/p2nr01/run/'
    selected=[p for p in files if p.startswith(prefix)]
    assert any('cross_nonuniform_postcontact_0p750/reference.npz' in p for p in selected)
    assert any('cross_nonuniform_postcontact_0p900/reference.npz' in p for p in selected)
    assert not any('left_hook_nonuniform' in p or 'right_hook_nonuniform' in p for p in selected)
    assert prefix+'manifest.json' in files and prefix+'plan.json' in files


def test_deadline_is_checked_before_unchanged_launch_gate_and_actor_append(monkeypatch):
    env=m.BoundedControlEnv.__new__(m.BoundedControlEnv);seen=[]
    env.control_deadline=SimpleNamespace(checkpoint=lambda stage:seen.append(stage))
    monkeypatch.setattr(m.route.NativeSpecialistEnv,'launch_gate',lambda self:seen.append('original_gate') or 'result')
    assert env.launch_gate()=='result' and seen==['nonuniform_control_actor_boundary','original_gate']
    def expired(stage):raise RuntimeError('bounded deadline')
    env.control_deadline=SimpleNamespace(checkpoint=expired);seen.clear()
    with pytest.raises(RuntimeError,match='bounded deadline'):env.launch_gate()
    assert not seen


def test_unallocated_cli_cannot_read_model_or_launch_native_execution(tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.setattr(m.torch.cuda,'is_available',lambda:pytest.fail('No CUDA query outside counted allocation'))
    with pytest.raises(RuntimeError,match='Counted Slurm'):m.execute(tmp_path/'absent.json',tmp_path/'run')
    assert not (tmp_path/'run').exists()


@pytest.fixture
def saved_media(tmp_path,plans):
    plan=deepcopy(plans['cross']);control=tmp_path/'control.json';control.write_text(json.dumps(plan))
    run=tmp_path/'run';panel=run/'validation_00000/summary.json';panel.parent.mkdir(parents=True)
    ctx,catalog,_,source,_=m.validate_plan(plan)
    _,paths=m.route.family_arguments(ctx.bound(source['release']).parent,'cross')
    launch=dict(schema='task10b_phase2_original_route_launch_v1',nonuniform_control_version=m.VERSION,mode='frozen',family='cross',
        catalog=str(catalog),catalog_sha256=m.sha256_file(catalog),planned_cells=plan['selected_candidate_ids'],
        control_plan=dict(path=str(control),sha256=m.sha256_file(control)),
        release_inputs={k:dict(path=str(p),sha256=m.sha256_file(p)) for k,p in paths.items()},
        sources={str(Path(m.__file__)):m.sha256_file(m.__file__)})
    (run/'launch.json').write_text(json.dumps(launch))
    panel.write_text(json.dumps(dict(schema='task10b_phase2_d0_panel_v1',planned_trials=16,
        trials=[dict(trial_id=f'{c}_bank_{i:02d}',candidate_id=c,numeric_complete_cycle=False) for c in plan['selected_candidate_ids'] for i in range(8)])))
    return run,panel,launch


def test_nonuniform_media_retains_only_two_requested_cells_and_all_sixteen_rows(saved_media):
    run,panel,_=saved_media;files=m.media_files(ROOT,dict(run=str(run),panel=str(panel)))
    assert str(panel.relative_to(ROOT)) in files and str((run/'launch.json').relative_to(ROOT)) in files


@pytest.mark.parametrize('change',['all_family_rows','missing_trial','checkpoint_hash','catalog_hash'])
def test_nonuniform_media_rejects_relabeling_old_rows_missing_trials_or_changed_weights(saved_media,change):
    run,panel,launch=saved_media
    if change=='all_family_rows':launch['planned_cells']+=['cross_original']
    elif change=='missing_trial':
        data=json.loads(panel.read_text());data['trials'].pop();panel.write_text(json.dumps(data))
    elif change=='checkpoint_hash':launch['release_inputs']['checkpoint']['sha256']='0'*64
    else:launch['catalog_sha256']='0'*64
    (run/'launch.json').write_text(json.dumps(launch))
    with pytest.raises(ValueError):m.media_files(ROOT,dict(run=str(run),panel=str(panel)))
