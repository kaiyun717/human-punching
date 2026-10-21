"""CPU metadata/dispatch tests; no candidate construction, actor, or scheduler."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util,json
from pathlib import Path
import pytest
from human_punching import phase2_family_target_probe_v1 as probe


@pytest.fixture
def selection(tmp_path):
    data=probe.read(probe.binding(probe.PROJECT_ROOT/'outputs/task10b/p2allfamilytargetprep01/selection.json'))
    path=tmp_path/'selection.json';path.write_text(json.dumps(data));return path


@pytest.fixture
def plans(selection,tmp_path):
    return probe.create_plans(selection,tmp_path/'plans')


@pytest.mark.parametrize('family,side',[('cross','Right'),('left_hook','Left'),('right_hook','Right')])
def test_exact_common_solver_settings_and_arm_only(family,side):
    settings=probe.family_settings(family)
    assert settings.base.active_side==side and len(settings.base.optimized_joint_names)==11
    assert all(name.startswith(side) for name in settings.base.optimized_joint_names)
    assert settings.base.trunk_joint_names==() and settings.base.maximum_joint_delta_rad==.205
    assert settings.base.maximum_iteration_delta_rad==.025 and settings.base.maximum_iterations==80
    assert settings.base.active_position_tolerance_m==.0005 and settings.preference_weight==.002


@pytest.mark.parametrize('family',['jab','unknown','left'])
def test_existing_jab_and_unknown_family_not_regenerated(family):
    with pytest.raises(ValueError):probe.family_settings(family)


def test_real_source_cells_allthree_families(plans):
    for family,path in plans.items():
        plan=probe.read(path);settings,cells,inputs=probe.resolve_cells(plan)
        assert len(cells)==4 and settings.base.active_side==('Left' if family=='left_hook' else 'Right')
        assert [(t['command_id'],p['requested_ttc_s']) for _,t,p,_,_ in cells]==[(g,t) for g in (4184,4251) for t in probe.TIMINGS[family]]
        assert plan['maximum_child_seconds']==210 and plan['maximum_cells']==4
        assert 'timing_selection' in inputs and all(t['split']=='train' for _,t,_,_,_ in cells)
        assert [p['candidate_id'] for _,_,p,_,_ in cells[:2]]==probe.read(plan['selected_d0_plan'])['training_candidate_ids']
        if family in ('cross','right_hook'):
            assert cells[1][2]['training_eligible'] is False  # Historical comparison flag is preserved.


@pytest.mark.parametrize('mutation',['split','cap','iterations','side','trunk','solver','source','qualification','implicit_bound','child'])
def test_no_silent_changes_to_frozen_contract(plans,mutation):
    plan=probe.read(plans['cross'])
    if mutation=='split':plan['split']='test'
    elif mutation=='cap':plan['retarget_settings']['maximum_joint_delta_rad']=.21
    elif mutation=='iterations':plan['retarget_settings']['maximum_iterations']=81
    elif mutation=='side':plan['retarget_settings']['active_side']='Left'
    elif mutation=='trunk':plan['retarget_settings']['trunk_joint_names']=['Chest_z']
    elif mutation=='solver':plan['path_solver_version']='fake'
    elif mutation=='source':plan['sources'][next(iter(plan['sources']))]['sha256']='0'*64
    elif mutation=='qualification':plan['physical_qualification']=True
    elif mutation=='implicit_bound':plan['new_family_engineering_bound_unqualified']=False
    else:plan['maximum_child_seconds']=211
    with pytest.raises(ValueError):probe.resolve_cells(plan)


def test_exact_candidate_order_and_no_nearest_substitution(plans):
    plan=probe.read(plans['right_hook']);plan['cells'][0]['parent_candidate_id']='right_hook_uniform_0p800'
    with pytest.raises(ValueError):probe.resolve_cells(plan)
    plan=probe.read(plans['cross']);plan['cells'].reverse()
    with pytest.raises(ValueError):probe.resolve_cells(plan)


def test_private_generator_instance_preserves_original_source_module():
    original=probe.engine_source.resolve_cells;engine=probe._engine()
    engine.resolve_cells=lambda _:None
    assert probe.engine_source.resolve_cells is original
    assert engine.retarget_sequential is probe.engine_source.retarget_sequential
    assert engine.finalize_original_feedforward is probe.engine_source.finalize_original_feedforward
    assert engine.screen_chunks is probe.engine_source.screen_chunks


def launcher():
    spec=importlib.util.spec_from_file_location('family_targets_test_driver',probe.PROJECT_ROOT/'scripts/task10b_phase2_family_targets_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_exact_scientific_closure_and_declared_family(plans):
    d=launcher()
    for family,item in plans.items():
        inputs={'plan':item['path']};scope=d.validate_spec(d.MODE,inputs,360,300)
        assert scope['family']==family and scope['fresh_transition_cap']==0
        assert scope['sequential_child_wall_seconds']==210 and scope['physical_joint_limits_relaxed'] is False
        closure=d.files(item['path']);plan=probe.read(item)
        assert str(Path(plan['timing_selection']['path']).relative_to(probe.PROJECT_ROOT)) in closure
        assert all(name in closure for name in probe.SOURCES)
        assert str(Path(plan['selected_d0_plan']['path']).relative_to(probe.PROJECT_ROOT)) in closure


@pytest.mark.parametrize('available',[None,270,240,211,210])
def test_finite_child_with_real_setup_margin(plans,available):
    d=launcher();record=dict(mode=d.MODE,inputs={'plan':plans['cross']['path']},output='outputs/task10b/future/run',
        sequential_child_wall_seconds=210,allocation_wall_seconds=360,application_wall_seconds=300,
        sequential_engineering_correction_cap_rad=.205,physical_joint_limits_relaxed=False)
    cmd=d.native_command(record,available)
    assert cmd[2:4]==['-m','human_punching.phase2_family_target_probe_v1'] and cmd[-1]=='210.0'


@pytest.mark.parametrize('available',[209.99,0,True,float('nan'),271])
def test_late_setup_cannot_truncate_or_overreserve_child(plans,available):
    d=launcher();record=dict(mode=d.MODE,inputs={'plan':plans['cross']['path']},output='outputs/task10b/future/run',
        sequential_child_wall_seconds=210,allocation_wall_seconds=360,application_wall_seconds=300,
        sequential_engineering_correction_cap_rad=.205,physical_joint_limits_relaxed=False)
    with pytest.raises(RuntimeError):d.native_command(record,available)


def test_no_actual_generation_without_counted_allocation(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='counted'):probe.engine_source.require_allocation(210)
