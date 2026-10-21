from copy import deepcopy
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from human_punching import phase2_family_target_probe_v2 as probe
from human_punching import phase2_family_target_path_v2 as solver


@pytest.fixture
def plans(tmp_path):
    return probe.create_plans(probe.PROJECT_ROOT/'outputs/task10b/p2familybounds01/bound_sensitivity.json',tmp_path/'plans')


def test_new_solver_diff_is_exact_cap_version_and_explanatory_error_only():
    old=(probe.PROJECT_ROOT/'src/human_punching/phase2_target_path_reference_v3.py').read_text()
    expected=old.replace('task10b_sequential_active_arm_target_path_v3',solver.VERSION).replace('MAXIMUM_CORRECTION_RAD = .205','MAXIMUM_CORRECTION_RAD = .210').replace('/.205-rad/','/.210-rad/')
    assert Path(solver.__file__).read_text()==expected
    assert solver.MAXIMUM_CORRECTION_RAD==.210


def test_real_cap_only_plans_and_same_target_source_requests(plans):
    assert set(plans)=={'cross','left_hook'}
    for family,p in plans.items():
        plan=probe.read(p);settings,cells,paths=probe.resolve_cells(plan)
        prior=probe.read(plan['previous_probe_plan'])
        assert len(cells)==4 and len(settings.base.optimized_joint_names)==11
        assert settings.base.maximum_joint_delta_rad==.21 and settings.base.maximum_iterations==80
        assert settings.base.maximum_iteration_delta_rad==.025 and settings.base.active_position_tolerance_m==.0005
        assert 'bound_sensitivity' in paths and 'previous_probe_plan' in paths
        assert [(r['parent_candidate_id'],r['target_candidate_identity']) for r in plan['cells']]==[(r['parent_candidate_id'],r['target_candidate_identity']) for r in prior['cells']]
        assert all(r['candidate_id'].endswith('_family_0210_v2_probe') for r in plan['cells'])


@pytest.mark.parametrize('family',['jab','right_hook','unknown'])
def test_no_other_family_cap_expansion(family):
    with pytest.raises(ValueError):probe.family_settings(family)


@pytest.mark.parametrize('mutation',['cap225','cap205','iterations','increment','tolerance','mask','inventory','sensitivity','prior'])
def test_no_silent_recipe_or_ancestry_expansion(plans,mutation):
    plan=probe.read(plans['cross'])
    if mutation=='cap225':plan['retarget_settings']['maximum_joint_delta_rad']=.225
    elif mutation=='cap205':plan['retarget_settings']['maximum_joint_delta_rad']=.205
    elif mutation=='iterations':plan['retarget_settings']['maximum_iterations']=81
    elif mutation=='increment':plan['retarget_settings']['maximum_iteration_delta_rad']=.05
    elif mutation=='tolerance':plan['retarget_settings']['active_position_tolerance_m']=.001
    elif mutation=='mask':plan['retarget_settings']['optimized_joint_names'][0]='Chest_z'
    elif mutation=='inventory':plan['bound_sensitivity_inputs'].pop(next(iter(plan['bound_sensitivity_inputs'])))
    elif mutation=='sensitivity':plan['bound_sensitivity']['sha256']='0'*64
    else:plan['previous_probe_plan']['sha256']='0'*64
    with pytest.raises(ValueError):probe.resolve_cells(plan)


def driver():
    spec=importlib.util.spec_from_file_location('family_bound_v2_driver_test',probe.PROJECT_ROOT/'scripts/task10b_phase2_family_targets_v2.py')
    d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d);return d


def test_exact_saved_diagnostic_inventory_closed(plans):
    d=driver();path=probe.original.owned_input(plans['cross']);plan=probe.read(plans['cross']);closure=d.files(path)
    for item in plan['bound_sensitivity_inputs'].values():
        p=probe.diagnostic_input(item)
        assert closure[str(p.relative_to(probe.PROJECT_ROOT))]==item['sha256']
    assert 'src/human_punching/phase2_family_target_path_v2.py' in closure


def test_dispatch_uses_private_new_cap_solver_and_retains_original_engine(plans,tmp_path,monkeypatch):
    original=probe.engine_source.retarget_sequential;original_version=probe.engine_source.VERSION
    captured={};engine=SimpleNamespace()
    def mock_generate(plan_path,output,**kwargs):
        captured.update(solver=engine.retarget_sequential,version=engine.VERSION,resolver=engine.resolve_cells,wall=kwargs['wall_seconds'])
        Path(output).mkdir();(Path(output)/'manifest.json').write_text('{}')
        return {'mock_only':True}
    engine.generate=mock_generate;monkeypatch.setattr(probe,'_engine',lambda:engine)
    result=probe.generate(probe.original.owned_input(plans['cross']),tmp_path/'mockout',wall_seconds=210)
    assert result=={'mock_only':True} and captured==dict(solver=solver.retarget_sequential,version=solver.VERSION,resolver=probe.resolve_cells,wall=210)
    assert probe.engine_source.retarget_sequential is original and probe.engine_source.VERSION==original_version
    sidecar=json.loads((tmp_path/'mockout/family_probe_entry.json').read_text())
    assert sidecar['engineering_cap_rad']==.21 and sidecar['physical_limits_changed'] is False
    assert sidecar['solver_math_other_than_cap_changed'] is False


@pytest.mark.parametrize('available',[None,270,240,210])
def test_fixed_child_and_no_native_dispatch_to_old_solver(available):
    d=driver();record=dict(mode=d.MODE,inputs={'plan':'tests/test_phase2_family_target_probe_v2.py'},
        allocation_wall_seconds=360,application_wall_seconds=300,sequential_child_wall_seconds=210,
        sequential_engineering_correction_cap_rad=.21,physical_joint_limits_relaxed=False,output='outputs/task10b/future/run')
    cmd=d.native_command(record,available)
    assert cmd[2:4]==['-m','human_punching.phase2_family_target_probe_v2'] and cmd[-1]=='210.0'
    record['sequential_engineering_correction_cap_rad']=.205
    with pytest.raises(ValueError):d.native_command(record,available)


@pytest.mark.parametrize('available',[209.99,271,float('nan'),True])
def test_late_setup_or_unbounded_child_refused(available):
    d=driver();record=dict(mode=d.MODE,inputs={'plan':'tests/test_phase2_family_target_probe_v2.py'},
        allocation_wall_seconds=360,application_wall_seconds=300,sequential_child_wall_seconds=210,
        sequential_engineering_correction_cap_rad=.21,physical_joint_limits_relaxed=False,output='outputs/task10b/future/run')
    with pytest.raises(RuntimeError):d.native_command(record,available)
