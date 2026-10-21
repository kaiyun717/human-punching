"""Exact selected control/pilot/target dispatch, caps and immutable closures."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('selected_route_driver',ROOT/'scripts/task10b_phase2_expand.py')
driver=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(driver)
NC=ROOT/'outputs/task10b/p2ncprep01/plan_c.json'
PILOT=ROOT/'outputs/task10b/p2jabpilotprep01/plan.json'
TARGET=ROOT/'outputs/task10b/p2targetprofileprep01/plan.json'


def record(mode,path,app):
    return dict(mode=mode,inputs=dict(plan=str(path)),application_wall_seconds=app,
        output='outputs/task10b/synthetic_selected/run',bundle='outputs/task10/release_baseline_02',
        release='outputs/four-punch-milestone',predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json')


def test_new_frozen_and_original_ppo_routes_keep_exact_initial_only_caps():
    control=driver.validate_spec('nonuniform-control',{'plan':str(NC)},480,420)
    pilot=driver.validate_spec('timing-pilot',{'plan':str(PILOT)},1200,1140)
    assert control['family']=='cross' and control['fresh_transition_cap']==0
    assert pilot['family']=='jab' and pilot['fresh_transition_cap']==128*64*16
    for mode,path,wall,app in [('nonuniform-control',NC,480,420),('timing-pilot',PILOT,1200,1140)]:
        assert driver.rounded_wall(wall)==wall
        assert driver.native_command({**record(mode,path,app),**driver.validate_spec(mode,{'plan':str(path)},wall,app)})
        with pytest.raises(ValueError,match='Conditional hours'):driver.validate_spec(mode,{'plan':str(path)},wall,app,proof={})
        assert driver.validate_spec(mode,{'plan':str(path)},wall,app)['d0_clock_versions']['independent_audit']=='task10b_phase2_independent_clock_v2'


@pytest.mark.parametrize('mode,path,wall,app',[
    ('nonuniform-control',NC,540,420),('nonuniform-control',NC,480,419),
    ('timing-pilot',PILOT,1260,1140),('timing-pilot',PILOT,1200,1139),
])
def test_new_route_allocation_or_application_changes_are_rejected(mode,path,wall,app):
    with pytest.raises(ValueError):driver.validate_spec(mode,{'plan':str(path)},wall,app)


def test_control_command_is_frozen_360_child_with_late_setup_rejection():
    r=record('nonuniform-control',NC,420)
    for remaining in (None,390.,375.,360.):
        command=driver.native_command(r,remaining)
        assert command[command.index('-m')+1]=='human_punching.phase2_nonuniform_control'
        assert command[-2:]==['--wall-seconds','360.0']
    with pytest.raises(RuntimeError,match='control child margin'):driver.native_command(r,359.999)


def test_pilot_uses_immutable_1080_wall_without_inventing_unsupported_cli_flags():
    r=record('timing-pilot',PILOT,1140)
    for remaining in (None,1110.,1090.,1080.):
        command=driver.native_command(r,remaining)
        assert command==[str(driver.base.PYTHON),'-B',str(ROOT/'scripts/task10b_phase2_timing_pilot.py'),
            '--plan',str(PILOT),'--output',str(ROOT/r['output'])]
    with pytest.raises(RuntimeError,match='pilot child margin'):driver.native_command(r,1079.999)


@pytest.mark.parametrize('mode,path,app',[('nonuniform-control',NC,420),('timing-pilot',PILOT,1140)])
def test_selected_control_and_pilot_closures_bind_actual_immutable_data_and_clock_auditor(mode,path,app):
    r=record(mode,path,app);files=driver.required_inputs(r)
    assert files[str(path.relative_to(ROOT))]==driver.sha256_file(path)
    for name in ('src/human_punching/phase2_d0_clock_v2.py','scripts/task10b_phase2_audit_clock_v2.py','src/human_punching/phase2_learning.py'):
        assert files[name]==driver.sha256_file(ROOT/name)
    if mode=='nonuniform-control':
        assert 'outputs/task10b/p2nr01/run/cross_nonuniform_postcontact_0p750/reference.npz' in files
        assert not any('left_hook_nonuniform' in p or 'right_hook_nonuniform' in p for p in files)
    else:
        assert files['scripts/task10b_phase2_timing_pilot.py']==driver.sha256_file(ROOT/'scripts/task10b_phase2_timing_pilot.py')
    assert not any('/home/gridsan' in p for p in files)


def test_standalone_target_branch_uses_exact_plan_helper_without_stale_body_xml_routing():
    r=record('target-references',TARGET,240);scope=driver.validate_spec(r['mode'],r['inputs'],300,240)
    assert scope['family']=='jab' and scope['fresh_transition_cap']==0
    files=driver.required_inputs(r)
    assert files['src/human_punching/phase2_target_inputs.py']==driver.sha256_file(ROOT/'src/human_punching/phase2_target_inputs.py')
    assert files['outputs/four-punch-milestone/jab/body_metadata.json']==driver.sha256_file(ROOT/'outputs/four-punch-milestone/jab/body_metadata.json')
    assert not any('/home/gridsan' in p for p in files)


@pytest.mark.parametrize('kind',['pilot','nonuniform'])
def test_media_dispatch_preserves_honest_launch_schema_and_selective_helper(tmp_path,monkeypatch,kind):
    run=tmp_path/'run';run.mkdir();panel=run/'summary.json';panel.write_text('{}')
    if kind=='pilot':
        launch=dict(schema='task10b_phase2_selected_d0_timing_pilot_v1',mode='timing_pilot',family='jab')
        from human_punching import phase2_timing_pilot_media_inputs as helper
        name='files'
    else:
        launch=dict(schema='task10b_phase2_original_route_launch_v1',mode='frozen',family='cross',
            nonuniform_control_version='task10b_phase2_six_nonuniform_frozen_control_v1')
        from human_punching import phase2_nonuniform_control as helper
        name='media_files'
    (run/'launch.json').write_text(json.dumps(launch));seen=[]
    def fake(*args,**kwargs):seen.append((args,kwargs));return {str(panel.relative_to(ROOT)):driver.sha256_file(panel)}
    monkeypatch.setattr(helper,name,fake)
    r=dict(mode='media',inputs=dict(run=str(run),panel=str(panel)),bundle='outputs/task10/release_baseline_02',
        release='outputs/four-punch-milestone',predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json')
    files=driver.required_inputs(r)
    assert len(seen)==1 and str(panel.relative_to(ROOT)) in files
    assert json.loads((run/'launch.json').read_text())==launch
