"""Counted nonuniform reference generation dispatch; no native execution."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('nonuniform_driver',ROOT/'scripts/task10b_phase2_expand.py')
driver=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(driver)
PLAN=ROOT/'outputs/task10b/p2nonuniformprep02/plan.json'
OLD_PLAN=ROOT/'outputs/task10b/p2nonuniformimpl01/nonuniform_plan.json'


def record():
    return dict(mode='nonuniform-references',inputs={'plan':str(PLAN)},application_wall_seconds=270,
        output='outputs/task10b/synthetic_nonuniform/run',bundle='outputs/task10/release_baseline_02',
        release='outputs/four-punch-milestone',predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json')


def test_nonuniform_mode_is_initial_only_exact_six_cell_no_actor_scope():
    scope=driver.validate_spec('nonuniform-references',{'plan':str(PLAN)},300,270)
    assert scope==dict(family=None,seed=21021,fresh_transition_cap=0)
    assert 'nonuniform-references' not in driver.CONDITIONAL_MODES
    with pytest.raises(ValueError,match='Conditional hours'):
        driver.validate_spec('nonuniform-references',{'plan':str(PLAN)},300,270,proof={})


@pytest.mark.parametrize('inputs,wall,app',[
    ({},300,270),({'plan':str(PLAN),'family':'cross'},300,270),
    ({'plan':str(PLAN)},360,270),({'plan':str(PLAN)},300,269),
    ({'plan':str(PLAN)},300,271),({'plan':str(PLAN)},240,210),
])
def test_nonuniform_mode_rejects_changed_inputs_or_allocation_caps(inputs,wall,app):
    with pytest.raises(ValueError):driver.validate_spec('nonuniform-references',inputs,wall,app)


def test_preserved_240_second_plan_cannot_be_silently_truncated():
    assert json.loads(OLD_PLAN.read_text())['application_wall_seconds']==240
    with pytest.raises(ValueError,match='exactly210'):
        driver.validate_spec('nonuniform-references',{'plan':str(OLD_PLAN)},300,270)


@pytest.mark.parametrize('remaining',[None,240.,225.,210.])
def test_nonuniform_command_keeps_exact_210_child_and_no_actor_arguments(remaining):
    command=driver.native_command(record(),remaining)
    assert command==[str(driver.base.PYTHON),'-B',str(ROOT/'scripts/task10b_phase2_nonuniform.py'),
        '--plan',str(PLAN),'--output',str(ROOT/record()['output']),'--wall-seconds','210.0']
    assert '--mode' not in command and '--family' not in command and '--max-updates' not in command


def test_late_setup_is_rejected_before_a_generation_child_is_launched():
    with pytest.raises(RuntimeError,match='nonuniform child deadline margin'):
        driver.native_command(record(),209.999)


def test_nonuniform_closure_binds_plan_assessment_and_both_direct_measured_receipts():
    closure=driver.required_inputs(record());plan=json.loads(PLAN.read_text())
    for item in [plan[k] for k in ('parent_catalog','domain','predicate','release','bundle','assessment')]:
        path=Path(item['path']);assert closure[str(path.relative_to(ROOT))]==item['sha256']
    assessment=json.loads(Path(plan['assessment']['path']).read_text())['measured']
    for prefix in ('cells','error_localization'):
        path=Path(assessment[prefix+'_path'])
        assert closure[str(path.relative_to(ROOT))]==assessment[prefix+'_sha256']
    for name in ('scripts/task10b_phase2_nonuniform.py','src/human_punching/phase2_nonuniform.py',
                 'src/human_punching/phase2_interiors.py','src/human_punching/phase2_reference.py'):
        assert closure[name]==driver.sha256_file(ROOT/name)
    for family in ('jab','cross','left_hook','right_hook'):
        key=f'outputs/four-punch-milestone/{family}/controller_schema.json'
        assert key in closure
    assert not any('/home/gridsan' in p for p in closure)


@pytest.mark.parametrize('change',['cell','clock','assessment_hash','measurement_hash'])
def test_nonuniform_plan_and_evidence_tampering_fail_closed(tmp_path,change):
    plan=json.loads(PLAN.read_text())
    if change=='cell':plan['cells'][0]['requested_ttc_s']+=.05
    elif change=='clock':plan['local_clock_rate_bounds']=[.4,1.5]
    elif change=='assessment_hash':plan['assessment']['sha256']='0'*64
    else:
        assessment=json.loads(Path(plan['assessment']['path']).read_text())
        assessment['measured']['cells_sha256']='0'*64
        path=tmp_path/'assessment.json';path.write_text(json.dumps(assessment))
        plan['assessment']=dict(path=str(path),sha256=driver.sha256_file(path))
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):driver.validate_spec('nonuniform-references',{'plan':str(path)},300,270)
