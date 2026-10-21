"""Interior packet admission and commands; no generation or GPU execution."""
import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('interior_driver',ROOT/'scripts/task10b_phase2_expand.py')
driver=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(driver)
PLAN=ROOT/'outputs/task10b/p2interiorimpl01/interior_plan.json'


def test_interiors_are_initial_only_exact_four_cell_plan_and_fixed_counted_wall():
    scope=driver.validate_spec('interior-references',{'plan':str(PLAN)},420,360)
    assert scope==dict(family=None,seed=21021,fresh_transition_cap=0)
    with pytest.raises(ValueError,match='Conditional hours'):driver.validate_spec('interior-references',{'plan':str(PLAN)},420,360,proof={})
    for inputs,wall,app in [({},420,360),({'plan':str(PLAN),'family':'jab'},420,360),({'plan':str(PLAN)},360,330),({'plan':str(PLAN)},420,359)]:
        with pytest.raises(ValueError):driver.validate_spec('interior-references',inputs,wall,app)


def test_interiors_never_shrink_immutable_child_wall_and_reject_late_setup():
    record=dict(mode='interior-references',inputs={'plan':str(PLAN)},application_wall_seconds=360,output='outputs/task10b/synthetic/run')
    for remaining in (None,330.,315.,300.):
        command=driver.native_command(record,remaining)
        assert command[-2:]==['--wall-seconds','300.0']
        assert command[2]==str(ROOT/'scripts/task10b_phase2_interiors.py')
        assert '--mode' not in command and '--max-updates' not in command
    with pytest.raises(RuntimeError,match='deadline margin'):driver.native_command(record,299.999)


def test_interiors_source_closure_contains_plan_native_cli_and_all_unchanged_generator_inputs():
    record=dict(mode='interior-references',inputs={'plan':str(PLAN)},bundle='outputs/task10/release_baseline_02',
        release='outputs/four-punch-milestone',predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json')
    files=driver.required_inputs(record)
    assert files['scripts/task10b_phase2_interiors.py']==driver.sha256_file(ROOT/'scripts/task10b_phase2_interiors.py')
    assert files['src/human_punching/phase2_interiors.py']==driver.sha256_file(ROOT/'src/human_punching/phase2_interiors.py')
    assert str(PLAN.relative_to(ROOT)) in files
