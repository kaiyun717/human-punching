"""Finite RH8 dispatch and prerequisite evidence; no environment or learning."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import pytest
from human_punching import phase2_right_hook_grouped_inputs_v1 as inputs
from human_punching.phase2_learning import PilotBudget
from human_punching.wbc.provenance import PROJECT_ROOT as ROOT,sha256_file


@pytest.fixture(scope='module')
def driver():
    spec=importlib.util.spec_from_file_location('rh_grouped_driver_tests',ROOT/'scripts/task10b_phase2_right_hook_grouped_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def declaration_fixture(tmp_path,driver):
    config=tmp_path/'configuration.json';config.write_text(json.dumps(dict(budget=asdict(PilotBudget(
        maximum_updates=8,maximum_wall_s=720.,shutdown_reserve_s=90.)))))
    values={k:str(config) for k in inputs.KEYS}
    record=dict(mode='grouped-pilot',inputs=values,family='right_hook',training=True,
        fresh_transition_cap=65536,grouped_child_wall_seconds=720,evaluation_reserve_seconds=4500,
        allocation_wall_seconds=960,application_wall_seconds=900,planned_validation_updates=[0,8],
        planned_validation_attempts=192,output='outputs/task10b/never_run/run',
        files={str(config.relative_to(ROOT)):sha256_file(config)})
    return config,values,record


def test_original_budget_and_existing_clock_parity_entry(driver,declaration_fixture,monkeypatch):
    path,values,record=declaration_fixture
    monkeypatch.setattr(driver,'_validate_spec',lambda *a,**k:dict(family='right_hook',seed=21021,fresh_transition_cap=65536))
    result=driver.validate_spec('grouped-pilot',values,960,900)
    assert result['planned_validation_updates']==[0,8] and result['planned_validation_attempts']==192
    assert result['grouped_integrity_gate_version']=='task10b_phase2_grouped_entry_integrity_v3'
    command=driver.native_command(record,720)
    assert command[command.index('-m')+1]=='human_punching.phase2_grouped_clock_entry_v2'
    assert command[command.index('--configuration-sha256')+1]==sha256_file(path)
    assert '--wall-seconds' not in command  # immutable original configuration owns720s deadline


@pytest.mark.parametrize('key,value',[('environments',256),('rollout_steps',32),('maximum_updates',16),
    ('maximum_wall_s',721),('shutdown_reserve_s',30),('validation_every',4)])
def test_budget_cannot_silently_expand_or_weaken(driver,declaration_fixture,monkeypatch,key,value):
    path,values,_=declaration_fixture;data=json.loads(path.read_text());data['budget'][key]=value;path.write_text(json.dumps(data))
    monkeypatch.setattr(driver,'_validate_spec',lambda *a,**k:dict(family='right_hook',seed=21021,fresh_transition_cap=65536))
    with pytest.raises(ValueError):driver.validate_spec('grouped-pilot',values,960,900)


@pytest.mark.parametrize('value',[719.999,871,float('nan'),float('inf'),True])
def test_late_setup_refuses_shortened_training(driver,declaration_fixture,value):
    with pytest.raises(RuntimeError):driver.native_command(declaration_fixture[2],value)


def test_wrong_scope_and_conditional_budget_refused(driver,declaration_fixture,monkeypatch):
    _,values,record=declaration_fixture
    for mode,wall,app in [('runtime-pack',960,900),('grouped-pilot',840,780),('grouped-pilot',1020,960)]:
        with pytest.raises(ValueError):driver.validate_spec(mode,values,wall,app)
    with pytest.raises(ValueError):driver.validate_spec('grouped-pilot',values,960,900,proof={})
    seen=[];monkeypatch.setattr(driver,'_admit_reservation',lambda *a,**k:seen.append((a,k)))
    driver.protected_reservation({},960)
    assert seen[0][1]['evaluation_reserve_seconds']==4500
    with pytest.raises(ValueError):driver.protected_reservation({},960,successful_route_frozen=True)


def test_original_semantic_dependency_restored_after_rejection(driver,declaration_fixture,monkeypatch):
    _,values,record=declaration_fixture;record['runtime_namespace_proof']={};record['successful_route_freeze']=None
    record['runtime_namespace_proof']={'proof':'present'}
    original=driver.legacy.semantic_files
    def fail(*a,**k):raise RuntimeError('synthetic closure interruption')
    monkeypatch.setattr(driver,'_required_inputs',fail)
    with pytest.raises(RuntimeError):driver.required_inputs(record)
    assert driver.legacy.semantic_files is original


def review_inputs():
    return dict(control_numerical_review=str(ROOT/'outputs/task10b/p2fcaudithost02/audit_right_hook/review.json'),
        control_visual_review=str(ROOT/'outputs/task10b/p2fcvisualreview01r/review.json'))


def test_actual_RH32_control_and_all25_manual_reviews_are_bound():
    collector=inputs.Inputs(ROOT);inputs.control_reviews(collector,review_inputs());closure=collector.finish()
    assert 'outputs/task10b/p2fcaudithost02/audit_right_hook/review.json' in closure
    assert 'outputs/task10b/p2fcaudithost02/audit_right_hook/trace_audit/audit.json' in closure
    assert 'outputs/task10b/p2fcvisualreview01r/review.json' in closure


@pytest.mark.parametrize('field,value',[('movement_exploit_observed',True),('manually_reviewed_clips',24),
    ('reviewed_numeric_failure_clips',0),('family','jab')])
def test_incomplete_or_exploit_visual_review_never_admits(tmp_path,field,value):
    data=review_inputs();view=json.loads(Path(data['control_visual_review']).read_text());view[field]=value
    path=tmp_path/'visual.json';path.write_text(json.dumps(view));data['control_visual_review']=str(path)
    with pytest.raises(ValueError):inputs.control_reviews(inputs.Inputs(ROOT),data)


def test_frozen_control_failures_cannot_be_silently_promoted(tmp_path):
    data=review_inputs();numeric=json.loads(Path(data['control_numerical_review']).read_text())
    numeric['independent_numeric_complete_cycles']=32
    path=tmp_path/'numeric.json';path.write_text(json.dumps(numeric));data['control_numerical_review']=str(path)
    with pytest.raises(ValueError):inputs.control_reviews(inputs.Inputs(ROOT),data)
