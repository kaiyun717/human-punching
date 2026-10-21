"""Exact saved-partial continuation admission; no generation or native physics."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest
from human_punching import phase2_right_hook_suffix_catalog_v1 as suffix
from human_punching.wbc.provenance import PROJECT_ROOT as ROOT,sha256_file


@pytest.fixture(scope='module')
def plan(tmp_path_factory):
    output=tmp_path_factory.mktemp('suffix_plans')/'new'
    item=suffix.create_plan(ROOT/'outputs/task10b/p2rhfullaudit01t/audit.json',output)
    path=Path(item['path']);return path,json.loads(path.read_text())


def test_real_exact_original_10_and_common_recipe(plan):
    path,data=plan;settings,cells,paths=suffix.resolve_cells(data)
    prior=suffix.read(data['prior_plan']);manifest=suffix.read(data['prior_manifest'])
    assert data['cells']==prior['cells'][54:] and len(cells)==10
    assert data['recipe']==prior['recipe'] and data['sources']==prior['sources']
    assert cells[0][0]['candidate_id']==manifest['cells'][54]['candidate_id']
    assert data['cells'][1:]==manifest['unattempted_cells']
    assert settings.base.maximum_joint_delta_rad==.205
    closure=suffix.files(path)
    for key in suffix.ANCESTRY:
        item=data[key];assert closure[str(Path(item['path']).relative_to(ROOT))]==item['sha256']
    assert closure[suffix.AUDITOR]==sha256_file(ROOT/suffix.AUDITOR)


@pytest.mark.parametrize('change',['cell','reorder','extra','cap','child','domain','recipe','retry','source','family'])
def test_no_scope_or_recipe_mutation(plan,change):
    _,original=plan;data=deepcopy(original)
    if change=='cell':data['cells'][0]['candidate_id']='different'
    elif change=='reorder':data['cells'].reverse()
    elif change=='extra':data['cells'].append(data['cells'][0])
    elif change=='cap':data['output_byte_cap']+=1
    elif change=='child':data['maximum_child_seconds']=120
    elif change=='domain':data['retarget_settings']['maximum_joint_delta_rad']=.210
    elif change=='recipe':data['recipe']=data['prior_plan']
    elif change=='retry':data['automatic_further_retry']=True
    elif change=='source':data['continuation_sources']={}
    elif change=='family':data['family']='cross'
    with pytest.raises(ValueError):suffix.resolve_cells(data)


@pytest.mark.parametrize('change',['geometry_failure','prefix_missing','all_complete','audit_failed','different_job','lost_unattempted'])
def test_prior_dispositions_cannot_be_reinterpreted(plan,tmp_path,change):
    _,original=plan;data=deepcopy(original)
    key={'audit_failed':'prior_audit','different_job':'prior_execution'}.get(change,'prior_manifest')
    value=suffix.read(data[key])
    if change=='geometry_failure':value['cells'][54]['generation_disposition']='failed'
    elif change=='prefix_missing':value['cells'][0]['kinematic_admission']=False
    elif change=='all_complete':value['complete']=True
    elif change=='audit_failed':value['passed']=False
    elif change=='different_job':value['job_id']='1'
    elif change=='lost_unattempted':value['unattempted_cells'].pop()
    path=tmp_path/'changed.json';path.write_text(json.dumps(value));data[key]=suffix.binding(path)
    with pytest.raises(ValueError):suffix.prior_evidence(data)


def test_original_sources_are_private_and_generator_not_reimplemented():
    engine=suffix.full._engine()
    assert engine is not suffix.full.engine_source
    assert engine.generate.__code__.co_code==suffix.full.engine_source.generate.__code__.co_code
    assert engine.resolve_cells is not suffix.resolve_cells


@pytest.fixture(scope='module')
def driver():
    spec=importlib.util.spec_from_file_location('suffix_driver_tests',ROOT/'scripts/task10b_phase2_right_hook_suffix_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_real_driver_scope_and_late_refusal(plan,driver):
    path,_=plan;scope=driver.validate_spec(driver.MODE,{'plan':str(path)},180,150)
    assert scope['sequential_cells']==10 and scope['sequential_child_wall_seconds']==90
    r=dict(**scope,mode=driver.MODE,inputs={'plan':str(path)},training=False,
        allocation_wall_seconds=180,application_wall_seconds=150,output='outputs/task10b/never_run/run')
    cmd=driver.native_command(r,90)
    assert cmd[cmd.index('--wall-seconds')+1]=='90.0'
    assert 'human_punching.phase2_right_hook_suffix_catalog_v1' in cmd
    for available in (89.999,121,float('nan'),True):
        with pytest.raises(RuntimeError):driver.native_command(r,available)
    with pytest.raises(ValueError):driver.validate_spec(driver.MODE,{'plan':str(path)},240,210)


def test_protected_initial_budget(driver,monkeypatch):
    seen=[];monkeypatch.setattr(driver,'_admit_reservation',lambda *a,**k:seen.append((a,k)))
    driver.protected_reservation({},180)
    assert seen[0][1]['evaluation_reserve_seconds']==4500
    with pytest.raises(ValueError):driver.protected_reservation({},360)


def test_generator_refuses_child_before_native(plan,tmp_path,monkeypatch):
    path,_=plan;monkeypatch.setattr(suffix.full,'_engine',lambda:pytest.fail('no engine'))
    with pytest.raises(ValueError):suffix.generate(path,tmp_path/'not_created',wall_seconds=89)
    assert not (tmp_path/'not_created').exists()
