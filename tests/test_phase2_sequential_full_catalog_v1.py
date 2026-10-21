"""Metadata and dispatch checks only; no reference construction or integration."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from human_punching import phase2_sequential_full_catalog_v1 as full
from human_punching import phase2_sequential_target_catalog_v3 as probe

ROOT=full.PROJECT_ROOT


@pytest.fixture(scope='module')
def plans(tmp_path_factory):
    folder=tmp_path_factory.mktemp('fullplans')/'plans'
    paths=full.create_plans(ROOT/'outputs/task10b/p2sequentialprobe02/plan.json',
        ROOT/'outputs/task10b/p2sequentialaudit02/audit.json',
        ROOT/'outputs/task10b/p2targetexpansionprep02/train_plan.json',
        ROOT/'outputs/task10b/p2targetexpansionprep02/validation_plan.json',folder)
    return {split:full.read(item) for split,item in paths.items()}


@pytest.fixture(scope='module')
def driver():
    spec=importlib.util.spec_from_file_location('test_full_driver',ROOT/'scripts/task10b_phase2_sequential_full_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_real_exact64_24_same_recipe_and_interior_exclusion(plans):
    assert plans['train']['recipe']==plans['validation']['recipe']
    for split,n in [('train',64),('validation',24)]:
        settings,cells,paths=full.resolve_cells(plans[split])
        assert len(cells)==n and settings.base.maximum_joint_delta_rad==.205
        assert {c[2]['requested_ttc_s'] for c in cells}==({.6,.75} if split=='train' else {.6,.675,.75})
        assert set(full.EVIDENCE).issubset(paths)
        assert all(c[1]['placement']==dict(forward_m=0.,left_m=0.,yaw_rad=0.) for c in cells)
    assert sum('c4184_' in r['candidate_id'] for r in plans['train']['cells'])==2


@pytest.mark.parametrize('change',[
    lambda p:p.update(split='test'),lambda p:p.update(family='cross'),
    lambda p:p.update(maximum_cells=63),lambda p:p['cells'].pop(),
    lambda p:p.update(maximum_child_seconds=359),lambda p:p.update(output_byte_cap=2**31),
    lambda p:p.update(physical_qualification=True),
    lambda p:p.update(native_frozen_control_required_before_learning=False),
    lambda p:p['retarget_settings'].update(maximum_joint_delta_rad=.21),
    lambda p:p['retarget_settings'].update(maximum_iterations=81),
    lambda p:p['cells'][0].update(parent_candidate_id='jab_uniform_interior_0p675'),
    lambda p:p['cells'][0].update(target_candidate_identity=p['cells'][2]['target_candidate_identity']),
    lambda p:p['cells'].reverse(),lambda p:p['sources'].pop(next(iter(p['sources']))),
])
def test_contract_changes_rejected(plans,change):
    p=deepcopy(plans['train']);change(p)
    with pytest.raises((ValueError,KeyError)):full.resolve_cells(p)


def test_failed_probe_cannot_unlock_full_catalog(plans,monkeypatch):
    original=full.read
    def read(item):
        result=original(item)
        if item==plans['train']['probe_audit']:result['passed']=False
        return result
    monkeypatch.setattr(full,'read',read)
    with pytest.raises(ValueError,match='counted and independently'):full.resolve_cells(plans['train'])


def test_recipe_cannot_change_between_splits(plans,monkeypatch):
    original=full.original.resolve_cells
    def resolve(plan):
        settings,cells,paths=original(plan)
        # No patched recipe semantics: direct mutation must be rejected by SHA.
        return settings,cells,paths
    monkeypatch.setattr(full.original,'resolve_cells',resolve)
    plan=deepcopy(plans['validation']);plan['recipe']['sha256']='0'*64
    with pytest.raises(ValueError,match='hash changed'):full.resolve_cells(plan)


def test_private_engine_has_exact_solver_and_does_not_mutate_probe():
    original_resolver=probe.resolve_cells
    engine=full._engine();engine.resolve_cells=lambda p:None
    assert probe.resolve_cells is original_resolver
    assert engine.retarget_sequential is probe.retarget_sequential
    assert engine.finalize_original_feedforward is probe.finalize_original_feedforward
    assert engine.screen_chunks is probe.screen_chunks
    assert Path(engine.__file__).resolve()==Path(probe.__file__).resolve()


@pytest.mark.parametrize('split',['train','validation'])
def test_exact_native_deadlines(plans,driver,split,tmp_path):
    p=tmp_path/(split+'.json');p.write_text(json.dumps(plans[split]));count,allocation,app,child=full.LIMITS[split]
    inputs={'plan':str(p)};meta=driver.validate_spec(driver.MODE,inputs,allocation,app)
    record=dict(mode=driver.MODE,inputs=inputs,output='outputs/task10b/synthetic_not_run',
        application_wall_seconds=app,allocation_wall_seconds=allocation,**meta)
    cmd=driver.native_command(record,child)
    assert cmd[-1]==str(float(child)) and 'human_punching.phase2_sequential_full_catalog_v1' in cmd
    for seconds in (child-.001,float('nan'),app,True):
        with pytest.raises(RuntimeError):driver.native_command(record,seconds)
    assert meta['fresh_transition_cap']==0


def test_real_input_closure_keeps_original_plans_and_all_evidence(plans,driver,tmp_path):
    p=tmp_path/'plan.json';p.write_text(json.dumps(plans['train']))
    result=driver.files(p)
    for item in [*plans['train']['original_plans'].values(),*(plans['train'][n] for n in full.EVIDENCE)]:
        assert result[str(full.own(item).relative_to(ROOT))]==item['sha256']
    probe_plan=full.read(plans['train']['probe_plan'])
    for key in ('engineering_bound_review','selection_audit'):
        item=probe_plan[key]
        assert result[str(full.own(item).relative_to(ROOT))]==item['sha256']
    assert not any('gridsan' in n for n in result)


def test_wrong_child_rejected_before_engine_or_output(plans,tmp_path,monkeypatch):
    p=tmp_path/'plan.json';p.write_text(json.dumps(plans['train']))
    monkeypatch.setattr(full,'_engine',lambda:pytest.fail('must not instantiate engine'))
    with pytest.raises(ValueError,match='Exact split child'):full.generate(p,tmp_path/'run',wall_seconds=240.)
    assert not (tmp_path/'run').exists()


def test_wrapper_preserves_negative_and_unattempted_result(plans,tmp_path,monkeypatch):
    p=tmp_path/'plan.json';p.write_text(json.dumps(plans['train']))
    result=dict(complete=False,cells=[dict(generation_disposition='timeout')],unattempted_cells=plans['train']['cells'][1:])
    def generate(path,output,*,wall_seconds):
        assert wall_seconds==360 and full.read(full.binding(path))==plans['train']
        Path(output).mkdir();(Path(output)/'manifest.json').write_text(json.dumps(result));return result
    engine=SimpleNamespace(generate=generate)
    monkeypatch.setattr(full,'_engine',lambda:engine)
    actual=full.generate(p,tmp_path/'run',wall_seconds=360.)
    assert actual is result and engine.resolve_cells is full.resolve_cells
    assert json.loads((tmp_path/'run/manifest.json').read_text())==result
    sidecar=json.loads((tmp_path/'run/full_catalog_entry.json').read_text())
    assert sidecar['physical_qualification'] is False and sidecar['solver_changed'] is False
