"""Single probe dispatch tests: no scheduler, model construction or references."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('sequential_v3_probe_launcher',
    Path(__file__).resolve().parents[1]/'scripts/task10b_phase2_sequential_v3.py')
launcher = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(launcher)


def test_durable_lock_and_watchdog_stay_original():
    assert launcher.legacy.SCHEMA == launcher.SCHEMA
    assert launcher.legacy.MODES == (launcher.MODE,) and launcher.legacy.CONDITIONAL_MODES == ()
    for name in ('submit','verify','execute'):
        assert getattr(launcher.legacy,name).__module__ == 'task10b_phase2_sequential_v3_frozen_driver'
    assert launcher.legacy.budget_status.__module__ == 'human_punching.phase2_budget'


@pytest.mark.parametrize('mode,inputs,wall,app,proof', [
    ('target-references',{'plan':'x'},360,300,None),
    ('sequential-target-references',{'plan':'x','command':'bad'},360,300,None),
    ('sequential-target-references',{'plan':'x'},420,300,None),
    ('sequential-target-references',{'plan':'x'},360,301,None),
    ('sequential-target-references',{'plan':'x'},360,270,None),
    ('sequential-target-references',{'plan':'x'},360,300,{}),
    ('sequential-target-references',{'plan':'x'},float('nan'),300,None),
])
def test_no_unreviewed_scope_or_limits(mode, inputs, wall, app, proof):
    with pytest.raises(ValueError): launcher.validate_spec(mode, inputs, wall, app, proof=proof)


def test_actual_scientific_resolver_is_required(tmp_path,monkeypatch):
    from human_punching import phase2_sequential_target_catalog_v3 as catalog
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'path_solver_version':'version'}))
    monkeypatch.setattr(launcher.legacy.base,'local',lambda p:Path(p).resolve())
    calls=[]
    def resolve(p):
        calls.append(p)
        return SimpleNamespace(base=SimpleNamespace(maximum_joint_delta_rad=.205)),[None]*4,{}
    monkeypatch.setattr(catalog,'resolve_cells',resolve)
    result=launcher.validate_spec(launcher.MODE,{'plan':str(plan)},360,300)
    assert len(calls)==1 and result['fresh_transition_cap']==0 and result['sequential_probe_cells']==4
    assert result['sequential_generator_output_byte_cap']==128*1024**2
    assert result['sequential_engineering_correction_cap_rad']==.205 and result['physical_joint_limits_relaxed'] is False
    def reject(p): raise ValueError('invalid immutable four-cell plan')
    monkeypatch.setattr(catalog,'resolve_cells',reject)
    with pytest.raises(ValueError,match='immutable'):launcher.validate_spec(launcher.MODE,{'plan':str(plan)},360,300)


def test_generator_budget_uses_exact_child_and_refuses_late_setup(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher.legacy.base,'local',lambda p:Path(p).resolve())
    record=dict(mode=launcher.MODE,inputs={'plan':str(tmp_path/'plan.json')},output='outputs/task10b/probe/run',
        sequential_child_wall_seconds=240,allocation_wall_seconds=360,application_wall_seconds=300,
        sequential_engineering_correction_cap_rad=.205,physical_joint_limits_relaxed=False)
    for available in (None,270,241,240):
        command=launcher.native_command(record,available)
        assert command[2:4]==['-m','human_punching.phase2_sequential_target_catalog_v3']
        assert command[-2:]==['--wall-seconds','240.0']
    for available in (239.999,120,-1,271,True,float('inf'),float('nan')):
        with pytest.raises(RuntimeError,match='Setup'):launcher.native_command(record,available)
    for key,value in [('mode','target-references'),('application_wall_seconds',301),('sequential_child_wall_seconds',241),
                      ('sequential_engineering_correction_cap_rad',.206),('physical_joint_limits_relaxed',True)]:
        with pytest.raises(ValueError):launcher.native_command({**record,key:value},260)


def test_only_batch_slot_changes(monkeypatch):
    expected=['sbatch','--time=6','--gres=gpu:1',str(launcher.ROOT/'scripts/task10b_phase2_expand.sbatch'),'packet']
    monkeypatch.setattr(launcher,'_submission_command',lambda *a:expected.copy())
    actual=launcher.submission_command({},'unused')
    assert actual[:-2]==expected[:-2] and actual[-1]==expected[-1]
    assert actual[-2]==str(launcher.ROOT/'scripts/task10b_phase2_sequential_v3.sbatch')


def test_required_inputs_refuses_missing_namespace_and_extra_seed(tmp_path,monkeypatch):
    record=dict(mode=launcher.MODE,inputs={'plan':str(tmp_path/'plan.json')})
    with pytest.raises(ValueError,match='proof'):launcher.required_inputs(record)
    record['runtime_namespace_proof']={'bound':True}
    monkeypatch.setattr(launcher.legacy.base,'local',lambda p:Path(p).resolve())
    original=launcher.legacy.semantic_files
    def seeds(r,**kwargs):return launcher.legacy.semantic_files([tmp_path/'wrong.json'])
    monkeypatch.setattr(launcher,'_required_inputs',seeds)
    with pytest.raises(ValueError,match='Unexpected'):launcher.required_inputs(record)
    assert launcher.legacy.semantic_files is original


def test_target_closure_injection_restores_on_scientific_failure(tmp_path,monkeypatch):
    from human_punching import phase2_target_inputs as target
    from human_punching import phase2_sequential_target_catalog_v3 as sequential
    old=target.resolve_cells
    def failed(p):
        assert target.resolve_cells is sequential.resolve_cells
        raise ValueError('exact target rejection')
    monkeypatch.setattr(target,'files',failed)
    with pytest.raises(ValueError,match='exact target'):launcher.files(tmp_path/'unused.json')
    assert target.resolve_cells is old


def test_prepare_cannot_omit_reviewed_namespace():
    with pytest.raises(ValueError,match='namespace'):launcher.prepare('unused',launcher.MODE,{})
