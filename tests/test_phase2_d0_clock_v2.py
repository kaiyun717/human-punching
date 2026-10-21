"""Saved-trace postprocessing only; no integration, actor inference or GPU calls."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from human_punching import phase2_learning as v1
from human_punching import phase2_d0_clock_v2 as v2

ROOT=Path(v1.__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('clock_v2_driver',ROOT/'scripts/task10b_phase2_expand.py')
driver=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(driver)


def test_static_v2_is_exact_v1_except_reviewed_clock_and_version_receipt():
    old=ast.parse(Path(v1.__file__).read_text());new=ast.parse(Path(v2.__file__).read_text())
    new.body=[n for n in new.body if not (isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id in ('D0_CLOCK_VERSION','PARENT_SOURCE_SHA256') for t in n.targets))
        and not (isinstance(n,ast.FunctionDef) and n.name=='evaluator_receipt')]
    evaluate=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_policy')
    assignment=next(n for n in ast.walk(evaluate) if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='own_steps' for t in n.targets))
    assert ast.unparse(assignment.value.body)=='int(np.ceil(row[\'episode_duration_s\'] / 0.005 - 1e-10))'
    assignment.value.body=ast.parse('int(np.ceil(row["episode_duration_s"]/.005))',mode='eval').body
    execute=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='execute_cli')
    guards=[n for n in execute.body if isinstance(n,ast.If) and 'Versioned D0 clock adapter' in ast.unparse(n)]
    assert len(guards)==1;execute.body.remove(guards[0])
    assignments=[n for n in execute.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='clock_adapter' for t in n.targets)]
    assert len(assignments)==1;execute.body.remove(assignments[0])
    bindings=[n for n in ast.walk(execute) if isinstance(n,ast.Call) and any(
        k.arg=='postprocessing_clock_adapter' for k in n.keywords)]
    assert len(bindings)==1
    bindings[0].keywords=[k for k in bindings[0].keywords if k.arg!='postprocessing_clock_adapter']
    assert ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False)
    assert v1.sha256_file(v1.__file__)==v2.PARENT_SOURCE_SHA256


def step_counts(module,duration,kind='original_control'):
    tree=ast.parse(Path(module.__file__).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_policy')
    node=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='own_steps' for t in n.targets))
    expression=compile(ast.Expression(node.value),str(module.__file__),'eval')
    return eval(expression,dict(np=np,math=math,row=dict(episode_duration_s=duration,kind=kind)))


@pytest.mark.parametrize('duration',[.02,.2,1.,4.1,4.8,5.])
def test_one_ulp_lattice_noise_does_not_invent_a_saved_step(duration):
    expected=int(round(duration/.005))
    for value in (np.nextafter(duration,-np.inf),duration,np.nextafter(duration,np.inf)):
        assert step_counts(v2,float(value))==expected
        assert step_counts(v2,float(value))<=int(math.ceil(value/.020-1e-10))*4


@pytest.mark.parametrize('duration',[.020001,.201,1.002,4.801,4.804999,4.805001])
def test_true_nonlattice_durations_remain_rounded_up(duration):
    assert step_counts(v2,duration)==step_counts(v1,duration)==int(math.ceil(duration/.005))
    assert step_counts(v2,duration)<=int(math.ceil(duration/.020-1e-10))*4


@pytest.mark.parametrize('duration',[.020000000000000004,4.800000000000001,4.803,5.])
def test_retimed_and_actor_padded_horizons_are_unchanged(duration):
    assert step_counts(v2,duration,'uniform')==step_counts(v1,duration,'uniform')


def test_adapter_is_frozen_only_and_pinned_before_any_runtime_or_cuda(monkeypatch):
    receipt=v2.evaluator_receipt()
    assert receipt['schema']==v2.D0_CLOCK_VERSION and receipt['native_execution_unchanged']
    assert not receipt['training_authorized'] and receipt['parent_source']['sha256']==v1.sha256_file(v1.__file__)
    monkeypatch.setattr(v2,'sha256_file',lambda path:'0'*64)
    with pytest.raises(ValueError,match='Pinned'):v2.evaluator_receipt()
    for mode,updates in [('pilot',0),('frozen',1)]:
        monkeypatch.setattr(v2.sys,'argv',['clock','--mode',mode,'--max-updates',str(updates),
            '--family','cross','--catalog','unused','--output','unused'])
        with pytest.raises(ValueError,match='zero updates'):v2.execute_cli()


def test_only_new_timing_control_commands_dispatch_the_v2_static_module():
    record=dict(inputs=dict(catalog=str(ROOT/'outputs/task10b/p2timingcatalog01/manifest.json')),
        application_wall_seconds=390,family='cross',output='outputs/task10b/synthetic/run',
        release='outputs/four-punch-milestone',predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json')
    for mode,module in [('timing-control','human_punching.phase2_d0_clock_v2'),('frozen-control','human_punching.phase2_learning')]:
        command=driver.native_command(dict(mode=mode,**record),355.)
        assert command[command.index('-m')+1]==module
        assert command[command.index('--mode')+1]=='frozen'
        assert command[command.index('--max-updates')+1]=='0'
        assert command[command.index('--wall-seconds')+1]=='355.0'


def postprocess(module):
    """Execute the actual evaluator's postprocessing AST against stored arrays."""
    tree=ast.parse(Path(module.__file__).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_policy')
    processing=next(n for n in fn.body if isinstance(n,ast.If) and ast.unparse(n.test)=='error is None')
    wrapper=ast.FunctionDef(name='replay',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=k) for k in
        ('folder','arrays','env','chosen','reference_ids','trial_ids','cfg','audit','reports','started')],
        kwonlyargs=[],kw_defaults=[],defaults=[]),body=copy.deepcopy(processing.body)+[ast.Return(ast.Name('reports',ast.Load()))],decorator_list=[])
    compiled=compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(module.__file__),'exec')
    namespace=dict(module.__dict__);exec(compiled,namespace);return namespace['replay']


def replay_environment(family,row,run):
    model=mujoco.MjModel.from_xml_path(str(run/'runtime.xml'))
    ids,qadr,_=v1.actuator_layout(model)
    with np.load(ROOT/'outputs/four-punch-milestone'/family/'reference.npz',allow_pickle=False) as archive:
        nominal_q=archive['qpos'][0].copy()
    library=v1.ReferenceLibrary(model,[row])
    return SimpleNamespace(model=model,ids=ids,qadr=qadr,device='cpu',library=library,
        active_hand_index=0 if family in ('jab','left_hook') else 1,
        nominal=v1.derive_guard_nominal(model,nominal_q,family=family))


@pytest.fixture(scope='module')
def actual_catalog():return json.loads((ROOT/'outputs/task10b/p2timingcatalog01/manifest.json').read_text())


def retained_inputs(run,candidate,worlds):
    path=run/'validation_00000'/candidate/'raw_trace.npz'
    with np.load(path,allow_pickle=False) as archive:
        return {k:archive[k][:,worlds].copy() for k in ('qpos','qvel','support_contact','launched')}


def call_replay(module,folder,arrays,env,row):
    folder.mkdir();n=arrays['qpos'].shape[1]
    return postprocess(module)(folder,arrays,env,[row]*n,np.zeros(n,dtype=int),
        [f"{row['candidate_id']}_bank_{i:02d}" for i in range(n)],v1.FunctionalGuardThresholds(),
        {'passed':True},[],0.)


def test_actual_cross_complete_trace_reconstructs_all_eight_worlds_without_changing_failed_packet(tmp_path,actual_catalog):
    run=ROOT/'outputs/task10b/p2t01c/run';row=next(r for r in actual_catalog['cells'] if r['candidate_id']=='cross_original')
    raw=run/'validation_00000/cross_original/raw_trace.npz';before=v1.sha256_file(raw)
    arrays=retained_inputs(run,row['candidate_id'],slice(None));assert len(arrays['qpos'])==961
    assert step_counts(v1,row['episode_duration_s'])+1==962
    assert step_counts(v2,row['episode_duration_s'])+1==961
    env=replay_environment('cross',row,run)
    with pytest.raises(ValueError,match='saved time axis'):call_replay(v1,tmp_path/'old',arrays,env,row)
    result=call_replay(v2,tmp_path/'v2',arrays,env,row)
    assert len(result)==8
    for world,report in enumerate(result):
        with np.load(tmp_path/'v2'/f'trial_{world:04d}.npz',allow_pickle=False) as archive:
            assert len(archive['time_s'])==len(archive['qpos'])==961
            np.testing.assert_array_equal(archive['qpos'],arrays['qpos'][:,world])
        assert report['candidate_id']=='cross_original'
    assert v1.sha256_file(raw)==before and not (run/'validation_00000/cross_original/summary.json').exists()


@pytest.mark.parametrize('candidate',['jab_original','jab_uniform_0p600','jab_uniform_0p750','jab_uniform_0p900'])
def test_actual_jab_original_and_retimed_postprocessing_parity(tmp_path,actual_catalog,candidate):
    run=ROOT/'outputs/task10b/p2t01j/run';row=next(r for r in actual_catalog['cells'] if r['candidate_id']==candidate)
    arrays=retained_inputs(run,candidate,[0]);env=replay_environment('jab',row,run)
    old=call_replay(v1,tmp_path/'old',arrays,env,row);new=call_replay(v2,tmp_path/'v2',arrays,env,row)
    for report in old+new:report.pop('elapsed_wall_s')
    assert old==new
    with np.load(tmp_path/'old/trial_0000.npz',allow_pickle=False) as a,np.load(tmp_path/'v2/trial_0000.npz',allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        for name in a.files:np.testing.assert_array_equal(a[name],b[name])


def test_fresh_timing_packet_predeclares_independent_clock_and_binds_both_sources():
    inputs=dict(catalog=str(ROOT/'outputs/task10b/p2timingcatalog01/manifest.json'),family='cross')
    scope=driver.validate_spec('timing-control',inputs,480,390)
    assert scope['d0_clock_versions']==dict(evaluator=v2.D0_CLOCK_VERSION,independent_audit='task10b_phase2_independent_clock_v2')
    record=dict(mode='timing-control',inputs=inputs,bundle='outputs/task10/release_baseline_02',
        release='outputs/four-punch-milestone',predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json',**scope)
    closure=driver.required_inputs(record)
    for path in ('src/human_punching/phase2_d0_clock_v2.py','scripts/task10b_phase2_audit_clock_v2.py',
                 'src/human_punching/phase2_learning.py','scripts/task10b_phase2_audit.py'):
        assert closure[path]==v1.sha256_file(ROOT/path)
