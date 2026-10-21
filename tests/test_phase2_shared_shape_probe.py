"""CPU integration of projected derivatives; no MuJoCo generation/execution."""
import ast
import copy
import json
import importlib.util

import numpy as np
import pytest

from human_punching import phase2_shared_probe_shape_v2 as probe
from human_punching.phase2_shape_derivatives import VERSION
from human_punching.wbc.provenance import PROJECT_ROOT
from test_phase2_shape_derivatives import synthetic_shared


def plan():
    p=json.loads((PROJECT_ROOT/'outputs/task10b/p2sharedprobeprep01/plan03.json').read_text())
    p['schema']=probe.VERSION;p['derivative_projection_version']=VERSION
    for name in ('predecessor_plan','predecessor_generation_manifest','algebraic_review'):
        p['inputs'][name]=dict(path='outputs/task10b/fixture.json',sha256='a'*64)
    return p


def test_shape_plan_requires_explicit_new_version_and_all_ancestry_bindings():
    p=plan();assert probe.validate_plan(p) is p
    for key in ('predecessor_plan','predecessor_generation_manifest','algebraic_review'):
        changed=copy.deepcopy(p);del changed['inputs'][key]
        with pytest.raises(ValueError):probe.validate_plan(changed)
    for value in (None,'task10b_legacy_fd'):
        changed=copy.deepcopy(p);changed['derivative_projection_version']=value
        with pytest.raises(ValueError):probe.validate_plan(changed)


@pytest.mark.parametrize('key,value',[('preparation_s',1.),('learned_qualification',True),
    ('selected_start_index',0),('requested_timings_s',[.6,.6,1.1,1.2])])
def test_shape_successor_keeps_fixed_physical_request(key,value):
    p=plan();p[key]=value
    with pytest.raises(ValueError):probe.validate_plan(p)


def test_final_feedforward_receives_actual_projected_all_hinge_derivatives(monkeypatch):
    model,candidate,metadata=synthetic_shared();before=copy.deepcopy(candidate);seen=[]
    def feedforward(model_arg,actual,receipt):
        assert model_arg is model
        assert actual['source_segment_coefficients'].shape==(16,6,48)
        assert actual['qpos'].tobytes()==before['qpos'].tobytes()
        assert actual['qvel'][:5].tobytes()==before['qvel'][:5].tobytes()
        assert np.any(actual['qvel'][5:]!=before['qvel'][5:])
        assert receipt['independent_polynomial_extrema_screen_passed']
        assert receipt['feedforward_recomputed'] is False
        seen.append(receipt)
        return actual,{**receipt,'feedforward_recomputed':True}
    monkeypatch.setattr(probe,'finalize_shared_target_feedforward',feedforward)
    out,receipt=probe.refine_then_finalize(model,candidate,metadata)
    assert len(seen)==1 and receipt['feedforward_recomputed']
    assert out['schedule']==before['schedule']


def test_invalid_projection_never_reaches_feedforward(monkeypatch):
    model,candidate,metadata=synthetic_shared();candidate['qpos'][11,7]=1.01
    def forbidden(*a):raise AssertionError('Invalid source must not reach feedforward')
    monkeypatch.setattr(probe,'finalize_shared_target_feedforward',forbidden)
    with pytest.raises(ValueError):probe.refine_then_finalize(model,candidate,metadata)


def test_static_successor_preserves_all_other_probe_algorithms_and_launch_protocol():
    parent=ast.parse((PROJECT_ROOT/'src/human_punching/phase2_shared_probe.py').read_text())
    child=ast.parse((PROJECT_ROOT/'src/human_punching/phase2_shared_probe_shape_v2.py').read_text())
    def declarations(tree):return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=declarations(parent),declarations(child)
    assert set(b)-set(a)=={'refine_then_finalize'}
    for name in a:
        if name not in ('generate','load_context','validate_plan'):
            assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False)
    calls=[n.func.id for n in ast.walk(b['generate']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert calls.index('compose_shared_reference')<calls.index('compose_shared_target')<calls.index('refine_then_finalize')
    assert probe.LAUNCH=='task10b_phase2_shared_development_probe_launch_v1'
    # All offline/cycle/launch code beyond the new derivative pipeline remains
    # identical; this version does not alter any physical or learned gate.


def driver():
    path=PROJECT_ROOT/'outputs/task10b/p2shapeprobeprep01/driver.py'
    spec=importlib.util.spec_from_file_location('shape_probe_transport_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_new_transport_preserves_original_allocation_guard_and_native_cleanup():
    def functions(path):
        return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    a=functions(PROJECT_ROOT/'outputs/task10b/p2sharedprobeprep01/driver.py')
    b=functions(PROJECT_ROOT/'outputs/task10b/p2shapeprobeprep01/driver.py')
    for name in ('submit','execute','submission_command','main'):
        assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False)
    source=ast.unparse(b['verify'])
    assert 'allocation_wall_seconds=480' in source and 'application_wall_seconds=420.0' in source
    assert 'evaluation_reserve_seconds=4500' in source
    assert 'child_wall_upper_bound_seconds=360' in source


@pytest.mark.parametrize('seconds',[0,180,361])
def test_transport_rejects_expanded_or_exhausted_child_wall(seconds):
    with pytest.raises(ValueError):driver().native_command({'mode':'shared-probe-generation'},seconds)


def test_transport_dispatch_is_fixed_to_new_shape_module():
    record=dict(mode='shared-probe-generation',inputs=dict(plan=dict(path='outputs/task10b/plan.json',sha256='a'*64)),output='outputs/task10b/fixture/run')
    command=driver().native_command(record,360)
    assert command[command.index('-m')+1]=='human_punching.phase2_shared_probe_shape_v2'
    assert command[-1]=='360'
