"""CPU dispatch/ancestry/order checks for the unprepared motion probe version."""
import ast
import copy
from dataclasses import asdict
import json
import pytest

from human_punching import phase2_shared_probe_motion_v3 as probe
from human_punching.phase2_shared_motion import LowerMotionSettings,VERSION
from human_punching.wbc.provenance import PROJECT_ROOT
from test_phase2_shared_motion import dynamic_candidate


def plan():
    p=json.loads((PROJECT_ROOT/'outputs/task10b/p2shapeprobeprep01/plan.json').read_text())
    p.update(schema=probe.VERSION,motion_version=VERSION,lower_motion_settings=asdict(LowerMotionSettings()))
    for key in ('motion_proposal','failed_shape_generation_manifest'):
        p['inputs'][key]=dict(path='outputs/task10b/fixture.json',sha256='a'*64)
    return p


def test_motion_plan_is_explicit_and_cannot_accept_static_recipe_or_larger_box():
    p=plan();assert probe.validate_plan(p) is p
    for key,value in (('schema','task10b_phase2_four_family_shared_shape_development_probe_v2'),
            ('motion_version','static_root'),('lower_motion_settings',asdict(LowerMotionSettings())|{'maximum_joint_difference_from_source_rad':.61})):
        changed=copy.deepcopy(p);changed[key]=value
        with pytest.raises(ValueError):probe.validate_plan(changed)
    del p['inputs']['failed_shape_generation_manifest']
    with pytest.raises(ValueError):probe.validate_plan(p)


def test_motion_events_and_hinge_derivatives_precede_original_feedforward(monkeypatch):
    model,candidate,metadata,source=dynamic_candidate();seen=[]
    def feedforward(model,actual,receipt):
        assert receipt['motion_event_correspondence_after_target']['observed_first_outward_actor_boundary_tick']==8
        assert actual['source_segment_coefficients'].shape==(16,6,48)
        seen.append(True);return actual,receipt
    monkeypatch.setattr(probe,'finalize_shared_target_feedforward',feedforward)
    probe.refine_then_finalize(model,candidate,metadata);assert seen==[True]


def test_changed_motion_boundary_never_reaches_feedforward(monkeypatch):
    model,candidate,metadata,source=dynamic_candidate();candidate['qpos'][7,0]=.001
    monkeypatch.setattr(probe,'finalize_shared_target_feedforward',lambda *a:pytest.fail('Clock mismatch reached FF'))
    with pytest.raises(ValueError):probe.refine_then_finalize(model,candidate,metadata)


def test_static_motion_probe_preserves_native_protocol_and_negative_ledger_algorithms():
    def definitions(name):
        tree=ast.parse((PROJECT_ROOT/'src/human_punching'/name).read_text())
        return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a=definitions('phase2_shared_probe_shape_v2.py');b=definitions('phase2_shared_probe_motion_v3.py')
    assert set(a)==set(b)
    for name in a:
        if name not in ('generate','load_context','validate_plan','refine_then_finalize'):
            assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False)
    calls=[n.func.id for n in ast.walk(b['generate']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert calls.index('compose_shared_reference')<calls.index('compose_shared_target')<calls.index('refine_then_finalize')
    assert probe.LAUNCH=='task10b_phase2_shared_development_probe_launch_v1'
