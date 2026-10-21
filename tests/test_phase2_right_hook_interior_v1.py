"""Pure scalar-clock contracts and saved-input checks; no native construction."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from human_punching import phase2_right_hook_interior_v1 as mixed


@pytest.fixture
def parents():
    root=Path(__file__).resolve().parents[1]
    selection=json.loads((root/"outputs/task10b/p2allfamilytargetprep01/selection.json").read_text())
    return deepcopy(selection["families"]["right_hook"]["selected_source_rows"])


def test_actual_parent_blend_uses_source_landmarks_not_rounded_parent_events(parents):
    old,new,left,right=mixed.blended_knots(*parents)
    assert np.array_equal(left,old*parents[0]["duration_scale"])
    assert np.array_equal(new,(left+right)/2)
    assert new[1] != parents[0]["launch_time_s"]
    assert new[-1] != (parents[0]["duration_s"]+parents[1]["duration_s"])/2
    curve,record=mixed.clock_contract(old,new,0.)
    np.testing.assert_allclose(curve(new),old,rtol=0,atol=1e-12)
    assert record["maximum_clock_velocity_jump"]<1e-10
    assert record["maximum_clock_acceleration_jump"]>0
    assert record["original_postcontact_intervals_claimed_preserved"] is False


@pytest.mark.parametrize("offset",[-.040,0.,.040])
def test_contact_offset_preserves_blended_intervals_and_prefix_only(parents,offset):
    old,new,_,_=mixed.blended_knots(*parents);curve,record=mixed.clock_contract(old,new,offset)
    actual=np.asarray(record["final_knots_s"])
    assert np.array_equal(actual[:2],new[:2])
    np.testing.assert_allclose(actual[2:]-new[2:],offset,rtol=0,atol=1e-15)
    np.testing.assert_allclose(np.diff(actual[2:]),np.diff(new[2:]),rtol=0,atol=1e-15)
    assert record["local_source_rate_min"]>=.5 and record["local_source_rate_max"]<=1.5
    sample=np.linspace(0,actual[-1],1001)
    assert np.all(np.diff(curve(sample))>0)


@pytest.mark.parametrize("offset",[.04001,-.04001,float("nan"),float("inf"),True])
def test_offsets_cannot_relax_contact_anchor_bound(parents,offset):
    old,new,_,_=mixed.blended_knots(*parents)
    with pytest.raises(ValueError,match="bounded"):mixed.clock_contract(old,new,offset)


@pytest.mark.parametrize("mutation",["different_source","different_family","wrong_ttc","wrong_kind","bad_scale"])
def test_no_nearest_or_different_ancestry_parent(parents,mutation):
    if mutation=="different_source":parents[1]["source_reference_sha256"]="0"*64
    elif mutation=="different_family":parents[1]["family"]="left_hook"
    elif mutation=="wrong_ttc":parents[1]["requested_ttc_s"]=1.25
    elif mutation=="wrong_kind":parents[0]["kind"]="nonuniform_contact_anchored"
    else:parents[0]["duration_scale"]=float("nan")
    with pytest.raises(ValueError):mixed.blended_knots(*parents)


def test_rate_envelope_checked_on_entire_polynomial(parents):
    old,new,_,_=mixed.blended_knots(*parents);new[-1]+=50
    with pytest.raises(ValueError,match="rate"):mixed.clock_contract(old,new,0.)


def test_sampled_derivative_and_event_build_delegates_exact_free_root_helper(parents,monkeypatch):
    old,new,_,_=mixed.blended_knots(*parents);t=np.linspace(0,old[-1],81)
    q=np.zeros((len(t),55));q[:,3]=1;q[:,7]=np.clip(t-.25,0,1)
    source=dict(t=t,qpos=q,head_center=np.array([.8,.2,1.2]),active_hand_index=1)
    calls=[]
    def tangent(model,grid,positions):
        calls.append((model,grid.copy(),positions.copy()))
        result=np.zeros((len(grid),54));result[:,6:]=np.gradient(positions[:,7:],grid,axis=0,edge_order=2)
        return result
    monkeypatch.setattr(mixed.reference,"tangent_velocities",tangent)
    candidate,_=mixed.retime_mixed(SimpleNamespace(nq=55),source,old,new,0.)
    assert len(calls)==1
    np.testing.assert_allclose(np.diff(candidate["t"]),.005,rtol=0,atol=1e-15)
    np.testing.assert_array_equal(candidate["qacc"],np.gradient(candidate["qvel"],candidate["t"],axis=0,edge_order=2))
    assert candidate["events"]["launch_tick"]%4==0
    # The unchanged shared generation cell writes full-duration+1s into the row;
    # the event helper supplies clocks, rather than a second phase denominator.
    assert "plus1s terminal hold" in candidate["events"]["phase_semantics"]
    assert candidate["t"][-1]>=new[-1] and candidate["t"][-1]-new[-1]<.005
    assert candidate["events"]["source_launch_annotation_is_not_actual_launch"] is True
    assert candidate["qpos"].shape[1]==55 and candidate["qvel"].shape[1]==54


def test_solve_has_bounded_correction_no_policy_lag_compensation(parents,monkeypatch):
    old,new,_,_=mixed.blended_knots(*parents);offsets=[]
    monkeypatch.setattr(mixed.prior,"source_landmarks",lambda *args:(old,{}))
    def retime(model,source,old,initial,offset):
        offsets.append(offset)
        return dict(t=np.array([0.,1.]),qpos=np.zeros((2,55)),head_center=[0,0,0],active_hand_index=1,
            events=dict(launch_time_s=.24,outward_completion_time_s=3.)),dict(offset=offset)
    monkeypatch.setattr(mixed,"retime_mixed",retime)
    monkeypatch.setattr(mixed.reference,"glove_positions",lambda *args:np.zeros((2,2,3)))
    monkeypatch.setattr(mixed.reference,"contact_diagnostics",lambda *args:dict(timing_error_s=.2,timing_valid=False))
    candidate,search=mixed.solve_mixed(None,{},1.1,*parents)
    assert offsets==[0.,-.04] and search["solver_residual_within_1ms"] is False
    assert search["reference_contact_compensation_for_policy_lag"] is False
    assert candidate["requested_ttc_s"]==1.1


def test_generation_refuses_cpu_without_allocated_cuda_before_creating_output(tmp_path):
    with pytest.raises(RuntimeError):mixed.generate(tmp_path/"missing_plan.json",tmp_path/"fresh")
    assert not (tmp_path/"fresh").exists()


def test_private_generation_dependency_leaves_frozen_module_unchanged(parents):
    before=mixed.prior.solve_contact_anchored
    engine=mixed._engine(*parents)
    assert engine is not mixed.prior and engine.solve_contact_anchored is not before
    assert mixed.prior.solve_contact_anchored is before
    assert mixed.prior.VERSION!=engine.VERSION==mixed.VERSION


@pytest.mark.parametrize("field,value",[("cells",[]),("application_wall_seconds",91),("blend_lambda",.6),
    ("contact_anchor_radius_s",.05),("training",True),("physics_dt_s",.0025)])
def test_plan_contract_rejects_scientific_or_budget_change_before_io(field,value):
    with pytest.raises(ValueError,match="Exact one-cell"):
        mixed.validate_plan({field:value})


@pytest.mark.parametrize("failure",[None,"geometric","timeout"])
def test_saved_generation_protocol_preserves_all22_rows_and_failure_extent(tmp_path,monkeypatch,failure):
    root=Path(__file__).resolve().parents[1]
    plan_path=root/"outputs/task10b/p2rhinteriorprep01/plan.json"
    plan=json.loads(plan_path.read_text());parent=json.loads(Path(plan["parent_catalog"]["path"]).read_text())
    before=deepcopy(parent["cells"])
    paths={k:Path(plan[k]["path"]) for k in ("selection","predicate")}
    # Every native/model/screen call is replaced; only the fresh saved ledger is exercised.
    monkeypatch.setattr(mixed,"require_allocation",lambda:None)
    monkeypatch.setattr(mixed,"validate_plan",lambda p:(paths,parent,1000))
    monkeypatch.setattr(mixed,"_native_context",lambda p:{})
    def cell(*args):
        if failure=="geometric":raise ValueError("synthetic reference rejection")
        if failure=="timeout":raise mixed.BoundedExecutionInterruption(dict(reason="synthetic_timeout",boundary="mock_cell"))
        return dict(**mixed.ENTRY,generation_disposition="generated",kinematic_admission=True)
    monkeypatch.setattr(mixed,"_engine",lambda *args:SimpleNamespace(_generate_cell=cell))
    output=tmp_path/"generated"
    result=mixed.generate(plan_path,output)
    assert result["cells"][:22]==before and parent["cells"]==before
    assert result["planned_cells"]==23 and result["executed_new_cells"]==1
    assert result["cells"][-1]["training_eligible"] is False
    assert result["cells"][-1]["generation_disposition"]=={None:"generated","geometric":"failed","timeout":"timeout"}[failure]
    assert result["complete"]==(failure!="timeout")
    assert (result["exception"] is not None)==(failure=="timeout")
    with pytest.raises(FileExistsError):mixed.generate(plan_path,output)


def test_real_plan_exact_saved_inputs_and_owned_source_closure():
    root=Path(__file__).resolve().parents[1];plan_path=root/"outputs/task10b/p2rhinteriorprep01/plan.json"
    plan=json.loads(plan_path.read_text());paths,parent,byte_count=mixed.validate_plan(plan)
    closure=mixed.files(plan_path)
    assert len(parent["cells"])==22 and byte_count+mixed.CELL_BYTES<=mixed.RESIDENT_BYTES
    assert str(plan_path.relative_to(root)) in closure
    assert all(name in closure for name in mixed.SOURCES)
    assert paths["selection"]==Path(plan["selection"]["path"])
