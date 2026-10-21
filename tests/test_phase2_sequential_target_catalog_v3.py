"""Four-cell selection, retained negative ledgers and mocked generation only."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from human_punching import phase2_sequential_target_catalog_v3 as catalog
from human_punching.phase2_target_path_reference_v3 import SequentialSettings
from human_punching.phase2_target_reference import RetargetSettings, RetargetRejection
from human_punching.phase2_deadline import BoundedExecutionInterruption


NAMES = tuple("Left" + s for s in ("Shoulder_x", "Shoulder_y", "Shoulder_z", "Arm_x", "Arm_y", "Arm_z", "ForeArm_z", "ForeArm_x", "Hand_x", "Hand_y", "Hand_z"))


def encoded(path, data):
    path.write_text(json.dumps(data))
    return dict(path=str(path), sha256=catalog.sha256_file(path))


def training_audit():
    cells = []
    for group in [4184, 4251] + list(range(30)):
        for timing in (.6, .75):
            row = dict(command_id=group, candidate_id=f"jab_train_c{group}_{timing}")
            if group == 4184:
                row["generation_disposition"] = "failed"
            else:
                row["maximum_5ms_correction_step_rad"] = 1. if group == 4251 else .01
            cells.append(row)
    return dict(passed=True, generation=dict(planned=64), cells=cells)


def test_selection_uses_distinct_worst_training_group_only():
    assert catalog.select_groups(training_audit()) == (4184, 4251)
    for change in ("validation", "two_failed", "count"):
        audit = training_audit()
        if change == "validation": audit["cells"][2]["candidate_id"] = "jab_validation_c4251"
        elif change == "two_failed": audit["cells"][2]["generation_disposition"] = "failed"
        else: audit["cells"].pop()
        with pytest.raises(ValueError): catalog.select_groups(audit)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    settings = RetargetSettings("Left", NAMES, maximum_iterations=80, maximum_joint_delta_rad=.205)
    selection = encoded(tmp_path/"selection.json", training_audit())
    sensitivity=encoded(tmp_path/"sensitivity.json",dict(
        schema="task10b_phase2_saved_pose_correction_bound_sensitivity_v1",
        physical_joint_limits_relaxed=False,cases=[dict(candidate_id=f"jab_train_c4184_{t}_sequential_v2_probe",
            samples=[dict(engineering_cap_rad=.205,physical_joint_ranges_retained=True,nonlinear_single_pose_residual_m=.000001)])
            for t in ("0p600","0p750")]))
    entries = [dict(candidate_id=f"jab_train_c{g}_{t}_sequential_v3_probe", parent_candidate_id=f"jab_uniform_{t}", target_candidate_identity=f"target{g}") for g in (4184,4251) for t in (.6,.75)]
    plan = dict(schema=catalog.SCHEMA,family="jab",split="train",start_mode="family_guard",
        path_solver_version=catalog.VERSION,maximum_cells=4,output_byte_cap=catalog.MAX_OUTPUT_BYTES,
        maximum_child_seconds=240.,sources=catalog.source_bindings(),selected_command_groups=[4184,4251],
        selection_audit=selection,engineering_bound_review=sensitivity,retarget_settings=asdict(settings),cells=entries,
        reference_byte_cap=2*1024**3,resident_byte_cap=256*1024**2)
    recipe = dict(path_solver_version=catalog.VERSION,sequential_preference_weight=.002,
        solver_sources=plan["sources"],selection_audit=selection,engineering_bound_review=sensitivity)
    plan["recipe"] = encoded(tmp_path/"recipe.json",recipe)
    q=np.zeros((5,55));q[:,3]=1
    source=tmp_path/"source.npz"
    np.savez_compressed(source,t=np.arange(5)*.005,qpos=q,qvel=np.zeros((5,54)),head_center=np.zeros(3),active_hand_index=0,guard_hold_start=.02)
    model=tmp_path/"model.xml";model.write_text("synthetic placeholder; never parsed")
    resolved=[]
    for entry,(group,timing) in zip(entries,[(g,t) for g in (4184,4251) for t in (.6,.75)],strict=True):
        target=dict(command_id=group,original_replica_id=0,domain_stage="D2",head_target_w=[.1,0,1])
        parent=dict(requested_ttc_s=timing,reference_sha256=catalog.sha256_file(source),
            launch_tick=0,outward_end_tick=3,duration_s=.02,episode_duration_s=1.02,phase_denominator_s=1.02)
        resolved.append((entry,target,parent,source,model))
    monkeypatch.setattr(catalog.original,"resolve_cells",lambda common:(settings,resolved,{"recipe":Path(plan["recipe"]["path"])}))
    return tmp_path,plan,resolved


@pytest.mark.parametrize("change",["schema","sources","count","groups","wall","cap","iterations"])
def test_changed_scientific_or_resource_contract_refused(fixture,change):
    _,plan,_=fixture;plan=deepcopy(plan)
    if change=="schema":plan["schema"]=catalog.original.SCHEMA
    elif change=="sources":plan["sources"]={}
    elif change=="count":plan["cells"].pop()
    elif change=="groups":plan["selected_command_groups"]=[4184,1]
    elif change=="wall":plan["maximum_child_seconds"]=241
    elif change=="cap":plan["output_byte_cap"]+=1
    else:plan["retarget_settings"]["maximum_iterations"]=40
    with pytest.raises(ValueError):catalog.validate_plan(plan)


def test_exact_four_cells_resolve_without_models(fixture):
    _,plan,_=fixture;settings,cells,inputs=catalog.resolve_cells(plan)
    assert isinstance(settings,SequentialSettings) and len(cells)==4
    assert "selection_audit" in inputs
    assert "engineering_bound_review" in inputs and settings.base.maximum_joint_delta_rad==.205


@pytest.mark.parametrize("change",["physical_limits","wrong_cells","missing_0205","failed_pose"])
def test_unrelated_or_unsuccessful_sensitivity_cannot_authorize_new_bound(fixture,change):
    folder,plan,_=fixture;plan=deepcopy(plan)
    evidence=json.loads(Path(plan["engineering_bound_review"]["path"]).read_text())
    if change=="physical_limits":evidence["physical_joint_limits_relaxed"]=True
    elif change=="wrong_cells":evidence["cases"][0]["candidate_id"]="other"
    elif change=="missing_0205":evidence["cases"][0]["samples"][0]["engineering_cap_rad"]=.21
    else:evidence["cases"][0]["samples"][0]["nonlinear_single_pose_residual_m"]=.001
    plan["engineering_bound_review"]=encoded(folder/f"bad_{change}.json",evidence)
    with pytest.raises(ValueError):catalog.validate_plan(plan)


def test_source_aliases_require_the_same_exact_owned_resolved_files(fixture):
    folder,plan,_=fixture;plan=deepcopy(plan)
    for index,(name,record) in enumerate(plan["sources"].items()):
        alias=folder/f"source_alias_{index}.py"
        alias.symlink_to(record["path"])
        record["path"]=str(alias)
    settings,groups,_=catalog.validate_plan(plan)
    assert groups==(4184,4251) and settings.base.maximum_iterations==80
    name=next(iter(plan["sources"]))
    impostor=folder/"source_copy.py";impostor.write_bytes(Path(plan["sources"][name]["path"]).read_bytes())
    plan["sources"][name]["path"]=str(impostor)
    with pytest.raises(ValueError,match="exact owned"):
        catalog.validate_plan(plan)


def native_stubs(monkeypatch):
    state=dict(calls=0,reject_first=False,timeout=False)
    monkeypatch.setattr(catalog,"require_allocation",lambda wall:None)
    model=SimpleNamespace(nq=55,nv=54,nu=48,actuator_forcerange=np.tile([-1.,1.],(48,1)))
    monkeypatch.setattr(catalog,"mujoco",SimpleNamespace(MjModel=SimpleNamespace(from_xml_path=lambda _:model)))
    def retarget(model,source,target,**kwargs):
        state["calls"]+=1
        assert isinstance(kwargs["settings"],SequentialSettings)
        kwargs["checkpoint"]()
        if state["reject_first"] and state["calls"]==1:
            raise RetargetRejection("synthetic_unsolved",iterations=80,saturated_joint_names=["LeftArm_x"])
        source=deepcopy(source);source.update(time_s=source["t"],head_center=np.asarray(target),qacc=np.zeros((5,54)),torque=np.zeros((5,48)),glove_pos=np.zeros((5,2,3)))
        return source,dict(events=dict(launch_tick=0,launch_time_s=0.,outward_end_tick=3,outward_completion_time_s=.015))
    monkeypatch.setattr(catalog,"retarget_sequential",retarget)
    monkeypatch.setattr(catalog,"finalize_original_feedforward",lambda m,c,d:(c,d))
    monkeypatch.setattr(catalog,"contact_diagnostics",lambda *a:dict(timing_valid=True,inactive_hand_first_or_simultaneous=False,any_contact_before_launch=False))
    def screen(*args):
        if state["timeout"]:raise BoundedExecutionInterruption(dict(reason="synthetic_timeout",boundary="screen"))
    monkeypatch.setattr(catalog,"screen_chunks",screen)
    return state


def test_mocked_generation_retains_one_negative_and_remaining_rows(fixture,monkeypatch):
    folder,plan,_=fixture;state=native_stubs(monkeypatch);state["reject_first"]=True
    pp=folder/"plan.json";encoded(pp,plan)
    result=catalog.generate(pp,folder/"run",wall_seconds=240.)
    assert result["complete"] and result["executed_cells"]==4 and result["generated_cells"]==3
    assert result["admitted_candidate_cells"]==3 and result["path_solver_version"]==catalog.VERSION
    failure=result["cells"][0]
    assert failure["details"]["iterations"]==80 and failure["physical_qualification"] is False
    assert catalog.sha256_file(failure["generation_failure_path"])==failure["reference_sha256"]
    assert all(row["recipe_sha256"]==plan["recipe"]["sha256"] for row in result["cells"])
    assert state["calls"]==4


def test_mocked_timeout_keeps_partial_reference_and_three_unattempted(fixture,monkeypatch):
    folder,plan,_=fixture;state=native_stubs(monkeypatch);state["timeout"]=True
    pp=folder/"plan.json";encoded(pp,plan)
    result=catalog.generate(pp,folder/"run",wall_seconds=240.)
    assert not result["complete"] and len(result["unattempted_cells"])==3
    assert result["cells"][0]["generation_disposition"]=="timeout"
    assert Path(result["cells"][0]["retained_partial_reference_path"]).exists()


def test_missing_allocation_prevents_model_or_output(monkeypatch,tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(RuntimeError,match="counted Slurm"):
        catalog.generate(tmp_path/"no_plan",tmp_path/"run",wall_seconds=240.)
    assert not (tmp_path/"run").exists()


@pytest.mark.parametrize("wall",[True,0,59,241,float("inf"),float("nan")])
def test_invalid_wall_fails_before_runtime(wall):
    with pytest.raises(ValueError):catalog.require_allocation(wall)


def test_output_cap_is_checked_at_writes_not_unchanged_ik_steps(tmp_path):
    events=[];deadline=SimpleNamespace(checkpoint=lambda *a,**kw:events.append(a[0]))
    guard=catalog.OutputBoundedDeadline(deadline,tmp_path,catalog.OUTPUT_RESERVE+1)
    (tmp_path/"evidence").write_bytes(b"xx")
    guard.checkpoint("target_IK_frame")
    with pytest.raises(ValueError,match="output limit"):
        guard.checkpoint("target_screen_chunk")
    assert events==["target_IK_frame","target_screen_chunk"]
