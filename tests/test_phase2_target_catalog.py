"""Exact catalog ancestry, split and interrupted-generation accounting."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from human_punching import phase2_target_catalog as catalog
from human_punching.phase2_domain import fingerprint, assign_trajectory_groups
from human_punching.phase2_deadline import BoundedExecutionInterruption
from human_punching.wbc.provenance import sha256_file


def encoded(path,data):
    path.write_text(json.dumps(data))
    return {"path":str(path),"sha256":sha256_file(path)}


@pytest.fixture
def example(tmp_path):
    group=next(g for g in range(100) if assign_trajectory_groups([g])[g]=="train")
    snapshot=dict(command_id=group,replica_id=0,slot_id=11,payload_sha256="a"*64)
    domain=encoded(tmp_path/"domain.json",dict(schema="Task10BPhase2Domain/v1",
        selected_command_groups=[dict(split="train",snapshot=snapshot)]))
    target=dict(command_id=group,split="train",original_replica_id=0,snapshot_slot_id=11,
        payload_sha256="a"*64,source_domain_sha256=domain["sha256"],head_target_w=[1.,0.,1.],domain_stage="D1")
    target["candidate_identity"]=fingerprint(target)
    geometry=encoded(tmp_path/"targets.json",dict(schema="Task10BPhase2RegisteredTargets/v1",geometry_candidates=[target]))
    q=np.zeros((5,55));q[:,3]=1.;q[1:,7]=.01
    source=tmp_path/"source.npz"
    np.savez_compressed(source,t=np.arange(5)*.005,qpos=q,qvel=np.zeros((5,54)),
        head_center=np.array([1.,0.,1.]),active_hand_index=0,guard_hold_start=.02)
    model=tmp_path/"model.xml";model.write_text("fixture model; native constructor mocked")
    parent=dict(candidate_id="jab_uniform_0p600",family="jab",requested_ttc_s=.6,
        kinematic_admission=True,reference_path=str(source),reference_sha256=sha256_file(source),
        source_model_path=str(model),source_model_sha256=sha256_file(model),launch_tick=0,outward_end_tick=3)
    parent.update(duration_s=.02,episode_duration_s=1.02,phase_denominator_s=1.02)
    parents=encoded(tmp_path/"parents.json",dict(schema=catalog.CATALOG_SCHEMA,cells=[parent]))
    settings=dict(active_side="Left",optimized_joint_names=["LeftShoulder"])
    recipe=encoded(tmp_path/"recipe.json",dict(schema="task10b_phase2_target_recipe_v1",family="jab",
        retarget_settings=settings,domain_sha256=domain["sha256"],target_geometry_sha256=geometry["sha256"],
        parent_catalog_sha256=parents["sha256"],start_mode="family_guard",physics_dt_s=.005,actor_dt_s=.020))
    plan=dict(schema=catalog.SCHEMA,family="jab",split="train",start_mode="family_guard",
        maximum_cells=2,reference_byte_cap=2**20,resident_byte_cap=2**20,retarget_settings=settings,
        domain=domain,target_geometry=geometry,parent_catalog=parents,recipe=recipe,
        body_metadata=encoded(tmp_path/"body.json",dict(kp=[1.]*48,kd=[1.]*48)),
        cells=[dict(candidate_id="jab_group_target_0600",parent_candidate_id=parent["candidate_id"],
            target_candidate_identity=target["candidate_identity"])])
    return tmp_path,plan,parent,target


def test_exact_plan_resolves_without_loading_native_model(example):
    _,plan,_,_=example
    settings,cells,paths=catalog.resolve_cells(plan)
    assert settings.active_side=="Left" and len(cells)==1 and set(paths)=={
        "domain","target_geometry","parent_catalog","body_metadata","recipe"}


@pytest.mark.parametrize("change",["split","side","duplicate","traversal","extra","cap"])
def test_unsupported_plan_rejected_before_output(example,change):
    _,plan,_,_=example
    if change=="split":plan["split"]="test"
    elif change=="side":plan["retarget_settings"]["active_side"]="Right"
    elif change=="duplicate":plan["cells"]*=2
    elif change=="traversal":plan["cells"][0]["candidate_id"]="../old"
    elif change=="extra":plan["cells"][0]["family"]="cross"
    else:plan["reference_byte_cap"]=catalog.BYTE_CAP+1
    with pytest.raises(ValueError):catalog.resolve_cells(plan)


def test_reference_hash_change_is_not_silently_substituted(example):
    _,plan,parent,_=example
    Path(parent["reference_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError,match="hash"):catalog.resolve_cells(plan)


def test_target_metadata_tamper_cannot_keep_old_identity(example):
    path,plan,_,_=example
    data=json.loads(Path(plan["target_geometry"]["path"]).read_text())
    data["geometry_candidates"][0]["head_target_w"][0]+=.05
    plan["target_geometry"]=encoded(path/"targets_changed.json",data)
    recipe=json.loads(Path(plan["recipe"]["path"]).read_text())
    recipe["target_geometry_sha256"]=plan["target_geometry"]["sha256"]
    plan["recipe"]=encoded(path/"recipe_changed.json",recipe)
    with pytest.raises(ValueError,match="binding"):catalog.resolve_cells(plan)


def native_stubs(monkeypatch):
    fake=SimpleNamespace(nq=55,nv=54,nu=48,actuator_forcerange=np.tile([-10.,10.],(48,1)))
    monkeypatch.setattr(catalog,"mujoco",SimpleNamespace(MjModel=SimpleNamespace(from_xml_path=lambda _:fake)))
    def retarget(model,source,target,**kwargs):
        kwargs["checkpoint"]()
        source=copy.deepcopy(source);source.update(time_s=source["t"],head_center=np.array(target),
            qacc=np.zeros((5,54)),torque=np.zeros((5,48)),glove_pos=np.zeros((5,2,3)))
        return source,dict(events=dict(launch_tick=0,launch_time_s=0.,outward_end_tick=3,outward_completion_time_s=.015))
    monkeypatch.setattr(catalog,"retarget_upper_body",retarget)
    monkeypatch.setattr(catalog,"finalize_original_feedforward",lambda model,candidate,metadata:(candidate,metadata))
    monkeypatch.setattr(catalog,"contact_diagnostics",lambda *a:dict(timing_valid=True,
        inactive_hand_first_or_simultaneous=False,any_contact_before_launch=False))
    monkeypatch.setattr(catalog,"screen_chunks",lambda *a:None)


def test_generated_row_binds_recipe_and_original_group(example,monkeypatch):
    path,plan,_,target=example;native_stubs(monkeypatch)
    plan_path=path/"plan.json";encoded(plan_path,plan)
    result=catalog.generate(plan_path,path/"run",wall_seconds=30.)
    row=result["cells"][0]
    assert result["complete"] and result["recipe_sha256"]==row["recipe_sha256"]==plan["recipe"]["sha256"]
    assert row["command_id"]==target["command_id"] and row["original_replica_id"]==0
    assert row["kinematic_admission"] and not row["physical_qualification"]
    assert sha256_file(row["reference_path"])==row["reference_sha256"]
    with pytest.raises(FileExistsError):catalog.generate(plan_path,path/"run",wall_seconds=30.)


def test_mid_screen_timeout_preserves_candidate_and_unattempted_remainder(example,monkeypatch):
    path,plan,_,_=example;native_stubs(monkeypatch)
    # A second declared target row must remain unattempted; resolution is fixed
    # independently of the injected timeout to isolate serialization accounting.
    settings,cells,inputs=catalog.resolve_cells(plan)
    other=copy.deepcopy(cells[0]);other[0]["candidate_id"]="jab_other"
    monkeypatch.setattr(catalog,"resolve_cells",lambda _:(settings,[cells[0],other],inputs))
    plan["cells"].append(dict(other[0]))
    def timeout(*args):raise BoundedExecutionInterruption(dict(reason="serialization_reserve",boundary="screen"))
    monkeypatch.setattr(catalog,"screen_chunks",timeout)
    plan_path=path/"plan.json";encoded(plan_path,plan)
    result=catalog.generate(plan_path,path/"run",wall_seconds=30.)
    assert not result["complete"] and result["executed_cells"]==1 and len(result["unattempted_cells"])==1
    failed=result["cells"][0]
    assert failed["generation_disposition"]=="timeout" and not failed["kinematic_admission"]
    assert failed["reference_identity_kind"]=="generation_failure_receipt"
    assert "reference_path" not in failed and "legacy_runtime_path" not in failed
    assert sha256_file(failed["generation_failure_path"])==failed["reference_sha256"]
    assert Path(failed["retained_partial_reference_path"]).exists()
    assert result["reference_uncompressed_array_bytes"]>0


def test_invalid_wall_rejected_without_output_creation(example):
    path,plan,_,_=example;plan_path=path/"plan.json";encoded(plan_path,plan)
    with pytest.raises(ValueError):catalog.generate(plan_path,path/"run",wall_seconds=0.)
    assert not (path/"run").exists()


def test_nonfinite_candidate_is_preserved_as_failure_not_an_npz(example,monkeypatch):
    path,plan,_,_=example;native_stubs(monkeypatch)
    original=catalog.retarget_upper_body
    def corrupt(*args,**kwargs):
        candidate,metadata=original(*args,**kwargs)
        candidate["qvel"][1,0]=np.nan
        return candidate,metadata
    monkeypatch.setattr(catalog,"retarget_upper_body",corrupt)
    plan_path=path/"plan.json";encoded(plan_path,plan)
    result=catalog.generate(plan_path,path/"run",wall_seconds=30.)
    assert result["generated_cells"]==0 and result["admitted_candidate_cells"]==0
    row=result["cells"][0]
    assert row["reference_identity_kind"]=="generation_failure_receipt"
    assert not list((path/"run").rglob("reference.npz"))


def test_chunk_screen_keeps_every_frame_and_all_six_base_diagnostics(tmp_path,monkeypatch):
    calls=[];checkpoints=[]
    def screen(model,chunk,caps):
        calls.append(chunk["t"].copy());n=len(chunk["t"])
        summary=dict(schema="screen",frames=n,within_effort_caps_frames=n-1,over_effort_caps_frames=1,
            support_status_counts={"feasible":n},joint_limit_excursion_max_rad=float(chunk["t"][-1]),
            selected_self_penetration_max_m=0.,floor_penetration_max_m=0.,joint_velocity_max_rad_s=1.,
            joint_acceleration_max_rad_s2=2.,maximum_minimax_effort_ratio=None,
            maximum_base_force_residual_n=3.,maximum_base_moment_residual_nm=4.)
        return summary,dict(time_s=chunk["t"],root_wrench_residual=np.zeros((n,6)))
    monkeypatch.setattr(catalog,"screen_diagnostics",screen)
    deadline=SimpleNamespace(checkpoint=lambda *args,**kwargs:checkpoints.append(kwargs["progress"]["first_frame"]))
    data=dict(t=np.arange(5)*.005,qpos=np.zeros((5,55)),qvel=np.zeros((5,54)),qacc=np.zeros((5,54)))
    result=catalog.screen_chunks("unused",data,np.ones(48),tmp_path,deadline,chunk_frames=2)
    assert np.array_equal(np.concatenate(calls),data["t"]) and checkpoints==[0,2,4]
    assert result["frames"]==5 and result["support_status_counts"]=={"feasible":5}
    assert result["maximum_base_force_residual_n"]==3. and result["maximum_base_moment_residual_nm"]==4.
    assert len(list(tmp_path.glob("screen_*.npz")))==3
