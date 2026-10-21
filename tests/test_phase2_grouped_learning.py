"""Synthetic train-split, bounded-input and reset tests; no native execution."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from human_punching.phase2_domain import fingerprint
from human_punching.phase2_grouped_learning import (
    GroupedTrainingDeclaration, GroupedSpecialistEnv, select_training_rows,
    preflight_training_references, joint_only_rehearsal_noise,
    validate_validation_descriptor, validate_callback_panel, train_grouped,
    grouped_stopping_rules, load_configuration, execute_configuration,
)
from human_punching.phase2_learning import NativeSpecialistEnv
from human_punching.phase2_learning import PilotBudget, validation_score
from human_punching.wbc.provenance import sha256_file
from human_punching.wbc.splits import assign_trajectory_groups


def fixture():
    assignment=assign_trajectory_groups(range(1000))
    train=tuple(k for k,v in assignment.items() if v=="train")[:32]
    validation=tuple(k for k,v in assignment.items() if v=="validation")[:8]
    geometry=[];groups=[];rows=[];placements=[]
    for split,replica,commands in (("train",0,train),("validation",1,validation)):
        for group in commands:
            snapshot=dict(command_id=group,replica_id=replica,slot_id=group,payload_sha256=fingerprint(group))
            groups.append(dict(split=split,snapshot=snapshot))
            target=dict(source_domain_sha256="a"*64,command_id=group,split=split,original_replica_id=replica,
                snapshot_slot_id=group,payload_sha256=fingerprint(group),head_target_w=[group*.001,0.,1.])
            target["candidate_identity"]=fingerprint(target);geometry.append(target)
            if split=="train":placements.append((group,target["candidate_identity"]))
            for timing in (.6,.675,.75):
                rows.append(dict(candidate_id=f"synthetic_{group}_{timing}",family="jab",kind="expanded_isolated",
                    command_id=group,split=split,original_replica_id=replica,requested_ttc_s=timing,
                    recipe_sha256="b"*64,target_candidate_identity=target["candidate_identity"],target_w=target["head_target_w"],
                    kinematic_admission=True,launch_tick=0,outward_end_tick=4,duration_s=.04,
                    episode_duration_s=1.04,phase_denominator_s=1.04))
    declaration=GroupedTrainingDeclaration("jab",21021,train,(.6,.75),.675,tuple(placements),
        "b"*64,"a"*64,"c"*64,"d"*64,"e"*64,"f"*64)
    domain=dict(schema="Task10BPhase2Domain/v1",selected_command_groups=groups)
    targets=dict(schema="Task10BPhase2RegisteredTargets/v1",geometry_candidates=geometry)
    catalog=dict(schema="task10b_phase2_reference_catalog_v1",cells=rows)
    return declaration,catalog,domain,targets


def validation_fixture(declaration,domain):
    groups=[r["snapshot"]["command_id"] for r in domain["selected_command_groups"] if r["split"]=="validation"]
    trials=[dict(trial_id=f"synthetic_validation_{g}_{t}_{s}",command_id=g,requested_ttc_s=t,start_index=s,
        numeric_complete_cycle=False,disposition="unattempted") for g in groups for t in (.6,.675,.75) for s in range(4)]
    descriptor=dict(schema="task10b_phase2_grouped_validation_v1",family="jab",seed=21021,split="validation",
        command_groups=groups,trained_timings_s=[.6,.75],interpolation_timing_s=.675,expected_attempts=96,
        recipe_sha256=declaration.recipe_sha256,domain_sha256=declaration.domain_sha256,
        target_geometry_sha256=declaration.target_geometry_sha256,predicate_sha256=declaration.predicate_sha256,
        declared_mask_sha256=declaration.declared_mask_sha256,trial_ids=[r["trial_id"] for r in trials])
    panel=dict(status="failed",hard_faults=[],trials=trials,grouped_validation_identity=fingerprint(descriptor))
    return descriptor,panel


def test_filter_retains_exactly_train_endpoints_without_opening_held_out_references():
    declaration,catalog,domain,targets=fixture()
    rows,receipt=select_training_rows(declaration,catalog,domain,targets)
    assert len(rows)==64 and {r["split"] for r in rows}=={"train"}
    assert {r["requested_ttc_s"] for r in rows}=={.6,.75}
    assert len(receipt["excluded_rows"])==56
    assert not receipt["interpolation_used_for_training"]
    rows[0]["target_w"][0]=999
    assert catalog["cells"][0]["target_w"][0]!=999


def test_group_declaration_forbids_calibration_held_out_and_unbounded_memory():
    declaration,_,domain,_=fixture()
    other=next(r["snapshot"]["command_id"] for r in domain["selected_command_groups"] if r["split"]=="validation")
    for kwargs in ({"command_groups":(3307,)+declaration.command_groups[1:]},
                   {"command_groups":(other,)+declaration.command_groups[1:]},
                   {"trained_timings_s":(.6,.675)}, {"resident_reference_bytes":2*1024**3},
                   {"seed":0},{"target_candidate_identities":declaration.target_candidate_identities[:-1]}):
        with pytest.raises(ValueError):replace(declaration,**kwargs)


@pytest.mark.parametrize("change",[{"original_replica_id":1},{"recipe_sha256":"0"*64},
    {"target_candidate_identity":"0"*64},{"target_w":[9,9,9]},{"kinematic_admission":False},
    {"launch_tick":1},{"preparation_complete_tick":4},{"start_mode":"shared_jab_saved_state"},
    {"phase_denominator_s":1.2}])
def test_selected_reference_binding_and_semantics_refuse_changes(change):
    declaration,catalog,domain,targets=fixture();catalog["cells"][0].update(change)
    with pytest.raises(ValueError):select_training_rows(declaration,catalog,domain,targets)


def test_missing_duplicate_or_conflicting_training_cells_are_not_resampled():
    declaration,catalog,domain,targets=fixture()
    with pytest.raises(ValueError,match="Exactly64"):
        select_training_rows(declaration,{**catalog,"cells":catalog["cells"][1:]},domain,targets)
    duplicate={**catalog["cells"][0],"candidate_id":"different_name_same_request"}
    with pytest.raises(ValueError,match="Multiple references"):
        select_training_rows(declaration,{**catalog,"cells":catalog["cells"]+[duplicate]},domain,targets)
    targets["geometry_candidates"][0]["snapshot_slot_id"]+=1
    with pytest.raises(ValueError,match="binding"):
        select_training_rows(declaration,catalog,domain,targets)


def write_reference(tmp_path,row):
    path=tmp_path/(row["candidate_id"]+".npz");n=9;t=np.arange(n)*.005
    q=np.zeros((n,55));q[:,3]=1.
    np.savez(path,t=t,qpos=q,qvel=np.zeros((n,54)),qacc=np.zeros((n,54)),torque=np.zeros((n,48)),
        head_center=np.array(row["target_w"]),active_hand_index=0)
    return {**row,"reference_path":str(path),"reference_sha256":sha256_file(path),
        "legacy_runtime_path":str(path),"legacy_runtime_sha256":sha256_file(path)}


def test_preflight_reference_bounds_hashes_and_metadata(tmp_path):
    declaration,catalog,domain,targets=fixture()
    rows,_=select_training_rows(declaration,catalog,domain,targets)
    one=write_reference(tmp_path,rows[0]);receipt=preflight_training_references([one])
    assert receipt["unique_archives"]==1 and not receipt["reference_payloads_loaded"]
    assert receipt["metadata_arrays_loaded"] and receipt["archive_bytes"]>0
    with pytest.raises(ValueError,match="allowance"):
        preflight_training_references([one],resident_limit=100)
    with pytest.raises(ValueError):
        preflight_training_references([{**one,"reference_sha256":"0"*64}])
    with pytest.raises(ValueError,match="target"):
        preflight_training_references([{**one,"target_w":[9.,0.,1.]}])
    # Native ReferenceLibrary copies repeated rows; resident allowance must too.
    doubled=preflight_training_references([one,{**one,"candidate_id":"second_row_same_payload"}])
    assert doubled["array_bytes"]==receipt["array_bytes"]
    assert doubled["conservative_resident_allowance_bytes"]==2*receipt["conservative_resident_allowance_bytes"]


def test_joint_only_rehearsal_noise_preserves_root_and_is_seeded_clipped():
    q=torch.zeros((128,55));q[:,3]=1.;v=torch.ones((128,54))*.01
    qadr=np.arange(7,55);vadr=np.arange(6,54);low=torch.ones(48)*-.01;high=-low
    torch.manual_seed(21021);actual_q,actual_v=joint_only_rehearsal_noise(q,v,qadr,vadr,low,high)
    assert torch.equal(actual_q[:,:7],q[:,:7]) and torch.equal(actual_v[:,:6],v[:,:6])
    assert not torch.equal(actual_q[:,qadr],q[:,qadr]) and not torch.equal(actual_v[:,vadr],v[:,vadr])
    assert (actual_q[:,qadr].abs()<=.01).all() and ((actual_v-v).abs()<=.06000001).all()
    assert not q[:,7:].any() and torch.equal(v,torch.ones_like(v)*.01)
    torch.manual_seed(21021);q2,v2=joint_only_rehearsal_noise(q,v,qadr,vadr,low,high)
    assert torch.equal(actual_q,q2) and torch.equal(actual_v,v2)
    with pytest.raises(ValueError):joint_only_rehearsal_noise(q,v,qadr,np.arange(48),low,high)


def test_subclass_noise_does_not_add_a_second_reset_or_change_other_worlds(monkeypatch):
    env=GroupedSpecialistEnv.__new__(GroupedSpecialistEnv);env.n=4;env.device="cpu"
    env.qadr=np.arange(7,55);env.vadr=np.arange(6,54)
    env.joint_low=-torch.ones(48);env.joint_high=torch.ones(48)
    q=torch.zeros((4,55));q[:,3]=1.;v=torch.zeros((4,54))
    calls=[];forward=[]
    env.sim=SimpleNamespace(data=SimpleNamespace(qpos=q,qvel=v),forward=lambda:forward.append(True))
    env.support_contact=lambda:torch.ones((4,2),dtype=torch.bool);env.observe=lambda:torch.cat((q,v),1)
    monkeypatch.setattr(NativeSpecialistEnv,"reset",lambda self,ids,**kwargs:calls.append((ids,kwargs)))
    env.reset(torch.tensor([1,3]),random_phase=False)
    assert len(calls)==1 and calls[0][1]["perturb"] is False and len(forward)==1
    assert not q[[0,2],7:].any() and not v[[0,2]].any()
    assert not v[:,:6].any() and not q[:,0:3].any()
    with pytest.raises(ValueError):env.reset(initial_qpos=q,initial_qvel=v,perturb=True)


def test_validation_callback_has_exact_held_out_panel_and_preserves_failed_attempts():
    declaration,_,domain,_=fixture();descriptor,panel=validation_fixture(declaration,domain)
    assert validate_validation_descriptor(declaration,descriptor,domain)==fingerprint(descriptor)
    assert validate_callback_panel(panel,descriptor) is panel and panel["status"]=="failed"
    with pytest.raises(ValueError):validate_validation_descriptor(declaration,{**descriptor,"split":"test"},domain)
    with pytest.raises(ValueError):validate_validation_descriptor(declaration,{**descriptor,"seed":21022},domain)
    with pytest.raises(ValueError):validate_callback_panel({**panel,"trials":panel["trials"][:-1]},descriptor)
    changed=[dict(r) for r in panel["trials"]];changed[0]["command_id"]=declaration.command_groups[0]
    with pytest.raises(ValueError):validate_callback_panel({**panel,"trials":changed},descriptor)


def test_no_grouped_training_without_counted_allocation(tmp_path,monkeypatch):
    declaration,_,domain,_=fixture();descriptor,_=validation_fixture(declaration,domain)
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(RuntimeError,match="counted Slurm"):
        train_grouped(declaration,catalog_path="unused",catalog_sha256="0"*64,domain_path="unused",
            target_geometry_path="unused",predicate_path="unused",mask_path="unused",recipe_path="unused",
            validation_descriptor=descriptor,validation_callback=None,release="unused",output=tmp_path/"not_created")
    assert not (tmp_path/"not_created").exists()


def test_stopping_rules_expose_initial_fall_constraint_and_finite_caps():
    rules=grouped_stopping_rules(PilotBudget())
    assert rules["validation_updates"]==[0,8,16,24,32]
    assert rules["initial_validation_hard_failure"]=="stop_before_learning"
    assert rules["fresh_transition_cap"]==262144
    assert rules["training_fall"]=="original_done_reset_no_commanded_cycle_credit"
    assert rules["training_nonfinite_root_assistance_effort_cap"]=="hard_stop"
    assert not rules["conditional_replication_budget_unlocked"]
    assert validation_score(dict(status="executed",hard_faults=[{"faults":{"fall":[0]}}],trials=[{}])) is None


def test_configuration_requires_immutable_binding_and_explicit_stopping_rules(tmp_path):
    import json
    from dataclasses import asdict
    declaration,catalog,domain,targets=fixture()
    items={}
    payloads={"catalog":catalog,"domain":domain,"target_geometry":targets,"predicate":{},"mask":{},"recipe":{}}
    for name,payload in payloads.items():
        path=tmp_path/(name+".json");path.write_text(json.dumps(payload))
        items[name]=dict(path=str(path),sha256=sha256_file(path))
    declaration=replace(declaration,**{attribute:items[name]["sha256"] for name,attribute in (
        ("domain","domain_sha256"),("target_geometry","target_geometry_sha256"),("predicate","predicate_sha256"),
        ("mask","declared_mask_sha256"),("recipe","recipe_sha256"))})
    for name,payload in (("declaration",asdict(declaration)),("validation_configuration",{})):
        path=tmp_path/(name+".json");path.write_text(json.dumps(payload));items[name]=dict(path=str(path),sha256=sha256_file(path))
    budget=PilotBudget(maximum_updates=8)
    config=dict(schema="task10b_phase2_grouped_training_configuration_v1",**items,
        budget=asdict(budget),stopping_rules=grouped_stopping_rules(budget),release=str(tmp_path))
    path=tmp_path/"configuration.json";path.write_text(json.dumps(config))
    loaded,_,actual,_,_=load_configuration(path,sha256_file(path))
    assert loaded.identity==declaration.identity and actual.transition_cap==65536
    config["stopping_rules"]["initial_validation_hard_failure"]="ignore"
    path2=tmp_path/"bad.json";path2.write_text(json.dumps(config))
    with pytest.raises(ValueError,match="stopping"):
        load_configuration(path2,sha256_file(path2))
    with pytest.raises(ValueError,match="changed"):
        load_configuration(path,"0"*64)
    with pytest.raises(ValueError):
        replace(declaration,resident_reference_bytes=2*1024**3)


def test_configuration_execution_never_runs_without_counted_allocation(tmp_path,monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(RuntimeError,match="counted Slurm"):
        execute_configuration("unused","0"*64,tmp_path/"not_created")
    assert not (tmp_path/"not_created").exists()
