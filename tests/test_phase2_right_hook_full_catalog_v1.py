"""Finite RH cache selection and evidence; only saved JSON/header operations."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from human_punching import phase2_right_hook_full_catalog_v1 as full


ROOT=Path(__file__).resolve().parents[1]
def load(name):return json.loads((ROOT/"outputs/task10b"/name).read_text())


def test_actual_proposal_preserves_exact64train24validation_and_interpolation_role():
    proposal=load("p2rhcacheproposal01/proposal.json")
    train=full.expected_cells(proposal,"train");validation=full.expected_cells(proposal,"validation")
    assert len(train)==64 and len(validation)==24
    assert len({r["target_candidate_identity"] for r in train})==32
    assert len({r["target_candidate_identity"] for r in validation})==8
    assert all("interior" not in r["parent_candidate_id"] for r in train)
    assert sum(r["parent_candidate_id"]==full.interior.ENTRY["candidate_id"] for r in validation)==8
    assert not ({r["target_candidate_identity"] for r in train}&{r["target_candidate_identity"] for r in validation})


@pytest.mark.parametrize("change",["drop","duplicate","order","timing","train_interior","calibration","group_overlap"])
def test_no_resampling_shrink_or_split_conflict(change):
    proposal=load("p2rhcacheproposal01/proposal.json")
    if change=="drop":proposal["cells"].pop(0)
    elif change=="duplicate":proposal["cells"][1]=deepcopy(proposal["cells"][0])
    elif change=="order":proposal["cells"][0],proposal["cells"][1]=proposal["cells"][1],proposal["cells"][0]
    elif change=="timing":proposal["cells"][0]["requested_ttc_s"]=1.1
    elif change=="train_interior":proposal["interior_used_for_training"]=True
    elif change=="calibration":proposal["groups"]["train"][0]=3307
    else:proposal["groups"]["validation"][0]=proposal["groups"]["train"][0]
    with pytest.raises(ValueError):full.expected_cells(proposal,"train")


@pytest.fixture
def evidence(monkeypatch):
    p=load("p2rhinteriorprep01/plan.json")
    old_parent=json.loads(Path(p["parent_catalog"]["path"]).read_text())
    parent=dict(schema=full.original.CATALOG_SCHEMA,generator_version=full.interior.VERSION,complete=True,
        exception=None,plan_sha256=full.sha256_file(ROOT/"outputs/task10b/p2rhinteriorprep01/plan.json"),
        planned_cells=23,executed_new_cells=1,cells=deepcopy(old_parent["cells"])+[
            dict(**full.interior.ENTRY,kinematic_admission=True,generation_disposition="generated",generator_version=full.interior.VERSION)])
    names={"timing_selection":"p2allfamilytargetprep01/selection.json","cache_proposal":"p2rhcacheproposal01/proposal.json",
        "probe_plan":"p2familytargetprep01/right_hook_plan.json","probe_audit":"p2familytargetaudit01r/audit.json",
        "probe_manifest":"p2ft02r/run/manifest.json","probe_packet":"p2ft02r/packet.json",
        "probe_execution":"p2ft02r/execution.json","probe_accounting":"p2ft02r/terminal_accounting.json",
        "interior_plan":"p2rhinteriorprep01/plan.json"}
    plan={k:full.binding(ROOT/"outputs/task10b"/v) for k,v in names.items()}
    synthetic={}
    def add(key,data):
        plan[key]=dict(path="synthetic:"+key,sha256=(str(len(synthetic)+1)*64)[:64]);synthetic[plan[key]["path"]]=data
    add("parent_catalog",parent)
    add("interior_packet",dict(schema="task10b_phase2_right_hook_interior_packet_v1",interior_cells=1,interior_ttc_s=1.1))
    add("interior_execution",dict(status="executed",returncode=0,job_id="123",packet_sha256=plan["interior_packet"]["sha256"]))
    add("interior_accounting",dict(state="COMPLETED",exit_code="0:0",job_id="123",packet_sha256=plan["interior_packet"]["sha256"]))
    add("interior_audit",dict(schema="task10b_phase2_rh_mixed_interior_saved_audit_v1",passed=True,physics_steps=0,
        reference_generation=False,packet_sha256=plan["interior_packet"]["sha256"],
        manifest_sha256=plan["parent_catalog"]["sha256"],plan_sha256=plan["interior_plan"]["sha256"]))
    original=full.read
    monkeypatch.setattr(full,"read",lambda item:synthetic[item["path"]] if item["path"] in synthetic else original(item))
    return plan,synthetic,parent


def test_completed_probe_does_not_require_its_frozen_actor_to_have_passed_all32(evidence):
    plan,_,_=evidence
    prior,proposal=full.validate_evidence(plan)
    assert prior["family"]=="right_hook" and prior["retarget_settings"]["maximum_joint_delta_rad"]==.205
    assert proposal["frozen_development_control"]["numeric_cycles"]==8
    assert proposal["frozen_development_control"]["attempts"]==32


@pytest.mark.parametrize("change",["changed_ancestor","missing_parent","wrong_interior","trained_interior","rejected_interior",
    "audit_fail","audit_wrong_packet","generation_job_failed","accounting_wrong_job"])
def test_interior_evidence_cannot_gain_admission_or_rewrite_ancestors(evidence,change):
    plan,data,parent=evidence
    if change=="changed_ancestor":parent["cells"][0]["requested_ttc_s"]=.1
    elif change=="missing_parent":parent["cells"].pop(0)
    elif change=="wrong_interior":parent["cells"][-1]["requested_ttc_s"]=1.09
    elif change=="trained_interior":parent["cells"][-1]["training_eligible"]=True
    elif change=="rejected_interior":parent["cells"][-1]["kinematic_admission"]=False
    elif change=="audit_fail":data[plan["interior_audit"]["path"]]["passed"]=False
    elif change=="audit_wrong_packet":data[plan["interior_audit"]["path"]]["packet_sha256"]="0"*64
    elif change=="generation_job_failed":data[plan["interior_execution"]["path"]]["returncode"]=1
    else:data[plan["interior_accounting"]["path"]]["job_id"]="456"
    with pytest.raises(ValueError):full.validate_evidence(plan)


def test_exact_finite_split_budgets_and_unchanged_solver_private_composition():
    assert full.LIMITS=={"train":(64,420,360,300),"validation":(24,240,180,120)}
    before=full.engine_source.resolve_cells;engine=full._engine()
    assert engine is not full.engine_source and full.engine_source.resolve_cells is before
    assert engine.retarget_sequential.__code__.co_code==full.engine_source.retarget_sequential.__code__.co_code
    assert full.measured_probe.family_settings("right_hook").base.maximum_joint_delta_rad==.205


@pytest.mark.parametrize("split",["test","train_and_validation",None])
def test_final_test_or_combined_split_refused_before_io(split):
    with pytest.raises(ValueError,match="Only RH"):full.validate_plan(dict(split=split))


def test_late_child_budget_is_not_reduced_to_fit(monkeypatch,tmp_path):
    path=tmp_path/"plan.json";path.write_text(json.dumps(dict(maximum_child_seconds=300)))
    monkeypatch.setattr(full,"binding",lambda p:dict(path=str(p),sha256="x"))
    monkeypatch.setattr(full,"own",lambda b:Path(b["path"]))
    monkeypatch.setattr(full,"resolve_cells",lambda p:None)
    with pytest.raises(ValueError,match="Exact finite"):
        full.generate(path,tmp_path/"output",wall_seconds=299)
    assert not (tmp_path/"output").exists()
