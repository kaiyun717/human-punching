"""Exact RH64train/24validation caches with one mixed-clock parent recipe.

The unchanged .205 active-arm solver constructs candidates, not qualified
execution. Every requested group remains declared even if construction fails.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from . import phase2_target_catalog as original
from . import phase2_sequential_target_catalog_v3 as engine_source
from . import phase2_family_target_probe_v1 as measured_probe
from . import phase2_right_hook_interior_v1 as interior
from .phase2_target_path_reference_v3 import SequentialSettings, VERSION as SOLVER_VERSION
from .wbc.provenance import PROJECT_ROOT,sha256_file,writable_path,write_json

SCHEMA="task10b_phase2_right_hook_full_catalog_plan_v1"
ENTRY_VERSION="task10b_phase2_right_hook_full_catalog_v1"
LIMITS={"train":(64,420,360,300),"validation":(24,240,180,120)}
OUTPUT_CAPS={"train":256*1024**2,"validation":128*1024**2}
SOURCES=(*engine_source.SOURCES,"src/human_punching/phase2_right_hook_full_catalog_v1.py")
EVIDENCE=("timing_selection","cache_proposal","probe_plan","probe_audit","probe_manifest",
    "probe_packet","probe_execution","probe_accounting","interior_plan","interior_audit",
    "interior_packet","interior_execution","interior_accounting")
own=original.owned_input


def binding(path):
    p=Path(path).resolve(strict=True)
    if not p.is_relative_to(PROJECT_ROOT.resolve()) or not p.is_file():raise ValueError("Owned RH cache input required")
    return dict(path=str(p),sha256=sha256_file(p))


def read(item):return json.loads(own(item).read_text())
def source_bindings():return {n:binding(PROJECT_ROOT/n) for n in SOURCES}


def expected_cells(proposal,split):
    if (proposal.get("schema")!="task10b_phase2_right_hook_grouped_cache_proposal_v1"
            or split not in LIMITS or proposal.get("training_timings_s")!=[1.,1.2]
            or proposal.get("validation_timings_s")!=[1.,1.1,1.2]
            or proposal.get("interior_used_for_training") is not False):
        raise ValueError("Exact prespecified RH cache proposal and splits required")
    rows=[r for r in proposal["cells"] if r["split"]==split]
    groups=proposal["groups"][split];timings=[1.,1.2] if split=="train" else [1.,1.1,1.2]
    if (len(rows)!=LIMITS[split][0] or len(groups)!={"train":32,"validation":8}[split]
            or len(set(groups))!=len(groups) or 3307 in groups
            or set(proposal["groups"]["train"])&set(proposal["groups"]["validation"])
            or [(r["command_id"],r["requested_ttc_s"]) for r in rows]!=[(g,t) for g in groups for t in timings]
            or any(r.get("family")!="right_hook" or r.get("training_eligible")!=(split=="train") for r in rows)):
        raise ValueError("Exact64/24 ordered requests; no resampling, test groups or interior training")
    return [dict(candidate_id=r["candidate_id"],parent_candidate_id=r["source_candidate_id"],
        target_candidate_identity=r["target_candidate_identity"]) for r in rows]


def validate_evidence(plan):
    data={key:read(plan[key]) for key in EVIDENCE}
    selection=data["timing_selection"];proposal=data["cache_proposal"];prior=data["probe_plan"]
    measured_probe.validate_plan(prior)
    if (prior.get("family")!="right_hook" or prior.get("path_solver_version")!=SOLVER_VERSION
            or plan["timing_selection"]!=prior["timing_selection"]
            or proposal.get("bindings",{}).get("selection")!=plan["timing_selection"]
            or proposal.get("retarget_settings")!=prior["retarget_settings"]):
        raise ValueError("Unchanged reviewed RH.205 engineering probe and proposal required")
    audit,manifest=data["probe_audit"],data["probe_manifest"]
    packet,execution,accounting=(data[k] for k in ("probe_packet","probe_execution","probe_accounting"))
    if (audit.get("schema")!="task10b_phase2_family_saved_target_reference_independent_audit_v1"
            or audit.get("passed") is not True or audit.get("family")!="right_hook" or audit.get("physics_steps")!=0
            or audit.get("packet_sha256")!=plan["probe_packet"]["sha256"]
            or audit.get("manifest_sha256")!=plan["probe_manifest"]["sha256"]
            or manifest.get("plan_sha256")!=plan["probe_plan"]["sha256"]
            or manifest.get("path_solver_version")!=SOLVER_VERSION or manifest.get("complete") is not True
            or manifest.get("exception") is not None or manifest.get("unattempted_cells")!=[]
            or any(manifest.get(k)!=4 for k in ("planned_cells","executed_cells","generated_cells","admitted_candidate_cells"))
            or packet.get("schema")!="task10b_phase2_family_target_packet_v1" or packet.get("family")!="right_hook"
            or execution.get("status")!="executed" or execution.get("returncode")!=0
            or accounting.get("state")!="COMPLETED" or accounting.get("exit_code")!="0:0"
            or any(x.get("packet_sha256")!=plan["probe_packet"]["sha256"] for x in (execution,accounting))
            or len({audit.get("job_id"),execution.get("job_id"),accounting.get("job_id")})!=1):
        raise ValueError("Completed counted independently audited four-reference RH probe required")
    ids={r["candidate_id"] for r in prior["cells"]}
    if ({r.get("candidate_id") for r in manifest["cells"]}!=ids
            or {r.get("candidate_id") for r in audit["cells"]}!=ids
            or any(r.get("all_checks_passed") is not True for r in audit["cells"])):
        raise ValueError("All four measured probe identities must be retained")
    ip=data["interior_plan"];_,parent22,_=interior.validate_plan(ip)
    if ip["selection"]!=plan["timing_selection"] or ip["proposal"]!=plan["cache_proposal"]:
        raise ValueError("Actual interior must use this exact selected mixed-clock recipe")
    ia=data["interior_audit"];packet=data["interior_packet"];execution=data["interior_execution"];accounting=data["interior_accounting"]
    parent=read(plan["parent_catalog"])
    if (ia.get("schema")!="task10b_phase2_rh_mixed_interior_saved_audit_v1" or ia.get("passed") is not True
            or ia.get("physics_steps")!=0 or ia.get("reference_generation") is not False
            or ia.get("packet_sha256")!=plan["interior_packet"]["sha256"]
            or ia.get("manifest_sha256")!=plan["parent_catalog"]["sha256"]
            or ia.get("plan_sha256")!=plan["interior_plan"]["sha256"]
            or packet.get("schema")!="task10b_phase2_right_hook_interior_packet_v1"
            or packet.get("interior_cells")!=1 or packet.get("interior_ttc_s")!=1.1
            or execution.get("status")!="executed" or execution.get("returncode")!=0
            or accounting.get("state")!="COMPLETED" or accounting.get("exit_code")!="0:0"
            or any(x.get("packet_sha256")!=plan["interior_packet"]["sha256"] for x in (execution,accounting))
            or execution.get("job_id")!=accounting.get("job_id")
            or parent.get("generator_version")!=interior.VERSION or parent.get("complete") is not True
            or parent.get("exception") is not None or parent.get("plan_sha256")!=plan["interior_plan"]["sha256"]
            or parent.get("planned_cells")!=23 or parent.get("executed_new_cells")!=1
            or len(parent.get("cells",[]))!=23 or parent["cells"][:22]!=parent22["cells"]):
        raise ValueError("Actual counted independently audited RH1.10 parent and unchanged22 ancestors required")
    row=parent["cells"][-1]
    if (any(row.get(k)!=v for k,v in interior.ENTRY.items()) or row.get("kinematic_admission") is not True
            or row.get("generation_disposition")!="generated" or row.get("generator_version")!=interior.VERSION):
        raise ValueError("Exact admitted1.10 reference required; no fallback or relabel")
    if selection["families"]["right_hook"]["trained_timings_s"]!=[1.,1.2]:
        raise ValueError("Fixed two trained bins changed")
    return prior,proposal


def validate_plan(plan):
    split=plan.get("split")
    if split not in LIMITS:raise ValueError("Only RH training or validation construction supported")
    count,_,_,child=LIMITS[split]
    if (plan.get("schema")!=SCHEMA or plan.get("entry_version")!=ENTRY_VERSION
            or plan.get("family")!="right_hook" or plan.get("path_solver_version")!=SOLVER_VERSION
            or plan.get("maximum_cells")!=count or len(plan.get("cells",[]))!=count
            or plan.get("maximum_child_seconds")!=child or plan.get("output_byte_cap")!=OUTPUT_CAPS[split]
            or plan.get("physical_qualification") is not False
            or plan.get("native_frozen_control_required_before_learning") is not True
            or set(plan.get("sources",{}))!=set(SOURCES)):
        raise ValueError("Exact RH64/24 finite .205 candidate-only contract required")
    for name,item in plan["sources"].items():
        if own(item)!=(PROJECT_ROOT/name).resolve(strict=True):raise ValueError("Exact owned source module required")
    prior,proposal=validate_evidence(plan)
    for key in ("domain","target_geometry","body_metadata","retarget_settings"):
        if plan[key]!=prior[key]:raise ValueError("Original measured targets/physical settings cannot change")
    if plan["cells"]!=expected_cells(proposal,split):raise ValueError("No shrink, resample or request substitution")
    common=deepcopy(plan);common["schema"]=original.SCHEMA;original.validate_plan(common)
    return measured_probe.family_settings("right_hook"),common


def resolve_cells(plan):
    settings,common=validate_plan(plan);_,cells,paths=original.resolve_cells(common)
    recipe=read(plan["recipe"])
    if (recipe.get("full_catalog_entry_version")!=ENTRY_VERSION or recipe.get("path_solver_version")!=SOLVER_VERSION
            or recipe.get("solver_sources")!=plan["sources"] or any(recipe.get(k)!=plan[k] for k in EVIDENCE)
            or recipe.get("sequential_preference_weight")!=.002 or recipe.get("interior_training_eligible") is not False
            or recipe.get("native_frozen_control_required_before_learning") is not True):
        raise ValueError("One common exact recipe must bind both splits and unchanged solver ancestry")
    proposal=read(plan["cache_proposal"]);domain=read(plan["domain"])
    groups=[g["snapshot"]["command_id"] for g in domain["selected_command_groups"] if g["split"]==plan["split"]]
    timings=[1.,1.2] if plan["split"]=="train" else [1.,1.1,1.2]
    if groups!=proposal["groups"][plan["split"]] or [(t["command_id"],p["requested_ttc_s"]) for _,t,p,_,_ in cells]!=[(g,t) for g in groups for t in timings]:
        raise ValueError("Exact32/8 deterministic source groups and timings required")
    declared={r["candidate_id"]:r for r in proposal["cells"]}
    for entry,target,parent,_,_ in cells:
        prior=declared[entry["candidate_id"]]
        if (target["placement"]!={"forward_m":0.,"left_m":0.,"yaw_rad":0.}
                or target["head_target_w"]!=prior["target_w"] or target["payload_sha256"]!=prior["snapshot_payload_sha256"]):
            raise ValueError("Exact registered static target and original snapshot payload required")
        if parent["requested_ttc_s"]==1.1 and (plan["split"]!="validation" or parent.get("training_eligible") is not False):
            raise ValueError("Prespecified mixed interior stays validation only")
    paths.update({key:own(plan[key]) for key in EVIDENCE})
    return settings,cells,paths


def files(plan_path):
    """Exact native semantic edges, without traversing documentary XML paths."""
    from . import phase2_target_inputs as target_inputs
    saved=target_inputs.resolve_cells
    try:
        target_inputs.resolve_cells=resolve_cells;closure=target_inputs.files(plan_path)
    finally:target_inputs.resolve_cells=saved
    plan=json.loads(Path(plan_path).read_text());prior=read(plan["probe_plan"])
    _,_,probe_paths=measured_probe.resolve_cells(prior)
    for p in probe_paths.values():closure[str(p.relative_to(PROJECT_ROOT))]=sha256_file(p)
    for item in (*plan["sources"].values(),*prior["sources"].values()):
        p=own(item);closure[str(p.relative_to(PROJECT_ROOT))]=item["sha256"]
    closure.update(interior.files(own(plan["interior_plan"])))
    for name,digest in closure.items():
        if sha256_file(PROJECT_ROOT/name)!=digest:raise ValueError("RH cache input changed during closure")
    return dict(sorted(closure.items()))


def _engine():
    spec=importlib.util.spec_from_file_location("human_punching._right_hook_full_engine_v1",Path(engine_source.__file__))
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine);return engine


def generate(plan_path,output,*,wall_seconds):
    plan_path=own(binding(plan_path));plan=json.loads(plan_path.read_text());resolve_cells(plan)
    if type(wall_seconds) not in (int,float) or wall_seconds!=plan["maximum_child_seconds"]:
        raise ValueError("Exact finite split child required; no late truncation")
    engine=_engine();engine.resolve_cells=resolve_cells
    engine.require_allocation=lambda value:engine_source.require_allocation(min(value,engine_source.MAX_CHILD_SECONDS))
    result=engine.generate(plan_path,output,wall_seconds=wall_seconds);folder=writable_path(output)
    write_json(folder/"right_hook_full_entry.json",dict(schema=ENTRY_VERSION,plan_sha256=sha256_file(plan_path),
        wrapper_source_sha256=sha256_file(__file__),frozen_generator_source_sha256=sha256_file(engine_source.__file__),
        manifest_sha256=sha256_file(folder/"manifest.json"),family="right_hook",split=plan["split"],
        child_wall_seconds=wall_seconds,solver_changed=False,engineering_cap_rad=.205,
        physical_qualification=False,native_frozen_control_required_before_learning=True))
    return result


def create_plans(interior_audit_path,output):
    """Create both complete plans only after the real1.10 parent passes its audit."""
    proposal_path=PROJECT_ROOT/"outputs/task10b/p2rhcacheproposal01/proposal.json"
    proposal=read(binding(proposal_path));selection=read(proposal["bindings"]["selection"])
    prior_path=PROJECT_ROOT/"outputs/task10b/p2familytargetprep01/right_hook_plan.json";prior=read(binding(prior_path))
    ia_path=Path(interior_audit_path).resolve(strict=True);ia=read(binding(ia_path))
    # The audit must provide explicit owned input paths; no guessed latest run.
    actual={k:binding(ia[k+"_path"]) for k in ("manifest","packet","plan","execution","accounting")}
    if any(actual[k]["sha256"]!=ia[k+"_sha256"] for k in actual):raise ValueError("Audited interior input bindings changed")
    probe_folder=PROJECT_ROOT/"outputs/task10b/p2ft02r"
    evidence=dict(timing_selection=proposal["bindings"]["selection"],cache_proposal=binding(proposal_path),
        probe_plan=binding(prior_path),probe_audit=binding(PROJECT_ROOT/"outputs/task10b/p2familytargetaudit01r/audit.json"),
        probe_manifest=binding(probe_folder/"run/manifest.json"),probe_packet=binding(probe_folder/"packet.json"),
        probe_execution=binding(probe_folder/"execution.json"),probe_accounting=binding(probe_folder/"terminal_accounting.json"),
        interior_plan=actual["plan"],interior_audit=binding(ia_path),interior_packet=actual["packet"],
        interior_execution=actual["execution"],interior_accounting=actual["accounting"])
    sources=source_bindings();recipe=read(prior["recipe"])
    recipe.update(**evidence,parent_catalog_sha256=actual["manifest"]["sha256"],solver_sources=sources,
        full_catalog_entry_version=ENTRY_VERSION,path_solver_version=SOLVER_VERSION,
        prior_probe_recipe=prior["recipe"],native_frozen_control_required_before_learning=True,
        interior_training_eligible=False,purpose="Exact RH64train/24validation candidates using measured .205 source path solver; same common recipe and new audited1.10 mixed clock; construction does not qualify execution.")
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False);write_json(folder/"recipe.json",recipe)
    results={}
    for split in LIMITS:
        count,_,_,child=LIMITS[split]
        plan=dict(schema=SCHEMA,entry_version=ENTRY_VERSION,family="right_hook",split=split,start_mode="family_guard",
            path_solver_version=SOLVER_VERSION,retarget_settings=prior["retarget_settings"],sources=sources,**evidence,
            **{k:deepcopy(prior[k]) for k in ("domain","target_geometry","body_metadata","reference_byte_cap","resident_byte_cap")},
            parent_catalog=actual["manifest"],recipe=binding(folder/"recipe.json"),maximum_cells=count,
            maximum_child_seconds=child,output_byte_cap=OUTPUT_CAPS[split],physical_qualification=False,
            native_frozen_control_required_before_learning=True,cells=expected_cells(proposal,split))
        resolve_cells(plan);path=folder/(split+"_plan.json");write_json(path,plan)
        write_json(folder/(split+"_inputs.json"),dict(plan=str(path)));results[split]=binding(path)
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--wall-seconds",type=float,required=True)
    a=p.parse_args(argv);result=generate(a.plan,a.output,wall_seconds=a.wall_seconds)
    print(json.dumps({k:result[k] for k in ("complete","planned_cells","executed_cells","generated_cells","exception")}))
    return 0 if result["complete"] and result["exception"] is None else 1


if __name__=="__main__":raise SystemExit(main())
