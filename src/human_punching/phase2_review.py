"""CPU adjudication of immutable native evidence; no inferred visual credit.

Manual judgments bind each retained trace. Predeclared videos and every flagged
native failure must also bind a fully decoded media product. This module does
not run physics, actors, training, rendering, retries or policy promotion.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .phase2_continuation import verified_run_ancestry
from .phase2_domain import fingerprint
from .phase2_expanded_evaluation import load_declaration,declared_mask_decisions
from .phase2_qualification import (qualify_panel,qualify_family,frozen_mask_catalog,
    target_start_reachability,validate_resolved_audit)
from .phase2_target_catalog import owned_input
from .wbc.provenance import PROJECT_ROOT,sha256_file,writable_path,write_json

PANEL="task10b_phase2_panel_review_configuration_v1"
MANUAL="task10b_phase2_visual_review_v1"
RECEIPT="task10b_phase2_panel_review_receipt_v1"
DIRECT_FIELDS=("fallen","nonfinite","root_assistance","effort_cap_violation",
    "joint_limit_excursion_rad","first_active_contact_s","first_inactive_contact_s",
    "first_any_contact_s","launch_s","observed_active_glove_rms_m",
    "observed_outward_active_glove_rms_m","observed_physical_metrics",
    "request_identity","checkpoint_sha256","predicate_sha256","readiness_pass",
    "immutable_launch_pass","continuity_history_pass","functional_recovery_pass","correct_hand",
    "request_s","preparation_complete_s","readiness_s","final_planning_s",
    "outward_complete_s","recovery_s","first_positional_support_readiness_s")


def read_binding(binding):
    path=owned_input(binding)
    return path,json.loads(path.read_text())


def retained_trial_audits(audit,run):
    """Accept parent traces only through verified same-execution ancestry."""
    ancestry=verified_run_ancestry(run)
    roots=tuple(Path(p).resolve(strict=True) for p in ancestry["run_directories"])
    rows={}
    for panel in audit.get("panels",[]):
        path=Path(panel["path"]).resolve(strict=True)
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError("Audit trace escapes the verified execution lineage")
        raw=owned_input(dict(path=path/"raw_trace.npz",sha256=panel["raw_sha256"]))
        for trial in panel.get("trials",[]):
            key=trial["trial_id"]
            if key in rows:raise ValueError("A physical attempt cannot be audited twice")
            rows[key]=dict(trial=trial,raw_trace_path=str(raw),raw_trace_sha256=panel["raw_sha256"])
    return rows


def verified_video(review,media,raw_sha):
    """Bind the reviewed exact1x50fps video to its independently audited trace."""
    matches=[r for r in media.get("results",[]) if r.get("trial_id")==review["trial_id"]]
    if len(matches)!=1:return None
    row=matches[0];product=row.get("media",{})
    if (row.get("status")!="rendered" or row.get("raw_trace_sha256")!=raw_sha
            or product.get("fps")!=50 or product.get("playback_speeds")!=[1.]
            or type(product.get("decoded_frames")) is not int or product["decoded_frames"]<=0
            or product.get("decoded_frames")!=product.get("video_frames")):
        return None
    if review.get("video") is None:return None
    path=owned_input(review["video"])
    if path!=Path(product["video"]).resolve(strict=True):raise ValueError("Reviewed video is not the bound media product")
    if review["video"]["sha256"] not in row.get("artifacts",{}).values():
        raise ValueError("Media artifact hashes do not include the reviewed video")
    return review["video"]["sha256"]


def adjudicate(configuration_path):
    """Recompute a panel with explicit independent and manual evidence."""
    path=Path(configuration_path).resolve(strict=True)
    config=json.loads(owned_input(dict(path=path,sha256=sha256_file(path))).read_text())
    required={"declaration","outcomes","launch","execution","trace_audit","visual_review","media","mask"}
    if config.get("schema")!=PANEL or set(config.get("inputs",{}))!=required:
        raise ValueError("Exact bounded native panel review inputs required")
    inputs={name:read_binding(record) for name,record in config["inputs"].items()}
    declaration=load_declaration(inputs["declaration"][0])
    run=Path(config["run"]).resolve(strict=True)
    if not run.is_relative_to((PROJECT_ROOT/"outputs/task10b").resolve(strict=True)):
        raise ValueError("Panel review must refer to an owned Phase2 execution")
    for name,file in (("declaration","declaration.json"),("outcomes","outcomes.json"),
                      ("launch","launch.json"),("execution","execution.json")):
        if inputs[name][0]!=run/file:raise ValueError("Review sources must be the exact execution records")
    launch=inputs["launch"][1];execution=inputs["execution"][1];audit=inputs["trace_audit"][1]
    if (launch.get("declaration_identity")!=declaration.identity
            or launch.get("schema")!="task10b_phase2_expanded_native_launch_v1" or not launch.get("slurm_job_id")
            or execution.get("outcomes_sha256")!=config["inputs"]["outcomes"]["sha256"]
            or Path(audit["run"]).resolve(strict=True)!=run
            or audit.get("schema")!="task10b_phase2_independent_trace_audit_v1"
            or config["inputs"]["mask"]["sha256"]!=declaration.declared_mask_sha256):
        raise ValueError("Native execution/audit/request identities differ")
    native_attempts=any(isinstance(row,dict) and row.get("attempted") is True
        for row in audit.get("trial_outcome_resolutions",[]))
    if native_attempts and (audit.get("closed_protocol_launch_audited") is not True
            or (declaration.start_mode=="shared_jab_saved_state"
                and audit.get("shared_saved_state_initialization_audited") is not True)):
        raise ValueError("Native execution/audit/request identities differ: required native protocol proof is missing")
    decisions=declared_mask_decisions(declaration,inputs["mask"][1])
    raw_audits=retained_trial_audits(audit,run)
    resolution=validate_resolved_audit(audit,declaration.bindings)
    records={r["trial_id"]:r for r in audit.get("trial_outcome_resolutions",[])}
    manual=inputs["visual_review"][1];media=inputs["media"][1]
    if (manual.get("schema")!=MANUAL or manual.get("declaration_identity")!=declaration.identity
            or not isinstance(manual.get("reviewer"),str) or not manual["reviewer"].strip()
            or manual.get("family_corridor_and_arm_trunk_sequence_inspected") is not True
            or media.get("schema")!="task10b_phase2_expanded_media_v1"):
        raise ValueError("Explicit trace/video reviewer and family/sequence evidence required")
    visual={r["trial_id"]:r for r in manual["reviews"]}
    expected={b.trial_id for b in declaration.bindings}
    if len(visual)!=len(manual["reviews"]) or not set(visual)<=expected:
        raise ValueError("Duplicate or undeclared manual judgments")
    supplied=inputs["outcomes"][1];outcomes=[]
    for source in supplied:
        row=json.loads(json.dumps(source,allow_nan=False));key=row["trial_id"]
        row.pop("audit_resolution",None)
        row.update(single_outward_stroke_pass=None,movement_exploit=None,
            visual_review="pending",video_sha256=None,execution_complete=False,
            full_active_glove_rms_m=None,outward_active_glove_rms_m=None,
            **{name:None for name in DIRECT_FIELDS})
        item=records.get(key);independent=raw_audits.get(key)
        row["trace_audit_pass"]=bool(audit.get("passed") is True and independent is not None
            and independent["trial"].get("passed") is True and resolution["passed"])
        if resolution["passed"] and item is not None:
            if (item.get("offline_rejection_kind")=="screen_or_mask_rejection"
                    and item.get("declared_mask_sha256")!=declaration.declared_mask_sha256):
                raise ValueError("Offline rejection changes the predeclared mask")
            row["audit_resolution"]=item
            row["execution_complete"]=item.get("execution_complete") is True
        if independent is not None:
            trial=independent["trial"]
            if item is None or item.get("raw_trace_sha256")!=independent["raw_trace_sha256"]:
                raise ValueError("Resolution and independent native trace hashes differ")
            for name in DIRECT_FIELDS:row[name]=trial.get(name)
            if item.get("execution_complete") is True:
                row["full_active_glove_rms_m"]=trial.get("observed_active_glove_rms_m")
                row["outward_active_glove_rms_m"]=trial.get("observed_outward_active_glove_rms_m")
            review=visual.get(key)
            if review is not None:
                if (review.get("raw_trace_sha256")!=independent["raw_trace_sha256"]
                        or type(review.get("single_outward_stroke_pass")) is not bool
                        or type(review.get("movement_exploit")) is not bool
                        or not isinstance(review.get("notes"),str) or not review["notes"].strip()
                        or review.get("evidence_scope") != ("full_retained_native_trace"
                            if item.get("execution_complete") is True else "partial_retained_native_trace")):
                    raise ValueError("Manual stroke/exploit decisions need explicit trace-bound evidence")
                for evidence in review.get("diagnostic_evidence",[]):owned_input(evidence)
                video=verified_video(review,media,independent["raw_trace_sha256"])
                if video is None and not review.get("diagnostic_evidence"):
                    raise ValueError("Manual judgment lacks video or stored diagnostic evidence")
                row.update(single_outward_stroke_pass=review["single_outward_stroke_pass"],
                    movement_exploit=review["movement_exploit"],visual_review=review.get("visual_review","trace_reviewed"),
                    video_sha256=video)
                if video is None and row["visual_review"] in ("reviewed_acceptable","reviewed_failure"):
                    raise ValueError("A visual-video decision requires the exact decoded video")
        if row.get("status")=="generation_failure" and decisions[key]["admitted"]:
            raise ValueError("Generation failure was not rejected in the immutable pre-test mask")
        outcomes.append(row)
    report=qualify_panel(declaration,outcomes)
    return report,dict(schema=RECEIPT,configuration=dict(path=str(path),sha256=sha256_file(path)),
        run=str(run),declaration_identity=declaration.identity,input_bindings=config["inputs"],
        independent_resolution=resolution,source_sha256=sha256_file(__file__),
        manual_judgments_inferred=False,new_physics_steps=0,actor_calls=0,formal_acceptance=False)


def review_panel(config,output):
    report,receipt=adjudicate(config)
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/"qualification.json",report)
    receipt["qualification"]=dict(path=str(folder/"qualification.json"),sha256=sha256_file(folder/"qualification.json"))
    write_json(folder/"review.json",receipt)
    return receipt


def reviewed_panel(binding):
    _,receipt=read_binding(binding)
    if receipt.get("schema")!=RECEIPT or receipt.get("source_sha256")!=sha256_file(__file__):
        raise ValueError("Immutable panel adjudication receipt from the same review implementation required")
    config,_=read_binding(receipt["configuration"])
    _,saved=read_binding(receipt["qualification"])
    current,_=adjudicate(config)
    if fingerprint(current)!=fingerprint(saved):raise ValueError("Stored qualification differs from its immutable audited/manual evidence")
    return saved


def review_family(plan_path,output):
    path=Path(plan_path).resolve(strict=True);plan=json.loads(owned_input(dict(path=path,sha256=sha256_file(path))).read_text())
    if plan.get("schema")!="task10b_phase2_family_review_plan_v1" or len(plan.get("panels",[]))!=3:
        raise ValueError("Exactly three independently reviewed learned final panels required")
    result=qualify_family([reviewed_panel(binding) for binding in plan["panels"]],plan["family"])
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/"qualification.json",result)
    write_json(folder/"review.json",dict(schema="task10b_phase2_family_review_receipt_v1",plan=dict(path=str(path),sha256=sha256_file(path)),
        qualification=dict(path=str(folder/"qualification.json"),sha256=sha256_file(folder/"qualification.json"))))
    return result


def publish_catalog(plan_path,output):
    path=Path(plan_path).resolve(strict=True);plan=json.loads(owned_input(dict(path=path,sha256=sha256_file(path))).read_text())
    if plan.get("schema")!="task10b_phase2_catalog_review_plan_v1" or len(plan.get("families",[]))!=4:
        raise ValueError("All four family review receipts and the complete predeclared requested domain required")
    qualifications=[]
    for binding in plan["families"]:
        _,receipt=read_binding(binding);family_plan,frozen=read_binding(receipt["plan"])
        if (receipt.get("schema")!="task10b_phase2_family_review_receipt_v1"
                or frozen.get("schema")!="task10b_phase2_family_review_plan_v1"
                or len(frozen.get("panels",[]))!=3):
            raise ValueError("Catalog requires complete immutable three-seed family review receipts")
        _,saved=read_binding(receipt["qualification"])
        checked=qualify_family([reviewed_panel(p) for p in frozen["panels"]],frozen["family"])
        if fingerprint(checked)!=fingerprint(saved):raise ValueError("Family qualification differs from its three reviewed seed panels")
        qualifications.append(saved)
    _,mask=read_binding(plan["requested_domain"])
    if mask.get("schema")!="task10b_phase2_declared_evaluation_mask_v1":raise ValueError("Frozen full requested mask required")
    digest=plan["requested_domain"]["sha256"]
    if any(q["declared_mask_sha256"]!=digest for q in qualifications):raise ValueError("Catalog cannot shrink or replace the final-test mask")
    requests=[];test_requests=[]
    for source in mask["requests"]:
        # Validation requests can share the mask. Preserve every requested row
        # in primary coverage and report the final-test subset separately.
        from .wbc.splits import assign_trajectory_groups
        request={k:source[k] for k in ("family","command_id","original_replica_id","ttc_s","start_index",
            "start_state_sha256","reference_sha256","target_w")}
        request.update(declared_mask_sha256=digest,admitted_before_final_test=source["admitted"])
        requests.append(request)
        if assign_trajectory_groups([source["command_id"]])[source["command_id"]]=="test":test_requests.append(request)
    catalog=frozen_mask_catalog(requests,qualifications);coverage=target_start_reachability(requests,qualifications)
    claims=predeclared_choice_claims(mask,coverage,plan["requested_domain"])
    final_catalog=frozen_mask_catalog(test_requests,qualifications)
    final_coverage=target_start_reachability(test_requests,qualifications)
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/"catalog.json",catalog);write_json(folder/"reachability.json",coverage)
    write_json(folder/"choice_claims.json",claims)
    write_json(folder/"final_test_catalog.json",final_catalog)
    write_json(folder/"final_test_reachability.json",final_coverage)
    write_json(folder/"provenance.json",dict(plan=dict(path=str(path),sha256=sha256_file(path)),
        requested_domain=plan["requested_domain"],family_evidence=plan["families"],
        no_policy_promotion=True,formal_acceptance=False,new_physics_steps=0,actor_calls=0))
    return catalog


def predeclared_choice_claims(mask,reachability,mask_binding):
    """Evaluate only the optional choice subsets frozen in the original mask."""
    scenarios={row["scenario_identity"]:row for row in reachability["scenarios"]}
    definitions=(("declared_shared_multi_choice_scenario_ids",2),
                 ("declared_nominal_shared_four_family_scenario_ids",4))
    claims={}
    for field,minimum in definitions:
        ids=mask.get(field,[])
        if (not isinstance(ids,list) or any(not isinstance(value,str) for value in ids)
                or len(set(ids))!=len(ids) or not set(ids)<=set(scenarios)):
            raise ValueError("Predeclared choice subsets require unique IDs from the full requested reachability domain")
        rows=[]
        for identity in ids:
            scenario=scenarios[identity]
            names=scenario["shared_start_reachable_families_at_any_qualified_bin"]
            bins=[set(scenario["families"][name]["shared_start_qualified_bins_s"]) for name in names]
            common=sorted(set.intersection(*bins)) if bins else []
            rows.append(dict(scenario_identity=identity,qualified_shared_families=names,
                qualified_shared_family_count=len(names),passed=len(names)>=minimum,
                common_qualified_shared_ttc_bins_s=common))
        claims[field]=dict(predeclared=bool(ids),minimum_shared_family_count=minimum,
            scenario_ids=ids,passed=bool(ids) and all(row["passed"] for row in rows),scenarios=rows)
    return dict(schema="task10b_phase2_predeclared_choice_claims_v1",requested_domain=mask_binding,
        claims=claims,coverage_is_not_a_predeclared_choice_claim=True,
        common_ttc_claim_inferred_from_any_bin_reachability=False,formal_acceptance=False)
