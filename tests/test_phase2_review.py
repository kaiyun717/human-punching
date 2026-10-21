"""Synthetic adjudication evidence; these fixtures establish no native result."""
from dataclasses import asdict,replace
import importlib.util
import json
from pathlib import Path

import pytest

from human_punching import phase2_review as review
from human_punching import phase2_target_catalog
from human_punching.wbc.provenance import sha256_file

SPEC=importlib.util.spec_from_file_location("review_qualification_fixture",Path(__file__).with_name("test_phase2_qualification.py"))
fixture=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(fixture)


def encoded(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
    return dict(path=str(path),sha256=sha256_file(path))


@pytest.fixture
def example(tmp_path,monkeypatch):
    monkeypatch.setattr(phase2_target_catalog,"PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(review,"PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(review,"writable_path",lambda path:Path(path).resolve())
    run=tmp_path/"outputs/task10b/run";run.mkdir(parents=True)
    monkeypatch.setattr(review,"verified_run_ancestry",lambda _:dict(run_directories=[str(run)]))
    declaration=fixture.declaration()
    mask=encoded(tmp_path/"mask.json",dict(schema="task10b_phase2_declared_evaluation_mask_v1",
        requests=[dict(family=declaration.family,admitted=True,**{k:v for k,v in asdict(b).items() if k!="trial_id"}) for b in declaration.bindings]))
    declaration=replace(declaration,declared_mask_sha256=mask["sha256"])
    inputs=dict(mask=mask,declaration=encoded(run/"declaration.json",asdict(declaration)))
    rows=fixture.success_rows(declaration);inputs["outcomes"]=encoded(run/"outcomes.json",rows)
    inputs["execution"]=encoded(run/"execution.json",dict(outcomes_sha256=inputs["outcomes"]["sha256"]))
    inputs["launch"]=encoded(run/"launch.json",dict(schema="task10b_phase2_expanded_native_launch_v1",
        slurm_job_id="synthetic_fixture_never_executed",declaration_identity=declaration.identity))
    batch=run/"batch_0000";batch.mkdir();raw=batch/"raw_trace.npz";raw.write_bytes(b"synthetic unexecuted trace")
    resolutions=[];trials=[]
    for binding,row in zip(declaration.bindings,rows,strict=True):
        item=fixture.resolution(binding);item["raw_trace_sha256"]=sha256_file(raw);resolutions.append(item)
        trials.append(dict(trial_id=binding.trial_id,passed=True,fallen=False,nonfinite=False,root_assistance=False,
            request_identity=binding.identity,checkpoint_sha256=declaration.checkpoint_sha256,
            predicate_sha256=declaration.predicate_sha256,readiness_pass=True,immutable_launch_pass=True,
            continuity_history_pass=True,functional_recovery_pass=True,correct_hand=True,
            request_s=0.,preparation_complete_s=.4,readiness_s=.46,final_planning_s=.46,
            outward_complete_s=1.5,recovery_s=2.,first_positional_support_readiness_s=.42,
            effort_cap_violation=False,joint_limit_excursion_rad=.001,launch_s=.46,first_active_contact_s=.46+binding.ttc_s,
            first_inactive_contact_s=None,first_any_contact_s=.46+binding.ttc_s,observed_active_glove_rms_m=.01,
            observed_outward_active_glove_rms_m=.01,observed_physical_metrics=dict(execution_complete=True)))
    audit=dict(schema="task10b_phase2_independent_trace_audit_v1",run=str(run),passed=True,outcome_resolution_passed=True,
        closed_protocol_launch_audited=True,
        trial_outcome_resolutions=resolutions,panels=[dict(path=str(batch),raw_sha256=sha256_file(raw),trials=trials)])
    inputs["trace_audit"]=encoded(tmp_path/"audit.json",audit)
    plot=tmp_path/"diagnostics.pdf";plot.write_bytes(b"synthetic trace diagnostic fixture")
    plot_binding=dict(path=str(plot),sha256=sha256_file(plot))
    manual=dict(schema=review.MANUAL,declaration_identity=declaration.identity,reviewer="synthetic test reviewer",
        family_corridor_and_arm_trunk_sequence_inspected=True,reviews=[dict(trial_id=b.trial_id,
            raw_trace_sha256=sha256_file(raw),single_outward_stroke_pass=True,movement_exploit=False,
            notes="Synthetic fixture only",evidence_scope="full_retained_native_trace",diagnostic_evidence=[plot_binding]) for b in declaration.bindings])
    video=tmp_path/"video.mp4";video.write_bytes(b"synthetic unrendered video fixture")
    manual["reviews"][0].update(video=dict(path=str(video),sha256=sha256_file(video)),visual_review="reviewed_acceptable")
    media=dict(schema="task10b_phase2_expanded_media_v1",results=[dict(trial_id=declaration.bindings[0].trial_id,status="rendered",
        raw_trace_sha256=sha256_file(raw),media=dict(video=str(video),video_frames=100,decoded_frames=100,fps=50,playback_speeds=[1.]),
        artifacts={"video.mp4":sha256_file(video)})])
    inputs["visual_review"]=encoded(tmp_path/"manual.json",manual)
    inputs["media"]=encoded(tmp_path/"media.json",media)
    config=dict(schema=review.PANEL,run=str(run),inputs=inputs)
    return tmp_path,config,declaration,audit,manual,media


def adjudicate(example):
    root,config,*_=example;path=root/"review_plan.json";encoded(path,config)
    return review.adjudicate(path)


def change_input(example,name,value):
    root,config,*_=example
    config["inputs"][name]=encoded(root/(name+"_changed.json"),value)


def test_explicit_independent_and_trace_video_review_produces_qualification(example):
    report,receipt=adjudicate(example)
    assert report["passed"] and report["successes"]==192 and receipt["independent_resolution"]["passed"]
    assert not receipt["manual_judgments_inferred"] and receipt["new_physics_steps"]==0


def test_application_visual_flags_cannot_replace_explicit_reviewer(example):
    root,config,_,_,manual,_=example;manual["reviews"]=[];change_input(example,"visual_review",manual)
    # The application file already contains true single-stroke/exploit flags.
    report,_=adjudicate(example)
    assert not report["passed"] and report["successes"]==0


def test_missing_predeclared_video_blocks_qualification_even_with_diagnostics(example):
    _,_,_,_,manual,_=example
    manual["reviews"][0].pop("video");manual["reviews"][0]["visual_review"]="trace_reviewed"
    change_input(example,"visual_review",manual)
    report,_=adjudicate(example)
    assert not report["passed"] and report["unreviewed_videos"]


def test_unknown_extent_or_failed_audit_cannot_receive_automatic_credit(example):
    _,_,_,audit,_,_=example;audit["outcome_resolution_passed"]=False
    change_input(example,"trace_audit",audit)
    report,_=adjudicate(example)
    assert not report["passed"] and report["successes"]==0


@pytest.mark.parametrize("change",["trace_hash","judgment_hash","video_rate","video_decode","missing_diagnostic","missing_flag"])
def test_review_rejects_tampered_or_unbound_evidence(example,change):
    _,_,_,audit,manual,media=example
    if change=="trace_hash":
        audit["panels"][0]["raw_sha256"]="0"*64;change_input(example,"trace_audit",audit)
    elif change=="judgment_hash":
        manual["reviews"][1]["raw_trace_sha256"]="0"*64;change_input(example,"visual_review",manual)
    elif change=="missing_diagnostic":
        manual["reviews"][1]["diagnostic_evidence"]=[];change_input(example,"visual_review",manual)
    elif change=="missing_flag":
        manual["reviews"][1].pop("movement_exploit");change_input(example,"visual_review",manual)
    else:
        if change=="video_rate":media["results"][0]["media"]["fps"]=25
        else:media["results"][0]["media"]["decoded_frames"]=50
        change_input(example,"media",media)
    if change in ("video_rate","video_decode"):
        with pytest.raises(ValueError,match="video"):adjudicate(example)
    else:
        with pytest.raises(ValueError):adjudicate(example)


def test_independent_physical_violation_cannot_be_hidden_by_application_metrics(example):
    _,_,_,audit,_,_=example;audit["panels"][0]["trials"][1]["joint_limit_excursion_rad"]=.03
    change_input(example,"trace_audit",audit)
    report,_=adjudicate(example)
    assert not report["passed"] and not report["checks"]["no_hard_physical_violation"]


@pytest.mark.parametrize("field",["functional_recovery_pass","readiness_pass","immutable_launch_pass",
    "continuity_history_pass","correct_hand","request_s","preparation_complete_s","readiness_s",
    "final_planning_s","launch_s","outward_complete_s","recovery_s","request_identity",
    "checkpoint_sha256","predicate_sha256"])
def test_missing_independent_gate_or_clock_cannot_fall_back_to_application(example,field):
    _,_,_,audit,_,_=example
    for trial in audit["panels"][0]["trials"]:trial.pop(field)
    change_input(example,"trace_audit",audit)
    report,_=adjudicate(example)
    assert not report["passed"] and report["successes"]==0


def test_positive_audit_requires_closed_launch_protocol_proof(example):
    _,_,_,audit,_,_=example;audit.pop("closed_protocol_launch_audited")
    change_input(example,"trace_audit",audit)
    with pytest.raises(ValueError,match="identities differ"):adjudicate(example)


def test_wholly_offline_resolved_rejections_report_zero_credit_without_native_protocol_proof(example):
    root,config,declaration,audit,manual,_=example
    mask=json.loads(Path(config["inputs"]["mask"]["path"]).read_text())
    for row in mask["requests"]:row.update(admitted=False,reason="synthetic failed generation")
    mask_binding=encoded(root/"mask_offline.json",mask)
    declaration=replace(declaration,declared_mask_sha256=mask_binding["sha256"])
    config["inputs"]["mask"]=mask_binding
    run=Path(config["run"])
    config["inputs"]["declaration"]=encoded(run/"declaration.json",asdict(declaration))
    launch=json.loads((run/"launch.json").read_text());launch["declaration_identity"]=declaration.identity
    config["inputs"]["launch"]=encoded(run/"launch.json",launch)
    rows=[dict(trial_id=b.trial_id,status="generation_failure",physical_execution_attempted=False) for b in declaration.bindings]
    config["inputs"]["outcomes"]=encoded(run/"outcomes.json",rows)
    config["inputs"]["execution"]=encoded(run/"execution.json",dict(outcomes_sha256=config["inputs"]["outcomes"]["sha256"]))
    audit.update(panels=[],closed_protocol_launch_audited=False,
        trial_outcome_resolutions=[fixture.resolution(b,"verified_offline_rejection") for b in declaration.bindings])
    change_input(example,"trace_audit",audit)
    manual.update(declaration_identity=declaration.identity,reviews=[])
    change_input(example,"visual_review",manual)
    report,_=adjudicate(example)
    assert not report["passed"] and report["successes"]==0
    assert report["checks"]["all_trials_resolved"] and not report["unreviewed_videos"]


def test_manual_review_cannot_label_partial_trace_as_complete(example):
    _,_,_,_,manual,_=example
    manual["reviews"][1]["evidence_scope"]="partial_retained_native_trace"
    change_input(example,"visual_review",manual)
    with pytest.raises(ValueError,match="trace-bound"):adjudicate(example)


def choice_reachability(names):
    return dict(scenarios=[dict(scenario_identity="a"*64,
        shared_start_reachable_families_at_any_qualified_bin=names,
        families={name:dict(shared_start_qualified_bins_s=[.6,.75] if name!="left_hook" else [.75,.9]) for name in names})])


def test_choice_coverage_does_not_imply_predeclared_choice_milestone():
    result=review.predeclared_choice_claims({},choice_reachability(["jab","cross","left_hook","right_hook"]),{"sha256":"b"*64})
    assert all(not row["predeclared"] and not row["passed"] for row in result["claims"].values())


def test_choice_claims_require_predeclared_reachable_shared_families_and_real_intersection():
    mask=dict(declared_shared_multi_choice_scenario_ids=["a"*64],
        declared_nominal_shared_four_family_scenario_ids=["a"*64])
    result=review.predeclared_choice_claims(mask,choice_reachability(["jab","cross","left_hook","right_hook"]),{"sha256":"b"*64})
    assert all(row["passed"] for row in result["claims"].values())
    assert result["claims"]["declared_nominal_shared_four_family_scenario_ids"]["scenarios"][0]["common_qualified_shared_ttc_bins_s"]==[.75]
    result=review.predeclared_choice_claims(mask,choice_reachability(["jab"]),{"sha256":"b"*64})
    assert all(not row["passed"] for row in result["claims"].values())


@pytest.mark.parametrize("ids",[["0"*64],["a"*64,"a"*64],"a"*64])
def test_choice_claims_reject_unknown_duplicate_or_unstructured_predeclarations(ids):
    with pytest.raises(ValueError,match="unique IDs"):
        review.predeclared_choice_claims(dict(declared_shared_multi_choice_scenario_ids=ids),choice_reachability(["jab","cross"]),{"sha256":"b"*64})


def test_publisher_preserves_all_frozen_requests_and_failed_family_without_mask_tuning(tmp_path,monkeypatch):
    """Publisher-only synthetic proofs; no native evidence is established."""
    monkeypatch.setattr(phase2_target_catalog,"PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(review,"writable_path",lambda path:Path(path).resolve())
    declarations={name:fixture.declaration(family=name,start_mode="shared_jab_saved_state")
        for name in ("jab","cross","left_hook","right_hook")}
    mask=dict(schema="task10b_phase2_declared_evaluation_mask_v1",requests=[])
    for family,declaration in declarations.items():
        for binding in declaration.bindings:
            mask["requests"].append(dict(family=family,admitted=True,
                **{key:value for key,value in asdict(binding).items() if key!="trial_id"}))
    # One unsupported validation request remains in primary coverage as zero.
    extra=fixture.declaration(split="validation").bindings[0]
    mask["requests"].append(dict(family="jab",admitted=False,
        **{key:value for key,value in asdict(extra).items() if key!="trial_id"}))
    first=declarations["jab"].bindings[0]
    from human_punching.phase2_domain import fingerprint
    scenario=fingerprint({key:value for key,value in asdict(first).items()
        if key in ("command_id","original_replica_id","start_index","start_state_sha256","target_w")})
    mask.update(declared_shared_multi_choice_scenario_ids=[scenario],
        declared_nominal_shared_four_family_scenario_ids=[scenario])
    mask_binding=encoded(tmp_path/"mask.json",mask)
    panel_reports={};family_bindings=[]
    for family,declaration in declarations.items():
        reports=[];panel_bindings=[]
        for seed in (21021,21022,21023):
            plan=replace(fixture.declaration(family=family,seed=seed,start_mode="shared_jab_saved_state"),
                declared_mask_sha256=mask_binding["sha256"])
            rows=fixture.success_rows(plan)
            if family=="cross" and seed==21023:
                for row in rows[:12]:row.update(functional_recovery_pass=False,visual_review="reviewed_failure")
            report=review.qualify_panel(plan,rows);reports.append(report)
            binding=encoded(tmp_path/f"{family}_{seed}_synthetic_review.json",{})
            panel_reports[binding["path"]]=report;panel_bindings.append(binding)
        family_plan=encoded(tmp_path/f"{family}_plan.json",dict(schema="task10b_phase2_family_review_plan_v1",
            family=family,panels=panel_bindings))
        qualification=encoded(tmp_path/f"{family}_qualification.json",review.qualify_family(reports,family))
        family_bindings.append(encoded(tmp_path/f"{family}_review.json",dict(schema="task10b_phase2_family_review_receipt_v1",
            plan=family_plan,qualification=qualification)))
    monkeypatch.setattr(review,"reviewed_panel",lambda binding:panel_reports[binding["path"]])
    plan=encoded(tmp_path/"catalog_plan.json",dict(schema="task10b_phase2_catalog_review_plan_v1",
        families=family_bindings,requested_domain=mask_binding))
    output=tmp_path/"catalog"
    catalog=review.publish_catalog(plan["path"],output)
    assert len(catalog["requests"])==769 and catalog["all_requested_scenarios_preserved"]
    assert not any(row["legal"] for row in catalog["requests"] if row["request"]["family"]=="cross")
    final=json.loads((output/"final_test_catalog.json").read_text())
    assert len(final["requests"])==768
    coverage=json.loads((output/"reachability.json").read_text())
    assert coverage["coverage_shared_start_any_qualified_bin_zero_to_four"]["0"]==1
    choices=json.loads((output/"choice_claims.json").read_text())["claims"]
    assert choices["declared_shared_multi_choice_scenario_ids"]["passed"]
    assert not choices["declared_nominal_shared_four_family_scenario_ids"]["passed"]


def test_review_receipt_recomputes_source_evidence_and_preserves_outcomes(example):
    root,config,*_=example;plan=root/"review_plan.json";encoded(plan,config)
    original=Path(config["inputs"]["outcomes"]["path"]).read_bytes()
    review.review_panel(plan,root/"reviewed")
    receipt=root/"reviewed/review.json";binding=dict(path=str(receipt),sha256=sha256_file(receipt))
    assert review.reviewed_panel(binding)["passed"]
    assert Path(config["inputs"]["outcomes"]["path"]).read_bytes()==original
    Path(config["inputs"]["visual_review"]["path"]).write_text('{"changed":true}')
    with pytest.raises(ValueError,match="hash"):review.reviewed_panel(binding)
