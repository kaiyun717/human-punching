"""Synthetic statistical/ledger gates only; these are not native results."""
from dataclasses import asdict, replace
import math

import pytest

from human_punching.phase2_domain import fingerprint
from human_punching.phase2_qualification import (
    PanelDeclaration, TrialBinding, clopper_pearson_lower, frozen_mask_catalog,
    qualify_family, qualify_panel, target_start_reachability, validate_resolved_audit,
)
from human_punching.wbc.splits import assign_trajectory_groups


def declaration(family="jab",split="test",seed=21021,start_mode="family_guard"):
    group_map=assign_trajectory_groups(range(1000))
    groups=tuple(k for k,v in group_map.items() if v == split)[:16 if split == "test" else 8]
    timings=(.6,.675,.75)
    bindings=tuple(TrialBinding(f"{family}_{seed}_{group}_{j}_{start}",group,2 if split == "test" else 1,
        ttc,start,fingerprint({"group":group,"start":start}),
        fingerprint({"family":family,"group":group,"ttc":ttc}), (0.,0.,1.2))
        for group in groups for j,ttc in enumerate(timings) for start in range(4))
    return PanelDeclaration(family,split,seed,groups,(.6,.75),.675,bindings,
        tuple(b.trial_id for b in bindings[:1]),"a"*64,fingerprint({"family":family,"seed":seed}),"b"*64,
        "c"*64,fingerprint({"training":family,"seed":seed}),8192,"d"*64,start_mode)


def success_rows(plan):
    return [dict(trial_id=b.trial_id,status="executed",request_identity=b.identity,
        checkpoint_sha256=plan.checkpoint_sha256,predicate_sha256=plan.predicate_sha256,
        trace_audit_pass=True,fallen=False,nonfinite=False,root_assistance=False,
        effort_cap_violation=False,movement_exploit=False,readiness_pass=True,
        immutable_launch_pass=True,continuity_history_pass=True,functional_recovery_pass=True,
        correct_hand=True,single_outward_stroke_pass=True,joint_limit_excursion_rad=.001,
        full_active_glove_rms_m=.01,outward_active_glove_rms_m=.01,
        request_s=0.,preparation_complete_s=.4,readiness_s=.42,final_planning_s=.44,
        launch_s=.46,outward_complete_s=1.5,recovery_s=2.,
        first_active_contact_s=.46+b.ttc_s,first_any_contact_s=.46+b.ttc_s,
        first_inactive_contact_s=None,visual_review="reviewed_acceptable",video_sha256="e"*64,
        extra_stability_pass=False) for b in plan.bindings]


def reports(family="jab",start_mode="family_guard",failed_seed=None):
    result=[]
    for seed in (21021,21022,21023):
        plan=declaration(family=family,seed=seed,start_mode=start_mode);outcomes=success_rows(plan)
        if seed == failed_seed:
            for row in outcomes[:12]:
                row.update(functional_recovery_pass=False,visual_review="reviewed_failure")
        result.append(qualify_panel(plan,outcomes))
    return result


def test_exact_one_sided_confidence_and_full_success():
    assert clopper_pearson_lower(192,192) == pytest.approx(.05**(1/192))
    assert clopper_pearson_lower(0,192) == 0
    assert clopper_pearson_lower(183,192) > .9
    assert clopper_pearson_lower(95,96) > .9
    with pytest.raises(ValueError):clopper_pearson_lower(1.5,192)
    with pytest.raises(ValueError):clopper_pearson_lower(192,192,confidence=.90)


@pytest.mark.parametrize("split,n",[("validation",96),("test",192)])
def test_planned_panel_sizes_and_additional_stability_remains_diagnostic(split,n):
    plan=declaration(split=split);result=qualify_panel(plan,success_rows(plan))
    assert result["passed"] and result["expected_attempts"] == n and result["successes"] == n
    assert not result["extra_stability_is_operational_gate"]
    assert [s["kind"] for s in result["timing_strata"]] == ["trained","untrained_interior","trained"]


def test_missing_rows_stay_denominator_and_cannot_qualify_incomplete_panel():
    plan=declaration();rows=success_rows(plan);result=qualify_panel(plan,rows[:-1])
    assert result["expected_attempts"] == 192 and result["successes"] == 191
    assert result["disposition_counts"]["unattempted"] == 1 and not result["passed"]


@pytest.mark.parametrize("status",["rejected","timeout","generation_failure"])
def test_nonexecution_dispositions_are_retained_failures(status):
    plan=declaration();rows=success_rows(plan)
    rows[0]={"trial_id":rows[0]["trial_id"],"status":status,"reason":"preserved_failure"}
    result=qualify_panel(plan,rows)
    assert result["successes"] == 191 and result["disposition_counts"][status] == 1
    assert result["trials"][0]["source_outcome"]["reason"] == "preserved_failure"
    assert not result["passed"] and not result["checks"]["all_trials_resolved"]


def resolution(binding,kind="audited_complete"):
    row=dict(trial_id=binding.trial_id,request_identity=binding.identity,
        outcome_resolution=kind,resolved=True,attempted=True,execution_complete=kind=="audited_complete",
        raw_trace_sha256="f"*64,prefix_integrity_pass=True,exact_counter_pass=True,
        safe_actor_boundary_pass=True,hard_faults_absent=True)
    if kind=="verified_offline_rejection":row.update(attempted=False,execution_complete=False,
        raw_trace_sha256=None,generation_failure_verified=True,generation_failure_sha256=binding.reference_sha256)
    return row


def test_exact_audited_failed_prefix_resolves_failure_without_success_credit():
    plan=declaration();rows=success_rows(plan);r=resolution(plan.bindings[0],"audited_failed_prefix")
    rows[0].update(status="timeout",audit_resolution=r,trace_audit_pass=True,
        observed_active_glove_rms_m=.01,observed_outward_active_glove_rms_m=.01)
    result=qualify_panel(plan,rows)
    assert result["successes"]==191 and result["checks"]["all_trials_resolved"] and result["passed"]
    rows[0]["fallen"]=True
    assert not qualify_panel(plan,rows)["checks"]["no_hard_physical_violation"]
    rows[0]["fallen"]=False;rows[0]["visual_review"]="pending"
    assert not qualify_panel(plan,rows)["passed"]


def test_known_prefix_before_outward_has_no_invented_outward_rms_or_cycle_credit():
    plan=declaration();rows=success_rows(plan)
    rows[0].update(status="timeout",audit_resolution=resolution(plan.bindings[0],"audited_failed_prefix"),
        observed_active_glove_rms_m=.01,observed_outward_active_glove_rms_m=None,
        observed_physical_metrics=dict(outward_active_sample_count=0),first_any_contact_s=None,
        first_active_contact_s=None,first_inactive_contact_s=None,launch_s=None)
    result=qualify_panel(plan,rows)
    assert result["passed"] and result["successes"]==191
    rows[0]["observed_physical_metrics"]["outward_active_sample_count"]=1
    assert not qualify_panel(plan,rows)["passed"]


def test_verified_offline_rejection_resolves_zero_execution_failure():
    plan=declaration();rows=success_rows(plan)
    rows[0]=dict(trial_id=plan.bindings[0].trial_id,status="generation_failure",
        audit_resolution=resolution(plan.bindings[0],"verified_offline_rejection"))
    result=qualify_panel(plan,rows)
    assert result["passed"] and result["successes"]==191
    rows[0]["status"]="unattempted"
    assert not qualify_panel(plan,rows)["checks"]["all_trials_resolved"]


def test_prefix_or_explicitly_incomplete_execution_never_gets_cycle_credit():
    plan=declaration();rows=success_rows(plan)
    rows[0]["execution_complete"]=False
    assert qualify_panel(plan,rows)["successes"]==191
    rows[0].pop("execution_complete")
    rows[0]["audit_resolution"]=resolution(plan.bindings[0],"audited_failed_prefix")
    assert qualify_panel(plan,rows)["successes"]==191


def test_invalid_complete_resolution_never_gets_cycle_credit():
    plan=declaration();rows=success_rows(plan)
    rows[0]["audit_resolution"]=resolution(plan.bindings[0])
    rows[0]["audit_resolution"]["safe_actor_boundary_pass"]=False
    result=qualify_panel(plan,rows)
    assert result["successes"]==191 and not result["checks"]["all_trials_resolved"]


def test_generated_reference_rejection_requires_independent_predeclared_mask_proof():
    plan=declaration();rows=success_rows(plan);binding=plan.bindings[0]
    r=resolution(binding,"verified_offline_rejection")
    for key in ("generation_failure_verified","generation_failure_sha256"):r.pop(key)
    r.update(offline_rejection_kind="screen_or_mask_rejection",rejected_request_verified=True,
        rejection_evidence_sha256="e"*64,reference_sha256=binding.reference_sha256,
        declared_mask_sha256=plan.declared_mask_sha256,declared_mask_verified=True,catalog_screen_verified=True)
    rows[0]=dict(trial_id=binding.trial_id,status="rejected",audit_resolution=r)
    assert qualify_panel(plan,rows)["passed"]
    r["declared_mask_sha256"]="0"*64
    assert not qualify_panel(plan,rows)["checks"]["all_trials_resolved"]
    r["declared_mask_sha256"]=plan.declared_mask_sha256;r["catalog_screen_verified"]=False
    assert not qualify_panel(plan,rows)["checks"]["all_trials_resolved"]


@pytest.mark.parametrize("field,value",[("resolved",False),("request_identity","0"*64),
    ("raw_trace_sha256",None),("exact_counter_pass",False),("safe_actor_boundary_pass",False),
    ("hard_faults_absent",False),("prefix_integrity_pass",False),("execution_complete",True)])
def test_failed_prefix_resolution_requires_lossless_independent_evidence(field,value):
    plan=declaration();r=resolution(plan.bindings[0],"audited_failed_prefix");r[field]=value
    rows=success_rows(plan);rows[0].update(status="timeout",audit_resolution=r)
    assert not qualify_panel(plan,rows)["checks"]["all_trials_resolved"]


def test_independent_resolution_audit_exact_ids_bindings_and_all_failure_types():
    plan=declaration();items=[resolution(b) for b in plan.bindings]
    items[0]=resolution(plan.bindings[0],"audited_failed_prefix")
    items[1]=resolution(plan.bindings[1],"verified_offline_rejection")
    audit=dict(passed=True,outcome_resolution_passed=True,trial_outcome_resolutions=items)
    assert validate_resolved_audit(audit,plan.bindings)["passed"]
    audit["trial_outcome_resolutions"]=items[:-1]
    assert not validate_resolved_audit(audit,plan.bindings)["passed"]
    audit["trial_outcome_resolutions"]=items+[items[-1]]
    assert not validate_resolved_audit(audit,plan.bindings)["passed"]
    audit["trial_outcome_resolutions"]=items;items[0]["outcome_resolution"]="unknown_or_missing"
    assert not validate_resolved_audit(audit,plan.bindings)["passed"]
    items[0]=resolution(plan.bindings[0]);audit["passed"]=False
    assert not validate_resolved_audit(audit,plan.bindings)["passed"]


def test_per_timing_threshold_is_independent_of_overall_rate():
    plan=declaration();rows=success_rows(plan)
    selected=[i for i,b in enumerate(plan.bindings) if b.ttc_s == .675][:7]
    for i in selected:rows[i].update(functional_recovery_pass=False,visual_review="reviewed_failure")
    result=qualify_panel(plan,rows)
    assert result["rate"] >= .95 and not result["passed"]
    assert not result["checks"]["each_timing_rate_at_least_90_percent"]


@pytest.mark.parametrize("field",["fallen","nonfinite","root_assistance","effort_cap_violation","movement_exploit"])
def test_single_hard_violation_fails_whole_panel(field):
    plan=declaration();rows=success_rows(plan);rows[0][field]=True
    result=qualify_panel(plan,rows)
    assert result["rate"] >= .95 and not result["passed"] and result["hard_failures"]


def test_first_contact_early_hit_cannot_be_repaired_and_wrong_hand_fails():
    plan=declaration();rows=success_rows(plan)
    rows[0].update(first_active_contact_s=.46+.49,first_any_contact_s=.46+.49)
    rows[1]["first_inactive_contact_s"] = .46+.50
    rows[1]["first_any_contact_s"] = .46+.50
    result=qualify_panel(plan,rows)
    assert not result["trials"][0]["passed"] and not result["trials"][1]["passed"]
    assert "first_contact_timing_failed" in result["trials"][0]["reasons"]
    assert any(r["reason"]=="early_contact_floor_or_unknown_launch" for r in result["hard_failures"])


@pytest.mark.parametrize("field,value",[("joint_limit_excursion_rad",.0201),
    ("full_active_glove_rms_m",.0201),("outward_active_glove_rms_m",.0201)])
def test_panel_physical_limits_cannot_hide_in_allowed_failed_cycle_fraction(field,value):
    plan=declaration();rows=success_rows(plan);rows[0][field]=value
    result=qualify_panel(plan,rows)
    assert result["rate"]>.95 and not result["checks"]["no_hard_physical_violation"]


def test_late_hit_is_failed_cycle_but_contact_above_floor_is_not_floor_violation():
    plan=declaration();rows=success_rows(plan)
    rows[0].update(first_active_contact_s=.46+.55,first_any_contact_s=.46+.55)
    result=qualify_panel(plan,rows)
    assert result["successes"]==191 and result["checks"]["no_hard_physical_violation"]


@pytest.mark.parametrize("field,value",[("joint_limit_excursion_rad",.0201),("full_active_glove_rms_m",.0201),
    ("outward_active_glove_rms_m",math.nan),("launch_s",.461),("first_active_contact_s","bad"),
    ("request_identity","0"*64),("checkpoint_sha256","0"*64),("continuity_history_pass",False)])
def test_metrics_clock_history_and_immutable_identities_fail_closed(field,value):
    plan=declaration();rows=success_rows(plan);rows[0][field]=value
    # Real artifacts disallow JSON NaN; this API additionally must reject it.
    result=qualify_panel(plan,rows)
    assert not result["trials"][0]["passed"]


def test_predeclared_sample_and_all_flagged_failures_need_review():
    plan=declaration();rows=success_rows(plan)
    rows[0]["visual_review"]="pending"
    rows[1].update(functional_recovery_pass=False,visual_review="pending")
    result=qualify_panel(plan,rows)
    assert not result["passed"] and set(result["unreviewed_videos"]) == {rows[0]["trial_id"],rows[1]["trial_id"]}


def test_split_timing_variants_and_original_replica_are_bound():
    plan=declaration()
    with pytest.raises(ValueError):replace(plan,command_groups=(3307,)+plan.command_groups[1:])
    with pytest.raises(ValueError):replace(plan,interpolation_timing_s=.6)
    binding=replace(plan.bindings[0],original_replica_id=0)
    with pytest.raises(ValueError):replace(plan,bindings=(binding,)+plan.bindings[1:])
    binding=replace(plan.bindings[0],start_state_sha256="f"*64)
    with pytest.raises(ValueError):replace(plan,bindings=(binding,)+plan.bindings[1:])
    with pytest.raises(ValueError):replace(plan,seed=0)
    with pytest.raises(ValueError):replace(plan,fresh_transitions=0)
    with pytest.raises(ValueError):replace(plan,checkpoint_sha256=plan.parent_checkpoint_sha256)


def test_duplicate_or_extra_outcomes_are_not_ignored():
    plan=declaration();rows=success_rows(plan)
    with pytest.raises(ValueError):qualify_panel(plan,rows+[rows[0]])
    with pytest.raises(ValueError):qualify_panel(plan,rows+[{"trial_id":"extra"}])


def test_three_independent_fine_tuning_seeds_required_and_failed_seed_not_legal():
    good=reports();assert qualify_family(good,"jab")["passed"]
    with pytest.raises(ValueError):qualify_family(good[:2],"jab")
    with pytest.raises(ValueError):qualify_family([good[0]]*3,"jab")
    bad=reports(failed_seed=21022);proof=qualify_family(bad,"jab")
    assert not proof["passed"] and proof["failed_seeds"] == [21022]
    request={**proof["accepted_binding_keys"][0],"family":"jab","declared_mask_sha256":"d"*64,
             "admitted_before_final_test":True}
    catalog=frozen_mask_catalog([request],[proof])
    assert not catalog["requests"][0]["legal"] and catalog["coverage_zero_to_four"]["0"] == 1


def test_mask_no_nearest_fallback_and_family_guards_do_not_claim_shared_choice():
    proofs=[qualify_family(reports(family=f),f) for f in ("jab","cross")]
    requests=[{**p["accepted_binding_keys"][0],"family":p["family"],"declared_mask_sha256":"d"*64,
               "admitted_before_final_test":True} for p in proofs]
    result=frozen_mask_catalog(requests,proofs)
    assert result["coverage_zero_to_four"]["2"] == 1
    assert not result["shared_multi_choice_subset"]
    requests[0]["ttc_s"] += .001
    result=frozen_mask_catalog(requests,proofs)
    assert not result["requests"][0]["legal"]


def test_shared_choice_requires_shared_saved_state_qualification_and_pretest_mask():
    proofs=[qualify_family(reports(family=f,start_mode="shared_jab_saved_state"),f) for f in ("jab","cross")]
    requests=[{**p["accepted_binding_keys"][0],"family":p["family"],"declared_mask_sha256":"d"*64,
               "admitted_before_final_test":True} for p in proofs]
    assert len(frozen_mask_catalog(requests,proofs)["shared_multi_choice_subset"]) == 1
    requests[0]["declared_mask_sha256"]="f"*64
    assert not frozen_mask_catalog(requests,proofs)["shared_multi_choice_subset"]


def reachability_fixture(start_mode="shared_jab_saved_state"):
    """Synthetic four-family passes with deliberately different timing intervals."""
    proofs=[];requests=[]
    for family,timings in (("jab",(.6,.675,.75)),("cross",(.6,.675,.75)),
                           ("left_hook",(.95,1.025,1.1)),("right_hook",(.8,.9,1.0))):
        panels=[]
        for seed in (21021,21022,21023):
            plan=declaration(family=family,seed=seed,start_mode=start_mode)
            lookup=dict(zip(plan.timings,timings))
            bindings=tuple(replace(b,ttc_s=lookup[b.ttc_s],reference_sha256=
                fingerprint({"family":family,"group":b.command_id,"ttc":lookup[b.ttc_s]})) for b in plan.bindings)
            plan=replace(plan,trained_timings_s=(timings[0],timings[2]),
                         interpolation_timing_s=timings[1],bindings=bindings)
            outcomes=success_rows(plan)
            for row in outcomes:row.update(outward_complete_s=2.,recovery_s=2.5)
            panels.append(qualify_panel(plan,outcomes))
        proof=qualify_family(panels,family);assert proof["passed"]
        proofs.append(proof)
        group=proof["accepted_binding_keys"][0]["command_id"]
        requests.extend({**b,"family":family,"declared_mask_sha256":"d"*64,
            "admitted_before_final_test":True} for b in proof["accepted_binding_keys"]
            if b["command_id"] == group and b["start_index"] == 0)
    return requests,proofs


def test_any_bin_four_family_shared_reachability_does_not_claim_common_ttc():
    requests,proofs=reachability_fixture()
    catalog_before=frozen_mask_catalog(requests,proofs)
    result=target_start_reachability(requests,proofs)
    assert result["requested_cells"] == 12 and result["requested_scenarios"] == 1
    assert result["coverage_any_qualified_bin_zero_to_four"]["4"] == 1
    assert result["coverage_shared_start_any_qualified_bin_zero_to_four"]["4"] == 1
    assert len(result["shared_four_family_any_bin_subset"]) == 1
    assert result["shared_four_family_common_ttc_subset"] == []
    scenario=result["scenarios"][0]
    assert scenario["all_four_common_qualified_bins_s"] == []
    assert scenario["pairwise_common_bins"][0]["qualified_bins_s"] == [.6,.675,.75]
    assert len(scenario["shared_state_proof"]["qualified_family_reports"]) == 4
    assert frozen_mask_catalog(requests,proofs) == catalog_before
    assert catalog_before["coverage_zero_to_four"]["4"] == 0


def test_any_bin_summary_does_not_infer_unrequested_families_or_bins():
    requests,proofs=reachability_fixture()
    result=target_start_reachability(requests[:1],proofs)
    scenario=result["scenarios"][0]
    assert result["coverage_any_qualified_bin_zero_to_four"]["1"] == 1
    assert scenario["families"]["jab"]["qualified_bins_s"] == [.6]
    assert scenario["families"]["cross"]["requested"] is False
    assert not scenario["families"]["cross"]["qualified_bins_s"]
    assert scenario["all_four_common_qualified_bins_s"] == []


def test_matching_hashes_of_family_guards_never_establish_shared_choice():
    requests,proofs=reachability_fixture(start_mode="family_guard")
    result=target_start_reachability(requests,proofs)
    assert result["coverage_any_qualified_bin_zero_to_four"]["4"] == 1
    assert result["coverage_shared_start_any_qualified_bin_zero_to_four"]["0"] == 1
    assert result["shared_four_family_any_bin_subset"] == []
    assert result["scenarios"][0]["shared_state_proof"]["qualified_family_reports"] == {}


def test_distinct_saved_guard_states_are_never_merged():
    requests,proofs=reachability_fixture(start_mode="family_guard")
    requests=[{**r,"start_state_sha256":fingerprint({"family":r["family"]})} for r in requests]
    result=target_start_reachability(requests,proofs)
    assert result["requested_scenarios"] == 4
    assert result["coverage_any_qualified_bin_zero_to_four"]["0"] == 4
    assert all(cell["status"] == "exact_request_not_qualified" for scenario in result["scenarios"]
        for family in scenario["families"].values() for cell in family["requests"])


def test_unqualified_excluded_and_failed_requests_remain_without_infeasibility_claim():
    requests,proofs=reachability_fixture()
    request=requests[0]
    missing=target_start_reachability([request],[])
    excluded=target_start_reachability([{**request,"admitted_before_final_test":False}],[])
    bad=qualify_family(reports(failed_seed=21022),"jab")
    failed=target_start_reachability([request],[bad])
    mismatch=target_start_reachability([{**request,"declared_mask_sha256":"f"*64}],proofs)
    for result,status in ((missing,"missing_family_qualification"),(excluded,"not_admitted_before_final_test"),
                          (failed,"failed_final_family_qualification"),(mismatch,"pretest_mask_mismatch")):
        assert result["requested_cells"] == 1 and result["coverage_any_qualified_bin_zero_to_four"]["0"] == 1
        assert result["scenarios"][0]["families"]["jab"]["requests"][0]["status"] == status
        assert result["measured_infeasibility_claimed"] is False


def test_reachability_exact_common_bins_and_request_validation():
    proofs=[qualify_family(reports(family=f,start_mode="shared_jab_saved_state"),f)
            for f in ("jab","cross","left_hook","right_hook")]
    requests=[{**p["accepted_binding_keys"][0],"family":p["family"],"declared_mask_sha256":"d"*64,
               "admitted_before_final_test":True} for p in proofs]
    result=target_start_reachability(requests,proofs)
    assert result["scenarios"][0]["all_four_common_shared_start_bins_s"] == [.6]
    assert len(result["shared_four_family_common_ttc_subset"]) == 1
    with pytest.raises(ValueError):target_start_reachability([{**requests[0],"family":"invented"}],proofs)
    with pytest.raises(ValueError):target_start_reachability([{**requests[0],"admitted_before_final_test":1}],proofs)
    with pytest.raises(ValueError):target_start_reachability([{**requests[0],"ttc_s":.1}],proofs)
