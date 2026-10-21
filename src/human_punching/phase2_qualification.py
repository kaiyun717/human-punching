"""Predeclared Phase 2 panels and three-seed qualification; no execution.

Thresholds and masks are inputs frozen before final tests. Missing, rejected and
timed-out attempts stay in the denominator. Synthetic callers establish no policy
competence; native trace audits and the declared video review are required.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import itertools
import math
from typing import Iterable

from scipy.stats import beta

from .phase2_domain import FAMILIES, SEEDS, fingerprint
from .wbc.splits import assign_trajectory_groups


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):
        return {"nonfinite_float":repr(value)}
    if isinstance(value,dict):return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_safe(v) for v in value]
    return value


@dataclass(frozen=True)
class TrialBinding:
    trial_id: str
    command_id: int
    original_replica_id: int
    ttc_s: float
    start_index: int
    start_state_sha256: str
    reference_sha256: str
    target_w: tuple[float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "target_w", tuple(float(x) for x in self.target_w))
        if (not self.trial_id or type(self.command_id) is not int or self.command_id < 0
                or self.command_id == 3307 or self.original_replica_id not in (0,1,2)
                or type(self.start_index) is not int or self.start_index not in range(4)
                or not math.isfinite(self.ttc_s) or self.ttc_s < .5
                or len(self.target_w) != 3 or not all(map(math.isfinite,self.target_w))
                or not _hash(self.start_state_sha256) or not _hash(self.reference_sha256)):
            raise ValueError("Complete original group/request/reference/start binding required")

    @property
    def identity(self):
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class PanelDeclaration:
    family: str
    split: str
    seed: int
    command_groups: tuple[int, ...]
    trained_timings_s: tuple[float, float]
    interpolation_timing_s: float
    bindings: tuple[TrialBinding, ...]
    video_trial_ids: tuple[str, ...]
    route_sha256: str
    checkpoint_sha256: str
    predicate_sha256: str
    parent_checkpoint_sha256: str
    training_receipt_sha256: str
    fresh_transitions: int
    declared_mask_sha256: str
    start_mode: str = "family_guard"
    predicate_version: str = "task10b_functional_guard_v1"

    def __post_init__(self):
        for name in ("command_groups","trained_timings_s","bindings","video_trial_ids"):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        count = {"validation":8,"test":16}.get(self.split)
        if (self.family not in FAMILIES or count is None or self.seed not in SEEDS
                or len(self.command_groups) != count or len(set(self.command_groups)) != count
                or any(type(g) is not int or g < 0 or g == 3307 for g in self.command_groups)
                or self.start_mode not in ("family_guard","shared_jab_saved_state")
                or self.predicate_version != "task10b_functional_guard_v1"
                or type(self.fresh_transitions) is not int or self.fresh_transitions <= 0
                or self.parent_checkpoint_sha256 == self.checkpoint_sha256
                or not all(_hash(getattr(self,k)) for k in ("route_sha256","checkpoint_sha256","predicate_sha256",
                    "parent_checkpoint_sha256","training_receipt_sha256","declared_mask_sha256"))):
            raise ValueError("Frozen 8/16-group validation/test panel and accepted ancestry required")
        groups = assign_trajectory_groups(self.command_groups)
        if set(groups.values()) != {self.split}:
            raise ValueError("Deterministic command-group split isolation violated")
        if (len(self.trained_timings_s) != 2 or not all(map(math.isfinite,self.trained_timings_s))
                or not math.isfinite(self.interpolation_timing_s)
                or not .5 <= self.trained_timings_s[0] < self.interpolation_timing_s < self.trained_timings_s[1]):
            raise ValueError("Declare two trained endpoints and a distinct untrained interior timing")
        if len(self.bindings) != count*3*4 or len({b.trial_id for b in self.bindings}) != len(self.bindings):
            raise ValueError("Complete 96/192-attempt declaration with unique trial IDs required")
        expected = set(itertools.product(self.command_groups,self.timings,range(4)))
        actual = [(b.command_id,b.ttc_s,b.start_index) for b in self.bindings]
        if set(actual) != expected or len(set(actual)) != len(actual):
            raise ValueError("Every group/timing/start combination must occur exactly once")
        replica = {"validation":1,"test":2}[self.split]
        if any(b.original_replica_id != replica for b in self.bindings):
            raise ValueError("Original-bank replica split conflict cannot be resampled")
        # A timing variant cannot quietly select a different physical/controller
        # start or snapshot target of the same command group.
        states,targets = {},{}
        for b in self.bindings:
            key=(b.command_id,b.start_index)
            if states.setdefault(key,b.start_state_sha256) != b.start_state_sha256:
                raise ValueError("Timing variants must retain the same saved start")
            if targets.setdefault(b.command_id,b.target_w) != b.target_w:
                raise ValueError("Timing variants must retain the same frozen snapshot target")
        if (not self.video_trial_ids or len(set(self.video_trial_ids)) != len(self.video_trial_ids)
                or not set(self.video_trial_ids) <= {b.trial_id for b in self.bindings}):
            raise ValueError("Predeclare a nonempty unique random-video sample")

    @property
    def timings(self):
        return (self.trained_timings_s[0],self.interpolation_timing_s,self.trained_timings_s[1])

    @property
    def identity(self):
        return fingerprint(asdict(self))


def clopper_pearson_lower(successes: int, attempts: int, *, confidence=.95) -> float:
    """Exact one-sided lower bound, using all planned attempts."""
    if (type(successes) is not int or type(attempts) is not int or attempts <= 0
            or not 0 <= successes <= attempts or confidence != .95):
        raise ValueError("Integer binomial counts and predeclared one-sided 95% confidence required")
    return 0. if successes == 0 else float(beta.ppf(1-confidence,successes,attempts-successes+1))


def _bounded(row,key,maximum):
    value=row.get(key)
    return (isinstance(value,(int,float)) and not isinstance(value,bool)
            and math.isfinite(value) and 0 <= value <= maximum+1.e-10)


def _resolution_diagnostics(row, binding):
    """Check independent evidence of a known outcome, never cycle success.

    Hashes bind evidence here; callers must also verify the retained files and
    independent audit identity. A cooperative prefix can be a resolved failure.
    A lost prefix or an unsupported assertion cannot resolve an attempt.
    """
    reasons=[]
    if not isinstance(row,dict):return ["missing_resolution_record"]
    if row.get("trial_id") != binding.trial_id:reasons.append("trial_id_mismatch")
    if row.get("request_identity") != binding.identity:reasons.append("request_identity_mismatch")
    if row.get("resolved") is not True:reasons.append("outcome_not_resolved")
    kind=row.get("outcome_resolution")
    if kind in ("audited_complete","audited_failed_prefix"):
        for name in ("attempted","prefix_integrity_pass","exact_counter_pass",
                     "safe_actor_boundary_pass","hard_faults_absent"):
            if row.get(name) is not True:reasons.append(name+"_failed_or_missing")
        if not _hash(row.get("raw_trace_sha256")):reasons.append("missing_raw_trace_identity")
        if row.get("execution_complete") is not (kind == "audited_complete"):
            reasons.append("execution_extent_mismatch")
    elif kind == "verified_offline_rejection":
        if row.get("attempted") is not False or row.get("execution_complete") is not False:
            reasons.append("offline_rejection_has_execution")
        if row.get("raw_trace_sha256") is not None:reasons.append("offline_rejection_has_trace")
        if row.get("offline_rejection_kind","generation_failure")=="generation_failure":
            if (row.get("generation_failure_verified") is not True
                    or row.get("generation_failure_sha256") != binding.reference_sha256):
                reasons.append("offline_rejection_evidence_mismatch")
        elif row.get("offline_rejection_kind")=="screen_or_mask_rejection":
            if (row.get("rejected_request_verified") is not True
                    or row.get("reference_sha256")!=binding.reference_sha256
                    or not _hash(row.get("rejection_evidence_sha256"))
                    or not _hash(row.get("declared_mask_sha256"))
                    or row.get("declared_mask_verified") is not True
                    or row.get("catalog_screen_verified") is not True):
                reasons.append("screen_or_mask_rejection_evidence_mismatch")
        else:reasons.append("unsupported_offline_rejection_kind")
    else:reasons.append("unknown_or_missing_execution_extent")
    return reasons


def validate_resolved_audit(audit, bindings):
    """Verify exact declared outcome coverage independently of cycle rate.

    This deliberately does not require every attempted horizon to complete.
    Complete and lossless failed-prefix traces require the same integrity and
    physical-fault evidence. Unknown/lost extent remains unresolved. The full
    96/192 declaration is still the qualification denominator.
    """
    bindings=tuple(bindings)
    expected={b.trial_id:b for b in bindings}
    diagnostics=[]
    if not bindings:diagnostics.append("empty_declared_panel")
    if len(expected)!=len(bindings):diagnostics.append("duplicate_declared_trial_ids")
    if not isinstance(audit,dict):return dict(passed=False,diagnostics=["missing_independent_audit"])
    if audit.get("passed") is not True:diagnostics.append("independent_audit_failed_or_missing")
    if audit.get("outcome_resolution_passed") is not True:diagnostics.append("audit_outcome_resolution_failed_or_missing")
    rows=audit.get("trial_outcome_resolutions")
    if not isinstance(rows,list):return dict(passed=False,diagnostics=diagnostics+["missing_resolution_ledger"])
    ids=[r.get("trial_id") if isinstance(r,dict) else None for r in rows]
    if (any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids)
            or set(ids)!=set(expected)):
        diagnostics.append("resolution_ids_do_not_match_exact_declaration")
    for row in rows:
        if isinstance(row,dict) and row.get("trial_id") in expected:
            diagnostics.extend(row["trial_id"]+":"+reason
                for reason in _resolution_diagnostics(row,expected[row["trial_id"]]))
    return dict(passed=not diagnostics,diagnostics=diagnostics,expected_trials=len(bindings),
                received_resolution_rows=len(rows),cycle_success_credit=0)


def _numeric_trial(row,binding,declaration):
    reasons=[]
    if row.get("status") != "executed":
        return False,[row.get("status","unattempted")]
    if (row.get("execution_complete") is False
            or (isinstance(row.get("audit_resolution"),dict)
                and row["audit_resolution"].get("outcome_resolution")!="audited_complete")):
        reasons.append("incomplete_or_unresolved_execution_has_no_cycle_credit")
    if row.get("request_identity") != binding.identity:
        reasons.append("immutable_request_mismatch")
    if row.get("checkpoint_sha256") != declaration.checkpoint_sha256:
        reasons.append("restored_checkpoint_mismatch")
    if row.get("predicate_sha256") != declaration.predicate_sha256:
        reasons.append("reviewed_predicate_mismatch")
    if row.get("trace_audit_pass") is not True:
        reasons.append("independent_trace_audit_failed_or_missing")
    for key in ("fallen","nonfinite","root_assistance","effort_cap_violation","movement_exploit"):
        if row.get(key) is not False:reasons.append(key+"_failed_or_missing")
    for key in ("readiness_pass","immutable_launch_pass","continuity_history_pass",
                "functional_recovery_pass","correct_hand","single_outward_stroke_pass"):
        if row.get(key) is not True:reasons.append(key+"_failed_or_missing")
    for key,maximum in (("joint_limit_excursion_rad",.02),("full_active_glove_rms_m",.02),
                        ("outward_active_glove_rms_m",.02)):
        if not _bounded(row,key,maximum):reasons.append(key+"_outside_gate")
    names=("request_s","preparation_complete_s","readiness_s","final_planning_s","launch_s","outward_complete_s","recovery_s")
    values=[row.get(k) for k in names]
    finite=all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values)
    if not finite or any(b < a for a,b in zip(values,values[1:])):
        reasons.append("event_clock_missing_or_nonmonotone")
    elif values[4] < 0 or abs(values[4]/.020-round(values[4]/.020)) > 1.e-7:
        reasons.append("launch_not_actor_boundary")
    first=row.get("first_active_contact_s")
    any_first=row.get("first_any_contact_s")
    inactive=row.get("first_inactive_contact_s")
    first_valid=isinstance(first,(int,float)) and not isinstance(first,bool) and math.isfinite(first)
    if (not finite or not first_valid
            or not isinstance(any_first,(int,float)) or isinstance(any_first,bool) or not math.isfinite(any_first)):
        reasons.append("miss_or_missing_first_contact")
    else:
        delay=first-values[4]
        if not max(.5,binding.ttc_s-.020)-1.e-10 <= delay <= binding.ttc_s+.020+1.e-10:
            reasons.append("first_contact_timing_failed")
        if first > values[5]+1.e-10 or any_first < first-1.e-10:
            reasons.append("contact_outside_outward_or_wrong_hand_first")
        if any_first > first+1.e-10:
            reasons.append("inconsistent_first_any_contact")
    if inactive is not None:
        if not isinstance(inactive,(int,float)) or isinstance(inactive,bool) or not math.isfinite(inactive):
            reasons.append("invalid_inactive_contact_clock")
        elif not first_valid or inactive <= first+1.e-10:
            reasons.append("inactive_hand_first_or_simultaneous")
    return not reasons,reasons


def qualify_panel(declaration: PanelDeclaration, outcomes: Iterable[dict]) -> dict:
    """Evaluate a declared panel; missing rows become explicit failed unattempted rows."""
    supplied=list(outcomes)
    ids=[row.get("trial_id") for row in supplied]
    expected={b.trial_id:b for b in declaration.bindings}
    if len(set(ids)) != len(ids) or not set(ids) <= set(expected):
        raise ValueError("Duplicate or undeclared trial IDs contaminate the frozen panel")
    received=dict(zip(ids,supplied))
    rows=[];hard_failures=[];visual_unreviewed=[];unresolved=[]
    for binding in declaration.bindings:
        row=received.get(binding.trial_id,{"trial_id":binding.trial_id,"status":"unattempted", "reason":"missing_result"})
        if row.get("status") not in ("executed","rejected","timeout","unattempted","generation_failure"):
            raise ValueError("Every trial requires an explicit known disposition")
        passed,reasons=_numeric_trial(row,binding,declaration)
        resolution=row.get("audit_resolution")
        known_resolution=resolution is not None and not _resolution_diagnostics(resolution,binding)
        if known_resolution:
            expected_kind={"executed":"audited_complete","timeout":"audited_failed_prefix",
                "rejected":"audited_failed_prefix",
                "generation_failure":"verified_offline_rejection"}.get(row.get("status"))
            known_resolution=resolution["outcome_resolution"]==expected_kind
            if (row.get("status")=="rejected" and resolution["outcome_resolution"]=="verified_offline_rejection"
                    and resolution.get("offline_rejection_kind")=="screen_or_mask_rejection"):
                known_resolution=resolution.get("declared_mask_sha256")==declaration.declared_mask_sha256
        if ((resolution is not None and not known_resolution)
                or (row.get("status") in ("unattempted","timeout","generation_failure","rejected") and not known_resolution)):
            unresolved.append(binding.trial_id)
            passed=False
            reasons.append("independent_outcome_resolution_failed_or_missing")
        # A partial physical attempt carries hard-fault and video obligations
        # even though its status is timeout and it earns no completed cycle.
        native_attempt=(row.get("status")=="executed" or
            (isinstance(resolution,dict) and resolution.get("attempted") is True))
        if native_attempt:
            for key in ("fallen","nonfinite","root_assistance","effort_cap_violation","movement_exploit"):
                if row.get(key) is not False:hard_failures.append({"trial_id":binding.trial_id,"reason":key})
            prefix=isinstance(resolution,dict) and resolution.get("outcome_resolution")=="audited_failed_prefix"
            metric_keys=("joint_limit_excursion_rad",)+( ("observed_active_glove_rms_m","observed_outward_active_glove_rms_m")
                if prefix else ("full_active_glove_rms_m","outward_active_glove_rms_m"))
            for key in metric_keys:
                if (prefix and key=="observed_outward_active_glove_rms_m" and row.get(key) is None
                        and row.get("observed_physical_metrics",{}).get("outward_active_sample_count")==0):
                    continue  # No outward samples observed; no RMS or cycle credit is inferred.
                if not _bounded(row,key,.02):hard_failures.append({"trial_id":binding.trial_id,"reason":key+"_panel_limit"})
            contact,launch=row.get("first_any_contact_s"),row.get("launch_s")
            if contact is not None:
                if (not isinstance(contact,(int,float)) or isinstance(contact,bool) or not math.isfinite(contact)
                        or not isinstance(launch,(int,float)) or isinstance(launch,bool) or not math.isfinite(launch)
                        or contact-launch<.5-1.e-10):
                    hard_failures.append({"trial_id":binding.trial_id,"reason":"early_contact_floor_or_unknown_launch"})
        visual_needed=(binding.trial_id in declaration.video_trial_ids or
                       (native_attempt and not passed))
        if visual_needed and native_attempt:
            if row.get("visual_review") not in ("reviewed_acceptable","reviewed_failure") or not _hash(row.get("video_sha256")):
                visual_unreviewed.append(binding.trial_id)
            elif row.get("visual_review") == "reviewed_failure":
                passed=False
                reasons.append("visual_behavior_failed")
        rows.append({"trial_id":binding.trial_id,"request_identity":binding.identity,
            "binding":asdict(binding),"status":row["status"],"passed":passed,"reasons":reasons,
            "source_outcome":_json_safe(row),"visual_review_required":visual_needed})
    n=len(rows);successes=sum(r["passed"] for r in rows)
    lower=clopper_pearson_lower(successes,n)
    counts=Counter(r["status"] for r in rows)
    strata=[]
    for timing in declaration.timings:
        selected=[r for r in rows if r["binding"]["ttc_s"] == timing]
        k=sum(r["passed"] for r in selected)
        strata.append({"ttc_s":timing,"kind":"untrained_interior" if timing == declaration.interpolation_timing_s else "trained",
                       "attempts":len(selected),"successes":k,"rate":k/len(selected),"pass":k/len(selected) >= .9})
    checks={"full_planned_ledger":n == (96 if declaration.split == "validation" else 192),
        "all_trials_resolved":not unresolved,
        "complete_cycle_rate_at_least_95_percent":successes/n >= .95,
        "one_sided_exact_95_lower_bound_at_least_90_percent":lower >= .90,
        "each_timing_rate_at_least_90_percent":all(s["pass"] for s in strata),
        "no_hard_physical_violation":not hard_failures,
        "predeclared_and_flagged_videos_reviewed":not visual_unreviewed}
    result={"schema":"Task10BPhase2PanelQualification/v1","declaration_identity":declaration.identity,
        "declaration":asdict(declaration),"family":declaration.family,"seed":declaration.seed,
        "split":declaration.split,"route_sha256":declaration.route_sha256,
        "expected_attempts":n,"received_outcomes":len(supplied),"successes":successes,"rate":successes/n,
        "confidence_method":"one-sided 95% exact Clopper-Pearson beta quantile; all planned attempts denominator",
        "confidence_scope":"Conditional attempt-level binomial bound under the declared panel; shared command groups/starts are clustered scenarios, not independent population samples or a population-level guarantee",
        "confidence_lower":lower,"timing_strata":strata,"disposition_counts":dict(counts),
        "checks":checks,"passed":all(checks.values()),"hard_failures":hard_failures,
        "unreviewed_videos":visual_unreviewed,"unresolved_trials":unresolved,"trials":rows,
        "extra_stability_is_operational_gate":False,
        "final_domain_test":declaration.split == "test", "formal_user_acceptance":False}
    result["report_identity"]=fingerprint(result)
    return result


def qualify_family(reports: Iterable[dict], family: str) -> dict:
    """All three independently fine-tuned seeds must pass untouched final tests."""
    rows=list(reports)
    if family not in FAMILIES or len(rows) != 3 or {r.get("seed") for r in rows} != set(SEEDS):
        raise ValueError("Exactly the three declared independent fine-tuning seeds required")
    if any(r.get("schema") != "Task10BPhase2PanelQualification/v1" or r.get("family") != family
           or r.get("split") != "test" for r in rows):
        raise ValueError("Frozen-policy/validation panels cannot substitute for learned final tests")
    for report in rows:
        unsigned={k:v for k,v in report.items() if k != "report_identity"}
        if report.get("report_identity") != fingerprint(unsigned):
            raise ValueError("Qualification report changed")
    declarations=[PanelDeclaration(**{**r["declaration"],"bindings":tuple(TrialBinding(**b) for b in r["declaration"]["bindings"])}) for r in rows]
    first=declarations[0]
    # Trial IDs may carry seed identity; compare physical request contents instead.
    def physical(d):
        return (d.command_groups,d.timings,d.start_mode,d.route_sha256,d.predicate_sha256,
                d.parent_checkpoint_sha256,d.declared_mask_sha256,
                sorted((b.command_id,b.ttc_s,b.start_index,b.start_state_sha256,b.reference_sha256,b.target_w) for b in d.bindings))
    if any(physical(d) != physical(first) for d in declarations[1:]):
        raise ValueError("Seeds must share the frozen domain, references, starts, timings and predicate")
    if len({d.checkpoint_sha256 for d in declarations}) != 3 or len({d.training_receipt_sha256 for d in declarations}) != 3:
        raise ValueError("Three independent fine-tuning seeds require distinct training/checkpoint receipts")
    passed=all(r["passed"] is True for r in rows)
    result={"schema":"Task10BPhase2FamilyQualification/v1","family":family,"seeds":list(SEEDS),
        "passed":passed,"route_sha256":first.route_sha256,"start_mode":first.start_mode,
        "declared_mask_sha256":first.declared_mask_sha256,"parent_checkpoint_sha256":first.parent_checkpoint_sha256,
        "panel_report_identities":sorted(r["report_identity"] for r in rows),
        "failed_seeds":sorted(r["seed"] for r in rows if not r["passed"]),
        "accepted_binding_keys":[{k:v for k,v in asdict(b).items() if k != "trial_id"} for b in first.bindings],
        "mask_adjustment_from_final_outcomes":False,"formal_user_acceptance":False}
    result["report_identity"]=fingerprint(result)
    return result


def frozen_mask_catalog(requests: Iterable[dict], qualifications: Iterable[dict]) -> dict:
    """Apply pre-test admissions plus all-seed family qualification, never tune a mask.

    Requests must bind the same target, physical/controller start, reference and
    timing as the qualified domain. Shared-family choice requires actual shared
    saved-start qualification; isolated family guards retain their separate IDs.
    """
    qualifications=list(qualifications)
    if len({q.get("family") for q in qualifications}) != len(qualifications):
        raise ValueError("Duplicate family qualification")
    proofs={q["family"]:q for q in qualifications}
    for q in qualifications:
        if (q.get("schema") != "Task10BPhase2FamilyQualification/v1"
                or q.get("report_identity") != fingerprint({k:v for k,v in q.items() if k != "report_identity"})):
            raise ValueError("Missing or changed family qualification report")
    rows=[];scenarios={};seen=set()
    for request in requests:
        required=("family","command_id","original_replica_id","ttc_s","start_index",
                  "start_state_sha256","reference_sha256","target_w","declared_mask_sha256","admitted_before_final_test")
        if any(k not in request for k in required) or not _hash(request["declared_mask_sha256"]):
            raise ValueError("A frozen exact pre-test request/mask identity is required")
        identity=fingerprint(request)
        if identity in seen:raise ValueError("Duplicate catalog request")
        seen.add(identity)
        proof=proofs.get(request["family"])
        binding={k:request[k] for k in required[1:8]}
        binding["target_w"]=tuple(request["target_w"])
        admitted=any(fingerprint(b) == fingerprint(binding) for b in proof["accepted_binding_keys"]) if proof else False
        legal=(request["admitted_before_final_test"] is True and proof is not None and proof["passed"] is True
               and request["declared_mask_sha256"] == proof["declared_mask_sha256"] and admitted)
        scenario=fingerprint({k:request[k] for k in ("command_id","ttc_s","start_index","start_state_sha256","target_w")})
        scenarios.setdefault(scenario,{"legal_families":set(),"shared_families":set()})
        if legal:
            scenarios[scenario]["legal_families"].add(request["family"])
            if proof["start_mode"] == "shared_jab_saved_state":scenarios[scenario]["shared_families"].add(request["family"])
        rows.append({"request_identity":identity,"request":request,"legal":legal,
                     "reason":"qualified_three_seed_frozen_domain" if legal else "unqualified_or_unsupported_exact_request",
                     "family_qualification_identity":None if proof is None else proof["report_identity"]})
    coverage=Counter(len(s["legal_families"]) for s in scenarios.values())
    return {"schema":"Task10BPhase2QualifiedCatalog/v1","requests":rows,"requested_scenarios":len(scenarios),
        "coverage_zero_to_four":{str(i):coverage[i] for i in range(5)},
        "shared_multi_choice_subset":[s for s,v in scenarios.items() if len(v["shared_families"]) >= 2],
        "shared_four_family_subset":[s for s,v in scenarios.items() if len(v["shared_families"]) == 4],
        "final_test_mask_tuning":False,"nearest_reference_substitution":False,
        "all_requested_scenarios_preserved":True,
        "failed_final_seed_semantics":"Keep the predeclared domain/mask and all requested rows; withhold the whole family's qualification, never silently reduce its final-test domain"}


def target_start_reachability(requests: Iterable[dict], qualifications: Iterable[dict]) -> dict:
    """Report any-bin reachability separately from actual common-TTC choice.

    Every supplied request, including unsupported and unqualified requests, stays
    in this reporting denominator. The caller supplies the complete predeclared
    requested domain; this function neither creates unrequested family cells nor
    changes a mask. Family guards never prove shared-start reachability, even if
    an input accidentally gives them identical state hashes. A failed family
    panel also does not prove that each exact cell is physically infeasible.
    """
    requests=list(requests);qualifications=list(qualifications)
    catalog=frozen_mask_catalog(requests,qualifications)
    proofs={q["family"]:q for q in qualifications}
    scenarios={}
    for row in catalog["requests"]:
        request=row["request"]
        if request["family"] not in FAMILIES or type(request["admitted_before_final_test"]) is not bool:
            raise ValueError("A known family and explicit pretest admission are required")
        # Reuse the strict binding validator without inventing an executed trial.
        TrialBinding(trial_id=row["request_identity"],**{k:request[k] for k in (
            "command_id","original_replica_id","ttc_s","start_index",
            "start_state_sha256","reference_sha256","target_w")})
        key={k:request[k] for k in ("command_id","original_replica_id",
                                  "start_index","start_state_sha256","target_w")}
        identity=fingerprint(key)
        scenario=scenarios.setdefault(identity,{"scenario_identity":identity,"binding":key,
            "families":{f:{"requested_bins_s":set(),"qualified_bins_s":set(),
                "shared_start_qualified_bins_s":set(),"requests":[]} for f in FAMILIES}})
        family=scenario["families"][request["family"]]
        family["requested_bins_s"].add(request["ttc_s"])
        proof=proofs.get(request["family"])
        if row["legal"]:
            status="qualified_exact_request"
            family["qualified_bins_s"].add(request["ttc_s"])
            if proof["start_mode"] == "shared_jab_saved_state":
                family["shared_start_qualified_bins_s"].add(request["ttc_s"])
        elif not request["admitted_before_final_test"]:
            status="not_admitted_before_final_test"
        elif proof is None:
            status="missing_family_qualification"
        elif proof["passed"] is not True:
            status="failed_final_family_qualification"
        elif proof["declared_mask_sha256"] != request["declared_mask_sha256"]:
            status="pretest_mask_mismatch"
        else:
            status="exact_request_not_qualified"
        family["requests"].append({"request_identity":row["request_identity"],
            "ttc_s":request["ttc_s"],"status":status,"legal":row["legal"],
            "family_qualification_identity":row["family_qualification_identity"],
            "measured_infeasibility":"not_inferred_from_qualification_or_admission"})
    rows=[]
    for scenario in scenarios.values():
        families=scenario["families"]
        for value in families.values():
            for field in ("requested_bins_s","qualified_bins_s","shared_start_qualified_bins_s"):
                value[field]=sorted(value[field])
            value["requested"]=bool(value["requested_bins_s"])
        qualified=[f for f in FAMILIES if families[f]["qualified_bins_s"]]
        shared=[f for f in FAMILIES if families[f]["shared_start_qualified_bins_s"]]
        def intersection(names,field):
            return sorted(set.intersection(*(set(families[f][field]) for f in names)))
        scenario.update({"reachable_families_at_any_qualified_bin":qualified,
            "shared_start_reachable_families_at_any_qualified_bin":shared,
            "all_four_common_qualified_bins_s":intersection(FAMILIES,"qualified_bins_s"),
            "all_four_common_shared_start_bins_s":intersection(FAMILIES,"shared_start_qualified_bins_s"),
            "pairwise_common_bins":[{"families":list(pair),
                "qualified_bins_s":intersection(pair,"qualified_bins_s"),
                "shared_start_qualified_bins_s":intersection(pair,"shared_start_qualified_bins_s")}
                for pair in itertools.combinations(FAMILIES,2)],
            "shared_state_proof":{"start_state_sha256":scenario["binding"]["start_state_sha256"],
                "required_start_mode":"shared_jab_saved_state",
                "qualified_family_reports":{f:proofs[f]["report_identity"] for f in shared},
                "four_family_same_saved_state_qualified":len(shared) == 4}})
        rows.append(scenario)
    any_counts=Counter(len(s["reachable_families_at_any_qualified_bin"]) for s in rows)
    shared_counts=Counter(len(s["shared_start_reachable_families_at_any_qualified_bin"]) for s in rows)
    result={"schema":"Task10BPhase2TargetStartReachability/v1","requested_scenarios":len(rows),
        "requested_cells":len(catalog["requests"]),"scenarios":rows,
        "coverage_any_qualified_bin_zero_to_four":{str(i):any_counts[i] for i in range(5)},
        "coverage_shared_start_any_qualified_bin_zero_to_four":{str(i):shared_counts[i] for i in range(5)},
        "shared_four_family_any_bin_subset":[s["scenario_identity"] for s in rows
            if len(s["shared_start_reachable_families_at_any_qualified_bin"]) == 4],
        "shared_four_family_common_ttc_subset":[s["scenario_identity"] for s in rows
            if s["all_four_common_shared_start_bins_s"]],
        "all_requested_scenarios_preserved":True,"unrequested_cells_inferred_legal":False,
        "final_test_mask_tuning":False,"nearest_reference_substitution":False,
        "measured_infeasibility_claimed":False,
        "scope":"Reporting over supplied predeclared requests only; no new qualification or publication. "
            "Any-bin family reachability does not establish a common TTC. Shared-state proof relies on "
            "the bound, audited family reports; matching family-guard labels or hashes alone is insufficient. "
            "Unsupported, missing and failed qualification remain distinct from measured cell infeasibility."}
    result["report_identity"]=fingerprint(result)
    return result
