"""Mandatory clock-aware review bridge for direct isolated expanded panels.

Original trace/manual adjudication and qualification math remain unchanged.
Every continuation ancestor must carry the same verified clock metadata, and
the saved clock claims are independently recomputed before review credit.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

from . import phase2_review as parent
from . import phase2_expanded_entry_v2 as entry
from .phase2_domain import fingerprint
from .phase2_qualification import PanelDeclaration, TrialBinding
from .phase2_target_catalog import owned_input
from .wbc.provenance import sha256_file, writable_path, write_json

PANEL = "task10b_phase2_direct_expanded_panel_review_configuration_v2"
RECEIPT = "task10b_phase2_direct_expanded_panel_review_receipt_v2"
FAMILY_PLAN = "task10b_phase2_direct_expanded_family_review_plan_v2"
FAMILY_RECEIPT = "task10b_phase2_direct_expanded_family_review_receipt_v2"
CATALOG_PLAN = "task10b_phase2_direct_expanded_catalog_review_plan_v2"
LAUNCH = "task10b_phase2_direct_expanded_native_launch_v2"


def binding(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256_file(path))


def read(record):
    path = owned_input(record)
    return path, json.loads(path.read_text())


def _same(actual, expected):
    return owned_input(actual) == owned_input(expected) and actual["sha256"] == expected["sha256"]


def _declaration_identity(value):
    # The native evaluator first-writes asdict(declaration), including defaults;
    # source JSON may omit those defaults or use different whitespace.
    return PanelDeclaration(**{**value, "bindings":tuple(TrialBinding(**row)
        for row in value["bindings"])}).identity


def verify_event_evidence(run):
    """Saved metadata/arrays only; never calls the native execute validator."""
    run = Path(run).resolve(strict=True)
    with entry.identity_scope():
        ancestry = parent.verified_run_ancestry(run)
    roots = [Path(p).resolve(strict=True) for p in ancestry["run_directories"]]
    if len(set(roots)) != len(roots) or run not in roots:
        raise ValueError("Exact acyclic execution ancestry required")
    expected_sources = entry.source_bindings(); results = []
    for root in roots:
        side_path, side = read(binding(root / entry.SIDECAR))
        _, launch = read(binding(root / "launch.json"))
        _, execution = read(binding(root / "execution.json"))
        _, declaration = read(binding(root / "declaration.json"))
        _, identity = read(binding(root / "execution_identity.json"))
        if (side.get("schema") != entry.VERSION or side.get("passed") is not True
                or side.get("event_clock_version") != entry.clocks.VERSION
                or side.get("all_native_batches_required") is not True
                or declaration.get("start_mode", "family_guard") != "family_guard"
                or launch.get("schema") != LAUNCH
                or execution.get("schema") != "task10b_phase2_expanded_execution_v1"):
            raise ValueError("Passing versioned direct isolated clock evidence is mandatory")
        for record in (launch, execution):
            if (record.get("direct_consumer_version") != entry.VERSION
                    or record.get("event_clock_version") != entry.clocks.VERSION
                    or record.get("event_clock_review_receipt_required") != entry.SIDECAR):
                raise ValueError("Native first-write consumer identity is missing or changed")
        if not _same(side["execution"], binding(root / "execution.json")):
            raise ValueError("Clock sidecar must bind this exact execution file")
        _, source_declaration = read(side["declaration"])
        if _declaration_identity(source_declaration) != _declaration_identity(declaration):
            raise ValueError("Clock declaration differs from the executed declaration")
        read(side["manifest"])
        frozen = identity.get("frozen", {})
        expected_metadata = dict(version=entry.VERSION,event_clock_version=entry.clocks.VERSION,
            source_sha256={k:v["sha256"] for k,v in expected_sources.items()})
        if (frozen.get("event_metadata") != expected_metadata or identity.get("identity") != fingerprint(frozen)
                or any(side[k]["sha256"] != frozen.get("input_sha256", {}).get(k) for k in ("declaration", "manifest"))
                or any(owned_input(side[k]) != Path(identity.get("input_paths", {}).get(k, "")).resolve(strict=True)
                       for k in ("declaration", "manifest"))
                or launch.get("execution_identity") != identity["identity"]
                or execution.get("execution_identity") != identity["identity"]):
            raise ValueError("Clock metadata/input identities differ from the frozen native execution")
        for sources in (side.get("sources", {}), launch.get("event_metadata_sources", {})):
            if set(sources) != set(expected_sources) or any(not _same(sources[k], expected_sources[k]) for k in expected_sources):
                raise ValueError("Event-clock source bindings differ from the reviewed implementation")
        started = execution.get("native_attempts_started")
        if type(started) is not int or started < 0 or side.get("native_attempts_started") != started:
            raise ValueError("Exact nonnegative durable attempt count required")
        batches = sorted(p for p in root.glob("batch_*") if p.is_dir())
        saved = side.get("batch_audits")
        if (not isinstance(saved, list) or any(not isinstance(a, dict) for a in saved)
                or [a.get("batch") for a in saved] != [p.name for p in batches]):
            raise ValueError("Clock receipt omits, duplicates or adds native batches")
        audits = []
        for batch, reported in zip(batches, saved, strict=True):
            current = dict(batch=batch.name, **entry.clock_audit.audit_batch(batch))
            if current.get("passed") is not True or fingerprint(current) != fingerprint(reported):
                raise ValueError("Stored clock audit differs from independently recomputed batch evidence")
            audits.append(current)
        if sum(a.get("trials", 0) for a in audits) != started:
            raise ValueError("Every started native attempt must retain its audited clock evidence")
        results.append(dict(run=str(root), sidecar=binding(side_path), execution=binding(root/"execution.json"),
                            native_attempts_started=started, batches=len(audits), passed=True))
    return dict(schema="task10b_phase2_direct_expanded_clock_consumer_review_v2", passed=True,
                runs=results, source_sha256=sha256_file(__file__), qualification=False, new_physics_steps=0)


def _typed_adjudicate(original_path,run):
    # Explicit typed consumer bridge: native v2 bytes remain unchanged and old
    # consumers reject them. Only this privately loaded original adjudicator
    # receives a v1-shaped view after the mandatory v2 proof succeeds.
    spec=importlib.util.spec_from_file_location("human_punching._phase2_direct_review_math_v2",parent.__file__)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    original_reader=legacy.read_binding
    def typed_read(record):
        actual,value=original_reader(record)
        if actual == run/"launch.json":
            if value.get("schema") != LAUNCH:raise ValueError("Only explicitly verified v2 launch may enter the typed bridge")
            value={**value,"schema":"task10b_phase2_expanded_native_launch_v1"}
        return actual,value
    legacy.read_binding=typed_read
    return legacy.adjudicate(original_path)


def adjudicate(configuration_path):
    path, config = read(binding(configuration_path))
    if config.get("schema") != PANEL or set(config) != {"schema", "original_review_configuration", "event_clock_entry"}:
        raise ValueError("Exact versioned clock-aware review configuration required")
    original_path, original = read(config["original_review_configuration"])
    run = Path(original["run"]).resolve(strict=True)
    if not _same(config["event_clock_entry"], binding(run/entry.SIDECAR)):
        raise ValueError("Review must bind this exact native clock sidecar")
    clocks = verify_event_evidence(run)
    report,inherited = _typed_adjudicate(original_path,run)
    return report, dict(schema=RECEIPT, configuration=binding(path), source_sha256=sha256_file(__file__),
        original_review_receipt=inherited, event_clock_evidence=clocks,
        typed_launch_view=dict(native_schema=LAUNCH,consumer_schema="task10b_phase2_expanded_native_launch_v1",
            native=binding(run/"launch.json"),source=binding(parent.__file__),disk_bytes_changed=False),
        qualification_math_changed=False, formal_acceptance=False)


def review_panel(configuration_path, output):
    report, receipt = adjudicate(configuration_path)
    folder = writable_path(output); folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/"qualification.json", report)
    receipt["qualification"] = binding(folder/"qualification.json")
    write_json(folder/"review.json", receipt)
    return receipt


def reviewed_panel(record):
    _, receipt = read(record)
    if receipt.get("schema") != RECEIPT or receipt.get("source_sha256") != sha256_file(__file__):
        raise ValueError("Clock-aware v2 review receipt required; an old generic review cannot substitute")
    config, _ = read(receipt["configuration"]); _, saved = read(receipt["qualification"])
    current, recomputed = adjudicate(config)
    if fingerprint(current) != fingerprint(saved) or any(fingerprint(receipt.get(k)) != fingerprint(v) for k,v in recomputed.items()):
        raise ValueError("Stored clock-aware qualification differs from immutable evidence")
    return saved


def review_family(plan_path, output):
    path, plan = read(binding(plan_path))
    if plan.get("schema") != FAMILY_PLAN or set(plan) != {"schema", "family", "panels"} or len(plan["panels"]) != 3:
        raise ValueError("Exactly three independently clock-reviewed final panels required")
    result = parent.qualify_family([reviewed_panel(p) for p in plan["panels"]], plan["family"])
    folder = writable_path(output); folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/"qualification.json", result)
    write_json(folder/"review.json", dict(schema=FAMILY_RECEIPT, plan=binding(path),
        qualification=binding(folder/"qualification.json"), source_sha256=sha256_file(__file__)))
    return result


def publish_catalog(plan_path, output):
    """Original catalog math, with mandatory v2 panel evidence and full inventory."""
    from .phase2_requested_inventory_v1 import verify_inventory, validate_final_mask
    from .wbc.splits import assign_trajectory_groups
    path, plan = read(binding(plan_path))
    if (plan.get("schema") != CATALOG_PLAN or set(plan) != {"schema", "families", "requested_domain", "requested_inventory"}
            or len(plan["families"]) != 4):
        raise ValueError("Four clock-reviewed families, frozen final mask and complete request inventory required")
    qualifications = []
    for record in plan["families"]:
        _, receipt = read(record)
        if receipt.get("schema") != FAMILY_RECEIPT or receipt.get("source_sha256") != sha256_file(__file__):
            raise ValueError("Only clock-aware family receipts may receive catalog credit")
        _, frozen = read(receipt["plan"]); _, saved = read(receipt["qualification"])
        if frozen.get("schema") != FAMILY_PLAN or len(frozen.get("panels", [])) != 3:
            raise ValueError("Exactly three clock-reviewed learned final panels required")
        current = parent.qualify_family([reviewed_panel(p) for p in frozen["panels"]], frozen["family"])
        if fingerprint(current) != fingerprint(saved):
            raise ValueError("Family result differs from the three immutable seed panels")
        qualifications.append(saved)
    _, mask = read(plan["requested_domain"]); digest = plan["requested_domain"]["sha256"]
    inventory = verify_inventory(plan["requested_inventory"])
    inventory_check = validate_final_mask(inventory, mask, digest)
    if inventory.get("final_mask", {}).get("mask_sha256") != digest or any(q["declared_mask_sha256"] != digest for q in qualifications):
        raise ValueError("Final mask cannot change between declaration, inventory and qualified panels")
    requests=[]; test=[]
    for row in mask["requests"]:
        request={k:row[k] for k in ("family","command_id","original_replica_id","ttc_s","start_index",
                                   "start_state_sha256","reference_sha256","target_w")}
        request.update(declared_mask_sha256=digest, admitted_before_final_test=row["admitted"])
        requests.append(request)
        if assign_trajectory_groups([row["command_id"]])[row["command_id"]] == "test":test.append(request)
    catalog=parent.frozen_mask_catalog(requests, qualifications); reachability=parent.target_start_reachability(requests, qualifications)
    folder=writable_path(output); folder.mkdir(parents=True, exist_ok=False)
    for name,value in (("catalog",catalog),("reachability",reachability),
        ("choice_claims",parent.predeclared_choice_claims(mask,reachability,plan["requested_domain"])),
        ("final_test_catalog",parent.frozen_mask_catalog(test,qualifications)),
        ("final_test_reachability",parent.target_start_reachability(test,qualifications))):write_json(folder/(name+".json"),value)
    write_json(folder/"requested_inventory.json",inventory)
    write_json(folder/"provenance.json",dict(schema="task10b_phase2_direct_expanded_catalog_review_v2",plan=binding(path),
        source_sha256=sha256_file(__file__), final_inventory_check=inventory_check,
        historical_exploratory_coverage_qualified=False, qualification_math_changed=False,
        no_policy_promotion=True, formal_acceptance=False))
    return catalog


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("panel","family","catalog"));parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    result={"panel":review_panel,"family":review_family,"catalog":publish_catalog}[args.mode](args.plan,args.output)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__ == "__main__":main()
