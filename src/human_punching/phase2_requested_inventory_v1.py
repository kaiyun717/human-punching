"""Exact, non-qualifying inventory of frozen requested Phase2 scenarios.

An inventory proves completeness relative to explicitly bound declarations;
it cannot prove that a caller disclosed every experiment ever requested. Final
domain and historical exploratory requests remain separate, without replacing
the frozen final-test mask or inferring feasibility from execution disposition.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
import math
from pathlib import Path

from .phase2_domain import FAMILIES, fingerprint
from .phase2_target_catalog import owned_input
from .wbc.provenance import sha256_file, writable_path, write_json

VERSION = "task10b_phase2_requested_inventory_v1"
PLAN = "task10b_phase2_requested_inventory_plan_v1"
DECLARATION = "task10b_phase2_requested_scenarios_v1"
DISPOSITIONS = "task10b_phase2_request_dispositions_v1"
SCOPES = ("final_domain", "historical_exploratory")
STATES = ("unattempted", "unsupported", "generation_failed", "rejected", "timeout", "executed")
FIELDS = {"request_id", "origin_request_id", "family", "command_id", "original_replica_id", "split", "ttc_s",
          "start_index", "start_mode", "start_state_sha256", "start_protocol_sha256",
          "target_w", "target_input_sha256", "style", "checkpoint_sha256", "predicate_sha256",
          "reference_sha256"}
MASK_FIELDS = ("family", "command_id", "original_replica_id", "ttc_s", "start_index",
               "start_state_sha256", "reference_sha256", "target_w")


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _ids(values):
    if (not isinstance(values, list) or not values or
            any(not isinstance(v, str) or not v.strip() for v in values) or len(set(values)) != len(values)):
        raise ValueError("An explicit nonempty unique frozen request-ID inventory is required")
    return values


def validate_request(row):
    if not isinstance(row, dict) or set(row) != FIELDS:
        raise ValueError("Exact physical request fields required; unknown fields cannot be silently discarded")
    if (row["family"] not in FAMILIES or any(not isinstance(row[k], str) or not row[k].strip() for k in ("request_id", "origin_request_id"))
            or any(type(row[k]) is not int or row[k] < 0 for k in ("command_id", "original_replica_id", "start_index"))
            or row["split"] not in ("train", "validation", "test", "calibration", "d0_development")
            or row["start_mode"] not in ("family_guard", "shared_jab_saved_state", "declared_other")
            or not isinstance(row["style"], str) or not row["style"].strip()
            or type(row["ttc_s"]) not in (int, float) or not math.isfinite(row["ttc_s"]) or row["ttc_s"] <= 0
            or not isinstance(row["target_w"], list) or len(row["target_w"]) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in row["target_w"])
            or any(not _hash(row[k]) for k in ("start_state_sha256", "start_protocol_sha256",
                "target_input_sha256", "checkpoint_sha256", "predicate_sha256"))
            or row["reference_sha256"] is not None and not _hash(row["reference_sha256"])):
        raise ValueError("Finite exact target/start/timing/checkpoint/predicate identities required")
    # No split assignment is invented for historical D0 or calibration requests.
    if row["split"] in ("train", "validation", "test"):
        from .wbc.splits import assign_trajectory_groups
        if row["command_id"] == 3307 or assign_trajectory_groups([row["command_id"]])[row["command_id"]] != row["split"]:
            raise ValueError("Command-group split differs from immutable deterministic assignment")
    return fingerprint(row)


def build_inventory(expected_request_ids, declarations, dispositions):
    """Pure assembly: every declared ID and every disposition must match exactly."""
    expected = _ids(expected_request_ids)
    by_id = {}
    for declaration in declarations:
        if (declaration.get("schema") != DECLARATION or declaration.get("scope") not in SCOPES
                or set(declaration) != {"schema", "scope", "request_ids", "requests", "origin"}):
            raise ValueError("Versioned frozen scope-specific request declaration required")
        ids = _ids(declaration["request_ids"])
        rows = declaration["requests"]
        if not isinstance(rows, list) or len(rows) != len(ids):
            raise ValueError("Missing or extra declared request rows")
        found = []
        for row in rows:
            identity = validate_request(row); key = row["request_id"]; found.append(key)
            if declaration["scope"] == "final_domain" and (row["split"] != "test"
                    or row["original_replica_id"] != 2 or row["start_index"] not in range(4)):
                raise ValueError("Final domain requires test groups, original replica2 and declared starts0..3")
            if key in by_id:
                raise ValueError("Request IDs cannot be merged across histories, checkpoints, or scopes")
            by_id[key] = dict(request=json.loads(json.dumps(row)), request_identity=identity, scope=declaration["scope"])
        if set(found) != set(ids) or len(set(found)) != len(found):
            raise ValueError("Missing, extra, or duplicate declared request IDs")
    if set(by_id) != set(expected):
        raise ValueError("Declaration union differs from the complete frozen request-ID inventory")
    if dispositions.get("schema") != DISPOSITIONS or set(dispositions) != {"schema", "rows"}:
        raise ValueError("Exact request disposition ledger required")
    observed = {}
    for row in dispositions["rows"]:
        if (set(row) != {"request_id", "request_identity", "status", "reason", "evidence"}
                or row["request_id"] not in by_id or row["request_id"] in observed
                or row["status"] not in STATES or row["request_identity"] != by_id[row["request_id"]]["request_identity"]
                or not isinstance(row["reason"], str) or not row["reason"].strip()
                or not isinstance(row["evidence"], list)
                or any(set(b) != {"path", "sha256"} or not isinstance(b["path"], str) or not b["path"]
                       or not _hash(b["sha256"]) for b in row["evidence"])):
            raise ValueError("Disposition must preserve exact request identity, reason, and evidence bindings")
        if row["status"] != "unattempted" and not row["evidence"]:
            raise ValueError("Reported disposition needs retained evidence; absence is not infeasibility")
        if row["status"] == "executed" and by_id[row["request_id"]]["request"]["reference_sha256"] is None:
            raise ValueError("Executed requests require an exact reference identity")
        observed[row["request_id"]] = json.loads(json.dumps(row))
    if set(observed) != set(expected):
        raise ValueError("Missing or extra disposition IDs; unattempted rows must be explicit")
    rows = [dict(**by_id[key], disposition=observed[key], legal=None, qualification="not_evaluated") for key in expected]
    return dict(schema=VERSION, request_ids=expected.copy(), requests=rows,
        inventory_identity=fingerprint(rows), count=len(rows),
        scopes={scope:dict(count=sum(r["scope"] == scope for r in rows),
            dispositions=dict(Counter(r["disposition"]["status"] for r in rows if r["scope"] == scope))) for scope in SCOPES},
        completeness="Exact union of explicitly bound declarations; original source completeness must also be verified",
        original_source_completeness_verified=False,
        physical_feasibility_inferred=False, qualification=False, native_execution=False,
        shared_start_choice_established=False, final_mask_replaced=False)


def validate_final_mask(inventory, mask, expected_sha256):
    """Bind final-domain cells to the existing mask; historical cells never enter it.

    Different seed checkpoint requests may map to one mask cell, but requests
    themselves remain distinct in the inventory. This grants no legal bits.
    """
    if (inventory.get("schema") != VERSION or inventory.get("inventory_identity") != fingerprint(inventory.get("requests"))
            or not _hash(expected_sha256) or mask.get("schema") != "task10b_phase2_declared_evaluation_mask_v1"):
        raise ValueError("Exact independently frozen final mask binding required")
    declared = {}; semantic = {}
    for item in inventory["requests"]:
        if item["scope"] != "final_domain":
            continue
        row = item["request"]
        if row["reference_sha256"] is None:
            raise ValueError("Final-mask cells must have explicit reference identities, even when rejected")
        key = fingerprint({k:row[k] for k in MASK_FIELDS})
        meaning = fingerprint({k:v for k,v in row.items() if k not in ("request_id", "origin_request_id", "checkpoint_sha256")})
        if semantic.setdefault(key, meaning) != meaning:
            raise ValueError("One final-mask cell cannot merge incompatible style, target input, protocol or predicate")
        declared.setdefault(key, []).append(row["request_id"])
    actual = []
    for row in mask.get("requests", []):
        if type(row.get("admitted")) is not bool:
            raise ValueError("Keep original predeclared mask admission flags")
        from .wbc.splits import assign_trajectory_groups
        if assign_trajectory_groups([row["command_id"]])[row["command_id"]] == "test":
            actual.append(fingerprint({k:row[k] for k in MASK_FIELDS}))
    if not declared or len(actual) != len(set(actual)) or set(actual) != set(declared):
        raise ValueError("Final mask is missing or adds cells relative to frozen final-domain requests")
    return dict(mask_sha256=expected_sha256, mask_cells=len(actual), request_ids_by_cell=declared,
                historical_requests_excluded=True, comparison_scope="deterministic_test_subset_of_unchanged_full_mask",
                mask_admission_unchanged=True, qualification=False)


def validate_origins(declarations, read):
    """Close completeness against original declarations, not a copied ID list."""
    receipts=[]
    for declaration in declarations:
        origin=declaration["origin"]; rows=declaration["requests"]
        if not isinstance(origin,dict) or "binding" not in origin:
            raise ValueError("Each derived declaration must bind its original requested source")
        source=read(origin["binding"])
        if declaration["scope"] == "final_domain":
            if set(origin)!={"kind","binding"} or origin["kind"]!="panel_declaration":
                raise ValueError("Final domain requires its original full physical PanelDeclaration")
            from .phase2_qualification import PanelDeclaration, TrialBinding
            from dataclasses import asdict
            panel=PanelDeclaration(**{**source,"bindings":tuple(TrialBinding(**b) for b in source["bindings"])})
            if panel.split!="test":raise ValueError("Final inventory origin must be a 192-attempt untouched test declaration")
            expected={b.trial_id:asdict(b) for b in panel.bindings}
            actual=[r["origin_request_id"] for r in rows]
            if len(actual)!=len(set(actual)) or set(actual)!=set(expected):
                raise ValueError("Final inventory omits or adds original physical panel trials")
            for row in rows:
                original=expected[row["origin_request_id"]]
                if (any(fingerprint(row[k])!=fingerprint(original[k]) for k in MASK_FIELDS if k!="family")
                        or any(row[k]!=getattr(panel,k) for k in ("family","split","checkpoint_sha256","predicate_sha256","start_mode"))):
                    raise ValueError("Final request changes original panel target/start/timing/checkpoint/predicate")
            receipts.append(dict(scope=declaration["scope"],origin=origin,original_count=len(expected),panel_identity=panel.identity))
        else:
            if (set(origin)!={"kind","binding","rows_key","id_key"} or origin["kind"]!="request_list"
                    or not isinstance(origin["rows_key"],str) or not isinstance(origin["id_key"],str)):
                raise ValueError("Historical inventory requires an explicit original request-list selector")
            original=source[origin["rows_key"]]
            if not isinstance(original,list) or any(not isinstance(r,dict) for r in original):
                raise ValueError("Original historical source must contain explicit requested rows")
            expected={r[origin["id_key"]]:r for r in original}
            actual=[r["origin_request_id"] for r in rows]
            if len(expected)!=len(original) or len(actual)!=len(set(actual)) or set(actual)!=set(expected):
                raise ValueError("Historical inventory omits or adds original requested scenarios")
            for row in rows:
                old=expected[row["origin_request_id"]]
                if any(k not in old or fingerprint(row[k])!=fingerprint(old[k]) for k in MASK_FIELDS):
                    raise ValueError("Historical inventory changes the original physical request")
            receipts.append(dict(scope=declaration["scope"],origin=origin,original_count=len(original)))
    return receipts


def inventory_from_plan(plan_path):
    path = Path(plan_path).resolve(strict=True)
    plan = json.loads(owned_input(dict(path=str(path), sha256=sha256_file(path))).read_text())
    if plan.get("schema") != PLAN or set(plan) != {"schema", "request_ids", "declarations", "dispositions", "final_mask"}:
        raise ValueError("Exact inventory plan schema required")
    bindings = []
    def read(binding):
        actual = owned_input(binding); bindings.append(dict(path=str(actual), sha256=sha256_file(actual)))
        return json.loads(actual.read_text())
    declarations=[read(b) for b in plan["declarations"]]
    result = build_inventory(plan["request_ids"], declarations, read(plan["dispositions"]))
    result["original_sources"]=validate_origins(declarations,read)
    result["original_source_completeness_verified"]=True
    for row in result["requests"]:
        for binding in row["disposition"]["evidence"]:
            actual = owned_input(binding); bindings.append(dict(path=str(actual), sha256=sha256_file(actual)))
    if plan["final_mask"] is not None:
        result["final_mask"] = validate_final_mask(result, read(plan["final_mask"]), plan["final_mask"]["sha256"])
    elif result["scopes"]["final_domain"]["count"]:
        raise ValueError("Final-domain inventory requires its immutable mask; exploratory-only inventory does not")
    result["provenance"] = dict(plan=dict(path=str(path), sha256=sha256_file(path)), inputs=bindings,
                                source_sha256=sha256_file(__file__))
    return result


def verify_inventory(binding):
    path = owned_input(binding); stored = json.loads(path.read_text())
    if stored.get("schema") != VERSION or stored.get("provenance", {}).get("source_sha256") != sha256_file(__file__):
        raise ValueError("Inventory from the exact versioned factory required")
    plan = owned_input(stored["provenance"]["plan"])
    recomputed = inventory_from_plan(plan)
    if fingerprint(stored) != fingerprint(recomputed):
        raise ValueError("Stored inventory differs from its immutable requests and evidence")
    return stored


def create_inventory(plan_path, output):
    result = inventory_from_plan(plan_path)
    folder = writable_path(output); folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "inventory.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = create_inventory(args.plan, args.output)
    print(json.dumps(dict(schema=VERSION, count=result["count"], qualification=False)))


if __name__ == "__main__":
    main()
