"""Append-only continuation of never-attempted frozen evaluation requests.

Completed, partial, timed-out, rejected and generation-failed requests are never
replayed. A child carries their original outcomes into the same full declared
denominator. Each child is a separately accounted bounded allocation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .phase2_domain import fingerprint
from .wbc.provenance import PROJECT_ROOT,sha256_file,writable_path,write_json,runtime_record

CONFIG="task10b_phase2_evaluation_continuation_v1"
IDENTITY="task10b_phase2_frozen_execution_identity_v1"
SOURCES=("phase2_continuation","phase2_expanded_evaluation","phase2_shared_evaluation","phase2_learning",
    "phase2_domain","phase2_qualification","phase2_launch","phase2_native_state","phase2_sampling",
    "phase2_deadline","phase2_cache","phase2_phases","functional_recovery")
STATUSES=frozenset(("executed","timeout","generation_failure","rejected","unattempted"))


def _owned(path):
    value=Path(path).resolve(strict=True)
    if not value.is_relative_to(PROJECT_ROOT.resolve()) or not value.is_file():
        raise ValueError("Continuation evidence must be an owned regular file")
    return value


def _read_bound(item):
    if not isinstance(item,dict) or set(item)!={"path","sha256"}:
        raise ValueError("Exact immutable continuation evidence path/SHA256 required")
    path=_owned(item["path"])
    if sha256_file(path)!=item["sha256"]:raise ValueError("Continuation parent evidence changed")
    return path,json.loads(path.read_text())


def execution_identity(declaration,inputs,*,release,protocol_version,maximum_worlds):
    module_root=Path(__file__).resolve().parent
    sources={name:sha256_file(module_root/(name+".py")) for name in SOURCES}
    release=Path(release).resolve(strict=True)
    manifest=_owned(release/"manifest.json")
    frozen=dict(declaration_identity=declaration.identity,family=declaration.family,seed=declaration.seed,
        split=declaration.split,input_sha256={name:sha256_file(_owned(path)) for name,path in inputs.items()},
        release_manifest_sha256=sha256_file(manifest),protocol_version=protocol_version,
        maximum_worlds=maximum_worlds,sources=sources,runtime=runtime_record())
    return dict(schema=IDENTITY,identity=fingerprint(frozen),frozen=frozen,
        input_paths={name:str(_owned(path)) for name,path in inputs.items()},
        release=str(release),slurm_job_id=os.environ.get("SLURM_JOB_ID"))


def load_continuation(path,current_identity,declaration):
    """Validate exact never-attempted remainder before any new native attempt."""
    config_path=_owned(path);config=json.loads(config_path.read_text())
    if config.get("schema")!=CONFIG:raise ValueError("Explicit frozen evaluation continuation configuration required")
    identity_path,parent_identity=_read_bound(config["parent_execution_identity"])
    execution_path,parent_execution=_read_bound(config["parent_execution"])
    outcomes_path,outcomes=_read_bound(config["parent_outcomes"])
    parent_run=Path(config["parent_run"]).resolve(strict=True)
    if (not parent_run.is_relative_to(PROJECT_ROOT.resolve()) or not parent_run.is_dir()
            or identity_path!=parent_run/"execution_identity.json"
            or parent_identity.get("schema")!=IDENTITY or parent_identity.get("identity")!=current_identity["identity"]
            or parent_identity.get("frozen")!=current_identity["frozen"]
            or parent_identity.get("identity")!=fingerprint(parent_identity.get("frozen"))):
        raise ValueError("Continuation changes frozen source/runtime/recipe/domain/weights/mask/start/declaration")
    if (parent_execution.get("schema")!="task10b_phase2_expanded_execution_v1"
            or parent_execution.get("status") not in ("failed","executed_pending_audit_and_video","recovered_external_termination")
            or parent_execution.get("planned_trials")!=len(declaration.bindings)
            or parent_execution.get("source_application_run",parent_execution.get("application_run"))!=str(parent_run)
            or parent_execution.get("execution_identity")!=current_identity["identity"]
            or parent_execution.get("outcomes_sha256")!=sha256_file(outcomes_path)):
        raise ValueError("A terminal complete parent ledger and matching immutable execution identity are required")
    expected={binding.trial_id for binding in declaration.bindings}
    rows={row["trial_id"]:row for row in outcomes}
    if len(rows)!=len(outcomes) or set(rows)!=expected or any(row.get("status") not in STATUSES for row in outcomes):
        raise ValueError("Parent must retain exactly the full declared 96/192-request ledger")
    candidates={trial for trial,row in rows.items() if row["status"]=="unattempted"}
    if any(rows[trial].get("physical_execution_attempted") is True or rows[trial].get("execution_complete") is True for trial in candidates):
        raise ValueError("An attempted physical request cannot be relabeled unattempted")
    for intent_path in sorted(parent_run.glob("native_attempt_intent_*.json")):
        intent=json.loads(intent_path.read_text())
        if intent.get("execution_identity")!=current_identity["identity"] or not set(intent["trial_ids"])<=expected:
            raise ValueError("Native attempt intent differs from the frozen parent declaration")
        if candidates.intersection(intent["trial_ids"]):
            raise ValueError("Parent marks a durably intended physical attempt as unattempted; recover the interrupted ledger first")
    ancestor_path=parent_run/"continuation.json"
    if ancestor_path.exists():
        ancestor=json.loads(ancestor_path.read_text())
        prior_path,prior_rows=_read_bound(dict(path=ancestor["parent_outcomes"],sha256=ancestor["parent_outcomes_sha256"]))
        for prior in prior_rows:
            if prior["status"]!="unattempted" and rows[prior["trial_id"]]!=prior:
                raise ValueError("A continuation may not replace or drop an ancestor's retained attempted outcome")
    requested=config.get("trial_ids")
    if (not isinstance(requested,list) or len(requested)!=len(set(requested))
            or set(requested)!=candidates or not candidates):
        raise ValueError("Continue exactly all and only never-attempted parent requests; no selected retries")
    return dict(configuration_path=str(config_path),configuration_sha256=sha256_file(config_path),
        parent_run=str(parent_run),parent_execution=str(execution_path),parent_execution_sha256=sha256_file(execution_path),
        parent_outcomes=str(outcomes_path),parent_outcomes_sha256=sha256_file(outcomes_path),
        trial_ids=requested,outcomes=outcomes,
        logical_origin_run=parent_execution.get("logical_origin_run",str(parent_run)),
        inherited_attempted_ids=sorted(expected-candidates),same_frozen_execution_identity=current_identity["identity"])


def claim_test_session(folder,declaration,identity,continuation=None):
    """One linear continuation chain per family/seed; never fork a parent."""
    if declaration.split!="test":raise ValueError("Exclusive claims apply to final tests only")
    key=fingerprint(dict(scope="task10b_phase2_untouched_final_test_v1",family=declaration.family,seed=declaration.seed))
    claim=writable_path(PROJECT_ROOT/"outputs/task10b/p2_final_test_claims"/key)
    origin=continuation["logical_origin_run"] if continuation else str(folder)
    newly_created=not claim.exists()
    if not newly_created:
        if continuation is None:raise ValueError("Final family/seed test already has an exclusive claim; use only its unattempted continuation")
        prior=json.loads((claim/"claim.json").read_text())
        if (prior.get("execution_identity")!=identity["identity"] or prior.get("declaration_identity")!=declaration.identity
                or prior.get("logical_origin_run")!=origin):
            raise ValueError("Final test claim belongs to a different frozen execution or origin")
        parent=continuation["parent_run"]
        claimed_parent=prior.get("run")==parent or any(json.loads(path.read_text()).get("run")==parent
            for path in (claim/"continuations").glob("*/claim.json"))
        if not claimed_parent:
            raise ValueError("Parent never acquired an exclusive test-session claim; continue from its last claimed terminal ancestor")
    else:
        claim.mkdir(parents=True,exist_ok=False)
        write_json(claim/"claim.json",dict(declaration_identity=declaration.identity,execution_identity=identity["identity"],
            logical_origin_run=origin,run=str(folder),phase="before_native_initialization",automatic_retry=False))
    session=claim
    if continuation is not None:
        child_key=fingerprint({key:continuation[key] for key in
            ("parent_run","parent_execution_sha256","parent_outcomes_sha256")})
        session=claim/"continuations"/child_key;session.mkdir(parents=True,exist_ok=False)
        write_json(session/"claim.json",dict(run=str(folder),parent_run=continuation["parent_run"],
            parent_execution_sha256=continuation["parent_execution_sha256"],
            parent_outcomes_sha256=continuation["parent_outcomes_sha256"],
            exact_never_attempted_trial_ids=continuation["trial_ids"],execution_identity=identity["identity"],
            logical_origin_run=origin,no_attempted_request_replayed=True))
    result=dict(path=str(claim),session_path=str(session),logical_origin_run=origin,
        continuation=continuation is not None,execution_identity=identity["identity"])
    write_json(Path(folder)/"final_test_claim.json",result)
    return result


def verified_run_ancestry(run,*,maximum_depth=32):
    """Return only hash-verified same-execution parent directories for auditing."""
    from .phase2_qualification import PanelDeclaration,TrialBinding
    if type(maximum_depth) is not int or not 1<=maximum_depth<=32:
        raise ValueError("Bounded continuation ancestry depth required")
    roots=[];seen=set();current=Path(run).resolve(strict=True);expected=None
    for _ in range(maximum_depth):
        if current in seen or not current.is_relative_to(PROJECT_ROOT.resolve()) or not current.is_dir():
            raise ValueError("Cyclic or unowned continuation ancestry")
        seen.add(current);identity=json.loads(_owned(current/"execution_identity.json").read_text())
        if (identity.get("schema")!=IDENTITY or identity.get("identity")!=fingerprint(identity.get("frozen"))
                or (expected is not None and identity["identity"]!=expected)):
            raise ValueError("Continuation ancestry changes frozen execution identity")
        expected=identity["identity"];roots.append(str(current))
        record_path=current/"continuation.json"
        if not record_path.exists():
            return dict(schema="task10b_phase2_verified_run_ancestry_v1",execution_identity=expected,
                run_directories=roots,depth=len(roots),same_declared_denominator=True,no_attempt_replayed=True)
        record=json.loads(record_path.read_text())
        configuration=_owned(record["configuration_path"])
        if sha256_file(configuration)!=record["configuration_sha256"]:
            raise ValueError("Continuation ancestry configuration changed")
        declaration_path=_owned(identity["input_paths"]["declaration"])
        if sha256_file(declaration_path)!=identity["frozen"]["input_sha256"]["declaration"]:
            raise ValueError("Continuation ancestry declaration changed")
        raw=json.loads(declaration_path.read_text())
        declaration=PanelDeclaration(**{**raw,"bindings":tuple(TrialBinding(**b) for b in raw["bindings"])})
        verified=load_continuation(configuration,identity,declaration)
        if any(record.get(key)!=value for key,value in verified.items() if key!="outcomes"):
            raise ValueError("Continuation ancestry receipt differs from its immutable parent evidence")
        current=Path(verified["parent_run"]).resolve(strict=True)
    raise ValueError("Continuation ancestry exceeds the declared bounded depth")


def recover_interrupted_parent(parent_run,terminal_accounting,output):
    """Read-only recovery into a new artifact after a terminal external stop.

    Last durable ledgers retain known results. Every later attempt intent is an
    unknown failed physical attempt, even when its trace disappeared. Requests
    without an intent remain unattempted. No unknown attempt is retried.
    """
    parent=Path(parent_run).resolve(strict=True);identity_path=_owned(parent/"execution_identity.json")
    identity=json.loads(identity_path.read_text());terminal_path=_owned(terminal_accounting)
    terminal=json.loads(terminal_path.read_text());packet=_owned(terminal_path.parent/"packet.json")
    state=terminal.get("state","").split()
    if (identity.get("schema")!=IDENTITY or identity.get("identity")!=fingerprint(identity.get("frozen"))
            or terminal.get("schema")!="task10b_phase2_terminal_accounting_v1"
            or not state or state[0] not in ("COMPLETED","FAILED","TIMEOUT","CANCELLED","PREEMPTED","NODE_FAIL","OUT_OF_MEMORY","BOOT_FAIL","DEADLINE","REVOKED")
            or terminal.get("packet_sha256")!=sha256_file(packet) or terminal.get("job_id")!=identity.get("slurm_job_id")
            or not parent.is_relative_to(terminal_path.parent) or (parent/"execution.json").exists()):
        raise ValueError("Interrupted recovery requires the same terminal accounted packet and an unfinished application ledger")
    declaration_path=_owned(identity["input_paths"]["declaration"])
    if sha256_file(declaration_path)!=identity["frozen"]["input_sha256"]["declaration"]:
        raise ValueError("Interrupted parent declaration changed")
    declaration=json.loads(declaration_path.read_text())
    ids=[row["trial_id"] for row in declaration["bindings"]]
    if len(ids)!=len(set(ids)) or len(ids) not in (96,192):raise ValueError("Complete frozen declaration required for recovery")
    ledgers=sorted(parent.glob("ledger_*.json"),key=lambda path:path.stat().st_mtime_ns)
    ancestor_path=parent/"continuation.json";ancestor={};inherited=None
    if ancestor_path.exists():
        ancestor=json.loads(ancestor_path.read_text())
        if ancestor.get("same_frozen_execution_identity")!=identity["identity"]:
            raise ValueError("Interrupted continuation ancestry differs from frozen execution")
        _,inherited=_read_bound(dict(path=ancestor["parent_outcomes"],sha256=ancestor["parent_outcomes_sha256"]))
    ledger=(json.loads(ledgers[-1].read_text()) if ledgers else inherited if inherited is not None else
        [dict(trial_id=trial,status="unattempted",physical_execution_attempted=False,
            reason="no_durable_native_attempt_intent") for trial in ids])
    rows={row["trial_id"]:row for row in ledger}
    if len(rows)!=len(ledger) or set(rows)!=set(ids):raise ValueError("Durable partial ledger does not cover the full declaration")
    if inherited is not None and any(row["status"]!="unattempted" and rows[row["trial_id"]]!=row for row in inherited):
        raise ValueError("Interrupted continuation changed an ancestor's retained attempted outcome")
    intents=[]
    for path in sorted(parent.glob("native_attempt_intent_*.json")):
        intent=json.loads(path.read_text());intents.append(dict(path=str(path),sha256=sha256_file(path)))
        if intent.get("execution_identity")!=identity["identity"] or not set(intent["trial_ids"])<=set(ids):
            raise ValueError("Durable physical attempt identity differs from declaration")
        for trial in intent["trial_ids"]:
            if rows[trial]["status"]=="unattempted":
                rows[trial]=dict(trial_id=trial,status="executed",physical_execution_attempted=True,execution_complete=False,
                    outcome_resolution="unknown_or_missing",execution_extent_unknown=True,numeric_complete_cycle=False,
                    reason="external_stop_after_durable_native_attempt_intent; never retry this request")
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    outcomes=[rows[trial] for trial in ids];write_json(folder/"outcomes.json",outcomes)
    claim=json.loads((parent/"final_test_claim.json").read_text()) if (parent/"final_test_claim.json").exists() else {}
    result=dict(schema="task10b_phase2_expanded_execution_v1",status="recovered_external_termination",planned_trials=len(ids),
        execution_identity=identity["identity"],outcomes_sha256=sha256_file(folder/"outcomes.json"),
        logical_origin_run=claim.get("logical_origin_run",ancestor.get("logical_origin_run",str(parent))),final_test_claim=claim.get("path"),
        unattempted_attempts=sum(row["status"]=="unattempted" for row in outcomes),
        terminal_accounting=dict(path=str(terminal_path),sha256=sha256_file(terminal_path)),
        selected_durable_ledger=None if not ledgers else dict(path=str(ledgers[-1]),sha256=sha256_file(ledgers[-1])),
        durable_attempt_intents=intents,source_application_run=str(parent),
        no_unknown_attempt_replayed=True,unknown_or_lost_extent_blocks_qualification=True,
        qualification_passed=False,phase2_complete=False,formal_acceptance=False)
    write_json(folder/"execution.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-parent",type=Path,required=True)
    parser.add_argument("--terminal-accounting",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    options=parser.parse_args();print(json.dumps(recover_interrupted_parent(options.recover_parent,options.terminal_accounting,options.output),sort_keys=True))


if __name__=="__main__":main()
