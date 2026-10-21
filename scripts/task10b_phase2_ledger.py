#!/usr/bin/env python3
"""Materialize declared D0 trials from saved evidence without executing physics.

An absent trace is not proof of zero execution. In particular, launch/progress
files are written before native work and cannot certify how far an interrupted
process advanced. Counts distinguish saved complete intervals, reported counters
and unknown execution extent. This ledger does not audit physical qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/task10b"
BANK_IDS = tuple(range(8))


def owned(path, *, exists=True):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve(strict=exists)
    if not path.is_relative_to(ROOT):
        raise ValueError("All ledger inputs/outputs must resolve inside human-punching")
    return path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


class Evidence:
    def __init__(self):
        self.sources = {}
        self.issues = []

    def read(self, path, *, required=False):
        path = owned(path, exists=False)
        if not path.exists():
            if required:
                raise ValueError(f"Required declaration is missing: {path}")
            return None
        data = path.read_bytes()
        self.sources[str(path)] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        try:
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            return result
        except (ValueError, UnicodeError) as error:
            if required:
                raise ValueError(f"Malformed required declaration: {path}") from error
            self.issues.append(dict(path=str(path), reason="malformed_json", detail=str(error)))
            return None

    def trace(self, path, worlds):
        """Read only clocks and NPY headers; no FK, dynamics or actor import."""
        path = owned(path, exists=False)
        if not path.exists():
            return None
        before = path.stat()
        identity = sha(path)
        self.sources[str(path)] = dict(sha256=identity, bytes=before.st_size)
        try:
            with zipfile.ZipFile(path) as archive:
                shapes = {}
                for name in ("time", "pre_qpos", "policy_episode_tick"):
                    member = name + ".npy"
                    if member not in archive.namelist():
                        continue
                    with archive.open(member) as stream:
                        version = np.lib.format.read_magic(stream)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                        else:
                            raise ValueError("Unreviewed trace array header version")
                    if dtype.hasobject:
                        raise ValueError("Object trace array")
                    shapes[name] = shape
                shape = shapes.get("time")
                if shape is None or len(shape) != 2 or shape[1] != worlds or shape[0] < 1:
                    raise ValueError("Trace time array does not match declared worlds")
                if archive.getinfo("time.npy").file_size > 8*1024**2:
                    raise ValueError("D0 clock array exceeds bounded audit size")
                time_s = np.load(io.BytesIO(archive.read("time.npy")), allow_pickle=False)
                if not np.isfinite(time_s).all() or not np.allclose(np.diff(time_s, axis=0), .005, atol=2.e-5, rtol=0):
                    raise ValueError("Trace does not contain successive finite 5ms clocks")
                if not np.allclose(time_s[0], time_s[0, 0], atol=1.e-6, rtol=0):
                    raise ValueError("World start clocks differ")
                steps = shape[0] - 1
                pre = shapes.get("pre_qpos", (0,))[0]
                if pre not in (steps, steps+1):
                    raise ValueError("Recorded drive intervals disagree with saved completed intervals")
                actors = shapes.get("policy_episode_tick", (0,))[0]
                if actors and actors not in (math.ceil(steps/4), steps//4+1):
                    raise ValueError("Actor boundary records disagree with completed intervals")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError("Trace changed during audit")
            return dict(valid=True, saved_completed_physics_steps=steps,
                recorded_preintegration_intervals=pre, recorded_actor_boundaries=actors,
                saved_completed_actor_blocks=steps//4, final_time_s=time_s[-1].tolist())
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError) as error:
            self.issues.append(dict(path=str(path), reason="unreadable_or_inconsistent_trace", detail=str(error)))
            return dict(valid=False, saved_completed_physics_steps=None, error=str(error))


def scheduler_evidence(paths, evidence):
    latest = {}
    for raw_path in paths:
        path = owned(raw_path)
        data = path.read_bytes()
        evidence.sources[str(path)] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                                          semantics="watch snapshot, not a live scheduler query")
        for line in data.splitlines():
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                evidence.issues.append(dict(path=str(path), reason="incomplete_scheduler_snapshot_line"))
                continue
            if record.get("returncode") != 0:
                continue
            for status in record.get("stdout", "").splitlines():
                fields = status.split("|")
                if len(fields) >= 2 and fields[0].isdigit():
                    job = fields[0]
                    row = dict(job_id=job, state=fields[1], utc=record.get("utc"), raw_line=status)
                    if job not in latest or (row["utc"] or "") > (latest[job]["utc"] or ""):
                        latest[job] = row
    return latest


def _constructor_proves_no_d0(error):
    # Match explicit saved caller lines, not a guessed exception type. Native
    # constructor warm-up may still have executed; this proves only no D0 panel.
    text = (error or {}).get("exception", {}).get("traceback", "")
    return ("evaluator = NativeSpecialistEnv(" in text or "env = NativeSpecialistEnv(args, selected" in text)


def _reports(record, evidence, path):
    result = {}
    for row in (record or {}).get("trials", []):
        trial = row.get("trial_id")
        if not trial or trial in result:
            evidence.issues.append(dict(path=str(path), reason="duplicate_or_missing_trial_identity"))
        else:
            result[trial] = row
    return result


def materialize_packet(packet_path, evidence, scheduler):
    packet_path = owned(packet_path)
    packet = evidence.read(packet_path, required=True)
    if packet.get("schema") != "task10b_phase2_execution_packet_v1" or packet.get("mode") not in ("frozen", "pilot"):
        raise ValueError("D0 ledgers require a frozen/pilot execution packet, not reference generation")
    family = packet["family"]
    base = packet_path.parent
    run = owned(packet["output"], exists=False)
    if run != base / "run":
        raise ValueError("Packet output is not its bound run directory")
    catalog_rel = packet["catalog"]
    expected = packet["files"].get(catalog_rel)
    if not expected:
        raise ValueError("Packet does not bind its declared catalog hash")
    snapshot = base / "source_snapshot" / catalog_rel
    catalog_path = snapshot if snapshot.exists() else owned(catalog_rel)
    catalog = evidence.read(catalog_path, required=True)
    if evidence.sources[str(catalog_path)]["sha256"] != expected:
        raise ValueError("Bound catalog hash mismatch")
    if catalog.get("schema") != "task10b_phase2_reference_catalog_v1":
        raise ValueError("Unknown reference catalog declaration")
    cells = sorted([row for row in catalog["cells"] if row["family"] == family],
                   key=lambda row: (row["kind"] != "original_control", row.get("requested_ttc_s") or 0))
    if not cells or len({row["candidate_id"] for row in cells}) != len(cells):
        raise ValueError("Missing or duplicate family candidate declarations")
    if any(Path(row["candidate_id"]).name != row["candidate_id"] or row["candidate_id"] in (".", "..") for row in cells):
        raise ValueError("Invalid candidate path identity")
    launch = evidence.read(run / "launch.json")
    if launch and (launch.get("planned_cells") != [row["candidate_id"] for row in cells]
                   or launch.get("diagnostic_bank_ids") != list(BANK_IDS)
                   or launch.get("catalog_sha256") != expected or launch.get("family") != family):
        evidence.issues.append(dict(path=str(run / "launch.json"), reason="launch_declaration_differs_from_packet"))
    wrapper = evidence.read(base / "execution.json")
    started = evidence.read(base / "execution_started.json")
    terminal = evidence.read(base / "terminal_accounting.json")
    submission = evidence.read(base / "submission.json")
    intent = evidence.read(base / "submission_intent.json")
    early_error = evidence.read(run / "execution_error.json")
    execution = evidence.read(run / "execution.json")
    training_plan = evidence.read(run / "training/training_plan.json")
    training = evidence.read(run / "training/summary.json")
    job = str((submission or {}).get("job_id", ""))
    scheduled = scheduler.get(job)
    packet_sha = evidence.sources[str(packet_path)]["sha256"]
    bindings_valid = True
    for path, record in ((base / "execution.json", wrapper), (base / "terminal_accounting.json", terminal),
                         (base / "submission.json", submission)):
        if record and record.get("packet_sha256") != packet_sha:
            bindings_valid = False
            evidence.issues.append(dict(path=str(path), reason="packet_hash_binding_mismatch"))
    ended = bool(wrapper or terminal or execution or early_error or training)
    native_zero = bool(bindings_valid and wrapper and wrapper.get("native_processes_started") == 0
                       and wrapper.get("actual_physics_steps") == 0)
    discovered = set()
    if run.exists():
        for folder in run.glob("validation_*"):
            suffix = folder.name.removeprefix("validation_")
            if folder.is_dir() and suffix.isdigit():
                discovered.add(int(suffix))
    updates = [0] if packet["mode"] == "frozen" else list(range(0, int(packet["max_updates"])+1, 8))
    if training_plan and training_plan.get("validation_updates") != updates:
        evidence.issues.append(dict(path=str(run / "training/training_plan.json"), reason="validation_schedule_differs"))
    if discovered - set(updates):
        evidence.issues.append(dict(path=str(run), reason="undeclared_validation_panels", updates=sorted(discovered-set(updates))))
    constructor_zero = bool(early_error and not discovered and _constructor_proves_no_d0(early_error))
    pending = bool(bindings_valid and scheduled and scheduled["state"] == "PENDING"
                   and not any((started, launch, wrapper, terminal, discovered)))
    hard_reasons = dict(wrapper_exception=(wrapper or {}).get("exception"),
        budget_stop=(wrapper or {}).get("budget_stop"), terminal_state=(terminal or {}).get("state"),
        early_exception=(early_error or {}).get("exception"), training_stop=(training or {}).get("stop_reason"))
    progress = []
    if run.exists():
        for path in sorted(run.glob("**/progress*.json")):
            record = evidence.read(path)
            if record is not None:
                progress.append(dict(path=str(path), record=record))
    trials, panels = [], []
    for update in sorted(set(updates) | discovered):
        folder = run / f"validation_{update:05d}"
        outer = evidence.read(folder / "summary.json")
        outer_reports = _reports(outer, evidence, folder / "summary.json")
        for cell in cells:
            candidate = cell["candidate_id"]
            cell_folder = folder / candidate
            plan = evidence.read(cell_folder / "plan.json")
            panel = evidence.read(cell_folder / "summary.json")
            panel_reports = _reports(panel, evidence, cell_folder / "summary.json")
            commitments = evidence.read(cell_folder / "launch_commitments.json")
            trace = evidence.trace(cell_folder / "raw_trace.npz", len(BANK_IDS))
            ids = [f"{candidate}_bank_{bank:02d}" for bank in BANK_IDS]
            if plan and (plan.get("trials") != ids or plan.get("candidates") != [candidate]*8):
                evidence.issues.append(dict(path=str(cell_folder / "plan.json"), reason="trial_world_identity_mismatch"))
                trace = dict(valid=False, error="trial/world identity mismatch", saved_completed_physics_steps=None)
            planned_steps = (plan or {}).get("planned_physics_steps")
            if planned_steps is None and cell.get("episode_duration_s") is not None:
                planned_steps = math.ceil(float(cell["episode_duration_s"])/.020-1.e-10)*4
            saved_steps = trace.get("saved_completed_physics_steps") if trace and trace["valid"] else None
            horizon_recorded = bool(saved_steps is not None and planned_steps is not None and saved_steps >= planned_steps)
            candidate_faults = (panel or {}).get("hard_faults", [])
            candidate_error = (panel or {}).get("exception")
            panels.append(dict(update=update, candidate_id=candidate, declared=update in updates,
                plan_present=plan is not None, summary_present=panel is not None,
                trace=trace, planned_physics_steps=planned_steps,
                reported_actual_physics_steps=(panel or {}).get("actual_physics_steps"),
                reported_actual_policy_updates=(panel or {}).get("actual_policy_updates"),
                reported_executed_trials=(panel or {}).get("executed_trials"),
                hard_faults=candidate_faults, exception=candidate_error))
            if panel and saved_steps is not None and panel.get("actual_physics_steps") != trace["recorded_preintegration_intervals"]:
                evidence.issues.append(dict(path=str(cell_folder / "summary.json"), reason="reported_preintegration_counter_mismatch"))
            for world, trial in enumerate(ids):
                standalone = evidence.read(cell_folder / f"trial_{world:04d}.json")
                outer_report = outer_reports.get(trial)
                report = standalone or panel_reports.get(trial)
                if report is None and outer_report and outer_report.get("disposition") == "executed":
                    report = outer_report
                if report and (report.get("trial_id") != trial or report.get("candidate_id") != candidate):
                    evidence.issues.append(dict(path=str(cell_folder), reason="outcome_identity_mismatch", trial_id=trial))
                    report = None
                reasons = list(cell.get("rejection_reasons", []))
                extent_unknown = False
                exact_steps = None
                if not cell.get("kinematic_admission", False):
                    disposition = "offline_rejection"
                    exact_steps = 0
                    if trace or plan or report:
                        evidence.issues.append(dict(path=str(cell_folder), reason="offline_rejected_cell_has_execution_evidence"))
                        disposition, exact_steps, extent_unknown = "offline_rejection_with_execution_conflict", None, True
                elif horizon_recorded:
                    disposition = "completed_reported" if report else "execution_horizon_recorded_outcome_missing"
                    extent_unknown = trace["recorded_preintegration_intervals"] > saved_steps
                    exact_steps = None if extent_unknown else saved_steps
                elif trace and trace["valid"]:
                    disposition = "incomplete_execution" if ended or panel else "execution_in_progress"
                    # An unclosed file cannot exclude an unsaved tail.
                    extent_unknown = panel is None
                    if panel is not None and trace["recorded_preintegration_intervals"] == saved_steps:
                        exact_steps = saved_steps
                    elif trace["recorded_preintegration_intervals"] > saved_steps:
                        extent_unknown = True
                elif pending:
                    disposition, exact_steps = "not_yet_executed_pending", 0
                elif native_zero:
                    disposition, exact_steps = "unattempted_native_not_started", 0
                elif constructor_zero:
                    disposition, exact_steps = "unattempted_before_d0_constructor_failure", 0
                elif outer_report and outer_report.get("disposition") == "unattempted_after_hard_failure" and not any((plan, trace, report)):
                    disposition, exact_steps = "unattempted_after_hard_failure", 0
                elif training and update > training.get("completed_updates", -1) and update not in discovered:
                    disposition, exact_steps = "unattempted_after_training_stop", 0
                elif not any((submission, intent, started, launch, wrapper, terminal, discovered, progress)):
                    disposition, exact_steps = "prepared_not_submitted", 0
                else:
                    disposition, extent_unknown = "execution_extent_unknown", True
                if report and not horizon_recorded:
                    reasons.append("reported_outcome_has_no_complete_saved_horizon")
                trials.append(dict(packet=str(packet_path), job_id=job or None, mode=packet["mode"], family=family,
                    validation_update=update, declared_panel=update in updates, trial_id=trial,
                    candidate_id=candidate, bank_id=world, reference_sha256=cell.get("reference_sha256"),
                    requested_ttc_s=cell.get("requested_ttc_s"), disposition=disposition, reasons=reasons,
                    saved_completed_physics_steps_lower_bound=saved_steps if saved_steps is not None else (0 if exact_steps == 0 else None),
                    actual_completed_physics_steps=exact_steps, execution_extent_unknown=extent_unknown,
                    execution_horizon_recorded=horizon_recorded, outcome_report=report,
                    reported_numeric_complete_cycle=report.get("numeric_complete_cycle") if report else None,
                    complete_cycle_qualified=False, source_outer_disposition=(outer_report or {}).get("disposition"),
                    source_outer_report=outer_report,
                    launch_commitment=next((item for item in (commitments or {}).get("commitments", [])
                                            if item.get("trial_id") == trial), None),
                    hard_faults=candidate_faults, panel_exception=candidate_error,
                    outer_panel_exception=(outer or {}).get("exception"),
                    outer_panel_hard_faults=(outer or {}).get("hard_faults", []), packet_stop_reasons=hard_reasons))
    return dict(packet=str(packet_path), packet_sha256=packet_sha, family=family, mode=packet["mode"],
        scheduler_snapshot=scheduled, terminal=terminal, launch_present=launch is not None,
        planned_validation_updates=updates, declared_cells=[cell["candidate_id"] for cell in cells],
        diagnostic_bank_ids=list(BANK_IDS), native_constructor_work_counts_unknown=constructor_zero,
        hard_stop_reasons=hard_reasons, progress_evidence=progress, panels=panels, trials=trials)


def build_ledger(packets, *, scheduler_paths=()):
    evidence = Evidence()
    scheduler = scheduler_evidence(scheduler_paths, evidence)
    records = [materialize_packet(path, evidence, scheduler) for path in packets]
    trials = [trial for record in records for trial in record["trials"]]
    panels = [panel for record in records for panel in record["panels"]]
    identities = [(row["packet"], row["validation_update"], row["trial_id"]) for row in trials]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate packet/trial declarations")
    summary = dict(schema="task10b_phase2_independent_d0_ledger_v1", utc=datetime.now(timezone.utc).isoformat(),
        auditor_source_sha256=sha(__file__), numpy_version=np.__version__,
        status="materialized_with_evidence_issues" if evidence.issues else "materialized",
        planned_trials=sum(row["declared_panel"] for row in trials), materialized_trials=len(trials),
        dispositions=dict(Counter(row["disposition"] for row in trials)),
        saved_complete_execution_horizons=sum(row["execution_horizon_recorded"] for row in trials),
        completed_reported_trials=sum(row["disposition"] == "completed_reported" for row in trials),
        reported_numeric_complete_cycles=sum(row["reported_numeric_complete_cycle"] is True for row in trials),
        unknown_execution_extent_trials=sum(row["execution_extent_unknown"] for row in trials),
        saved_d0_completed_physics_intervals_lower_bound=sum(
            (panel.get("trace") or {}).get("saved_completed_physics_steps") or 0 for panel in panels),
        saved_d0_completed_world_intervals_lower_bound=8*sum(
            (panel.get("trace") or {}).get("saved_completed_physics_steps") or 0 for panel in panels),
        evidence_issues=evidence.issues, full_domain_qualification=False, formal_acceptance=False,
        count_semantics="Saved post-state intervals are lower bounds unless a closed trace proves exact extent. Pre-integration counters and planned actor boundaries do not themselves prove completed integration. Missing interrupted traces remain unknown.",
        packets=[{key: value for key, value in record.items() if key != "trials"} for record in records])
    return summary, trials, evidence.sources


def write_ledger(output, packets, *, scheduler_paths=()):
    output = owned(output, exists=False)
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError("Write a fresh audit directory under outputs/task10b")
    if output.exists():
        raise FileExistsError("Ledger output already exists; prior results cannot be overwritten")
    if any(output.is_relative_to(owned(packet).parent) for packet in packets):
        raise ValueError("Audit output must be separate from every immutable packet/run directory")
    summary, trials, sources = build_ledger(packets, scheduler_paths=scheduler_paths)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("ledger.json", summary), ("sources.json", sources)):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    with (output / "trials.jsonl").open("x") as stream:
        for row in trials:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", action="append", type=Path, required=True)
    parser.add_argument("--scheduler-status", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_ledger(args.output, args.packet, scheduler_paths=args.scheduler_status)
    print(json.dumps({key: result[key] for key in ("status", "planned_trials", "dispositions",
                                                 "unknown_execution_extent_trials")}))


if __name__ == "__main__":
    main()
