"""Saved-evidence ledger tests, including interrupted and never-started trials."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("phase2_ledger", Path(__file__).resolve().parents[1]/"scripts/task10b_phase2_ledger.py")
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream)
    return path


def fixture(tmp_path, *, mode="frozen", rejected=False):
    base = tmp_path/"packet"
    base.mkdir()
    catalog_path = tmp_path/"catalog.json"
    cells = [dict(candidate_id="jab_original", family="jab", kind="original_control",
                  requested_ttc_s=None, kinematic_admission=True, episode_duration_s=.04,
                  reference_sha256="b"*64, rejection_reasons=[])]
    if rejected:
        cells.append(dict(candidate_id="jab_uniform_060", family="jab", kind="uniform",
            requested_ttc_s=.6, kinematic_admission=False, rejection_reasons=["unsupported_path"],
            reference_sha256=None))
    catalog = dict(schema="task10b_phase2_reference_catalog_v1", cells=cells)
    put(catalog_path, catalog)
    relative = str(catalog_path.relative_to(ledger.ROOT))
    digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    packet = dict(schema="task10b_phase2_execution_packet_v1", mode=mode, family="jab",
        output=str((base/"run").relative_to(ledger.ROOT)), catalog=relative,
        files={relative: digest}, max_updates=16 if mode == "pilot" else 0)
    path = put(base/"packet.json", packet)
    return path, base/"run", digest


def launch(path, run, digest, *, rejected=False):
    put(run/"launch.json", dict(planned_cells=["jab_original"]+(["jab_uniform_060"] if rejected else []),
        diagnostic_bank_ids=list(range(8)), family="jab", catalog_sha256=digest))
    put(path.parent/"execution_started.json", dict(native_processes_started=0))


def wrapper(path, *, native=1, **kwargs):
    return put(path.parent/"execution.json", dict(packet_sha256=ledger.sha(path), status="failed",
        native_processes_started=native, **kwargs))


def panel(run, *, steps=8, update=0, summary=True, outcomes=True, extra_pre=False):
    folder = run/f"validation_{update:05d}/jab_original"
    ids = [f"jab_original_bank_{i:02d}" for i in range(8)]
    put(folder/"plan.json", dict(trials=ids, candidates=["jab_original"]*8, planned_physics_steps=8))
    if steps is not None:
        np.savez_compressed(folder/"raw_trace.npz",
            time=np.broadcast_to(np.arange(steps+1)[:, None]*.005, (steps+1, 8)),
            pre_qpos=np.zeros((steps+int(extra_pre), 8, 55)),
            policy_episode_tick=np.arange(0, steps+int(extra_pre), 4))
    reports = [dict(trial_id=trial, candidate_id="jab_original", numeric_complete_cycle=i != 0)
               for i, trial in enumerate(ids)] if outcomes else []
    if summary:
        put(folder/"summary.json", dict(status="executed" if steps == 8 else "failed", trials=reports,
            actual_physics_steps=steps+int(extra_pre) if steps is not None else 0,
            actual_policy_updates=int(np.ceil((steps or 0)/4)), executed_trials=len(reports),
            hard_faults=[] if steps == 8 else [{"faults": {"effort_cap": [2]}}],
            exception=None if steps == 8 else {"type": "RuntimeError", "message": "hard physical stop"}))
    return folder


def test_every_declared_cell_and_eight_banks_retains_offline_rejections(tmp_path):
    path, _, _ = fixture(tmp_path, rejected=True)
    summary, rows, _ = ledger.build_ledger([path])
    assert summary["planned_trials"] == 16 and len(rows) == 16
    assert summary["dispositions"] == {"prepared_not_submitted": 8, "offline_rejection": 8}
    assert {row["bank_id"] for row in rows} == set(range(8))
    assert rows[-1]["reasons"] == ["unsupported_path"]


def test_watchdog_absent_active_trace_is_unknown_not_zero(tmp_path):
    path, run, digest = fixture(tmp_path)
    launch(path, run, digest)
    wrapper(path, budget_stop="application_wall")
    panel(run, steps=None, summary=False, outcomes=False)
    summary, rows, _ = ledger.build_ledger([path])
    assert summary["dispositions"] == {"execution_extent_unknown": 8}
    assert all(row["actual_completed_physics_steps"] is None for row in rows)
    assert summary["unknown_execution_extent_trials"] == 8
    assert rows[0]["packet_stop_reasons"]["budget_stop"] == "application_wall"


def test_launch_alone_and_misleading_native_counts_never_prove_zero(tmp_path):
    path, run, digest = fixture(tmp_path)
    launch(path, run, digest)
    put(run/"execution_error.json", dict(exception={"type": "RuntimeError", "message": "failed"},
        native_counts="absent attempts remain unattempted"))
    summary, _, _ = ledger.build_ledger([path])
    assert summary["dispositions"] == {"execution_extent_unknown": 8}


def test_constructor_failure_proves_only_no_d0_not_no_native_physics(tmp_path):
    path, run, digest = fixture(tmp_path)
    launch(path, run, digest)
    put(run/"execution_error.json", dict(exception={"traceback":
        'File "phase2_learning.py", in execute_cli\n    evaluator = NativeSpecialistEnv(args, admitted, num_envs=8)\nRuntimeError: compile'}))
    summary, rows, _ = ledger.build_ledger([path])
    assert summary["dispositions"] == {"unattempted_before_d0_constructor_failure": 8}
    assert summary["packets"][0]["native_constructor_work_counts_unknown"]
    assert rows[0]["actual_completed_physics_steps"] == 0


def test_terminal_wrapper_native_zero_proves_unattempted(tmp_path):
    path, _, _ = fixture(tmp_path)
    wrapper(path, native=0, actual_physics_steps=0, exception={"message": "CUDA health failed"})
    summary, _, _ = ledger.build_ledger([path])
    assert summary["dispositions"] == {"unattempted_native_not_started": 8}


def test_wrong_packet_zero_receipt_and_conflicting_unattempted_rows_do_not_prove_zero(tmp_path):
    path, run, digest = fixture(tmp_path)
    put(path.parent/"execution.json", dict(packet_sha256="f"*64, native_processes_started=0,
                                          actual_physics_steps=0))
    folder = panel(run, steps=None, summary=False, outcomes=False)
    put(folder.parent/"summary.json", dict(trials=[dict(trial_id=f"jab_original_bank_{i:02d}",
        candidate_id="jab_original", disposition="unattempted_after_hard_failure") for i in range(8)]))
    summary, rows, _ = ledger.build_ledger([path])
    assert summary["dispositions"] == {"execution_extent_unknown": 8}
    assert all(row["actual_completed_physics_steps"] is None for row in rows)
    assert any(item["reason"] == "packet_hash_binding_mismatch" for item in summary["evidence_issues"])


def test_pending_requires_scheduler_snapshot_not_just_submission(tmp_path):
    path, _, _ = fixture(tmp_path)
    put(path.parent/"submission.json", dict(job_id="123", packet_sha256=ledger.sha(path)))
    summary, _, _ = ledger.build_ledger([path])
    assert summary["dispositions"] == {"execution_extent_unknown": 8}
    status = tmp_path/"status.jsonl"
    status.write_text(json.dumps(dict(utc="2026-10-21T00:00:00Z", returncode=0,
                                    stdout="123|PENDING|0:00|N/A|(Priority)\n"))+"\n")
    summary, _, _ = ledger.build_ledger([path], scheduler_paths=[status])
    assert summary["dispositions"] == {"not_yet_executed_pending": 8}


def test_complete_horizon_outcomes_and_negative_results_kept(tmp_path):
    path, run, digest = fixture(tmp_path)
    launch(path, run, digest)
    panel(run)
    wrapper(path)
    summary, rows, _ = ledger.build_ledger([path])
    assert summary["completed_reported_trials"] == 8
    assert summary["reported_numeric_complete_cycles"] == 7
    assert rows[0]["reported_numeric_complete_cycle"] is False
    assert all(row["actual_completed_physics_steps"] == 8 for row in rows)
    assert not any(row["complete_cycle_qualified"] for row in rows)


def test_full_trace_with_missing_reports_retains_execution_not_invented_success(tmp_path):
    path, run, digest = fixture(tmp_path)
    launch(path, run, digest)
    panel(run, summary=False, outcomes=False)
    wrapper(path, budget_stop="application_wall")
    summary, _, _ = ledger.build_ledger([path])
    assert summary["dispositions"] == {"execution_horizon_recorded_outcome_missing": 8}
    assert summary["saved_complete_execution_horizons"] == 8
    assert summary["completed_reported_trials"] == 0 and summary["reported_numeric_complete_cycles"] == 0


def test_partial_trace_preintegration_is_not_completed_step(tmp_path):
    path, run, digest = fixture(tmp_path)
    launch(path, run, digest)
    panel(run, steps=3, extra_pre=True, outcomes=False)
    wrapper(path)
    summary, rows, _ = ledger.build_ledger([path])
    assert summary["dispositions"] == {"incomplete_execution": 8}
    assert rows[0]["saved_completed_physics_steps_lower_bound"] == 3
    assert rows[0]["actual_completed_physics_steps"] is None
    assert rows[0]["hard_faults"] == [{"faults": {"effort_cap": [2]}}]


def test_extra_unsaved_interval_after_full_horizon_is_still_unknown(tmp_path):
    path, run, digest = fixture(tmp_path)
    launch(path, run, digest)
    panel(run, steps=8, extra_pre=True, outcomes=False)
    wrapper(path)
    summary, rows, _ = ledger.build_ledger([path])
    assert summary["saved_complete_execution_horizons"] == 8
    assert summary["unknown_execution_extent_trials"] == 8
    assert all(row["actual_completed_physics_steps"] is None for row in rows)


def test_corrupted_trace_after_interruption_is_unknown(tmp_path):
    path, run, digest = fixture(tmp_path)
    launch(path, run, digest)
    folder = panel(run, steps=None, summary=False, outcomes=False)
    (folder/"raw_trace.npz").write_bytes(b"interrupted zip")
    wrapper(path, budget_stop="application_wall")
    summary, _, _ = ledger.build_ledger([path])
    assert summary["dispositions"] == {"execution_extent_unknown": 8}
    assert summary["status"] == "materialized_with_evidence_issues"


def test_pilot_declared_future_panels_remain_visible_after_stop(tmp_path):
    path, run, digest = fixture(tmp_path, mode="pilot")
    launch(path, run, digest)
    panel(run)
    put(run/"training/summary.json", dict(completed_updates=2, stop_reason="wall_limit"))
    summary, _, _ = ledger.build_ledger([path])
    assert summary["planned_trials"] == 24
    assert summary["dispositions"] == {"completed_reported": 8, "unattempted_after_training_stop": 16}


def test_outer_hard_abort_reason_and_declared_unattempted_rows_retained(tmp_path):
    path, run, _ = fixture(tmp_path)
    put(run/"validation_00000/summary.json", dict(exception={"message": "prior cell effort violation"},
        hard_faults=[{"effort_cap": [2]}], trials=[dict(trial_id=f"jab_original_bank_{i:02d}",
            candidate_id="jab_original", disposition="unattempted_after_hard_failure") for i in range(8)]))
    summary, rows, _ = ledger.build_ledger([path])
    assert summary["dispositions"] == {"unattempted_after_hard_failure": 8}
    assert rows[0]["outer_panel_exception"]["message"] == "prior cell effort violation"
    assert rows[0]["outer_panel_hard_faults"] == [{"effort_cap": [2]}]


def test_binding_mismatch_and_prior_output_are_never_overwritten(tmp_path):
    path, _, _ = fixture(tmp_path)
    catalog = json.loads(path.read_text())["catalog"]
    with (ledger.ROOT/catalog).open("a") as stream:
        stream.write(" ")  # Deliberately corrupt this synthetic fixture's identity.
    with pytest.raises(ValueError, match="catalog hash mismatch"):
        ledger.build_ledger([path])
    with pytest.raises(FileExistsError):
        ledger.write_ledger(tmp_path, [path])
