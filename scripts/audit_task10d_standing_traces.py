"""Independent saved-output audit for the standing Phase 3C dynamic sample.

No simulator execution, replay, production event-classifier import, or writes to
the bank/reference/fixture payloads. The output describes evidence limitations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_time(t, mask):
    ix = np.flatnonzero(mask)
    return None if not len(ix) else float(t[ix[0]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("Audit output must be a new file under human-punching")
    bank_root, fixture_root = args.bank.resolve(strict=True), args.fixture.resolve(strict=True)
    fixture = dict(np.load(fixture_root / "fixture.npz", allow_pickle=False))
    fixture_meta = json.loads((fixture_root / "fixture.json").read_text())
    if sha256(fixture_root / "fixture.npz") != fixture_meta["payload_sha256"]:
        raise ValueError("Fixture identity changed")
    selected = json.loads((bank_root / "selected_trials.json").read_text())
    results = json.loads((bank_root / "dynamic/results.json").read_text())
    rows, errors = [], []
    records = {row["label"]: row for row in results}
    expected = selected["trials"]
    if len(records) != len(results) or set(records) != {row["label"] for row in expected}:
        errors.append("Selected/completed labels differ or duplicate")
    if not selected["selection_before_dynamic_outcomes"]:
        errors.append("Trial selection not declared before dynamic outcomes")
    first_targets = []
    baseline_trace = None
    shared_prefix_errors = {"qpos": 0., "qvel": 0.}
    shared_position_errors_m = {"pelvis": 0., "head": 0.}
    for selection in expected:
        label = selection["label"]
        if label not in records:
            continue
        summary = records[label]
        case = bank_root / "dynamic" / label
        trace = dict(np.load(case / "trace.npz", allow_pickle=False))
        if baseline_trace is None:
            baseline_trace = trace
        overlap = min(len(trace["time_s"]), len(baseline_trace["time_s"]))
        for key in shared_prefix_errors:
            shared_prefix_errors[key] = max(shared_prefix_errors[key], float(np.max(
                np.abs(trace[key][:overlap]-baseline_trace[key][:overlap]))))
        for name, key in (("pelvis", "qpos"), ("head", "head")):
            shared_position_errors_m[name] = max(shared_position_errors_m[name], float(np.max(
                np.linalg.norm(trace[key][:overlap, :3]-baseline_trace[key][:overlap, :3],axis=1))))
        camera = json.loads((case / "camera.json").read_text())
        checks = {}
        t, source = trace["time_s"], trace["source_s"]
        checks["first_qpos_exact_fixture"] = np.array_equal(trace["qpos"][0], fixture["sim/data/qpos"][0])
        checks["first_qvel_exact_fixture"] = np.array_equal(trace["qvel"][0], fixture["sim/data/qvel"][0])
        checks["first_head_exact_fixture"] = np.array_equal(trace["head"][0], fixture["geometry/head_world"])
        checks["first_head_velocity_exact_fixture"] = np.array_equal(trace["head_velocity"][0], fixture["geometry/head_velocity_world"])
        checks["no_launch_integration_runtime_record"] = summary["reset"]["physics_steps"] == 0 and t[0] == 0
        checks["all_action_history_restored_runtime_record"] = all(
            summary["reset"]["restored_max_abs_error"][name] == 0 for name in
            ("action/_action", "action/_prev_action", "action/_prev_prev_action"))
        checks["all_atomic_state_fields_unchanged_runtime_record"] = all(summary["atomic_preserved"].values())
        checks["captured_noise_and_controller_context"] = summary["reset"]["restored_solver_and_measurement_context"]
        checks["no_camera_history_restored"] = not summary["reset"]["camera_history_restored"]
        checks["physics_trace_every_5ms"] = np.allclose(np.diff(t), .005, atol=1e-12, rtol=0)
        end = summary["reference_end_s"]
        checks["unit_source_clock_and_terminal_hold"] = np.allclose(source, np.minimum(t, end), atol=1e-12, rtol=0)
        checks["complete_reference_and_400ms_hold"] = t[-1] >= end + .4 - 1e-12
        checks["no_arrival_before_recovery"] = summary["safe_arrival_evaluated"] is False and summary["recovery_complete"]
        checks["fresh96_only"] = summary["avoidance_residual"] == 0 and summary["clock_rate"] == 1 and summary["requested_ttc_s"] is None
        checks["first_camera_truthful_padding"] = bool(camera[0]["padding"]) and all(camera[0][key] == 0 for key in
            ("time_s", "capture_frame_id", "delivered_frame_id", "delivered_capture_timestamp_s", "delivery_timestamp_s"))
        capture_times = np.asarray([row["time_s"] for row in camera])
        checks["camera_acquisitions_every_20ms"] = np.allclose(np.diff(capture_times), .020, atol=1e-12, rtol=0)
        checks["camera_covers_entire_trace"] = abs(capture_times[-1] - t[-1]) < 1e-12
        checks["no_future_delivered_images"] = all(row["delivered_capture_timestamp_s"] <= row["time_s"] + 1e-12 for row in camera)
        checks["padding_only_repeats_initial_image"] = all(row["delivered_frame_id"] == 0 and row["delivered_capture_timestamp_s"] == 0
            for row in camera if row["padding"])
        checks["media_complete_record"] = summary["media_complete"] and summary["video_frames"] == len(camera)
        frozen = np.linalg.norm(trace["gloves"] - trace["head"][0][None, None, :], axis=-1)
        moving = np.linalg.norm(trace["gloves"] - trace["head"][:, None, :], axis=-1)
        checks["saved_distances_match_geometry"] = np.allclose(frozen, trace["frozen_distances"], atol=1e-8, rtol=0) and np.allclose(moving, trace["moving_distances"], atol=1e-8, rtol=0)
        active = ("left", "right").index(summary["active_hand"])
        outward = source <= summary["outward_end_source_s"] + 1e-12
        events = dict(
            frozen_head_active_outward_entry_s=first_time(t, outward & (frozen[:, active] <= .14)),
            moving_head_first_encounter_s=first_time(t, (moving <= .14).any(axis=1)),
            first_glove_clearance_violation_s=first_time(t, (moving < .16).any(axis=1)),
            first_mechanical_safety_violation_s=first_time(t, ~trace["mechanically_safe"]))
        checks["reported_first_events_match_direct_trace"] = all(events[key] == summary[key] for key in events)
        checks["mechanical_flags_consistent_with_finite_and_height"] = bool(np.all(~trace["mechanically_safe"] | (trace["finite"] & (trace["qpos"][:, 2] >= .55))))
        checks["all_numerics_finite"] = bool(trace["finite"].all() and all(np.isfinite(trace[key]).all() for key in
            ("qpos", "qvel", "head", "gloves", "physical_targets")))
        first_targets.append(trace["physical_targets"][0])
        for name, passed in checks.items():
            if not passed:
                errors.append(f"{label}: {name}")
        rows.append(dict(label=label, primitive_id=summary["primitive_id"], checks={k:bool(v) for k,v in checks.items()}, events=events,
            trace_samples=len(t), camera_samples=len(camera), reference_end_s=end, logged_end_s=float(t[-1]),
            minimum_head_distances_m=moving.min(axis=0).tolist(),
            minimum_root_height_m=float(trace["qpos"][:, 2].min()),
            mechanical_failure=not bool(trace["mechanically_safe"].all()),
            maximum_head_displacement_m=float(np.linalg.norm(trace["head"]-trace["head"][0], axis=1).max()),
            inputs_sha256={name:sha256(case/name) for name in ("trace.npz", "summary.json", "camera.json")}))
    first_targets = np.asarray(first_targets)
    target_spread = 0. if not len(first_targets) else float(np.max(np.abs(first_targets-first_targets[0])))
    if target_spread > 0:
        errors.append("Identical fixture first physical targets differ across trials")
    result = dict(schema="task10d_phase3c_independent_trace_audit_v1", passed=not errors,
        selected_count=len(expected), completed_count=len(results), audited_count=len(rows),
        primitive_exposure=dict(Counter(row["primitive_id"] for row in rows)),
        all_first_physical_targets_max_abs_difference=target_spread,
        repeated_fixture_shared_prefix_max_abs_difference=shared_prefix_errors,
        repeated_fixture_shared_prefix_max_position_difference_m=shared_position_errors_m,
        fixture_sha256=sha256(fixture_root/"fixture.npz"),
        source_script_sha256=sha256(__file__), checks_failed=errors, trials=rows,
        limitations=["Raw action histories are preserved by native reset assertions; dynamic traces do not duplicate these arrays. All first physical targets are compared independently.",
            "Mechanical boolean is the accepted native diagnostic. Finite values and root height are checked independently; self-contact is not recomputed offline.",
            "Media frame-count declarations are checked against trace/camera rows; this audit does not decode videos."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key:result[key] for key in ("passed", "audited_count", "primitive_exposure", "checks_failed")},indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
