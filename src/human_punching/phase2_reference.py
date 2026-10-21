"""Task 10B Phase 2 fixed-path timing candidates; no physical qualification.

The untouched release remains the control. New references use a uniform clock
first, fresh tangent derivatives, original joint-only inverse-statics feedforward,
and freshly derived actor-boundary events. Offline force allocations are diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .contact import first_contact_time
from .wbc.references import interpolate_poses, tangent_velocities, sha256

PHYSICS_DT = .005
ACTOR_DT = .020
CANDIDATE_TTCS = {"jab": (.60, .75, .90), "cross": (.60, .75, .90),
                  "left_hook": (.95, 1.10, 1.25), "right_hook": (.80, 1., 1.20)}


def exact_grid(duration_s: float) -> np.ndarray:
    """Pad to the next physics boundary, never shorten the requested duration."""
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("A positive finite duration is required")
    return np.arange(int(np.ceil(duration_s / PHYSICS_DT - 1e-10)) + 1) * PHYSICS_DT


def reference_events(times, qpos, qvel, outward_source_time_s, scale,
                     *, zero_tolerance=1e-7) -> dict[str, Any]:
    """The first outward actor interval is derived from this reference's motion.

    qvel counts: a finite-difference velocity can start one sample before a pose
    changes. The outward endpoint remains the source-playback proxy, not contact.
    """
    t, q, v = map(np.asarray, (times, qpos, qvel))
    if (t.ndim != 1 or len(t) < 2 or q.shape[0] != len(t) or v.shape[0] != len(t)
            or not all(np.isfinite(a).all() for a in (t, q, v))
            or not np.allclose(np.diff(t), PHYSICS_DT, atol=1e-12, rtol=0)
            or t[0] != 0 or not np.isfinite(scale) or scale <= 0
            or not np.isfinite(outward_source_time_s) or outward_source_time_s <= 0):
        raise ValueError("Finite arrays on the exact 5ms clock are required")
    moving = ((np.abs(q - q[0]).max(axis=1) > zero_tolerance)
              | (np.abs(v).max(axis=1) > zero_tolerance))
    indices = np.flatnonzero(moving)
    if not len(indices):
        raise ValueError("A constant reference has no outward launch")
    first = int(indices[0])
    launch = (first // 4) * 4
    outward = int(np.ceil(outward_source_time_s * scale / PHYSICS_DT - 1e-10))
    if not launch < outward < len(t):
        raise ValueError("Invalid derived launch/outward completion ordering")
    return {"launch_tick": launch, "launch_time_s": float(t[launch]),
            "first_nonconstant_reference_tick": first,
            "first_nonconstant_reference_time_s": float(t[first]),
            "outward_end_tick": outward, "outward_completion_time_s": float(t[outward]),
            "reference_numerical_zero_tolerance": zero_tolerance,
            "phase_semantics": "Complete reset-relative uniform reference clock plus 1s terminal hold; first outward actor interval derived afresh from pose/velocity, source playback end including withdrawal is conservative outward-completion proxy; no phase/history reset"}


def retime_uniform(model, source: dict, scale: float) -> dict:
    """Resample the complete path and recompute tangent velocities/accelerations.

    Root angular qvel is MuJoCo's local free-joint tangent convention. The finite
    differences include clock scaling; qacc differentiates that qvel convention.
    Endpoint padding holds the last source pose and is explicit in the metadata.
    """
    if not np.isfinite(scale) or not .35 <= scale <= 3.:
        raise ValueError("Uniform duration scale must lie in the declared [.35,3] search bounds")
    old_t = np.asarray(source.get("time_s", source.get("t")), dtype=float)
    old_q = np.asarray(source["qpos"], dtype=float)
    if (old_t.ndim != 1 or len(old_t) < 3 or old_t[0] != 0
            or np.any(np.diff(old_t) <= 0) or old_q.shape != (len(old_t), model.nq)
            or not np.isfinite(old_t).all() or not np.isfinite(old_q).all()):
        raise ValueError("Malformed source reference")
    duration = float(old_t[-1] * scale)
    t = exact_grid(duration)
    source_t = np.minimum(t / scale, old_t[-1])
    q = interpolate_poses(old_t, old_q, source_t)
    v = tangent_velocities(model, t, q)
    a = np.gradient(v, t, axis=0, edge_order=2)
    events = reference_events(t, q, v, float(source["return_start"]), scale)
    return {"t": t, "time_s": t, "qpos": q, "qvel": v, "qacc": a,
            "source_reference_time_s": source_t, "duration_scale": float(scale),
            "unrounded_duration_s": duration, "terminal_padding_s": float(t[-1]-duration),
            "launch_time": float(source["launch_time"])*scale,
            "return_start": float(source["return_start"])*scale,
            "guard_hold_start": float(source["guard_hold_start"])*scale,
            "head_center": np.asarray(source["head_center"], dtype=float).copy(),
            "active_hand_index": int(source["active_hand_index"]),
            "attack_window_reference_s": np.array([events["launch_time_s"], events["outward_completion_time_s"]]),
            "events": events}


def glove_positions(model, poses) -> np.ndarray:
    data = mujoco.MjData(model)
    ids = [model.site(side+"Hand_center").id for side in ("Left", "Right")]
    result = np.empty((len(poses), 2, 3))
    for i, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        result[i] = data.site_xpos[ids]
    return result


def contact_diagnostics(times, gloves, target, active_hand, launch_time_s,
                        outward_completion_time_s, requested_ttc_s) -> dict:
    """Latch first crossings over the full trace; an early hit cannot be repaired."""
    if active_hand not in (0, 1) or not np.isfinite(requested_ttc_s) or requested_ttc_s < .5:
        raise ValueError("Valid active hand and requested TTC >=.5s required")
    t = np.asarray(times)
    gloves = np.asarray(gloves)
    crossings = [first_contact_time(t, gloves[:, [hand, hand]], target) for hand in (0, 1)]
    active, inactive = crossings[active_hand], crossings[1-active_hand]
    ttc = active-launch_time_s if np.isfinite(active) else None
    earliest = max(.5, requested_ttc_s-.020)
    late_bound = requested_ttc_s+.020
    timing = bool(ttc is not None and earliest-1e-10 <= ttc <= late_bound+1e-10
                  and active <= outward_completion_time_s+1e-10)
    wrong = bool(np.isfinite(inactive) and inactive <= active)
    return {"first_active_contact_time_s": float(active) if np.isfinite(active) else None,
            "first_inactive_contact_time_s": float(inactive) if np.isfinite(inactive) else None,
            "frozen_target_ttc_s": float(ttc) if ttc is not None else None,
            "requested_ttc_s": float(requested_ttc_s),
            "earliest_admissible_ttc_s": earliest,
            "timing_error_s": float(ttc-requested_ttc_s) if ttc is not None else None,
            "timing_valid": timing, "inactive_hand_first_or_simultaneous": wrong,
            "any_contact_before_launch": bool(min(crossings) < launch_time_s),
            "minimum_glove_target_distance_m": np.linalg.norm(gloves-target, axis=-1).min(axis=0).tolist(),
            "contact_method": "First piecewise-linear glove-center sphere crossing at radius .14m, offline FK diagnostic"}


def solve_uniform_timing(model, source, requested_ttc_s, *, maximum_trials=18):
    """Bounded scalar solve; the actor launch discontinuity is measured every trial."""
    if not np.isfinite(requested_ttc_s) or requested_ttc_s < .5 or maximum_trials < 1:
        raise ValueError("TTC floor and finite positive search trial cap are required")
    scale, best, trials = 1., None, []
    for trial in range(maximum_trials):
        candidate = retime_uniform(model, source, scale)
        candidate["glove_pos"] = glove_positions(model, candidate["qpos"])
        events = candidate["events"]
        contact = contact_diagnostics(candidate["t"], candidate["glove_pos"],
            candidate["head_center"], candidate["active_hand_index"], events["launch_time_s"],
            events["outward_completion_time_s"], requested_ttc_s)
        error = contact["timing_error_s"]
        trials.append({"trial": trial, "duration_scale": scale, "events": events, **contact})
        if error is None:
            raise ValueError("Fixed source path has no active contact; uniform retiming cannot repair geometry")
        if best is None or abs(error) < abs(best[1]["timing_error_s"]):
            best = candidate, contact
        if abs(error) <= .001:
            break
        # The first-outward actor interval changes only on .020s boundaries.
        # Shrink oscillations around those boundaries, retaining every trial.
        proposal = scale * requested_ttc_s / max(contact["frozen_target_ttc_s"], .01)
        if trial >= 3:
            proposal = .5 * (proposal + scale)
        proposal = float(np.clip(proposal, .35, 3.))
        if abs(proposal-scale) < 1e-8:
            break
        scale = proposal
    assert best is not None
    candidate, contact = best
    candidate["requested_ttc_s"] = float(requested_ttc_s)
    return candidate, {"method": "uniform_complete_reference_clock",
                       "search_bounds_duration_scale": [.35, 3.],
                       "maximum_trials": maximum_trials, "executed_trials": len(trials),
                       "trials": trials, "selected_contact": contact,
                       "qualified_physical_execution": False}


def screen_diagnostics(model_path, candidate, effort_caps):
    """Geometry and all six free-base equations, with declared support assumptions.

    A failed planted-corner model remains a visible diagnostic, never a hidden
    physical gate. This routine does not change the executed model or apply forces.
    """
    from .wbc.effort_diagnostics import (contact_edges, support_effort_solution,
                                       unconstrained_inverse)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    t, q, v, a = (candidate[name] for name in ("t", "qpos", "qvel", "qacc"))
    data = mujoco.MjData(model)
    lower, upper = model.jnt_range[1:].T
    excursion = np.maximum(np.maximum(lower-q[:, 7:], q[:, 7:]-upper), 0.).max(axis=1)
    self_depth, floor_depth = np.zeros(len(t)), np.zeros(len(t))
    support_status, ratios, residual, loads = [], [], [], []
    required_all, torque_all = [], []
    floor = model.geom("floor").id
    for index, (pose, velocity) in enumerate(zip(q, v, strict=True)):
        data.qpos[:], data.qvel[:] = pose, velocity
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            depth = max(0., -float(contact.dist))
            if floor in (contact.geom1, contact.geom2):
                floor_depth[index] = max(floor_depth[index], depth)
            elif model.geom_bodyid[contact.geom1] and model.geom_bodyid[contact.geom2]:
                self_depth[index] = max(self_depth[index], depth)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    if hasattr(mujoco.mjtEnableBit, "mjENBL_INVDISCRETE"):
        model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
    for pose, velocity, acceleration in zip(q, v, a, strict=True):
        required = unconstrained_inverse(model, data, pose, velocity, acceleration)
        required_all.append(required)
        try:
            edges, _, _ = contact_edges(model, data)
        except ValueError:
            support_status.append("declared_support_geometry_unavailable")
            ratios.append(np.nan); residual.append(np.full(6, np.nan))
            loads.append(np.full(32, np.nan)); torque_all.append(np.full(48, np.nan))
            continue
        solution = support_effort_solution(required, edges, np.asarray(effort_caps))
        support_status.append(solution["status"])
        ratios.append(solution["minimum_peak_effort_ratio"] if solution["support_feasible"] else np.nan)
        residual.append(solution["base_residual"]); loads.append(solution["edge_loads"])
        torque_all.append(solution["joint_torque"])
    ratios, residual = np.asarray(ratios), np.asarray(residual)
    feasible = np.asarray(support_status) == "feasible"
    finite = ratios[np.isfinite(ratios)]
    summary = {"schema": "task10b_phase2_fixed_path_screen_v1", "physics_executed": False,
        "acceptance_use": "diagnostic_only_native_execution_required",
        "frames": len(t), "joint_limit_excursion_max_rad": float(excursion.max()),
        "selected_self_penetration_max_m": float(self_depth.max()),
        "floor_penetration_max_m": float(floor_depth.max()),
        "joint_velocity_max_rad_s": float(np.abs(v[:, 6:]).max()),
        "joint_acceleration_max_rad_s2": float(np.abs(a[:, 6:]).max()),
        "support_status_counts": {str(value): int(np.sum(np.asarray(support_status) == value))
                                   for value in np.unique(support_status)},
        "within_effort_caps_frames": int(np.sum(feasible & (ratios <= 1.+1e-6))),
        "over_effort_caps_frames": int(np.sum(feasible & (ratios > 1.+1e-6))),
        "maximum_minimax_effort_ratio": float(finite.max()) if len(finite) else None,
        "maximum_base_force_residual_n": float(np.linalg.norm(residual[feasible, :3], axis=1).max()) if feasible.any() else None,
        "maximum_base_moment_residual_nm": float(np.linalg.norm(residual[feasible, 3:], axis=1).max()) if feasible.any() else None,
        "free_base_contract": "No root actuator; six exact root equations use contact Jacobian-transpose forces only",
        "support_assumption": "Original eight box-bottom corners projected to floor, each within [-1,+3]mm; nonnegative loads and conservative |fx|+|fy|<=mu*fz. Geometry failure is recorded, not replaced by fictitious root torque.",
        "limitations": "Exact imposed path and ideal declared double support, not a physical feasibility proof or expanded operational gate"}
    arrays = {"time_s": t, "qacc": a, "joint_limit_excursion_rad": excursion,
              "selected_self_penetration_m": self_depth, "floor_penetration_m": floor_depth,
              "required_generalized_force": np.asarray(required_all),
              "joint_torque": np.asarray(torque_all), "root_wrench_residual": residual,
              "contact_edge_loads": np.asarray(loads), "support_status": np.asarray(support_status),
              "minimum_peak_effort_ratio": ratios}
    return summary, arrays


def generate_catalog(bundle_path, release_root, output_root, predicate_path):
    """Write twelve new candidates and four immutable controls in a new directory."""
    from .wbc.provenance import writable_path
    from .wbc.runtime.bundle import load_runtime_bundle
    from .wbc.release_baseline_probe import load_release_inputs
    from .wbc.release_baseline import frozen_source
    output = writable_path(Path(output_root))
    if output.exists():
        raise FileExistsError("Choose a fresh Phase2 reference output")
    bundle = load_runtime_bundle(Path(bundle_path))
    inputs = load_release_inputs(bundle, Path(release_root), num_envs=16)
    predicate = json.loads(Path(predicate_path).read_text())
    if predicate["thresholds"]["version"] != "task10b_functional_guard_v1":
        raise ValueError("Reviewed Phase1 predicate required")
    output.mkdir(parents=True, exist_ok=False)
    planned = [{"candidate_id": family+"_original", "family": family,
                "kind": "original_control", "requested_ttc_s": None,
                "initial_disposition": "unattempted"}
               for family in CANDIDATE_TTCS]
    planned.extend({"candidate_id": family+"_uniform_"+f"{ttc:.3f}".replace(".", "p"),
                    "family": family, "kind": "uniform", "requested_ttc_s": ttc,
                    "initial_disposition": "unattempted"}
                   for family, ttcs in CANDIDATE_TTCS.items() for ttc in ttcs)
    (output/"planned_cells.json").write_text(json.dumps({"schema": "task10b_phase2_reference_plan_v1",
        "cells": planned, "timeout_accounting": "Every planned ID absent from the latest progress receipt remains unattempted; a created folder without a disposition is interrupted, never silently successful"}, indent=2)+"\n")
    rows = []
    def record_progress():
        attempted_ids = {row["candidate_id"] for row in rows}
        (output/f"progress_{len(rows):02d}.json").write_text(json.dumps({"cells": rows,
            "unattempted_cells": [row for row in planned if row["candidate_id"] not in attempted_ids],
            "complete": False}, indent=2)+"\n")
    record_progress()
    for family, requests in CANDIDATE_TTCS.items():
        origin = inputs[family]
        source_path = origin["paths"]["reference"]
        model_path = origin["paths"]["model"]
        model = mujoco.MjModel.from_xml_path(str(model_path))
        module, _ = frozen_source(Path(release_root), family)
        with np.load(source_path, allow_pickle=False) as archive:
            source = {key: archive[key].copy() for key in archive.files}
        legacy = bundle.path(origin["clip"]["legacy_runtime_path"])
        schedule = predicate["schedule"][family]
        common = {"family": family, "source_reference_sha256": sha256(source_path),
            "source_model_path": str(model_path), "source_model_sha256": sha256(model_path),
            "head_center": np.asarray(source["head_center"]).tolist(),
            "target_w": np.asarray(source["head_center"]).tolist(),
            "active_hand_index": int(source["active_hand_index"]), "split": "d0_development",
            "physical_qualification": False, "fine_tuned": False}
        original_duration = float(origin["clip"]["source_duration_s"])
        control_folder = output/(family+"_original_diagnostics")
        control_folder.mkdir(exist_ok=False)
        original_t = np.asarray(source.get("time_s", source.get("t")), dtype=float)
        control_screen_input = {"t": original_t, "qpos": source["qpos"],
            "qvel": source["qvel"], "qacc": np.gradient(source["qvel"], original_t, axis=0, edge_order=2)}
        control_screen, control_arrays = screen_diagnostics(model_path, control_screen_input,
            np.asarray(origin["schema"]["torque_limits"])[:, 1])
        control_screen.update(reference_identity="Unchanged accepted source pose/velocity grid; qacc differentiated from preserved original qvel only for diagnostics",
            sampling_time_step_min_max_s=[float(np.diff(original_t).min()), float(np.diff(original_t).max())],
            original_reference_sha256=sha256(source_path), runtime_reference_overwritten=False)
        np.savez_compressed(control_folder/"screen_arrays.npz", **control_arrays)
        (control_folder/"screen.json").write_text(json.dumps(control_screen, indent=2, allow_nan=False)+"\n")
        rows.append({**common, "candidate_id": family+"_original", "kind": "original_control",
            "requested_ttc_s": None, "reference_path": str(source_path), "reference_sha256": sha256(source_path),
            "legacy_runtime_path": str(legacy), "legacy_runtime_sha256": sha256(legacy),
            "duration_s": original_duration, "episode_duration_s": original_duration+1.,
            "phase_denominator_s": original_duration+1., "launch_tick": schedule["launch_actor_tick"],
            "launch_time_s": schedule["launch_time_s"], "outward_end_tick": schedule["outward_completion_tick"],
            "outward_completion_time_s": schedule["outward_completion_time_s"],
            "guard_time_s": float(source["guard_hold_start"]), "kinematic_admission": True,
            "rejection_reasons": [], "diagnostic_screen_path": str(control_folder/"screen.json"),
            "generation_disposition": "preserved_original_control_diagnostics_executed",
            "events_identity": "unchanged_phase1_reviewed_schedule", "preserved_original_control": True,
            "episode_clock_semantics": "Original unrounded source duration plus original 1.0s terminal hold is actor phase denominator; original evaluation horizon and explicit actor padding remain separate"})
        record_progress()
        for requested in requests:
            candidate_id = family+"_uniform_"+f"{requested:.3f}".replace(".", "p")
            folder = output/candidate_id
            folder.mkdir(exist_ok=False)
            try:
                candidate, search = solve_uniform_timing(model, source, requested)
                reference_path = folder/"reference.npz"
                # Compute original feedforward directly using the retimed qpos/qvel;
                # no acceleration compensation or optimized support force is applied.
                candidate["torque"] = module.support_feedforward(model, candidate,
                    np.asarray(origin["schema"]["dof_addresses"]))
                events = candidate.pop("events")
                candidate["punch_family"] = np.asarray(family)
                candidate["timing_kind"] = np.asarray("uniform_complete_reference_clock")
                candidate["source_reference_sha256"] = np.asarray(sha256(source_path))
                candidate["source_timing_modified"] = np.asarray(True)
                np.savez_compressed(reference_path, **candidate)
                screen, screen_arrays = screen_diagnostics(model_path, candidate,
                    np.asarray(origin["schema"]["torque_limits"])[:, 1])
                np.savez_compressed(folder/"screen_arrays.npz", **screen_arrays)
                (folder/"screen.json").write_text(json.dumps(screen, indent=2, allow_nan=False)+"\n")
                (folder/"search.json").write_text(json.dumps(search, indent=2, allow_nan=False)+"\n")
                (folder/"events.json").write_text(json.dumps(events, indent=2, allow_nan=False)+"\n")
                contact = search["selected_contact"]
                reasons = []
                if not contact["timing_valid"]:
                    reasons.append("uniform_reference_first_contact_timing_outside_contract")
                if contact["inactive_hand_first_or_simultaneous"]:
                    reasons.append("wrong_hand_first_or_simultaneous")
                if contact["any_contact_before_launch"]:
                    reasons.append("contact_before_launch")
                rows.append({**common, **events, "candidate_id": candidate_id, "kind": "uniform",
                    "requested_ttc_s": requested, "reference_path": str(reference_path),
                    "reference_sha256": sha256(reference_path), "legacy_runtime_path": str(reference_path),
                    "legacy_runtime_sha256": sha256(reference_path), "duration_s": float(candidate["t"][-1]),
                    "episode_duration_s": float(candidate["t"][-1])+1.,
                    "phase_denominator_s": float(candidate["t"][-1])+1.,
                    "guard_time_s": float(candidate["guard_hold_start"]),
                    "duration_scale": float(candidate["duration_scale"]),
                    "generation_disposition": "generated",
                    "kinematic_admission": not reasons, "rejection_reasons": reasons,
                    "diagnostic_screen_path": str(folder/"screen.json"),
                    "kinematic_contact": contact, "preserved_original_control": False})
            except (ValueError, RuntimeError) as error:
                failure = {**common, "candidate_id": candidate_id, "kind": "uniform",
                    "requested_ttc_s": requested, "kinematic_admission": False,
                    "rejection_reasons": ["reference_generation_failure"],
                    "generation_disposition": "failed",
                    "error_type": type(error).__name__, "error": str(error), "reference_path": None}
                (folder/"failure.json").write_text(json.dumps(failure, indent=2, allow_nan=False)+"\n")
                rows.append(failure)
            # Preserve a current ledger if a later candidate generation times out.
            record_progress()
    manifest = {"schema": "task10b_phase2_reference_catalog_v1", "complete": True,
        "scope": "D0 timing development only; no target/shared-start/final-test qualification",
        "cells": rows, "planned_cells": 16, "executed_cells": len(rows),
        "unattempted_cells": [],
        "reference_bytes": sum(p.stat().st_size for p in output.rglob("*.npz")),
        "initial_reference_byte_cap": 2*1024**3, "generation_source_sha256": sha256(Path(__file__)),
        "predicate_path": str(Path(predicate_path).resolve()), "predicate_sha256": sha256(Path(predicate_path)),
        "physics_executed": False, "learned_trajectory_generator": False,
        "fine_tuning_required_before_expanded_competence_claim": True}
    if manifest["reference_bytes"] > manifest["initial_reference_byte_cap"]:
        raise RuntimeError("Reference byte cap exceeded; preserve packet and reject catalog")
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return manifest
