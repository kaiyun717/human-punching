#!/usr/bin/env python3
"""Independent, read-only Phase 0 evidence reconstruction; no physics stepping.

Writes only a new audit directory. The executed packet's archived sources and
inputs are authoritative, so later live-source edits are reported separately.
CPU actor inference and fresh FK do not replay or integrate the simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("jab", "cross", "left_hook", "right_hook")
SOLVERS = {"historical": (1, 30, 15, 1e-7), "accepted": (0, 10, 20, 1e-8)}
SOURCE_KEYS = {"checkpoint": "checkpoint.pt", "schema": "controller_schema.json",
    "arguments": "arguments.json", "bank": "evaluation/evaluation_bank.npz",
    "bank_manifest": "evaluation/evaluation_bank_manifest.json", "reference": "reference.npz",
    "model": "model/human.xml", "source": "jab_learning_source.py"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def expected_conditions(mode):
    if mode == "parity":
        return [(family, solver, True, residual) for family in FAMILIES
                for solver in SOLVERS for residual in ("learned", "zero")]
    if mode in ("contract", "diagnostic"):
        return [(family, solver, ff, "learned") for family in FAMILIES
                for solver, ff in (("historical", True), ("accepted", True), ("accepted", False))]
    raise ValueError("Unknown saved mode")


def condition_name(row, mode):
    family, solver, ff, residual = row
    return f"{family}_{solver}_{'ff' if ff else 'zero_ff'}" + ("_" + residual if mode == "parity" else "")


def strict_precision(metrics):
    tracking, timing = metrics.get("reference_tracking") or {}, metrics.get("contact_timing") or {}
    values = [tracking.get("active_glove_position_rms_m"), tracking.get("attack_active_glove_position_rms_m"),
              timing.get("absolute_interpolated_contact_error_s")]
    return bool(metrics["pilot_pass"] and all(v is not None and np.isfinite(v) and v <= .02 for v in values)
                and metrics["precision_targets"]["active_glove_rms_target_met"]
                and metrics["precision_targets"]["contact_timing_target_met"])


def interpolate(times, values, query):
    """Independent float32 implementation of the archived sample convention."""
    times, values, query = np.asarray(times, np.float32), np.asarray(values, np.float32), np.asarray(query, np.float32)
    query = np.clip(query, times[0], times[-1])
    upper = np.clip(np.searchsorted(times, query, side="right"), 1, len(times)-1)
    lower = upper-1
    alpha = (query-times[lower])/(times[upper]-times[lower])
    return values[lower] + alpha.reshape(*alpha.shape, *([1]*(values.ndim-1))) * (values[upper]-values[lower])


def observation(qpos, qvel, gloves, previous, clock, reference, duration):
    def sample(name, offset=0.):
        value = interpolate(reference["t"], reference[name], clock + np.float32(offset))
        if name == "qpos":
            value[:, 3:7] /= np.maximum(np.linalg.norm(value[:, 3:7], axis=-1, keepdims=True), 1e-8)
        return value
    desired, velocity = sample("qpos"), sample("qvel")
    phase = clock / np.float32(duration)
    parts = [qpos[:, 2:7], qvel[:, :6]*np.float32(.2), qpos[:, 7:], qvel[:, 6:]*np.float32(.1),
        (desired[:, :3]-qpos[:, :3])*5, desired[:, 3:7], desired[:, 7:]-qpos[:, 7:],
        (velocity[:, 6:]-qvel[:, 6:])*np.float32(.1),
        np.stack((phase, np.sin(phase*np.float32(2*np.pi)), np.cos(phase*np.float32(2*np.pi))), -1), previous]
    for offset in (.1, .25):
        future = sample("qpos", offset)
        parts.extend(((future[:, :3]-qpos[:, :3])*5, future[:, 7:]-qpos[:, 7:]))
    parts.append((sample("glove_pos")-gloves).reshape(len(qpos), -1)*5)
    return np.clip(np.concatenate(parts, axis=-1), -20, 20)


def audit_arrays(arrays, schema, master, initial_qpos, initial_qvel, *, feedforward, residual):
    """Reconstruct all intervals independently of the probe's own audit."""
    frames, worlds = arrays["time"].shape
    steps = frames-1
    policies = len(arrays.get("policy_action", []))
    checks, errors = {}, {}
    checks["finite_raw_evidence"] = all(np.isfinite(v).all() for v in arrays.values()
                                       if np.issubdtype(v.dtype, np.number))
    tolerance = max(5e-8, 8*np.finfo(np.float32).eps*float(np.abs(arrays["time"]).max()))
    errors["physics_interval_s"] = float(np.max(np.abs(np.diff(arrays["time"].astype(float), axis=0)-.005), initial=0))
    checks["real_native_clock"] = bool(np.all(arrays["time"][0] == 0) and errors["physics_interval_s"] <= tolerance)
    checks["complete_actor_intervals"] = steps == policies*4
    ticks = np.broadcast_to(np.arange(steps+1)[:, None], (frames, worlds))
    checks["reference_ticks"] = np.array_equal(arrays["reference_physics_steps"], ticks)
    checks["simulation_counter"] = np.array_equal(arrays["sim_step_index"], np.arange(frames))
    checks["policy_counter"] = np.array_equal(arrays["policy_step_index"], np.maximum(np.arange(frames)-1, 0)//4)
    checks["one_reset"] = bool(np.all(arrays["reset_counts"] == 1))
    checks["no_overflow"] = bool(np.all(arrays["overflow"] == 0))
    checks["unactuated_root"] = bool(np.all(np.abs(arrays["qfrc_actuator"][..., :6]) <= 1e-9))
    checks["no_external_assistance"] = bool(np.all(np.abs(arrays["qfrc_applied"]) <= 1e-9)
                                            and np.all(np.abs(arrays["xfrc_applied"]) <= 1e-9))
    origins = arrays["qpos"][0, :, :3] - initial_qpos.astype(np.float32)[:, :3]
    expected_q = initial_qpos.astype(np.float32).copy()
    expected_q[:, :3] += origins
    checks["initial_bank"] = bool(np.array_equal(arrays["qpos"][0], expected_q)
                                  and np.array_equal(arrays["qvel"][0], initial_qvel.astype(np.float32)))
    if not checks["complete_actor_intervals"] or not steps:
        return dict(checks=checks, errors=errors, steps=steps, worlds=worlds, policies=policies), origins
    action = arrays["policy_action"]
    held = np.repeat(action, 4, axis=0)
    checks["held_raw_action"] = bool(np.array_equal(arrays["pre_actor_action"], held)
                                     and np.array_equal(arrays["raw_action"][1:], held))
    checks["previous_raw_action"] = bool(np.all(arrays["policy_previous_raw_action"][0] == 0)
        and np.array_equal(arrays["policy_previous_raw_action"][1:], action[:-1]))
    checks["actor_boundaries"] = np.array_equal(arrays["policy_reference_tick"], ticks[:-1:4])
    checks["preintegration_ticks"] = np.array_equal(arrays["pre_reference_tick"], ticks[:-1])
    checks["force_interval_state_pairing"] = bool(np.array_equal(arrays["pre_qpos"], arrays["qpos"][:-1])
        and np.array_equal(arrays["pre_qvel"], arrays["qvel"][:-1]))
    ids = np.minimum(np.arange(steps), len(master["time_s"])-1)
    qref, vref = master["qpos"][ids].astype(np.float32), master["qvel"][ids].astype(np.float32)
    ff = master["joint_feedforward"][ids].astype(np.float32) if feedforward else np.zeros((steps, 48), np.float32)
    checks["immutable_references"] = bool(np.allclose(arrays["pre_qref"][..., 7:], qref[:, None, 7:], atol=2e-6, rtol=0)
        and np.allclose(arrays["pre_vref"], vref[:, None], atol=2e-6, rtol=0)
        and np.allclose(arrays["pre_feedforward"], ff[:, None], atol=5e-6, rtol=0))
    kp, kd, caps = np.asarray(schema["kp"]), np.asarray(schema["kd"]), np.asarray(schema["torque_limits"])[:, 1]
    # The compiled source joint limits are supplied by the caller, independently
    # of the recorded target and drive.
    limits = np.asarray(schema["_compiled_joint_limits"])
    target = np.clip(qref[:, None, 7:] + .25*np.clip(held, -1, 1), limits[:, 0], limits[:, 1])
    drive = kp*target+kd*vref[:, None, 6:]+ff[:, None]
    force = np.clip(arrays["ctrl"][1:]-kp*arrays["pre_qpos"][..., 7:]-kd*arrays["pre_qvel"][..., 6:], -caps, caps)
    for name, actual, expected, bound in (
        ("joint_target", arrays["joint_position_target"][1:], target, 2e-6),
        ("joint_velocity_target", arrays["joint_velocity_target"][1:], vref[:, None, 6:], 2e-6),
        ("joint_feedforward", arrays["joint_effort_target"][1:], ff[:, None], 5e-6),
        ("native_drive", arrays["ctrl"][1:], drive, 2e-4),
        ("native_force", arrays["actuator_force"][1:], force, 5e-4),
        ("generalized_force", arrays["qfrc_actuator"][..., 6:], arrays["actuator_force"], 1e-6)):
        errors[name] = float(np.abs(actual-expected).max(initial=0))
        checks[name] = errors[name] <= bound
    errors["effort_cap_excess_nm"] = float(np.maximum(np.abs(arrays["actuator_force"])-caps, 0).max(initial=0))
    checks["effort_caps"] = errors["effort_cap_excess_nm"] <= 1e-4
    checks["zero_residual"] = residual != "zero" or bool(np.all(action == 0))
    return dict(checks=checks, errors=errors, steps=steps, worlds=worlds, policies=policies), origins


def sampled_actor(model, arrays, origins, reference, duration, checkpoint, residual, sample_count):
    import mujoco
    import torch
    from torch import nn
    policies, worlds = arrays["policy_action"].shape[:2]
    selected = np.unique(np.linspace(0, policies-1, min(sample_count, policies)).astype(int))
    rows = selected*4
    qpos = arrays["qpos"][rows].copy()
    qvel = arrays["qvel"][rows].copy()
    hands = [model.site("robot/"+side+"Hand_center").id for side in ("Left", "Right")]
    data = mujoco.MjData(model)
    gloves = np.empty((len(rows), worlds, 2, 3), np.float32)
    for sample in range(len(rows)):
        for world in range(worlds):
            data.qpos[:] = qpos[sample, world]
            mujoco.mj_kinematics(model, data)
            gloves[sample, world] = data.site_xpos[hands].astype(np.float32)-origins[world]
    qpos[:, :, :3] -= origins
    clock = (arrays["policy_reference_tick"][selected].astype(np.float32)*np.float32(.005)).reshape(-1)
    rebuilt = observation(qpos.reshape(-1, 55), qvel.reshape(-1, 54), gloves.reshape(-1, 2, 3),
        arrays["policy_previous_raw_action"][selected].reshape(-1, 48), clock, reference, duration)
    recorded = arrays["policy_observation"][selected].reshape(-1, 369)
    obs_error = float(np.abs(rebuilt-recorded).max(initial=0))
    # Use recorded observations to isolate inference arithmetic from FK error.
    if residual == "learned":
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        actor = nn.Sequential(nn.Linear(369, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 48))
        actor.load_state_dict({k.removeprefix("actor."): v for k, v in saved["policy"].items() if k.startswith("actor.")})
        with torch.no_grad():
            action = actor(torch.from_numpy(recorded)).numpy()
    else:
        action = np.zeros((len(recorded), 48), np.float32)
    error = float(np.abs(action-arrays["policy_action"][selected].reshape(-1, 48)).max(initial=0))
    return dict(policy_indices=selected.tolist(), worlds=worlds, sampled_observations=len(recorded),
        maximum_observation_error=obs_error, observation_tolerance=5e-5,
        maximum_cpu_actor_output_error=error, actor_tolerance=5e-5,
        checks={"fresh_fk_observations": obs_error <= 5e-5, "frozen_actor_outputs": error <= 5e-5},
        scope="Deterministic actor-boundary samples across every world; fresh CPU FK and CPU inference, no integration")


def recorded_origin_matches_initial(raw_initial, bank_initial, recorded_origin):
    """Check forward float32 translation; subtraction need not recover an origin.

    For example float32(a + 1) - a can differ from one by one ULP. The
    authoritative source bank and recorded origin must reproduce raw qpos
    exactly before that origin is used to link normalized rollout evidence.
    """
    origin = np.asarray(recorded_origin)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        return False
    expected = np.asarray(bank_initial, np.float32).copy()
    expected[:3] += origin.astype(np.float32)
    return bool(np.array_equal(raw_initial, expected))


def parity_evidence(folder, arrays, condition, origins, archive, files, model, initial_qpos, expected_steps):
    """Verify every persisted trial and re-evaluate nominal/last trial independently."""
    summary = read_json(folder/"historical_evaluation/summary.json")
    worlds = arrays["qpos"].shape[1]
    checks = {"full_trial_coverage": summary["trial_ids"] == list(range(worlds)) and summary["trial_count"] == worlds,
        "historical_evaluation_uses_original_horizon": summary["historical_physics_steps"] == expected_steps}
    pilot, strict, rows = 0, 0, []
    prefix = Path("src/human_punching")
    for name in ("jab_evaluation.py", "contact.py"):
        if digest(ROOT/prefix/name) != files[str(prefix/name)]:
            raise ValueError("Live independent evaluator differs from executed packet archive")
    sys.path.insert(0, str(ROOT/"src"))
    evaluator = importlib.import_module("human_punching.jab_evaluation")
    family = condition["family"]
    reference = archive/"outputs/four-punch-milestone"/family/"reference.npz"
    bank_manifest = read_json(archive/"outputs/four-punch-milestone"/family/"evaluation/evaluation_bank_manifest.json")
    steps = int(summary["historical_physics_steps"])
    import mujoco
    evaluation_model = mujoco.MjModel.from_xml_path(str(folder/"historical_evaluation/historical_evaluation_runtime.xml"))
    native_body_ids = [0] + [model.body("robot/"+evaluation_model.body(i).name).id
                             for i in range(1, evaluation_model.nbody)]
    support = np.zeros((steps+1, worlds, 2), bool)
    feet = [model.geom("robot/physical/"+side+"Foot").id for side in ("Left", "Right")]
    floor = model.geom("terrain").id
    for tick in range(steps+1):
        start, end = arrays["contact_offsets"][tick:tick+2]
        geoms, forces, owners = (arrays["contact_"+key][start:end] for key in ("geom", "force", "worldid"))
        for side, foot in enumerate(feet):
            selected = np.any(geoms == floor, axis=1) & np.any(geoms == foot, axis=1) & (forces[:, 0] > 1e-6)
            support[tick, owners[selected], side] = True
    by_trial = {row["trial_id"]: row for row in summary["trials"]}
    for trial in range(worlds):
        path = folder/f"historical_evaluation/rollout_{trial:02d}.npz"
        payload = read_npz(path)
        metrics = read_json(folder/f"historical_evaluation/trial_{trial:02d}.json")
        saved_origin = payload["world_origin"]
        expected_q = arrays["qpos"][:steps+1, trial].copy()
        expected_q[:, :3] -= saved_origin
        ok = digest(path) == by_trial[trial]["rollout_sha256"]
        ok &= recorded_origin_matches_initial(arrays["qpos"][0, trial], initial_qpos[trial], saved_origin)
        ok &= np.allclose(saved_origin, origins[trial], atol=1e-6, rtol=0)
        ok &= np.array_equal(payload["qpos"], expected_q)
        ok &= all(np.array_equal(payload[key], arrays[key][:steps+1, trial]) for key in
                  ("qvel", "ctrl", "actuator_force", "qfrc_actuator", "qfrc_applied"))
        ok &= np.array_equal(payload["xfrc_applied"], arrays["xfrc_applied"][:steps+1, trial][:, native_body_ids])
        ok &= np.array_equal(payload["support_contact"], support[:, trial])
        ok &= str(payload["trial_id"].item()) == str(trial)
        ok &= int(payload["physics_step_count"].item()) == steps
        computed_strict = strict_precision(metrics)
        ok &= computed_strict == metrics["strict_historical_release_pass"] == by_trial[trial]["strict_historical_release_pass"]
        ok &= metrics["pilot_pass"] == by_trial[trial]["pilot_pass"]
        pilot += int(metrics["pilot_pass"])
        strict += int(computed_strict)
        if trial in (0, worlds-1):
            again = evaluator.evaluate_rollout(folder/"historical_evaluation/historical_evaluation_runtime.xml", path, reference,
                attack_window_reference_s=bank_manifest["attack_window_reference_s"],
                required_physics_timestep_s=.005, required_policy_timestep_s=.020)
            ok &= (again["acceptance_checks"] == metrics["acceptance_checks"] and strict_precision(again) == computed_strict)
        rows.append(bool(ok))
    checks.update(all_trial_evidence_matches_raw=all(rows), counts_match=pilot == summary["pilot_pass_count"]
                  and strict == summary["strict_historical_release_pass_count"])
    return dict(checks=checks, trial_count=worlds, pilot_pass_count=pilot,
        strict_historical_release_pass_count=strict, independently_reevaluated_trials=[0, worlds-1],
        raw_padding_steps=arrays["qpos"].shape[0]-1-steps)


def audit_packet(packet_path, sample_count):
    import mujoco
    import torch
    torch.set_num_threads(1)
    packet_path = Path(packet_path).resolve(strict=True)
    packet = read_json(packet_path)
    archive = packet_path.parent/"source_snapshot"
    files = packet["files"]
    archive_bad, live_changed = [], []
    for relative, expected in files.items():
        archived = (archive/relative).resolve(strict=True)
        if not archived.is_relative_to(archive.resolve()) or digest(archived) != expected:
            archive_bad.append(relative)
        live = ROOT/relative
        if not live.is_file() or digest(live) != expected:
            live_changed.append(relative)
    checks = {"all_archived_input_hashes": not archive_bad, "phase0_no_learning_scope": packet["program_phase"] == 0
        and packet["training"] is False and packet["learning_updates"] == 0}
    if archive_bad:
        raise ValueError("Archived input bytes changed: " + ", ".join(archive_bad))
    execution_path = packet_path.parent/"execution.json"
    execution = read_json(execution_path) if execution_path.is_file() else {}
    checks["execution_binds_packet"] = execution.get("packet_sha256") == digest(packet_path)
    checks["execution_completed_all_modes"] = execution.get("status") == "passed" and execution.get("completed_modes") == packet["modes"]
    health_summary = None
    health_path = packet_path.parent/"cuda_health.json"
    if health_path.is_file() or "cuda_health_sha256" in execution:
        health = read_json(health_path)
        measured = health.get("torch", {}).get("result", {})
        checks["cuda_health_receipt_bound"] = (digest(health_path) == execution.get("cuda_health_sha256")
            and health["status"] == execution.get("cuda_health_status")
            and health["selected_environment"]["SLURM_JOB_ID"] == execution.get("slurm_job_id"))
        checks["cuda_health_has_no_physics_or_learning"] = all(health.get(k) == 0 for k in
            ("physics_steps", "actor_updates", "learning_updates"))
        checks["cuda_health_passed"] = (health["status"] == "passed" and measured.get("initialized") is True
            and measured.get("tensor_roundtrip") is True and measured.get("cuda_available") is True
            and measured.get("torch_version") == packet["runtime"]["packages"]["torch"])
        uuid = measured.get("device_uuid")
        checks["health_gpu_identity_in_driver_receipt"] = bool(uuid and uuid != "unavailable"
            and uuid in health.get("nvidia_smi", {}).get("stdout", "") and measured.get("device_name") == "NVIDIA L40S")
        health_summary = dict(path=str(health_path), sha256=digest(health_path), status=health["status"],
            hostname=health["hostname"], device_uuid=uuid, device_name=measured.get("device_name"))
    bundle = archive/packet["bundle"]
    manifest = read_json(bundle/"manifest.json")
    clips = {c["family"]: c for c in manifest["clips"]}
    modes = []
    for mode in packet["modes"]:
        path = ROOT/packet["output"]/mode
        audit_path = path/"audit.json"
        expected = expected_conditions(mode)
        if not audit_path.is_file():
            modes.append(dict(mode=mode, complete=False, missing_conditions=len(expected), conditions=[]))
            checks[mode+"_complete"] = False
            continue
        reported = read_json(audit_path)
        records = reported["conditions"]
        complete = len(records) == len(expected) and reported.get("status") == "passed"
        identities = [(r["family"], r["solver_profile"], r["feedforward_enabled"], r["residual_mode"]) for r in records]
        mode_checks = dict(exact_ordered_condition_coverage=identities == expected,
            missing_conditions_honest=reported["unattempted_conditions"] == len(expected)-len(records),
            no_learning=reported["training"] is False and reported["includes_ppo"] is False)
        receipt = next((entry for entry in execution.get("modes", []) if entry.get("mode") == mode), {})
        mode_checks["execution_receipt_binds_mode_audit"] = receipt.get("audit_sha256") == digest(audit_path)
        outcomes = []
        pair_initial = {}
        for condition in records:
            name, family = condition["name"], condition["family"]
            folder = path/name
            result = dict(name=name, family=family, solver=condition["solver_profile"], residual=condition["residual_mode"])
            try:
                arrays = read_npz(folder/"raw_trace.npz")
                model_path = folder/"compiled_model.mjb"
                model = mujoco.MjModel.from_binary_path(str(model_path))
                source = archive/packet["release"]/family
                schema = read_json(source/"controller_schema.json")
                schema["_compiled_joint_limits"] = model.jnt_range[1:].copy()
                bank = read_npz(source/"evaluation/evaluation_bank.npz")
                worlds = 64 if mode == "parity" else 16
                data, origins = audit_arrays(arrays, schema, read_npz(bundle/clips[family]["master_path"]),
                    bank["qpos"][:worlds], bank["qvel"][:worlds],
                    feedforward=condition["feedforward_enabled"], residual=condition["residual_mode"])
                result.update(data)
                c = result["checks"]
                # Match the archived original trainer's literal ceil, including
                # cross 4.800000000000001/.005 -> 961. Actor rounding cannot
                # remove that original final physical interval.
                original_steps = int(np.ceil(float(clips[family]["legacy_episode_duration_s"])/.005))
                result["original_horizon"] = dict(source_rule="ceil(original_duration / 0.005)",
                    reference_duration_s=clips[family]["legacy_episode_duration_s"],
                    historical_physics_steps=original_steps,
                    complete_actor_interval_physics_steps=((original_steps+3)//4)*4,
                    actual_minus_original_physics_steps=data["steps"]-original_steps)
                if mode == "parity":
                    c["historical_full_horizon_recorded"] = data["steps"] >= original_steps
                c["raw_trace_hash"] = digest(folder/"raw_trace.npz") == condition["raw_trace_sha256"]
                c["compiled_model_hash"] = digest(model_path) == condition["compiled_model"]["sha256"]
                c["standalone_condition_receipt_matches"] = read_json(folder/"audit.json") == condition
                c["input_hashes"] = all(digest(source/relative) == reported["source_inputs"][family][key]
                                        for key, relative in SOURCE_KEYS.items())
                cone, iterations, ls, tolerance = SOLVERS[condition["solver_profile"]]
                c["actual_compiled_solver"] = (int(model.opt.integrator), int(model.opt.solver), int(model.opt.cone),
                    int(model.opt.iterations), int(model.opt.ls_iterations), model.opt.tolerance, model.opt.timestep
                    ) == (3, 2, cone, iterations, ls, tolerance, .005)
                c["actual_compiled_controller"] = bool(model.nq == 55 and model.nv == 54 and model.nu == 48
                    and np.all(model.actuator_gainprm[:, 0] == 1)
                    and np.array_equal(model.actuator_biasprm[:, 1], -np.asarray(schema["kp"]))
                    and np.array_equal(model.actuator_biasprm[:, 2], -np.asarray(schema["kd"]))
                    and np.array_equal(model.actuator_forcerange, schema["torque_limits"])
                    and np.all(model.actuator_forcelimited) and not np.any(model.actuator_ctrllimited)
                    and np.array_equal(model.actuator_trnid[:, 0], np.arange(1, 49)))
                c["trial_ids"] = condition["trial_ids"] == list(range(worlds))
                c["actual_recorded_counts"] = condition["actual_physics_steps"] == data["steps"]
                c["actual_recorded_counts"] &= condition["actual_completed_policy_steps"] == data["policies"]
                c["actual_recorded_counts"] &= condition["actual_preintegration_records"] == data["steps"]
                c["actual_recorded_counts"] &= condition["actual_postintegration_records"] == data["steps"]
                c["actual_recorded_counts"] &= condition["actual_actor_updates"] == data["policies"]
                c["actual_recorded_counts"] &= condition["actual_physics_steps_per_world"] == [data["steps"]]*worlds
                c["actual_recorded_counts"] &= condition["actual_integrated_duration_s"] == data["steps"]*.005
                c["native_elapsed_receipt_matches_trace"] = np.array_equal(
                    np.asarray(condition["actual_native_elapsed_s"]),
                    arrays["time"][-1].astype(float)-arrays["time"][0].astype(float))
                c["reported_contract_pass"] = condition["status"] == "passed" and all(condition["audit"]["checks"].values())
                if data["policies"] and data["checks"]["complete_actor_intervals"]:
                    result["sampled_actor"] = sampled_actor(model, arrays, origins,
                        read_npz(bundle/clips[family]["legacy_runtime_path"]), clips[family]["legacy_episode_duration_s"],
                        source/"checkpoint.pt", condition["residual_mode"], sample_count)
                    c.update(result["sampled_actor"]["checks"])
                if mode == "parity":
                    result["historical"] = parity_evidence(folder, arrays, condition, origins, archive, files, model,
                        bank["qpos"][:worlds], original_steps)
                    c.update(result["historical"]["checks"])
                    key = (family, condition["solver_profile"])
                    state = (arrays["qpos"][0], arrays["qvel"][0])
                    if condition["residual_mode"] == "learned":
                        pair_initial[key] = state
                    else:
                        c["matched_learned_zero_initial_state"] = key in pair_initial and all(
                            np.array_equal(a, b) for a, b in zip(state, pair_initial[key]))
                result["passed"] = all(c.values())
            except Exception as error:
                result.update(passed=False, error={"type": type(error).__name__, "message": str(error)})
            outcomes.append(result)
        mode_checks["every_saved_condition_independently_passes"] = all(r["passed"] for r in outcomes)
        if all("steps" in result for result in outcomes):
            mode_checks["aggregate_world_transition_count"] = reported["actual_world_physics_transitions"] == sum(
                result["steps"]*result["worlds"] for result in outcomes)
        checks[mode+"_complete"] = complete and all(mode_checks.values())
        modes.append(dict(mode=mode, complete=complete, checks=mode_checks,
            requested_conditions=len(expected), completed_conditions=len(records),
            missing_conditions=len(expected)-len(records), conditions=outcomes))
    return dict(schema="task10b_phase0_independent_audit_v1", packet=str(packet_path), packet_sha256=digest(packet_path),
        auditor_sha256=digest(__file__), status="passed" if all(checks.values()) else "failed_or_incomplete",
        checks=checks, archived_files_verified=len(files), later_live_source_changes=live_changed,
        cuda_health=health_summary,
        sampled_policy_boundaries_per_condition=sample_count, modes=modes,
        no_physics_integration=True, no_learning=True,
        physical_success_is_audit_success=False,
        limitations="All interval/controller/history checks; deterministic sampled observation/actor reconstruction; all parity ledgers checked with nominal and last trials independently re-evaluated. Physical failed trials remain failures.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-policy-boundaries", type=int, default=9)
    args = parser.parse_args(argv)
    destination = args.output.resolve()
    if (not destination.is_relative_to((ROOT/"outputs/task10b").resolve()) or destination.exists()
            or args.sample_policy_boundaries < 1):
        raise ValueError("A new Task 10B output directory and positive sample count are required")
    destination.mkdir(parents=True, exist_ok=False)
    source_copy = destination/"auditor_source.py"
    source_copy.write_bytes(Path(__file__).read_bytes())
    result = audit_packet(args.packet, args.sample_policy_boundaries)
    if digest(source_copy) != result["auditor_sha256"]:
        raise ValueError("Independent auditor source changed during the audit")
    result["auditor_source_snapshot"] = source_copy.name
    with (destination/"audit.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(destination/"audit.json"),
                      "checks": result["checks"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
