"""Task 10B Phase 0 historical gates over saved native diagnostic evidence.

This adapter never integrates physics. It maps the isolated entity into the
unchanged Task 09 evaluator's unprefixed model, derives support from recorded
MuJoCo-Warp contact forces, and evaluates only the original physical horizon.
The full raw trace, including any final actor-rounding padding, stays intact.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .provenance import sha256_file, writable_path, write_json

EVALUATOR_SHA256 = "654cccd911eb6d04c5adcdc5eba5a09cb46817080d7dfe79fef92595db63fc66"
CONTACT_SHA256 = "4a9644712706445dc66432c502e7a21cc8d1065a0de03cf7641271a1ae583d1f"


def evaluator_identity():
    """Fail rather than silently replacing the historical measurement rules."""
    from human_punching import contact, jab_evaluation
    identities = {}
    for module, expected in ((jab_evaluation, EVALUATOR_SHA256), (contact, CONTACT_SHA256)):
        path = Path(module.__file__).resolve(strict=True)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Historical evaluator dependency changed: {path}")
        identities[module.__name__] = dict(path=str(path), sha256=actual)
    return identities


def historical_precision(metrics):
    """The three additional precision gates from the frozen release packager."""
    tracking = metrics.get("reference_tracking") or {}
    timing = metrics.get("contact_timing") or {}
    targets = metrics["precision_targets"]
    full = tracking.get("active_glove_position_rms_m")
    attack = tracking.get("attack_active_glove_position_rms_m")
    error = timing.get("absolute_interpolated_contact_error_s")
    return {
        "full_cycle_active_glove_rms_2cm": bool(full is not None and np.isfinite(full)
            and full <= .02 and targets["active_glove_rms_target_met"]),
        "attack_active_glove_rms_2cm": bool(attack is not None and np.isfinite(attack) and attack <= .02),
        "contact_timing_20ms": bool(error is not None and np.isfinite(error)
            and error <= .02 and targets["contact_timing_target_met"]),
    }


def recorded_support(model, arrays, worlds):
    """Decode either cone using recorded public contact-frame normal forces.

    The >1e-6 N threshold exactly matches the historical support predicate.
    This does not substitute the newer 1 N forefoot-patch diagnostic.
    Each row after row zero describes its preceding integration interval.
    """
    rows = len(arrays["time"])
    offsets = np.asarray(arrays["contact_offsets"])
    geom = np.asarray(arrays["contact_geom"])
    force = np.asarray(arrays["contact_force"])
    world = np.asarray(arrays["contact_worldid"])
    if (offsets.shape != (rows + 1,) or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
            or geom.shape != (int(offsets[-1]), 2) or force.shape != (len(geom), 6)
            or world.shape != (len(geom),) or not np.isfinite(force).all()
            or not np.issubdtype(world.dtype, np.integer)
            or np.any(world < 0) or np.any(world >= worlds)
            or np.any(geom < 0) or np.any(geom >= model.ngeom)):
        raise ValueError("Malformed recorded native contact forces or ownership")
    floor = model.geom("terrain").id
    feet = [model.geom(f"robot/physical/{side}Foot").id for side in ("Left", "Right")]
    result = np.zeros((rows, worlds, 2), bool)
    for tick in range(rows):
        span = slice(offsets[tick], offsets[tick + 1])
        pairs, loads, owners = geom[span], force[span, 0], world[span]
        positive_floor = (loads > 1.e-6) & np.any(pairs == floor, axis=-1)
        for side, foot in enumerate(feet):
            result[tick, owners[positive_floor & np.any(pairs == foot, axis=-1)], side] = True
    return result


def historical_model(model, inputs, destination):
    """Export unprefixed source body with the *actual* compiled controller/solver."""
    import mujoco
    from human_punching.scalar_control import configure_implicit_pd, export_runtime_model

    source = mujoco.MjModel.from_xml_path(str(inputs["paths"]["model"]))
    if (source.nq, source.nv, source.nu) != (model.nq, model.nv, model.nu):
        raise ValueError("Historical evaluation requires an isolated full-human model")
    configure_implicit_pd(source, inputs["schema"]["kp"], inputs["schema"]["kd"])
    for index in range(source.njnt):
        if model.joint("robot/" + source.joint(index).name).id != index:
            raise ValueError("Historical evaluation entity joint order differs")
    for index in range(source.nu):
        joint = source.joint(int(source.actuator_trnid[index, 0])).name
        if model.actuator("robot/" + joint + "_native_pd").id != index:
            raise ValueError("Historical evaluation entity actuator order differs")
    body_ids = np.asarray([0] + [model.body("robot/" + source.body(i).name).id
                                  for i in range(1, source.nbody)])
    for name in ("actuator_trnid", "actuator_trntype", "actuator_dyntype", "actuator_gaintype",
                 "actuator_biastype", "actuator_ctrllimited", "actuator_forcelimited",
                 "actuator_gear", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
                 "jnt_range", "jnt_pos", "jnt_axis", "dof_armature", "dof_damping", "dof_frictionloss"):
        if not np.array_equal(getattr(source, name), getattr(model, name)):
            raise ValueError(f"Historical evaluation controller/body differs: {name}")
    for name in ("body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat"):
        if not np.array_equal(getattr(source, name)[1:], getattr(model, name)[body_ids[1:]]):
            raise ValueError(f"Historical evaluation body mapping differs: {name}")
    for key in ("timestep", "impratio", "tolerance", "ls_tolerance", "iterations", "ls_iterations",
                "ccd_iterations", "integrator", "cone", "jacobian", "solver", "gravity",
                "disableflags", "enableflags"):
        setattr(source.opt, key, getattr(model.opt, key))
    export_runtime_model(inputs["paths"]["model"], destination, source)
    return body_ids


def evaluate_historical_panel(model, arrays, inputs, origins, output, *, residual_mode, solver_profile):
    """Save per-world historical evidence and return physical/precision counts.

    ``inputs`` is the hash-checked family record from load_release_inputs.
    Contract failures raise; physical failures remain complete counted trials.
    """
    if residual_mode not in ("learned", "zero") or solver_profile not in ("historical", "accepted"):
        raise ValueError("Unknown Phase 0 historical panel mode")
    identities = evaluator_identity()
    from human_punching.jab_evaluation import evaluate_rollout

    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    frames, worlds = np.asarray(arrays["time"]).shape
    origins = np.asarray(origins, dtype=np.float32)
    if origins.shape != (worlds, 3) or not np.isfinite(origins).all():
        raise ValueError("Invalid world origins")
    steps = int(np.ceil(float(inputs["clip"]["legacy_episode_duration_s"]) / .005))
    count = steps + 1
    if frames < count:
        raise ValueError("Incomplete original-horizon evidence cannot become a parity trial")
    expected = np.asarray(inputs["initial_qpos"], dtype=np.float32).copy()
    expected[:, :3] += origins
    if (not np.array_equal(arrays["qpos"][0], expected)
            or not np.array_equal(arrays["qvel"][0], np.asarray(inputs["initial_qvel"], np.float32))
            or not np.all(arrays["reset_counts"][:count] == 1)):
        raise ValueError("Historical initial bank/reset identity differs")
    if residual_mode == "zero" and np.any(arrays["policy_action"] != 0):
        raise ValueError("Zero-residual panel contains a nonzero policy action")
    runtime_path = destination / "historical_evaluation_runtime.xml"
    body_ids = historical_model(model, inputs, runtime_path)
    runtime_hash = sha256_file(runtime_path)
    support = recorded_support(model, arrays, worlds)
    bank_manifest = json.loads(Path(inputs["paths"]["bank_manifest"]).read_text())
    bank_hash = sha256_file(inputs["paths"]["bank"])
    bank = np.load(inputs["paths"]["bank"], allow_pickle=False)
    seed = int(bank["seed"]) if "seed" in bank else None
    trial_ids = np.asarray(inputs.get("trial_ids", np.arange(worlds)), dtype=int)
    if trial_ids.shape != (worlds,) or len(np.unique(trial_ids)) != worlds:
        raise ValueError("Invalid preserved bank trial identities")
    reports = []
    for world, trial_id in enumerate(trial_ids):
        qpos = np.asarray(arrays["qpos"][:count, world]).copy()
        qpos[:, :3] -= origins[world]
        policy_ticks = np.asarray(arrays["policy_reference_tick"][:, world])
        policy_ticks = policy_ticks[policy_ticks < steps]
        if not np.array_equal(policy_ticks, np.arange(0, steps, 4)):
            raise ValueError("Recorded historical actor boundary schedule differs")
        payload = {name: np.asarray(arrays[name][:count, world]).copy()
                   for name in ("qvel", "ctrl", "actuator_force", "qfrc_actuator", "qfrc_applied")}
        payload.update(qpos=qpos, t=np.arange(count)*.005, time_s=np.arange(count)*.005,
            simulation_time_s=np.asarray(arrays["time"][:count, world]),
            xfrc_applied=np.asarray(arrays["xfrc_applied"][:count, world])[:, body_ids],
            support_contact=support[:count, world], state_reset_count=1, physics_step_count=steps,
            physics_timestep_s=float(model.opt.timestep), policy_timestep_s=.020,
            policy_update_step_indices=policy_ticks, controller_mode="implicit_pd", reference_speed=1.,
            runtime_model_sha256=runtime_hash, evaluation_bank_sha256=bank_hash,
            evaluation_bank_world=int(trial_id), trial_id=str(trial_id),
            evaluation_kind="learned_policy" if residual_mode == "learned" else "reference_pd",
            model_sha256=inputs["input_sha256"]["model"], reference_sha256=inputs["input_sha256"]["reference"],
            checkpoint_sha256=inputs["input_sha256"]["checkpoint"] if residual_mode == "learned" else "",
            world_origin=origins[world], source_raw_frames=frames,
            normalization_note="float32 origin subtraction; native initial bank equality audited before subtraction")
        if seed is not None:
            payload["initial_state_seed"] = seed
        rollout = destination / f"rollout_{trial_id:02d}.npz"
        with rollout.open("xb") as stream:
            np.savez_compressed(stream, **payload)
        metrics = evaluate_rollout(runtime_path, rollout, inputs["paths"]["reference"],
            attack_window_reference_s=bank_manifest["attack_window_reference_s"],
            required_physics_timestep_s=.005, required_policy_timestep_s=.020)
        precision = historical_precision(metrics)
        strict = bool(metrics["pilot_pass"] and all(precision.values()))
        metrics.update(trial_id=int(trial_id), strict_historical_release_pass=strict,
                       historical_release_precision=precision)
        write_json(destination / f"trial_{trial_id:02d}.json", metrics)
        reports.append(dict(trial_id=int(trial_id), pilot_pass=metrics["pilot_pass"],
            strict_historical_release_pass=strict, precision=precision,
            failed_checks=metrics["failed_checks"], acceptance_checks=metrics["acceptance_checks"],
            first_contact_reset_s=metrics["first_sampled_contact_after_reset_s"],
            reference_tracking=metrics["reference_tracking"], contact_timing=metrics["contact_timing"],
            final_guard=metrics["final_guard"], root_height_min_m=metrics["root_height_min_m"],
            root_tilt_max_rad=metrics["root_tilt_max_rad"],
            joint_limit_violation_max_rad=metrics["joint_limit_violation_max_rad"],
            sole_position_drift_max_m=metrics["sole_position_drift_max_m"],
            actual_joint_torque_max_nm=metrics["actual_joint_torque_max_nm"],
            rollout_sha256=sha256_file(rollout)))
    bank.close()
    summary = dict(schema="task10b_phase0_historical_evaluation_v1", family=inputs["clip"]["family"],
        residual_mode=residual_mode, solver_profile=solver_profile, worlds=worlds, trial_count=worlds,
        trial_ids=trial_ids.tolist(), pilot_pass_count=sum(row["pilot_pass"] for row in reports),
        strict_historical_release_pass_count=sum(row["strict_historical_release_pass"] for row in reports),
        gate_pass_counts={key: sum(row["acceptance_checks"][key] for row in reports)
                          for key in reports[0]["acceptance_checks"]},
        precision_target_pass_counts={key: sum(row["precision"][key] for row in reports)
                                      for key in reports[0]["precision"]},
        historical_physics_steps=steps, historical_evaluation_duration_s=steps*.005,
        raw_recorded_physics_steps=frames-1, excluded_terminal_padding_steps=frames-count,
        historical_initial_bank_sha256=bank_hash, source_inputs=inputs["input_sha256"],
        runtime_model_sha256=runtime_hash, evaluator_source_identity=identities,
        adapter_sha256=sha256_file(__file__), training=False, learning_updates=0,
        scope="Historical fixed-reference exact-guard gates; no functional-recovery or requested-TTC claim",
        support_semantics="Public contact_force normal component >1e-6 N, both feet, preceding integration interval",
        clock_semantics="5 ms samples; native simulator clock audited; integer reference clock differs from historical float32 accumulation",
        trials=reports)
    write_json(destination / "summary.json", summary)
    return summary
