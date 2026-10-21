"""Physical qualification of one static common guard, separate from punching.

Six seconds of native 5 ms physics, 20 ms zero residual, one initial reset and
zero joint feedforward by default, or an explicit hashed static-support LP
feedforward artifact. No punch is launched and no targeting/TTC result is
computed. The final continuous second must be quiet, supported and in guard.
A passed fixture never qualifies a target/TTC domain or reference for training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .cycle_trace import CycleTraceRecorder
from .evaluation import EvaluationThresholds
from .native_cycle_probe import (ENVIRONMENTS, POLICY_STEPS, ProbeMetrics,
    _events, _probe_command_types, _save_npz, allocated_device_metadata,
    build_cycle_probe_cfg, deterministic_clip_ids, initialize_launch_drive,
    placed_request, save_compiled_model)
from .native_probe import _configure_caches, _copy, _require_allocated_cuda, audit_probe_arrays
from .provenance import runtime_record, sha256_file, writable_path, write_json

SCHEMA = "human_wbc_static_guard_probe_v1"


def common_guard_identity(bundle) -> dict:
    """Require the actual same initial qpos across all four source families."""
    rows, records = [], []
    for clip in bundle.clips:
        with np.load(bundle.path(clip["master_path"]), allow_pickle=False) as master:
            qpos = np.asarray(master["qpos"][0], dtype=np.float64)
        if qpos.shape != (55,) or not np.isfinite(qpos).all():
            raise ValueError("Finite constrained-human initial qpos is required")
        if clip.get("guard_source") != "explicit_common_guard":
            raise ValueError("Static guard qualification requires an explicit common guard")
        rows.append(qpos)
        records.append({"reference_id": clip["reference_id"], "family": clip["family"],
                        "master_sha256": clip["master_sha256"],
                        "initial_qpos_sha256": hashlib.sha256(qpos.astype("<f8").tobytes()).hexdigest()})
    if set(clip["family"] for clip in bundle.clips) != {"jab", "cross", "left_hook", "right_hook"}:
        raise ValueError("Common guard needs four-family source coverage")
    maximum_difference = float(np.max(np.abs(np.asarray(rows)-rows[0])))
    if maximum_difference > 1e-9:
        raise ValueError("Source families do not share the same initial common qpos")
    native_controller = {**bundle.controller_contract(), "feedforward_mode": "zero_joint_feedforward"}
    native_controller_sha256 = hashlib.sha256(json.dumps(native_controller, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"qpos": rows[0].tolist(), "qpos_sha256": records[0]["initial_qpos_sha256"],
            "qpos_dtype": "little-endian float64", "qvel": [0.]*54,
            "maximum_source_qpos_difference": maximum_difference,
            "unique_guard_configurations": 1, "independent_guard_replicates": False,
            "source_references": records, "feedforward_mode": "zero_joint_feedforward",
            "model_sha256": sha256_file(bundle.model_path),
            "source_controller_sha256": bundle.controller_sha256(),
            "native_guard_controller_contract": native_controller,
            "native_guard_controller_sha256": native_controller_sha256}


def _guard_command_types():
    import torch
    probe_command, probe_cfg = _probe_command_types()

    class StaticGuardCommand(probe_command):
        def sample_master(self, name: str, offset_steps: int = 0):
            del offset_steps
            value = self.master[name][self.master_starts[self.trajectory_ids]]
            if name == "joint_feedforward":
                configured = getattr(getattr(self, "cfg", None), "static_guard_feedforward_nm", ())
                if configured:
                    if not hasattr(self, "_static_guard_feedforward"):
                        self._static_guard_feedforward = torch.as_tensor(configured, dtype=value.dtype, device=value.device)
                    return self._static_guard_feedforward.expand_as(value)
            if name in ("qvel", "joint_feedforward", "body_lin_vel_w", "body_ang_vel_w",
                        "anchor_lin_acc_w", "anchor_ang_acc_w"):
                return torch.zeros_like(value)
            return value

        def _update_command(self, env_ids):
            ids = slice(None) if env_ids is None else env_ids
            self.time_steps[ids] = self.motion.segment_start_idx[self.trajectory_ids[ids]]
            self.update_relative_body_poses()

    @dataclass(kw_only=True)
    class StaticGuardCommandCfg(probe_cfg):
        static_guard_feedforward_nm: tuple[float, ...] = ()

        def build(self, env):
            return StaticGuardCommand(self, env)

    return StaticGuardCommand, StaticGuardCommandCfg


def build_guard_probe_cfg(bundle, metrics: ProbeMetrics, *, feedforward=None):
    common_guard_identity(bundle)
    cfg = build_cycle_probe_cfg(bundle, metrics)
    _, cfg_type = _guard_command_types()
    command = cfg.commands["motion"]
    cfg.commands["motion"] = cfg_type(**{field.name: getattr(command, field.name) for field in fields(command)})
    if feedforward is not None:
        torque = np.asarray(feedforward, dtype=float)
        if torque.shape != (48,) or not np.isfinite(torque).all() or np.any(abs(torque) > np.asarray(bundle.body["effort"])+1e-8):
            raise ValueError("Static guard feedforward must contain 48 finite capped joint torques")
        cfg.commands["motion"].static_guard_feedforward_nm = tuple(torque.tolist())
    return cfg


def audit_static_drive(native: dict, bundle, expected_feedforward: np.ndarray) -> dict:
    """Independently reconstruct the native drive, including declared FF."""
    target = np.asarray(native["joint_position_target"], float)
    velocity = np.asarray(native["joint_velocity_target"], float)
    recorded_ff = np.asarray(native["joint_effort_target"], float)
    actual = np.asarray(native["ctrl"], float)
    expected_ff = np.asarray(expected_feedforward, float)
    if (target.ndim != 3 or target.shape[-1] != 48 or target.shape != velocity.shape
            or target.shape != recorded_ff.shape or target.shape != actual.shape
            or expected_ff.shape != (48,) or not all(np.isfinite(a).all() for a in (target, velocity, recorded_ff, actual, expected_ff))):
        return {"status": "failed", "reason": "invalid_native_drive_arrays"}
    reference = np.asarray(common_guard_identity(bundle)["qpos"])[7:]
    ff_error = float(np.max(abs(recorded_ff-expected_ff)))
    target_error = float(np.max(abs(target-reference)))
    predicted = np.asarray(bundle.body["kp"])*target + np.asarray(bundle.body["kd"])*velocity + expected_ff
    drive_error = float(np.max(abs(actual-predicted)))
    checks = {"declared_joint_feedforward": ff_error <= 5e-6, "constant_guard_target": target_error <= 2e-6,
              "zero_reference_velocity": bool(np.all(velocity == 0)), "native_affine_drive_includes_feedforward": drive_error <= 2e-4}
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "maximum_feedforward_error_nm": ff_error, "maximum_target_error_rad": target_error,
            "maximum_drive_error_nm": drive_error, "drive_is_not_effort_clipped": True}


def _angle(quaternions, reference):
    return 2*np.arccos(np.clip(abs(np.sum(quaternions*reference, axis=-1)), 0., 1.))


def evaluate_guard_trace(trace: dict, head_position_w, *, native_contract_passed: bool) -> dict:
    """Pure offline guard checks; no use of punch family, contact TTC or rewards."""
    thresholds = EvaluationThresholds()
    required = {"time_s", "root_position_w", "root_quat_wxyz", "root_linear_velocity_w",
        "root_angular_velocity_w", "joint_position", "joint_velocity", "joint_limits",
        "effort", "effort_limits", "glove_position_w", "glove_position_guard_frame",
        "glove_quat_guard_frame", "glove_velocity_guard_frame", "support_anchor_position_w",
        "support_contact", "support_patch_contact", "external_wrench", "external_generalized_force",
        "root_actuator_force", "external_support", "reset_event", "numerical_failure",
        "contact_overflow", "forbidden_collision", "human_g1_collision", "evidence_kind"}
    missing = sorted(required-set(trace))
    if missing:
        return {"guard_admission_passed": False, "failure_categories": ["missing_evidence"], "missing": missing}
    a = {key: np.asarray(value) for key, value in trace.items() if isinstance(value, np.ndarray)}
    time_s = np.asarray(trace["time_s"], float)
    if time_s.shape != (1201,) or not np.allclose(time_s-time_s[0], np.arange(1201)*.005, atol=1e-9, rtol=0):
        return {"guard_admission_passed": False, "failure_categories": ["incomplete_physics_clock"]}
    for name in required-{"evidence_kind"}:
        values = np.asarray(trace[name])
        if not np.issubdtype(values.dtype, np.number) and values.dtype.kind != "b":
            return {"guard_admission_passed": False, "failure_categories": ["malformed_evidence"], "field": name}
        if not np.isfinite(values).all():
            return {"guard_admission_passed": False, "failure_categories": ["nonfinite_evidence"], "field": name}
    shapes = {"root_position_w": (1201, 3), "root_quat_wxyz": (1201, 4),
        "root_linear_velocity_w": (1201, 3), "root_angular_velocity_w": (1201, 3),
        "glove_position_w": (1201, 2, 3), "glove_position_guard_frame": (1201, 2, 3),
        "glove_quat_guard_frame": (1201, 2, 4), "glove_velocity_guard_frame": (1201, 2, 3),
        "support_anchor_position_w": (1201, 2, 3), "support_contact": (1201, 2),
        "support_patch_contact": (1201, 2), "root_actuator_force": (1201, 6),
        "external_generalized_force": (1201, 54), "joint_position": (1201, 48),
        "joint_velocity": (1201, 48), "joint_limits": (48, 2),
        "effort": (1201, 48), "effort_limits": (48,)}
    boolean_fields = {"support_contact", "support_patch_contact", "external_support", "reset_event",
                      "numerical_failure", "contact_overflow", "forbidden_collision", "human_g1_collision"}
    shapes.update({name: (1201,) for name in boolean_fields if name not in shapes})
    for name, shape in shapes.items():
        if a[name].shape != shape or (name in boolean_fields and a[name].dtype.kind != "b"):
            return {"guard_admission_passed": False, "failure_categories": ["malformed_evidence"], "field": name}
    wrench = a["external_wrench"]
    if wrench.ndim != 3 or wrench.shape[0] != 1201 or wrench.shape[1] < 1 or wrench.shape[2] != 6:
        return {"guard_admission_passed": False, "failure_categories": ["malformed_evidence"], "field": "external_wrench"}
    if (np.any(a["effort_limits"] <= 0) or np.any(a["joint_limits"][:, 0] >= a["joint_limits"][:, 1])
            or np.any(abs(np.linalg.norm(a["root_quat_wxyz"], axis=-1)-1.) > 1e-6)
            or np.any(abs(np.linalg.norm(a["glove_quat_guard_frame"], axis=-1)-1.) > 1e-6)):
        return {"guard_admission_passed": False, "failure_categories": ["invalid_limits_or_quaternion"]}
    final = time_s-time_s[0] >= 5.-1e-10
    root = a["root_position_w"]
    quat = a["root_quat_wxyz"]
    tilt = np.arccos(np.clip(1-2*(quat[:, 1]**2+quat[:, 2]**2), -1., 1.))
    q, limits = a["joint_position"], a["joint_limits"]
    joint_excess = np.maximum(limits[:, 0]-q, q-limits[:, 1])
    cap_excess = abs(a["effort"])-a["effort_limits"]
    anchors = a["support_anchor_position_w"]
    drift = np.linalg.norm(anchors[..., :2]-anchors[0, :, :2], axis=-1)
    loaded = a["support_patch_contact"].astype(bool)
    loaded_drift = float(drift[loaded].max(initial=0.))
    distance = np.linalg.norm(a["glove_position_w"]-np.asarray(head_position_w), axis=-1)
    root_error = np.linalg.norm(root-root[0], axis=-1)
    root_angle = _angle(quat, quat[0])
    hand_error = np.linalg.norm(a["glove_position_guard_frame"]-a["glove_position_guard_frame"][0], axis=-1)
    hand_angle = _angle(a["glove_quat_guard_frame"], a["glove_quat_guard_frame"][0])
    checks = {
        "complete_native_clock_controller_contract": bool(native_contract_passed),
        "native_guard_hold_evidence": trace["evidence_kind"] == "physics_guard_hold",
        "no_fall": bool(np.all(root[:, 2] >= .75*root[0, 2]) and np.all(tilt <= .60)),
        "last_second_double_support": bool(a["support_contact"][final].all()),
        "last_second_root_pose": bool(np.all(root_error[final] <= thresholds.guard_root_position_m)
            and np.all(root_angle[final] <= thresholds.guard_root_orientation_rad)),
        "last_second_joint_guard": bool(np.all(abs(q[final]-q[0]) <= thresholds.guard_joint_position_rad)),
        "last_second_glove_guard": bool(np.all(hand_error[final] <= thresholds.guard_glove_relative_position_m)
            and np.all(hand_angle[final] <= thresholds.guard_glove_relative_orientation_rad)),
        "last_second_root_quiet": bool(np.all(np.linalg.norm(a["root_linear_velocity_w"][final], axis=-1) <= thresholds.guard_root_linear_speed_m_s)
            and np.all(np.linalg.norm(a["root_angular_velocity_w"][final], axis=-1) <= thresholds.guard_root_angular_speed_rad_s)),
        "last_second_joints_quiet": bool(np.all(abs(a["joint_velocity"][final]) <= thresholds.guard_joint_speed_rad_s)),
        "last_second_gloves_quiet": bool(np.all(np.linalg.norm(a["glove_velocity_guard_frame"][final], axis=-1) <= thresholds.guard_glove_relative_speed_m_s)),
        "glove_clearance": bool(np.all(distance >= .16)),
        "loaded_forefoot_drift": loaded_drift <= thresholds.maximum_support_anchor_drift_m,
        "last_second_footprint": bool(np.all(drift[final] <= thresholds.maximum_support_anchor_drift_m)),
        "static_footprint_all_times": bool(np.all(drift <= thresholds.maximum_support_anchor_drift_m)),
        "joint_limits": bool(np.all(joint_excess <= thresholds.maximum_joint_limit_violation_rad)),
        "effort_caps": bool(np.all(cap_excess <= thresholds.maximum_effort_limit_violation)),
        "zero_external_body_wrenches": bool(np.all(abs(a["external_wrench"]) <= 1e-9)),
        "zero_external_generalized_force": bool(np.all(abs(a["external_generalized_force"]) <= 1e-9)),
        "unactuated_root": bool(np.all(abs(a["root_actuator_force"]) <= 1e-9)),
        "no_assistance": bool(not a["external_support"].any()),
        "one_initial_reset": bool(a["reset_event"][0] and not a["reset_event"][1:].any()),
        "no_numerical_failure": bool(not a["numerical_failure"].any()),
        "no_contact_overflow": bool(not a["contact_overflow"].any()),
        "no_forbidden_collision": bool(not a["forbidden_collision"].any()),
        "no_g1_collision": bool(not a["human_g1_collision"].any()),
    }
    return {"guard_admission_passed": all(checks.values()), "checks": checks,
        "failure_categories": [key for key, passed in checks.items() if not passed],
        "quiet_window_start_s": 5., "quiet_window_duration_s": 1.,
        "punch_execution_tested": False, "targeting_tested": False, "ttc_tested": False,
        "qualified_for_training": False, "target_ttc_domain_qualified": False,
        "thresholds": asdict(thresholds), "metrics": {
            "initial_double_support": bool(a["support_contact"][0].all()),
            "initial_support_contact": a["support_contact"][0].tolist(),
            "initial_support_is_admission_gate": False,
            "initial_support_semantics": "descriptive reset state before unassisted physical settling; only the final quiet second must remain supported",
            "maximum_root_tilt_rad": float(tilt.max()), "minimum_glove_clearance_m": float(distance.min()),
            "maximum_loaded_forefoot_drift_m": loaded_drift,
            "maximum_final_footprint_drift_m": float(drift[final].max()),
            "maximum_joint_limit_excess_rad": float(max(0., joint_excess.max())),
            "maximum_effort_cap_excess_nm": float(max(0., cap_excess.max())),
            "last_second_max_root_position_error_m": float(root_error[final].max()),
            "last_second_max_root_orientation_error_rad": float(root_angle[final].max()),
            "last_second_max_glove_guard_error_m": float(hand_error[final].max()),
            "last_second_max_joint_guard_error_rad": float(abs(q[final]-q[0]).max()),
            "last_second_max_root_speed_m_s": float(np.linalg.norm(a["root_linear_velocity_w"][final], axis=-1).max()),
            "last_second_max_root_angular_speed_rad_s": float(np.linalg.norm(a["root_angular_velocity_w"][final], axis=-1).max()),
            "last_second_max_joint_speed_rad_s": float(abs(a["joint_velocity"][final]).max()),
            "last_second_max_glove_speed_m_s": float(np.linalg.norm(a["glove_velocity_guard_frame"][final], axis=-1).max()),
            "contact_evidence_semantics": "forces/support at row k describe preceding integration interval"}}


def run_guard_probe(bundle, output, *, feedforward_artifact: Path | None = None) -> dict:
    _require_allocated_cuda()
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise RuntimeError("Guard probe requires the pinned Task 10 runtime")
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    _configure_caches(destination)
    from .runtime.bundle import load_runtime_bundle
    checked = load_runtime_bundle(bundle)
    guard = common_guard_identity(checked)
    metrics = ProbeMetrics()
    feedforward, artifact = np.zeros(48), None
    if feedforward_artifact is not None:
        from .guard_feedforward import load_guard_feedforward
        from .provenance import PROJECT_ROOT
        import shutil
        artifact = load_guard_feedforward(feedforward_artifact, checked)
        feedforward = np.asarray(artifact["joint_feedforward_nm"])
        with (destination/"static_guard_feedforward.json").open("xb") as stream:
            stream.write(Path(feedforward_artifact).read_bytes())
        if sha256_file(destination/"static_guard_feedforward.json") != sha256_file(feedforward_artifact):
            raise ValueError("Static feedforward artifact changed during archival")
        for name, expected_hash in artifact["source_sha256"].items():
            archived = writable_path(destination/"feedforward_sources"/name)
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT/"src/human_punching/wbc"/name, archived)
            if sha256_file(archived) != expected_hash:
                raise ValueError("Feedforward source changed during archival")
        guard["feedforward_mode"] = artifact["mode"]
        guard["native_guard_controller_contract"] = {**checked.controller_contract(),
            "feedforward_mode": artifact["mode"], "feedforward_artifact_sha256": sha256_file(feedforward_artifact),
            "static_guard_qpos_sha256": guard["qpos_sha256"], "joint_feedforward_nm": feedforward.tolist()}
        guard["native_guard_controller_sha256"] = hashlib.sha256(json.dumps(
            guard["native_guard_controller_contract"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cfg = build_guard_probe_cfg(checked, metrics, feedforward=feedforward if artifact is not None else None)
    executed_solver = {"mujoco": asdict(cfg.sim.mujoco), "decimation": cfg.decimation,
                       "nconmax": cfg.sim.nconmax, "njmax": cfg.sim.njmax}
    if artifact is not None and json.dumps(executed_solver, sort_keys=True) != json.dumps(
            artifact["executed_solver_contract"], sort_keys=True):
        raise ValueError("Actual guard-probe solver differs from the feedforward artifact")
    cfg.sim.nan_guard.output_dir = str(destination / "nan_guard")
    report = dict(schema=SCHEMA, status="starting", runtime=runtime,
        environments=ENVIRONMENTS, duration_s=6., policy_steps=POLICY_STEPS,
        guard_identity=guard, simulation_contract=asdict(cfg.sim.mujoco),
        executed_solver_contract=executed_solver,
        action_mode="zero_residual_static_guard", joint_feedforward_mode="zero" if artifact is None else guard["feedforward_mode"],
        feedforward_artifact_sha256=None if artifact is None else sha256_file(feedforward_artifact),
        includes_ppo=False, qualified_for_training=False, target_ttc_domain_qualified=False,
        punch_execution_tested=False, targeting_tested=False, ttc_tested=False,
        inter_character_collision_enabled=False, g1_entity_present=False,
        reference_manifest_sha256=sha256_file(checked.root / "manifest.json"),
        slurm_job_id=os.environ["SLURM_JOB_ID"],
        time_origin="guard diagnostic start; no punch launch",
        source_context_semantics="PunchRequest is only a frozen provenance/head-clearance carrier; no request is executed",
        contact_api="mujoco_warp.contact_force; contact-frame forces from preceding integration interval",
        state_write_scope="local command reset/write/reset_to_frame; no events/automatic reset; all applied forces recorded",
        performance_scope="synchronized diagnostic recording, not PPO throughput")
    write_json(destination / "launch.json", report)
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    env, command, recorder, error = None, None, None, None
    clips, origins = [], None
    started = time.monotonic()
    try:
        device = torch.device("cuda:0")
        report["allocated_device"] = allocated_device_metadata(device)
        stream = torch.cuda.ExternalStream(wp.get_stream("cuda:0").cuda_stream, device=device)
        caller = torch.cuda.current_stream(device)
        stream.wait_stream(caller)
        with torch.cuda.stream(stream), torch.no_grad():
            env = ManagerBasedRlEnv(cfg, device="cuda:0")
            report["compiled_scene"] = save_compiled_model(env.sim.mj_model, destination / "compiled_scene.mjb")
            env.reset(seed=10010)
            command = env.command_manager.get_term("motion")
            command.write_ledger.seal()
            report["initial_drive"] = initialize_launch_drive(env)
            if np.any(_copy(env.sim.data.qvel) != 0):
                raise RuntimeError("Static guard initialization must have exactly zero physical velocity")
            ids = _copy(command.trajectory_ids).astype(int)
            np.testing.assert_array_equal(ids, deterministic_clip_ids(command.clips, np.arange(ENVIRONMENTS)))
            clips = [command.clips[i] for i in ids]
            origins, times = _copy(env.scene.env_origins), _copy(env.sim.data.time)
            expected_qpos = np.broadcast_to(np.asarray(guard["qpos"]), (ENVIRONMENTS, 55)).copy()
            expected_qpos[:, :3] += origins
            np.testing.assert_allclose(_copy(env.sim.data.qpos), expected_qpos, atol=2e-6, rtol=0)
            report["initial_guard_qpos_matches_input"] = True
            requests = [placed_request(clip, origin, timestamp)
                        for clip, origin, timestamp in zip(clips, origins, times, strict=True)]
            write_json(destination / "source_contexts.json", [asdict(request) for request in requests])
            recorder = CycleTraceRecorder(requests, event_provider=_events)
            metrics.attach(recorder)
            metrics.native.capture(env)
            recorder.capture_launch(env)
            action = torch.zeros((ENVIRONMENTS, env.action_manager.total_action_dim), device=device)
            for _ in range(POLICY_STEPS):
                env.step(action)
        caller.wait_stream(stream)
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exception:
                error = error or {"type": type(exception).__name__, "message": str(exception)}
    report["simulation_wall_time_s"] = time.monotonic()-started
    raw = recorder.raw_arrays() if recorder is not None else {}
    native = metrics.native.arrays()
    report["raw_trace_sha256"] = _save_npz(destination / "raw_trace.npz", raw)
    report["native_controller_trace_sha256"] = _save_npz(destination / "native_controller_trace.npz", native)
    report["recorded_samples"] = len(raw.get("time", []))
    report["status"] = "failed"
    report["guard_admission_passed"] = False
    if command is not None:
        report["state_write_ledger"] = {"reset_counts": command.write_ledger.resets.tolist(),
            "pose_initialization_attempts": command.write_ledger.writes.tolist(),
            "blocked_post_launch_attempts": command.write_ledger.blocked.tolist(), "sealed": command.write_ledger.sealed}
    if recorder is not None and recorder.model is not None and native:
        model = recorder.model
        root = np.flatnonzero(model.jnt_type == 0)
        audit = audit_probe_arrays(native, policy_steps=POLICY_STEPS,
            physics_dt_s=float(model.opt.timestep), effort_limits=model.actuator_forcerange[:, 1],
            root_dof_start=int(model.jnt_dofadr[root[0]]) if len(root) == 1 else -1)
        static_checks = {
            "constant_guard_targets": bool(np.all(native["joint_position_target"] == native["joint_position_target"][0])),
            "zero_reference_velocity": bool(np.all(native["joint_velocity_target"] == 0)),
            "declared_joint_feedforward": bool(np.allclose(native["joint_effort_target"], feedforward, atol=5e-6, rtol=0)),
            "zero_initial_physical_velocity": bool(np.all(native["qvel"][0] == 0)),
        }
        if artifact is None:
            static_checks["zero_joint_feedforward"] = bool(np.all(native["joint_effort_target"] == 0))
        drive_audit = audit_static_drive(native, checked, feedforward)
        native_pass = audit["status"] == "passed" and all(static_checks.values()) and drive_audit["status"] == "passed"
        report.update(clock_controller_audit=audit, static_reference_checks=static_checks, independent_drive_audit=drive_audit)
        outcomes = []
        for index, (request, clip) in enumerate(zip(recorder.requests, clips, strict=True)):
            item = {"environment": index, "source_family": request.family.value, "reference_id": clip["reference_id"]}
            try:
                with np.load(checked.path(clip["master_path"]), allow_pickle=False) as master:
                    reference = np.broadcast_to(master["glove_pos_w"][0]+origins[index], (len(recorder.rows), 2, 3)).copy()
                trace = recorder.episode_arrays(index, reference_glove_position_w=reference,
                    joint_names=checked.joint_names, outward_end_s=0., evidence_kind="physics_guard_hold")
                # The guard schema deliberately contains no outward phase or TTC result.
                trace.pop("outward_end_s")
                trace["task"] = "static_guard_hold"
                item["trace_sha256"] = _save_npz(destination / f"guard_{index:02d}.npz", trace)
                item["result"] = evaluate_guard_trace(trace, request.snapshot.head_position_w,
                                                       native_contract_passed=native_pass)
                item["status"] = "evaluated"
            except Exception as exception:
                item.update(status="evaluation_failed", exception={"type": type(exception).__name__, "message": str(exception)})
            outcomes.append(item)
        report["episodes"] = outcomes
        complete = native_pass and len(outcomes) == ENVIRONMENTS and all(item["status"] == "evaluated" for item in outcomes)
        report["status"] = "recorded" if complete else "failed"
        report["guard_admission_passed"] = complete and all(item["result"]["guard_admission_passed"] for item in outcomes)
    if error is not None:
        report.update(status="failed", guard_admission_passed=False, exception=error)
    report["total_wall_time_s"] = time.monotonic()-started
    write_json(destination / "audit.json", report)
    return report


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--environments", type=int, choices=(ENVIRONMENTS,), default=ENVIRONMENTS)
    parser.add_argument("--seconds", type=float, choices=(6.,), default=6.)
    parser.add_argument("--feedforward-artifact", type=Path,
                        help="Explicit hashed static support-LP artifact; default is zero joint feedforward")
    args = parser.parse_args(argv)
    result = run_guard_probe(args.bundle, args.output, feedforward_artifact=args.feedforward_artifact)
    print(json.dumps({"status": result["status"], "guard_admission_passed": result["guard_admission_passed"],
                      "punch_execution_tested": False}, sort_keys=True))
    return 0 if result["status"] == "recorded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
