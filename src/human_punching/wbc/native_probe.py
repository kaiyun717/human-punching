"""Bounded native clock/controller diagnostic for a later allocated GPU run.

This is neither punch acceptance nor PPO capacity measurement. Derived MuJoCo
forces at the post-step metric hook describe the preceding integration interval;
only integrated qpos/qvel/time are current. Glove positions are recomputed by
offline CPU kinematics after recording, without stepping or replacing a pose.
"""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .provenance import runtime_record, sha256_file, writable_path, write_json

SCHEMA = "human_wbc_native_probe_v1"
RAW_FIELDS = ("time", "qpos", "qvel", "ctrl", "actuator_force", "qfrc_actuator",
              "qfrc_applied", "xfrc_applied", "overflow")


def _copy(value) -> np.ndarray:
    value = getattr(value, "_tensor", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


class NativeProbeRecorder:
    """A supported MetricsTerm callback, including its standard reset hook."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.reset_counts = np.zeros(num_envs, dtype=np.int64)
        self.rows: list[dict[str, np.ndarray]] = []

    def note_reset(self, env_ids=None) -> None:
        if env_ids is None:
            ids = slice(None)
        elif isinstance(env_ids, slice):
            ids = env_ids
        else:
            ids = _copy(env_ids).astype(int)
        self.reset_counts[ids] += 1

    def capture(self, env) -> None:
        command = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        row = {name: _copy(getattr(env.sim.data, name)) for name in RAW_FIELDS}
        row.update(reference_physics_steps=_copy(command.physics_steps),
                   trajectory_id=_copy(command.trajectory_ids),
                   joint_position_target=_copy(robot.data.joint_pos_target),
                   joint_velocity_target=_copy(robot.data.joint_vel_target),
                   joint_effort_target=_copy(robot.data.joint_effort_target),
                   raw_action=_copy(env.action_manager.action),
                   reset_counts=self.reset_counts.copy(),
                   sim_step_index=np.asarray(env._sim_step_counter),
                   policy_step_index=np.asarray(env.common_step_counter))
        self.rows.append(row)

    def metric_function(self):
        # A plain closure survives manager/config deepcopy without duplicating
        # recorder state. MetricsManager explicitly supports a callable.reset.
        def record(env):
            import torch
            self.capture(env)
            return torch.zeros(env.num_envs, device=env.device)

        record.reset = self.note_reset
        return record

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.rows:
            return {}
        return {name: np.stack([row[name] for row in self.rows]) for name in self.rows[0]}


def build_probe_cfg(bundle, recorder: NativeProbeRecorder, *, policy_steps: int = 50):
    """Construct only; this function never creates a simulation or runs physics."""
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from .runtime.task import build_human_env_cfg

    cfg = build_human_env_cfg(bundle, num_envs=recorder.num_envs, evaluation=True)
    cfg.terminations = {}  # Untrained zero residual is a controller diagnostic.
    cfg.rewards = {}       # There is no optimization or imitation-score claim.
    cfg.metrics = {"native_probe": MetricsTermCfg(func=recorder.metric_function(), per_substep=True)}
    cfg.episode_length_s = max(10., (policy_steps + 1) * .020)
    cfg.seed = 10010
    return cfg


def audit_probe_arrays(arrays: dict[str, np.ndarray], *, policy_steps: int,
                       physics_dt_s: float, effort_limits: np.ndarray,
                       root_dof_start: int = 0) -> dict[str, Any]:
    """Pure CPU audit. Failure does not become a guessed successful measurement."""
    required = set(RAW_FIELDS) | {
        "reference_physics_steps", "reset_counts", "sim_step_index",
        "policy_step_index", "raw_action", "trajectory_id",
        "joint_position_target", "joint_velocity_target", "joint_effort_target",
    }
    if required - set(arrays):
        return {"status": "failed", "checks": {"required_fields": False},
                "missing_fields": sorted(required - set(arrays))}
    finite = all(np.isfinite(value).all() for value in arrays.values()
                 if np.issubdtype(value.dtype, np.number))
    timestamps = np.asarray(arrays["time"], dtype=float)
    expected_count = policy_steps * 4 + 1
    shape_ok = timestamps.ndim == 2 and len(timestamps) == expected_count
    clock_ok, actor_clock_ok, clock_error = False, False, None
    if shape_ok and np.isfinite(timestamps).all():
        # MuJoCo-Warp time is float32: measure against one actual .005 step,
        # allowing its accumulator roundoff, not an alternate integration dt.
        tolerance = max(5e-8, 8*np.finfo(np.float32).eps*float(np.abs(timestamps).max()))
        clock_error = float(np.max(np.abs(np.diff(timestamps, axis=0) - .005)))
        clock_ok = clock_error <= tolerance
        actor_clock_ok = bool(np.allclose(np.diff(timestamps[::4], axis=0), .020,
                                         atol=4*tolerance, rtol=0))
    reference = np.asarray(arrays["reference_physics_steps"])
    reference_ok = (shape_ok and reference.shape == timestamps.shape
                    and np.array_equal(reference, np.broadcast_to(
                        np.arange(expected_count)[:, None], reference.shape)))
    sim_steps = np.asarray(arrays["sim_step_index"])
    sim_count_ok = (sim_steps.shape == (expected_count,) and np.array_equal(
        sim_steps - sim_steps[0], np.arange(expected_count)))
    # The metric hook runs before the environment increments its policy-step
    # counter. Rows 1..4 belong to actor interval 0, rows 5..8 to interval 1.
    policy_steps_recorded = np.asarray(arrays["policy_step_index"])
    expected_policy_steps = np.maximum(np.arange(expected_count) - 1, 0) // 4
    policy_count_ok = (policy_steps_recorded.shape == (expected_count,)
                      and np.array_equal(policy_steps_recorded, expected_policy_steps))
    caps = np.asarray(effort_limits, dtype=float)
    forces = np.asarray(arrays["actuator_force"], dtype=float)
    caps_valid = (caps.ndim == 1 and np.isfinite(caps).all() and (caps > 0).all()
                  and forces.ndim == 3 and forces.shape[-1] == len(caps))
    cap_excess = float(np.max(np.abs(forces) - caps)) if caps_valid else None
    env_count = timestamps.shape[1] if timestamps.ndim == 2 else 0
    joint_shape = (expected_count, env_count, len(caps)) if caps.ndim == 1 else ()
    joint_fields = ("ctrl", "actuator_force", "raw_action", "joint_position_target",
                    "joint_velocity_target", "joint_effort_target")
    joint_shapes_ok = all(np.asarray(arrays[name]).shape == joint_shape for name in joint_fields)
    state_shapes_ok = (np.asarray(arrays["qpos"]).shape == (expected_count, env_count, 55)
                       and np.asarray(arrays["qvel"]).shape == (expected_count, env_count, 54)
                       and np.asarray(arrays["qfrc_actuator"]).shape == (expected_count, env_count, 54)
                       and np.asarray(arrays["qfrc_applied"]).shape == (expected_count, env_count, 54))
    body_force = np.asarray(arrays["xfrc_applied"])
    wrench_shape_ok = (body_force.ndim == 4 and body_force.shape[:2] == (expected_count, env_count)
                       and body_force.shape[2] > 0 and body_force.shape[3] == 6)
    trajectory = np.asarray(arrays["trajectory_id"])
    trajectory_ok = (trajectory.shape == timestamps.shape and shape_ok
                     and np.all(trajectory == trajectory[0]))
    root_force = np.asarray(arrays["qfrc_actuator"])[..., root_dof_start:root_dof_start+6]
    checks = dict(required_fields=True, finite=bool(finite),
                  actual_model_timestep=physics_dt_s == .005,
                  sample_count=bool(shape_ok), physics_clock=bool(clock_ok),
                  twenty_ms_action_clock=bool(actor_clock_ok),
                  policy_counter=bool(policy_count_ok),
                  recorded_state_and_controller_shapes=bool(joint_shapes_ok and state_shapes_ok and wrench_shape_ok),
                  zero_learned_residual=bool(joint_shapes_ok and np.all(arrays["raw_action"] == 0)),
                  committed_trajectory_id=bool(trajectory_ok),
                  one_reference_sample_per_physics_step=bool(reference_ok),
                  native_step_counter=bool(sim_count_ok),
                  one_initial_reset=bool(np.asarray(arrays["reset_counts"]).shape == timestamps.shape
                                         and np.all(arrays["reset_counts"] == 1)),
                  actual_actuator_force_caps=bool(caps_valid and cap_excess <= 1e-4),
                  no_native_overflow=bool(np.all(arrays["overflow"] == 0)),
                  unactuated_root=bool(root_force.shape[-1] == 6
                                      and np.all(np.abs(root_force) <= 1e-9)),
                  zero_external_generalized_force=bool(np.all(np.abs(arrays["qfrc_applied"]) <= 1e-9)),
                  zero_external_body_wrench=bool(np.all(np.abs(arrays["xfrc_applied"]) <= 1e-9)))
    qpos = np.asarray(arrays["qpos"])
    fall_count = None
    if qpos.ndim == 3 and qpos.shape[-1] >= 7 and np.isfinite(qpos).all():
        quat = qpos[..., 3:7]
        tilt = np.arccos(np.clip(1-2*(quat[..., 1]**2+quat[..., 2]**2), -1, 1))
        falls = (qpos[..., 2] < .75*qpos[0, :, 2]) | (tilt > .60)
        fall_count = int(falls.any(axis=0).sum())
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "maximum_physics_clock_error_s": clock_error,
            "maximum_force_cap_excess_nm": cap_excess if cap_excess is not None and np.isfinite(cap_excess) else None,
            "fall_diagnostic_environments": fall_count,
            "fall_diagnostic_is_acceptance_gate": False}


def _require_allocated_cuda() -> None:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Native probe requires an existing explicitly allocated Slurm CUDA job")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Allocated CUDA is unavailable; there is no CPU dynamics fallback")


def _configure_caches(destination: Path) -> None:
    for variable, child in (("XDG_CACHE_HOME", "xdg"), ("WARP_CACHE_PATH", "warp"),
                            ("CUDA_CACHE_PATH", "cuda"), ("TORCH_HOME", "torch"),
                            ("TMPDIR", "tmp"), ("MPLCONFIGDIR", "matplotlib")):
        path = writable_path(destination / "cache" / child)
        path.mkdir(parents=True, exist_ok=False)
        os.environ[variable] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def _gloves_from_qpos(model, qpos: np.ndarray) -> np.ndarray:
    import mujoco
    ids = [model.site("robot/" + side + "Hand_center").id for side in ("Left", "Right")]
    data = mujoco.MjData(model)
    result = np.empty((*qpos.shape[:2], 2, 3))
    for sample in range(qpos.shape[0]):
        for world in range(qpos.shape[1]):
            data.qpos[:] = qpos[sample, world]
            mujoco.mj_kinematics(model, data)
            result[sample, world] = data.site_xpos[ids]
    return result


def run_native_probe(bundle, output, envs: int = 16, policy_steps: int = 50) -> dict[str, Any]:
    """Execute ONLY when called inside an authorized existing GPU allocation.

    One initial reset, zero learned residual, no assistance, no optimizer. The
    retained trace is deliberately not an evaluate_episode acceptance schema.
    """
    if type(envs) is not int or not 1 <= envs <= 64:
        raise ValueError("A bounded native probe requires 1..64 environments")
    if type(policy_steps) is not int or not 1 <= policy_steps <= 1000:
        raise ValueError("A bounded native probe requires 1..1000 policy steps")
    _require_allocated_cuda()
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise RuntimeError("Native probe runtime does not match the pinned Task 10 candidate")
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    _configure_caches(destination)
    from .runtime.bundle import load_runtime_bundle
    checked = load_runtime_bundle(bundle)
    recorder = NativeProbeRecorder(envs)
    cfg = build_probe_cfg(checked, recorder, policy_steps=policy_steps)
    cfg.sim.nan_guard.output_dir = str(destination / "nan_guard")
    report = {"schema": SCHEMA, "status": "starting", "runtime": runtime,
              "reference_manifest_sha256": sha256_file(checked.root / "manifest.json"),
              "model_sha256": sha256_file(checked.model_path),
              "controller_sha256": checked.controller_sha256(),
              "environments": envs, "requested_policy_steps": policy_steps,
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "simulation_contract": asdict(cfg.sim.mujoco),
              "action_mode": "zero_learned_residual_reference_controller",
              "includes_ppo": False, "qualified_for_training": False,
              "complete_cycle_acceptance": False,
              "trace_semantics": {
                  "qpos_qvel_time": "current integrated post-step state",
                  "forces_ctrl_targets": "preceding integration interval; initial row is reset forward",
                  "glove_position_w_fk": "CPU kinematics from recorded integrated qpos; no physics steps",
                  "missing_acceptance_evidence": "support-patch kinematics, collision classification and guard-frame motion",
                  "reference_physics_steps": "count after applying sample k-1 and integrating interval k",
                  "performance": "CPU copies synchronize every substep; not a throughput or PPO capacity test"}}
    write_json(destination / "launch.json", report)
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    env, model, error = None, None, None
    started = time.monotonic()
    try:
        device = torch.device("cuda:0")
        stream = torch.cuda.ExternalStream(wp.get_stream("cuda:0").cuda_stream, device=device)
        caller = torch.cuda.current_stream(device)
        stream.wait_stream(caller)
        with torch.cuda.stream(stream), torch.no_grad():
            env = ManagerBasedRlEnv(cfg, device="cuda:0")
            model = env.sim.mj_model
            env.reset(seed=10010)
            recorder.capture(env)
            command = env.command_manager.get_term("motion")
            initial_targets = _copy(command.target_w)
            initial_families = _copy(command.family_ids[command.trajectory_ids])
            action = torch.zeros((envs, env.action_manager.total_action_dim), device=device)
            for _ in range(policy_steps):
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
    report["wall_time_s"] = time.monotonic() - started
    arrays = recorder.arrays()
    if arrays and model is not None:
        if np.isfinite(arrays["qpos"]).all():
            arrays["glove_position_w_fk"] = _gloves_from_qpos(model, arrays["qpos"])
        if "initial_targets" in locals():
            arrays["target_w"] = initial_targets
            arrays["family_id"] = initial_families
        free = np.flatnonzero(model.jnt_type == 0)
        root_dof = int(model.jnt_dofadr[free[0]]) if len(free) == 1 else -1
        audit = audit_probe_arrays(arrays, policy_steps=policy_steps,
                                   physics_dt_s=float(model.opt.timestep),
                                   effort_limits=model.actuator_forcerange[:, 1],
                                   root_dof_start=root_dof)
        report.update(audit)
        with (destination / "raw_trace.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        report["raw_trace_sha256"] = sha256_file(destination / "raw_trace.npz")
        report["recorded_samples"] = len(arrays["time"])
    else:
        report.update(status="failed", checks={"recorded_samples": False})
    if error is not None:
        report.update(status="failed", exception=error)
    write_json(destination / "audit.json", report)
    return report
