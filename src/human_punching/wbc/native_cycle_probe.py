"""Bounded six-second, zero-residual diagnostics; never trains or qualifies data.

Run only inside an allocated CUDA job. Every reference remains unqualified. The
full trace is retained; deadline-prefix evaluations are descriptive, not Task 10
acceptance. Native contact forces describe the preceding integration interval;
current geometry comes from CPU FK of the recorded integrated state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import os
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .contracts import LaunchSnapshot, PunchRequest
from .cycle_trace import CycleTraceRecorder
from .evaluation import evaluate_episode
from .native_probe import (NativeProbeRecorder, _configure_caches, _copy,
                           _require_allocated_cuda, audit_probe_arrays)
from .provenance import runtime_record, sha256_file, writable_path, write_json

SCHEMA = "human_wbc_native_cycle_probe_v1"
ENVIRONMENTS = 16
POLICY_STEPS = 300
FAMILIES = ("jab", "cross", "left_hook", "right_hook")


def deterministic_clip_ids(clips: Sequence[dict], env_ids) -> np.ndarray:
    """Round robin families; successive groups select variants by reference ID."""
    candidates = [[i for i, clip in sorted(enumerate(clips),
                   key=lambda item: item[1]["reference_id"])
                   if clip["family"] == family] for family in FAMILIES]
    if any(not group for group in candidates):
        raise ValueError("The cycle probe requires all four families")
    ids = np.asarray(env_ids, dtype=int)
    if ids.ndim != 1 or (ids < 0).any():
        raise ValueError("Nonnegative one-dimensional environment IDs are required")
    return np.asarray([candidates[e % 4][(e // 4) % len(candidates[e % 4])]
                       for e in ids], dtype=np.int64)


def placed_request(clip: dict, origin: np.ndarray, launch_time_s: float) -> PunchRequest:
    """Translate the actual frozen G1 payload with its parallel-world placement."""
    if not clip.get("g1_snapshot_present") or not clip.get("launch_snapshot"):
        raise ValueError("Cycle diagnostics require actual launch snapshot payloads")
    snapshot = LaunchSnapshot(**clip["launch_snapshot"])
    if snapshot.fingerprint != clip.get("snapshot_fingerprint"):
        raise ValueError("Launch snapshot fingerprint mismatch")
    if not np.allclose(snapshot.head_position_w, clip["target_w"], atol=1e-9, rtol=0):
        raise ValueError("Reference target differs from frozen launch head position")
    original = PunchRequest(clip["family"], clip["requested_ttc_s"], snapshot,
                            clip["guard_id"], clip["launch_time_s"])
    if clip.get("request_fingerprint") != original.fingerprint:
        raise ValueError("Original launch request fingerprint mismatch")
    origin = np.asarray(origin, float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Finite parallel-world origin required")
    qpos = np.asarray(snapshot.g1_qpos).copy()
    qpos[:3] += origin
    shifted = replace(snapshot, g1_qpos=qpos,
        head_position_w=np.asarray(snapshot.head_position_w) + origin,
        human_origin_w=np.asarray(snapshot.human_origin_w) + origin)
    return replace(original, snapshot=shifted, launch_time_s=float(launch_time_s))


def load_phase_endpoints(bundle, phase_file=None) -> dict[str, float]:
    """Require explicit phase evidence bound to this exact reference bundle."""
    if phase_file is None:
        records = bundle.clips
    else:
        payload = json.loads(Path(phase_file).read_text())
        if (payload.get("schema") != "human_wbc_reference_phases_v1"
                or payload.get("bundle_manifest_sha256") != sha256_file(bundle.root / "manifest.json")):
            raise ValueError("Phase evidence is not bound to this bundle manifest")
        records = payload["references"]
    by_id = {entry["reference_id"]: entry for entry in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate reference phase identity")
    result = {}
    for clip in bundle.clips:
        entry = by_id.get(clip["reference_id"], {})
        end = entry.get("outward_end_s")
        if (entry.get("master_sha256") != clip["master_sha256"]
                or not isinstance(end, (int, float)) or not np.isfinite(end)
                or not 0 < end < clip["guard_time_s"]
                or abs(entry.get("guard_time_s", -1)-clip["guard_time_s"]) > 1e-9):
            raise ValueError("Missing or inconsistent exact reference outward phase")
        result[clip["reference_id"]] = float(end)
    return result


def reference_diagnostic(bundle, clip: dict, sample_count: int, origin,
                         outward_end_s: float) -> np.ndarray:
    """Sample the committed 200 Hz reference, holding its final guard position."""
    with np.load(bundle.path(clip["master_path"]), allow_pickle=False) as master:
        times = master["time_s"]
        gloves = master["glove_pos_w"].copy()
    side = FAMILIES.index(clip["family"]) % 2
    distance = np.linalg.norm(gloves[:, side] - clip["target_w"], axis=1)
    if not np.any((times > 0) & (times <= outward_end_s) & (distance <= .14)):
        raise ValueError("No reference analytic contact in the declared outward phase")
    return gloves[np.minimum(np.arange(sample_count), len(gloves)-1)] + np.asarray(origin)


def deadline_prefix(arrays: dict, request: PunchRequest) -> dict:
    """Keep a complete deadline panel without discarding the saved six-second trace."""
    times = np.asarray(arrays["time_s"])
    count = int(np.searchsorted(times, request.launch_time_s + request.ttc_s + 4. + 1e-10,
                                side="right"))
    # Constant arrays (joint limits/caps) must never be sliced as sample arrays.
    constants = {"joint_limits", "effort_limits"}
    result = {key: (value[:count] if isinstance(value, np.ndarray) and value.ndim > 0
                    and len(value) == len(times) and key not in constants else value)
              for key, value in arrays.items()}
    result["physics_step_count"] = count-1
    return result


class ProbeWriteLedger:
    """Instrument the only enabled state-write path; reject post-launch writes.

    This does not pretend to intercept arbitrary native memory edits. The probe
    disables events, assistance and auto resets; its local command subclasses
    every command reset/pose-write entry point. Native forces are audited too.
    """
    def __init__(self, count: int):
        self.resets = np.zeros(count, np.int64)
        self.writes = np.zeros(count, np.int64)
        self.blocked = np.zeros(count, bool)
        self.sealed = False

    def event(self, kind: str, env_ids) -> None:
        ids = _copy(env_ids).astype(int)
        counter = {"reset": self.resets, "write": self.writes}[kind]
        counter[ids] += 1
        if self.sealed:
            self.blocked[ids] = True
            raise RuntimeError(f"Physical {kind} attempted after committed launch")

    def seal(self) -> None:
        if not np.all(self.resets == 1) or not np.all(self.writes == 1):
            raise RuntimeError("Expected exactly one command reset and pose initialization per environment")
        self.sealed = True


def _probe_command_types():
    # Lazy imports: pure request/trace tests do not initialize Warp or CUDA.
    import torch
    from .runtime.command import PunchMotionCommand, PunchMotionCommandCfg

    class ProbeCommand(PunchMotionCommand):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.write_ledger = ProbeWriteLedger(env.num_envs)
            self._fixed_clip_ids = torch.as_tensor(deterministic_clip_ids(
                self.clips, np.arange(env.num_envs)), device=self.device)

        def reset(self, env_ids):
            self.write_ledger.event("reset", env_ids)
            return super().reset(env_ids)

        def _write_reference_state_to_sim(self, env_ids, *args, **kwargs):
            self.write_ledger.event("write", env_ids)
            return super()._write_reference_state_to_sim(env_ids, *args, **kwargs)

        def reset_to_frame(self, env_ids, frame):
            self.write_ledger.event("write", env_ids)
            raise RuntimeError("Reference scrubbing is forbidden in a cycle diagnostic")

        def _resample_command(self, env_ids):
            if not getattr(self, "_reset_in_progress", False):
                raise RuntimeError("Cycle probe references only initialize at episode reset")
            self.trajectory_ids[env_ids] = self._fixed_clip_ids[env_ids]
            self.physics_steps[env_ids] = 0
            self.time_steps[env_ids] = self.motion.segment_start_idx[self.trajectory_ids[env_ids]]
            self.episode_assist_gain[env_ids] = 0.
            self._episode_start_step[env_ids] = self.time_steps[env_ids]
            self._episode_start_bin[env_ids] = 0
            self._episode_similarity_sum[env_ids] = 0.
            self._episode_step_count[env_ids] = 0
            self._write_reference_state_to_sim(env_ids,
                self.body_pos_w[env_ids, 0], self.body_quat_w[env_ids, 0],
                self.body_lin_vel_w[env_ids, 0], self.body_ang_vel_w[env_ids, 0],
                self.joint_pos[env_ids], self.joint_vel[env_ids])
            self.update_relative_body_poses()

    @dataclass(kw_only=True)
    class ProbeCommandCfg(PunchMotionCommandCfg):
        def build(self, env):
            return ProbeCommand(self, env)

    return ProbeCommand, ProbeCommandCfg


class ProbeMetrics:
    """Supported metric/reset hooks, including resets before requests are captured."""
    def __init__(self):
        self.native = NativeProbeRecorder(ENVIRONMENTS)
        self.cycle: CycleTraceRecorder | None = None

    def attach(self, recorder: CycleTraceRecorder) -> None:
        if self.cycle is not None:
            raise RuntimeError("A committed cycle recorder cannot be replaced")
        recorder.reset_counts[:] = self.native.reset_counts
        self.cycle = recorder

    def metric_function(self):
        def reset(env_ids=None):
            self.native.note_reset(env_ids)
            if self.cycle is not None:
                self.cycle.note_reset(env_ids)

        def capture(env):
            self.native.capture(env)
            if self.cycle is None:
                raise RuntimeError("Launch requests must be committed before integration")
            return self.cycle.metric_function()(env)
        capture.reset = reset
        return capture


def build_cycle_probe_cfg(bundle, metrics: ProbeMetrics):
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from .runtime.task import build_human_env_cfg
    _, cfg_type = _probe_command_types()
    cfg = build_human_env_cfg(bundle, num_envs=ENVIRONMENTS, evaluation=True)
    old = cfg.commands["motion"]
    cfg.commands["motion"] = cfg_type(**{field.name: getattr(old, field.name) for field in fields(old)})
    cfg.events, cfg.terminations, cfg.rewards = {}, {}, {}
    cfg.auto_reset = False
    cfg.episode_length_s = 10.
    cfg.metrics = {"cycle_trace": MetricsTermCfg(func=metrics.metric_function(), per_substep=True)}
    cfg.seed = 10010
    return cfg


def _events(env):
    command = env.command_manager.get_term("motion")
    if env.cfg.events or env.cfg.auto_reset or env.sim.mj_model.neq:
        raise RuntimeError("Cycle instrumentation requires no events, auto resets or equality assistance")
    if command.cfg.assistive_wrench_enabled:
        raise RuntimeError("Assistance must remain disabled")
    # Applied generalized forces and all body wrenches are separately recorded;
    # none is summed away. This flag also captures the command assistance state.
    return {"external_support": _copy(command.episode_assist_gain) != 0,
            "pose_replacement": command.write_ledger.blocked.copy()}


def initialize_launch_drive(env) -> dict:
    """Set sample-zero native drive and refresh forward quantities without stepping.

    Reset leaves action targets at zero. Calling apply_actions here would advance
    the reference clock; explicit targets avoid that phase error. This is not
    physical settling and does not alter qpos/qvel or add an integration.
    """
    import torch
    command = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    before = {name: _copy(getattr(env.sim.data, name)) for name in ("qpos", "qvel", "time")}
    ticks = _copy(command.physics_steps)
    limits = robot.data.joint_pos_limits
    robot.set_joint_position_target(torch.clamp(command.joint_pos, limits[..., 0], limits[..., 1]))
    robot.set_joint_velocity_target(command.joint_vel)
    robot.set_joint_effort_target(command.sample_master("joint_feedforward"))
    env.scene.write_data_to_sim()
    env.sim.forward()
    for name, value in before.items():
        np.testing.assert_array_equal(_copy(getattr(env.sim.data, name)), value)
    np.testing.assert_array_equal(_copy(command.physics_steps), ticks)
    return {"sample_zero_reference_drive_initialized": True, "forward_calls": 1,
            "integration_steps": 0, "qpos_qvel_time_unchanged": True,
            "reference_counter_unchanged": True, "physical_guard_settling": False}


def save_compiled_model(model, path: Path) -> dict:
    """Preserve the actual compiled options, addresses, sensors and native actuators."""
    import mujoco
    path = writable_path(path)
    # Reserve the new evidence path; never overwrite a preceding run's model.
    with path.open("xb"):
        pass
    mujoco.mj_saveModel(model, str(path), None)
    return {"path": path.name, "sha256": sha256_file(path), "format": "MuJoCo MJB",
            "mujoco_version": mujoco.__version__, "nq": model.nq, "nv": model.nv,
            "nu": model.nu, "nbody": model.nbody, "ngeom": model.ngeom,
            "portability": "Requires a compatible MuJoCo binary-model version; source assets are retained separately"}


def allocated_device_metadata(device) -> dict:
    """Read actual public Torch/Warp APIs; driver API version is not wheel CUDA."""
    import torch
    import warp as wp
    properties = torch.cuda.get_device_properties(device)
    native = wp.get_device(str(device))
    return {"torch_device_name": properties.name,
            "torch_total_memory_bytes": int(properties.total_memory),
            "torch_compute_capability": [int(properties.major), int(properties.minor)],
            "torch_multiprocessor_count": int(properties.multi_processor_count),
            "torch_cuda_build": torch.version.cuda,
            "warp_device_name": native.name, "warp_device_uuid": native.uuid,
            "warp_device_arch": native.arch, "warp_device_sm_count": native.sm_count,
            "warp_cuda_toolkit_version": wp.get_cuda_toolkit_version(),
            "warp_cuda_driver_api_version": wp.get_cuda_driver_version(),
            "driver_version_semantics": "CUDA driver API major/minor reported by Warp, not NVIDIA package build number"}


def _save_npz(path: Path, arrays: dict) -> str:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return sha256_file(path)


def run_native_cycle_probe(bundle, output, *, phases=None) -> dict[str, Any]:
    """Exactly 16 environments, 6 s, zero residual, one reset; no CPU dynamics."""
    _require_allocated_cuda()
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise RuntimeError("Cycle probe requires the pinned Task 10 runtime")
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    _configure_caches(destination)
    from .runtime.bundle import load_runtime_bundle
    checked = load_runtime_bundle(bundle)
    phase_endpoints = load_phase_endpoints(checked, phases)
    metrics = ProbeMetrics()
    cfg = build_cycle_probe_cfg(checked, metrics)
    cfg.sim.nan_guard.output_dir = str(destination / "nan_guard")
    report = dict(schema=SCHEMA, status="starting", runtime=runtime,
        environments=ENVIRONMENTS, duration_s=6., policy_steps=POLICY_STEPS,
        action_mode="zero_residual_only", includes_ppo=False,
        qualified_for_training=False, admission_passed=False, acceptance_eligible=False,
        inter_character_collision_enabled=False, g1_entity_present=False,
        reference_manifest_sha256=sha256_file(checked.root / "manifest.json"),
        model_sha256=sha256_file(checked.model_path), controller_sha256=checked.controller_sha256(),
        simulation_contract=asdict(cfg.sim.mujoco), slurm_job_id=os.environ["SLURM_JOB_ID"],
        contact_api="mujoco_warp.contact_force(model, data, ids, to_world_frame=False, force)",
        contact_time_semantics="preceding integration interval; launch row is reset forward state",
        state_write_scope="local command reset/write/reset_to_frame; events disabled; no auto reset; all native applied forces retained",
        evaluation_scope="descriptive deadline prefixes of full six-second traces; no domain admission",
        performance_scope="synchronized per-substep CPU copies, not production throughput")
    report["phase_sidecar_sha256"] = sha256_file(phases) if phases is not None else None
    report["outward_phase_source"] = "hash_bound_phase_sidecar" if phases is not None else "reference_manifest"
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
            report["launch_drive_initialization"] = initialize_launch_drive(env)
            ids = _copy(command.trajectory_ids).astype(int)
            if not np.array_equal(ids, deterministic_clip_ids(command.clips, np.arange(ENVIRONMENTS))):
                raise RuntimeError("Deterministic family/variant assignment changed")
            clips = [command.clips[i] for i in ids]
            origins = _copy(env.scene.env_origins)
            times = _copy(env.sim.data.time)
            requests = [placed_request(clip, origin, timestamp)
                        for clip, origin, timestamp in zip(clips, origins, times, strict=True)]
            write_json(destination / "launch_requests.json", [asdict(request) for request in requests])
            recorder = CycleTraceRecorder(requests, event_provider=_events)
            metrics.attach(recorder)
            metrics.native.capture(env)
            recorder.capture_launch(env)
            initial_target = _copy(command.target_w)
            np.testing.assert_allclose(initial_target, [r.snapshot.head_position_w for r in requests],
                                       atol=2e-6, rtol=0)
            action = torch.zeros((ENVIRONMENTS, env.action_manager.total_action_dim), device=device)
            for _ in range(POLICY_STEPS):
                env.step(action)
            if not np.array_equal(_copy(command.target_w), initial_target):
                raise RuntimeError("Committed target changed during execution")
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
    # Persist raw evidence first, even if contact decoding or downstream FK fails.
    raw = recorder.raw_arrays() if recorder is not None else {}
    native = metrics.native.arrays()
    report["raw_trace_sha256"] = _save_npz(destination / "raw_trace.npz", raw)
    report["native_controller_trace_sha256"] = _save_npz(destination / "native_controller_trace.npz", native)
    report["recorded_samples"] = len(raw.get("time", []))
    report["native_state_samples"] = len(native.get("time", []))
    report["status"] = "failed"
    if command is not None:
        report["state_write_ledger"] = {
            "reset_counts": command.write_ledger.resets.tolist(),
            "pose_write_counts": command.write_ledger.writes.tolist(),
            "blocked_post_launch_attempts": command.write_ledger.blocked.tolist(),
            "sealed": command.write_ledger.sealed}
    if recorder is not None and recorder.model is not None and native:
        model = recorder.model
        free = np.flatnonzero(model.jnt_type == 0)
        audit = audit_probe_arrays(native, policy_steps=POLICY_STEPS,
            physics_dt_s=float(model.opt.timestep), effort_limits=model.actuator_forcerange[:, 1],
            root_dof_start=int(model.jnt_dofadr[free[0]]) if len(free) == 1 else -1)
        report["clock_controller_audit"] = audit
        outcomes = []
        for index, (request, clip) in enumerate(zip(recorder.requests, clips, strict=True)):
            item = dict(environment=index, family=request.family.value,
                        reference_id=clip["reference_id"], request_sha256=request.fingerprint)
            try:
                outward = phase_endpoints[clip["reference_id"]]
                reference = reference_diagnostic(
                    checked, clip, len(recorder.rows), origins[index], outward)
                trace = recorder.episode_arrays(index, reference_glove_position_w=reference,
                    joint_names=checked.joint_names, outward_end_s=outward, admission_passed=False)
                item["trace_sha256"] = _save_npz(destination / f"episode_{index:02d}.npz", trace)
                prefix = deadline_prefix(trace, request)
                result = evaluate_episode(request, prefix, training_seed=0,
                    recipe_id=clip.get("recipe_id", clip["reference_id"]),
                    spatial_cell_id="unqualified_candidate", ttc_cell_id="unqualified_candidate")
                item.update(status="evaluated_descriptively", outward_endpoint_definition=report["outward_phase_source"],
                            outward_end_s=outward, evaluated_samples=len(prefix["time_s"]),
                            full_trace_samples=len(trace["time_s"]), result=asdict(result))
                if result.acceptance_eligible:
                    raise RuntimeError("Unqualified diagnostic became acceptance eligible")
            except Exception as exception:
                item.update(status="evaluation_failed", exception={"type": type(exception).__name__,
                                                                  "message": str(exception)})
            outcomes.append(item)
        report["episodes"] = outcomes
        report["status"] = ("recorded_descriptively" if audit["status"] == "passed"
            and all(item["status"] == "evaluated_descriptively" for item in outcomes)
            and report["recorded_samples"] == POLICY_STEPS*4+1 else "failed")
    if error is not None:
        report.update(status="failed", exception=error)
    report["total_wall_time_s"] = time.monotonic()-started
    write_json(destination / "audit.json", report)
    return report


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phases", type=Path, help="Hash-bound phase sidecar for older candidate manifests")
    parser.add_argument("--environments", type=int, choices=(ENVIRONMENTS,), default=ENVIRONMENTS)
    parser.add_argument("--seconds", type=float, choices=(6.,), default=6.)
    args = parser.parse_args(argv)
    report = run_native_cycle_probe(args.bundle, args.output, phases=args.phases)
    print(json.dumps({"status": report["status"], "recorded_samples": report["recorded_samples"],
                      "acceptance_eligible": False}, sort_keys=True))
    return 0 if report["status"] == "recorded_descriptively" else 1


if __name__ == "__main__":
    raise SystemExit(main())
