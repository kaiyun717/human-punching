"""Six-second frozen learned-guard diagnostic; no punching, PPO or job submission."""
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
import time

import numpy as np

from .conditioning import Conditioning, apply_conditioning
from .cycle_trace import CycleTraceRecorder
from .guard_probe import _guard_command_types, build_guard_probe_cfg, common_guard_identity, evaluate_guard_trace
from .native_cycle_probe import (ENVIRONMENTS, POLICY_STEPS, ProbeMetrics, _events, _save_npz,
    allocated_device_metadata, deterministic_clip_ids, initialize_launch_drive, placed_request, save_compiled_model)
from .native_probe import _configure_caches, _copy, _require_allocated_cuda
from .provenance import PROJECT_ROOT, runtime_record, sha256_file, writable_path, write_json
from .runtime.bundle import load_runtime_bundle
from .trainedactor_evaluation import actor_identity, audit_actor_clock, native_interfaces
from .validation import digest

SCHEMA = "human_wbc_learned_guard_evaluation_v1"
ORIGIN = "new_scratch_shared_guard_stabilization"


def evaluation_caches(destination):
    """Reuse the shallow session caches; do not lengthen NVRTC temporary paths."""
    names = ("XDG_CACHE_HOME", "WARP_CACHE_PATH", "CUDA_CACHE_PATH", "TORCH_HOME", "TMPDIR", "MPLCONFIGDIR")
    existing = {name: os.environ.get(name) for name in names}
    if not all(value and Path(value).resolve().is_relative_to(PROJECT_ROOT) and Path(value).is_dir()
               for value in existing.values()):
        _configure_caches(destination)
    return {name: str(Path(os.environ[name]).resolve()) for name in names}


def verify_guard_actor_identity(actor, bundle, identity, joint_feedforward_nm, conditioning):
    expected = {**actor_identity(actor),
        "reference_manifest_sha256": sha256_file(bundle.root/"manifest.json"),
        "model_sha256": sha256_file(bundle.model_path), "controller_sha256": bundle.controller_sha256(),
        "conditioning": Conditioning(conditioning).value,
        "joint_feedforward_sha256": digest(np.asarray(joint_feedforward_nm, dtype=float).tolist()),
        "guard_training_origin": ORIGIN}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Learned guard actor/body/controller/reference/feedforward identity differs")
    for key in ("guard_admission_sha256", "training_checkpoint_sha256"):
        value = identity.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Learned guard requires an explicit {key}")
    if not identity.get("observation_schema"):
        raise ValueError("Native training observation schema is required")
    torque = np.asarray(joint_feedforward_nm, float)
    if torque.shape != (48,) or not np.isfinite(torque).all() or np.any(abs(torque) > bundle.body["effort"]):
        raise ValueError("Guard feedforward must contain 48 finite capped joint torques")


def learned_guard_command_types():
    """Preserve the probe's physical-write ledger while matching guard conditioning."""
    import torch
    command_type, cfg_type = _guard_command_types()

    class LearnedGuardCommand(command_type):
        @property
        def elapsed(self):
            return torch.zeros_like(self.physics_steps, dtype=torch.float32)

    @dataclass(kw_only=True)
    class LearnedGuardCommandCfg(cfg_type):
        def build(self, env):
            return LearnedGuardCommand(self, env)

    return LearnedGuardCommand, LearnedGuardCommandCfg


def build_learned_guard_cfg(bundle, metrics, *, joint_feedforward_nm, conditioning, seed):
    cfg = build_guard_probe_cfg(bundle, metrics, feedforward=joint_feedforward_nm)
    _, command_cfg = learned_guard_command_types()
    old = cfg.commands["motion"]
    cfg.commands["motion"] = command_cfg(**{field.name: getattr(old, field.name) for field in fields(old)})
    apply_conditioning(cfg, conditioning)
    cfg.seed = seed
    return cfg


def verify_guard_observation_schema(actual: dict, training: dict) -> dict:
    """Require the complete actor interface; disclose the unused critic change.

    Diagnostic probes disable rewards. WBC's privileged critic includes one
    feature per motion reward, so its width legitimately differs here. The
    frozen actor consumes only the actor group. Keep its term order, functions,
    processing, normalization convention and shared semantics strictly bound.
    """
    for value in (actual, training):
        if (not isinstance(value, dict) or value.get("schema") != "human_wbc_observations_v1"
                or not isinstance(value.get("groups"), dict)
                or set(value["groups"]) != {"actor", "critic"}
                or not all(isinstance(value["groups"][name], dict)
                           and value["groups"][name].get("terms")
                           for name in ("actor", "critic"))):
            raise ValueError("Native actor and critic observation schemas are required")
    shared = lambda value: {key: item for key, item in value.items() if key != "groups"}
    if digest(shared(actual)) != digest(shared(training)):
        raise ValueError("Learned guard shared observation semantics differ from training")
    actor, expected_actor = actual["groups"]["actor"], training["groups"]["actor"]
    if digest(actor) != digest(expected_actor):
        raise ValueError("Learned guard actor observations differ from its training schema")
    critic, expected_critic = actual["groups"]["critic"], training["groups"]["critic"]
    differences = []

    def compare(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
            for key in left:
                compare(left[key], right[key], path+"."+key)
        elif isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                compare(a, b, f"{path}[{index}]")
        elif left != right:
            differences.append({"path": path, "training": left, "evaluation": right})

    compare(expected_critic, critic, "groups.critic")
    return {"status": "passed", "comparison_scope": "strict_actor_group_and_shared_semantics",
        "actor_schema_sha256": digest(actor), "critic_used_for_inference": False,
        "critic_schema_matches_training": not differences,
        "training_critic_schema_sha256": digest(expected_critic),
        "evaluation_critic_schema_sha256": digest(critic), "critic_differences": differences,
        "reason": "Frozen actor consumes actor observations only; diagnostic rewards are disabled."}


def audit_residual_guard_drive(native, bundle, joint_feedforward_nm):
    """Reconstruct targets, affine drives and preceding-interval capped forces."""
    import mujoco
    fields_needed = ("qpos", "qvel", "raw_action", "joint_position_target", "joint_velocity_target",
                     "joint_effort_target", "ctrl", "actuator_force")
    if any(name not in native for name in fields_needed):
        return {"status": "failed", "reason": "missing_native_arrays"}
    a = {key: np.asarray(native[key], float) for key in fields_needed}
    shape = a["raw_action"].shape
    if (len(shape) != 3 or shape[-1] != 48 or shape[0] < 2
            or a["qpos"].shape != (*shape[:2], 55) or a["qvel"].shape != (*shape[:2], 54)
            or any(a[name].shape != shape for name in fields_needed[2:])
            or not all(np.isfinite(value).all() for value in a.values())):
        return {"status": "failed", "reason": "malformed_native_arrays"}
    model = mujoco.MjModel.from_xml_path(str(bundle.model_path))
    limits = model.jnt_range[1:]
    qguard = np.asarray(common_guard_identity(bundle)["qpos"])[7:]
    expected_target = np.clip(qguard+.25*a["raw_action"], limits[:, 0], limits[:, 1])
    kp, kd, cap = (np.asarray(bundle.body[key]) for key in ("kp", "kd", "effort"))
    expected_ff = np.asarray(joint_feedforward_nm, float)
    drive = kp*a["joint_position_target"]+kd*a["joint_velocity_target"]+expected_ff
    q = np.concatenate((a["qpos"][:1, :, 7:], a["qpos"][:-1, :, 7:]))
    v = np.concatenate((a["qvel"][:1, :, 6:], a["qvel"][:-1, :, 6:]))
    force = np.clip(a["ctrl"]-kp*q-kd*v, -cap, cap)
    errors = {"target_rad": float(np.max(abs(a["joint_position_target"]-expected_target))),
              "feedforward_nm": float(np.max(abs(a["joint_effort_target"]-expected_ff))),
              "drive_nm": float(np.max(abs(a["ctrl"]-drive))),
              "capped_force_nm": float(np.max(abs(a["actuator_force"]-force)))}
    checks = {"residual_target_and_joint_clamp": errors["target_rad"] <= 2e-6,
        "declared_joint_feedforward": errors["feedforward_nm"] <= 5e-6,
        "zero_reference_velocity": bool(np.all(a["joint_velocity_target"] == 0)),
        "unclipped_affine_drive": errors["drive_nm"] <= 2e-4,
        "preceding_interval_capped_native_force": errors["capped_force_nm"] <= 2e-3}
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "maximum_errors": errors, "force_semantics": "row k uses pre-integration q/v from row k-1; initial row uses initial q/v"}


def run_learned_guard_evaluation(bundle, actor, output, *, joint_feedforward_nm,
                                 conditioning, seed, actor_contract_identity) -> Path:
    """Evaluate one exact guard with 16 deterministic contexts; retain every failure.

    This is a guard diagnostic with a frozen actor, not punch/target/TTC acceptance
    or a production physical-validation callback. No recipe is discarded.
    """
    checked = load_runtime_bundle(bundle)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Evaluation seed must be a nonnegative integer")
    verify_guard_actor_identity(actor, checked, actor_contract_identity, joint_feedforward_nm, conditioning)
    _require_allocated_cuda()
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise RuntimeError("Learned guard evaluation requires the pinned runtime")
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    caches = evaluation_caches(destination)
    guard, metrics = common_guard_identity(checked), ProbeMetrics()
    cfg = build_learned_guard_cfg(checked, metrics, joint_feedforward_nm=joint_feedforward_nm,
                                  conditioning=conditioning, seed=seed)
    cfg.sim.nan_guard.output_dir = str(destination/"nan_guard")
    report = {"schema": SCHEMA, "status": "starting", "includes_ppo": False,
        "actor_contract_identity": actor_contract_identity, "guard_identity": guard,
        "actor_contract_sha256": digest(actor_contract_identity), "runtime": runtime, "cache_paths": caches,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "environments": ENVIRONMENTS,
        "duration_s": 6., "physics_dt_s": .005, "policy_dt_s": .020, "seed": seed,
        "joint_feedforward_nm": np.asarray(joint_feedforward_nm, float).tolist(),
        "punch_execution_tested": False, "targeting_tested": False, "ttc_tested": False,
        "target_ttc_domain_qualified": False, "qualified_for_training": False,
        "inter_character_collision_enabled": False, "g1_entity_present": False,
        "time_origin": "physical guard diagnostic start; no punch launch",
        "conditioning_elapsed_s": 0., "normalization_updated": False,
        "independent_guard_replicates": False, "unique_guard_configurations": 1,
        "latency_semantics": "synchronized actor inference wall time; separate from policy period"}
    write_json(destination/"launch.json", report)
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from .checkpointing import capture_rng, restore_rng
    env = command = recorder = frozen = None
    contexts, clips, origins, error, rng = [], [], None, None, None
    observations_saved, actions_raw, executed, latency = [], [], [], []
    identity_before = actor_identity(actor)
    started = time.monotonic()
    try:
        rng = capture_rng("cuda:0")
        frozen = deepcopy(actor).to("cuda:0").eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        if actor_identity(frozen) != identity_before:
            raise RuntimeError("Frozen independent actor copy changed")
        report["allocated_device"] = allocated_device_metadata(torch.device("cuda:0"))
        stream = torch.cuda.ExternalStream(wp.get_stream("cuda:0").cuda_stream, device="cuda:0")
        caller = torch.cuda.current_stream("cuda:0")
        stream.wait_stream(caller)
        with torch.cuda.stream(stream), torch.no_grad():
            env = ManagerBasedRlEnv(cfg, device="cuda:0")
            interfaces = native_interfaces(env, checked)
            report["native_interfaces"] = interfaces
            report["observation_schema_audit"] = verify_guard_observation_schema(
                interfaces["observation_contract"], actor_contract_identity["observation_schema"])
            report["compiled_scene"] = save_compiled_model(env.sim.mj_model, destination/"compiled_scene.mjb")
            env.reset(seed=seed)
            command = env.command_manager.get_term("motion")
            command.write_ledger.seal()
            report["initial_drive"] = initialize_launch_drive(env)
            if np.any(_copy(env.sim.data.qvel) != 0):
                raise RuntimeError("Initial guard must have zero physical velocity")
            ids = _copy(command.trajectory_ids).astype(int)
            np.testing.assert_array_equal(ids, deterministic_clip_ids(command.clips, np.arange(ENVIRONMENTS)))
            clips = [command.clips[index] for index in ids]
            origins, times = _copy(env.scene.env_origins), _copy(env.sim.data.time)
            qpos = np.broadcast_to(guard["qpos"], (ENVIRONMENTS, 55)).copy()
            qpos[:, :3] += origins
            np.testing.assert_allclose(_copy(env.sim.data.qpos), qpos, atol=2e-6, rtol=0)
            contexts = [placed_request(clip, origin, stamp) for clip, origin, stamp in zip(clips, origins, times, strict=True)]
            recorder = CycleTraceRecorder(contexts, event_provider=_events)
            metrics.attach(recorder); metrics.native.capture(env); recorder.capture_launch(env)
            target, g1 = _copy(command.target_w), _copy(command.g1_context)
            observations = env.observation_manager.compute(update_history=False)
            for _ in range(POLICY_STEPS):
                if torch.any(command.elapsed != 0):
                    raise RuntimeError("Learned guard conditioning must remain prelaunch elapsed zero")
                values = observations["actor"]
                if values.ndim != 2 or not torch.isfinite(values).all():
                    raise RuntimeError("Malformed actor observation")
                torch.cuda.synchronize("cuda:0")
                before = time.perf_counter()
                raw = frozen(values)
                if raw.shape != (ENVIRONMENTS, 48) or not torch.isfinite(raw).all():
                    raise RuntimeError("Malformed actor actions")
                action = raw.clamp(-1., 1.)
                torch.cuda.synchronize("cuda:0")
                latency.append(time.perf_counter()-before)
                observations_saved.append(_copy(values)); actions_raw.append(_copy(raw)); executed.append(_copy(action))
                observations, *_ = env.step(action)
            np.testing.assert_array_equal(_copy(command.target_w), target)
            np.testing.assert_array_equal(_copy(command.g1_context), g1)
        caller.wait_stream(stream)
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exception:
                error = error or {"type": type(exception).__name__, "message": str(exception)}
        if rng is not None:
            restore_rng(rng, "cuda:0")
    if actor_identity(actor) != identity_before or (frozen is not None and actor_identity(frozen) != identity_before):
        error = {"type": "ActorMutation", "message": "Policy/normalization changed"}
        report["normalization_updated"] = True
    native = metrics.native.arrays()
    raw = recorder.raw_arrays() if recorder is not None else {}
    report["raw_trace_sha256"] = _save_npz(destination/"raw_trace.npz", raw)
    report["native_controller_trace_sha256"] = _save_npz(destination/"native_controller_trace.npz", native)
    report["actor_trace_sha256"] = _save_npz(destination/"actor_trace.npz", {
        "observations": np.asarray(observations_saved), "raw_actions": np.asarray(actions_raw),
        "executed_actions": np.asarray(executed), "inference_wall_time_s": np.asarray(latency)})
    clock = audit_actor_clock(native, np.asarray(executed), effort_limits=checked.body["effort"]) if native else {"status": "failed"}
    drive = audit_residual_guard_drive(native, checked, joint_feedforward_nm)
    report.update(clock_controller_audit=clock, independent_drive_audit=drive,
                  recorded_samples=len(raw.get("time", [])))
    native_pass = clock["status"] == drive["status"] == "passed" and error is None
    outcomes = []
    for index in range(ENVIRONMENTS):
        row = {"environment": index, "status": "failed_or_incomplete"}
        try:
            if recorder is None or index >= len(contexts):
                raise RuntimeError("No complete native recording for planned guard context")
            context, clip = contexts[index], clips[index]
            row.update(source_family=context.family.value, reference_id=clip["reference_id"], source_context=asdict(context))
            with np.load(checked.path(clip["master_path"]), allow_pickle=False) as saved:
                reference = np.broadcast_to(saved["glove_pos_w"][0]+origins[index], (len(recorder.rows), 2, 3)).copy()
            trace = recorder.episode_arrays(index, reference_glove_position_w=reference, joint_names=checked.joint_names,
                outward_end_s=0., evidence_kind="physics_guard_hold")
            trace.pop("outward_end_s")
            trace.update(task="learned_static_guard_hold", **{key: actor_contract_identity[key]
                for key in ("policy_sha256", "normalization_sha256", "reference_manifest_sha256", "training_checkpoint_sha256")})
            name = f"guard_{index:02d}.npz"
            row.update(trace_path=name, trace_sha256=_save_npz(destination/name, trace), status="evaluated",
                       result=evaluate_guard_trace(trace, context.snapshot.head_position_w, native_contract_passed=native_pass))
        except Exception as exception:
            row["exception"] = {"type": type(exception).__name__, "message": str(exception)}
        outcomes.append(row)
    report["episodes"] = outcomes
    complete = native_pass and all(row["status"] == "evaluated" for row in outcomes)
    report["status"] = "recorded" if complete else "failed"
    report["guard_success_count"] = sum(row.get("result", {}).get("guard_admission_passed", False) for row in outcomes)
    report["guard_admission_passed"] = bool(complete and report["guard_success_count"] == ENVIRONMENTS)
    report["state_write_ledger"] = None if command is None else {"resets": command.write_ledger.resets.tolist(),
        "writes": command.write_ledger.writes.tolist(), "blocked": command.write_ledger.blocked.tolist()}
    report.update(exception=error, wall_time_s=time.monotonic()-started)
    write_json(destination/"audit.json", report)
    return destination/"audit.json"
