"""Independent frozen-actor cycle recording; production readiness fails closed.

This first backend supports descriptive six-second panels inside allocated CUDA
jobs. It never trains, submits jobs or edits the training environment. Production
validation remains unsupported until physical guard settling before launch and
counted failed-readiness receipts are implemented and qualified. In particular,
initializing at a reference guard is not evidence of physical launch readiness.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import time

import numpy as np

from .conditioning import Conditioning, apply_conditioning
from .cycle_trace import CycleTraceRecorder
from .evaluation import evaluate_episode
from .native_cycle_probe import (POLICY_STEPS, ProbeMetrics, _events, _probe_command_types,
    _save_npz, allocated_device_metadata, build_cycle_probe_cfg, deadline_prefix,
    initialize_launch_drive, load_phase_endpoints, placed_request, reference_diagnostic,
    save_compiled_model)
from .native_probe import NativeProbeRecorder, _configure_caches, _copy, _require_allocated_cuda, audit_probe_arrays
from .provenance import runtime_record, sha256_file, writable_path, write_json
from .runtime.bundle import SOURCE_PINS, load_runtime_bundle
from .training_session import observation_schema
from .validation import PanelExpectation, digest, tensor_digest

SCHEMA = "human_wbc_frozen_actor_diagnostic_panel_v1"
CONTRACT_SCHEMA = "human_wbc_actor_evaluation_contract_v1"
PRODUCTION_BLOCKER = ("Production physical validation is unsupported: guard readiness must be reached "
    "by physical settling before frozen t0; failed-readiness receipts and caller training-environment "
    "state/counter parity are not yet implemented and qualified.")


def actor_identity(actor) -> dict:
    """Bind every parameter/buffer and the embedded normalizer separately."""
    import torch
    state = actor.state_dict()
    normalizer = {name: value for name, value in state.items() if name.startswith("obs_normalizer.")}
    if (not normalizer or not all(torch.isfinite(value).all() for value in state.values())
            or any(module.training for module in actor.modules())
            or any(parameter.requires_grad for parameter in actor.parameters())):
        raise ValueError("Evaluation requires a finite frozen eval-mode actor with embedded normalization")
    return {"policy_sha256": tensor_digest(state), "normalization_sha256": tensor_digest(normalizer)}


def evaluation_contract(actor, bundle, *, observation_contract: dict, action_contract: dict,
                        simulation_contract: dict, checkpoint_contract_sha256: str,
                        conditioning: str = Conditioning.COMBINED.value) -> dict:
    """Construct a reviewable contract from an already frozen learner export.

    The caller supplies exact native interface schemas recorded with its learner.
    This binds identity only; it does not assert a trained actor or qualification.
    """
    checked = load_runtime_bundle(bundle)
    if len(checkpoint_contract_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_contract_sha256):
        raise ValueError("A checkpoint contract SHA256 is required")
    if not observation_contract or not action_contract or not simulation_contract:
        raise ValueError("Exact observation, action and simulation contracts are required")
    payload = {"schema": CONTRACT_SCHEMA, **actor_identity(actor),
        "checkpoint_contract_sha256": checkpoint_contract_sha256,
        "reference_manifest_sha256": sha256_file(checked.root/"manifest.json"),
        "model_sha256": sha256_file(checked.model_path), "controller_sha256": checked.controller_sha256(),
        "source_pins": SOURCE_PINS, "conditioning": Conditioning(conditioning).value,
        "observation_contract": observation_contract, "action_contract": action_contract,
        "simulation_contract": simulation_contract}
    return {**payload, "sha256": digest(payload)}


def verify_actor_contract(actor, bundle, contract: dict, expectation: PanelExpectation) -> None:
    if (contract.get("schema") != CONTRACT_SCHEMA
            or contract.get("sha256") != digest({k:v for k,v in contract.items() if k != "sha256"})):
        raise ValueError("Actor evaluation contract schema/digest mismatch")
    rebuilt = evaluation_contract(actor, bundle,
        observation_contract=contract["observation_contract"], action_contract=contract["action_contract"],
        simulation_contract=contract["simulation_contract"], conditioning=contract["conditioning"],
        checkpoint_contract_sha256=contract["checkpoint_contract_sha256"])
    if rebuilt != contract:
        raise ValueError("Actor/reference/body/controller/source identity changed")
    for name in ("policy_sha256", "normalization_sha256", "checkpoint_contract_sha256", "reference_manifest_sha256"):
        if getattr(expectation, name) != contract[name]:
            raise ValueError(f"Panel expectation does not match actor contract: {name}")


def planned_recipes(bundle, recipe_file: Path, expectation: PanelExpectation, *, production=False) -> tuple[dict, ...]:
    """Bind every recipe to an explicit immutable clip, guard, snapshot and TTC."""
    if sha256_file(recipe_file) != expectation.recipe_manifest_sha256:
        raise ValueError("Predeclared recipe manifest hash changed")
    data = json.loads(recipe_file.read_text())
    if (data.get("schema") != "human_wbc_validation_recipes_v1"
            or data.get("domain_version") != expectation.domain_version
            or data.get("split") != ("validation" if production else "diagnostic")):
        raise ValueError("Recipe split/domain/schema is inconsistent with the requested panel")
    rows = data.get("recipes", [])
    if not 1 <= len(rows) <= 256 or len({row.get("recipe_id") for row in rows}) != len(rows):
        raise ValueError("A bounded panel requires 1..256 unique predeclared recipe IDs")
    clips = {clip["reference_id"]: clip for clip in bundle.clips}
    for row in rows:
        clip = clips.get(row.get("reference_id"))
        if clip is None or not all(row.get(key) for key in ("recipe_id", "spatial_cell_id", "ttc_cell_id")):
            raise ValueError("Every recipe requires an exact clip and explicit cell IDs")
        source = placed_request(clip, np.zeros(3), clip["launch_time_s"])
        if row.get("request_sha256") != source.fingerprint or row.get("family") != source.family.value:
            raise ValueError("Recipe differs from committed clip guard/snapshot/target/TTC")
        if source.ttc_s+4. > POLICY_STEPS*.020:
            raise ValueError("Six-second diagnostic cannot cover this request's complete recovery deadline")
    if production:
        bundle.require_training_qualification()
        # Even qualified references cannot replace physical readiness or provide
        # proof that callback execution leaves the caller's simulation unchanged.
        raise NotImplementedError(PRODUCTION_BLOCKER)
    return tuple(rows)


class PanelMetrics(ProbeMetrics):
    def __init__(self, count):
        self.native = NativeProbeRecorder(count)
        self.cycle = None


def build_actor_panel_cfg(bundle, recipes: tuple[dict, ...], metrics: PanelMetrics,
                          conditioning=Conditioning.COMBINED.value):
    """Use the existing physical-cycle ledger with explicit preplanned clip IDs."""
    import torch
    base_command, base_cfg = _probe_command_types()

    class PlannedCommand(base_command):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            lookup = {clip["reference_id"]: i for i, clip in enumerate(self.clips)}
            self._fixed_clip_ids = torch.as_tensor([lookup[name] for name in cfg.planned_reference_ids],
                                                   device=self.device)
            if len(self._fixed_clip_ids) != env.num_envs:
                raise ValueError("Planned clip count differs from the physical environment count")

    @dataclass(kw_only=True)
    class PlannedCommandCfg(base_cfg):
        planned_reference_ids: tuple[str, ...] = ()

        def build(self, env):
            return PlannedCommand(self, env)

    cfg = build_cycle_probe_cfg(bundle, metrics)
    cfg.scene.num_envs = len(recipes)
    old = cfg.commands["motion"]
    cfg.commands["motion"] = PlannedCommandCfg(
        **{field.name: getattr(old, field.name) for field in fields(old)},
        planned_reference_ids=tuple(row["reference_id"] for row in recipes))
    apply_conditioning(cfg, conditioning)
    return cfg


def verify_compiled_controller(model, bundle) -> None:
    """Reject different transmissions, integration semantics or hidden base drives."""
    import mujoco
    expected_gain = np.zeros((48, 10)); expected_gain[:, 0] = 1.
    expected_bias = np.zeros((48, 10))
    expected_bias[:, 1:3] = -np.column_stack((bundle.body["kp"], bundle.body["kd"]))
    expected_gear = np.zeros((48, 6)); expected_gear[:, 0] = 1.
    if (model.nq, model.nv, model.nu) != (55, 54, 48):
        raise ValueError("Compiled human controller requires one free root and 48 scalar hinges")
    joints = model.actuator_trnid[:, 0]
    names = [model.joint(int(index)).name.rsplit("/", 1)[-1] for index in joints]
    checks = [
        tuple(names) == bundle.joint_names,
        np.array_equal(model.jnt_type, [mujoco.mjtJoint.mjJNT_FREE] + [mujoco.mjtJoint.mjJNT_HINGE]*48),
        np.array_equal(model.jnt_qposadr[joints], np.arange(7, 55)),
        np.array_equal(model.jnt_dofadr[joints], np.arange(6, 54)),
        np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT),
        np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE),
        np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED),
        np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE),
        not np.any(model.actuator_ctrllimited), np.all(model.actuator_forcelimited),
        np.allclose(model.actuator_gainprm, expected_gain, rtol=0, atol=1e-10),
        np.allclose(model.actuator_biasprm, expected_bias, rtol=0, atol=1e-10),
        np.allclose(model.actuator_gear, expected_gear, rtol=0, atol=1e-10),
        np.allclose(model.actuator_forcerange,
                    np.column_stack((-np.asarray(bundle.body["effort"]), bundle.body["effort"])), rtol=0, atol=1e-10),
    ]
    if not all(checks):
        raise ValueError("Compiled human action/controller differs from the frozen actor contract")


def native_interfaces(env, bundle) -> dict:
    """Read actual ordered schemas and compiled native affine actuator values."""
    verify_compiled_controller(env.sim.mj_model, bundle)
    action = env.action_manager.get_term("joint_pos")
    scale = action._scale
    if hasattr(scale, "detach"):
        scale = scale.detach().cpu().numpy()
    if not np.all(np.asarray(scale) == .25):
        raise ValueError("Native actor residual scale differs from 0.25 rad")
    contract = {"observation_contract": observation_schema(env),
        "action_contract": {"joint_names": list(action.target_names), "dimension": action.action_dim,
                            "raw_clip": 1., "scale_rad": .25, "clock_s": .020},
        "simulation_contract": {"mujoco": asdict(env.cfg.sim.mujoco), "decimation": env.cfg.decimation,
                                "nconmax": env.cfg.sim.nconmax, "njmax": env.cfg.sim.njmax}}
    if tuple(action.target_names) != bundle.joint_names or action.action_dim != 48:
        raise ValueError("Compiled human action/controller differs from the frozen actor contract")
    return contract


def audit_actor_clock(native: dict, executed_actions: np.ndarray, *, effort_limits) -> dict:
    """Retain native checks while replacing only the zero-policy requirement."""
    result = audit_probe_arrays(native, policy_steps=POLICY_STEPS, physics_dt_s=.005,
                                effort_limits=effort_limits, root_dof_start=0)
    checks = dict(result["checks"])
    checks.pop("zero_learned_residual", None)
    actions = np.asarray(executed_actions)
    count = np.asarray(native.get("time", [])).shape
    shape = (POLICY_STEPS, count[1], 48) if len(count) == 2 else ()
    valid = actions.shape == shape and np.isfinite(actions).all() and np.all(abs(actions) <= 1.)
    checks["bounded_actor_output"] = bool(valid)
    recorded = np.asarray(native.get("raw_action", []))
    checks["actor_action_held_for_four_physics_steps"] = bool(valid and recorded.shape == (1201, count[1], 48)
        and np.array_equal(recorded[1:], np.repeat(actions, 4, axis=0)) and np.all(recorded[0] == 0))
    return {**result, "checks": checks, "status": "passed" if all(checks.values()) else "failed"}


def run_actor_panel(bundle, actor, actor_contract: dict, recipes_path: Path, output: Path, *,
                    expectation: PanelExpectation, phases: Path | None = None, production=False) -> Path:
    """Run a separate frozen actor environment and retain every planned outcome.

    This backend does not receive the live training environment. Its diagnostic
    report cannot be used by verify_physical_panel or select a checkpoint.
    Production raises before environment construction until readiness and caller
    state-parity proof are implemented; no elapsed-time or pose-reset substitute.
    """
    checked = load_runtime_bundle(bundle)
    verify_actor_contract(actor, checked, actor_contract, expectation)
    recipes = planned_recipes(checked, recipes_path, expectation, production=production)
    _require_allocated_cuda()
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise RuntimeError("Actor panel requires the pinned candidate runtime")
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    _configure_caches(destination)
    phase_ends = load_phase_endpoints(checked, phases)
    metrics = PanelMetrics(len(recipes))
    cfg = build_actor_panel_cfg(checked, recipes, metrics, actor_contract["conditioning"])
    cfg.sim.nan_guard.output_dir = str(destination/"nan_guard")
    cfg.seed = expectation.training_seed
    report = {"schema": SCHEMA, "status": "starting", "native_execution": False,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "expectation": asdict(expectation),
        "actor_contract": actor_contract, "acceptance_eligible": False,
        "qualified_for_training": False, "guard_settling_supported": False,
        "production_blocker": PRODUCTION_BLOCKER, "normalization_updated": False,
        "training_environment_ownership": "separate environment; no caller environment argument",
        "training_environment_parity_proof": "not implemented; production unsupported",
        "inter_character_collision_enabled": False, "g1_entity_present": False,
        "includes_ppo": False, "planned_recipe_count": len(recipes), "runtime": runtime,
        "time_origin": "diagnostic start; no accepted punch launch because physical readiness is not established",
        "phase_sidecar_sha256": sha256_file(phases) if phases is not None else None,
        "duration_s": 6., "physics_dt_s": .005, "policy_dt_s": .020,
        "latency_semantics": "actor forward/clipping/device synchronization wall time, separate from simulated policy period"}
    write_json(destination/"launch.json", report)
    with (destination/"recipes.json").open("xb") as stream:
        stream.write(recipes_path.read_bytes())
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    env = command = recorder = None
    error, clips, requests, origins = None, [], [], None
    actor_inputs, actor_raw, executed, latency = [], [], [], []
    caller_identity = actor_identity(actor)
    frozen = None
    started = time.monotonic()
    try:
        frozen = deepcopy(actor).to("cuda:0").eval()
        # Module.requires_grad_ is not supported by torch ScriptModules, but
        # their native Parameter tensors retain the ordinary tensor API.
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        if actor_identity(frozen) != caller_identity:
            raise RuntimeError("Independent actor copy changed policy or normalization state")
        report["allocated_device"] = allocated_device_metadata(torch.device("cuda:0"))
        stream = torch.cuda.ExternalStream(wp.get_stream("cuda:0").cuda_stream, device="cuda:0")
        caller_stream = torch.cuda.current_stream("cuda:0")
        stream.wait_stream(caller_stream)
        with torch.cuda.stream(stream), torch.no_grad():
            env = ManagerBasedRlEnv(cfg, device="cuda:0")
            actual = native_interfaces(env, checked)
            for name, value in actual.items():
                if digest(value) != digest(actor_contract[name]):
                    raise ValueError(f"Native actor panel {name} differs from frozen learner schema")
            report["compiled_scene"] = save_compiled_model(env.sim.mj_model, destination/"compiled_scene.mjb")
            env.reset(seed=expectation.training_seed)
            command = env.command_manager.get_term("motion")
            command.write_ledger.seal()
            report["initial_drive"] = initialize_launch_drive(env)
            ids = _copy(command.trajectory_ids).astype(int)
            clips = [command.clips[i] for i in ids]
            if [clip["reference_id"] for clip in clips] != [r["reference_id"] for r in recipes]:
                raise RuntimeError("Physical clip assignment differs from the predeclared panel")
            origins, times = _copy(env.scene.env_origins), _copy(env.sim.data.time)
            requests = [placed_request(clip, origin, stamp) for clip, origin, stamp in zip(clips, origins, times, strict=True)]
            recorder = CycleTraceRecorder(requests, event_provider=_events)
            metrics.attach(recorder)
            metrics.native.capture(env)
            recorder.capture_launch(env)
            target, context = _copy(command.target_w), _copy(command.g1_context)
            np.testing.assert_allclose(target, [r.snapshot.head_position_w for r in requests], atol=2e-6, rtol=0)
            observations = env.observation_manager.compute(update_history=False)
            for _ in range(POLICY_STEPS):
                values = observations["actor"]
                if values.ndim != 2 or not torch.isfinite(values).all():
                    raise RuntimeError("Nonfinite/malformed actor observation")
                torch.cuda.synchronize("cuda:0")
                before = time.perf_counter()
                raw = frozen(values)
                if raw.shape != (len(recipes), 48) or not torch.isfinite(raw).all():
                    raise RuntimeError("Frozen actor returned nonfinite/malformed actions")
                action = torch.clamp(raw, -1., 1.)
                torch.cuda.synchronize("cuda:0")
                latency.append(time.perf_counter()-before)
                actor_inputs.append(_copy(values)); actor_raw.append(_copy(raw)); executed.append(_copy(action))
                observations, *_ = env.step(action)
            np.testing.assert_array_equal(_copy(command.target_w), target)
            np.testing.assert_array_equal(_copy(command.g1_context), context)
        caller_stream.wait_stream(stream)
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exception:
                error = error or {"type": type(exception).__name__, "message": str(exception)}
    if actor_identity(actor) != caller_identity or (frozen is not None and actor_identity(frozen) != caller_identity):
        error = {"type": "ActorMutation", "message": "Actor or normalization changed during evaluation"}
        report["normalization_updated"] = True
    report["native_execution"] = recorder is not None and len(recorder.rows) > 1
    report["raw_trace_sha256"] = _save_npz(destination/"raw_trace.npz", recorder.raw_arrays() if recorder else {})
    native = metrics.native.arrays()
    report["native_controller_trace_sha256"] = _save_npz(destination/"native_controller_trace.npz", native)
    report["actor_trace_sha256"] = _save_npz(destination/"actor_trace.npz", {
        "observations": np.asarray(actor_inputs), "raw_actions": np.asarray(actor_raw),
        "executed_actions": np.asarray(executed), "inference_wall_time_s": np.asarray(latency)})
    audit = audit_actor_clock(native, np.asarray(executed), effort_limits=checked.body["effort"]) if native else {"status": "failed"}
    report["clock_action_controller_audit"] = audit
    outcomes = []
    for index, recipe in enumerate(recipes):
        row = {"recipe_id": recipe["recipe_id"], "reference_id": recipe["reference_id"],
               "status": "unexecuted", "acceptance_eligible": False}
        try:
            if recorder is None or index >= len(requests):
                raise RuntimeError("No launch was recorded for this planned recipe")
            request, clip = requests[index], clips[index]
            row.update(source_request=asdict(placed_request(clip, np.zeros(3), clip["launch_time_s"])),
                       request=asdict(request), scene_origin_w=origins[index].tolist())
            ref = reference_diagnostic(checked, clip, len(recorder.rows), origins[index], phase_ends[clip["reference_id"]])
            trace = recorder.episode_arrays(index, reference_glove_position_w=ref,
                joint_names=checked.joint_names, outward_end_s=phase_ends[clip["reference_id"]],
                admission_passed=False, domain_version=expectation.domain_version)
            trace.update({name: getattr(expectation, name) for name in (
                "policy_sha256", "normalization_sha256", "checkpoint_contract_sha256", "reference_manifest_sha256")})
            trace["physical_guard_settling_performed"] = False
            filename = f"episode_{index:06d}.npz"
            row.update(trace_path=filename, trace_sha256=_save_npz(destination/filename, trace))
            result = evaluate_episode(request, deadline_prefix(trace, request),
                training_seed=expectation.training_seed, recipe_id=recipe["recipe_id"],
                spatial_cell_id=recipe["spatial_cell_id"], ttc_cell_id=recipe["ttc_cell_id"])
            if result.acceptance_eligible:
                raise RuntimeError("Unadmitted diagnostic became eligible for acceptance")
            row.update(status="evaluated_descriptively", result=asdict(result))
        except Exception as exception:
            row.update(status="failed_or_incomplete", exception={"type": type(exception).__name__, "message": str(exception)})
        outcomes.append(row)
    report["episodes"] = outcomes
    report["state_write_ledger"] = None if command is None else {
        "resets": command.write_ledger.resets.tolist(), "writes": command.write_ledger.writes.tolist(),
        "blocked": command.write_ledger.blocked.tolist(), "sealed": command.write_ledger.sealed}
    report["status"] = "recorded_descriptively" if (error is None and audit["status"] == "passed"
        and all(row["status"] == "evaluated_descriptively" for row in outcomes)) else "failed"
    report["exception"] = error
    report["wall_time_s"] = time.monotonic()-started
    write_json(destination/"report.json", report)
    return destination/"report.json"


def physical_validation_callback(bundle, actor_contract, recipes_path, output_factory, *, phases=None):
    """Protocol-compatible future hook; production deliberately remains blocked."""
    def callback(actor, expectation: PanelExpectation) -> Path:
        return run_actor_panel(bundle, actor, actor_contract, recipes_path, output_factory(expectation),
                               expectation=expectation, phases=phases, production=True)
    return callback
