"""Narrow scratch guard-stabilization fallback using the inherited WBC learner.

This module prepares/registers a task; it does not launch training or submit jobs.
Its admission permits learning stabilization of one screened posture, even when
fixed PD falls. It never admits punches, target/TTC requests, or production data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .budget import TrainingScale
from .conditioning import Conditioning, apply_conditioning
from .provenance import PROJECT_ROOT, sha256_file
from .runtime.bundle import load_runtime_bundle
from .validation import digest

SCHEMA = "human_wbc_guard_learning_admission_v1"
ORIGIN = "new_scratch_shared_guard_stabilization"
TASK_ID = "HumanPunching-WBC-GuardEngineeringSmoke-v0"


@dataclass(frozen=True)
class GuardSmokePlan:
    environments: int = 256
    iterations: int = 64
    seed: int = 10010
    conditioning: str = Conditioning.COMBINED.value
    episode_seconds: float = 6.

    def __post_init__(self):
        for name in ("environments", "iterations", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("Guard smoke counts and seed must be exact integers")
        if not 16 <= self.environments <= 1024 or self.environments % 4:
            raise ValueError("Guard engineering smoke uses 16..1024 environments in multiples of four")
        if not 1 <= self.iterations < 250 or self.seed < 0 or self.episode_seconds != 6.:
            raise ValueError("Guard smoke is scratch-only, below 250 iterations, with six-second episodes")
        Conditioning(self.conditioning)

    def accounting(self):
        scale = TrainingScale(global_environments=self.environments, world_size=1)
        counts = scale.report(self.iterations)
        counts.pop("allocation_authorized")
        return {**counts, "scope": "guard_engineering_smoke",
                "authorization_semantics": "resource counts only; authorization is recorded in the session packet",
                "production_global_environments": 8192, "production_batch_qualified": False,
                "initialization": "scratch", "historical_specialists_allowed": False,
                "automatic_punch_stage_transition": False}


def guard_geometry(bundle) -> dict:
    """Fresh FK/collision screening of the common posture, not dynamic stability."""
    import mujoco
    from scipy.spatial import ConvexHull
    from .guard_probe import common_guard_identity
    checked = load_runtime_bundle(bundle)
    identity = common_guard_identity(checked)
    model = mujoco.MjModel.from_xml_path(str(checked.model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = identity["qpos"]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_collision(model, data)
    foot_ids = [model.geom("physical/"+side+"Foot").id for side in ("Left", "Right")]
    floor = model.geom("floor").id
    if (model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE
            or not np.array_equal(model.geom_pos[floor], np.zeros(3))
            or not np.array_equal(model.geom_quat[floor], [1., 0., 0., 0.])):
        raise ValueError("Guard screening expects the declared level floor")
    points = []
    for geom in foot_ids:
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Static guard support expects solid foot boxes")
        size, rotation = model.geom_size[geom], data.geom_xmat[geom].reshape(3, 3)
        points.extend(data.geom_xpos[geom]+rotation@np.array([x*size[0], y*size[1], -size[2]])
                      for x, y in itertools.product((-1, 1), repeat=2))
    points = np.asarray(points)
    hull = ConvexHull(points[:, :2])
    com = data.subtree_com[model.body("Hips").id, :2]
    margin = float(np.min(-(hull.equations[:, :2]@com+hull.equations[:, 2])))
    floor_depth = self_depth = nonfoot_depth = 0.
    for contact in data.contact:
        pair = {int(contact.geom1), int(contact.geom2)}
        depth = max(0., -float(contact.dist))
        if floor in pair:
            floor_depth = max(floor_depth, depth)
            if not pair.intersection(foot_ids):
                nonfoot_depth = max(nonfoot_depth, depth)
        else:
            self_depth = max(self_depth, depth)
    gloves = data.site_xpos[[model.site(side+"Hand_center").id for side in ("Left", "Right")]]
    clearance = min(float(np.linalg.norm(gloves-np.asarray(clip["target_w"]), axis=1).min())
                    for clip in checked.clips)
    joints = data.qpos[7:]
    checks = {"joint_limits": bool(np.all(joints >= model.jnt_range[1:, 0]-1e-9)
                                   and np.all(joints <= model.jnt_range[1:, 1]+1e-9)),
              "selected_self_collision": self_depth <= .002,
              "floor_penetration": floor_depth <= .001, "nonfoot_floor_clearance": nonfoot_depth <= .001,
              "foot_support_geometry": bool(points[:, 2].min() >= -.001 and points[:, 2].max() <= .003),
              "com_inside_projected_foot_hull": margin >= 0., "both_gloves_clear": clearance >= .16,
              "four_frozen_g1_contexts": all(clip.get("g1_snapshot_present") is True
                  and len(clip.get("launch_features", [])) == 77 for clip in checked.clips)}
    return {"checks": checks, "passed": all(checks.values()), "qpos_sha256": identity["qpos_sha256"],
            "minimum_projected_support_margin_m": margin, "minimum_glove_clearance_m": clearance,
            "scope": "kinematics and selected collisions only; no loaded-contact or stability claim"}


def prepare_guard_admission(bundle, native_guard_report: Path, support_artifact: Path, *,
                            feedforward_mode: str = "static_lp") -> dict:
    """Verify real native clock/force evidence; fixed-controller falls are retained."""
    from .provenance import configure_local_caches
    # Direct callers may arrive before the CLI establishes its session cache.
    import os
    if not os.environ.get("MPLCONFIGDIR"):
        configure_local_caches()
    from .guard_feedforward import load_guard_feedforward
    from .guard_probe import audit_static_drive, common_guard_identity
    from .native_probe import audit_probe_arrays
    checked = load_runtime_bundle(bundle)
    if feedforward_mode not in ("static_lp", "zero"):
        raise ValueError("Only zero or explicitly qualified joint-only static feedforward is supported")
    report_path, artifact_path = Path(native_guard_report).resolve(strict=True), Path(support_artifact).resolve(strict=True)
    if not report_path.is_relative_to(PROJECT_ROOT) or not artifact_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Guard evidence must remain inside the human project")
    report = json.loads(report_path.read_text())
    support = load_guard_feedforward(artifact_path, checked)
    guard = common_guard_identity(checked)
    if (report.get("schema") != "human_wbc_static_guard_probe_v1" or not report.get("slurm_job_id")
            or report.get("recorded_samples") != 1201 or report.get("includes_ppo") is not False
            or report.get("reference_manifest_sha256") != sha256_file(checked.root/"manifest.json")
            or report.get("guard_identity", {}).get("qpos_sha256") != guard["qpos_sha256"]
            or report.get("guard_identity", {}).get("model_sha256") != sha256_file(checked.model_path)
            or report.get("guard_identity", {}).get("source_controller_sha256") != checked.controller_sha256()
            or report.get("initial_guard_qpos_matches_input") is not True
            or report.get("runtime", {}).get("declared_pins_match") is not True):
        raise ValueError("Native guard evidence does not bind the current posture/body/controller/runtime")
    files = {}
    for name, field in (("native_controller_trace.npz", "native_controller_trace_sha256"),
                        ("raw_trace.npz", "raw_trace_sha256")):
        path = (report_path.parent/name).resolve(strict=True)
        if not path.is_relative_to(report_path.parent) or sha256_file(path) != report.get(field):
            raise ValueError("Native guard trace hash/path mismatch")
        files[name] = {"path": str(path), "sha256": sha256_file(path)}
    import mujoco
    from .trainedactor_evaluation import verify_compiled_controller
    compiled = (report_path.parent/"compiled_scene.mjb").resolve(strict=True)
    if (not compiled.is_relative_to(report_path.parent)
            or sha256_file(compiled) != report.get("compiled_scene", {}).get("sha256")):
        raise ValueError("Compiled native guard scene hash/path mismatch")
    model = mujoco.MjModel.from_binary_path(str(compiled))
    verify_compiled_controller(model, checked)
    if (model.opt.timestep != .005 or model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            or model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL
            or model.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON
            or model.opt.iterations != 10 or model.opt.ls_iterations != 20
            or model.opt.tolerance != 1e-8
            or report.get("simulation_contract") != support["executed_solver_contract"]["mujoco"]):
        raise ValueError("Actual compiled native solver differs from the guard learning contract")
    files["compiled_scene.mjb"] = {"path": str(compiled), "sha256": sha256_file(compiled)}
    with np.load(files["native_controller_trace.npz"]["path"], allow_pickle=False) as saved:
        native = {name: saved[name].copy() for name in saved.files}
    audit = audit_probe_arrays(native, policy_steps=300, physics_dt_s=.005,
                               effort_limits=checked.body["effort"], root_dof_start=0)
    if not np.array_equal(native["qvel"][0], np.zeros_like(native["qvel"][0])):
        raise ValueError("Guard learning native evidence must start with zero initial velocity")
    torque = np.asarray(support["joint_feedforward_nm"]) if feedforward_mode == "static_lp" else np.zeros(48)
    drive = audit_static_drive(native, checked, torque)
    if (audit["status"] != "passed" or drive["status"] != "passed"
            or (feedforward_mode == "static_lp" and report.get("feedforward_artifact_sha256") != sha256_file(artifact_path))):
        raise ValueError("Actual native clock/force/drive contract failed")
    geometry = guard_geometry(checked)
    if not geometry["passed"]:
        raise ValueError("Guard geometry is unsuitable even for stabilization learning")
    payload = {"schema": SCHEMA, "scope": "guard_stabilization_learning_only", "origin": ORIGIN,
        "reference_manifest_sha256": sha256_file(checked.root/"manifest.json"),
        "model_sha256": sha256_file(checked.model_path), "controller_sha256": checked.controller_sha256(),
        "qpos_sha256": guard["qpos_sha256"], "source_sha256": sha256_file(Path(__file__)),
        "native_guard_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "support_artifact": {"path": str(artifact_path), "sha256": sha256_file(artifact_path)},
        "native_traces": files, "geometry": geometry, "feedforward_mode": feedforward_mode,
        "joint_feedforward_nm": torque.tolist(), "native_clock_force_passed": True,
        "fixed_controller_guard_admission_passed": report.get("guard_admission_passed", False),
        "fixed_controller_falls": audit["fall_diagnostic_environments"],
        "punch_reference_training_qualified": False, "target_ttc_domain_qualified": False,
        "zero_residual_stability_required_for_learning": False,
        "support_lp_is_hypothetical_not_applied": True, "external_assistance": False}
    return {**payload, "sha256": digest(payload)}


def verify_guard_admission(bundle, admission: dict) -> None:
    if admission.get("sha256") != digest({k: v for k, v in admission.items() if k != "sha256"}):
        raise ValueError("Guard admission digest changed")
    fresh = prepare_guard_admission(bundle, Path(admission["native_guard_report"]["path"]),
        Path(admission["support_artifact"]["path"]), feedforward_mode=admission["feedforward_mode"])
    if fresh != admission:
        raise ValueError("Guard-only posture/controller/native evidence changed")


def guard_command_types():
    import torch
    from .runtime.command import PunchMotionCommand, PunchMotionCommandCfg

    class GuardLearningCommand(PunchMotionCommand):
        @property
        def elapsed(self):
            # Preserve clock counters used by native actuation; conditioning is
            # frozen in the prelaunch guard phase. Episode timeout uses env time.
            return torch.zeros_like(self.physics_steps, dtype=torch.float32)

        def sample_master(self, name, offset_steps=0):
            del offset_steps
            value = self.master[name][self.master_starts[self.trajectory_ids]]
            if name == "joint_feedforward":
                return torch.as_tensor(self.cfg.static_joint_feedforward_nm,
                    dtype=value.dtype, device=value.device).expand_as(value)
            if name in ("qvel", "body_lin_vel_w", "body_ang_vel_w", "anchor_lin_acc_w", "anchor_ang_acc_w"):
                return torch.zeros_like(value)
            return value

        def _update_command(self, env_ids):
            ids = slice(None) if env_ids is None else env_ids
            self.time_steps[ids] = self.motion.segment_start_idx[self.trajectory_ids[ids]]
            self.update_relative_body_poses()

    @dataclass(kw_only=True)
    class GuardLearningCommandCfg(PunchMotionCommandCfg):
        static_joint_feedforward_nm: tuple[float, ...] = ()

        def build(self, env):
            return GuardLearningCommand(self, env)

    return GuardLearningCommand, GuardLearningCommandCfg


def build_guard_training_cfg(bundle, admission: dict, plan: GuardSmokePlan, *, evaluation=False):
    """Same actor/critic dimensions and native actuator semantics, static reference."""
    from .runtime.task import build_human_env_cfg
    verify_guard_admission(bundle, admission)
    cfg = build_human_env_cfg(bundle, num_envs=plan.environments, evaluation=True)
    _, command_cfg = guard_command_types()
    old = cfg.commands["motion"]
    cfg.commands["motion"] = command_cfg(**{field.name: getattr(old, field.name) for field in fields(old)},
        static_joint_feedforward_nm=tuple(admission["joint_feedforward_nm"]))
    apply_conditioning(cfg, plan.conditioning)
    cfg.rewards.pop("timed_target", None)
    cfg.terminations.pop("cycle_complete", None)
    cfg.episode_length_s = plan.episode_seconds
    cfg.auto_reset = not evaluation
    cfg.seed = plan.seed
    cfg.sim.nan_guard.enabled = True
    # Scene._add_terrain writes these effective scene values into the terrain
    # config during construction. Bind them before measuring the contract so
    # the same native configuration has one identity before and after init.
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.num_envs = cfg.scene.num_envs
        cfg.scene.terrain.env_spacing = cfg.scene.env_spacing
    return cfg


def guard_measurement_contract(bundle, admission, plan, env_cfg, agent_cfg):
    """Guard-specific bound configuration; never calls punch qualification."""
    from .measurement_contract import configuration_value, contract_from_parts, source_inventory
    from .provenance import runtime_record
    checked = load_runtime_bundle(bundle)
    environment = configuration_value(env_cfg, bundle_root=checked.root)
    environment["fields"] = [[key, value] for key, value in environment["fields"] if key not in ("seed", "viewer")]
    for key, value in environment["fields"]:
        if key == "sim":
            for name, config in value["fields"]:
                if name == "nan_guard":
                    config["fields"] = [[field, setting] for field, setting in config["fields"] if field != "output_dir"]
    learner = configuration_value(agent_cfg, bundle_root=checked.root)
    excluded = {"seed", "max_iterations", "save_interval", "experiment_name", "run_name", "resume",
                "load_run", "load_checkpoint", "wandb_project", "wandb_tags", "neptune_project"}
    learner["fields"] = [[key, value] for key, value in learner["fields"] if key not in excluded]
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise ValueError("Guard capacity requires the pinned runtime")
    if (env_cfg.scene.num_envs != plan.environments or not env_cfg.sim.nan_guard.enabled
            or (agent_cfg.num_steps_per_env, agent_cfg.algorithm.num_learning_epochs,
                agent_cfg.algorithm.num_mini_batches) != (24, 5, 4)):
        raise ValueError("Guard capacity native environment/PPO configuration changed")
    return contract_from_parts(
        bundle_identity={"reference_manifest_sha256": sha256_file(checked.root/"manifest.json"),
            "body_sha256": sha256_file(checked.model_path), "controller_sha256": checked.controller_sha256(),
            "guard_admission_sha256": admission["sha256"], "scope": "static_guard_only"},
        scale=TrainingScale(global_environments=plan.environments, world_size=1),
        conditioning=plan.conditioning, environment_configuration=environment, learner_configuration=learner,
        actor_observation_schema=configuration_value(env_cfg.observations["actor"], bundle_root=checked.root),
        critic_observation_schema=configuration_value(env_cfg.observations["critic"], bundle_root=checked.root),
        sources=source_inventory(), runtime={"python": runtime["python"], "packages": runtime["packages"]})


def require_guard_capacity(capacity: dict, admission: dict, plan: GuardSmokePlan) -> None:
    """No measured guard capacity exists yet; estimates cannot authorize execution."""
    if (capacity.get("schema") != "human_wbc_guard_capacity_v1" or capacity.get("status") != "passed"
            or capacity.get("admission_sha256") != admission["sha256"]
            or capacity.get("plan_sha256") != digest(asdict(plan))
            or capacity.get("includes_inherited_ppo") is not True
            or capacity.get("gpu_count") != 1 or "L40S" not in str(capacity.get("gpu_name", ""))
            or capacity.get("normalization_finite") is not True
            or capacity.get("optimizer_steps_per_iteration") != 20
            or not capacity.get("slurm_job_id")):
        raise ValueError("Guard smoke needs matching native capacity/PPO evidence")
    evidence = capacity.get("native_measurement", {})
    path = Path(evidence.get("path", "")).resolve(strict=True)
    if (not path.is_relative_to(PROJECT_ROOT) or not path.is_file()
            or sha256_file(path) != evidence.get("sha256")
            or json.loads(path.read_text()) != {k: v for k, v in capacity.items() if k != "native_measurement"}):
        raise ValueError("Guard capacity requires an intact native measurement artifact")
    rate, headroom = capacity.get("transitions_per_second"), capacity.get("vram_headroom_fraction")
    if (isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0
            or isinstance(headroom, bool) or not isinstance(headroom, (int, float)) or not .15 <= headroom <= 1.):
        raise ValueError("Guard capacity rate/headroom invalid")
    if plan.accounting()["total_transitions"]/rate > 30*60:
        raise ValueError("Proposed guard smoke exceeds its 30-minute measured learning budget")


def make_guard_runner(bundle, admission: dict, plan: GuardSmokePlan, *, capacity: dict):
    """Public runner-class hook; inherited WBC/RSL PPO with a narrower data gate."""
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner
    from .training_session import observation_schema
    from .measurement_contract import verify_measurement_contract, gpu_hardware_identity, verify_gpu_hardware

    class GuardSmokeRunner(PolicyOnlyMotionTrackingRunner):
        def load(self, *args, **kwargs):
            raise RuntimeError("Guard engineering smoke starts from scratch; checkpoint initialization is forbidden")

        def learn(self, num_learning_iterations, init_at_random_ep_len=False):
            del init_at_random_ep_len
            verify_guard_admission(bundle, admission)
            require_guard_capacity(capacity, admission, plan)
            if (getattr(self, "guard_learning_started", False) or self.is_distributed
                    or num_learning_iterations != plan.iterations or self.current_learning_iteration != 0):
                raise ValueError("Guard smoke is one bounded scratch session on one GPU")
            from .native_probe import _require_allocated_cuda
            from .runtime.task import human_ppo_cfg
            _require_allocated_cuda()
            expected = guard_measurement_contract(bundle, admission, plan, self.env.unwrapped.cfg, human_ppo_cfg())
            verify_measurement_contract(capacity.get("measurement_contract", {}), expected)
            verify_gpu_hardware(capacity["gpu_hardware"], gpu_hardware_identity(self.device))
            self.guard_learning_started = True
            # The inherited public implementation owns rollout storage, PPO,
            # optimization and normalization. No punch qualification is changed.
            return super().learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=False)

        def save(self, path, infos=None):
            if not getattr(self, "guard_learning_started", False):
                raise RuntimeError("Cannot label an unstarted actor as guard-trained")
            from .validation import tensor_digest
            provenance = {"schema": "human_wbc_guard_checkpoint_v1", "origin": ORIGIN,
                "scratch_initialized": True, "historical_initialization": False,
                "completed_iterations": self.current_learning_iteration+1,
                "seed": plan.seed, "plan": asdict(plan), "admission": admission,
                "observation_schema": observation_schema(self.env.unwrapped),
                "actor_state_sha256": tensor_digest(self.alg.get_policy().state_dict()),
                "punch_execution_trained": False, "target_ttc_domain_qualified": False,
                "automatic_punch_initialization": False}
            MjlabOnPolicyRunner.save(self, path, {**(infos or {}), "guard_training": provenance})
            export_directory = Path(path).parent/"params"
            self.export_policy_to_jit(str(export_directory), "policy_jit.pt")
            self.export_policy_to_onnx(str(export_directory), "policy.onnx")

    return GuardSmokeRunner


def validate_guard_checkpoint(checkpoint: dict, *, expected_admission: dict, expected_observation_schema: dict):
    """Inspect allowed future initialization ancestry; never load weights here."""
    from .validation import tensor_digest
    info = checkpoint.get("infos", {}).get("guard_training", {})
    if (info.get("schema") != "human_wbc_guard_checkpoint_v1" or info.get("origin") != ORIGIN
            or info.get("scratch_initialized") is not True or info.get("historical_initialization") is not False
            or info.get("admission") != expected_admission
            or info.get("observation_schema") != expected_observation_schema
            or isinstance(info.get("completed_iterations"), bool)
            or not isinstance(info.get("completed_iterations"), int) or not 1 <= info["completed_iterations"] < 250
            or "actor_state_dict" not in checkpoint or "critic_state_dict" not in checkpoint
            or tensor_digest(checkpoint["actor_state_dict"]) != info.get("actor_state_sha256")):
        raise ValueError("Checkpoint is not the bound new scratch shared guard learner")
    return info


def register_guard_smoke_task(bundle, admission: dict, plan: GuardSmokePlan, *, capacity: dict):
    """Register using public mjlab hooks; never construct a live environment."""
    from mjlab.tasks.registry import register_mjlab_task
    from .runtime.task import human_ppo_cfg
    require_guard_capacity(capacity, admission, plan)
    cfg = build_guard_training_cfg(bundle, admission, plan)
    agent = human_ppo_cfg()
    agent.seed, agent.max_iterations, agent.resume = plan.seed, plan.iterations, False
    agent.experiment_name = "human_guard_engineering_smoke"
    register_mjlab_task(TASK_ID, env_cfg=cfg,
        play_env_cfg=build_guard_training_cfg(bundle, admission, plan, evaluation=True),
        rl_cfg=agent, runner_cls=make_guard_runner(bundle, admission, plan, capacity=capacity))
    return TASK_ID


def _run_guard_learning_phase(bundle, admission, plan, destination, *, phase, deadline, capacity=None):
    """Single native phase; stock WBC learn owns every PPO update."""
    import gc
    import random
    import time
    from copy import deepcopy
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.tasks.registry import register_mjlab_task, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner
    from .capacity import CapacityConfig, _RunnerTiming
    from .measurement_contract import gpu_hardware_identity, verify_measurement_contract
    from .provenance import write_json
    from .runtime.task import human_ppo_cfg
    from .runtime.runner import require_float32_observations
    from .training_session import observation_schema
    from .validation import tensor_digest

    class DisposableCapacityRunner(PolicyOnlyMotionTrackingRunner):
        def save(self, path, infos=None):
            MjlabOnPolicyRunner.save(self, path, {**(infos or {}), "guard_capacity_only": True,
                                                "initialization_for_training_allowed": False})
            self.export_policy_to_jit(str(Path(path).parent/"params"), "policy_jit.pt")
            self.export_policy_to_onnx(str(Path(path).parent/"params"), "policy.onnx")

    destination.mkdir(exist_ok=False)
    cfg = build_guard_training_cfg(bundle, admission, plan)
    cfg.sim.nan_guard.output_dir = str(destination/"nan_guard")
    seed = plan.seed+1 if phase == "capacity" else plan.seed
    cfg.seed = seed
    agent = human_ppo_cfg()
    iterations = 5 if phase == "capacity" else plan.iterations
    agent.seed, agent.max_iterations, agent.resume, agent.save_interval = seed, iterations, False, iterations+1
    contract = guard_measurement_contract(bundle, admission, plan, cfg, agent)
    record = {"schema": "human_wbc_guard_learning_phase_v1", "phase": phase,
              "status": "starting", "native_execution": False, "seed": seed,
              "requested_iterations": iterations, "measurement_contract": contract,
              "scratch_initialized": True, "historical_initialization": False}
    write_json(destination/"launch.json", record)
    env = runner = timer = None
    frozen = None
    started = time.monotonic()
    try:
        device = torch.device("cuda:0")
        configure_torch_backends()
        torch.cuda.set_device(device)
        torch.manual_seed(seed); torch.cuda.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        record["gpu_hardware"] = gpu_hardware_identity(device)
        if "L40S" not in record["gpu_hardware"]["name"]:
            raise RuntimeError("Guard learning qualification requires the allocated L40S")
        stream = torch.cuda.ExternalStream(wp.get_stream("cuda:0").cuda_stream, device=device)
        caller = torch.cuda.current_stream(device)
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            torch.cuda.reset_peak_memory_stats(device)
            initial_free, total = torch.cuda.mem_get_info(device)
            runner_type = DisposableCapacityRunner if phase == "capacity" else make_guard_runner(
                bundle, admission, plan, capacity=capacity)
            task_id = TASK_ID+"-"+phase
            play = deepcopy(cfg)
            play.auto_reset = False
            register_mjlab_task(task_id, env_cfg=cfg, play_env_cfg=play, rl_cfg=agent, runner_cls=runner_type)
            env = ManagerBasedRlEnv(cfg, device="cuda:0")
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            record["observation_dtype_audit"] = require_float32_observations(env)
            record["initialized_measurement_contract"] = guard_measurement_contract(
                bundle, admission, plan, env.cfg, agent)
            verify_measurement_contract(contract, record["initialized_measurement_contract"])
            registered_runner = load_runner_cls(task_id)
            runner = registered_runner(wrapped, asdict(agent), str(destination/"learner"), "cuda:0")
            if env.num_envs != plan.environments or runner.is_distributed:
                raise RuntimeError("Native guard learner environment/rank count differs from its plan")
            record.update(actual_environments=int(env.num_envs), observation_schema=observation_schema(env),
                actor_width=int(runner.alg.get_policy().obs_dim), critic_width=int(runner.alg._raw_critic.obs_dim),
                initial_actor_state_sha256=tensor_digest(runner.alg.get_policy().state_dict()),
                initialization_wall_time_s=time.monotonic()-started)
            if (list(env.observation_manager.group_obs_dim["actor"]) != [record["actor_width"]]
                    or list(env.observation_manager.group_obs_dim["critic"]) != [record["critic_width"]]):
                raise RuntimeError("Actual actor/critic widths differ from the native observation schema")
            original_act = runner.alg.act
            def bounded_act(*args, **kwargs):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Guard session application wall budget reached")
                return original_act(*args, **kwargs)
            runner.alg.act = bounded_act
            timing_cfg = CapacityConfig(global_environments=plan.environments, gpu_count=1,
                warmup_iterations=2 if phase == "capacity" else 1,
                measurement_iterations=3 if phase == "capacity" else 1,
                seed=seed, conditioning=plan.conditioning)
            timer = _RunnerTiming(runner, timing_cfg, torch, device)
            timer.install()
            # Exactly one inherited call; no duplicated completed-iteration index.
            runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
            if runner.logger.writer is not None:
                runner.logger.writer.flush(); runner.logger.writer.close()
            metrics = timer.finish()
            record.update(metrics, native_execution=True)
            rows = metrics["iterations"]
            if (len(rows) != iterations or runner.current_learning_iteration != iterations-1
                    or any(row["policy_steps"] != 24 or row["physics_steps"] != 96
                           or row["optimizer_steps"] != 20 for row in rows)):
                raise RuntimeError("Actual inherited rollout/optimization counts do not match requested learning")
            states = [runner.alg.get_policy().state_dict(), runner.alg._raw_critic.state_dict()]
            record["normalization_finite"] = all(bool(torch.isfinite(value).all()) for state in states
                                                   for key, value in state.items() if key.startswith("obs_normalizer."))
            if not record["normalization_finite"] or not metrics["finite_learning"]:
                raise FloatingPointError("Nonfinite guard actor, critic, normalization or PPO loss")
            checkpoint = destination/"learner"/f"model_{iterations-1}.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if tensor_digest(saved["actor_state_dict"]) != tensor_digest(runner.alg.get_policy().state_dict()):
                raise RuntimeError("Saved guard checkpoint does not match the final actor")
            import onnx
            onnx_path = destination/"learner/params/policy.onnx"
            jit_path = destination/"learner/params/policy_jit.pt"
            onnx.checker.check_model(str(onnx_path))
            jit = torch.jit.load(str(jit_path), map_location=device).eval()
            frozen = runner.alg.get_policy().as_jit().to(device).eval()
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
            with torch.inference_mode():
                observations = env.observation_manager.compute(update_history=False)["actor"][:16]
                error = float((jit(observations)-frozen(observations)).abs().max())
            if not math.isfinite(error) or error > 1e-5:
                raise RuntimeError("Guard JIT checkpoint/normalization parity failed")
            free_after, _ = torch.cuda.mem_get_info(device)
            record.update(status="passed", completed_iterations=iterations,
                total_fresh_transitions=sum(row["policy_steps"] for row in rows)*plan.environments,
                total_optimizer_steps=sum(row["optimizer_steps"] for row in rows),
                final_actor_state_sha256=tensor_digest(runner.alg.get_policy().state_dict()),
                checkpoint_path=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                jit_path=str(jit_path), jit_sha256=sha256_file(jit_path), onnx_sha256=sha256_file(onnx_path),
                jit_native_action_max_abs_error=error, checkpoint_export_verified=True)
            record["conservative_peak_used_bytes"] = max(record["conservative_peak_used_bytes"],
                int(total-min(initial_free, free_after)))
        caller.wait_stream(stream)
    except Exception as error:
        record.update(status="failed", exception={"type": type(error).__name__, "message": str(error)})
        if timer is not None:
            record["completed_timing_rows"] = timer.rows
        raise
    finally:
        if timer is not None:
            timer.restore()
        if runner is not None and runner.logger.writer is not None:
            runner.logger.writer.close()
        if env is not None:
            env.close()
        record["wall_time_s"] = time.monotonic()-started
        write_json(destination/"phase.json", record)
        del runner, env, timer
        gc.collect()
        torch.cuda.empty_cache()
    return record, frozen


def _record_evaluation_outcome(record: dict, audit_path: Path) -> None:
    """Distinguish a completed diagnostic from a failed/unexecuted evaluation."""
    audit = json.loads(audit_path.read_text())
    episodes = audit.get("episodes", [])
    complete = (audit.get("schema") == "human_wbc_learned_guard_evaluation_v1"
        and audit.get("status") == "recorded" and audit.get("recorded_samples") == 1201
        and audit.get("environments") == 16 and len(episodes) == 16
        and all(row.get("status") == "evaluated" for row in episodes))
    record.update(status="recorded" if complete else "evaluation_failed",
        evaluation_path=str(audit_path), evaluation_sha256=sha256_file(audit_path),
        evaluation_status=audit.get("status"), evaluation_completed=complete)
    if not complete:
        record["evaluation_exception"] = audit.get("exception")
    # Guard success is intentionally separate: a complete panel with observed
    # physical failures is valid evidence, unlike a panel that never executed.


def run_guard_session(bundle, native_guard_report: Path, support_artifact: Path, output: Path,
                      plan: GuardSmokePlan = GuardSmokePlan(), *, application_seconds=2580.) -> dict:
    """Capacity 2+3 PPO iterations, then a fresh bounded smoke and physical audit."""
    import gc
    import os
    import time
    import torch
    from .capacity import CapacityConfig, _local_caches, _require_allocation
    from .provenance import runtime_record, writable_path, write_json
    from .trainedactor_evaluation import actor_identity
    if plan.environments < 256 or plan.iterations < 2:
        raise ValueError("Executable capacity+smoke session requires at least 256 environments and two smoke iterations")
    if not 0 < application_seconds <= 2580:
        raise ValueError("Guard application budget cannot exceed 43 minutes inside its 45-minute allocation")
    _require_allocation(CapacityConfig(global_environments=plan.environments, gpu_count=1))
    checked = load_runtime_bundle(bundle)
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    _local_caches(destination)  # One shallow cache for all phases; never nested per learner/rank.
    if len(os.fsencode(Path(os.environ["TMPDIR"]).resolve())) > 120:
        raise ValueError("Use a shallower guard output path for NVRTC temporary filenames")
    admission = prepare_guard_admission(checked, native_guard_report, support_artifact)
    previous = Path.cwd()
    os.chdir(destination)
    started = time.monotonic()
    deadline = started+application_seconds
    record = {"schema": "human_wbc_guard_learning_session_v1", "status": "starting",
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "admission": admission, "plan": asdict(plan),
        "accounting": plan.accounting(), "runtime": runtime_record(), "application_seconds": application_seconds,
        "punch_training": False, "production_training": False, "target_ttc_domain_qualified": False}
    write_json(destination/"launch.json", record)
    try:
        measured, disposable = _run_guard_learning_phase(checked, admission, plan, destination/"capacity",
            phase="capacity", deadline=deadline)
        del disposable
        gc.collect(); torch.cuda.empty_cache()
        headroom = 1-measured["conservative_peak_used_bytes"]/measured["total_vram_bytes"]
        capacity = {"schema": "human_wbc_guard_capacity_v1", "status": "passed",
            "admission_sha256": admission["sha256"], "plan_sha256": digest(asdict(plan)),
            "includes_inherited_ppo": True, "gpu_count": 1, "gpu_name": measured["gpu_hardware"]["name"],
            "gpu_hardware": measured["gpu_hardware"], "normalization_finite": measured["normalization_finite"],
            "optimizer_steps_per_iteration": 20, "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "transitions_per_second": measured["measured_fresh_transitions"]/measured["measurement_wall_time_s"],
            "vram_headroom_fraction": headroom, "measurement_contract": measured["measurement_contract"],
            "phase_path": str(destination/"capacity/phase.json"), "phase_sha256": sha256_file(destination/"capacity/phase.json"),
            "capacity_learner_discarded": True, "warmup_iterations": measured["warmup_iterations"],
            "measurement_iterations": measured["measurement_iterations"]}
        if (measured["warmup_iterations"], measured["measurement_iterations"]) != (2, 3):
            raise RuntimeError("Guard capacity did not complete exactly two warmup plus three measurement iterations")
        capacity_path = destination/"capacity.json"
        write_json(capacity_path, capacity)
        capacity["native_measurement"] = {"path": str(capacity_path), "sha256": sha256_file(capacity_path)}
        require_guard_capacity(capacity, admission, plan)
        record["capacity"] = capacity
        remaining = deadline-time.monotonic()
        estimate = plan.accounting()["total_transitions"]/capacity["transitions_per_second"]
        if remaining < estimate*1.25+measured["initialization_wall_time_s"]+300:
            record.update(status="capacity_passed_smoke_not_started", reason="insufficient_remaining_wall_budget")
            return record
        learned, actor = _run_guard_learning_phase(checked, admission, plan, destination/"smoke",
            phase="smoke", deadline=deadline, capacity=capacity)
        record["smoke"] = learned
        record["learning_completed"] = learned["status"] == "passed" and learned["completed_iterations"] == plan.iterations
        if learned["initial_actor_state_sha256"] == measured["final_actor_state_sha256"]:
            raise RuntimeError("Smoke actor improperly reused capacity learner weights")
        if deadline-time.monotonic() < 180:
            record.update(status="smoke_completed_evaluation_not_started", reason="insufficient_evaluation_wall_budget")
            return record
        from .learned_guard_evaluation import run_learned_guard_evaluation
        identity = {**actor_identity(actor), "reference_manifest_sha256": sha256_file(checked.root/"manifest.json"),
            "model_sha256": sha256_file(checked.model_path), "controller_sha256": checked.controller_sha256(),
            "conditioning": plan.conditioning, "observation_schema": learned["observation_schema"],
            "guard_admission_sha256": admission["sha256"], "joint_feedforward_sha256": digest(admission["joint_feedforward_nm"]),
            "training_checkpoint_sha256": learned["checkpoint_sha256"], "guard_training_origin": ORIGIN}
        evaluated = run_learned_guard_evaluation(checked, actor, destination/"evaluation",
            joint_feedforward_nm=admission["joint_feedforward_nm"], conditioning=plan.conditioning,
            seed=plan.seed, actor_contract_identity=identity)
        _record_evaluation_outcome(record, Path(evaluated))
        return record
    except Exception as error:
        record.update(status="failed", exception={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        record["wall_time_s"] = time.monotonic()-started
        write_json(destination/"session.json", record)
        os.chdir(previous)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "native-guard-report", "support-artifact", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--environments", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10010)
    args = parser.parse_args(argv)
    result = run_guard_session(args.bundle, args.native_guard_report, args.support_artifact, args.output,
        GuardSmokePlan(environments=args.environments, iterations=args.iterations, seed=args.seed))
    print(json.dumps({"status": result["status"], "punch_training": False}, sort_keys=True))
    return 0 if result["status"] == "recorded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
