"""Qualified WBC launcher; optimization is entirely inherited from mjlab/RSL-RL."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path

from .budget import STAGE_CEILINGS, TrainingScale
from .conditioning import Conditioning
from .measurement_contract import build_measurement_contract, verify_measurement_contract
from .provenance import runtime_record, sha256_file, writable_path, write_json


def checked_compute_plan(path: Path, bundle: Path) -> tuple[dict, TrainingScale]:
    plan = json.loads(path.read_text())
    if plan.get("schema") != "human_wbc_measured_compute_plan_v1":
        raise ValueError("A measured Task10 compute plan is required; a planning estimate is insufficient")
    scale = TrainingScale(world_size=plan["gpu_count"])
    if scale.world_size not in (1, 2, 4):
        raise ValueError("Unsupported GPU count")
    stage, iterations = plan["stage"], plan["iterations"]
    if stage not in STAGE_CEILINGS or isinstance(iterations, bool) or not isinstance(iterations, int):
        raise ValueError("Unknown training stage or noninteger iteration budget")
    if not 0 < iterations <= STAGE_CEILINGS[stage]:
        raise ValueError("Run exceeds the accepted per-stage iteration ceiling")
    # Reject unsupported production before constructing an expensive GPU scene.
    # The runner repeats this guard before any inherited optimizer updates.
    from .validation import require_bounded_smoke_without_validation
    require_bounded_smoke_without_validation(plan, iterations)
    if stage != "canonical":
        # Do not restart a later stage from scratch and call it continuation.
        # Full learner/sampler resume needs its own tested checkpoint contract.
        raise ValueError("Later-stage continuation requires the pending qualified resume interface")
    expected = sha256_file(bundle / "manifest.json")
    if plan.get("reference_manifest_sha256") != expected:
        raise ValueError("Compute plan references a different data bundle")
    measurement = plan.get("measurement", {})
    measurement_path = (path.parent / measurement.get("path", "")).resolve()
    if not measurement_path.is_file() or sha256_file(measurement_path) != measurement.get("sha256"):
        raise ValueError("Compute plan requires intact native throughput evidence")
    measured = json.loads(measurement_path.read_text())
    if (measured.get("schema") != "human_wbc_capacity_measurement_v1"
            or measured.get("status") != "passed"
            or measured.get("reference_manifest_sha256") != expected
            or measured.get("gpu_count") != scale.world_size
            or measured.get("global_environments") != scale.global_environments
            or measured.get("includes_ppo") is not True):
        raise ValueError("Capacity evidence does not qualify this full training configuration")
    conditioning = Conditioning(plan.get("conditioning", Conditioning.COMBINED.value)).value
    expected_contract = build_measurement_contract(bundle, scale=scale, conditioning=conditioning)
    verify_measurement_contract(measured.get("measurement_contract", {}), expected_contract)
    if measured.get("measurement_contract_sha256") != expected_contract["sha256"]:
        raise ValueError("Capacity report contract digest differs from this configuration")
    hardware = measured.get("gpu_hardware", {})
    if not isinstance(hardware, dict) or "L40S" not in str(hardware.get("name", "")):
        raise ValueError("Production requires measured L40S hardware identity")
    rate, headroom = measured.get("transitions_per_second", 0), measured.get("vram_headroom_fraction", 0)
    if not math.isfinite(rate) or rate <= 0 or not .15 <= headroom <= 1:
        raise ValueError("Invalid throughput or insufficient GPU memory headroom")
    if plan.get("global_environments") != 8192:
        raise ValueError("Default Task10 production uses 8192 global environments")
    plan["conditioning"] = conditioning
    plan["measured_gpu_hardware"] = hardware
    plan["measurement_contract_sha256"] = expected_contract["sha256"]
    if plan.get("resume"):
        resume = plan["resume"]
        resume["path"] = str((path.parent / resume["path"]).resolve(strict=True))
        if sha256_file(Path(resume["path"]) / "manifest.json") != resume.get("manifest_sha256"):
            raise ValueError("Resume checkpoint manifest does not match its declared hash")
    return plan, scale


def _train_worker(bundle: str, scale_values: dict, plan: dict, seed: int, output: str) -> None:
    # Register in EACH torchrunx process; parent registration is not inherited.
    from mjlab.scripts.train import TrainConfig, run_train
    from dataclasses import asdict
    from .conditioning import Conditioning, apply_conditioning
    from .runtime.task import register_human_task, human_ppo_cfg
    from .training_session import make_session_runner
    from .measurement_contract import gpu_hardware_identity, verify_gpu_hardware
    scale = TrainingScale(**scale_values)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    worker_root = writable_path(Path(output) / f"rank_{rank:02d}")
    worker_root.mkdir(parents=True, exist_ok=False)
    for variable, name in (("XDG_CACHE_HOME", "xdg"), ("XDG_CONFIG_HOME", "config"),
            ("WARP_CACHE_PATH", "warp"), ("CUDA_CACHE_PATH", "cuda"),
            ("TMPDIR", "tmp"), ("MPLCONFIGDIR", "matplotlib"), ("TORCH_HOME", "torch")):
        cache = worker_root / "cache" / name
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    agent_values = asdict(human_ppo_cfg())
    learner = {k: agent_values[k] for k in ("actor", "critic", "algorithm", "obs_groups",
                                           "num_steps_per_env", "clip_actions")}
    runner_cls = make_session_runner(plan, Path(output), learner)
    task = register_human_task(bundle, scale.environments_per_rank, evaluation=False,
                               runner_cls=runner_cls)
    cfg = TrainConfig.from_task(task)
    apply_conditioning(cfg.env, plan.get("conditioning", Conditioning.COMBINED.value))
    cfg.agent.seed = seed
    cfg.agent.max_iterations = plan["iterations"]
    cfg.agent.num_steps_per_env = scale.rollout_steps
    cfg.agent.algorithm.num_learning_epochs = scale.epochs
    cfg.agent.algorithm.num_mini_batches = scale.minibatches
    cfg.agent.resume = False
    cfg = replace(cfg, log_root=output, enable_nan_guard=True)
    cfg.env.sim.nan_guard.output_dir = str(worker_root / "nan_guard")
    cfg.env.sim.nan_guard.enabled = True
    current_contract = build_measurement_contract(bundle, scale=scale,
        conditioning=plan["conditioning"], env_cfg=cfg.env, agent_cfg=cfg.agent)
    if current_contract["sha256"] != plan["measurement_contract_sha256"]:
        raise ValueError("Worker configuration changed after capacity qualification")
    import torch
    import warp as wp
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    verify_gpu_hardware(plan["measured_gpu_hardware"], gpu_hardware_identity(device))
    stream = torch.cuda.ExternalStream(wp.get_stream(str(device)).cuda_stream, device=device)
    caller = torch.cuda.current_stream(device)
    stream.wait_stream(caller)
    previous_directory = Path.cwd()
    os.chdir(worker_root)
    try:
        with torch.cuda.stream(stream):
            run_train(task, cfg, Path(output) / "run")
        caller.wait_stream(stream)
    finally:
        os.chdir(previous_directory)


def launch_qualified_training(bundle: Path, compute_plan: Path, output: Path, seed: int) -> None:
    from dataclasses import asdict
    bundle = bundle.resolve(strict=True)
    plan, scale = checked_compute_plan(compute_plan.resolve(strict=True), bundle)
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise RuntimeError("Training runtime differs from qualified candidate pins")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Training requires an existing explicitly allocated Slurm GPU job")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Invoke the training launcher once; it creates its own local workers")
    import torch
    if torch.cuda.device_count() != scale.world_size:
        raise RuntimeError("Visible GPU count differs from the measured global batch plan")
    from .runtime.bundle import load_runtime_bundle
    load_runtime_bundle(bundle).require_training_qualification()
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "launch.json", {
        "schema": "human_wbc_launch_v1", "seed": seed, "scale": scale.report(plan["iterations"]),
        "compute_plan_sha256": sha256_file(compute_plan), "runtime": runtime,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "status": "starting",
    })
    if scale.world_size == 1:
        _train_worker(str(bundle), asdict(scale), plan, seed, str(destination))
    else:
        import torchrunx
        os.environ["TORCHRUNX_LOG_DIR"] = str(destination / "torchrunx")
        torchrunx.Launcher(
            hostnames=["localhost"], workers_per_host=scale.world_size, backend=None,
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + (
                "MUJOCO*", "WARP*", "SLURM_JOB_ID", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
                "TMPDIR", "MPLCONFIGDIR", "CUDA_CACHE_PATH", "TORCH_HOME", "PYTHONDONTWRITEBYTECODE"),
        ).run(_train_worker, str(bundle), asdict(scale), plan, seed, str(destination))
