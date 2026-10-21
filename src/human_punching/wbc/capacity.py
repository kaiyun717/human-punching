"""Bounded inherited WBC PPO capacity with explicit production or engineering entry.

This module submits no jobs and never qualifies references. The small learner is
a capacity experiment, not a production checkpoint or an acceptance evaluation.
All optimization and normalization remain in WBC/mjlab/RSL-RL. Timing hooks are
scoped to one runner instance and restored even if learning/export fails.
Engineering entry retains unqualified references and requires its own typed
native-controller, geometry and observer evidence; it cannot qualify production.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from .budget import TrainingScale
from .conditioning import Conditioning, apply_conditioning
from .measurement_contract import (build_measurement_contract, digest, gpu_hardware_identity,
                                   verify_measurement_contract)
from .provenance import runtime_record, sha256_file, writable_path, write_json
from .runtime.engineering_recipe import LEGACY_RECIPE_ID, RECIPE_IDS, resolve_engineering_recipe, verify_recipe_manifest

SCHEMA = "human_wbc_capacity_measurement_v1"


@dataclass(frozen=True)
class CapacityConfig:
    global_environments: int = 256
    gpu_count: int = 1
    warmup_iterations: int = 2
    measurement_iterations: int = 3
    seed: int = 10010
    conditioning: str = Conditioning.COMBINED.value
    engineering_recipe: str = LEGACY_RECIPE_ID

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if key in ("conditioning", "engineering_recipe"):
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
        if not 256 <= self.global_environments <= 8192:
            raise ValueError("Capacity probes require 256..8192 global environments")
        if self.gpu_count not in (1, 2, 4):
            raise ValueError("Capacity probes support one, two or four GPUs")
        if not 1 <= self.warmup_iterations <= 20 or not 1 <= self.measurement_iterations <= 20:
            raise ValueError("Warmup and measurement each require 1..20 iterations")
        if self.seed < 0:
            raise ValueError("Seed must be nonnegative")
        object.__setattr__(self, "conditioning", Conditioning(self.conditioning).value)
        object.__setattr__(self, "engineering_recipe", resolve_engineering_recipe(self.engineering_recipe).recipe_id)
        self.scale  # Also validate per-rank rollout/minibatch divisibility.

    @property
    def scale(self) -> TrainingScale:
        return TrainingScale(global_environments=self.global_environments, world_size=self.gpu_count)


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def aggregate_capacity(config: CapacityConfig, ranks: list[dict], manifest_sha256: str,
                       measurement_contract: dict) -> dict:
    """Audit native rank records; synthetic test records do not execute a learner.

    Throughput uses the slowest synchronized rank's complete measured phase,
    including PPO, logging, checkpoint/ONNX saves, and instrumentation overhead.
    Memory combines sampled driver usage with Torch's reserved-memory peak and
    the largest sampled non-Torch allocation; this is conservative, not a claim
    that sampled driver usage captures every transient external allocation.
    """
    scale = config.scale
    iterations = config.measurement_iterations
    required_ranks = set(range(config.gpu_count))
    rank_ids = [r.get("rank") for r in ranks]
    contract_valid = True
    try:
        verify_measurement_contract(measurement_contract, measurement_contract)
    except (TypeError, ValueError, KeyError):
        contract_valid = False
    contract_hash = measurement_contract.get("sha256")
    contract_payload = measurement_contract.get("payload", {})
    schemas = [r.get("native_observation_schema") for r in ranks]
    checks = {
        "all_ranks_present": len(ranks) == config.gpu_count and set(rank_ids) == required_ranks,
        "native_rank_execution": all(r.get("status") == "passed" and r.get("native_execution") is True for r in ranks),
        "matching_reference": all(r.get("reference_manifest_sha256") == manifest_sha256 for r in ranks),
        "matching_dimensions": all(r.get("environments_per_rank") == scale.environments_per_rank
                                   and r.get("actual_environments_per_rank") == scale.environments_per_rank
                                   and r.get("gpu_world_size") == config.gpu_count for r in ranks),
        "matching_conditioning": all(r.get("conditioning") == config.conditioning for r in ranks),
        "matching_measurement_contract": contract_valid and all(
            r.get("measurement_contract_sha256") == contract_hash for r in ranks)
            and contract_payload.get("conditioning") == config.conditioning
            and contract_payload.get("reference_and_body", {}).get("reference_manifest_sha256") == manifest_sha256
            and contract_payload.get("scale") == {
                "global_environments": config.global_environments, "gpu_count": config.gpu_count,
                "environments_per_rank": scale.environments_per_rank, "rollout_steps": 24,
                "epochs": 5, "minibatches": 4},
        "native_observation_schema": all(isinstance(schema, dict)
            and _positive(schema.get("actor_width")) and _positive(schema.get("critic_width"))
            and r.get("native_observation_schema_sha256") == digest(schema)
            for r, schema in zip(ranks, schemas))
            and len({r.get("native_observation_schema_sha256") for r in ranks}) == 1,
        "matching_l40s_hardware": all(isinstance(r.get("gpu_hardware"), dict)
            and "L40S" in r["gpu_hardware"].get("name", "")
            and _positive(r["gpu_hardware"].get("total_memory_bytes")) for r in ranks)
            and len({digest(r.get("gpu_hardware")) for r in ranks}) == 1,
        "complete_warmup": all(r.get("warmup_iterations") == config.warmup_iterations for r in ranks),
        "complete_measurement": all(r.get("measurement_iterations") == iterations for r in ranks),
        "actual_rollout_counts": all(r.get("measured_policy_steps") == iterations * 24
                                     and r.get("measured_physics_steps") == iterations * 24 * 4
                                     and r.get("measured_fresh_transitions") == iterations * scale.transitions_per_rank
                                     for r in ranks),
        "actual_optimizer_counts": all(r.get("measured_optimizer_steps") == iterations * 20 for r in ranks),
        "finite_timing": all(_positive(r.get("measurement_wall_time_s"))
                             and _positive(r.get("measured_ppo_update_wall_time_s"))
                             and _positive(r.get("initialization_wall_time_s"))
                             and _positive(r.get("warmup_wall_time_s")) for r in ranks),
        "finite_learning": all(r.get("finite_learning") is True for r in ranks),
        "checkpoint_and_export": any(r.get("rank") == 0 and r.get("checkpoint_export_verified") is True for r in ranks),
        "valid_memory": all(_positive(r.get("total_vram_bytes"))
                            and _positive(r.get("conservative_peak_used_bytes"))
                            and 0 < r.get("conservative_peak_used_bytes", 0) <= r.get("total_vram_bytes", 0)
                            for r in ranks),
    }
    identity = contract_payload.get("reference_and_body", {})
    engineering = identity.get("scope") == "canonical_four_family_scratch_engineering"
    checks["engineering_recipe_scope"] = engineering or config.engineering_recipe == LEGACY_RECIPE_ID
    if engineering:
        from .capacity_evidence import fresh_imitation_health_matches, initialization_health_matches
        recipe = resolve_engineering_recipe(config.engineering_recipe)
        recipe_valid = True
        try:
            verify_recipe_manifest(identity.get("engineering_recipe"), identity.get("engineering_recipe_sha256"))
        except (TypeError, ValueError):
            recipe_valid = False
        checks["matching_engineering_recipe"] = (recipe_valid
            and identity.get("engineering_recipe_sha256") == recipe.sha256
            and all(row.get("engineering_recipe_sha256") == recipe.sha256 for row in ranks))
        checks["initialization_start_accounting"] = all(
            initialization_health_matches(row.get(name), recipe)
            for row in ranks for name in ("cycle_after_warmup", "cycle_after"))
        checks["matching_training_initialization_evidence"] = all(
            row.get("training_initialization_evidence_sha256") == identity.get("training_initialization_evidence_sha256")
            for row in ranks) and (
                bool(identity.get("training_initialization_evidence_sha256")) if recipe.training.reference_start_probability > 0
                else identity.get("training_initialization_evidence_sha256") is None)
        if recipe.training.reference_start_probability > 0:
            checks["mixed_start_workloads_exercised"] = all(
                row.get("cycle_after", {}).get("cumulative_training_reference_starts", 0) > 0
                and row.get("cycle_after", {}).get("cumulative_strict_guard_starts", 0) > 0 for row in ranks)
        checks["matching_engineering_entry"] = all(
            row.get("engineering_admission_sha256") == identity.get("engineering_admission_sha256")
            and row.get("observer_evidence_sha256") == identity.get("observer_evidence_sha256")
            and row.get("bridge_evidence_sha256") == identity.get("bridge_evidence_sha256") for row in ranks)
        checks["native_observer_enabled"] = all(row.get("native_observer_enabled") is True for row in ranks)
        checks["cycle_bridge_accounting"] = all(row.get("cycle_bridge_enabled") is True
            and row.get("task_objectives_enabled") is True
            and row.get("cycle_after", {}).get("terminal_accounting_consistent") is True
            and row.get("cycle_after", {}).get("readiness") == {"quiet_ticks": 40, "maximum_settling_ticks": 1200}
            and row.get("native_observation_schema", {}).get("actor_width") == 398 for row in ranks)
        checks["raw_wbc_imitation_rewards"] = all(
            fresh_imitation_health_matches(row.get("cycle_after_warmup", {}).get("fresh_imitation"),
                                           config.warmup_iterations*24)
            and fresh_imitation_health_matches(row.get("cycle_after", {}).get("fresh_imitation"),
                                               (config.warmup_iterations+iterations)*24)
            for row in ranks)
        checks["distributed_normalization"] = all(
            row.get("normalization_after", {}).get("all_ranks_match") is True
            and row.get("normalization_after", {}).get("normalization_finite") is True
            and row.get("normalization_after", {}).get("rank_state_sha256")
                == [digest(row.get("normalization_after", {}).get("states"))]*config.gpu_count
            and all(row.get("normalization_after", {}).get("states", {}).get(name, {}).get("count", -1)
                    > row.get("normalization_before", {}).get("states", {}).get(name, {}).get("count", 0)
                    and len(row.get("normalization_after", {}).get("states", {}).get(name, {}).get("model_state_sha256", "")) == 64
                    for name in ("actor", "critic"))
            for row in ranks) and len({digest(row.get("normalization_after", {}).get("states"))
                                      for row in ranks}) == 1
    complete = all(checks.values())
    wall = max(r["measurement_wall_time_s"] for r in ranks) if complete else None
    headroom = min(1-r["conservative_peak_used_bytes"]/r["total_vram_bytes"] for r in ranks) if complete else None
    checks["fifteen_percent_vram_headroom"] = headroom is not None and headroom >= .15
    transitions = iterations * scale.fresh_transitions
    return {
        "schema": SCHEMA, "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "gpu_count": config.gpu_count,
        "global_environments": config.global_environments,
        "conditioning": config.conditioning,
        "measurement_contract": measurement_contract,
        "measurement_contract_sha256": contract_hash,
        "native_observation_schema": schemas[0] if complete else None,
        "native_observation_schema_sha256": ranks[0].get("native_observation_schema_sha256") if complete else None,
        "gpu_hardware": ranks[0].get("gpu_hardware") if complete else None,
        "environments_per_rank": scale.environments_per_rank,
        "rollout_steps": 24, "epochs": 5, "minibatches": 4,
        "configured_optimizer_steps_per_iteration": 20,
        "fresh_transitions_per_iteration": scale.fresh_transitions,
        "measurement_iterations": iterations, "warmup_iterations": config.warmup_iterations,
        "measured_fresh_transitions": transitions if complete else None,
        "measurement_wall_time_s": wall,
        "transitions_per_second": transitions / wall if wall else None,
        "vram_headroom_fraction": headroom,
        "includes_ppo": True, "reference_manifest_sha256": manifest_sha256,
        "production_checkpoint": False, "complete_cycle_acceptance": False,
        "learning_scope": identity.get("scope", "production_reference_capacity"),
        "production_reference_qualified": not engineering,
        "engineering_admission_sha256": identity.get("engineering_admission_sha256"),
        "observer_evidence_sha256": identity.get("observer_evidence_sha256"),
        "bridge_evidence_sha256": identity.get("bridge_evidence_sha256"),
        "engineering_recipe": identity.get("engineering_recipe"),
        "engineering_recipe_sha256": identity.get("engineering_recipe_sha256"),
        "training_initialization_evidence_sha256": identity.get("training_initialization_evidence_sha256"),
        "strict_ready_launch_workload_exercised": any(row.get("cycle_after", {}).get("cumulative_strict_ready_launches", 0) > 0 for row in ranks),
        "training_reference_start_workload_exercised": any(row.get("cycle_after", {}).get("cumulative_training_reference_starts", 0) > 0 for row in ranks),
        "launched_workload_exercised": any(row.get("cycle_after", {}).get("launch_workload_exercised", False) for row in ranks),
        "launch_workload_limitation": "Zero launches measures settling/reset workload, not launched-cycle throughput.",
        "timing_scope": "synchronized phase includes rollout, PPO, logging, saves, export and instrumentation",
        "memory_scope": "max(sampled driver used, max Torch reserved + max sampled non-Torch used); samples are not a continuous external-allocation peak",
        "compilation_scope": "initialization and warmup include compilation/first-use; compiler-only time is not separately instrumented",
        "ranks": ranks,
    }


def _require_allocation(config: CapacityConfig) -> None:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Capacity measurement requires an existing allocated Slurm CUDA job")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != config.gpu_count:
        raise RuntimeError("Visible CUDA devices must match the requested GPU count")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Invoke the capacity launcher once; it creates its own local workers")


def _local_caches(destination: Path, *, shared_tmp: Path | None = None) -> None:
    for variable, directory in {
        "XDG_CACHE_HOME": "xdg", "XDG_CONFIG_HOME": "config", "TMPDIR": "tmp",
        "MPLCONFIGDIR": "matplotlib", "TORCH_HOME": "torch", "WARP_CACHE_PATH": "warp",
        "CUDA_CACHE_PATH": "cuda", "WANDB_CACHE_DIR": "wandb", "WANDB_CONFIG_DIR": "wandb_config",
    }.items():
        path = writable_path(shared_tmp if variable == "TMPDIR" and shared_tmp is not None
                             else destination / "cache" / directory)
        if variable == "TMPDIR" and len(os.fsencode(path)) > 120:
            raise ValueError("Native NVRTC requires a shallow resolved TMPDIR (at most 120 bytes)")
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


class _RunnerTiming:
    """Instance-scoped observers; all wrapped methods delegate unchanged."""

    def __init__(self, runner, config: CapacityConfig, torch, device, *, phase_boundary=None):
        self.runner, self.config, self.torch, self.device = runner, config, torch, device
        self.rows: list[dict] = []
        self.row: dict | None = None
        self.restorations: list[tuple[Any, str, Any]] = []
        self.start = 0.
        self.measure_start = 0.
        self.total_vram = 0
        self.max_driver_used = 0
        self.max_non_torch_used = 0
        self.max_torch_reserved = 0
        self.max_torch_allocated = 0
        self.finite_learning = True
        self.optimizer_hook = None
        self.phase_boundary = phase_boundary

    def sync(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def memory(self) -> None:
        free, total = self.torch.cuda.mem_get_info(self.device)
        reserved = self.torch.cuda.memory_reserved(self.device)
        self.total_vram = int(total)
        self.max_driver_used = max(self.max_driver_used, int(total-free))
        self.max_non_torch_used = max(self.max_non_torch_used, int(total-free-reserved))
        self.max_torch_reserved = max(self.max_torch_reserved, int(self.torch.cuda.max_memory_reserved(self.device)))
        self.max_torch_allocated = max(self.max_torch_allocated, int(self.torch.cuda.max_memory_allocated(self.device)))
        if self.row is not None:
            for key, value in {
                "sampled_peak_driver_used_bytes": int(total-free),
                "sampled_peak_non_torch_used_bytes": max(0, int(total-free-reserved)),
                "torch_peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved(self.device)),
                "torch_peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
            }.items():
                self.row[key] = max(self.row.get(key, 0), value)

    def _wrap(self, owner, name: str, function) -> None:
        original = getattr(owner, name)
        self.restorations.append((owner, name, original))
        setattr(owner, name, function(original))

    def _finish_iteration(self) -> None:
        if self.row is None:
            return
        self.sync()
        self.memory()
        env = self.runner.env.unwrapped
        self.row.update(
            wall_time_s=time.perf_counter()-self.row.pop("started"),
            policy_steps=int(env.common_step_counter)-self.row.pop("initial_policy_counter"),
            physics_steps=int(env._sim_step_counter)-self.row.pop("initial_physics_counter"),
        )
        if self.row["policy_steps"] != 24 or self.row["physics_steps"] != 96 or self.row["optimizer_steps"] != 20:
            raise RuntimeError("Measured rollout/optimizer counts differ from 24 steps, decimation four and 5x4 PPO")
        self.rows.append(self.row)
        self.row = None

    def _begin_iteration(self) -> None:
        self._finish_iteration()
        index = len(self.rows)
        if index == self.config.warmup_iterations:
            if self.phase_boundary is not None:
                self.phase_boundary("warmup")
            if self.runner.is_distributed:
                self.torch.distributed.barrier()
            self.sync()
            self.measure_start = time.perf_counter()
        env = self.runner.env.unwrapped
        self.row = {
            "index": index, "phase": "warmup" if index < self.config.warmup_iterations else "measurement",
            "started": time.perf_counter(), "initial_policy_counter": int(env.common_step_counter),
            "initial_physics_counter": int(env._sim_step_counter), "act_calls": 0,
            "optimizer_steps": 0, "ppo_update_wall_time_s": 0., "returns_wall_time_s": 0.,
            "save_export_wall_time_s": 0., "save_calls": 0,
        }
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.memory()

    def install(self) -> None:
        def act(original):
            def timed(*args, **kwargs):
                if self.row is None or self.row["act_calls"] == 24:
                    self._begin_iteration()
                self.row["act_calls"] += 1
                return original(*args, **kwargs)
            return timed

        def duration(field):
            def wrap(original):
                def timed(*args, **kwargs):
                    self.sync()
                    self.memory()
                    started = time.perf_counter()
                    result = original(*args, **kwargs)
                    self.sync()
                    self.row[field] += time.perf_counter()-started
                    self.memory()
                    return result
                return timed
            return wrap

        def log(original):
            def timed(*args, **kwargs):
                # RSL 5.5 calls log with named metrics. Missing or nonfinite losses
                # are a failed capacity run, not usable throughput evidence.
                losses = kwargs.get("loss_dict", {})
                finite = bool(losses) and all(math.isfinite(float(v)) for v in losses.values())
                finite = finite and all(bool(self.torch.isfinite(p).all()) for model in
                                        (self.runner.alg.actor, self.runner.alg.critic) for p in model.parameters())
                self.finite_learning = self.finite_learning and finite
                if not finite:
                    raise FloatingPointError("Nonfinite PPO losses or model parameters")
                self.row["losses"] = {k: float(v) for k, v in losses.items()}
                self.row["rsl_collect_time_s"] = float(kwargs["collect_time"])
                self.row["rsl_learn_time_s"] = float(kwargs["learn_time"])
                return original(*args, **kwargs)
            return timed

        def save(original):
            counted = duration("save_export_wall_time_s")(original)
            def timed(*args, **kwargs):
                result = counted(*args, **kwargs)
                self.row["save_calls"] += 1
                return result
            return timed

        self._wrap(self.runner.alg, "act", act)
        self._wrap(self.runner.alg, "compute_returns", duration("returns_wall_time_s"))
        self._wrap(self.runner.alg, "update", duration("ppo_update_wall_time_s"))
        self._wrap(self.runner.logger, "log", log)
        self._wrap(self.runner, "save", save)
        self.optimizer_hook = self.runner.alg.optimizer.register_step_post_hook(self._optimizer_step)
        self.sync()
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.memory()
        self.start = time.perf_counter()

    def _optimizer_step(self, optimizer, args, kwargs):
        del optimizer, args, kwargs
        self.row["optimizer_steps"] += 1

    def finish(self) -> dict:
        # Final save/export is inside the inherited learn call and precedes this
        # barrier. No warmup checkpoint is charged to the measurement phase.
        if self.runner.is_distributed:
            self.torch.distributed.barrier()
        self.sync()
        self._finish_iteration()
        ended = time.perf_counter()
        measured = [r for r in self.rows if r["phase"] == "measurement"]
        warmup = [r for r in self.rows if r["phase"] == "warmup"]
        return {
            "warmup_iterations": len(warmup), "measurement_iterations": len(measured),
            "warmup_wall_time_s": self.measure_start-self.start,
            "measurement_wall_time_s": ended-self.measure_start,
            "measured_policy_steps": sum(r["policy_steps"] for r in measured),
            "measured_physics_steps": sum(r["physics_steps"] for r in measured),
            "measured_fresh_transitions": sum(r["policy_steps"] for r in measured)*self.config.scale.environments_per_rank,
            "measured_optimizer_steps": sum(r["optimizer_steps"] for r in measured),
            "measured_ppo_update_wall_time_s": sum(r["ppo_update_wall_time_s"] for r in measured),
            "measured_save_export_wall_time_s": sum(r["save_export_wall_time_s"] for r in measured),
            "finite_learning": self.finite_learning, "total_vram_bytes": self.total_vram,
            "sampled_peak_driver_used_bytes": self.max_driver_used,
            "sampled_peak_non_torch_used_bytes": self.max_non_torch_used,
            "torch_peak_reserved_bytes": self.max_torch_reserved,
            "torch_peak_allocated_bytes": self.max_torch_allocated,
            "conservative_peak_used_bytes": max(self.max_driver_used, self.max_torch_reserved+self.max_non_torch_used),
            "iterations": self.rows,
            "phase_memory": {phase: {
                key: max(row[key] for row in rows)
                for key in ("sampled_peak_driver_used_bytes", "sampled_peak_non_torch_used_bytes",
                            "torch_peak_reserved_bytes", "torch_peak_allocated_bytes")}
                for phase, rows in (("warmup", warmup), ("measurement", measured))},
        }

    def restore(self) -> None:
        if self.optimizer_hook is not None:
            self.optimizer_hook.remove()
        for owner, name, original in reversed(self.restorations):
            setattr(owner, name, original)


def _capacity_worker(bundle: str, values: dict, output: str, expected_contract: dict,
                     engineering_values: dict | None = None, observer_report: str | None = None,
                     bridge_report: str | None = None, panel_recipes: str | None = None,
                     training_initialization_report: str | None = None) -> None:
    config = CapacityConfig(**values)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if int(os.environ.get("WORLD_SIZE", "1")) != config.gpu_count:
        raise RuntimeError("Worker world size differs from the global capacity configuration")
    destination = writable_path(Path(output) / f"rank_{rank:02d}")
    destination.mkdir(exist_ok=False)
    _local_caches(destination, shared_tmp=Path(output)/"cache/tmp")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    previous_directory = Path.cwd()
    os.chdir(destination)  # Confine native warning logs as well as explicit caches.
    env, timer, attempt_sink = None, None, None
    record = {"rank": rank, "gpu_world_size": config.gpu_count,
              "environments_per_rank": config.scale.environments_per_rank,
              "conditioning": config.conditioning,
              "reference_manifest_sha256": sha256_file(Path(bundle)/"manifest.json"),
              "status": "failed", "native_execution": True}
    try:
        worker_runtime = runtime_record()
        if not worker_runtime["declared_pins_match"]:
            raise RuntimeError("Capacity worker runtime differs from the candidate pins")
        import torch
        import warp as wp
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.scripts.train import TrainConfig
        from mjlab.tasks.registry import load_runner_cls
        from mjlab.utils.os import dump_yaml
        from mjlab.utils.torch import configure_torch_backends
        from .runtime.task import register_human_task

        admission = None
        runner_override = None
        if engineering_values is not None:
            from .engineering_admission import EngineeringAdmission, engineering_runner
            from .capacity_evidence import (engineering_train_config, distributed_normalization_evidence,
                attach_engineering_cycle, cycle_capacity_health, fresh_imitation_health_matches)
            admission = EngineeringAdmission.from_dict(engineering_values)
            runner_override = engineering_runner(admission)

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        configure_torch_backends()
        task_id = register_human_task(bundle, config.scale.environments_per_rank,
            evaluation=admission is not None, runner_cls=runner_override)
        cfg = TrainConfig.from_task(task_id)
        if admission is not None:
            cfg = engineering_train_config(cfg, bundle, admission,
                num_envs=config.scale.environments_per_rank, conditioning=config.conditioning,
                engineering_recipe=config.engineering_recipe)
            record["engineering_admission_sha256"] = admission.sha256
            record["native_observer_enabled"] = True
            record["cycle_bridge_enabled"] = record["task_objectives_enabled"] = True
            record["conditioning_metadata"] = {"actor_variant": config.conditioning,
                "phase_aware": True, "explicit_actor_command_dimension": 14, "frozen_actor_g1_dimension": 77}
        else:
            record["conditioning_metadata"] = apply_conditioning(cfg.env, config.conditioning)
        cfg.agent.seed = cfg.env.seed = config.seed+rank
        cfg.agent.num_steps_per_env = 24
        cfg.agent.algorithm.num_learning_epochs = 5
        cfg.agent.algorithm.num_mini_batches = 4
        cfg.agent.resume = False
        cfg.agent.max_iterations = config.warmup_iterations+config.measurement_iterations
        cfg.agent.save_interval = cfg.agent.max_iterations+1
        cfg.env.sim.nan_guard.enabled = True
        cfg.env.sim.nan_guard.output_dir = str(destination/"nan_guard")
        measured_contract = build_measurement_contract(bundle, scale=config.scale,
            conditioning=config.conditioning, env_cfg=cfg.env, agent_cfg=cfg.agent,
            engineering_admission=admission, observer_report=observer_report, bridge_report=bridge_report,
            engineering_recipe=config.engineering_recipe, training_initialization_report=training_initialization_report)
        verify_measurement_contract(measured_contract, expected_contract)
        record["measurement_contract_sha256"] = measured_contract["sha256"]
        if admission is not None:
            record["observer_evidence_sha256"] = measured_contract["payload"]["reference_and_body"]["observer_evidence_sha256"]
            record["bridge_evidence_sha256"] = measured_contract["payload"]["reference_and_body"]["bridge_evidence_sha256"]
            record["engineering_recipe_sha256"] = measured_contract["payload"]["reference_and_body"]["engineering_recipe_sha256"]
            record["training_initialization_evidence_sha256"] = measured_contract["payload"]["reference_and_body"]["training_initialization_evidence_sha256"]
        record["gpu_hardware"] = gpu_hardware_identity(device)
        dump_yaml(destination/"params"/"env.yaml", asdict(cfg.env))
        dump_yaml(destination/"params"/"agent.yaml", asdict(cfg.agent))
        runner_cls = load_runner_cls(task_id)
        if runner_cls is None:
            raise RuntimeError("Missing registered human WBC runner")
        stream = torch.cuda.ExternalStream(wp.get_stream(str(device)).cuda_stream, device=device)
        caller = torch.cuda.current_stream(device)
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            initial_free, total_vram = torch.cuda.mem_get_info(device)
            started = time.perf_counter()
            env = ManagerBasedRlEnv(cfg=cfg.env, device=str(device))
            record["actual_environments_per_rank"] = int(env.num_envs)
            if env.num_envs != config.scale.environments_per_rank:
                raise RuntimeError("Native environment count differs from the global capacity plan")
            if admission is not None:
                _, attempt_sink = attach_engineering_cycle(env, admission, output=destination/"attempts",
                    measurement_contract=measured_contract)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
            runner = runner_cls(wrapped, asdict(cfg.agent), str(destination/"learner"), str(device))
            manager = env.observation_manager
            schema = {
                "actor_width": int(runner.alg._raw_actor.obs_dim),
                "critic_width": int(runner.alg._raw_critic.obs_dim),
                "groups": {group: {"names": list(manager.active_terms[group]),
                                   "term_dimensions": [list(shape) for shape in manager.group_obs_term_dim[group]],
                                   "total_shape": list(manager.group_obs_dim[group])}
                           for group in ("actor", "critic")},
            }
            if any(schema["groups"][group]["names"] != list(cfg.env.observations[group].terms)
                   or schema["groups"][group]["total_shape"] != [schema[f"{group}_width"]]
                   for group in ("actor", "critic")):
                raise RuntimeError("Native actor/critic dimensions or ordering differ from the configured observation schema")
            record["native_observation_schema"] = schema
            record["native_observation_schema_sha256"] = digest(schema)
            torch.cuda.synchronize(device)
            record["initialization_wall_time_s"] = time.perf_counter()-started
            free_after, _ = torch.cuda.mem_get_info(device)
            reserved_after = torch.cuda.memory_reserved(device)
            record["initialization_memory"] = {
                "total_vram_bytes": int(total_vram),
                "sampled_peak_driver_used_bytes": int(total_vram-min(initial_free, free_after)),
                "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "sampled_non_torch_used_bytes": max(0, int(total_vram-free_after-reserved_after)),
            }
            if runner.gpu_world_size != config.gpu_count:
                raise RuntimeError("RSL distributed world size differs from requested GPU count")
            if runner.is_distributed:
                torch.distributed.barrier()
            if admission is not None:
                record["normalization_before"] = distributed_normalization_evidence(runner)
                record["cycle_before"] = cycle_capacity_health(env, attempt_sink)
            def phase_boundary(phase):
                if attempt_sink is not None:
                    attempt_sink.flush()
                    record["cycle_after_"+phase] = cycle_capacity_health(env, attempt_sink)
                    expected = config.warmup_iterations*24
                    if phase == "measurement":
                        expected += config.measurement_iterations*24
                    if not fresh_imitation_health_matches(record["cycle_after_"+phase]["fresh_imitation"], expected):
                        raise FloatingPointError("Actual raw WBC imitation outputs were nonfinite, skipped or duplicated")
            timer = _RunnerTiming(runner, config, torch, device, phase_boundary=phase_boundary)
            timer.install()
            # One call preserves inherited iteration numbering and normalization.
            runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=False)
            writer = runner.logger.writer
            if writer is not None:
                writer.flush()
                writer.close()
            phase_boundary("measurement")
            if attempt_sink is not None:
                record["cycle_after"] = record["cycle_after_measurement"]
            record.update(timer.finish())
            if admission is not None:
                record["normalization_after"] = distributed_normalization_evidence(runner, include_parameters=True)
            initial_memory = record["initialization_memory"]
            record["conservative_peak_used_bytes"] = max(
                record["conservative_peak_used_bytes"], initial_memory["sampled_peak_driver_used_bytes"],
                initial_memory["torch_peak_reserved_bytes"]+initial_memory["sampled_non_torch_used_bytes"])
            if rank == 0:
                final = destination/"learner"/f"model_{cfg.agent.max_iterations-1}.pt"
                export = destination/"learner"/"params"/"policy.onnx"
                controller = destination/"learner"/"params"/"human_controller.json"
                record["checkpoint_export_verified"] = all(p.is_file() and p.stat().st_size > 0 for p in (final, export, controller))
                if not record["checkpoint_export_verified"]:
                    raise RuntimeError("Inherited checkpoint/export did not produce all required files")
                import onnx
                onnx.checker.check_model(str(export))
                record["final_checkpoint_sha256"] = sha256_file(final)
                record["onnx_sha256"] = sha256_file(export)
            record.update(status="passed", device_name=torch.cuda.get_device_name(device), runtime=worker_runtime)
            if panel_recipes is not None:
                from .capacity_panel import prepare_terminal_panel, run_post_capacity_panel
                record["post_measurement_panel_requested"] = True
                # All measured timing, memory and collective normalization audits
                # are final before the disposable panel. No NCCL operation waits
                # for rank-zero compilation, physical recording or CPU auditing.
                prepare_terminal_panel(runner, destination, record)
                record["distributed_group_destroyed_before_panel"] = True
                if rank == 0:
                    try:
                        record["post_measurement_panel"] = run_post_capacity_panel(runner, bundle, admission,
                            measured_contract, config, record, Path(panel_recipes), destination/"post_panel")
                    except Exception as error:
                        record["post_measurement_panel"] = {"status": "failed",
                            "exception": {"type": type(error).__name__, "message": str(error)},
                            "capacity_phase_already_completed": True, "production_acceptance": False}
        caller.wait_stream(stream)
    except Exception as error:
        record.update(status="failed", exception={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if timer is not None:
            timer.restore()
        try:
            if attempt_sink is not None:
                attempt_sink.flush()
                record["terminal_receipts_final"] = attempt_sink.report()
            if env is not None:
                env.close()
        except Exception as error:
            record.update(status="failed", close_exception={"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            write_json(destination/"measurement.json", record)
            os.chdir(previous_directory)


def run_capacity(bundle: Path, output: Path, config: CapacityConfig = CapacityConfig(), *,
                 engineering_admission: Path | None = None, observer_report: Path | None = None,
                 bridge_report: Path | None = None, panel_recipes: Path | None = None,
                 training_initialization_report: Path | None = None) -> dict:
    """Run only inside an existing allocation; preserve failures and never submit."""
    from .runtime.bundle import load_runtime_bundle
    checked = load_runtime_bundle(bundle.resolve(strict=True))
    admission = None
    if engineering_admission is None:
        checked.require_training_qualification()
    else:
        from .capacity_evidence import load_engineering_admission
        admission = load_engineering_admission(engineering_admission)
    if panel_recipes is not None:
        if admission is None:
            raise ValueError("Post-capacity engineering panel requires explicit engineering admission")
        from .engineering_cycle_panel import build_engineering_recipes, engineering_recipe_bytes
        panel_recipes = Path(panel_recipes).resolve(strict=True)
        if panel_recipes.read_bytes() != engineering_recipe_bytes(build_engineering_recipes(checked, episodes_per_family=16)):
            raise ValueError("Capacity panel requires its predeclared exact 64 canonical recipes")
    if config.engineering_recipe != LEGACY_RECIPE_ID and panel_recipes is None:
        raise ValueError("Revised capacity requires its separate strict 64-episode panel")
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise RuntimeError("Capacity runtime differs from the Task 10 candidate pins")
    contract = build_measurement_contract(checked, scale=config.scale, conditioning=config.conditioning,
        engineering_admission=admission, observer_report=observer_report, bridge_report=bridge_report,
            engineering_recipe=config.engineering_recipe, training_initialization_report=training_initialization_report)
    _require_allocation(config)
    destination = writable_path(output)
    destination.mkdir(parents=True, exist_ok=False)
    _local_caches(destination)
    manifest_hash = sha256_file(checked.root/"manifest.json")
    write_json(destination/"launch.json", {
        "schema": "human_wbc_capacity_launch_v1", "config": asdict(config),
        "scale": config.scale.report(config.warmup_iterations+config.measurement_iterations),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "runtime": runtime,
        "reference_manifest_sha256": manifest_hash, "production_checkpoint": False,
        "measurement_contract": contract,
        "learning_scope": contract["payload"]["reference_and_body"].get("scope", "production_reference_capacity"),
        "engineering_admission": admission.as_dict() if admission is not None else None,
        "capacity_weights_may_initialize_training": False,
        "post_panel_recipe_sha256": sha256_file(panel_recipes) if panel_recipes is not None else None,
    })
    error = None
    try:
        if config.gpu_count == 1:
            _capacity_worker(str(checked.root), asdict(config), str(destination), contract,
                admission.as_dict() if admission is not None else None,
                str(observer_report) if observer_report is not None else None,
                str(bridge_report) if bridge_report is not None else None,
                str(panel_recipes) if panel_recipes is not None else None,
                str(training_initialization_report) if training_initialization_report is not None else None)
        else:
            import torchrunx
            os.environ["TORCHRUNX_LOG_DIR"] = str(destination/"torchrunx")
            torchrunx.Launcher(
                hostnames=["localhost"], workers_per_host=config.gpu_count, backend=None,
                copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY+(
                    "MUJOCO*", "WARP*", "SLURM_JOB_ID", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
                    "TMPDIR", "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH", "PYTHONDONTWRITEBYTECODE"),
            ).run(_capacity_worker, str(checked.root), asdict(config), str(destination), contract,
                admission.as_dict() if admission is not None else None,
                str(observer_report) if observer_report is not None else None,
                str(bridge_report) if bridge_report is not None else None,
                str(panel_recipes) if panel_recipes is not None else None,
                str(training_initialization_report) if training_initialization_report is not None else None)
    except Exception as exception:
        error = exception
    ranks = []
    for rank in range(config.gpu_count):
        path = destination/f"rank_{rank:02d}"/"measurement.json"
        if path.is_file():
            ranks.append(json.loads(path.read_text()))
    report = aggregate_capacity(config, ranks, manifest_hash, contract)
    report["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    report["runtime"] = runtime
    if panel_recipes is not None:
        record_zero = next((row for row in ranks if row.get("rank") == 0), {})
        report["post_measurement_panel"] = record_zero.get("post_measurement_panel", {
            "status": "incomplete", "reason": "No completed rank-zero panel receipt"})
        report["session_status"] = ("passed" if report["status"] == "passed"
            and report["post_measurement_panel"].get("status") == "passed" else "incomplete_or_failed")
    if error is not None:
        report.update(status="failed", exception={"type": type(error).__name__, "message": str(error)})
    write_json(destination/"measurement.json", report)
    if error is not None:
        raise RuntimeError(f"Capacity measurement failed; preserved evidence: {destination}") from error
    if report["status"] != "passed":
        raise RuntimeError(f"Capacity gates failed; inspect {destination/'measurement.json'}")
    if panel_recipes is not None and report["session_status"] != "passed":
        raise RuntimeError(f"Capacity completed but its separate panel failed; inspect {destination/'measurement.json'}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-environments", type=int, default=256)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10010)
    parser.add_argument("--conditioning", choices=tuple(item.value for item in Conditioning),
                        default=Conditioning.COMBINED.value)
    parser.add_argument("--engineering-admission", type=Path)
    parser.add_argument("--observer-report", type=Path)
    parser.add_argument("--bridge-report", type=Path)
    parser.add_argument("--panel-recipes", type=Path)
    parser.add_argument("--engineering-recipe", choices=RECIPE_IDS, default=LEGACY_RECIPE_ID)
    parser.add_argument("--training-initialization-report", type=Path)
    args = parser.parse_args(argv)
    config = CapacityConfig(args.global_environments, args.gpus, args.warmup_iterations,
                            args.measurement_iterations, args.seed, args.conditioning, args.engineering_recipe)
    report = run_capacity(args.bundle, args.output, config,
        engineering_admission=args.engineering_admission, observer_report=args.observer_report,
        bridge_report=args.bridge_report, panel_recipes=args.panel_recipes,
        training_initialization_report=args.training_initialization_report)
    print(json.dumps({k: report[k] for k in ("status", "transitions_per_second", "vram_headroom_fraction")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
