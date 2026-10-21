"""Capacity accounting/gating only: these fixtures are not native evidence."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from human_punching.wbc.capacity import CapacityConfig, aggregate_capacity, run_capacity, _require_allocation
from human_punching.wbc.measurement_contract import contract_from_parts, digest
from human_punching.wbc.provenance import sha256_file
from human_punching.wbc.training import checked_compute_plan


def contract_fixture(config, manifest="a"*64):
    return contract_from_parts(bundle_identity={"reference_manifest_sha256": manifest},
        scale=config.scale, conditioning=config.conditioning, environment_configuration={"fixture": True},
        learner_configuration={"fixture": True}, actor_observation_schema=["fixture_actor"],
        critic_observation_schema=["fixture_critic"], sources={"fixture": True}, runtime={"fixture": True})


def rank_fixture(config: CapacityConfig, rank: int, manifest_digest: str = "a"*64) -> dict:
    """Synthetic records for arithmetic tests; never written as real run output."""
    return {
        "rank": rank, "gpu_world_size": config.gpu_count, "status": "passed",
        "native_execution": True, "reference_manifest_sha256": manifest_digest,
        "environments_per_rank": config.scale.environments_per_rank,
        "actual_environments_per_rank": config.scale.environments_per_rank,
        "conditioning": config.conditioning,
        "measurement_contract_sha256": contract_fixture(config, manifest_digest)["sha256"],
        "native_observation_schema": {"actor_width": 400, "critic_width": 800},
        "native_observation_schema_sha256": digest({"actor_width": 400, "critic_width": 800}),
        "gpu_hardware": {"name": "NVIDIA L40S", "total_memory_bytes": 1000,
                         "compute_capability": [8, 9], "multiprocessor_count": 142, "torch_cuda_build": "12.8",
                         "nvidia_driver_version": "fixture-driver", "warp_cuda_toolkit_build": [12, 9],
                         "warp_cuda_driver_api": [13, 1], "cuda_nvrtc_package_version": "fixture-compiler"},
        "warmup_iterations": config.warmup_iterations,
        "measurement_iterations": config.measurement_iterations,
        "measured_policy_steps": config.measurement_iterations*24,
        "measured_physics_steps": config.measurement_iterations*96,
        "measured_fresh_transitions": config.measurement_iterations*config.scale.transitions_per_rank,
        "measured_optimizer_steps": config.measurement_iterations*20,
        "measurement_wall_time_s": 20.+rank,
        "measured_ppo_update_wall_time_s": 4.,
        "initialization_wall_time_s": 10., "warmup_wall_time_s": 15.,
        "finite_learning": True, "checkpoint_export_verified": rank == 0,
        "total_vram_bytes": 1000, "conservative_peak_used_bytes": 800,
    }


@pytest.mark.parametrize("gpu_count,per_rank", [(1, 8192), (2, 4096), (4, 2048)])
def test_global_batch_and_slowest_rank_throughput(gpu_count, per_rank):
    config = CapacityConfig(global_environments=8192, gpu_count=gpu_count)
    ranks = [rank_fixture(config, rank) for rank in range(gpu_count)]
    report = aggregate_capacity(config, ranks, "a"*64, contract_fixture(config))
    assert report["status"] == "passed"
    assert report["environments_per_rank"] == per_rank
    assert report["fresh_transitions_per_iteration"] == 196608
    assert report["measured_fresh_transitions"] == 196608*3
    assert report["transitions_per_second"] == pytest.approx(196608*3/(19+gpu_count))
    assert report["configured_optimizer_steps_per_iteration"] == 20
    assert report["vram_headroom_fraction"] == pytest.approx(.20)


@pytest.mark.parametrize("changes", [
    {"gpu_count": 3}, {"gpu_count": True}, {"global_environments": 128},
    {"global_environments": 16384}, {"global_environments": 257, "gpu_count": 2},
    {"warmup_iterations": 0}, {"warmup_iterations": 21},
    {"measurement_iterations": 0}, {"measurement_iterations": 21}, {"seed": -1},
    {"conditioning": "unknown"},
])
def test_bounds_and_integer_gates(changes):
    with pytest.raises(ValueError):
        replace(CapacityConfig(), **changes)


@pytest.mark.parametrize("key,value,failed_check", [
    ("native_execution", False, "native_rank_execution"),
    ("reference_manifest_sha256", "wrong", "matching_reference"),
    ("measured_policy_steps", 24, "actual_rollout_counts"),
    ("measured_physics_steps", 72, "actual_rollout_counts"),
    ("measured_optimizer_steps", 19, "actual_optimizer_counts"),
    ("measurement_wall_time_s", float("nan"), "finite_timing"),
    ("measured_ppo_update_wall_time_s", 0., "finite_timing"),
    ("finite_learning", False, "finite_learning"),
    ("checkpoint_export_verified", False, "checkpoint_and_export"),
    ("conservative_peak_used_bytes", 851, "fifteen_percent_vram_headroom"),
    ("total_vram_bytes", 0, "valid_memory"),
    ("conditioning", "reference_only", "matching_conditioning"),
    ("actual_environments_per_rank", 8192, "matching_dimensions"),
    ("measurement_contract_sha256", "changed", "matching_measurement_contract"),
    ("native_observation_schema_sha256", "changed", "native_observation_schema"),
    ("gpu_hardware", {"name": "V100", "total_memory_bytes": 1000}, "matching_l40s_hardware"),
])
def test_missing_or_failed_native_checks_cannot_pass(key, value, failed_check):
    config = CapacityConfig()
    row = rank_fixture(config, 0)
    row[key] = value
    report = aggregate_capacity(config, [row], "a"*64, contract_fixture(config))
    assert report["status"] == "failed"
    assert report["checks"][failed_check] is False


def test_missing_duplicate_and_empty_ranks_fail_closed():
    config = CapacityConfig(gpu_count=2)
    for ranks in ([], [rank_fixture(config, 0)], [rank_fixture(config, 0)]*2):
        report = aggregate_capacity(config, ranks, "a"*64, contract_fixture(config))
        assert report["status"] == "failed"
        assert report["transitions_per_second"] is None
        assert report["checks"]["all_ranks_present"] is False


def test_full_scale_schema_is_compatible_with_compute_plan(tmp_path, monkeypatch):
    bundle = tmp_path/"bundle"
    bundle.mkdir()
    (bundle/"manifest.json").write_text("{}")
    digest = sha256_file(bundle/"manifest.json")
    config = CapacityConfig(global_environments=8192, gpu_count=2)
    evidence = aggregate_capacity(config, [rank_fixture(config, i, digest) for i in range(2)],
                                  digest, contract_fixture(config, digest))
    monkeypatch.setattr("human_punching.wbc.training.build_measurement_contract",
                        lambda *a, **kw: evidence["measurement_contract"])
    measurement = tmp_path/"synthetic_accounting_fixture.json"
    measurement.write_text(json.dumps(evidence))
    plan = {
        "schema": "human_wbc_measured_compute_plan_v1", "gpu_count": 2,
        "stage": "canonical", "iterations": 100, "global_environments": 8192,
        "validation_mode": "bounded_smoke_without_validation",
        "reference_manifest_sha256": digest,
        "measurement": {"path": measurement.name, "sha256": sha256_file(measurement)},
    }
    path = tmp_path/"synthetic_plan_fixture.json"
    path.write_text(json.dumps(plan))
    _, scale = checked_compute_plan(path, bundle)
    assert scale.fresh_transitions == 196608


def test_qualification_is_checked_before_allocation_or_output(tmp_path, monkeypatch):
    from human_punching.wbc.runtime import bundle as runtime_bundle

    class Unqualified:
        def require_training_qualification(self):
            raise ValueError("No physically screened reference bundle")

    monkeypatch.setattr(runtime_bundle, "load_runtime_bundle", lambda path: Unqualified())
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    destination = tmp_path/"must_not_exist"
    with pytest.raises(ValueError, match="physically screened"):
        run_capacity(tmp_path, destination)
    assert not destination.exists()


def test_no_allocation_rejected_before_torch_or_dynamics(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="allocated Slurm"):
        _require_allocation(CapacityConfig())
