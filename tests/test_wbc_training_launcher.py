import json
from pathlib import Path

import pytest

from human_punching.wbc.provenance import sha256_file
from human_punching.wbc.training import checked_compute_plan
from human_punching.wbc.measurement_contract import digest, SCHEMA


CONTRACT = {"schema": SCHEMA, "payload": {"fixture": "unit-only-not-native-evidence"},
            "sha256": digest({"fixture": "unit-only-not-native-evidence"})}


@pytest.fixture(autouse=True)
def mock_configuration_only(monkeypatch):
    # Exercise plan/measurement matching independently of physical bundle gates.
    monkeypatch.setattr("human_punching.wbc.training.build_measurement_contract",
                        lambda *args, **kwargs: CONTRACT)


def measured_plan(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text('{}\n')
    digest = sha256_file(bundle / "manifest.json")
    measurement = {
        "schema": "human_wbc_capacity_measurement_v1", "status": "passed",
        "reference_manifest_sha256": digest, "gpu_count": 2,
        "global_environments": 8192, "includes_ppo": True,
        "transitions_per_second": 25000., "vram_headroom_fraction": .20,
        "measurement_contract": CONTRACT, "measurement_contract_sha256": CONTRACT["sha256"],
        "gpu_hardware": {"name": "NVIDIA L40S"},
    }
    plan = {
        "schema": "human_wbc_measured_compute_plan_v1", "stage": "canonical",
        "iterations": 100, "gpu_count": 2, "global_environments": 8192,
        "validation_mode": "bounded_smoke_without_validation",
        "reference_manifest_sha256": digest,
    }
    return bundle, plan, measurement


def save_plan(tmp_path, plan, measurement):
    evidence = tmp_path / "capacity.json"
    evidence.write_text(json.dumps(measurement))
    plan["measurement"] = {"path": evidence.name, "sha256": sha256_file(evidence)}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_measured_plan_preserves_global_batch_and_hashes(tmp_path):
    bundle, plan, measurement = measured_plan(tmp_path)
    path = save_plan(tmp_path, plan, measurement)
    _, scale = checked_compute_plan(path, bundle)
    assert scale.environments_per_rank == 4096
    assert scale.fresh_transitions == 196608
    (tmp_path / "capacity.json").write_text('{}')
    with pytest.raises(ValueError, match="intact native"):
        checked_compute_plan(path, bundle)


@pytest.mark.parametrize("key,value", [
    ("includes_ppo", False), ("global_environments", 4096),
    ("gpu_count", 4), ("status", "estimated"),
    ("vram_headroom_fraction", .149), ("transitions_per_second", 0),
    ("measurement_contract_sha256", "0" * 64),
    ("measurement_contract", {}), ("gpu_hardware", {"name": "unmeasured"}),
])
def test_probe_or_incompatible_capacity_cannot_launch_production(tmp_path, key, value):
    bundle, plan, measurement = measured_plan(tmp_path)
    measurement[key] = value
    path = save_plan(tmp_path, plan, measurement)
    with pytest.raises(ValueError):
        checked_compute_plan(path, bundle)


def test_later_stage_cannot_silently_restart_from_scratch(tmp_path):
    bundle, plan, measurement = measured_plan(tmp_path)
    plan["stage"] = "domain"
    path = save_plan(tmp_path, plan, measurement)
    with pytest.raises(RuntimeError, match="physical whole-cycle validation"):
        checked_compute_plan(path, bundle)


def test_production_rejected_before_capacity_or_gpu_construction(tmp_path):
    bundle, plan, measurement = measured_plan(tmp_path)
    plan.pop("validation_mode")
    path = save_plan(tmp_path, plan, measurement)
    (tmp_path / "capacity.json").unlink()
    with pytest.raises(RuntimeError, match="physical whole-cycle validation"):
        checked_compute_plan(path, bundle)
