"""Configuration/source drift checks; no native dynamics or GPU allocation."""
from copy import deepcopy
from dataclasses import dataclass

import pytest

from human_punching.wbc.budget import TrainingScale
from human_punching.wbc.conditioning import Conditioning, apply_conditioning
from human_punching.wbc.measurement_contract import (
    build_measurement_contract, configuration_value, contract_from_parts, digest,
    source_inventory, verify_gpu_hardware, verify_measurement_contract,
)


def contract_parts():
    return dict(bundle_identity={"reference_manifest_sha256": "a"*64, "body_sha256": "b"*64,
                                 "controller_sha256": "c"*64},
        scale=TrainingScale(), conditioning=Conditioning.COMBINED.value,
        environment_configuration={"physics_dt": .005, "solver": "newton"},
        learner_configuration={"actor": [512, 256], "normalization": True, "epochs": 5},
        actor_observation_schema=["joint_pos", "reference", "g1_launch"],
        critic_observation_schema=["joint_pos", "reference", "g1_launch", "privileged"],
        sources={"source_sha256": "d"*64}, runtime={"torch": "2.8.0+cu128"})


@pytest.mark.parametrize("key,changed", [
    ("conditioning", Conditioning.REFERENCE.value),
    ("bundle_identity", {"reference_manifest_sha256": "e"*64}),
    ("environment_configuration", {"physics_dt": .0025, "solver": "newton"}),
    ("learner_configuration", {"actor": [128, 128], "normalization": True, "epochs": 5}),
    ("actor_observation_schema", ["reference", "joint_pos", "g1_launch"]),
    ("sources", {"source_sha256": "f"*64}),
    ("runtime", {"torch": "2.11.0+cu128"}),
    ("scale", TrainingScale(global_environments=4096)),
])
def test_capacity_cannot_qualify_changed_configuration(key, changed):
    parts = contract_parts()
    measured = contract_from_parts(**parts)
    parts[key] = changed
    with pytest.raises(ValueError, match="different source/runtime/configuration"):
        verify_measurement_contract(measured, contract_from_parts(**parts))


def test_contract_payload_tampering_is_detected():
    measured = contract_from_parts(**contract_parts())
    expected = deepcopy(measured)
    measured["payload"]["conditioning"] = Conditioning.REFERENCE.value
    with pytest.raises(ValueError, match="schema/digest mismatch"):
        verify_measurement_contract(measured, expected)


def test_mapping_order_and_callable_identity_are_deterministic():
    @dataclass
    class Fixture:
        terms: dict
        callback: object
    first = configuration_value(Fixture({"a": 1, "b": 2}, configuration_value))
    second = configuration_value(Fixture({"b": 2, "a": 1}, configuration_value))
    assert digest(first) != digest(second)
    assert configuration_value(configuration_value) == {
        "callable": "human_punching.wbc.measurement_contract.configuration_value"}
    with pytest.raises(TypeError, match="Unsupported deterministic"):
        configuration_value(object())


def test_gpu_memory_model_and_cuda_build_must_match():
    hardware = {"name": "NVIDIA L40S", "total_memory_bytes": 45000000000,
                "compute_capability": [8, 9], "multiprocessor_count": 142, "torch_cuda_build": "12.8",
                "nvidia_driver_version": "fixture-driver", "warp_cuda_toolkit_build": [12, 9],
                "warp_cuda_driver_api": [13, 1], "cuda_nvrtc_package_version": "fixture-compiler"}
    verify_gpu_hardware(hardware, dict(hardware))
    for key, value in (("name", "V100"), ("total_memory_bytes", 16000000000), ("torch_cuda_build", "13.0")):
        with pytest.raises(ValueError, match="GPU hardware differs"):
            verify_gpu_hardware(hardware, {**hardware, key: value})


def test_current_imported_upstream_code_matches_source_export():
    pytest.importorskip("wbc_mjlab")
    inventory = source_inventory()
    assert inventory["human_files"]
    assert inventory["human_files_sha256"] == digest(inventory["human_files"])
    assert set(inventory["upstream"]) == {"wbc-mjlab", "mjlab"}


def test_real_config_contract_excludes_budget_seed_paths_but_binds_conditioning(tmp_path, monkeypatch):
    pytest.importorskip("wbc_mjlab")
    from human_punching.wbc import measurement_contract as module
    from human_punching.wbc.references import PROJECT_ROOT, convert_release
    from human_punching.wbc.runtime.bundle import RuntimeBundle, load_runtime_bundle
    from human_punching.wbc.runtime.task import build_human_env_cfg, human_ppo_cfg
    import mujoco

    monkeypatch.setattr(mujoco, "mj_step", lambda *a, **kw: pytest.fail("No dynamics in a contract audit"))
    root = tmp_path/"legacy_fixture"
    convert_release(PROJECT_ROOT/"outputs/four-punch-milestone", root)
    bundle = load_runtime_bundle(root)
    with pytest.raises(ValueError, match="unqualified"):
        build_measurement_contract(bundle, scale=TrainingScale())
    # Configuration-only fixture; no production data are marked qualified and no
    # environment or learner is constructed. Production always uses the gate.
    monkeypatch.setattr(RuntimeBundle, "require_training_qualification", lambda self: None)
    monkeypatch.setattr(module, "source_inventory", lambda: {"fixture": "stable source identity"})
    cfg = build_human_env_cfg(bundle, num_envs=4096, evaluation=False)
    cfg.sim.nan_guard.enabled = True
    agent = human_ppo_cfg()
    first = build_measurement_contract(bundle, scale=TrainingScale(), env_cfg=cfg, agent_cfg=agent)
    agent.seed, cfg.seed = 7, 7
    agent.max_iterations, agent.save_interval = 10000, 100
    cfg.sim.nan_guard.output_dir = str(tmp_path/"different_rank_output")
    second = build_measurement_contract(bundle, scale=TrainingScale(), env_cfg=cfg, agent_cfg=agent)
    verify_measurement_contract(first, second)
    apply_conditioning(cfg, Conditioning.REFERENCE)
    third = build_measurement_contract(bundle, scale=TrainingScale(), env_cfg=cfg, agent_cfg=agent,
                                       conditioning=Conditioning.REFERENCE.value)
    with pytest.raises(ValueError, match="different source/runtime/configuration"):
        verify_measurement_contract(first, third)
