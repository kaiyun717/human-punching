"""CPU contract/actor/recorder arithmetic checks; no environment dynamics."""
from copy import deepcopy
from dataclasses import replace
import json

import mujoco
import numpy as np
import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from human_punching.wbc.provenance import PROJECT_ROOT, configure_local_caches, sha256_file
from human_punching.wbc.runtime.bundle import load_runtime_bundle, RuntimeBundle
from human_punching.wbc.trainedactor_evaluation import (
    PanelMetrics, actor_identity, audit_actor_clock, build_actor_panel_cfg,
    evaluation_contract, physical_validation_callback, planned_recipes,
    run_actor_panel, verify_actor_contract, verify_compiled_controller,
)
from human_punching.wbc.validation import PanelExpectation


@pytest.fixture(scope="module")
def bundle(task10_common_bundle):
    return task10_common_bundle


@pytest.fixture
def inputs(bundle, tmp_path):
    observation = TensorDict({"actor": torch.arange(48, dtype=torch.float32).reshape(8, 6)}, batch_size=[8])
    model = MLPModel(observation, {"actor": ["actor"]}, "actor", 48,
                     hidden_dims=(16, 8), obs_normalization=True)
    model.update_normalization(observation)
    actor = torch.jit.script(model.as_jit().eval())
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    contract = evaluation_contract(actor, bundle,
        observation_contract={"ordered_terms": [{"name": "fixture", "shape": [6]}]},
        action_contract={"dimension": 48, "joint_names": list(bundle.joint_names)},
        simulation_contract={"physics_dt_s": .005}, checkpoint_contract_sha256="a"*64)
    recipes = {"schema": "human_wbc_validation_recipes_v1", "split": "diagnostic", "domain_version": "unqualified_fixture",
        "recipes": [{"recipe_id": str(i), "reference_id": clip["reference_id"],
                     "family": clip["family"], "request_sha256": clip["request_fingerprint"],
                     "spatial_cell_id": "unqualified", "ttc_cell_id": "unqualified"}
                    for i, clip in enumerate(bundle.clips)]}
    path = tmp_path/"recipes.json"
    path.write_text(json.dumps(recipes))
    expected = PanelExpectation(completed_iterations=1, stage="canonical", training_seed=1,
        **{key:contract[key] for key in ("policy_sha256", "normalization_sha256", "checkpoint_contract_sha256", "reference_manifest_sha256")},
        recipe_manifest_sha256=sha256_file(path), domain_version="unqualified_fixture", episodes_per_family=1)
    return actor, contract, path, expected


def test_frozen_native_normalized_actor_identity_and_output_are_stable(bundle, inputs):
    actor, contract, _, expected = inputs
    verify_actor_contract(actor, bundle, contract, expected)
    before = actor_identity(actor)
    values = torch.randn(5, 6)
    with torch.inference_mode():
        result = actor(values)
    assert result.shape == (5, 48) and torch.isfinite(result).all()
    assert actor_identity(actor) == before
    altered = deepcopy(actor)
    altered.obs_normalizer._mean.add_(.1)
    with pytest.raises(ValueError, match="identity changed"):
        verify_actor_contract(altered, bundle, contract, expected)
    actor.train()
    with pytest.raises(ValueError, match="frozen eval-mode"):
        actor_identity(actor)


def test_actor_contract_rejects_tampering_and_wrong_checkpoint(bundle, inputs):
    actor, contract, _, expected = inputs
    changed = deepcopy(contract)
    changed["action_contract"]["dimension"] = 47
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_actor_contract(actor, bundle, changed, expected)
    with pytest.raises(ValueError, match="checkpoint_contract"):
        verify_actor_contract(actor, bundle, contract, replace(expected, checkpoint_contract_sha256="b"*64))


def test_planned_clips_guard_requests_and_order_are_preserved(bundle, inputs, monkeypatch):
    _, contract, path, expected = inputs
    recipes = planned_recipes(bundle, path, expected)
    assert [r["reference_id"] for r in recipes] == [c["reference_id"] for c in bundle.clips]
    configure_local_caches()
    monkeypatch.setattr(mujoco, "mj_step", lambda *a, **kw: pytest.fail("No dynamics"))
    metrics = PanelMetrics(len(recipes))
    cfg = build_actor_panel_cfg(bundle, recipes, metrics, contract["conditioning"])
    assert cfg.scene.num_envs == 4 and cfg.sim.mujoco.timestep == .005 and cfg.decimation == 4
    assert not cfg.auto_reset and cfg.events == cfg.terminations == cfg.rewards == {}
    assert cfg.commands["motion"].planned_reference_ids == tuple(r["reference_id"] for r in recipes)
    assert not cfg.commands["motion"].assistive_wrench_enabled
    assert metrics.native.num_envs == 4


@pytest.mark.parametrize("key,value", [("reference_id", "missing"), ("request_sha256", "f"*64),
                                      ("family", "wrong"), ("spatial_cell_id", "")])
def test_changed_recipe_cannot_be_executed(bundle, inputs, key, value):
    _, _, path, expected = inputs
    data = json.loads(path.read_text())
    data["recipes"][0][key] = value
    path.write_text(json.dumps(data))
    expected = replace(expected, recipe_manifest_sha256=sha256_file(path))
    with pytest.raises(ValueError):
        planned_recipes(bundle, path, expected)


def test_production_remains_blocked_even_if_reference_gate_is_satisfied(bundle, inputs, tmp_path, monkeypatch):
    actor, contract, path, expected = inputs
    data = json.loads(path.read_text()); data["split"] = "validation"
    path.write_text(json.dumps(data)); expected = replace(expected, recipe_manifest_sha256=sha256_file(path))
    with pytest.raises(ValueError, match="unqualified"):
        planned_recipes(bundle, path, expected, production=True)
    # Exercise the separate readiness/proof gate only; no artifact is qualified.
    monkeypatch.setattr(RuntimeBundle, "require_training_qualification", lambda _: None)
    import human_punching.wbc.trainedactor_evaluation as module
    monkeypatch.setattr(module, "_require_allocated_cuda", lambda: pytest.fail("Must fail before CUDA"))
    with pytest.raises(NotImplementedError, match="guard readiness.*physical settling"):
        run_actor_panel(bundle, actor, contract, path, tmp_path/"blocked", expectation=expected, production=True)
    callback = physical_validation_callback(bundle, contract, path, lambda _: tmp_path/"callback")
    with pytest.raises(NotImplementedError, match="failed-readiness receipts"):
        callback(actor, expected)
    assert not (tmp_path/"blocked").exists() and not (tmp_path/"callback").exists()


def native_fixture():
    count, envs = 1201, 2
    q = np.zeros((count, envs, 55)); q[..., 2:4] = [.9, 1.]
    actions = np.zeros((300, envs, 48)); actions[::2, :, 0] = .6; actions[1::2, :, 1] = -.8
    native = {"time": np.broadcast_to((np.arange(count)*.005)[:, None], (count, envs)),
        "qpos": q, "qvel": np.zeros((count, envs, 54)), "ctrl": np.zeros((count, envs, 48)),
        "actuator_force": np.zeros((count, envs, 48)), "qfrc_actuator": np.zeros((count, envs, 54)),
        "qfrc_applied": np.zeros((count, envs, 54)), "xfrc_applied": np.zeros((count, envs, 24, 6)),
        "overflow": np.zeros((count, envs), int),
        "reference_physics_steps": np.broadcast_to(np.arange(count)[:, None], (count, envs)),
        "reset_counts": np.ones((count, envs), int), "sim_step_index": np.arange(count),
        "policy_step_index": np.maximum(np.arange(count)-1, 0)//4,
        "raw_action": np.concatenate((np.zeros((1, envs, 48)), np.repeat(actions, 4, axis=0))),
        "trajectory_id": np.broadcast_to(np.arange(envs), (count, envs)),
        "joint_position_target": np.zeros((count, envs, 48)),
        "joint_velocity_target": np.zeros((count, envs, 48)), "joint_effort_target": np.zeros((count, envs, 48))}
    return native, actions


def test_nonzero_actor_actions_must_hold_exactly_four_physics_ticks():
    native, actions = native_fixture()
    result = audit_actor_clock(native, actions, effort_limits=np.full(48, 20.))
    assert result["status"] == "passed"
    assert "zero_learned_residual" not in result["checks"]
    native["raw_action"][2, 0, 0] += .01
    result = audit_actor_clock(native, actions, effort_limits=np.full(48, 20.))
    assert not result["checks"]["actor_action_held_for_four_physics_steps"]
    assert result["status"] == "failed"


@pytest.mark.parametrize("field", ["qfrc_applied", "qfrc_actuator", "xfrc_applied", "reset_counts"])
def test_actor_clock_does_not_remove_force_or_reset_gates(field):
    native, actions = native_fixture()
    native[field][1].flat[0] += 1
    assert audit_actor_clock(native, actions, effort_limits=np.full(48, 20.))["status"] == "failed"


@pytest.mark.parametrize("field,index,value", [
    ("actuator_gear", (0, 0), 2.),
    ("actuator_gainprm", (0, 0), 2.),
    ("actuator_biasprm", (0, 0), .1),
    ("actuator_trnid", (0, 0), 0),
    ("actuator_biastype", 0, mujoco.mjtBias.mjBIAS_NONE),
    ("actuator_ctrllimited", 0, 1),
    ("actuator_forcerange", (0, 1), 999.),
])
def test_compiled_controller_rejects_changed_force_law_and_base_transmission(bundle, monkeypatch, field, index, value):
    from human_punching.wbc.runtime.body import build_human_entity_cfg
    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("No dynamics"))
    model = build_human_entity_cfg(bundle).build().spec.compile()
    verify_compiled_controller(model, bundle)
    getattr(model, field)[index] = value
    with pytest.raises(ValueError, match="controller differs"):
        verify_compiled_controller(model, bundle)
