"""Native RSL CPU serialization checks; no simulator, rollout or GPU job."""
from copy import deepcopy
from dataclasses import replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from human_punching.wbc.checkpointing import (
    BoundaryProgress, CheckpointContract, CONTINUATION, StageTransition,
    assert_checkpoint_rank_parity, build_checkpoint_contract, capture_rng, load_boundary_checkpoint,
    restore_rng, save_boundary_checkpoint,
)


SAMPLER = {"name": "uniform_family_then_variant", "version": 1}


def make_runner():
    observations = TensorDict({"actor": torch.randn(8, 6)*2+1,
                               "critic": torch.randn(8, 9)*3-1}, batch_size=[8])
    groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = MLPModel(observations, groups, "actor", 3, hidden_dims=(16, 8),
                     obs_normalization=True, distribution_cfg={"class_name": "GaussianDistribution",
                                                               "init_std": .7, "std_type": "scalar"})
    critic = MLPModel(observations, groups, "critic", 1, hidden_dims=(16, 8), obs_normalization=True)
    actor.update_normalization(observations)
    critic.update_normalization(observations)
    storage = RolloutStorage("rl", 8, 24, observations, [3], "cpu")
    algorithm = PPO(actor, critic, storage, learning_rate=.0003, device="cpu")
    # Populate real Adam moments with a deterministic serialization fixture.
    # This is no PPO learning call, simulator rollout or performance result.
    for parameter in list(actor.parameters())+list(critic.parameters()):
        parameter.grad = torch.full_like(parameter, .01)
    algorithm.optimizer.step()
    algorithm.optimizer.zero_grad()
    runner = SimpleNamespace(alg=algorithm, device="cpu", gpu_global_rank=0,
        gpu_world_size=1, is_distributed=False, current_learning_iteration=7,
        logger=SimpleNamespace(tot_timesteps=4096, tot_time=12.5),
        env=SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=192)))
    return runner, observations


def contract_fixture(tmp_path):
    body = tmp_path/"body.xml"
    body.write_text("<mujoco/>")
    reference = tmp_path/"references.json"
    reference.write_text('{"fixture":true}')
    source = tmp_path/"source.py"
    source.write_text("# serialization fixture\n")
    return build_checkpoint_contract(body_path=body, controller={"law": "native_affine"},
        actor_observation_schema={"terms": [{"name": "fixture", "width": 6}], "normalization": True},
        action_schema={"dimension": 3, "clip": 1.}, simulation_contract={"physics_dt_s": .005},
        learner_configuration={"rollout_steps": 24, "epochs": 5, "minibatches": 4},
        reference_manifest=reference, implementation_sources={"fixture": source})


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert left == right


def draws():
    return random.random(), np.random.random(3), torch.randn(3)


def assert_draws_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert torch.equal(left[2], right[2])


def save_fixture(tmp_path):
    runner, observations = make_runner()
    contract = contract_fixture(tmp_path)
    progress = BoundaryProgress(8, "canonical", 8, 2, {"level": 0}, {"best_panel": 1})
    output = tmp_path/"boundary"
    receipt = save_boundary_checkpoint(output, runner, contract=contract, progress=progress,
        sampler_identity=SAMPLER, sampler_state={"family_weights": [1., 1., 1., 1.]})
    return runner, observations, contract, output, receipt


def test_real_normalized_rsl_learner_optimizer_rng_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("Never collect peer-device RNG"))
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    runner, observations, contract, output, receipt = save_fixture(tmp_path)
    expected = deepcopy(runner.alg.save())
    with torch.no_grad():
        expected_action = runner.alg.actor(observations).clone()
    expected_draws = draws()
    restored, _ = make_runner()  # Constructors intentionally consume random values.
    sampler_calls, reset_calls = [], []
    result = load_boundary_checkpoint(output, restored, manifest_sha256=receipt["manifest_sha256"],
        expected_contract=contract, stage="canonical", sampler_identity=SAMPLER,
        restore_sampler=sampler_calls.append, reset_environment=lambda: reset_calls.append(draws()))
    equal_tree(expected, restored.alg.save())
    with torch.no_grad():
        assert torch.equal(expected_action, restored.alg.actor(observations))
    assert any("obs_normalizer" in name for name in expected["actor_state_dict"])
    assert expected["optimizer_state_dict"]["state"]
    assert restored.alg.learning_rate == .0003
    assert restored.current_learning_iteration == 8  # Saved RSL index was 7.
    assert restored.logger.tot_timesteps == 4096
    assert restored.env.unwrapped.common_step_counter == 192
    assert_draws_equal(expected_draws, reset_calls[0])
    assert sampler_calls == [{"family_weights": [1., 1., 1., 1.]}]
    assert result["continuation"] == CONTINUATION and result["physical_state_restored"] is False
    assert not output.with_name("boundary.incomplete").exists()
    with pytest.raises(FileExistsError):
        save_boundary_checkpoint(output, runner, contract=contract,
            progress=BoundaryProgress(8, "canonical", 8), sampler_identity=SAMPLER)


@pytest.mark.parametrize("field", ["body_sha256", "controller_sha256", "actor_observation_schema_sha256",
                                   "action_schema_sha256", "simulation_contract_sha256",
                                   "learner_configuration_sha256", "implementation_sources_sha256",
                                   "reference_manifest_sha256"])
def test_contract_drift_rejected_before_unpickling(tmp_path, monkeypatch, field):
    runner, _, contract, output, receipt = save_fixture(tmp_path)
    changed = replace(contract, **{field: "f"*64})
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Mismatch must fail before unpickling"))
    with pytest.raises(ValueError, match="contract/stage mismatch"):
        load_boundary_checkpoint(output, runner, manifest_sha256=receipt["manifest_sha256"],
            expected_contract=changed, stage="canonical", sampler_identity=SAMPLER,
            reset_environment=lambda: None)


def test_corrupt_rank_bytes_rejected_before_unpickling(tmp_path, monkeypatch):
    runner, _, contract, output, receipt = save_fixture(tmp_path)
    with (output/"rank_00.pt").open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Hash mismatch before torch.load"))
    with pytest.raises(ValueError, match="path/hash mismatch"):
        load_boundary_checkpoint(output, runner, manifest_sha256=receipt["manifest_sha256"],
            expected_contract=contract, stage="canonical", sampler_identity=SAMPLER,
            reset_environment=lambda: None)


def test_named_stage_transition_changes_only_reference_and_resets_stage_counts(tmp_path):
    runner, _, contract, output, receipt = save_fixture(tmp_path)
    new_contract = replace(contract, reference_manifest_sha256="d"*64)
    transition = StageTransition("canonical_to_domain_v1", "canonical", "domain",
                                contract.reference_manifest_sha256, "d"*64, "Qualified expanded target/TTC library")
    calls = []
    result = load_boundary_checkpoint(output, runner, manifest_sha256=receipt["manifest_sha256"],
        expected_contract=new_contract, stage="domain", sampler_identity=SAMPLER,
        transition=transition, transition_curriculum={"target_fraction": .2},
        restore_sampler=calls.append, reset_environment=lambda: None)
    assert result["progress"]["completed_iterations"] == 8
    assert result["progress"]["stage_completed_iterations"] == 0
    assert result["progress"]["stage"] == "domain"
    assert result["progress"]["curriculum"] == {"target_fraction": .2}
    assert result["progress"]["validation"]["selection_history_reset"] is True
    assert result["prior_progress"]["validation"] == {"best_panel": 1}
    assert calls == [{}]
    with pytest.raises(ValueError, match="cannot change body"):
        load_boundary_checkpoint(output, runner, manifest_sha256=receipt["manifest_sha256"],
            expected_contract=replace(new_contract, controller_sha256="e"*64), stage="domain",
            sampler_identity=SAMPLER, transition=transition, reset_environment=lambda: None)


def test_wrong_boundary_or_unfinished_rollout_cannot_be_saved(tmp_path):
    runner, _ = make_runner()
    contract = contract_fixture(tmp_path)
    with pytest.raises(ValueError, match="last iteration index"):
        save_boundary_checkpoint(tmp_path/"wrong", runner, contract=contract,
            progress=BoundaryProgress(7, "canonical", 7), sampler_identity=SAMPLER)
    runner.alg.storage.step = 1
    with pytest.raises(ValueError, match="empty rollout buffer"):
        save_boundary_checkpoint(tmp_path/"unfinished", runner, contract=contract,
            progress=BoundaryProgress(8, "canonical", 8), sampler_identity=SAMPLER)
    assert not (tmp_path/"unfinished.incomplete").exists()


def test_cpu_rng_restore_uses_no_cuda_context(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *a, **kw: pytest.fail("No CUDA context in CPU check"))
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("No peer RNG collection"))
    state = capture_rng("cpu")
    expected = draws()
    restore_rng(state, "cpu")
    assert_draws_equal(expected, draws())


def test_distributed_normalizer_divergence_is_rejected_instead_of_hidden(tmp_path):
    _, _, _, output, _ = save_fixture(tmp_path)
    receipt = json.loads((output/"manifest.json").read_text())["ranks"][0]
    other_rank = {**receipt, "rank": 1}
    assert_checkpoint_rank_parity([receipt, other_rank])
    other_rank["model_buffers_sha256"] = "d"*64
    with pytest.raises(ValueError, match="model_buffers_sha256"):
        assert_checkpoint_rank_parity([receipt, other_rank])
    assert receipt["learned_parameters_sha256"] == other_rank["learned_parameters_sha256"]


def test_source_pin_drift_and_invalid_stage_skip_rejected(tmp_path):
    contract = contract_fixture(tmp_path)
    with pytest.raises(ValueError, match="pinned WBC"):
        replace(contract, source_pins={"wbc_mjlab": "wrong", "mjlab": "wrong"})
    with pytest.raises(ValueError, match="Only canonical-to-domain"):
        StageTransition("skip", "canonical", "guard_start", "a"*64, "b"*64, "invalid skip")
