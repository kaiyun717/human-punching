"""CPU orchestration fixtures only; these tests are not training evidence."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from human_punching.wbc.validation import (
    PanelExpectation, collective_agreement, digest, isolated_validation_actor,
    iteration_chunks, require_bounded_smoke_without_validation,
    run_validated_chunks, validation_decision, verify_physical_panel, validate_placed_request,
    validate_trace_identity,
)
from human_punching.wbc.provenance import sha256_file


def test_chunks_stop_at_absolute_250_boundaries_without_repeating_indices():
    assert iteration_chunks(0, 600) == (250, 250, 100)
    assert iteration_chunks(249, 252) == (1, 250, 1)
    assert iteration_chunks(500, 250) == (250,)
    with pytest.raises(ValueError):
        iteration_chunks(True, 10)


def test_production_cannot_silently_bypass_missing_physical_runner():
    for settings, count in [({"stage": "canonical"}, 100),
        ({"stage": "canonical", "validation_mode": "bounded_smoke_without_validation"}, 250),
        ({"stage": "domain", "validation_mode": "bounded_smoke_without_validation"}, 20),
        ({"stage": "canonical", "validation_mode": "bounded_smoke_without_validation", "resume": {"path": "old"}}, 20)]:
        with pytest.raises(RuntimeError, match="physical whole-cycle"):
            require_bounded_smoke_without_validation(settings, count)
    require_bounded_smoke_without_validation(
        {"stage": "canonical", "validation_mode": "bounded_smoke_without_validation"}, 24)


def panel(step, rate=.8, stage="canonical", timing=.1, target=.1, recovery=2.):
    # Unit fixture marked explicitly; not written as native evidence.
    return {"unit_test_fixture": True, "native_trace_evidence_verified": True,
            "expectation": {"completed_iterations": step, "stage": stage, "training_seed": 1,
                            "checkpoint_contract_sha256": "a"*64, "reference_manifest_sha256": "b"*64,
                            "recipe_manifest_sha256": "c"*64, "domain_version": "fixture", "episodes_per_family": 256},
            "family_success": {f: rate for f in ("jab", "cross", "left_hook", "right_hook")},
            "worst_family_success": rate, "worst_cell_success": rate,
            "timing_error": timing, "targeting_error": target, "recovery_time": recovery}


def test_stopping_requires_each_family_and_three_final_stage_panels():
    p = panel(250, .95)
    p["family_success"]["left_hook"] = .89
    p["worst_family_success"] = .89
    assert not validation_decision([p], "canonical")["stop_learning"]
    assert validation_decision([panel(250, .90)], "canonical")["stage_ready_for_review"]
    stages = [panel(250*i, .95, "guard_start") for i in (1, 2, 3)]
    assert not validation_decision(stages[:2], "guard_start")["stop_learning"]
    answer = validation_decision(stages, "guard_start")
    assert answer["stage_ready_for_review"] and not answer["automatic_stage_transition"]


def test_plateau_retains_metric_improvement_and_rejects_unverified_panels():
    rows = [panel(i*250) for i in range(1, 5)]
    assert validation_decision(rows, "canonical")["reason"] == "validation_plateau"
    rows[-1]["timing_error"] = .09
    assert not validation_decision(rows, "canonical")["stop_learning"]
    rows[-1]["native_trace_evidence_verified"] = False
    with pytest.raises(ValueError, match="unverified"):
        validation_decision(rows, "canonical")
    with pytest.raises(ValueError, match="increasing"):
        validation_decision([panel(250), panel(250)], "canonical")


class CopyableActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Linear(3, 2)
        self.obs_normalizer = torch.nn.Linear(3, 3, bias=False)
    def as_jit(self):
        return deepcopy(self)
    def forward(self, x):
        return self.mlp(self.obs_normalizer(x))


def test_actor_and_rng_isolation_restores_rng_without_switching_live_mode():
    live = CopyableActor().train()
    runner = SimpleNamespace(device="cpu", alg=SimpleNamespace(get_policy=lambda: live))
    torch.manual_seed(7); np.random.seed(8); random.seed(9)
    torch_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
    before = {key: value.clone() for key, value in live.state_dict().items()}
    with isolated_validation_actor(runner) as (actor, identity):
        assert actor is not live and not actor.training and live.training
        assert len(identity["normalization_sha256"]) == 64
        torch.rand(4); np.random.rand(4); random.random()
        actor(torch.ones(1, 3))
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert random.getstate() == python_rng
    assert all(torch.equal(value, before[key]) for key, value in live.state_dict().items())


def test_actor_modification_fails_and_rng_still_restores():
    live = CopyableActor()
    runner = SimpleNamespace(device="cpu", alg=SimpleNamespace(get_policy=lambda: live))
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="frozen actor"):
        with isolated_validation_actor(runner) as (actor, _):
            actor.mlp.weight.add_(1.)
            torch.rand(1)
    assert torch.equal(torch.get_rng_state(), before)


def test_frozen_actor_isolation_accepts_an_actual_torchscript_module():
    live = CopyableActor()
    scripted = torch.jit.trace(deepcopy(live), torch.zeros(1, 3))
    live.as_jit = lambda: scripted
    runner = SimpleNamespace(device="cpu", alg=SimpleNamespace(get_policy=lambda: live))
    with isolated_validation_actor(runner) as (actor, identity):
        assert isinstance(actor, torch.jit.ScriptModule)
        assert all(not parameter.requires_grad for parameter in actor.parameters())
        assert identity["normalization_sha256"]


def test_orchestration_uses_inherited_loop_then_validation_then_collective_save():
    order = []
    writer = SimpleNamespace(flush=lambda: order.append("flush"), close=lambda: order.append("close"))
    runner = SimpleNamespace(current_learning_iteration=0, is_distributed=False,
                             logger=SimpleNamespace(writer=writer))
    def learn(num_learning_iterations, init_at_random_ep_len):
        assert not init_at_random_ep_len
        order.append(("learn", runner.current_learning_iteration, num_learning_iterations))
        runner.current_learning_iteration += num_learning_iterations-1
    def validate(completed):
        order.append(("validation", completed))
        return {"stop_learning": completed >= 500, "verified_panel_sha256": "a"*64}
    def save(completed, decision):
        order.append(("save", completed))
        assert runner.current_learning_iteration == completed-1
    result = run_validated_chunks(runner, requested_iterations=1000, completed_iterations=0,
        inherited_learn=learn, validate_boundary=validate, checkpoint_boundary=save)
    assert result["new_iterations"] == 500 and result["stopped_early"]
    assert order == [("learn", 0, 250), "flush", "close", ("validation", 250), ("save", 250),
                     ("learn", 250, 250), "flush", "close", ("validation", 500), ("save", 500)]


def test_no_callback_cannot_start_learning():
    with pytest.raises(ValueError, match="physical validation"):
        run_validated_chunks(SimpleNamespace(), requested_iterations=1000, completed_iterations=0,
            inherited_learn=lambda **kwargs: pytest.fail("learning started"),
            validate_boundary=None, checkpoint_boundary=lambda *args: None)


def test_panel_verifier_rejects_synthetic_claim_before_loading_episodes(tmp_path):
    recipe = tmp_path / "recipes.json"
    recipe.write_text(json.dumps({"schema": "human_wbc_validation_recipes_v1", "split": "validation",
                                  "domain_version": "test", "recipes": []}))
    expected = PanelExpectation(250, "canonical", 1, *(["a"*64]*4), sha256_file(recipe), "test", 1)
    report = tmp_path / "panel.json"
    report.write_text(json.dumps({"schema": "human_wbc_native_validation_panel_v1", "native_execution": False,
        "slurm_job_id": "fixture-not-a-job", "expectation": asdict(expected),
        "training_environment_mutated": False, "normalization_updated": False, "episodes": []}))
    with pytest.raises(ValueError, match="native physical"):
        verify_physical_panel(report, recipe, expected)


def test_guard_consolidation_cannot_get_three_panels_by_one_iteration_sessions():
    rows = [panel(step, .95, "guard_start") for step in (250, 251, 252)]
    assert not validation_decision(rows, "guard_start")["stage_ready_for_review"]


def test_recipe_identity_allows_actual_launch_time_and_scene_translation_only():
    from human_punching.wbc.contracts import LaunchSnapshot, PunchRequest
    from dataclasses import replace
    q = np.zeros(36); q[3] = 1.
    source = PunchRequest("jab", .7, LaunchSnapshot("test", q, np.zeros(35),
        [1, 0, 1.2], [.1, 0, 0], [0, 0, 0], [1, 0, 0, 0]), "guard")
    shift = np.array([5., -3., 0.]); q[:3] += shift
    moved = replace(source.snapshot, g1_qpos=q, head_position_w=np.array([1,0,1.2])+shift,
                    human_origin_w=shift)
    actual = replace(source, snapshot=moved, launch_time_s=2.325)
    row = {"source_request": asdict(source), "request": asdict(actual), "scene_origin_w": shift.tolist()}
    recipe = {"request_sha256": source.fingerprint, "family": "jab"}
    assert validate_placed_request(row, recipe).fingerprint == actual.fingerprint
    row["request"]["ttc_s"] = .8
    with pytest.raises(ValueError, match="changes more"):
        validate_placed_request(row, recipe)


def test_failed_validation_never_saves_an_apparent_complete_panel():
    runner = SimpleNamespace(current_learning_iteration=0, is_distributed=False,
                             logger=SimpleNamespace(writer=None))
    def learn(**kwargs):
        runner.current_learning_iteration = 249
    def failed(_):
        raise ValueError("no physical trace")
    with pytest.raises(RuntimeError, match="no physical trace"):
        run_validated_chunks(runner, requested_iterations=250, completed_iterations=0,
            inherited_learn=learn, validate_boundary=failed,
            checkpoint_boundary=lambda *args: pytest.fail("incomplete panel was checkpointed as complete"))


def test_stopping_cannot_mix_seeds_or_changed_validation_recipes():
    for changed in ("training_seed", "recipe_manifest_sha256", "checkpoint_contract_sha256", "domain_version"):
        rows = [panel(step) for step in (250, 500, 750, 1000)]
        rows[-1]["expectation"][changed] = 2 if changed == "training_seed" else "changed"
        with pytest.raises(ValueError, match="unchanged recipes"):
            validation_decision(rows, "canonical")


def test_trace_binding_rejects_wrong_policy_even_when_outer_report_would_match():
    expectation = PanelExpectation(250, "canonical", 1, *(str(i)*64 for i in range(1, 6)), "fixture", 1)
    request = SimpleNamespace(fingerprint="f"*64)
    arrays = {key: np.asarray(value) for key, value in asdict(expectation).items() if key.endswith("sha256")}
    arrays.update(request_sha256=np.asarray(request.fingerprint), domain_version=np.asarray("fixture"))
    validate_trace_identity(arrays, request, expectation)
    arrays["policy_sha256"] = np.asarray("0"*64)
    with pytest.raises(ValueError, match="policy_sha256"):
        validate_trace_identity(arrays, request, expectation)


def test_panel_expectation_rejects_boolean_counts_and_missing_hash():
    values = dict(completed_iterations=250, stage="canonical", training_seed=1, policy_sha256="a"*64,
                  normalization_sha256="b"*64, checkpoint_contract_sha256="c"*64,
                  reference_manifest_sha256="d"*64, recipe_manifest_sha256="e"*64,
                  domain_version="fixture", episodes_per_family=256)
    for key, bad in (("training_seed", True), ("episodes_per_family", 1.5), ("policy_sha256", None)):
        with pytest.raises(ValueError):
            PanelExpectation(**{**values, key: bad})
