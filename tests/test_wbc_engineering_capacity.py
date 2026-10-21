"""Capacity configuration/normalization checks; no allocation or native dynamics."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest
import torch

from human_punching.wbc import capacity_evidence, engineering_admission, measurement_contract
from human_punching.wbc.budget import TrainingScale
from human_punching.wbc.capacity import CapacityConfig, _local_caches
from human_punching.wbc.capacity_evidence import (
    configure_engineering_capacity, distributed_normalization_evidence,
    require_engineering_configuration, fresh_imitation_health_matches, engineering_train_config,
)
from human_punching.wbc.engineering_admission import EngineeringAdmission, SCHEMA, SCOPE
from human_punching.wbc.measurement_contract import build_measurement_contract, digest, verify_measurement_contract
from human_punching.wbc.provenance import PROJECT_ROOT
from human_punching.wbc.references import convert_release
from human_punching.wbc.runtime.engineering_recipe import LEGACY_RECIPE, REVISED_RECIPE


def admission_fixture():
    payload = {"schema": SCHEMA, "scope": SCOPE, "qualified_for_training": False,
               "target_ttc_domain_qualified": False, "synthetic_configuration_fixture": True}
    return EngineeringAdmission.from_dict({**payload, "sha256": digest(payload)})


def test_native_observer_payloads_bind_compiled_model_for_reaudit(tmp_path, monkeypatch):
    from human_punching.wbc.provenance import sha256_file
    monkeypatch.setattr(engineering_admission, "PROJECT_ROOT", tmp_path)
    directory=tmp_path/"guard_b";directory.mkdir()
    arrays=directory/"native_trace.npz";arrays.write_bytes(b"synthetic trace fixture")
    model=directory/"compiled_scene.mjb";model.write_bytes(b"synthetic compiled fixture")
    row={"hashes":{"native_trace":sha256_file(arrays)},"compiled_scene_sha256":sha256_file(model)}
    result=capacity_evidence.observer_run_payloads(directory,row)
    assert set(result)=={"native_trace","compiled_scene"}
    assert result["compiled_scene"]=={"path":"guard_b/compiled_scene.mjb","sha256":row["compiled_scene_sha256"]}
    model.write_bytes(b"changed compiled model")
    with pytest.raises(ValueError,match="compiled scene hash"):
        capacity_evidence.observer_run_payloads(directory,row)
    model.unlink()
    with pytest.raises(FileNotFoundError):
        capacity_evidence.observer_run_payloads(directory,row)


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("engineering_capacity")/"references"
    convert_release(PROJECT_ROOT/"outputs/four-punch-milestone", root)
    # Deliberately synthetic command metadata for configuration-only tests;
    # historical fixed-target clips do not constitute commanded TTC evidence.
    path = root/"manifest.json"
    manifest = json.loads(path.read_text())
    for row in manifest["clips"]:
        row["requested_ttc_s"] = 1.2
    path.write_text(json.dumps(manifest))
    return root


def test_real_engineering_config_preserves_ppo_and_explicit_reset_at_production_scale(converted, monkeypatch):
    from human_punching.wbc.runtime.task import human_ppo_cfg
    from human_punching.wbc.runtime.bundle import load_runtime_bundle
    from mjlab.scene import Scene
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, lambda *a, **kw: pytest.fail("No dynamics in capacity config test"))
    # Configuration-only fixture; synthetic admission never becomes saved or
    # executable native evidence. The production gate remains intact.
    monkeypatch.setattr(engineering_admission, "verify_engineering_admission", lambda *args: None)
    monkeypatch.setattr(capacity_evidence, "observer_capacity_evidence", lambda *args: {"synthetic_fixture": True})
    monkeypatch.setattr(capacity_evidence, "bridge_capacity_evidence", lambda *args: {"synthetic_fixture": True})
    monkeypatch.setattr(measurement_contract, "source_inventory", lambda: {"fixture": "stable"})
    admission = admission_fixture()
    scale = CapacityConfig(global_environments=8192, gpu_count=2).scale
    cfg = configure_engineering_capacity(converted, admission, num_envs=scale.environments_per_rank)
    require_engineering_configuration(cfg)
    assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == 4096
    assert cfg.auto_reset and cfg.sim.nan_guard.enabled and not cfg.events
    assert cfg.sim.mujoco.timestep == .005 and cfg.decimation == 4
    assert cfg.commands["motion"].rsi.sampling_mode == "start"
    assert cfg.metrics["human_native_substep_valid"].per_substep
    from human_punching.wbc.telemetry.cycle_runtime import cycle_condition
    assert cfg.observations["actor"].terms["punch_condition"].func is cycle_condition
    assert "cycle_complete" in cfg.terminations and "timed_target" not in cfg.rewards
    ppo = human_ppo_cfg()
    assert (ppo.num_steps_per_env, ppo.algorithm.num_learning_epochs, ppo.algorithm.num_mini_batches) == (24, 5, 4)
    assert ppo.actor.obs_normalization and ppo.critic.obs_normalization
    first = build_measurement_contract(converted, scale=scale, env_cfg=cfg, agent_cfg=ppo,
        engineering_admission=admission, observer_report=Path("synthetic"), bridge_report=Path("synthetic"))
    scene = Scene(cfg.scene, device="cpu")
    scene.compile()
    second = build_measurement_contract(converted, scale=scale, env_cfg=cfg, agent_cfg=ppo,
        engineering_admission=admission, observer_report=Path("synthetic"), bridge_report=Path("synthetic"))
    verify_measurement_contract(first, second)
    identity = first["payload"]["reference_and_body"]
    assert identity["scope"] == SCOPE and identity["production_reference_qualified"] is False
    assert identity["engineering_admission_sha256"] == admission.sha256
    with pytest.raises(ValueError, match="unqualified"):
        load_runtime_bundle(converted).require_training_qualification()
    with pytest.raises(ValueError, match="unqualified"):
        build_measurement_contract(converted, scale=scale)
    cfg.auto_reset = False
    with pytest.raises(ValueError, match="explicit unassisted"):
        require_engineering_configuration(cfg)
    cfg.auto_reset = True
    from human_punching.wbc.conditioning import apply_conditioning, Conditioning
    apply_conditioning(cfg, Conditioning.COMBINED)
    with pytest.raises(ValueError, match="cycle bridge"):
        require_engineering_configuration(cfg)
    with pytest.raises(ValueError, match="phase-aware"):
        configure_engineering_capacity(converted, admission, num_envs=4096, conditioning=Conditioning.REFERENCE)


def test_engineering_contract_cannot_omit_native_observer_evidence(converted):
    with pytest.raises(ValueError, match="completed native observer"):
        build_measurement_contract(converted, scale=TrainingScale(), engineering_admission=admission_fixture())
    with pytest.raises(ValueError, match="completed native bridge"):
        build_measurement_contract(converted, scale=TrainingScale(), engineering_admission=admission_fixture(),
            observer_report=Path("synthetic"))


def test_worker_adapter_replaces_actual_frozen_train_config_and_keeps_agent(converted, monkeypatch):
    from dataclasses import FrozenInstanceError
    from mjlab.scripts.train import TrainConfig
    from human_punching.wbc.runtime.task import build_human_env_cfg, human_ppo_cfg
    monkeypatch.setattr(engineering_admission, "verify_engineering_admission", lambda *args: None)
    original_env = build_human_env_cfg(converted, num_envs=4096, evaluation=True)
    original = TrainConfig(env=original_env, agent=human_ppo_cfg(), video=True, log_root="retained_fixture")
    with pytest.raises(FrozenInstanceError):
        original.env = original_env
    updated = engineering_train_config(original, converted, admission_fixture(), num_envs=4096)
    assert type(updated) is TrainConfig and updated is not original
    assert updated.env is not original.env and original.env is original_env
    assert updated.agent is original.agent and updated.video and updated.log_root == "retained_fixture"
    require_engineering_configuration(updated.env)
    assert updated.env.scene.num_envs == updated.env.scene.terrain.num_envs == 4096
    # Nested runner settings are mutable in the accepted pinned API, unlike the
    # launcher envelope. The actual workers still set their per-rank seed here.
    updated.agent.seed = 101
    updated.agent.algorithm.num_learning_epochs = 5
    assert original.agent.seed == 101 and updated.agent.algorithm.num_learning_epochs == 5


def runner_fixture(worlds=1):
    from rsl_rl.modules.normalization import EmpiricalNormalization
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.obs_normalizer = EmpiricalNormalization(4)
    actor, critic = Model(), Model()
    actor.obs_normalizer.update(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    critic.obs_normalizer.update(torch.arange(20, dtype=torch.float32).reshape(5, 4))
    return SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: actor, _raw_critic=critic),
                           gpu_world_size=worlds, is_distributed=worlds > 1)


def test_actual_rsl_normalization_buffers_are_finite_and_counted():
    runner = runner_fixture()
    result = distributed_normalization_evidence(runner, include_parameters=True)
    assert result["states"]["actor"]["count"] == 3
    assert result["states"]["critic"]["count"] == 5
    assert result["all_ranks_match"] and result["normalization_finite"]
    assert len(result["states"]["actor"]["model_state_sha256"]) == 64
    runner.alg._raw_critic.obs_normalizer._mean[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        distributed_normalization_evidence(runner)


def test_rank_normalizer_disagreement_is_rejected(monkeypatch):
    runner = runner_fixture(2)
    def gather(outputs, state):
        outputs[:] = [deepcopy(state), deepcopy(state)]
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    result = distributed_normalization_evidence(runner)
    assert len(result["rank_state_sha256"]) == 2
    def changed(outputs, state):
        gather(outputs, state)
        outputs[1]["critic"]["count"] += 1
    monkeypatch.setattr(torch.distributed, "all_gather_object", changed)
    with pytest.raises(ValueError, match="differ across ranks"):
        distributed_normalization_evidence(runner)


def test_native_tmp_is_shared_shallow_and_never_rank_nested(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    shared = PROJECT_ROOT/".cache/capacity_tmp_test"
    _local_caches(tmp_path/"rank_01", shared_tmp=shared)
    assert Path(os.environ["TMPDIR"]) == shared
    assert "rank_01" not in os.environ["TMPDIR"]
    assert "rank_01" in os.environ["WARP_CACHE_PATH"]
    with pytest.raises(ValueError, match="shallow resolved TMPDIR"):
        _local_caches(tmp_path, shared_tmp=shared/("x"*120))


def test_engineering_capacity_aggregation_retains_scope_and_verifies_rank_buffers():
    from test_wbc_capacity import contract_fixture, rank_fixture
    from human_punching.wbc.capacity import aggregate_capacity
    config = CapacityConfig(global_environments=8192, gpu_count=2)
    contract = contract_fixture(config)
    contract["payload"]["reference_and_body"].update(scope=SCOPE,
        engineering_admission_sha256="b"*64, observer_evidence_sha256="c"*64, bridge_evidence_sha256="f"*64,
        engineering_recipe=LEGACY_RECIPE.manifest(), engineering_recipe_sha256=LEGACY_RECIPE.sha256,
        training_initialization_evidence_sha256=None)
    contract["sha256"] = digest(contract["payload"])
    rows = [rank_fixture(config, rank) for rank in range(2)]
    state = {name: {"count": 983040, "sha256": "d"*64, "model_state_sha256": "e"*64}
             for name in ("actor", "critic")}
    for row in rows:
        row.update(measurement_contract_sha256=contract["sha256"], engineering_admission_sha256="b"*64,
            observer_evidence_sha256="c"*64, native_observer_enabled=True,
            bridge_evidence_sha256="f"*64, cycle_bridge_enabled=True, task_objectives_enabled=True,
            engineering_recipe_sha256=LEGACY_RECIPE.sha256, training_initialization_evidence_sha256=None,
            cycle_after={**initialization_health_fixture(), "terminal_accounting_consistent": True,
                "fresh_imitation": imitation_health_fixture(120),
                "readiness": {"quiet_ticks": 40, "maximum_settling_ticks": 1200}},
            cycle_after_warmup={**initialization_health_fixture(), "fresh_imitation": imitation_health_fixture(48)},
            normalization_before={"states": {name: {"count": 0} for name in ("actor", "critic")}},
            normalization_after={"all_ranks_match": True, "normalization_finite": True,
                "states": deepcopy(state), "rank_state_sha256": [digest(state)]*2})
        row["native_observation_schema"]["actor_width"] = 398
        row["native_observation_schema_sha256"] = digest(row["native_observation_schema"])
    report = aggregate_capacity(config, rows, "a"*64, contract)
    assert report["status"] == "passed"
    assert report["learning_scope"] == SCOPE and report["production_reference_qualified"] is False
    assert report["launched_workload_exercised"] is False
    rows[1]["normalization_after"]["rank_state_sha256"][0] = "changed"
    report = aggregate_capacity(config, rows, "a"*64, contract)
    assert report["status"] == "failed" and not report["checks"]["distributed_normalization"]


def imitation_health_fixture(windows):
    from human_punching.wbc.runtime.imitation_rewards import UPSTREAM_NAMES
    return {"configured_terms": list(UPSTREAM_NAMES), "observed_windows": windows,
        "term_nonfinite_counts": dict.fromkeys(UPSTREAM_NAMES, 0),
        "term_call_counts": dict.fromkeys(UPSTREAM_NAMES, windows),
        "all_raw_terms_finite": True, "every_term_each_window": True}


@pytest.mark.parametrize("mutation", (None, "sanitized", "skipped", "duplicate", "missing", "boolean", "no_windows"))
def test_raw_reward_evidence_does_not_trust_sanitized_manager_outputs(mutation):
    health = imitation_health_fixture(120)
    name = health["configured_terms"][0]
    if mutation == "sanitized": health["term_nonfinite_counts"][name] = 1
    elif mutation == "skipped": health["term_call_counts"][name] -= 1
    elif mutation == "duplicate": health["term_call_counts"][name] += 1
    elif mutation == "missing": del health["term_nonfinite_counts"][name]
    elif mutation == "boolean": health["observed_windows"] = True
    elif mutation == "no_windows": health = imitation_health_fixture(0)
    assert fresh_imitation_health_matches(health, 120) is (mutation is None)
    assert fresh_imitation_health_matches(imitation_health_fixture(0), 0)


def bridge_report_fixture():
    checks = {name: True for name in ("complete_native_samples", "complete_policy_samples",
        "ordinary_reset_exercised", "one_initialization_per_reset", "fresh_endpoint_fk",
        "fresh_actor_anchor", "finite_task_rewards", "finite_observations",
        "zero_reference_before_launch", "future_held_before_launch")}
    report = dict(schema="human_wbc_cycle_native_qualification_v1", status="passed", training=False,
        exception=None, runtime={"declared_pins_match": True},
        readiness={"quiet_ticks": 40, "maximum_settling_ticks": 1200}, support_artifact_sha256=None,
        actor_observation_dimension=[398], requested_policy_steps=400, completed_policy_steps=400,
        actual_physics_samples=1600, environments=16,
        audit=dict(status="passed", checks=checks, launch_observed=False, launch_checks_exercised=False,
                   launch_qualification="not_tested_unready"))
    execution = dict(schema="human_wbc_cycle_qualification_execution_v1", returncode=0,
        budget_stop=None, slurm_job_id="synthetic-not-native-evidence", packet_sha256="a"*64)
    packet = dict(schema="human_wbc_cycle_qualification_packet_v1", policy_steps=400)
    return report, execution, packet


def test_unlaunched_native_bridge_evidence_never_becomes_launched_qualification():
    report, execution, packet = bridge_report_fixture()
    capacity_evidence._require_bridge_report(report, execution, packet, "a"*64)
    report["audit"]["launch_qualification"] = "diagnostic_only"
    with pytest.raises(ValueError, match="native bridge"):
        capacity_evidence._require_bridge_report(report, execution, packet, "a"*64)


@pytest.mark.parametrize("change", ("check", "step", "dwell", "actor", "support", "job", "hash"))
def test_bridge_evidence_rejects_incomplete_or_different_semantics(change):
    report, execution, packet = bridge_report_fixture()
    if change == "check": report["audit"]["checks"]["fresh_endpoint_fk"] = False
    elif change == "step": report["completed_policy_steps"] -= 1
    elif change == "dwell": report["readiness"]["quiet_ticks"] = 1
    elif change == "actor": report["actor_observation_dimension"] = [394]
    elif change == "support": report["support_artifact_sha256"] = "b"*64
    elif change == "job": execution["slurm_job_id"] = None
    else: execution["packet_sha256"] = "b"*64
    with pytest.raises(ValueError, match="native bridge"):
        capacity_evidence._require_bridge_report(report, execution, packet, "a"*64)


def initialization_health_fixture(recipe=LEGACY_RECIPE):
    revised = recipe.training.reference_start_probability > 0
    return {"cumulative_attempts": 20, "cumulative_launches": 15 if revised else 0,
        "cumulative_strict_guard_starts": 5 if revised else 20,
        "cumulative_strict_ready_launches": 0, "cumulative_training_reference_starts": 15 if revised else 0,
        "cumulative_training_support_grace_ticks": 30 if revised else 0,
        "training_initialization": recipe.training.initialization_parameters(),
        "launch_count_semantics": "all_reference_starts_including_training_initialization",
        "terminal_receipts": {"terminal_attempts": 12, "launched_attempts": 8 if revised else 0,
            "strict_ready_launch_attempts": 0, "training_reference_start_attempts": 8 if revised else 0,
            "unclassified_launch_attempts": 0, "initialization_metadata_missing_attempts": 0,
            "initial_support_grace_ticks_used": 16 if revised else 0}}


@pytest.mark.parametrize("recipe", (LEGACY_RECIPE, REVISED_RECIPE))
def test_explicit_recipe_preserves_controller_and_has_distinct_contract(converted, monkeypatch, recipe):
    from human_punching.wbc.runtime.training_initialization import TrainingInitializationCfg
    monkeypatch.setattr(engineering_admission, "verify_engineering_admission", lambda *args: None)
    cfg = configure_engineering_capacity(converted, admission_fixture(), num_envs=4096, engineering_recipe=recipe)
    require_engineering_configuration(cfg, engineering_recipe=recipe)
    assert cfg.commands["motion"].training_initialization == TrainingInitializationCfg(**recipe.training.initialization_parameters())
    assert cfg.rewards["human_guard_joint_posture"].weight == recipe.training.guard_joint_posture_weight
    assert cfg.sim.mujoco.timestep == .005 and cfg.decimation == 4
    assert cfg.commands["motion"].assistive_wrench_enabled is False
    other = REVISED_RECIPE if recipe is LEGACY_RECIPE else LEGACY_RECIPE
    with pytest.raises(ValueError, match="explicit unassisted"):
        require_engineering_configuration(cfg, engineering_recipe=other)


@pytest.mark.parametrize("mutation", ("probability", "grace", "posture_metric", "posture_weight", "other_weight"))
def test_revised_configuration_rejects_hidden_recipe_changes(converted, monkeypatch, mutation):
    from human_punching.wbc.runtime.training_initialization import TrainingInitializationCfg
    monkeypatch.setattr(engineering_admission, "verify_engineering_admission", lambda *args: None)
    cfg = configure_engineering_capacity(converted, admission_fixture(), num_envs=256, engineering_recipe=REVISED_RECIPE)
    if mutation == "probability": cfg.commands["motion"].training_initialization = TrainingInitializationCfg(.5, 4)
    elif mutation == "grace": cfg.commands["motion"].training_initialization = TrainingInitializationCfg(.75, 3)
    elif mutation == "posture_metric": cfg.rewards["human_guard_joint_posture"].params["guard_joint_posture_metric"] = "rms"
    elif mutation == "posture_weight": cfg.rewards["human_guard_joint_posture"].weight = .25
    else: cfg.rewards["human_guard_root_position"].weight += .1
    with pytest.raises(ValueError, match="explicit unassisted"):
        require_engineering_configuration(cfg, engineering_recipe=REVISED_RECIPE)


def test_native_initializer_evidence_is_mandatory_and_not_borrowed_by_legacy(converted):
    from human_punching.wbc.capacity_evidence import training_initialization_capacity_evidence
    with pytest.raises(ValueError, match="completed native training initialization"):
        build_measurement_contract(converted, scale=TrainingScale(), engineering_admission=admission_fixture(),
            observer_report=Path("synthetic"), bridge_report=Path("synthetic"), engineering_recipe=REVISED_RECIPE)
    with pytest.raises(ValueError, match="Legacy recipe"):
        build_measurement_contract(converted, scale=TrainingScale(), engineering_admission=admission_fixture(),
            observer_report=Path("synthetic"), bridge_report=Path("synthetic"), training_initialization_report=Path("synthetic"))
    with pytest.raises(ValueError, match="Legacy recipe"):
        training_initialization_capacity_evidence(converted, Path("synthetic"), engineering_recipe=LEGACY_RECIPE)


@pytest.mark.parametrize("recipe", (LEGACY_RECIPE, REVISED_RECIPE))
@pytest.mark.parametrize("mutation", (None, "missing", "boolean", "relabel", "missing_metadata", "unclassified", "grace", "probability", "counts"))
def test_initialization_health_counts_never_masquerade_as_strict(recipe, mutation):
    from human_punching.wbc.capacity_evidence import initialization_health_matches
    health = initialization_health_fixture(recipe)
    if mutation == "missing": del health["cumulative_training_reference_starts"]
    elif mutation == "boolean": health["cumulative_strict_ready_launches"] = False
    elif mutation == "relabel": health["terminal_receipts"]["strict_ready_launch_attempts"] += 1
    elif mutation == "missing_metadata": health["terminal_receipts"]["initialization_metadata_missing_attempts"] = 1
    elif mutation == "unclassified": health["terminal_receipts"]["unclassified_launch_attempts"] = 1
    elif mutation == "grace": health["cumulative_training_support_grace_ticks"] = 1000
    elif mutation == "probability": health["training_initialization"]["reference_start_probability"] = .5
    elif mutation == "counts": health["cumulative_strict_guard_starts"] += 1
    assert initialization_health_matches(health, recipe) is (mutation is None)


def initialization_mode_fixture(recipe):
    checks = dict.fromkeys(("full_reference_arrays_unchanged", "strict_initialization_exercised",
        "training_initialization_exercised", "training_rows_start_at_frame_zero", "strict_rows_do_not_launch_at_reset",
        "reset_has_no_fabricated_readiness_dwell", "training_start_has_explicit_preparation",
        "first_observation_training_phase", "first_observation_strict_phase", "first_observation_has_zero_elapsed",
        "actor_future_reference_matches_command", "first_action_condition_unchanged_after_reset_commit",
        "first_action_reference_unchanged_after_reset_commit", "full_plan_identity_target_and_g1_context_frozen",
        "grace_exactly_matches_initial_unsupported_endpoints", "grace_has_four_tick_absolute_cap",
        "raw_support_flags_preserved", "all_receipts_explicitly_ineligible_and_start_typed"), True)
    native = dict.fromkeys(("complete_native_samples", "complete_policy_samples", "ordinary_reset_exercised",
        "one_initialization_per_reset", "fresh_endpoint_fk", "fresh_actor_anchor", "finite_task_rewards", "finite_observations"), True)
    return {"recipe": recipe.manifest(), "recipe_sha256": recipe.sha256,
        "schema": "human_wbc_training_initialization_native_v1", "status": "passed", "exception": None,
        "training": False, "learning_updates": 0, "acceptance_qualified": False, "reference_domain_qualified": False,
        "runtime": {"declared_pins_match": True}, "actor": "zero_residual_no_network", "seed": 10010,
        "later_g1_metadata_perturbation": True, "initialization": recipe.training.initialization_parameters(),
        "environments": 16, "requested_policy_steps": 400, "completed_policy_steps": 400,
        "actual_physics_samples": 1600, "observer_samples": 1600, "actor_observation_dimension": [398],
        "readiness": {"quiet_ticks": 40, "maximum_settling_ticks": 1200}, "stream_capture": {"passed": True},
        "fresh_imitation_health": imitation_health_fixture(400),
        "initialization_audit": {"status": "passed", "checks": checks,
            "strict_launch_qualification": "not_granted_by_this_instrumentation_audit",
            "ordinary_reset_starts": 20, "strict_guard_reset_starts": 5 if recipe is REVISED_RECIPE else 20,
            "training_frame_zero_starts": 15 if recipe is REVISED_RECIPE else 0},
        "inherited_native_audit": {"status": "passed", "checks": native,
            "launch_qualification": "instrumentation_only_training_starts_separate"}}


@pytest.mark.parametrize("recipe", (LEGACY_RECIPE, REVISED_RECIPE))
@pytest.mark.parametrize("mutation", (None, "recipe", "clock", "steps", "raw_reward", "failed_check", "missing_check",
    "support_grace", "acceptance", "strict_label", "missing_strict", "count_partition"))
def test_native_initialization_evidence_fails_closed(recipe, mutation):
    report = initialization_mode_fixture(recipe)
    if mutation == "recipe": report["recipe"]["training"]["guard_joint_posture_weight"] = 99
    elif mutation == "clock": report["actual_physics_samples"] -= 1
    elif mutation == "steps": report["completed_policy_steps"] -= 1
    elif mutation == "raw_reward": report["fresh_imitation_health"]["observed_windows"] -= 1
    elif mutation == "failed_check": report["initialization_audit"]["checks"]["raw_support_flags_preserved"] = False
    elif mutation == "missing_check": del report["initialization_audit"]["checks"]["first_observation_training_phase"]
    elif mutation == "support_grace": report["initialization"]["initial_support_grace_ticks"] = 5
    elif mutation == "acceptance": report["acceptance_qualified"] = True
    elif mutation == "strict_label": report["initialization_audit"]["strict_launch_qualification"] = "qualified"
    elif mutation == "missing_strict": report["initialization_audit"]["strict_guard_reset_starts"] = 0
    elif mutation == "count_partition": report["initialization_audit"]["ordinary_reset_starts"] += 1
    if mutation is None:
        capacity_evidence._require_initialization_mode(report, recipe)
    else:
        with pytest.raises(ValueError):
            capacity_evidence._require_initialization_mode(report, recipe)


def initialization_panel_fixture(tmp_path, monkeypatch):
    """Synthetic immutable-file fixture, never submitted or reported as native."""
    from human_punching.wbc.provenance import sha256_file
    monkeypatch.setattr(engineering_admission, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(capacity_evidence, "PROJECT_ROOT", tmp_path)
    bundle = tmp_path/"bundle"; bundle.mkdir(); (bundle/"manifest.json").write_text('{"synthetic":true}')
    monkeypatch.setattr(capacity_evidence, "load_runtime_bundle", lambda value: SimpleNamespace(root=bundle))
    run = tmp_path/"fixture/run"; run.mkdir(parents=True)
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return sha256_file(path)
    sources = set(capacity_evidence.OBSERVER_REQUIRED_SOURCES)
    sources.update("src/human_punching/wbc/runtime/"+name+".py" for name in (
        "action", "body", "bundle", "command", "mdp", "task", "task_rewards", "task_reward_adapter",
        "training_initialization", "engineering_recipe", "imitation_rewards"))
    sources.update("src/human_punching/wbc/telemetry/"+name+".py" for name in (
        "cycle_runtime", "event_state", "readiness_clock", "attempt_receipts", "training_initialization_qualification"))
    inputs = {}
    for relative in sources:
        path=tmp_path/relative;path.parent.mkdir(parents=True, exist_ok=True);path.write_text('# synthetic fixture\n')
        inputs[relative]=sha256_file(path)
    rows, mode_sha, completion_modes, configuration = [], {}, {}, {}
    for name, recipe in (("strict_default", LEGACY_RECIPE), ("mixed_training", REVISED_RECIPE)):
        directory=run/name;directory.mkdir()
        report=initialization_mode_fixture(recipe);report["bundle_manifest_sha256"]=sha256_file(bundle/"manifest.json")
        report["hashes"]={}
        for key in ("native", "pre_interval", "endpoint", "policy_boundary", "initial_observation"):
            path=directory/(key+'.npz');path.write_bytes(b'synthetic non-native array')
            report['hashes'][key]=sha256_file(path)
        (directory/'compiled_scene.mjb').write_bytes(b'synthetic compiled scene')
        save(directory/'attempts/attempt_0.json', {'synthetic_fixture': True})
        report['terminal_receipts']=1
        mode_sha[name]=save(directory/'report.json', report)
        rows.append({'mode':name,'status':'passed','report_sha256':mode_sha[name]})
        configuration[name]={'recipe':recipe.manifest(),'recipe_sha256':recipe.sha256}
        completion_modes[name]={'status':'passed','recipe_sha256':recipe.sha256,
            'compiled_scene_sha256':sha256_file(directory/'compiled_scene.mjb')}
    panel={'schema':'human_wbc_training_initialization_panel_v1','modes':rows,'status':'passed',
        'training':False,'acceptance_qualified':False,'strict_launch_qualified':False}
    report_sha=save(run/'report.json',panel)
    products={str(p.relative_to(run)):sha256_file(p) for p in run.rglob('*') if p.is_file()}
    save(run/'completion_manifest.json', {'schema':'human_wbc_training_initialization_completion_v1',
        'status':'passed','driver_returncode':0,'driver_exception':None,'acceptance_qualified':False,
        'strict_launch_qualified':False,'source_schema':panel['schema'],'aggregate_report_sha256':report_sha,
        'completed_modes':['strict_default','mixed_training'],'learning_updates':0,'products':products,
        'modes':completion_modes,'mode_report_sha256':mode_sha,
        'qualification_source_sha256':inputs['src/human_punching/wbc/telemetry/training_initialization_qualification.py']})
    packet_sha=save(run.parent/'packet.json', {'schema':'human_wbc_training_initialization_packet_v1',
        'modes':['strict_default','mixed_training'],'training':False,'learning_updates':0,
        'acceptance_qualified':False,'strict_launch_qualified':False,'environments_per_mode':16,
        'policy_steps_per_mode':400,'physics_steps_per_mode':1600,'configuration':configuration,'inputs':inputs})
    save(run.parent/'execution.json', {'schema':'human_wbc_training_initialization_execution_v1',
        'returncode':0,'budget_stop':None,'job_id':'synthetic-not-native-evidence','packet_sha256':packet_sha,'learning_updates':0})
    return run, bundle


@pytest.mark.parametrize('mutation',(None,'compiled_model','raw_array','receipt','source','execution','completion'))
def test_native_initialization_file_closure_rejects_changed_artifacts(tmp_path,monkeypatch,mutation):
    run,bundle=initialization_panel_fixture(tmp_path,monkeypatch)
    if mutation=='compiled_model':(run/'mixed_training/compiled_scene.mjb').write_bytes(b'changed')
    elif mutation=='raw_array':(run/'mixed_training/native.npz').write_bytes(b'changed')
    elif mutation=='receipt':(run/'mixed_training/attempts/attempt_0.json').unlink()
    elif mutation=='source':(tmp_path/'src/human_punching/wbc/runtime/training_initialization.py').write_text('# changed')
    elif mutation=='execution':
        path=run.parent/'execution.json';value=json.loads(path.read_text());value['packet_sha256']='0'*64;path.write_text(json.dumps(value))
    elif mutation=='completion':
        path=run/'completion_manifest.json';value=json.loads(path.read_text());value['status']='failed';path.write_text(json.dumps(value))
    if mutation is None:
        result=capacity_evidence.training_initialization_capacity_evidence(bundle,run/'report.json',engineering_recipe=REVISED_RECIPE)
        assert result['recipe_sha256']==REVISED_RECIPE.sha256
        assert not result['acceptance_qualified'] and not result['strict_ready_launch_qualified']
        assert len(result['trace_payloads'])==17
    else:
        with pytest.raises((ValueError,FileNotFoundError)):
            capacity_evidence.training_initialization_capacity_evidence(bundle,run/'report.json',engineering_recipe=REVISED_RECIPE)
