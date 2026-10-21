"""Guard-only fallback preparation tests; no dynamics, learner update or job."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from human_punching.wbc import guard_training as module
from human_punching.wbc.guard_training import (
    GuardSmokePlan, build_guard_training_cfg, guard_command_types, guard_geometry,
    make_guard_runner, require_guard_capacity, validate_guard_checkpoint,
)
from human_punching.wbc.validation import digest, tensor_digest


@pytest.mark.parametrize("values", [
    {"environments": 8192}, {"environments": 17}, {"environments": True},
    {"iterations": 250}, {"iterations": 0}, {"seed": True}, {"episode_seconds": 2.},
])
def test_guard_smoke_cannot_masquerade_as_production_or_long_training(values):
    with pytest.raises(ValueError):
        GuardSmokePlan(**values)


def test_smoke_keeps_inherited_rollout_and_optimizer_counts():
    report = GuardSmokePlan().accounting()
    assert report["global_environments"] == 256
    assert report["fresh_transitions_per_iteration"] == 256*24
    assert report["configured_optimizer_steps_per_iteration"] == 20
    assert report["production_global_environments"] == 8192
    assert report["production_batch_qualified"] is False
    assert report["historical_specialists_allowed"] is False


def test_real_guard_geometry_uses_fk_without_dynamics_or_stability_claim(task10_common_bundle, monkeypatch):
    for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward"):
        monkeypatch.setattr(mujoco, name, lambda *a, **kw: pytest.fail("Geometry-only test cannot execute dynamics"))
    answer = guard_geometry(task10_common_bundle)
    assert answer["passed"]
    assert answer["minimum_projected_support_margin_m"] > .05
    assert answer["minimum_glove_clearance_m"] >= .16
    assert "no loaded-contact or stability claim" in answer["scope"]
    assert task10_common_bundle.manifest["qualified_for_training"] is False


def test_static_reference_cannot_leak_future_outward_motion_or_initial_velocity():
    cls, _ = guard_command_types()
    q = torch.arange(4*55, dtype=torch.float32).reshape(4, 55)
    fake = SimpleNamespace(master={"qpos": q, "qvel": torch.ones(4, 54),
        "joint_feedforward": torch.zeros(4, 48)},
        master_starts=torch.tensor([0, 2]), trajectory_ids=torch.tensor([0, 1]),
        physics_steps=torch.tensor([9, 110]), cfg=SimpleNamespace(static_joint_feedforward_nm=tuple(range(48))))
    np.testing.assert_array_equal(cls.sample_master(fake, "qpos", 24), q[[0, 2]])
    assert torch.count_nonzero(cls.sample_master(fake, "qvel")) == 0
    torch.testing.assert_close(cls.sample_master(fake, "joint_feedforward"), torch.arange(48).float().repeat(2, 1))
    torch.testing.assert_close(cls.elapsed.fget(fake), torch.zeros(2))
    torch.testing.assert_close(fake.physics_steps, torch.tensor([9, 110]))
    with pytest.raises(RuntimeError, match="Mid-cycle"):
        cls.reset_to_frame(fake, [0], 10)


def test_guard_config_retains_actor_contract_but_removes_punch_objective(task10_common_bundle, monkeypatch):
    # This test inspects config changes only; it never declares native evidence.
    monkeypatch.setattr(module, "verify_guard_admission", lambda *args: None)
    from human_punching.wbc.runtime.task import build_human_env_cfg
    baseline = build_human_env_cfg(task10_common_bundle, 256, evaluation=True)
    cfg = build_guard_training_cfg(task10_common_bundle, {"joint_feedforward_nm": [0.]*48}, GuardSmokePlan())
    assert cfg.auto_reset and cfg.episode_length_s == 6.
    assert cfg.events == {} and not cfg.commands["motion"].assistive_wrench_enabled
    assert cfg.commands["motion"].rsi.sampling_mode == "start"
    assert cfg.actions == baseline.actions
    for group in ("actor", "critic"):
        assert list(cfg.observations[group].terms) == list(baseline.observations[group].terms)
    assert "timed_target" not in cfg.rewards and "cycle_complete" not in cfg.terminations
    assert cfg.sim.mujoco.timestep == .005 and cfg.decimation == 4
    assert task10_common_bundle.manifest["qualified_for_training"] is False
    with pytest.raises(ValueError, match="unqualified"):
        task10_common_bundle.require_training_qualification()


def test_missing_guard_capacity_never_becomes_a_launch_estimate():
    with pytest.raises(ValueError, match="capacity/PPO"):
        require_guard_capacity({}, {"sha256": "a"*64}, GuardSmokePlan())


def test_guard_runner_rejects_historical_or_any_checkpoint_initialization():
    runner = make_guard_runner(None, {}, GuardSmokePlan(), capacity={})
    with pytest.raises(RuntimeError, match="scratch"):
        runner.load(SimpleNamespace(), "historical_specialist.pt")


def test_guard_initialization_requires_new_shared_ancestry_and_exact_actor_state():
    actor = {"weight": torch.ones(2, 3), "obs_normalizer.mean": torch.zeros(3)}
    admission, observations = {"sha256": "a"*64}, {"actor": "fixed-schema"}
    info = {"schema": "human_wbc_guard_checkpoint_v1", "origin": module.ORIGIN,
        "scratch_initialized": True, "historical_initialization": False,
        "completed_iterations": 10, "admission": admission, "observation_schema": observations,
        "actor_state_sha256": tensor_digest(actor)}
    checkpoint = {"infos": {"guard_training": info}, "actor_state_dict": actor, "critic_state_dict": {}}
    assert validate_guard_checkpoint(checkpoint, expected_admission=admission,
        expected_observation_schema=observations)["completed_iterations"] == 10
    info["historical_initialization"] = True
    with pytest.raises(ValueError, match="new scratch shared"):
        validate_guard_checkpoint(checkpoint, expected_admission=admission, expected_observation_schema=observations)
    info["historical_initialization"] = False
    actor["weight"][0, 0] = 2.
    with pytest.raises(ValueError, match="new scratch shared"):
        validate_guard_checkpoint(checkpoint, expected_admission=admission, expected_observation_schema=observations)


def test_report_digest_cannot_be_changed_to_waive_native_evidence():
    with pytest.raises(ValueError, match="digest changed"):
        module.verify_guard_admission(None, {"sha256": "a"*64, "scope": "punch_production"})


def test_guard_capacity_contract_never_calls_punch_admission_and_ignores_run_length(task10_common_bundle, monkeypatch):
    from human_punching.wbc.runtime.bundle import RuntimeBundle
    from human_punching.wbc.runtime.task import human_ppo_cfg
    monkeypatch.setattr(module, "verify_guard_admission", lambda *args: None)
    monkeypatch.setattr(RuntimeBundle, "require_training_qualification",
                        lambda *args: pytest.fail("Guard learning must not need prior punch success"))
    admission = {"joint_feedforward_nm": [0.]*48, "sha256": "a"*64}
    plan = GuardSmokePlan()
    cfg = build_guard_training_cfg(task10_common_bundle, admission, plan)
    agent = human_ppo_cfg()
    first = module.guard_measurement_contract(task10_common_bundle, admission, plan, cfg, agent)
    cfg.seed = 999
    cfg.sim.nan_guard.output_dir = "different-output"
    agent.max_iterations, agent.seed, agent.save_interval = 5, 999, 6
    second = module.guard_measurement_contract(task10_common_bundle, admission, plan, cfg, agent)
    assert first["sha256"] == second["sha256"]
    cfg.actions["joint_pos"].scale = .30
    changed = module.guard_measurement_contract(task10_common_bundle, admission, plan, cfg, agent)
    assert changed["sha256"] != first["sha256"]


def test_guard_capacity_does_not_accept_unbacked_success_dictionary():
    plan = GuardSmokePlan()
    fields = {"schema": "human_wbc_guard_capacity_v1", "status": "passed",
        "admission_sha256": "a"*64, "plan_sha256": digest(asdict(plan)),
        "includes_inherited_ppo": True, "gpu_count": 1, "gpu_name": "NVIDIA L40S",
        "normalization_finite": True, "optimizer_steps_per_iteration": 20,
        "slurm_job_id": "cpu_fixture_not_a_job"}
    with pytest.raises(ValueError, match="intact native measurement"):
        require_guard_capacity(fields, {"sha256": "a"*64}, plan)


def test_guard_contract_survives_real_scene_construction_without_dropping_terrain_fields(
        task10_common_bundle, monkeypatch):
    from mjlab.scene import Scene
    from human_punching.wbc.runtime.task import human_ppo_cfg
    from human_punching.wbc.measurement_contract import verify_measurement_contract
    monkeypatch.setattr(module, "verify_guard_admission", lambda *args: None)
    for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward"):
        monkeypatch.setattr(mujoco, name, lambda *a, **kw: pytest.fail("No dynamics during Scene construction"))
    admission = {"joint_feedforward_nm": [0.]*48, "sha256": "a"*64}
    plan = GuardSmokePlan()
    cfg = build_guard_training_cfg(task10_common_bundle, admission, plan)
    agent = human_ppo_cfg()
    before = module.guard_measurement_contract(task10_common_bundle, admission, plan, cfg, agent)
    assert cfg.scene.terrain.num_envs == cfg.scene.num_envs == 256
    assert cfg.scene.terrain.env_spacing == cfg.scene.env_spacing
    Scene(cfg.scene, device="cpu")
    after = module.guard_measurement_contract(task10_common_bundle, admission, plan, cfg, agent)
    assert before == after
    verify_measurement_contract(before, after)
    cfg.scene.terrain.num_envs += 1
    changed = module.guard_measurement_contract(task10_common_bundle, admission, plan, cfg, agent)
    with pytest.raises(ValueError, match="different.*contract"):
        verify_measurement_contract(before, changed)


@pytest.mark.parametrize("status,samples,executed,expected", [
    ("failed", 0, False, "evaluation_failed"),
    ("recorded", 0, False, "evaluation_failed"),
    ("recorded", 1201, True, "recorded"),
])
def test_session_evaluation_failure_retains_completed_learning_and_nonzero_exit(
        tmp_path, monkeypatch, status, samples, executed, expected):
    audit = {"schema": "human_wbc_learned_guard_evaluation_v1", "status": status,
        "recorded_samples": samples, "environments": 16, "guard_success_count": 0,
        "episodes": [{"status": "evaluated" if executed else "failed_or_incomplete"} for _ in range(16)],
        "exception": None if executed else {"type": "ValueError", "message": "CPU fixture schema mismatch"}}
    path = tmp_path/"fixture_audit.json"
    path.write_text(json.dumps(audit))
    record = {"learning_completed": True,
        "smoke": {"completed_iterations": 64, "total_fresh_transitions": 393216}}
    module._record_evaluation_outcome(record, path)
    assert record["status"] == expected and record["evaluation_completed"] is executed
    assert record["learning_completed"] and record["smoke"]["total_fresh_transitions"] == 393216
    monkeypatch.setattr(module, "run_guard_session", lambda *args, **kwargs: record)
    code = module.main(["--bundle", str(tmp_path), "--native-guard-report", str(path),
        "--support-artifact", str(path), "--output", str(tmp_path/"unused")])
    assert code == (0 if executed else 2)
