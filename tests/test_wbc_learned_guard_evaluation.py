"""Frozen guard evaluator CPU contracts; no native simulation or learned actor."""
from copy import deepcopy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from human_punching.wbc.conditioning import Conditioning
from human_punching.wbc.guard_probe import common_guard_identity
from human_punching.wbc.learned_guard_evaluation import (
    ORIGIN, audit_residual_guard_drive, build_learned_guard_cfg, evaluation_caches, learned_guard_command_types,
    run_learned_guard_evaluation, verify_guard_actor_identity, verify_guard_observation_schema,
)
from human_punching.wbc.native_cycle_probe import ProbeMetrics
from human_punching.wbc.provenance import configure_local_caches, sha256_file
from human_punching.wbc.trainedactor_evaluation import actor_identity
from human_punching.wbc.validation import digest


@pytest.fixture
def actor_case(task10_common_bundle):
    bundle = task10_common_bundle
    observations = TensorDict({"actor": torch.randn(4, 6)}, batch_size=[4])
    model = MLPModel(observations, {"actor": ["actor"]}, "actor", 48,
                     hidden_dims=(16, 8), obs_normalization=True)
    model.update_normalization(observations)
    actor = torch.jit.script(model.as_jit().eval())
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    ff = np.linspace(-1., 1., 48)
    identity = {**actor_identity(actor), "reference_manifest_sha256": sha256_file(bundle.root/"manifest.json"),
        "model_sha256": sha256_file(bundle.model_path), "controller_sha256": bundle.controller_sha256(),
        "conditioning": Conditioning.COMBINED.value, "joint_feedforward_sha256": digest(ff.tolist()),
        "guard_training_origin": ORIGIN, "guard_admission_sha256": "a"*64,
        "training_checkpoint_sha256": "b"*64, "observation_schema": {"fixture": True}}
    return actor, identity, ff


def test_frozen_actor_identity_binds_reference_normalization_and_controller(task10_common_bundle, actor_case):
    actor, identity, ff = actor_case
    verify_guard_actor_identity(actor, task10_common_bundle, identity, ff, Conditioning.COMBINED.value)
    for key in ("policy_sha256", "normalization_sha256", "controller_sha256", "reference_manifest_sha256"):
        changed = deepcopy(identity); changed[key] = "f"*64
        with pytest.raises(ValueError, match="identity differs"):
            verify_guard_actor_identity(actor, task10_common_bundle, changed, ff, Conditioning.COMBINED.value)
    changed = deepcopy(identity); changed.pop("training_checkpoint_sha256")
    with pytest.raises(ValueError, match="training_checkpoint"):
        verify_guard_actor_identity(actor, task10_common_bundle, changed, ff, Conditioning.COMBINED.value)


def test_actor_gate_precedes_cuda_and_output_creation(task10_common_bundle, actor_case, tmp_path, monkeypatch):
    actor, identity, ff = actor_case
    import human_punching.wbc.learned_guard_evaluation as module
    monkeypatch.setattr(module, "_require_allocated_cuda", lambda: pytest.fail("Identity before CUDA"))
    actor.obs_normalizer._mean.add_(.01)
    with pytest.raises(ValueError, match="identity differs"):
        run_learned_guard_evaluation(task10_common_bundle, actor, tmp_path/"absent", joint_feedforward_nm=ff,
            conditioning=Conditioning.COMBINED.value, seed=1, actor_contract_identity=identity)
    assert not (tmp_path/"absent").exists()


def test_static_elapsed_preserves_physics_clock_and_probe_ledger(task10_common_bundle, monkeypatch):
    configure_local_caches()
    monkeypatch.setattr(mujoco, "mj_step", lambda *a, **k: pytest.fail("No dynamics"))
    cls, _ = learned_guard_command_types()
    command = object.__new__(cls)
    command.physics_steps = torch.tensor([0, 1200])
    command.master_starts = torch.tensor([0, 2]); command.trajectory_ids = torch.tensor([0, 1])
    command.cfg = SimpleNamespace(static_guard_feedforward_nm=tuple(np.ones(48)))
    command.master = {"qpos": torch.arange(4*55).reshape(4, 55).float(),
                      "qvel": torch.ones(4, 54), "joint_feedforward": torch.zeros(4, 48)}
    assert not command.elapsed.any()
    for offset in (0, 10, 999):
        assert not command.sample_master("qvel", offset).any()
        assert torch.all(command.sample_master("joint_feedforward", offset) == 1.)
        torch.testing.assert_close(command.sample_master("qpos", offset), command.master["qpos"][[0, 2]])
    torch.testing.assert_close(command.physics_steps, torch.tensor([0, 1200]))
    cfg = build_learned_guard_cfg(task10_common_bundle, ProbeMetrics(), joint_feedforward_nm=np.ones(48),
                                  conditioning=Conditioning.COMBINED.value, seed=3)
    assert not cfg.auto_reset and cfg.events == cfg.rewards == cfg.terminations == {}
    assert cfg.decimation == 4 and cfg.sim.mujoco.timestep == .005 and cfg.seed == 3
    assert not cfg.commands["motion"].assistive_wrench_enabled


def residual_fixture(bundle):
    model = mujoco.MjModel.from_xml_path(str(bundle.model_path))
    qguard = np.asarray(common_guard_identity(bundle)["qpos"])
    qpos = np.broadcast_to(qguard, (5, 2, 55)).copy()
    qvel = np.zeros((5, 2, 54)); qvel[1:, :, 6:] = .1
    action = np.zeros((5, 2, 48)); action[1:, :, 0] = .8
    target = np.clip(qguard[7:]+.25*action, model.jnt_range[1:, 0], model.jnt_range[1:, 1])
    kp, kd, cap = (np.asarray(bundle.body[key]) for key in ("kp", "kd", "effort"))
    ff = np.linspace(-2., 2., 48)
    drive = kp*target+ff
    oldq = np.concatenate((qpos[:1, :, 7:], qpos[:-1, :, 7:]))
    oldv = np.concatenate((qvel[:1, :, 6:], qvel[:-1, :, 6:]))
    force = np.clip(drive-kp*oldq-kd*oldv, -cap, cap)
    return dict(qpos=qpos, qvel=qvel, raw_action=action, joint_position_target=target,
        joint_velocity_target=np.zeros_like(target), joint_effort_target=np.broadcast_to(ff, target.shape).copy(),
        ctrl=drive, actuator_force=force), ff


def test_independent_drive_uses_residual_limits_and_preceding_interval_force(task10_common_bundle):
    native, ff = residual_fixture(task10_common_bundle)
    result = audit_residual_guard_drive(native, task10_common_bundle, ff)
    assert result["status"] == "passed"
    assert all(value == 0. for value in result["maximum_errors"].values())
    native["actuator_force"][1, 0, 0] += .01
    result = audit_residual_guard_drive(native, task10_common_bundle, ff)
    assert not result["checks"]["preceding_interval_capped_native_force"]


@pytest.mark.parametrize("field", ["joint_position_target", "joint_effort_target", "ctrl", "joint_velocity_target"])
def test_drive_mismatch_cannot_be_hidden_by_finite_actions(task10_common_bundle, field):
    native, ff = residual_fixture(task10_common_bundle)
    native[field][1, 0, 2] += .01
    assert audit_residual_guard_drive(native, task10_common_bundle, ff)["status"] == "failed"


def test_existing_shallow_session_caches_are_preserved(tmp_path, monkeypatch):
    import human_punching.wbc.learned_guard_evaluation as module
    paths = {}
    for name in ("XDG_CACHE_HOME", "WARP_CACHE_PATH", "CUDA_CACHE_PATH", "TORCH_HOME", "TMPDIR", "MPLCONFIGDIR"):
        path = tmp_path/name
        path.mkdir()
        monkeypatch.setenv(name, str(path))
        paths[name] = str(path.resolve())
    monkeypatch.setattr(module, "_configure_caches", lambda *a: pytest.fail("Do not deepen NVRTC paths"))
    assert evaluation_caches(tmp_path/"nested/evaluation") == paths


def test_real_scene_actor_values_match_with_unused_critic_reward_width_change(task10_common_bundle, monkeypatch):
    """Actual configs, Scene name resolution, command properties and observations.

    CPU state tensors supply known observations without constructing a simulator.
    The critic group is narrowed to its actual reward feature to isolate the
    regression; the full actor group and every actor function remain untouched.
    """
    from mjlab.scene import Scene
    from mjlab.managers.observation_manager import ObservationManager, ObservationGroupCfg
    from human_punching.wbc import guard_training
    from human_punching.wbc.measurement_contract import configuration_value
    from human_punching.wbc.training_session import observation_schema
    from human_punching.wbc.references import FAMILIES

    monkeypatch.setattr(guard_training, "verify_guard_admission", lambda *args: None)
    for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward"):
        monkeypatch.setattr(mujoco, name, lambda *a, **kw: pytest.fail("No dynamics or forward"))
    bundle = task10_common_bundle
    ff = np.linspace(-1., 1., 48)
    train = guard_training.build_guard_training_cfg(bundle, {"joint_feedforward_nm": ff.tolist()},
        guard_training.GuardSmokePlan(environments=16))
    evaluate = build_learned_guard_cfg(bundle, ProbeMetrics(), joint_feedforward_nm=ff,
        conditioning=Conditioning.COMBINED.value, seed=10010)
    for group in ("actor", "critic"):
        assert configuration_value(train.observations[group]) == configuration_value(evaluate.observations[group])
    assert len([name for name in train.rewards if name.startswith("motion_")]) == 10
    assert evaluate.rewards == {}
    scene = Scene(evaluate.scene, device="cpu")
    robot = scene["robot"]
    assert tuple(robot.joint_names) == bundle.joint_names
    assert tuple(robot.body_names) == bundle.body_names
    first = []
    for clip in bundle.clips:
        with np.load(bundle.path(clip["master_path"]), allow_pickle=False) as arrays:
            first.append({key: arrays[key][0].copy() for key in
                ("qpos", "qvel", "joint_feedforward", "body_pos_w", "body_quat_w",
                 "body_lin_vel_w", "body_ang_vel_w")})
    master = {key: torch.tensor(np.stack([row[key] for row in first]), dtype=torch.float32)
              for key in first[0]}
    ids = torch.arange(16) % 4
    generator = torch.Generator().manual_seed(701)
    joint_pos = master["qpos"][ids, 7:] + .01*torch.randn(16, 48, generator=generator)
    robot._data = SimpleNamespace(joint_pos=joint_pos, joint_vel=torch.randn(16, 48, generator=generator),
        default_joint_pos=master["qpos"][ids, 7:], default_joint_vel=torch.zeros(16, 48),
        root_link_lin_vel_b=torch.randn(16, 3, generator=generator),
        root_link_ang_vel_b=torch.randn(16, 3, generator=generator),
        projected_gravity_b=torch.tensor([0., 0., -1.]).repeat(16, 1),
        gravity_vec_w=torch.tensor([0., 0., -1.]).repeat(16, 1),
        body_link_pos_w=master["body_pos_w"][ids]+scene.env_origins[:, None],
        body_link_quat_w=master["body_quat_w"][ids])
    action = SimpleNamespace(raw_action=torch.randn(16, 48, generator=generator).clamp(-1., 1.))
    results = []
    for cfg, cls in ((train, guard_training.guard_command_types()[0]),
                     (evaluate, learned_guard_command_types()[0])):
        command = object.__new__(cls)
        env = SimpleNamespace(cfg=cfg, scene=scene, device="cpu", num_envs=16,
            action_manager=SimpleNamespace(get_term=lambda name: action))
        command._env, command.cfg, command.robot = env, cfg.commands["motion"], robot
        command.master, command.master_starts = master, torch.arange(4)
        command.trajectory_ids, command.physics_steps = ids, torch.arange(16)*75
        command._tracked_joint_ids = None
        command.motion_anchor_body_index = command.robot_anchor_body_index = bundle.body_names.index("Hips")
        command.family_ids = torch.tensor([FAMILIES.index(c["family"]) for c in bundle.clips])
        command.targets = torch.tensor([c["target_w"] for c in bundle.clips], dtype=torch.float32)
        command.launch_times = torch.tensor([c["launch_time_s"] for c in bundle.clips])
        command.ttcs = torch.tensor([c["requested_ttc_s"] for c in bundle.clips])
        command.g1_context = torch.tensor([c["launch_features"] for c in bundle.clips])
        env.command_manager = SimpleNamespace(get_term=lambda name, command=command: command)
        term = cfg.observations["critic"].terms["motion_tracking_step_rewards"]
        cfg.observations["critic"] = ObservationGroupCfg(terms={"motion_tracking_step_rewards": term})
        env.observation_manager = ObservationManager(cfg.observations, env)
        results.append((observation_schema(env), env.observation_manager.compute(update_history=False)))
    training_schema, training_values = results[0]
    evaluation_schema, evaluation_values = results[1]
    assert training_values["actor"].shape == evaluation_values["actor"].shape == (16, 394)
    torch.testing.assert_close(training_values["actor"], evaluation_values["actor"], rtol=0, atol=0)
    assert training_values["critic"].shape == (16, 10)
    assert evaluation_values["critic"].shape == (16, 0)
    audit = verify_guard_observation_schema(evaluation_schema, training_schema)
    assert audit["status"] == "passed" and not audit["critic_used_for_inference"]
    assert not audit["critic_schema_matches_training"]
    assert {row["path"] for row in audit["critic_differences"]} == {
        "groups.critic.shape[0]", "groups.critic.terms[0].shape[0]"}
    for field, value in (("function", "wrong.function"), ("shape", [395]),
                         ("processing", {"scale": 2.})):
        changed = deepcopy(evaluation_schema)
        changed["groups"]["actor"]["terms"][0][field] = value
        with pytest.raises(ValueError, match="actor observations differ"):
            verify_guard_observation_schema(changed, training_schema)
    changed = deepcopy(evaluation_schema)
    changed["groups"]["actor"]["terms"].reverse()
    with pytest.raises(ValueError, match="actor observations differ"):
        verify_guard_observation_schema(changed, training_schema)
    changed = deepcopy(evaluation_schema)
    changed["previous_action"] = "unclipped_policy_output"
    with pytest.raises(ValueError, match="shared observation semantics differ"):
        verify_guard_observation_schema(changed, training_schema)
    changed = deepcopy(evaluation_schema)
    changed["groups"]["actor"]["normalization"] = "updated_in_evaluation"
    with pytest.raises(ValueError, match="actor observations differ"):
        verify_guard_observation_schema(changed, training_schema)


@pytest.mark.parametrize("value", [{}, {"schema": "human_wbc_observations_v1", "groups": {}}])
def test_guard_schema_comparison_never_accepts_missing_interfaces(value):
    with pytest.raises(ValueError, match="schemas are required"):
        verify_guard_observation_schema(value, value)
