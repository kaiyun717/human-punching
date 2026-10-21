"""CPU-only checks of the real WBC extension, with no mj_step or GPU creation."""

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

pytest.importorskip("wbc_mjlab")

from human_punching.wbc.references import PROJECT_ROOT, convert_release  # noqa: E402
from human_punching.wbc.runtime.action import PunchJointPositionAction  # noqa: E402
from human_punching.wbc.runtime.bundle import load_runtime_bundle, sha256  # noqa: E402
from human_punching.wbc.runtime.command import PunchMotionCommand  # noqa: E402
from human_punching.wbc.runtime.task import build_human_env_cfg, human_ppo_cfg  # noqa: E402


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("runtime") / "references"
    convert_release(PROJECT_ROOT / "outputs/four-punch-milestone", root)
    return root


def test_real_bundle_config_preserves_shared_ppo_and_clocks(converted, monkeypatch):
    def no_step(*args, **kwargs):
        raise AssertionError("Configuration must not execute dynamics")
    monkeypatch.setattr(mujoco, "mj_step", no_step)
    cfg = build_human_env_cfg(converted, num_envs=4096, evaluation=True)
    assert cfg.scene.num_envs == 4096
    assert cfg.sim.mujoco.timestep == .005 and cfg.decimation == 4
    assert cfg.sim.mujoco.cone == "pyramidal"
    assert (cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations) == (10, 20)
    assert cfg.events == {} and not cfg.commands["motion"].assistive_wrench_enabled
    assert cfg.commands["motion"].joint_position_range == (0., 0.)
    assert not cfg.auto_reset and cfg.actions["joint_pos"].scale == .25
    assert cfg.rewards["foot_slip"].weight < 0
    assert "punch_condition" in cfg.observations["actor"].terms
    assert "future_joint_reference" in cfg.observations["actor"].terms
    ppo = human_ppo_cfg()
    assert ppo.num_steps_per_env == 24
    assert ppo.algorithm.num_learning_epochs * ppo.algorithm.num_mini_batches == 20
    assert ppo.actor.obs_normalization and ppo.critic.obs_normalization
    assert ppo.clip_actions == 1.
    with pytest.raises(ValueError, match="unqualified"):
        build_human_env_cfg(converted, num_envs=8192, evaluation=False)


def test_native_affine_entity_compiles_and_preserves_body(converted, monkeypatch):
    from mjlab.terrains import TerrainEntity

    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("No dynamics in CPU preflight"))
    bundle = load_runtime_bundle(converted)
    original = mujoco.MjModel.from_xml_path(str(bundle.model_path))
    cfg = build_human_env_cfg(bundle)
    entity = cfg.scene.entities["robot"].build()
    model = entity.spec.compile()
    assert (model.nq, model.nv, model.nu) == (55, 54, 48)
    np.testing.assert_allclose(model.body_mass, original.body_mass, atol=1e-12)
    np.testing.assert_allclose(model.body_inertia, original.body_inertia, atol=1e-12)
    np.testing.assert_allclose(model.dof_armature, original.dof_armature, atol=1e-12)
    np.testing.assert_allclose(model.dof_damping, original.dof_damping, atol=1e-12)
    assert not np.any(model.actuator_ctrllimited)
    assert np.all(model.actuator_forcelimited)
    assert np.all(model.actuator_trnid[:, 0] > 0)
    np.testing.assert_allclose(model.actuator_biasprm[:, 1], -np.asarray(bundle.body["kp"]))
    np.testing.assert_allclose(model.actuator_biasprm[:, 2], -np.asarray(bundle.body["kd"]))
    np.testing.assert_allclose(model.actuator_forcerange[:, 1], bundle.body["effort"])
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = np.linspace(-.1, .1, model.nv)
    data.ctrl[:] = np.linspace(-200., 200., model.nu)
    mujoco.mj_forward(model, data)
    qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    vadr = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    kp, kd, cap = (np.asarray(bundle.body[k]) for k in ("kp", "kd", "effort"))
    expected = np.clip(data.ctrl - kp * data.qpos[qadr] - kd * data.qvel[vadr], -cap, cap)
    np.testing.assert_allclose(data.actuator_force, expected, atol=1e-10)
    np.testing.assert_array_equal(data.qfrc_applied, 0.)
    np.testing.assert_array_equal(data.xfrc_applied, 0.)
    terrain = TerrainEntity(cfg.scene.terrain, device="cpu").spec.compile()
    source_floor = original.geom("floor")
    new_floor = terrain.geom("terrain")
    for field in ("friction", "solref", "solimp", "contype", "conaffinity", "condim",
                  "priority", "margin", "gap", "solmix"):
        np.testing.assert_allclose(getattr(new_floor, field), getattr(source_floor, field),
                                   atol=1e-12)


def test_actual_action_constructor_resolves_all_human_hinges(converted, monkeypatch):
    """Exercise upstream action/name resolution without a dynamics allocation."""
    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("No CPU dynamics"))
    bundle = load_runtime_bundle(converted)
    cfg = build_human_env_cfg(bundle)
    entity = cfg.scene.entities["robot"].build()
    command = PunchMotionCommand.__new__(PunchMotionCommand)
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": entity},
                          command_manager=SimpleNamespace(get_term=lambda _: command))
    action = cfg.actions["joint_pos"].build(env)
    assert action.action_dim == 48
    assert tuple(action.target_names) == bundle.joint_names
    np.testing.assert_array_equal(action.target_ids.numpy(), np.arange(48))
    action.process_actions(torch.full((2, 48), 3.))
    torch.testing.assert_close(action.raw_action, torch.ones(2, 48))
    torch.testing.assert_close(action._processed_actions, torch.full((2, 48), .25))


def test_full_scene_sensor_slots_match_native_contact_layout(converted, monkeypatch):
    from mjlab.scene import Scene

    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("No dynamics in CPU preflight"))
    bundle = load_runtime_bundle(converted)
    scene = Scene(build_human_env_cfg(bundle).scene, device="cpu")
    model = scene.compile()
    sensor = scene["keybodies_ground_contact"]
    assert sensor.primary_names == list(bundle.body_names)
    assert sensor.cfg.num_slots == 1
    for slot in sensor._slots:
        expected = 3 if slot.field_name == "force" else 1
        assert model.sensor(slot.sensor_name).dim[0] == expected
    assert tuple(scene["robot"].joint_names) == bundle.joint_names
    assert tuple(scene["robot"].body_names) == bundle.body_names


def test_hash_only_training_claim_is_rejected(converted, tmp_path):
    import shutil
    root = tmp_path / "tampered"
    shutil.copytree(converted, root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["qualified_for_training"] = True
    for clip in manifest["clips"]:
        clip["requested_ttc_s"] = .5
        clip["launch_time_s"] = 0.
        clip["launch_features"] = [0.] * 77
        clip["g1_snapshot_present"] = True
        clip["snapshot_fingerprint"] = "test_only"
        clip["guard_source"] = "explicit_common_guard"
        clip["common_guard_validated"] = True
    fake = root / "qualification.json"
    fake.write_text("{}")
    manifest["qualification_evidence"] = {"path": fake.name, "sha256": sha256(fake)}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="qualification report"):
        load_runtime_bundle(root).require_training_qualification()


@pytest.mark.parametrize("tamper", ["master_names", "view_values"])
def test_master_and_lower_rate_view_cannot_disagree(converted, tmp_path, tamper):
    import shutil
    root = tmp_path / tamper
    shutil.copytree(converted, root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    clip = manifest["clips"][0]
    field = "master" if tamper == "master_names" else "motion"
    path = root / clip[f"{field}_path"]
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    if tamper == "master_names":
        arrays["joint_names"][[0, 1]] = arrays["joint_names"][[1, 0]]
    else:
        arrays["joint_pos"][1, 0] += .1
    np.savez_compressed(path, **arrays)
    clip[f"{field}_sha256"] = sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Master reference joint order|differs from its 200 Hz"):
        load_runtime_bundle(root)


def test_training_rejects_nonlaunch_clock_and_family_guard(converted):
    from copy import deepcopy
    from dataclasses import replace

    bundle = load_runtime_bundle(converted)
    clips = deepcopy(bundle.clips)
    for clip in clips:
        clip.update(requested_ttc_s=.5, launch_features=[0.] * 77,
                    g1_snapshot_present=True, snapshot_fingerprint="test_only")
    candidate = replace(bundle, clips=clips,
                        manifest={**bundle.manifest, "qualified_for_training": True})
    with pytest.raises(ValueError, match="clock must start"):
        candidate.require_training_qualification()
    for clip in clips:
        clip["launch_time_s"] = 0.
    with pytest.raises(ValueError, match="admitted common guard"):
        candidate.require_training_qualification()


def test_targeting_reward_uses_geometry_timestamp():
    from human_punching.wbc.runtime.mdp import timed_target_reward

    command = SimpleNamespace(trajectory_ids=torch.tensor([0]), family_ids=torch.tensor([0]),
        target_w=torch.zeros(1, 3), elapsed=torch.tensor([.505]),
        launch_times=torch.tensor([0.]), ttcs=torch.tensor([.5]))
    robot = SimpleNamespace(find_sites=lambda *a, **kw: ([0, 1], []),
                            data=SimpleNamespace(site_pos_w=torch.zeros(1, 2, 3)))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command),
                          scene={"robot": robot}, num_envs=1, device="cpu", physics_dt=.005)
    assert timed_target_reward(env).item() == pytest.approx(1.)


def clock_command():
    command = PunchMotionCommand.__new__(PunchMotionCommand)
    command._env = SimpleNamespace(device="cpu", num_envs=1,
                                   scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))
    command.physics_steps = torch.zeros(1, dtype=torch.long)
    command.master_starts = torch.tensor([0])
    command.master_lengths = torch.tensor([9])
    command.trajectory_ids = torch.zeros(1, dtype=torch.long)
    command._tracked_joint_ids = None
    command.master = {"qpos": torch.zeros(9, 55), "qvel": torch.ones(9, 54),
                      "joint_feedforward": torch.ones(9, 48) * 3.,
                      "body_lin_vel_w": torch.ones(9, 1, 3),
                      "body_ang_vel_w": torch.ones(9, 1, 3),
                      "anchor_lin_acc_w": torch.ones(9, 3),
                      "anchor_ang_acc_w": torch.ones(9, 3)}
    command.master["qpos"][:, 7:] = torch.arange(9)[:, None] * .01
    command.motion = SimpleNamespace(segment_start_idx=torch.tensor([0]),
                                     segment_length=torch.tensor([3]))
    command.time_steps = torch.zeros(1, dtype=torch.long)
    command.update_relative_body_poses = lambda: None
    return command


def test_four_physics_reference_samples_per_held_action():
    cmd = clock_command()
    action = PunchJointPositionAction.__new__(PunchJointPositionAction)
    action._command = cmd
    action._target_ids = torch.arange(48)
    action._processed_actions = torch.ones(1, 48) * .25
    samples = []
    action._entity = SimpleNamespace(
        data=SimpleNamespace(joint_pos_limits=torch.tensor([[[-.2, .2]] * 48])),
        set_joint_position_target=lambda value, **kw: samples.append(value.clone()),
        set_joint_velocity_target=lambda value, **kw: None,
        set_joint_effort_target=lambda value, **kw: None,
    )
    action._processed_actions.zero_()
    for _ in range(4):
        action.apply_actions()
    np.testing.assert_allclose([value[0, 0] for value in samples], [0., .01, .02, .03], atol=1e-8)
    assert cmd.physics_steps.item() == 4
    cmd._update_command(None)
    assert cmd.physics_steps.item() == 4 and cmd.time_steps.item() == 1
    action._processed_actions.fill_(.25)
    action.apply_actions()
    assert samples[-1].max().item() == pytest.approx(.2)


def test_clip_end_holds_reference_and_never_resets_state():
    cmd = clock_command()
    cmd.physics_steps.fill_(100)
    cmd._update_command(None)
    assert cmd.time_steps.item() == 2
    assert cmd.joint_pos[0, 0].item() == pytest.approx(.08)
    assert torch.count_nonzero(cmd.joint_vel) == 0
    assert torch.count_nonzero(cmd.body_lin_vel_w) == 0
    assert torch.count_nonzero(cmd.body_ang_vel_w) == 0
    assert torch.count_nonzero(cmd.anchor_lin_acc_w) == 0
    assert torch.count_nonzero(cmd.anchor_ang_acc_w) == 0
    assert cmd.physics_steps.item() == 100
    with pytest.raises(RuntimeError, match="episode reset"):
        cmd._resample_command(torch.tensor([0]))


def test_reference_plan_does_not_realign_to_live_human():
    command = PunchMotionCommand.__new__(PunchMotionCommand)
    # Overwrite only master lookup. A current-human pose cannot enter the
    # committed pose transform because update_relative_body_poses never reads it.
    command._env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))
    command.master = {}
    command.sample_master = lambda name: (torch.tensor([[[1., 2., 3.]]])
                                          if name == "body_pos_w"
                                          else torch.tensor([[[1., 0., 0., 0.]]]))
    command.body_pos_relative_w = torch.zeros(1, 1, 3)
    command.body_quat_relative_w = torch.zeros(1, 1, 4)
    command.update_relative_body_poses()
    np.testing.assert_array_equal(command.body_pos_relative_w, [[[1., 2., 3.]]])


def test_actual_wbc_loader_reset_and_clip_boundary(converted):
    bundle = load_runtime_bundle(converted)
    cfg = build_human_env_cfg(bundle, num_envs=128).commands["motion"]
    writes = []
    robot = SimpleNamespace(
        joint_names=bundle.joint_names, body_names=bundle.body_names,
        find_joints=lambda names, **kw: ([bundle.joint_names.index(n) for n in names], names),
        find_bodies=lambda names, **kw: ([bundle.body_names.index(n) for n in names], names),
        data=SimpleNamespace(soft_joint_pos_limits=torch.tensor(bundle.body["joint_limits_rad"])
                             .unsqueeze(0).repeat(128, 1, 1)),
        write_joint_state_to_sim=lambda *args, **kw: writes.append("joint"),
        write_root_state_to_sim=lambda *args, **kw: writes.append("root"),
        reset=lambda **kw: None,
    )

    class Scene(dict):
        env_origins = torch.zeros(128, 3)

    env = SimpleNamespace(device="cpu", num_envs=128, step_dt=.02, scene=Scene(robot=robot))
    command = PunchMotionCommand(cfg, env)
    torch.manual_seed(17)
    command.reset(torch.arange(128))
    assert writes == ["joint", "root"]
    assert set(command.family_ids[command.trajectory_ids].tolist()) == {0, 1, 2, 3}
    command.physics_steps.fill_(10_000)
    command._update_command(None)
    assert writes == ["joint", "root"]
    assert torch.all(command.time_steps < command.motion.segment_end_idx[command.trajectory_ids])
    with pytest.raises(RuntimeError, match="pose replacement"):
        command.reset_to_frame(torch.tensor([0]), 20)


def test_entity_defaults_real_wbc_loader_and_actor_share_float32(converted, monkeypatch):
    """Reproduce the first-forward failure without constructing a simulator."""
    from mjlab.envs.mdp.observations import joint_pos_rel
    from mjlab.utils.string import resolve_expr
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    from wbc_mjlab.env.mdp.commands import MotionLoader

    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("No CPU dynamics"))
    bundle = load_runtime_bundle(converted)
    master_path = bundle.path(bundle.clips[0]["master_path"])
    master_hash = sha256(master_path)
    with np.load(master_path) as archive:
        master_dtype = archive["qpos"].dtype
        initial = archive["qpos"][0, 7:]
    # This is the previous bug: mjlab's actual default-state construction
    # preserves NumPy scalar dtype, promoting float32 physical joint positions.
    historical_values = dict(zip(bundle.joint_names, initial, strict=True))
    assert torch.tensor(resolve_expr(historical_values, bundle.joint_names, 0.)).dtype == torch.float64

    cfg = build_human_env_cfg(bundle)
    entity = cfg.scene.entities["robot"].build()
    assert all(type(value) is float for value in entity.cfg.init_state.joint_pos.values())
    assert all(type(value) is float for value in (*entity.cfg.init_state.pos, *entity.cfg.init_state.rot))
    default = torch.tensor(resolve_expr(entity.cfg.init_state.joint_pos, entity.joint_names, 0.))[None]
    assert default.dtype == torch.float32
    assert entity._build_default_root_state(1, "cpu").dtype == torch.float32

    loader = MotionLoader(str(bundle.root/"motions"), torch.arange(len(bundle.body_names)), 0, .020)
    for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
                 "body_ang_vel_w", "anchor_lin_acc_w", "anchor_ang_acc_w"):
        assert getattr(loader, name).dtype == torch.float32
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
        default_joint_pos=default, joint_pos=loader.joint_pos[:1].clone()))})
    relative = joint_pos_rel(env)
    assert relative.dtype == torch.float32
    value = torch.cat((relative, loader.joint_pos[:1], loader.joint_vel[:1]), dim=-1)
    observations = TensorDict({"actor": value}, batch_size=[1])
    actor = MLPModel(observations, {"actor": ["actor"]}, "actor", output_dim=48,
                     hidden_dims=(32, 16), activation="elu", obs_normalization=True)
    with torch.no_grad():
        action = actor(observations)
    assert action.dtype == torch.float32 and torch.isfinite(action).all()
    assert sha256(master_path) == master_hash
    with np.load(master_path) as archive:
        assert archive["qpos"].dtype == master_dtype == np.float64


def test_observation_dtype_gate_reports_term_without_casting_or_history_update():
    from human_punching.wbc.runtime.runner import require_float32_observations
    calls = []
    values = {"actor": torch.zeros(2, 3, dtype=torch.float64), "critic": torch.zeros(2, 4)}
    manager = SimpleNamespace(
        compute=lambda **kw: calls.append(kw) or values,
        active_terms={"actor": ["joint_pos"]},
        get_term_cfg=lambda *args: SimpleNamespace(func=lambda env: values["actor"], params={}))
    env = SimpleNamespace(observation_manager=manager)
    with pytest.raises(TypeError, match="joint_pos.*torch.float64"):
        require_float32_observations(env)
    assert values["actor"].dtype == torch.float64
    assert calls == [{"update_history": False}]
    values["actor"] = values["actor"].float()
    assert require_float32_observations(env)["actor"] == {"dtype": "torch.float32", "shape": [2, 3]}
    values["critic"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="critic.*nonfinite"):
        require_float32_observations(env)


def test_untrained_native_rsl_normalized_actor_exports_and_matches(tmp_path):
    """Export plumbing only: no optimizer, training, physical execution or policy claim."""
    import onnx
    from onnx.reference import ReferenceEvaluator
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    from human_punching.wbc.runtime.runner import HumanMotionTrackingRunner

    torch.manual_seed(91)
    observations = TensorDict({"actor": torch.randn(64, 16) * 3 + 2}, batch_size=[64])
    actor = MLPModel(observations, {"actor": ["actor"]}, "actor", output_dim=48,
                     hidden_dims=(32, 16), activation="elu", obs_normalization=True,
                     distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.,
                                       "std_type": "scalar"})
    actor.update_normalization(observations)
    actor.eval()
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    runner = HumanMotionTrackingRunner.__new__(HumanMotionTrackingRunner)
    runner.alg = SimpleNamespace(get_policy=lambda: actor)
    runner.export_policy_to_onnx(str(tmp_path), "untrained_smoke.onnx")
    runner.export_policy_to_jit(str(tmp_path), "untrained_smoke_jit.pt")
    portable = torch.jit.load(str(tmp_path / "untrained_smoke_jit.pt"), map_location="cpu").eval()
    graph = onnx.load(tmp_path / "untrained_smoke.onnx")
    onnx.checker.check_model(graph)
    assert {"Sub", "Div"} <= {node.op_type for node in graph.graph.node}
    assert any("obs_normalizer" in value.name for value in graph.graph.initializer)
    evaluator = ReferenceEvaluator(graph)
    test_inputs = torch.randn(8, 16) * 2
    for row in test_inputs:
        batch = row.unsqueeze(0)
        with torch.no_grad():
            expected = actor(TensorDict({"actor": batch}, batch_size=[1])).numpy()
        actual = evaluator.run(None, {graph.graph.input[0].name: batch.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        with torch.no_grad():
            np.testing.assert_allclose(portable(batch).numpy(), expected, atol=1e-5, rtol=1e-5)
    assert all(torch.equal(value, actor.state_dict()[key]) for key, value in before.items())
