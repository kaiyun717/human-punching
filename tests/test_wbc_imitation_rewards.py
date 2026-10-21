"""Fresh geometry with the actual pinned WBC kernels; no physics integration."""
from copy import deepcopy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from human_punching.wbc.runtime import imitation_rewards as m
from human_punching.wbc.runtime.task import build_human_env_cfg
from human_punching.wbc.telemetry.runtime_telemetry import ObserverOptions, configure_observer
from wbc_mjlab.env.mdp import rewards


@pytest.fixture(scope="module")
def physical(task10_common_bundle):
    bundle = task10_common_bundle
    model = mujoco.MjModel.from_xml_path(str(bundle.model_path))
    with np.load(bundle.path(bundle.clips[0]["master_path"])) as master:
        nominal = master["qpos"][0].copy()
    names = tuple(bundle.body_names)
    body_ids = [int(model.body(name).id) for name in names]
    rows, expected = [], []
    generator = np.random.default_rng(10010)
    for index in range(3):
        data = mujoco.MjData(model)
        data.qpos[:] = nominal
        data.qpos[:3] += index * np.array([2., -1., 0.])
        data.qpos[7:] += generator.uniform(-.01, .01, 48)
        data.qvel[:] = generator.uniform(-.4, .4, 54)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        rows.append({name: getattr(data, name).copy() for name in ("xpos", "xquat", "cvel", "subtree_com")})
        linear, angular = [], []
        for body in body_ids:
            jacobian_position, jacobian_rotation = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacobian_position, jacobian_rotation, body)
            linear.append(jacobian_position @ data.qvel)
            angular.append(jacobian_rotation @ data.qvel)
        expected.append({"position": data.xpos[body_ids].copy(), "quaternion": data.xquat[body_ids].copy(),
            "linear_velocity": np.array(linear), "angular_velocity": np.array(angular)})
    arrays = {key: torch.tensor(np.stack([row[key] for row in rows]), dtype=torch.float32) for key in rows[0]}
    expected = {key: torch.tensor(np.stack([row[key] for row in expected]), dtype=torch.float32) for key in expected[0]}
    return model, names, arrays, expected


def environment(physical):
    model, names, arrays, expected = physical
    arrays = {key: value.clone() for key, value in arrays.items()}
    anchor = names.index("Hips")
    command = SimpleNamespace(cfg=SimpleNamespace(body_names=names, anchor_body_name="Hips"),
        body_pos_relative_w=expected["position"], body_quat_relative_w=expected["quaternion"],
        body_lin_vel_w=expected["linear_velocity"], body_ang_vel_w=expected["angular_velocity"],
        anchor_pos_w=expected["position"][:, anchor], anchor_quat_w=expected["quaternion"][:, anchor],
        anchor_lin_vel_w=expected["linear_velocity"][:, anchor], anchor_ang_vel_w=expected["angular_velocity"][:, anchor],
        marker="original-command")
    observer = SimpleNamespace(actual_samples=4, complete=True,
        fk=SimpleNamespace(_arrays=arrays, bindings=SimpleNamespace(spec=SimpleNamespace(prefix=""))))
    def window():
        if not observer.complete:
            raise RuntimeError("Window requires four actual physics intervals")
        return {}
    observer.window_summary = window
    env = SimpleNamespace(device="cpu", sim=SimpleNamespace(mj_model=model),
        command_manager=SimpleNamespace(get_term=lambda name: command))
    return env, observer, arrays, command


def test_all_named_body_origins_and_twists_match_independent_cpu_jacobians(physical):
    model, names, arrays, expected = physical
    ids = torch.tensor([int(model.body(name).id) for name in names])
    trees = torch.tensor([int(model.body_rootid[i]) for i in ids])
    result = m.body_endpoint(arrays, ids, trees)
    for name, value in expected.items():
        torch.testing.assert_close(result[name], value, rtol=2e-6, atol=2e-6)
    # cvel's raw linear components must not be mistaken for body-origin velocity.
    assert (result["linear_velocity"] - arrays["cvel"][:, ids, 3:]).abs().max() > .1


@pytest.mark.parametrize("name", m.UPSTREAM_NAMES)
def test_real_wbc_kernel_sees_fresh_geometry_without_mutating_state(physical, monkeypatch, name):
    env, observer, arrays, command = environment(physical)
    monkeypatch.setattr(m, "get_observer", lambda _: observer)
    original = {key: value.clone() for key, value in arrays.items()}
    func = getattr(rewards, name)
    kwargs = {"command_name": "motion", "std": .2}
    actual = m.fresh_imitation_reward(env, upstream_reward=func, **kwargs)
    torch.testing.assert_close(actual, torch.ones(3), rtol=1e-6, atol=1e-6)
    for key, value in original.items():
        torch.testing.assert_close(arrays[key], value, rtol=0, atol=0)
    assert getattr(env, m.CACHE_ATTRIBUTE).get().get_term("motion").marker == command.marker
    with pytest.raises(ValueError, match="motion command"):
        getattr(env, m.CACHE_ATTRIBUTE).get().get_term("different-entity")


def test_cached_geometry_updates_once_per_complete_window_and_never_updates_fk(physical):
    env, observer, arrays, _ = environment(physical)
    cache = m.FreshImitationCache(env, observer)
    first = cache.get()
    before = first.get_term("motion").robot_body_pos_w.clone()
    arrays["xpos"].add_(.03)  # Synthetic replacement of private endpoint buffer.
    assert cache.get() is first
    torch.testing.assert_close(first.get_term("motion").robot_body_pos_w, before)
    observer.actual_samples += 4
    second = cache.get()
    torch.testing.assert_close(second.get_term("motion").robot_body_pos_w, before + .03)
    observer.complete = False
    with pytest.raises(RuntimeError, match="four actual"):
        cache.get()


def test_raw_nonfinite_reward_remains_visible_before_manager_sanitization(physical, monkeypatch):
    env, observer, _, command = environment(physical)
    monkeypatch.setattr(m, "get_observer", lambda _: observer)
    assert m.fresh_imitation_health(env)["observed_windows"] == 0
    for name in m.UPSTREAM_NAMES:
        m.fresh_imitation_reward(env, upstream_reward=getattr(rewards, name), command_name="motion", std=.2)
    health = m.fresh_imitation_health(env)
    assert health["observed_windows"] == 1 and health["all_raw_terms_finite"]
    assert health["every_term_each_window"]
    observer.actual_samples += 4
    broken_name = m.UPSTREAM_NAMES[0]
    command.anchor_pos_w = torch.full_like(command.anchor_pos_w, float("nan"))
    raw = m.fresh_imitation_reward(env, upstream_reward=getattr(rewards, broken_name),
                                 command_name="motion", std=.2)
    assert torch.isnan(raw).all()  # Explicit corrupt-reference fault injection.
    assert torch.isfinite(torch.nan_to_num(raw)).all()  # Manager sanitization hides this.
    cache = getattr(env, m.CACHE_ATTRIBUTE)
    assert torch.isnan(cache.raw_values[broken_name]).all()
    raw.zero_()
    assert torch.isnan(cache.raw_values[broken_name]).all()  # Retained evidence owns its storage.
    for name in m.UPSTREAM_NAMES[1:]:
        m.fresh_imitation_reward(env, upstream_reward=getattr(rewards, name), command_name="motion", std=.2)
    health = m.fresh_imitation_health(env)
    assert health["observed_windows"] == 2 and health["every_term_each_window"]
    assert not health["all_raw_terms_finite"]
    assert health["term_nonfinite_counts"][broken_name] == 3
    assert all(count == 2 for count in health["term_call_counts"].values())
    with pytest.raises(RuntimeError, match="twice"):
        m.fresh_imitation_reward(env, upstream_reward=getattr(rewards, broken_name),
                                 command_name="motion", std=.2)


def test_actual_configuration_retains_kernels_parameters_weights_and_term_names(task10_common_bundle):
    cfg = build_human_env_cfg(task10_common_bundle, num_envs=16, evaluation=True)
    before = deepcopy(cfg.rewards)
    with pytest.raises(ValueError, match="native observer"):
        m.configure_fresh_imitation_rewards(cfg)
    configure_observer(cfg, ObserverOptions(enabled=True))
    contract = m.configure_fresh_imitation_rewards(cfg)
    m.require_fresh_imitation_rewards(cfg)
    assert list(cfg.rewards) == list(before)
    assert len(contract["terms"]) == 8 and not contract["extra_integration_or_forward"]
    for name, term in cfg.rewards.items():
        assert term.weight == before[name].weight
        if name in contract["terms"]:
            assert term.func is m.fresh_imitation_reward
            assert term.params["upstream_reward"] is before[name].func
            assert {key: value for key, value in term.params.items() if key != "upstream_reward"} == before[name].params
        else:
            assert term.func is before[name].func and term.params == before[name].params
    cfg.rewards[contract["terms"][0]] = before[contract["terms"][0]]
    with pytest.raises(ValueError, match="eight WBC"):
        m.require_fresh_imitation_rewards(cfg)
