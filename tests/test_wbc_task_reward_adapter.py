from types import SimpleNamespace

import pytest
import torch

from human_punching.wbc.runtime.task_reward_adapter import RewardCache, configure_task_rewards, native_task_reward
from human_punching.wbc.runtime.task_rewards import DEFAULT_WEIGHTS, task_reward_rates
from test_wbc_task_rewards import fixture


def test_shared_reward_cache_uses_actual_samples_and_preserves_event_bits():
    current, goal, window, context, caps = fixture()
    context["goal"] = goal
    context["new_recovery"][0] = True
    calls = []
    complete = [True]

    def checked_window():
        if not complete[0]:
            raise RuntimeError("Four physical intervals required")
        return window

    def provider():
        calls.append(1)
        return context

    observer = SimpleNamespace(actual_samples=4, window=SimpleNamespace(caps=caps),
        window_summary=checked_window, endpoint_state=lambda: current)
    cache = RewardCache(observer, provider)
    first = cache.values()
    for name in DEFAULT_WEIGHTS:
        assert cache.values()[name] is first[name]
    assert len(calls) == 1
    assert context["new_recovery"].tolist() == [True, False]
    # Even an already cached value cannot escape a half-completed new window.
    complete[0] = False
    with pytest.raises(RuntimeError, match="intervals"):
        cache.values()
    complete[0] = True
    observer.actual_samples = 8
    context["new_recovery"].zero_()
    current["root_position_w"][0, 0] = .1
    second = cache.values()
    assert len(calls) == 2
    assert second["guard_root_position"][0] < first["guard_root_position"][0]
    assert not second["guard_recovered_bonus"].any()
    assert first["guard_recovered_bonus"][0] == 50.


def configuration():
    return SimpleNamespace(decimation=4, sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=.005)),
        scale_rewards_by_dt=True, metrics={"human_native_substep_valid": object()},
        rewards={"whole_body_tracking": object(), "timed_target": object(),
                 "foot_slip": SimpleNamespace(weight=-.1)})


def test_configuration_replaces_stale_terms_and_keeps_wbc_imitation():
    cfg = configuration()
    tracking = cfg.rewards["whole_body_tracking"]
    result = configure_task_rewards(cfg)
    assert result is cfg
    assert cfg.rewards["whole_body_tracking"] is tracking
    assert "timed_target" not in cfg.rewards
    assert cfg.rewards["foot_slip"].weight == 0.
    assert all(cfg.rewards["human_"+name].weight == weight for name, weight in DEFAULT_WEIGHTS.items())
    assert all(cfg.rewards["human_"+name].params == {"name": name} for name in DEFAULT_WEIGHTS)
    with pytest.raises(ValueError, match="already"):
        configure_task_rewards(cfg)


@pytest.mark.parametrize("change", ["dt", "decimation", "integration", "observer"])
def test_configuration_requires_actual_clocks_and_native_observer(change):
    cfg = configuration()
    if change == "dt":
        cfg.sim.mujoco.timestep = .0025
    elif change == "decimation":
        cfg.decimation = 8
    elif change == "integration":
        cfg.scale_rewards_by_dt = False
    else:
        cfg.metrics.clear()
    with pytest.raises(ValueError):
        configure_task_rewards(cfg)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), True])
def test_bad_weight_rejected_before_mutating_configuration(weight):
    cfg = configuration()
    weights = dict(DEFAULT_WEIGHTS, guard_quiet=weight)
    with pytest.raises(ValueError, match="Finite"):
        configure_task_rewards(cfg, weights)
    assert "timed_target" in cfg.rewards
    assert cfg.rewards["foot_slip"].weight == -.1


def native_fixture():
    args = fixture()
    current, goal, window, context, caps = args
    current["joint_position"][0, 0] = .18
    current["joint_position"][1] = .18 / 3**.5
    context["goal"] = goal
    calls = []

    def provider():
        calls.append(1)
        return context

    observer = SimpleNamespace(actual_samples=4, window=SimpleNamespace(caps=caps),
        window_summary=lambda: window, endpoint_state=lambda: current)
    bridge = SimpleNamespace(reward_context=provider)
    env = SimpleNamespace(num_envs=2, device="cpu", _human_cycle_runtime=bridge,
                          _human_substep_observer=observer)
    return env, args, calls


@pytest.mark.parametrize("metric", ["rms", "maximum"])
def test_native_reward_manager_propagates_selected_metric_and_shares_one_cache(metric):
    from mjlab.managers.reward_manager import RewardManager
    env, args, calls = native_fixture()
    cfg = configure_task_rewards(configuration(), guard_joint_posture_metric=metric)
    terms = {name: value for name, value in cfg.rewards.items() if name.startswith("human_")}
    manager = RewardManager(terms, env)
    actual = manager.compute(.020)
    expected_rates = task_reward_rates(*args, guard_joint_posture_metric=metric)
    expected = sum(DEFAULT_WEIGHTS[name] * value * .020 for name, value in expected_rates.items())
    torch.testing.assert_close(actual, expected)
    assert len(calls) == 1
    cache = env._human_cycle_runtime._task_reward_cache
    assert cache.guard_joint_posture_metric == metric
    # Reading again does not consume events or recalculate the same window.
    assert native_task_reward(env, "guard_joint_posture", guard_joint_posture_metric=metric) is cache.values()["guard_joint_posture"]
    assert len(calls) == 1
    env._human_substep_observer.actual_samples = 8
    args[3]["guard_mask"][0] = False
    manager.compute(.020)
    assert len(calls) == 2
    assert cache.values()["guard_joint_posture"][0] == 0
    assert cache.values()["guard_joint_posture"][1] > 0


@pytest.mark.parametrize("first,conflicting", [("rms", "maximum"), ("maximum", "rms")])
def test_native_cache_rejects_metric_conflicts_without_replacing_values(first, conflicting):
    env, _, calls = native_fixture()
    original = native_task_reward(env, "guard_joint_posture", guard_joint_posture_metric=first)
    cache = env._human_cycle_runtime._task_reward_cache
    with pytest.raises(ValueError, match="Conflicting"):
        native_task_reward(env, "guard_root_position", guard_joint_posture_metric=conflicting)
    assert env._human_cycle_runtime._task_reward_cache is cache
    assert cache.values()["guard_joint_posture"] is original
    assert len(calls) == 1
    with pytest.raises(AttributeError):
        cache.guard_joint_posture_metric = conflicting


def test_maximum_metric_is_explicit_in_all_term_params_and_configuration_digest():
    from human_punching.wbc.measurement_contract import configuration_value, digest
    legacy = configure_task_rewards(configuration())
    explicit_rms = configure_task_rewards(configuration(), guard_joint_posture_metric="rms")
    maximum = configure_task_rewards(configuration(), guard_joint_posture_metric="maximum")
    for name in DEFAULT_WEIGHTS:
        old, same, new = (cfg.rewards["human_"+name] for cfg in (legacy, explicit_rms, maximum))
        assert old.params == same.params == {"name": name}
        assert new.params == {"name": name, "guard_joint_posture_metric": "maximum"}
        assert old.weight == same.weight == new.weight == DEFAULT_WEIGHTS[name]
        assert digest(configuration_value(old)) == digest(configuration_value(same))
        assert digest(configuration_value(old)) != digest(configuration_value(new))


@pytest.mark.parametrize("metric", [None, True, 1, "max", "mean", []])
def test_invalid_metric_rejected_before_configuration_or_cache_mutation(metric):
    cfg = configuration()
    with pytest.raises(ValueError, match="posture metric"):
        configure_task_rewards(cfg, guard_joint_posture_metric=metric)
    assert "timed_target" in cfg.rewards
    assert cfg.rewards["foot_slip"].weight == -.1
    with pytest.raises(ValueError, match="posture metric"):
        RewardCache(None, None, guard_joint_posture_metric=metric)
    env, _, calls = native_fixture()
    with pytest.raises(ValueError, match="posture metric"):
        native_task_reward(env, "guard_joint_posture", guard_joint_posture_metric=metric)
    assert not hasattr(env._human_cycle_runtime, "_task_reward_cache")
    assert calls == []
