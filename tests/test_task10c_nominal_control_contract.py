"""Focused contracts for the paired single-command diagnostic experiment."""
from copy import deepcopy

import pytest

from human_punching.task10c.env_cfg import make_env_cfg
from human_punching.task10c.evaluate import (
    _digest, create_nominal_control_recipes, reference_env_kwargs,
    resolve_reference_contract, validate_recipes, validate_reference_resume,
    verify_reference_contract,
)
from human_punching.task10c.robot import ROOT


def contracts():
    return [resolve_reference_contract(ROOT / f'configs/task10c/jab_nominal_control_{name}_20261022.json')
            for name in ('a_relative', 'b_launch_registered')]


def test_pair_changes_only_imitation_placement_and_removes_task_reward_terms():
    a, b = contracts()
    assert {k for k in a if a[k] != b[k]} == {'imitation_frame'}
    for contract in (a, b):
        cfg = make_env_cfg(num_envs=1, **reference_env_kwargs(contract))
        assert not {'encounter', 'interaction'} & cfg.rewards.keys()
        assert set(cfg.rewards) == {'anchor_height', 'anchor_orientation', 'body_position',
            'body_orientation', 'body_linear_velocity', 'body_angular_velocity',
            'command_excess', 'action_rate', 'effort', 'joint_margin'}
        c = cfg.commands['punch']
        assert (c.imitation_budget, c.recovery_budget, c.command_excess_weight) == (7., 6., 1.)
        assert c.recovery_fraction == .2
        assert c.fixed_target_snapshot_id == 53874
        assert c.ttc_range == c.easy_ttc_range == (.7, .7)
        assert c.easy_fraction == .5
        assert cfg.metrics['encounter_refresh'].per_substep
    with pytest.raises(ValueError, match='contract changed'):
        validate_reference_resume({'reference_contract': a}, b)


def test_nominal_recipe_keeps_real_snapshot_and_explicit_diagnostic_scope():
    path = ROOT / 'outputs/task10c/setup/target_pool.npz'
    recipes = create_nominal_control_recipes(path, 4, 112223)
    validate_recipes(recipes, path)
    assert recipes['split'] == 'train'
    assert len({tuple(r['target_world']) for r in recipes['rows']}) == 1
    assert {r['snapshot_id'] for r in recipes['rows']} == {53874}
    assert {r['requested_ttc'] for r in recipes['rows']} == {.7}
    invalid = deepcopy(recipes)
    invalid['rows'][0]['target_world'][0] -= .01
    invalid['sha256'] = _digest({k: v for k, v in invalid.items() if k != 'sha256'})
    with pytest.raises(ValueError):
        validate_recipes(invalid, path)


def test_nominal_contract_does_not_silently_reintroduce_task_rewards_or_vary_commands():
    a, _ = contracts()
    for change in ({'task_rewards_enabled': True}, {'fixed_target_snapshot_id': 53875},
                   {'ttc_range': [.5, .9]}, {'easy_ttc_range': [.555, .6783333333333333]}):
        with pytest.raises(ValueError):
            verify_reference_contract({**a, **change})
