"""Phase 2 learning regressions without GPU simulation or learning claims."""
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from human_punching.task10c.events import PunchState, physics_event_callback
from human_punching.task10c.targets import target_subset_indices


def _state(n=1):
    state = PunchState(n, 'cpu')
    state.reset(torch.arange(n), torch.zeros(n, 3), torch.full((n,), 0.6),
                torch.zeros(n), torch.ones(n, dtype=torch.bool))
    return state


def test_central_target_curriculum_retains_original_points_and_height():
    points = np.array([[2., 0., 1.21], [0., 0., 1.23], [-2., 0., 1.22],
                       [0., 1., 1.24], [0., -1., 1.20]])
    original = points.copy()
    narrow = target_subset_indices(points, 0.4)
    broad = target_subset_indices(points, 0.8)
    assert narrow.tolist() == [1, 3]  # Stable ordering of equal XY radii.
    assert broad[:len(narrow)].tolist() == narrow.tolist()
    assert len(target_subset_indices(points, 1.0)) == len(points)
    np.testing.assert_array_equal(points, original)
    np.testing.assert_array_equal(points[narrow, 2], original[narrow, 2])


@pytest.mark.parametrize('fraction', [0, -0.1, 1.1, float('nan')])
def test_target_curriculum_rejects_empty_or_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        target_subset_indices(np.ones((4, 3)), fraction)


def test_curriculum_update_preserves_active_world_goal_and_clock():
    from human_punching.task10c.commands import PunchCommand

    points = np.array([[0., 0., 1.22], [0.1, 0., 1.23], [-0.1, 0., 1.21],
                       [0., 0.1, 1.20]])
    c = SimpleNamespace(reference=SimpleNamespace(strike_time=0.6), device='cpu',
                        cfg=SimpleNamespace(target_split='train', family='jab'),
                        pool=SimpleNamespace(points=lambda *_: points.copy()), state=_state())
    c.state.physics_steps[:] = 71
    target, requested, elapsed = (c.state.target_world.clone(),
                                  c.state.requested_ttc.clone(), c.state.elapsed.clone())
    PunchCommand.set_curriculum(c, {'stage': 1, 'target_fraction': 1.0,
                                   'ttc_range': [0.5, 1.0], 'easy_fraction': 0.5})
    torch.testing.assert_close(c.state.target_world, target)
    torch.testing.assert_close(c.state.requested_ttc, requested)
    torch.testing.assert_close(c.state.elapsed, elapsed)
    assert len(c.target_candidates) == 4 and len(c.easy_target_candidates) == 1
    assert c.cfg.ttc_range == (0.5, 1.0)


def _reset_command():
    """Exercise the real command/reset sampler against inert state writers."""
    from human_punching.task10c.commands import PunchCommand

    n = 4
    zeros = lambda *shape: torch.zeros(*shape)
    action = SimpleNamespace(target_ids=list(range(31)), offset=zeros(n, 31),
                             scale=torch.ones(n, 31), _raw_actions=zeros(n, 31))
    env = SimpleNamespace(step_dt=0.02, scene=SimpleNamespace(env_origins=torch.tensor(
        [[0., 0., 0.], [4., 0., 0.], [8., 0., 0.], [12., 0., 0.]])),
        action_manager=SimpleNamespace(get_term=lambda _: action, action=zeros(n, 31),
                                       prev_action=zeros(n, 31), prev_prev_action=zeros(n, 31)))
    robot = SimpleNamespace(write_root_state_to_sim=lambda *_: None,
                            write_joint_state_to_sim=lambda *_, **__: None,
                            set_joint_position_target=lambda *_, **__: None)
    sample = {'root_pos_w': zeros(n, 3),
              'root_quat_w': torch.tensor([1., 0., 0., 0.]).expand(n, 4),
              'body_lin_vel_w': zeros(n, 1, 3), 'body_ang_vel_w': zeros(n, 1, 3),
              'joint_pos': torch.full((n, 31), 0.2), 'joint_vel': zeros(n, 31)}
    c = SimpleNamespace(device='cpu', _env=env, robot=robot, root_index=0,
        cfg=SimpleNamespace(target_split='validation', family='jab', recovery_fraction=1.0),
        targets=torch.tensor([[0., 0., 1.21], [.1, 0., 1.22], [.2, 0., 1.23]]),
        target_index=torch.zeros(n, dtype=torch.long), state=_state(n),
        reference=SimpleNamespace(recovery=0.8), ref=sample,
        refresh_reference=lambda: None, horizon=torch.ones(n),
        interaction_z=torch.ones(n), guard_z=torch.ones(n),
        pending_launch=torch.zeros(n, dtype=torch.bool), ready=torch.zeros(n, dtype=torch.bool),
        steps_seen=torch.zeros(n, dtype=torch.long), settled_duration=zeros(n),
        first_settled_elapsed=torch.full((n,), float('nan')), component_totals={},
        reset_counts={'full': 0, 'snippet': 0}, evaluation_recipes=None)
    c.set_evaluation_recipes = MethodType(PunchCommand.set_evaluation_recipes, c)
    c._resample_command = MethodType(PunchCommand._resample_command, c)
    return c


def test_fixed_recipes_are_repeatable_full_starts_with_coherent_previous_actions():
    c = _reset_command()
    c.set_evaluation_recipes({'split': 'validation', 'rows': [
        {'target_index': 2, 'requested_ttc': 0.55, 'recipe_id': 'a'},
        {'target_index': 0, 'requested_ttc': 0.75, 'recipe_id': 'b'}]})
    c._resample_command(torch.tensor([3, 0]))
    assert c.target_index[[0, 3]].tolist() == [2, 0]
    torch.testing.assert_close(c.state.target_world[3], torch.tensor([12., 0., 1.21]))
    torch.testing.assert_close(c.state.requested_ttc[[0, 3]], torch.tensor([.55, .75]))
    assert c.state.history_known[[0, 3]].all()
    assert c.state.elapsed[[0, 3]].tolist() == [0, 0]
    assert c.reset_counts == {'full': 2, 'snippet': 0}
    for actions in (c._env.action_manager.action, c._env.action_manager.prev_action,
                    c._env.action_manager.prev_prev_action):
        torch.testing.assert_close(actions[3], c.ref['joint_pos'][3])
    c.state.physics_steps[3] = 50
    c._resample_command(torch.tensor([3]))
    assert c.state.elapsed[3] == 0
    assert c.state.requested_ttc[3] == pytest.approx(0.75)
    assert c.evaluation_recipe_ids == ['a', 'b']


@pytest.mark.parametrize('split,index,ttc', [('train', 0, .6), ('validation', 3, .6),
                                           ('validation', 0, float('nan'))])
def test_fixed_recipes_reject_wrong_split_or_invalid_command(split, index, ttc):
    c = _reset_command()
    with pytest.raises(ValueError):
        c.set_evaluation_recipes({'split': split, 'rows': [
            {'target_index': index, 'requested_ttc': ttc, 'recipe_id': 'bad'}]})


def test_reference_refresh_optimization_keeps_first_physics_encounter(monkeypatch):
    state = _state()
    state.physics_steps[:] = 116  # Start the actor step at t=0.580 s.
    sites = torch.ones(1, 2, 3)
    refresh_times, geometry_ticks = [], []
    c = SimpleNamespace(state=state, robot=SimpleNamespace(data=SimpleNamespace(site_pos_w=sites)),
        site_ids=[0, 1], active_index=0, glove_velocity=torch.zeros(1, 2, 3),
        previous_glove=sites.clone(), refresh_reference=lambda: refresh_times.append(float(state.elapsed[0])))
    env = SimpleNamespace(_sim_step_counter=116, cfg=SimpleNamespace(decimation=4), physics_dt=.005,
        command_manager=SimpleNamespace(get_term=lambda _: c),
        sim=SimpleNamespace(nan_guard=SimpleNamespace(enabled=False), wp_device='cpu',
            wp_model=None, wp_data=None, data=SimpleNamespace(qpos=torch.zeros(1, 38), qvel=torch.zeros(1, 37))))

    def kinematics(*_):
        geometry_ticks.append(env._sim_step_counter)
        sites[:, 0] = torch.tensor([.13, 0., 0.]) if env._sim_step_counter == 117 else 1.

    monkeypatch.setitem(sys.modules, 'mujoco_warp', SimpleNamespace(kinematics=kinematics))
    monkeypatch.setitem(sys.modules, 'warp', SimpleNamespace(ScopedDevice=lambda _: nullcontext()))
    for _ in range(8):
        env._sim_step_counter += 1
        physics_event_callback(env)
    assert geometry_ticks == list(range(117, 125))
    assert refresh_times == pytest.approx([.60, .62])
    assert state.first_encounter_elapsed[0] == pytest.approx(.585)
    expected = 2 + 8 * np.exp(-.5 * ((.585 - .6) / .1) ** 2)
    assert state.event_bonus()[0] == pytest.approx(expected, rel=1e-6)
    assert state.event_bonus()[0] == 0


def test_dense_target_credit_stops_at_crossing_and_for_unknown_history():
    from human_punching.task10c import mdp

    state = _state(3)
    state.physics_steps[:] = 120
    state.history_known[2] = False
    state.observe(torch.zeros(1, 3), torch.ones(1, 3), ids=torch.tensor([1]))
    q = torch.tensor([1., 0., 0., 0.]).expand(3, 2, 4)
    c = SimpleNamespace(state=state, active_index=0, site_ids=[0, 1],
        robot=SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.zeros(3, 2, 3), site_quat_w=q)),
        glove_velocity=torch.zeros(3, 2, 3),
        ref={'glove_lin_vel_w': torch.zeros(3, 2, 3), 'glove_quat_w': q},
        interaction_z=torch.ones(3), accumulate=lambda _, value: value)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: c))
    reward = mdp.interaction(env)
    assert reward[0] > 0
    assert reward[1:].tolist() == [0, 0]


def test_complete_punch_horizon_is_terminal_and_has_expected_actor_timing():
    from human_punching.task10c.env_cfg import make_env_cfg

    cfg = make_env_cfg(num_envs=2, recovery_fraction=.2, target_fraction=.25,
                       ttc_range=(.54, .66))
    assert cfg.sim.mujoco.timestep == .005 and cfg.decimation == 4
    assert cfg.is_finite_horizon and not cfg.terminations['complete'].time_out
    assert cfg.commands['punch'].recovery_fraction == .2
    assert cfg.commands['punch'].target_fraction == .25


def test_resume_guard_accepts_only_declared_phase2_lineage():
    from human_punching.task10c.runner import PHASE, validate_resume

    valid = {'phase': PHASE, 'family': 'jab', 'lineage_id': 'fresh-jab-lineage',
             'reference_sha256': 'reference', 'target_pool_sha256': 'targets',
             'seed': 102202, 'completed_iterations': 150,
             'curriculum': {'stage': 0, 'target_fraction': .25}}
    validate_resume(valid, 'reference', 'targets', 102202)
    for change in ({'phase': 'phase1_setup'}, {'family': 'cross'}, {'lineage_id': None},
                   {'reference_sha256': 'changed'}, {'target_pool_sha256': 'changed'},
                   {'seed': 1}, {'completed_iterations': -1}, {'curriculum': None}):
        with pytest.raises(ValueError):
            validate_resume({**valid, **change}, 'reference', 'targets', 102202)


def test_raw_episode_stream_drains_without_losing_snippets_or_interruptions(tmp_path):
    import gzip
    import json
    from human_punching.task10c.runner import EpisodeStream

    command = SimpleNamespace(episode_records=[
        {'reset_kind': 'full', 'hit': True, 'fall': False, 'complete_horizon': True,
         'raw_component_totals': {'encounter': 8.5, 'guard': 1.2}},
        {'reset_kind': 'recovery_snippet', 'hit': None, 'fall': False, 'complete_horizon': True,
         'raw_component_totals': {'encounter': 0., 'guard': .7}},
        {'reset_kind': 'full', 'hit': False, 'fall': False, 'complete_horizon': False,
         'raw_component_totals': {'encounter': 0., 'guard': .1}},
    ])
    stream = EpisodeStream(tmp_path)
    try:
        batch = stream.drain(command, 150)
        assert command.episode_records == []
        assert batch['full_attempts'] == 2 and batch['full_hits'] == 1
        assert stream.totals['full']['interrupted'] == 1
        assert stream.totals['recovery_snippet']['hits'] == 0
        assert stream.drain(command, 151)['episodes'] == 0
        assert stream.count == 3
    finally:
        stream.close()
    with gzip.open(tmp_path / 'episodes.jsonl.gz', 'rt') as file:
        rows = [json.loads(line) for line in file]
    assert [row['episode_record_id'] for row in rows] == [0, 1, 2]
    assert all(row['completed_iteration'] == 150 for row in rows)
    assert rows[1]['reset_kind'] == 'recovery_snippet' and rows[1]['hit'] is None


def test_continuation_reward_budgets_inherit_parent_without_mutation():
    from human_punching.task10c.runner import resolve_reward_scales

    assert resolve_reward_scales() == {'balance': 1., 'guard': 1., 'style': 1.}
    parent = {'balance': 5., 'guard': 4., 'style': 1.}
    assert resolve_reward_scales(parent, balance=None, guard=None, style=None) == parent
    assert resolve_reward_scales(parent, guard=6.) == {'balance': 5., 'guard': 6., 'style': 1.}
    assert parent == {'balance': 5., 'guard': 4., 'style': 1.}


@pytest.mark.parametrize('overrides', [{'guard': float('nan')}, {'balance': float('inf')},
                                       {'style': -.01}, {'encounter': 2.}])
def test_continuation_reward_budget_rejects_invalid_override(overrides):
    from human_punching.task10c.runner import resolve_reward_scales

    with pytest.raises(ValueError):
        resolve_reward_scales(**overrides)
    with pytest.raises(ValueError):
        resolve_reward_scales(overrides)


def test_scaled_reward_rate_matches_logged_actor_dt_contribution_once():
    from human_punching.task10c.commands import PunchCommand

    c = SimpleNamespace(cfg=SimpleNamespace(reward_scales={'guard': 4., 'balance': 5.}),
                        component_totals={}, num_envs=2, device='cpu',
                        _env=SimpleNamespace(step_dt=.02))
    c.accumulate = MethodType(PunchCommand.accumulate, c)
    with torch.inference_mode():
        first = c.accumulate('guard', torch.tensor([2., 3.]))
        second = c.accumulate('guard', torch.tensor([1., 4.]))
        event = c.accumulate('encounter', torch.tensor([500., 0.]))
    torch.testing.assert_close(first, torch.tensor([8., 12.]))
    torch.testing.assert_close(second, torch.tensor([4., 16.]))
    torch.testing.assert_close(c.component_totals['guard'], (first + second) * .02)
    torch.testing.assert_close(event, torch.tensor([500., 0.]))
    torch.testing.assert_close(c.component_totals['encounter'], torch.tensor([10., 0.]))
    assert not c.component_totals['guard'].is_inference()
    c.component_totals['guard'][0] = 0.  # Explicit ordinary reset remains valid.


def test_default_guard_speed_score_retains_original_gaussian_exactly():
    from human_punching.task10c.mdp import guard_speed_score

    speeds = torch.tensor([0., .3, .5, 1., 2.4, 5., 20.])
    original = torch.exp(-.5 * (speeds / .5) ** 2)
    assert torch.equal(guard_speed_score(speeds), original)
    assert torch.equal(guard_speed_score(speeds, 'gaussian'), original)


def test_rational_guard_speed_score_is_bounded_monotone_and_dense():
    from human_punching.task10c.mdp import guard_speed_score

    speeds = torch.tensor([0., .3, .5, 1., 2.4, 5., 20.])
    score = guard_speed_score(speeds, 'rational')
    assert bool(torch.isfinite(score).all())
    assert score[0] == 1. and bool((score > 0).all()) and bool((score <= 1).all())
    assert bool((score[1:] < score[:-1]).all())
    assert score[4].item() == pytest.approx(1. / (1. + (2.4 / .5) ** 2))
    assert score[4] > guard_speed_score(speeds, 'gaussian')[4]
    with pytest.raises(ValueError):
        guard_speed_score(speeds, 'unsupported')


def test_guard_speed_profile_defaults_to_original_behavior():
    from human_punching.task10c.commands import PunchCommandCfg

    cfg = PunchCommandCfg(reference_path='reference.npz', target_pool_path='target.npz')
    assert cfg.guard_speed_profile == 'gaussian'


def test_guard_speed_profile_inherits_continuation_unless_explicitly_changed():
    from human_punching.task10c.runner import resolve_guard_speed_profile

    assert resolve_guard_speed_profile() == 'gaussian'
    assert resolve_guard_speed_profile('rational') == 'rational'
    assert resolve_guard_speed_profile('rational', None) == 'rational'
    assert resolve_guard_speed_profile('rational', 'gaussian') == 'gaussian'
    with pytest.raises(ValueError):
        resolve_guard_speed_profile('invalid')
    with pytest.raises(ValueError):
        resolve_guard_speed_profile('gaussian', 'invalid')
