"""Constructed timing/event checks independent of random-policy punch success."""
import math

import pytest
import torch
import numpy as np
import json

from human_punching.task10c.events import (
    PunchState, guard_weight, interaction_weight, reference_clock, scheduled_normalizers,
)


def state(n=1):
    value = PunchState(n, 'cpu')
    value.reset(torch.arange(n), torch.zeros(n, 3), torch.full((n,), 0.6),
                torch.zeros(n), torch.ones(n, dtype=torch.bool))
    return value


def test_first_of_four_substeps_retained_and_no_reentry_bonus():
    s = state()
    near, far = torch.tensor([[0.13, 0.0, 0.0]]), torch.ones(1, 3)
    for active in (near, far, near, near):
        s.physics_steps += 1
        s.observe(active, far)
    assert float(s.first_encounter_elapsed[0]) == pytest.approx(0.005)
    assert s.pending.item()
    first_bonus = s.event_bonus().item()
    assert first_bonus >= 2
    assert s.event_bonus().item() == 0
    s.physics_steps += 116
    s.observe(near, far)
    assert float(s.first_encounter_elapsed[0]) == pytest.approx(0.005)
    assert s.event_bonus().item() == 0


def test_launch_overlap_survives_first_substep():
    s = state()
    s.observe(torch.zeros(1, 3), torch.ones(1, 3), launch=True)
    assert s.launch_overlap.item()
    s.physics_steps += 1
    s.observe(torch.ones(1, 3), torch.ones(1, 3))
    assert s.pending.item()
    assert s.first_encounter_elapsed.item() == 0
    assert s.event_bonus().item() >= 2


def test_inactive_glove_does_not_earn_active_credit():
    s = state()
    s.physics_steps += 1
    s.observe(torch.ones(1, 3), torch.zeros(1, 3))
    assert s.inactive_encountered.item()
    assert not s.encountered.item()
    assert s.event_bonus().item() == 0
    assert math.isnan(s.first_encounter_elapsed.item())


def test_late_encounters_are_logged_without_strike_credit():
    s = state()
    s.physics_steps += 160
    s.observe(torch.zeros(1, 3), torch.ones(1, 3))
    assert s.encountered.item()
    assert s.first_encounter_elapsed.item() == pytest.approx(0.8)
    assert s.event_bonus().item() == 0


def test_snippet_command_age_and_unknown_history():
    s = state()
    s.reset(torch.tensor([0]), torch.zeros(1, 3), torch.tensor([0.6]),
            torch.tensor([0.82]), torch.tensor([False]))
    assert s.remaining.item() == pytest.approx(-0.22)
    s.observe(torch.zeros(1, 3), torch.ones(1, 3), launch=True)
    assert s.encountered.item()  # First observed snippet overlap, never actual TTC.
    assert math.isnan(s.first_encounter_elapsed.item())
    assert not s.launch_overlap.item()
    assert s.event_bonus().item() == 0


def test_independent_masked_reset_preserves_other_env_clock_pulse():
    s = state(2)
    s.physics_steps += 3
    s.observe(torch.zeros(2, 3), torch.ones(2, 3))
    s.reset(torch.tensor([0]), torch.ones(1, 3), torch.tensor([0.8]),
            torch.tensor([1.0]), torch.tensor([False]))
    assert s.physics_steps.tolist() == [0, 3]
    assert s.pending.tolist() == [False, True]
    assert s.first_encounter_elapsed[1].item() == pytest.approx(0.015)
    assert math.isnan(s.first_encounter_elapsed[0].item())


def test_frozen_target_copy_and_moving_h2_frame():
    from mjlab.utils.lab_api.math import quat_apply_inverse
    s = state()
    target = torch.tensor([[1.0, 0.0, 1.2]])
    s.reset(torch.tensor([0]), target, torch.tensor([0.6]), torch.zeros(1), torch.tensor([True]))
    target += 4  # The external G1 source cannot mutate the frozen copy.
    identity = torch.tensor([[1.0, 0, 0, 0]])
    a = quat_apply_inverse(identity, s.target_world - torch.tensor([[0.0, 0, 0]]))
    b = quat_apply_inverse(identity, s.target_world - torch.tensor([[0.2, 0, 0]]))
    torch.testing.assert_close(a - b, torch.tensor([[0.2, 0, 0]]))
    torch.testing.assert_close(s.target_world, torch.tensor([[1.0, 0, 1.2]]))


def test_reference_clock_rates_and_terminal_guard_hold():
    t = torch.tensor([0.2, 0.6, 0.7, 1.4, 1.8])
    requested = torch.full_like(t, 0.6)
    source, rate = reference_clock(t, requested, 0.9, 1.7, 0.8)
    torch.testing.assert_close(source, torch.tensor([0.3, 0.9, 1.0, 1.7, 1.7]))
    torch.testing.assert_close(rate, torch.tensor([1.5, 1.5, 1.0, 0.0, 0.0]))
    eps = 0.0001
    sample = torch.tensor([0.2, 0.7])
    source0, speed = reference_clock(sample, torch.tensor([0.6, 0.6]), 0.9, 1.7, 0.8)
    source1, _ = reference_clock(sample + eps, torch.tensor([0.6, 0.6]), 0.9, 1.7, 0.8)
    torch.testing.assert_close((source1 - source0) / eps, speed, atol=0.002, rtol=0.002)


def test_integrated_reward_budgets_and_event_dt_scaling():
    requested = torch.tensor([0.5, 0.83, 1.1])
    horizon, zi, zg = scheduled_normalizers(requested, 0.8, 0.02)
    grid = torch.arange(1, int((horizon.max() / 0.02).round()) + 1) * 0.02
    valid = grid[None] <= horizon[:, None] + 1e-6
    interaction = 6 * interaction_weight(grid[None], requested[:, None]) / zi[:, None]
    guard = 2 * guard_weight(grid[None], requested[:, None], 0.8) / zg[:, None]
    torch.testing.assert_close((interaction * valid).sum(-1) * 0.02, torch.full((3,), 6.0))
    torch.testing.assert_close((guard * valid).sum(-1) * 0.02, torch.full((3,), 2.0))
    torch.testing.assert_close((2 / horizon[:, None] * valid).sum(-1) * 0.02, torch.full((3,), 2.0))
    s = state()
    s.physics_steps += 120
    s.observe(torch.zeros(1, 3), torch.ones(1, 3))
    rate = s.event_bonus() / 0.02
    assert (rate * 0.02).item() == pytest.approx(10.0)
    # Even the maximum non-task full-episode style/balance/guard budget is 5.
    assert (rate * 0.02).item() > 2 + 1 + 2


def test_stored_reference_named_order_quaternions_and_retimed_velocity(tmp_path):
    from human_punching.task10c.commands import TorchReference
    names = [f'joint_{i}' for i in range(31)]
    t = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    joint_pos = t[:, None] + np.arange(31)[None, :]
    qpos = np.zeros((3, 38), np.float32)
    qpos[:, 3] = 1
    qpos[:, 7:] = joint_pos
    quat = np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    path = tmp_path / 'reference.npz'
    np.savez(path, time_s=t, joint_names=names, body_names=['pelvis'],
             joint_pos=joint_pos, joint_vel=np.ones((3, 31)), qpos=qpos,
             root_pos_w=np.zeros((3, 3)), root_quat_w=quat,
             body_lin_vel_w=np.ones((3, 1, 3)), body_ang_vel_w=np.ones((3, 1, 3)),
             glove_lin_vel_w=np.ones((3, 2, 3)), glove_ang_vel_w=np.ones((3, 2, 3)))
    metadata = {'family': 'jab', 'active_hand': 'left', 'strike_time_s': 0.5, 'recovery_duration_s': 0.6}
    path.with_suffix('.json').write_text(json.dumps(metadata))
    reference = TorchReference(path, 'cpu', list(reversed(names)), family='jab')
    with pytest.raises(ValueError, match='Reference family'):
        TorchReference(path, 'cpu', names, family='cross')
    path.with_suffix('.json').write_text(json.dumps({**metadata, 'active_hand': 'right'}))
    with pytest.raises(ValueError, match='active_hand'):
        TorchReference(path, 'cpu', names, family='jab')
    sample = reference.sample(torch.tensor([0.25, 1.0]), torch.tensor([2.0, 0.0]))
    assert sample['joint_pos'][0, 0].item() == pytest.approx(30.25)
    for key in ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w', 'glove_lin_vel_w', 'glove_ang_vel_w'):
        torch.testing.assert_close(sample[key][0], torch.full_like(sample[key][0], 2.0))
        torch.testing.assert_close(sample[key][1], torch.zeros_like(sample[key][1]))
    torch.testing.assert_close(sample['root_quat_w'].norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(sample['qpos'][:, 3:7], sample['root_quat_w'])


def test_inference_collection_guard_buffers_allow_explicit_reset():
    from types import SimpleNamespace, MethodType
    from human_punching.task10c.commands import PunchCommand
    from human_punching.task10c import mdp
    body_pos = torch.zeros(2, 1, 3)
    body_quat = torch.tensor([1., 0, 0, 0]).expand(2, 1, 4).clone()
    glove_pos = torch.zeros(2, 2, 3)
    glove_quat = body_quat.expand(2, 2, 4).clone()
    data = SimpleNamespace(body_link_pos_w=body_pos, body_link_quat_w=body_quat,
                           site_pos_w=glove_pos, site_quat_w=glove_quat,
                           projected_gravity_b=torch.tensor([[0., 0, -1]]).expand(2, 3),
                           root_link_lin_vel_b=torch.zeros(2, 3))
    c = SimpleNamespace(robot=SimpleNamespace(data=data), state=state(2),
                        torso_index=0, reference_torso_index=0, site_ids=[0, 1],
                        reference=SimpleNamespace(recovery=0.8), guard_z=torch.ones(2),
                        glove_velocity=torch.zeros(2, 2, 3),
                        settled_duration=torch.zeros(2), first_settled_elapsed=torch.full((2,), float('nan')),
                        component_totals={}, num_envs=2, device='cpu',
                        terminal={'body_pos_w': body_pos, 'body_quat_w': body_quat,
                                  'glove_pos_w': glove_pos, 'glove_quat_w': glove_quat})
    c.state.physics_steps[:] = 200
    env = SimpleNamespace(step_dt=0.02, command_manager=SimpleNamespace(get_term=lambda _: c))
    c._env = env
    c.accumulate = MethodType(PunchCommand.accumulate, c)
    original_duration = c.settled_duration
    with torch.inference_mode():
        mdp.guard(env)
    assert c.settled_duration is original_duration
    assert not c.settled_duration.is_inference()
    assert not c.component_totals['guard'].is_inference()
    # Same indexed writes used by the explicit command reset, outside collection.
    c.settled_duration[0] = 0
    c.first_settled_elapsed[0] = float('nan')
    c.component_totals['guard'][0] = 0
    assert c.settled_duration[1].item() == pytest.approx(0.02)
    assert c.component_totals['guard'][1].item() > 0


def test_default_reference_selection_and_explicit_override():
    from pathlib import Path
    from human_punching.task10c.env_cfg import ROOT, make_env_cfg
    for family in ('left_hook', 'right_hook'):
        revision = 'right_compact_return' if family == 'right_hook' else 'initial_guard_return'
        selected = ROOT / f'outputs/task10c/reference_correction/hooks_final/{revision}/selected/{family}.npz'
        cfg = make_env_cfg(family=family, num_envs=1)
        assert Path(cfg.commands['punch'].reference_path) == selected
        assert selected.is_file()
        historical = ROOT / f'outputs/task10c/references/{family}.npz'
        explicit = make_env_cfg(family=family, reference_path=historical, num_envs=1)
        assert Path(explicit.commands['punch'].reference_path) == historical
    for family in ('jab', 'cross'):
        cfg = make_env_cfg(family=family, num_envs=1)
        assert Path(cfg.commands['punch'].reference_path) == ROOT / f'outputs/task10c/references/{family}.npz'
