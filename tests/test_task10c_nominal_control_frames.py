"""Focused paired-control checks: one pose-frame change, same frozen launch."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from human_punching.task10c.commands import PunchCommand, TorchReference
from human_punching.task10c.env_cfg import make_env_cfg
from human_punching.task10c.imitation import whole_body_errors, whole_body_scores
from human_punching.task10c.robot import JOINT_NAMES, ROOT
from human_punching.task10c.targets import TargetPool
from human_punching.task10c.whole_body_metrics import WholeBodyMetrics

CANDIDATE = ROOT / 'outputs/task10c/jab/nominal_reference_adaptation_20261022_01/final/candidate/jab.npz'
POOL = ROOT / 'outputs/task10c/setup/target_pool.npz'
TIMING = dict(clock_mode='smooth_entry', ttc_anchor_time=.510, post_entry_duration=.910)


@pytest.fixture(scope='module')
def reference():
    return TorchReference(CANDIDATE, 'cpu', JOINT_NAMES, family='jab', **TIMING)


def state():
    positions = torch.tensor([[[0., 0., 1.], [.5, .2, 1.3], [-.2, .1, .2]]], dtype=torch.float64)
    quaternions = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(1, 3, 4).clone()
    return positions, quaternions, torch.zeros_like(positions), torch.zeros_like(positions)


def test_horizontal_pelvis_drift_is_visible_only_to_launch_registered_body_pose():
    expected = state()
    actual = tuple(value.clone() for value in expected)
    actual[0][..., :2] += torch.tensor([.30, -.20])
    relative = whole_body_errors(*actual, *expected)
    registered = whole_body_errors(*actual, *expected, imitation_frame='launch_registered')
    assert relative['body_position'].item() == pytest.approx(0, abs=1e-15)
    assert registered['body_position'].item() == pytest.approx(.13)
    for name in ('anchor_height', 'anchor_orientation', 'body_orientation',
                 'body_linear_velocity', 'body_angular_velocity'):
        torch.testing.assert_close(registered[name], relative[name])


def test_body_yaw_alignment_changes_only_pose_errors_and_keeps_anchor_kernel():
    expected = state()
    actual = [value.clone() for value in expected]
    angle = torch.tensor(.35, dtype=torch.float64)
    rotation = torch.tensor([torch.cos(angle / 2), 0., 0., torch.sin(angle / 2)]).expand(1, 3, 4)
    actual[0] = expected[0][:, :1] + quat_apply(rotation, expected[0] - expected[0][:, :1])
    actual[1] = quat_mul(rotation, expected[1])
    relative = whole_body_errors(*actual, *expected)
    registered = whole_body_errors(*actual, *expected, imitation_frame='launch_registered')
    assert relative['body_position'].item() < 1e-20
    assert relative['body_orientation'].item() < 1e-20
    assert registered['body_position'].item() > .01
    assert registered['body_orientation'].item() == pytest.approx(.35 ** 2)
    assert relative['anchor_orientation'].item() == pytest.approx(.35 ** 2)
    for name in ('anchor_height', 'anchor_orientation', 'body_linear_velocity', 'body_angular_velocity'):
        torch.testing.assert_close(registered[name], relative[name])


def test_launch_registration_tracks_the_sources_own_moving_root_and_velocities(reference):
    sample = reference.sample(torch.tensor([0., .35, .51, .68, 1.42]), torch.tensor([.7, .7, .7, 1.2, 0.]))
    origins = torch.tensor([[0., 0., 0.], [4., 0., 0.], [0., 4., 0.], [-4., 0., 0.], [0., -4., 0.]])
    expected = (sample['body_pos_w'] + origins[:, None], sample['body_quat_w'],
                sample['body_lin_vel_w'], sample['body_ang_vel_w'])
    pelvis = reference.body_names.index('pelvis')
    assert float((sample['body_pos_w'][:, pelvis, 0].max() - sample['body_pos_w'][:, pelvis, 0].min())) > .1
    assert float(sample['body_lin_vel_w'].abs().max()) > .1
    for score in whole_body_scores(*expected, *expected, imitation_frame='launch_registered').values():
        torch.testing.assert_close(score, torch.ones_like(score), atol=2e-6, rtol=0)


def command_harness(reference, frame, recovery_fraction):
    """Real command constructor/reset, replacing only simulator state writers."""
    n = 8
    origins = torch.tensor([[4.*i, (-1.)**i * 4., 0.] for i in range(n)])
    captured = {}
    def root_writer(value, ids):
        captured['root'] = value.clone()
    def joint_writer(position, velocity, *, env_ids):
        captured['joint_pos'], captured['joint_vel'] = position.clone(), velocity.clone()
    robot = SimpleNamespace(joint_names=JOINT_NAMES, body_names=reference.body_names,
        find_sites=lambda *_, **__: ([0, 1], ['left_glove_center', 'right_glove_center']),
        write_root_state_to_sim=root_writer, write_joint_state_to_sim=joint_writer,
        set_joint_position_target=lambda *_, **__: None)
    class Scene(dict):
        pass
    scene = Scene(robot=robot)
    scene.env_origins = origins
    action = SimpleNamespace(target_ids=list(range(31)), offset=torch.zeros(n, 31),
                             scale=torch.ones(n, 31), _raw_actions=torch.zeros(n, 31))
    env = SimpleNamespace(num_envs=n, device='cpu', physics_dt=.005, step_dt=.02, scene=scene,
        action_manager=SimpleNamespace(get_term=lambda _: action, action=torch.zeros(n, 31),
            prev_action=torch.zeros(n, 31), prev_prev_action=torch.zeros(n, 31)))
    cfg = make_env_cfg(reference_path=CANDIDATE, target_pool_path=POOL, num_envs=n,
        recovery_fraction=recovery_fraction, imitation_frame=frame, task_rewards_enabled=False,
        fixed_target_snapshot_id=53874, ttc_range=(.7, .7), easy_ttc_range=(.7, .7), **TIMING)
    return PunchCommand(cfg.commands['punch'], env), captured


@pytest.mark.parametrize('recovery_fraction', [0., 1.])
def test_both_controls_reset_to_same_frozen_reference_and_real_nominal_target(reference, recovery_fraction):
    captures = []
    pool = TargetPool.load(POOL)
    nominal = pool.points('train', 'jab')[pool.snapshot_ids('train') == 53874][0]
    for frame in ('relative', 'launch_registered'):
        command, captured = command_harness(reference, frame, recovery_fraction)
        assert command.reference_origin_w.data_ptr() != command._env.scene.env_origins.data_ptr()
        assert command.target_candidates.tolist() == [command.fixed_target_index]
        assert command.easy_target_candidates.tolist() == [command.fixed_target_index]
        with torch.random.fork_rng():
            torch.manual_seed(102223)
            command._resample_command(torch.arange(8))
        expected = reference.sample(command.source_time, command.source_rate)
        torch.testing.assert_close(captured['root'][:, :3], expected['root_pos_w'] + command.reference_origin_w)
        torch.testing.assert_close(captured['root'][:, 3:7], expected['root_quat_w'])
        torch.testing.assert_close(captured['joint_pos'], expected['joint_pos'])
        torch.testing.assert_close(command.state.target_world,
            torch.tensor(nominal, dtype=torch.float32)[None] + command._env.scene.env_origins)
        torch.testing.assert_close(command.state.requested_ttc, torch.full((8,), .7))
        if recovery_fraction == 0:
            assert bool((captured['root'][:, 7:] == 0).all())
            assert bool((captured['joint_vel'] == 0).all())
            assert bool((command.source_time == 0).all())
        else:
            torch.testing.assert_close(captured['root'][:, 7:10], expected['body_lin_vel_w'][:, command.root_index])
            torch.testing.assert_close(captured['root'][:, 10:13], expected['body_ang_vel_w'][:, command.root_index])
            torch.testing.assert_close(captured['joint_vel'], expected['joint_vel'])
            assert bool((command.source_time > .510).all())
        captures.append(captured)
    for key in captures[0]:
        torch.testing.assert_close(captures[0][key], captures[1][key], atol=0, rtol=0)


def test_disabling_task_rewards_preserves_six_terms_budgets_controller_and_event_callback():
    original = make_env_cfg()
    control = make_env_cfg(task_rewards_enabled=False, imitation_frame='launch_registered',
                           fixed_target_snapshot_id=53874)
    assert set(control.rewards) == set(original.rewards) - {'encounter', 'interaction'}
    assert control.commands['punch'].imitation_budget == original.commands['punch'].imitation_budget == 7
    assert control.commands['punch'].recovery_budget == original.commands['punch'].recovery_budget == 6
    assert control.commands['punch'].recovery_fraction == .2
    assert control.commands['punch'].easy_fraction == .5
    assert control.metrics['encounter_refresh'].func is original.metrics['encounter_refresh'].func
    assert control.metrics['encounter_refresh'].per_substep
    assert control.actions == original.actions
    assert control.decimation == original.decimation == 4
    assert control.sim.mujoco.timestep == original.sim.mujoco.timestep == .005


def test_terminal_tracking_uses_selected_frame_and_registered_scene_offset():
    expected = state()
    origins = torch.tensor([[12., -4., 0.]], dtype=torch.float64)
    body_positions = expected[0] + origins[:, None]
    context = SimpleNamespace(
        c=SimpleNamespace(cfg=SimpleNamespace(imitation_frame='launch_registered'),
            reference_origin_w=origins, ref=dict(zip(
                ('body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'), expected))),
        body_ids=[0, 1, 2], reference_ids=[0, 1, 2],
        robot=SimpleNamespace(data=SimpleNamespace(body_link_pos_w=body_positions,
            body_link_quat_w=expected[1], body_link_lin_vel_w=expected[2], body_link_ang_vel_w=expected[3])))
    errors, _ = WholeBodyMetrics._tracking(context)
    assert errors['body_position'].item() == pytest.approx(0, abs=1e-20)
    context.robot.data.body_link_pos_w = body_positions + torch.tensor([.3, 0., 0.])
    errors, _ = WholeBodyMetrics._tracking(context)
    assert errors['body_position'].item() == pytest.approx(.09)
    context.c.cfg.imitation_frame = 'relative'
    errors, _ = WholeBodyMetrics._tracking(context)
    assert errors['body_position'].item() == pytest.approx(0, abs=1e-20)
