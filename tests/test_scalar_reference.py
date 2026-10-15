"""Regression checks for coordinate derivatives and bound-aware full-body IK."""
import mujoco
import numpy as np

from human_punching.scalar_body import build_scalar_human
from human_punching.scalar_reference import (
    PoseSolver,
    cycle_clock,
    derivatives,
    smooth_scalar_poses,
)


def test_punch_speed_changes_playback_without_shortening_return():
    times = np.array([0., .5, .75, 1.5, 2., 2.5, 3.])
    phase, source_time, attack = cycle_clock(times, 1., 1., .5, 1.)
    np.testing.assert_array_equal(phase, [0, 0, 1, 1, 2, 3, 3])
    np.testing.assert_allclose(source_time, [0., 0., .25, 1., 1., 1., 1.])
    np.testing.assert_allclose(attack, [.5, 1.5])
    slow_phase, _, slow_attack = cycle_clock(times, 1., .5, .5, 1.)
    np.testing.assert_allclose(slow_attack, [.5, 2.5])
    assert slow_phase[-1] == 2


def test_scalar_smoothing_preserves_root_rotations_bounds_and_quiet_guards():
    times = np.linspace(0, 2, 121)
    poses = np.zeros((len(times), 9))
    poses[:, 3] = 1.
    moving = (times > .5) & (times < 1.5)
    poses[moving, 7] = np.sin(np.linspace(0, np.pi, moving.sum()))
    poses[moving, 8] = .2*np.sign(np.sin(np.linspace(0, 5*np.pi, moving.sum())))
    angles = .3*poses[:, 7]
    poses[:, 3], poses[:, 6] = np.cos(angles/2), np.sin(angles/2)
    result = smooth_scalar_poses(poses, times, .025, .5, 1.5)
    np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(result[~moving], np.repeat(poses[0][None], (~moving).sum(), axis=0))
    assert result[:, 7:].min() >= poses[:, 7:].min()-1e-12
    assert result[:, 7:].max() <= poses[:, 7:].max()+1e-12
    assert np.abs(np.diff(result[:, 8])).max() < np.abs(np.diff(poses[:, 8])).max()


def test_reference_derivatives_use_free_root_tangent_space(tmp_path):
    metadata = build_scalar_human(tmp_path / "body")
    model = mujoco.MjModel.from_xml_path(metadata["xml_path"])
    poses = np.repeat(model.qpos0[None], 3, axis=0)
    times = np.array([0., .1, .2])
    poses[:, 0] = [0., .02, .04]
    angles = np.deg2rad([170., 180., 190.])
    poses[:, 3] = np.cos(angles/2)
    poses[:, 4:6] = 0
    poses[:, 6] = np.sin(angles/2)
    velocity = derivatives(model, times, poses)
    np.testing.assert_allclose(velocity[1, :3], [.2, 0., 0.], atol=1e-10)
    np.testing.assert_allclose(velocity[1, 3:6], [0., 0., np.deg2rad(100.)], atol=1e-10)
    np.testing.assert_array_equal(velocity[[0, -1]], 0.)


def test_bound_aware_ik_recovers_planted_pose_from_root_offset(tmp_path):
    metadata = build_scalar_human(tmp_path / "body")
    model = mujoco.MjModel.from_xml_path(metadata["xml_path"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    points, rotations = [], []
    for side in ("Left", "Right"):
        for part in ("Foot_sole", "Hand_center"):
            name = side + part
            site = model.site(name).id
            points.append(("site", name, data.site_xpos[site].copy(), 12.))
        name = side + "Foot_sole"
        site = model.site(name).id
        rotations.append(("site", name, data.site_xmat[site].reshape(3, 3).copy(), 1.5))
    shifted = model.qpos0.copy()
    shifted[0] += .04
    solver = PoseSolver(model)
    result = solver.solve(shifted, points, rotations, prior=model.qpos0, iterations=100)
    assert np.all(result[solver.address] >= solver.lower-1e-9)
    assert np.all(result[solver.address] <= solver.upper+1e-9)
    data.qpos[:] = result
    mujoco.mj_forward(model, data)
    for _, name, target, _ in points:
        assert np.linalg.norm(data.site_xpos[model.site(name).id]-target) < .001
