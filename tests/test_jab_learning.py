"""Numerical contracts that matter for the learning pilot, independent of CUDA."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from human_punching.jab_learning import (  # noqa: E402
    actuator_layout,
    advantages,
    interpolate_reference,
    load_reference,
    maximum_joint_limit_excursion,
    points_in_body_frame,
    quaternion_error,
    validate_controller_schema,
)


def test_interpolation_clamps_and_handles_irregular_times():
    times = torch.tensor([0., .2, 1.])
    values = torch.tensor([[0., 3.], [2., 5.], [10., 13.]])
    result = interpolate_reference(times, values, torch.tensor([-.1, .1, .6, 2.]))
    torch.testing.assert_close(result, torch.tensor([[0., 3.], [1., 4.], [6., 9.], [10., 13.]]))


def test_limit_cost_catches_a_violation_between_policy_boundaries():
    low, high = torch.tensor([-1., -.5]), torch.tensor([1., .5])
    # First/last states are legal; an intermediate 5 ms sample exceeds a stop.
    positions = torch.tensor([[[0., 0.]], [[0., .53]], [[-.99, .49]], [[0., 0.]]])
    peaks = torch.stack([maximum_joint_limit_excursion(q, low, high) for q in positions])
    torch.testing.assert_close(peaks.amax(0), torch.tensor([.03]))
    torch.testing.assert_close(peaks[-1], torch.zeros(1))


def test_relative_guard_is_invariant_to_world_translation_and_yaw():
    local = torch.tensor([[[.3, .1, .2], [.3, -.1, .2]]])
    rotation = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    origin = torch.tensor([[2., -3., 1.5]])
    world = torch.einsum('bij,bhj->bhi', rotation, local)+origin[:, None]
    torch.testing.assert_close(points_in_body_frame(world, origin, rotation), local)


def test_gae_does_not_leak_across_episode_reset():
    reward = torch.tensor([[1.], [100.]])
    value = torch.zeros_like(reward)
    done = torch.tensor([[True], [False]])
    advantage, target = advantages(reward, value, done, torch.tensor([10.]), gamma=.9, lam=1.)
    torch.testing.assert_close(advantage, torch.tensor([[1.], [109.]]))
    torch.testing.assert_close(target, advantage)


def test_reference_rejects_dimensions_before_simulating(tmp_path):
    import mujoco
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/>'
                                          '<geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    path = tmp_path/'bad.npz'
    np.savez(path, t=[0., 1.], qpos=np.zeros((2, 6)), qvel=np.zeros((2, 6)))
    with pytest.raises(ValueError, match='dimensions'):
        load_reference(model, path)


def test_controller_refuses_free_root_actuation():
    import mujoco
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint name="root"/>'
        '<geom type="sphere" size=".1"/></body></worldbody>'
        '<actuator><motor joint="root" ctrlrange="-10 10"/></actuator></mujoco>')
    with pytest.raises(ValueError, match='scalar hinge'):
        actuator_layout(model)


def test_controller_refuses_unbounded_motor():
    import mujoco
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/>'
        '<body pos="0 0 .2"><joint name="hinge"/><geom type="sphere" size=".1"/>'
        '</body></body></worldbody><actuator><motor joint="hinge"/></actuator></mujoco>')
    with pytest.raises(ValueError, match='bounded ctrlrange'):
        actuator_layout(model)


def test_root_rotation_feedback_uses_local_axis_and_quaternion_double_cover():
    half = np.sqrt(.5)
    current = torch.tensor([[half, 0., 0., half]], dtype=torch.float64)
    desired = torch.tensor([[half*np.cos(.05), half*np.sin(.05),
                             half*np.sin(.05), half*np.cos(.05)]], dtype=torch.float64)
    expected = torch.tensor([[.1, 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(quaternion_error(current, desired), expected)
    torch.testing.assert_close(quaternion_error(current, -desired), expected)


def test_checkpoint_rejects_different_effective_pd_gains():
    # Same model and tensor dimensions do not make a changed controller equivalent.
    with pytest.raises(ValueError, match='kp'):
        validate_controller_schema({'kp': [100., 200.]}, {'kp': [100., 250.]})
