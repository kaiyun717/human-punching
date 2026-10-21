"""Synthetic timing/derivative fixtures, never physical punch evidence."""
import mujoco
import numpy as np
import pytest

from human_punching.phase2_reference import (contact_diagnostics, exact_grid,
    reference_events, retime_uniform)


@pytest.fixture
def free_model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom type="sphere" size=".1" mass="1"/>
        <body><joint axis="0 0 1"/><geom type="sphere" size=".1" mass="1"/></body>
      </body></worldbody></mujoco>''')


def source_fixture(model):
    t = np.arange(101)*.01
    q = np.tile(model.qpos0, (len(t), 1))
    moving = np.clip((t-.1)/.8, 0, 1)
    q[:, 0] = .1*moving**2
    angle = .2*moving**2
    q[:, 3] = np.cos(angle/2)
    q[:, 6] = np.sin(angle/2)
    q[:, 7] = .3*moving**2
    return {"time_s": t, "qpos": q, "launch_time": .1,
            "return_start": .7, "guard_hold_start": .9,
            "head_center": [0., 0., 1.], "active_hand_index": 0}


def test_exact_grid_pads_without_retiming_sample_interval():
    t = exact_grid(1.003)
    assert t[-1] == pytest.approx(1.005)
    np.testing.assert_allclose(np.diff(t), .005, atol=1e-15)
    assert len(exact_grid(1.)) == 201


def test_uniform_derivatives_include_free_root_and_scale(free_model):
    source = source_fixture(free_model)
    original = retime_uniform(free_model, source, 1.)
    slowed = retime_uniform(free_model, source, 2.)
    # Matched points away from source knots: tangent angular velocity, translation
    # and hinge speed all obey the same time transformation.
    indices = np.arange(25, 155, 2)
    np.testing.assert_allclose(slowed["qvel"][indices*2], original["qvel"][indices]/2,
                               atol=2e-13)
    # Piecewise-linear source sampling has acceleration impulses at source
    # knots. Different clock grids distribute them differently; compare the
    # integrated acceleration over equal source intervals, not one spike.
    normal_delta = np.trapezoid(original["qacc"][30:151, 3:6], original["t"][30:151], axis=0)
    slow_delta = np.trapezoid(slowed["qacc"][60:301, 3:6], slowed["t"][60:301], axis=0)
    np.testing.assert_allclose(slow_delta, normal_delta/2, atol=2e-11)
    assert abs(original["qvel"][70, 5]) > 0
    assert np.allclose(np.linalg.norm(slowed["qpos"][:, 3:7], axis=1), 1.)


def test_events_rederive_launch_from_velocity_not_old_annotation():
    t = np.arange(101)*.005
    q, v = np.zeros((101, 8)), np.zeros((101, 7))
    q[23:, 7] = .01
    v[22:25, 6] = 1.
    event = reference_events(t, q, v, .4, 1.)
    assert event["first_nonconstant_reference_tick"] == 22
    assert event["launch_tick"] == 20
    assert event["launch_time_s"] == pytest.approx(.1)
    assert event["outward_end_tick"] == 80


def test_velocity_endpoint_is_not_silently_zeroed(free_model):
    source = source_fixture(free_model)
    # Remove terminal hold; a moving endpoint retains one-sided tangent velocity.
    source["qpos"][-1, 7] += .01
    candidate = retime_uniform(free_model, source, 1.)
    assert candidate["qvel"][-1, -1] > 0


def test_first_early_crossing_cannot_be_repaired():
    t = np.arange(201)*.005
    gloves = np.full((len(t), 2, 3), 1.)
    gloves[:, :, 1:] = 0.
    gloves[70:76, 0, 0] = .1  # Early crossing at about .345 s.
    gloves[119:125, 0, 0] = .1  # Requested .6 s recrossing cannot repair it.
    result = contact_diagnostics(t, gloves, np.zeros(3), 0, 0., 1., .6)
    assert result["frozen_target_ttc_s"] < .5
    assert not result["timing_valid"]


def test_wrong_hand_and_missing_active_contact_remain_failures():
    t = np.arange(201)*.005
    gloves = np.ones((len(t), 2, 3))
    gloves[120:, 1] = .01
    result = contact_diagnostics(t, gloves, np.zeros(3), 0, 0., 1., .6)
    assert result["frozen_target_ttc_s"] is None
    assert result["inactive_hand_first_or_simultaneous"]
    assert not result["timing_valid"]


def test_floor_dominates_requested_tolerance_at_half_second():
    t = np.array([0., .49, .5, .51, 1.])
    gloves = np.ones((len(t), 2, 3))
    gloves[:, 0, 1:] = 0.
    gloves[:, 0, 0] = [.4, .15, .14, .1, .4]
    result = contact_diagnostics(t, gloves, np.zeros(3), 0, 0., 1., .5)
    assert result["earliest_admissible_ttc_s"] == .5
    assert result["timing_valid"]


def test_malformed_clock_and_constant_reference_rejected():
    with pytest.raises(ValueError, match="constant"):
        reference_events(np.arange(101)*.005, np.zeros((101, 8)), np.zeros((101, 7)), .4, 1.)
    with pytest.raises(ValueError, match="5ms"):
        reference_events(np.arange(101)*.006, np.zeros((101, 8)), np.zeros((101, 7)), .4, 1.)
    with pytest.raises(ValueError):
        exact_grid(float("nan"))
