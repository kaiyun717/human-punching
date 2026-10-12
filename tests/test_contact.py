"""Published boundary, attachment, and swept-center contact regressions."""

import numpy as np
import pytest

from human_punching.contact import (
    CONTACT_RADIUS_M,
    GEOMETRY_VERSION,
    GLOVE_RADIUS_M,
    HEAD_RADIUS_M,
    SAFE_CLEARANCE_RADIUS_M,
    SAFETY_MARGIN_M,
    evaluate_clearance,
    first_contact_time,
    world_attached_centers,
)


def test_task08_constants_and_boundary_equalities():
    assert GEOMETRY_VERSION == "task08_head06_gloves08_margin02_v1"
    assert GLOVE_RADIUS_M == .08 and HEAD_RADIUS_M == .06 and SAFETY_MARGIN_M == .02
    assert CONTACT_RADIUS_M == .14 and SAFE_CLEARANCE_RADIUS_M == .16
    gloves = np.zeros((5, 2, 3))
    gloves[:, 0, 0] = [.139, .14, .15, .16, .161]
    gloves[:, 1, 0] = .3
    result = evaluate_clearance(gloves, np.zeros(3))
    np.testing.assert_array_equal(result.any_contact, [True, True, False, False, False])
    np.testing.assert_array_equal(result.fist_clearance_safe, [False, False, False, True, True])
    np.testing.assert_allclose(result.min_safety_clearance_m, [-.021, -.02, -.01, 0, .001])


def test_inactive_glove_and_moving_head_are_not_ignored():
    head = np.array([[0., 0, 0], [1., 0, 0]])
    gloves = np.array([[[.3, 0, 0], [.13, 0, 0]], [[1.3, 0, 0], [1.17, 0, 0]]])
    result = evaluate_clearance(gloves, head)
    np.testing.assert_array_equal(result.any_contact, [True, False])
    np.testing.assert_array_equal(result.fist_clearance_safe, [False, True])
    # A global rigid translation cannot change center-distance labels.
    translated = evaluate_clearance(gloves + [10, -3, 2], head + [10, -3, 2])
    np.testing.assert_allclose(translated.distances_m, result.distances_m, atol=1e-14)


def test_hand_rotation_moves_offset_center_but_not_wrist_origin():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    body = np.array([1, 2, 3])
    offset = np.array([.066, -.004, .011])
    center = world_attached_centers(body, rotation, offset)
    np.testing.assert_allclose(center, [1.004, 2.066, 3.011])
    assert np.linalg.norm(center - body) > .06


def test_ttc_uses_physical_envelope_and_catches_between_frame_crossing():
    times = np.array([10., 11.])
    gloves = np.array([[[-.3, 0, 0], [0, 1, 0]], [[.3, 0, 0], [0, 1, 0]]])
    assert not evaluate_clearance(gloves, np.zeros(3)).any_contact.any()
    expected = 10 + (.3 - .14) / .6
    assert first_contact_time(times, gloves, np.zeros(3)) == pytest.approx(expected)
    # Tangency is contact under the inclusive physical boundary.
    gloves[:, 0, 1] = .14
    assert first_contact_time(times, gloves, np.zeros(3)) == pytest.approx(10.5)


def test_frozen_target_ttc_differs_from_actual_moving_head_contact():
    gloves = np.array([[[.4, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 1, 0]]])
    assert first_contact_time([0, 1], gloves, [0, 0, 0]) == pytest.approx(.65)
    # The head retreats at the glove's speed; separation stays .4 m.
    moving_head = np.array([[0, 0, 0], [-.4, 0, 0]])
    assert np.isinf(first_contact_time([0, 1], gloves, moving_head))


def test_first_sample_contact_and_stationary_miss():
    gloves = np.array([[[.14, 0, 0], [0, 1, 0]]])
    assert first_contact_time([2], gloves, [0, 0, 0]) == 2
    gloves[0, 0, 0] = .15
    assert np.isinf(first_contact_time([2, 3], np.repeat(gloves, 2, axis=0), [0, 0, 0]))


@pytest.mark.parametrize("bad", [np.full((2, 3), np.nan), np.zeros((1, 3)), np.zeros(3)])
def test_invalid_glove_states_are_rejected(bad):
    with pytest.raises(ValueError):
        evaluate_clearance(bad, [0, 0, 0])


def test_invalid_clock_and_attachment_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        first_contact_time([0, 0], np.zeros((2, 2, 3)), np.zeros(3))
    with pytest.raises(ValueError, match="body_rotations"):
        world_attached_centers([0, 0, 0], np.eye(2), [0, 0, 0])
