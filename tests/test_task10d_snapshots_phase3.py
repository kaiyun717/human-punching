"""Focused Phase3 domain/geometry selection checks; no dynamics or GPU launch."""
import numpy as np
import pytest

from human_punching.task10d.g1 import transform_snapshot
from human_punching.task10d.snapshots_phase3 import (
    FEATURE_NAMES, deterministic_training_subset, selection_features,
)


def test_subset_retains_camera_extremes_when_head_positions_are_identical():
    ids = np.arange(53874, 53974)
    features = np.zeros((100, len(FEATURE_NAMES)))
    features[:, 5] = np.linspace(-.8, .8, 100)
    features[33, 3] = -1.
    features[44, 4] = 1.
    selected = deterministic_training_subset(ids, features, count=12)
    assert {0, 33, 44, 99}.issubset(set(selected))
    assert len(np.unique(selected)) == 12
    # Input reorder cannot change the selected original identities.
    reverse = np.arange(len(ids))[::-1]
    reordered = deterministic_training_subset(ids[reverse], features[reverse], count=12)
    np.testing.assert_array_equal(ids[selected], ids[reverse][reordered])


def test_constant_features_use_original_id_ties_without_duplicate_rows():
    ids = np.array([53879, 53875, 53874, 53876])
    selected = deterministic_training_subset(ids, np.zeros((4, 3)), count=4)
    np.testing.assert_array_equal(ids[selected], [53874, 53875, 53876, 53879])
    with pytest.raises(ValueError, match="outside TRAIN"):
        deterministic_training_subset(ids, np.zeros((4, 3)), count=5)


def test_feature_space_distinguishes_equal_head_different_camera_postures():
    arrays = dict(head_world=np.tile([0., 0., 1.2], (2, 1)),
        pelvis_world=np.tile([0., 0., .8], (2, 1)),
        pelvis_heading_rad=np.zeros(2), pelvis_rotation=np.tile(np.eye(3), (2, 1, 1)),
        torso_rotation=np.tile(np.eye(3), (2, 1, 1)),
        camera_position_world=np.array([[.08, 0., 1.25], [.12, -.05, 1.21]]),
        baseline_camera_rotation=np.tile(np.eye(3), (2, 1, 1)))
    features = selection_features(arrays)
    assert features.shape == (2, len(FEATURE_NAMES))
    np.testing.assert_array_equal(features[0, :3], features[1, :3])
    assert not np.array_equal(features[0, 7:10], features[1, 7:10])


def test_registration_preserves_height_articulation_and_nonzero_velocity():
    q = np.zeros(36, dtype=np.float32); q[2] = .79; q[3] = 1.; q[7:] = .1
    v = np.arange(35, dtype=np.float32) * .01
    transform = np.eye(4); transform[:2, :2] = [[0., -1.], [1., 0.]]
    transform[:2, 3] = [1., .3]
    placed_q, placed_v = transform_snapshot(q, v, transform)
    assert placed_q[2] == q[2]
    np.testing.assert_array_equal(placed_q[7:], q[7:])
    np.testing.assert_array_equal(placed_v[3:], v[3:])
    np.testing.assert_allclose(placed_v[:3], [-v[1], v[0], v[2]], atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(placed_v), np.linalg.norm(v), rtol=1e-7)
