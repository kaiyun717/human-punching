"""Constant actual-pelvis registration and non-convex sampled reach unions."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from human_punching.task10d.common_start_phase3 import (
    accepted_to_common, common_start_transform, overlap_counts,
    sample_ball_membership, transform_points_yaw,
)
from human_punching.task10d.placement_phase3 import cache_library
from human_punching.task10d.replay import Placement


def test_nonidentity_accepted_placement_is_composed_exactly_once():
    original = np.array([.4, -.3, 1.1])
    accepted = Placement((.8, -.2), .7)
    path = SimpleNamespace(initial_pelvis=accepted.transform_points(original),
        initial_heading_rad=.9, primitive=SimpleNamespace(placement=accepted))
    desired = common_start_transform(path, pelvis_xy=(.2, .6), heading_rad=-.1)
    np.testing.assert_allclose(desired.transform_points(original), [.2, .6, 1.1], atol=1e-14)
    assert np.isclose(.2 + desired.yaw, -.1)


def test_sample_ball_union_preserves_holes_and_height_cutoff():
    samples = np.array([[-1., 0., 1.2], [1., 0., 1.2]])
    queries = np.array([[0., 0., 1.2], [-1., .14, 1.2], [-1., .140001, 1.2], [-1., 0., 1.4]])
    np.testing.assert_array_equal(sample_ball_membership(samples, queries), [False, True, False, False])


def test_two_variants_count_as_one_family():
    masks = np.array([[[True, False]], [[True, True]], [[True, False]]])
    primitives, families = overlap_counts(masks, ("jab", "jab", "cross"))
    np.testing.assert_array_equal(primitives, [[3, 1]])
    np.testing.assert_array_equal(families, [[2, 1]])


@pytest.fixture(scope="module")
def paths():
    return cache_library()


def test_native_full_body_root_and_foot_motion_preserved(paths):
    for path in paths:
        placement = common_start_transform(path)
        origin = path.primitive.reference.sample(0., placement)
        np.testing.assert_allclose(origin.body_pos[0, :2], 0., atol=2e-14)
        forward = Rotation.from_quat(origin.body_quat[0], scalar_first=True).as_matrix()[:, 0]
        assert abs(np.arctan2(forward[1], forward[0])) < 1e-13
        source_start = path.primitive.reference.sample(0., path.primitive.placement)
        for t in (0., float(path.time_s[np.flatnonzero(path.outward)[-1]]), float(path.time_s[-1])):
            source = path.primitive.reference.sample(t, path.primitive.placement)
            common = path.primitive.reference.sample(t, placement)
            np.testing.assert_array_equal(common.qpos, source.qpos)
            np.testing.assert_allclose(common.body_pos, accepted_to_common(path, source.body_pos), atol=2e-14)
            np.testing.assert_allclose(common.glove_pos, accepted_to_common(path, source.glove_pos), atol=2e-14)
            # Existing scipy Rotation application can round Z by a few ulps.
            np.testing.assert_allclose(common.body_pos[:, 2], source.body_pos[:, 2], atol=2e-14, rtol=0)
            # One constant transform preserves root/foot/body displacement from launch.
            np.testing.assert_allclose(np.linalg.norm(common.body_pos - origin.body_pos, axis=1),
                np.linalg.norm(source.body_pos - source_start.body_pos, axis=1), atol=2e-14)


def test_points_keep_z_and_arbitrary_trajectory_dimensions():
    points = np.arange(30, dtype=float).reshape(5, 2, 3) / 10
    result = transform_points_yaw(points, .83, (.6, -.3))
    np.testing.assert_array_equal(result[..., 2], points[..., 2])
    np.testing.assert_allclose(np.linalg.norm(np.diff(result, axis=0), axis=-1),
                               np.linalg.norm(np.diff(points, axis=0), axis=-1), atol=1e-14)
