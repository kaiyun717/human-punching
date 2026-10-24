"""Focused target-center, optical-frame and outward-only coverage checks."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from human_punching.task10d.placement_phase3 import (
    CachedPath, PRIMITIVE_IDS, SearchBounds, cache_library, mount_optical_rotation,
    search_primitive, target_centered_placement,
)
from human_punching.task10d.replay import Placement


def test_target_centered_composition_preserves_height_and_equal_height_distances():
    accepted = Placement((.2, -.4), .3)
    target = np.array([1.1, .3, 1.2])
    head = np.array([-2., 4., 1.2])
    native = np.array([[.3, .7, 1.25], [1.4, .2, .9]])
    composed = target_centered_placement(accepted, target, head, .7, -.2, 30)
    actual = composed.transform_points(native)
    expected = accepted.transform_points(native) - target
    angle = .7 - (-.2) + np.deg2rad(30)
    expected = Rotation.from_rotvec([0, 0, angle]).apply(expected) + head
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_array_equal(actual[:, 2], native[:, 2])
    np.testing.assert_allclose(np.linalg.norm(actual - head, axis=-1),
        np.linalg.norm(accepted.transform_points(native) - target, axis=-1))
    identity = target_centered_placement(accepted, target, target, -.2, -.2)
    np.testing.assert_allclose(identity.xy, accepted.xy, atol=1e-14)
    assert identity.yaw == accepted.yaw


def test_pitch_sign_and_native_camera_conversion():
    from boxing.perception.camera import CameraConfig
    for pitch in (-15., 0., 15., 35.):
        config = CameraConfig(mount_pitch_down_deg=-pitch)
        native = Rotation.from_quat(config.mujoco_quaternion_wxyz, scalar_first=True).as_matrix()
        native[:, 1:] *= -1
        actual = mount_optical_rotation(pitch)
        np.testing.assert_allclose(actual, native, atol=1e-14)
        assert actual[2, 2] == pytest.approx(np.sin(np.deg2rad(pitch)))
        np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-14)


def _synthetic_path(outward_reaches=True):
    # Camera/pelvis forward is +x; opponent initial gloves straddle the image.
    initial = [[.6, -.1, 1.], [.6, .1, 1.]]
    middle = [[.1 if outward_reaches else .3, 0., 1.], [.6, .1, 1.]]
    end = [[.0, 0., 1.], [.6, .1, 1.]]
    gloves = np.array([initial, middle, end])
    outward = np.array([True, True, False])
    primitive = SimpleNamespace(active_hand="left", placement=Placement())
    return CachedPath(primitive, np.array([0., .005, .01]), gloves, outward,
        np.array([.9, .2, .8]), np.pi / 2, np.array([0., 0., 1.]),
        cKDTree(gloves[outward, 0]))


def _states():
    return dict(snapshot_id=np.array([1]), head_world=np.array([[0., 0., 1.]]),
        pelvis_world=np.array([[0., 0., .8]]), pelvis_heading_rad=np.array([0.]),
        torso_rotation=np.eye(3)[None], camera_position_world=np.array([[0., 0., 1.]]))


def test_recovery_only_touch_is_excluded_and_facing_is_diagnostic():
    bounds = SearchBounds(yaw_degrees=(0.,), residual_xy_m=(0.,))
    missed = search_primitive(_states(), _synthetic_path(False), nominal_heading_rad=0.,
                              pitches=(0.,), bounds=bounds)
    assert not missed["reachable"][0]
    assert not missed["available"][0, 0]
    reached = search_primitive(_states(), _synthetic_path(True), nominal_heading_rad=0.,
                               pitches=(0.,), bounds=bounds)
    assert reached["available"][0, 0]
    assert reached["top_facing_relative_deg"][0, 0, 0] == pytest.approx(90.)
    assert reached["top_first_entry_sample_s"][0, 0, 0] == .005
    assert reached["top_previous_sample_s"][0, 0, 0] == 0.


def test_yaw_and_residual_apply_to_same_reach_and_frustum_candidate():
    states = _states()
    states["head_world"] = np.array([[1., 2., 1.]])
    states["pelvis_world"] = np.array([[1., 2., .8]])
    states["camera_position_world"] = np.array([[1., 2., 1.]])
    states["pelvis_heading_rad"] = np.array([.5])
    states["torso_rotation"] = Rotation.from_rotvec([0, 0, .5]).as_matrix()[None]
    path = _synthetic_path()
    result = search_primitive(states, path, nominal_heading_rad=0., pitches=(0.,),
        bounds=SearchBounds(yaw_degrees=(-10., 0., 10.), residual_xy_m=(-.075, 0., .075)))
    for rank in range(result["top_available"].shape[-1]):
        if not result["top_available"][0, 0, rank]:
            continue
        placement = Placement(tuple(result["top_xy"][0, 0, rank]),
                              result["top_yaw_rad"][0, 0, rank])
        world = placement.transform_points(path.gloves.reshape(-1, 3)).reshape(path.gloves.shape)
        distances = np.linalg.norm(world[path.outward, 0] - states["head_world"][0], axis=-1)
        assert distances.min() == pytest.approx(result["top_outward_min_m"][0, 0, rank])
        assert distances.min() <= .14


def test_all_eight_cached_paths_match_production_and_nominal_outward_reach():
    paths = cache_library()
    assert tuple(p.primitive.primitive_id for p in paths) == PRIMITIVE_IDS
    for path in paths:
        for index in (0, len(path.time_s)//2, len(path.time_s)-1):
            sample = path.primitive.reference.sample(path.time_s[index], path.primitive.placement)
            np.testing.assert_array_equal(path.gloves[index], sample.glove_pos)
        assert path.tree.query(path.target)[0] <= .14
        assert np.all(np.linalg.norm(path.gloves[0] - path.target, axis=-1) >= .16)
        assert path.time_s[path.outward].max() <= path.primitive.record["annotations"]["withdrawal_start_s"] + 1e-12
