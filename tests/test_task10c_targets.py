"""Focused scene-frame and retained snapshot checks; no physics rollout."""

import json

import numpy as np
import pytest

from human_punching.task10c.targets import (
    FAMILIES,
    SCHEMA,
    TargetPool,
    canonical_from_collection,
    sha256,
    transform_points,
    yaw_transform,
)


def test_yaw_removal_retains_height_and_posture_tilt():
    # q = yaw(90deg) * pitch(30deg); the head's forward tilt remains after yaw removal.
    yaw, pitch = np.pi / 2, np.pi / 6
    q = np.array([np.cos(yaw/2)*np.cos(pitch/2), -np.sin(yaw/2)*np.sin(pitch/2),
                  np.cos(yaw/2)*np.sin(pitch/2), np.sin(yaw/2)*np.cos(pitch/2)])
    qpos = np.r_[2., -3., .8, q]
    head = np.array([2., -2.7, 1.32])
    transform = canonical_from_collection(qpos)
    np.testing.assert_allclose(transform_points(transform, head), [.3, 0., 1.32], atol=1e-14)
    np.testing.assert_allclose(transform_points(np.linalg.inv(transform),
                                               transform_points(transform, head)), head)
    assert transform[2, 3] == 0


def test_scene_placement_does_not_fabricate_height():
    head = np.array([.1, -.02, 1.28])
    transform = yaw_transform(np.pi, np.array([.6, .15, 0.]))
    np.testing.assert_allclose(transform_points(transform, head), [.5, .17, 1.28])


def test_reject_undefined_root_heading():
    with pytest.raises(ValueError, match="heading"):
        canonical_from_collection(np.array([0, 0, 1, 2**-.5, 0, 2**-.5, 0]))


def test_pool_split_identity_and_freeze_copy(tmp_path):
    path = tmp_path / "pool.npz"
    arrays = {
        "snapshot_id": np.array([4, 105844, 211684]),
        "split_id": np.array([0, 1, 2]),
        "canonical_head": np.array([[.1, 0, 1.2], [.2, 0, 1.3], [.3, 0, 1.4]]),
        "canonical_from_collection": np.tile(np.eye(4), (3, 1, 1)),
        "scene_from_canonical": np.tile(yaw_transform(0, np.array([.5, 0, 0])), (4, 1, 1)),
    }
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps({"schema": SCHEMA, "payload_sha256": sha256(path)}))
    pool = TargetPool.load(path)
    for family in FAMILIES:
        np.testing.assert_allclose(pool.points("validation", family), [[.7, 0, 1.3]])
        assert pool.snapshot_ids("validation").tolist() == [105844]
    command_target = pool.points("train", "jab")
    command_target[:] = -100
    np.testing.assert_allclose(pool.points("train", "jab"), [[.6, 0, 1.2]])
    np.testing.assert_allclose(pool.transforms("test", "jab")[0, :3, 3], [.5, 0, 0])
