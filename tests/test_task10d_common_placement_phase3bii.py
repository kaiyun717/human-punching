"""Focused common actual pose and equivalent-envelope checks."""
from pathlib import Path
import json

import numpy as np
import pytest

from human_punching.task10d.g1 import PROJECT
from human_punching.task10d.common_placement_phase3bii import (
    common_pose_candidates, evaluate_common_pose,
)
from human_punching.task10d.common_start_phase3 import common_start_transform
from human_punching.task10d.placement_phase3 import SearchBounds, cache_library


@pytest.fixture(scope="module")
def inputs():
    paths = cache_library()
    with np.load(PROJECT/"outputs/task10d/phase3/snapshots/geometry.npz") as geometry:
        index = int(np.flatnonzero(geometry["snapshot_id"] == 53874)[0])
        state = {key:geometry[key][index].copy() for key in geometry.files
                 if geometry[key].shape[:1] == geometry["snapshot_id"].shape}
    bounds = SearchBounds(**json.loads((PROJECT/"configs/task10d/phase3_review_bounds.json").read_text()))
    poses = common_pose_candidates(paths, PROJECT/"outputs/task10d/phase3/search01", 53874)
    return paths,state,bounds,poses


def test_union_is_bounded_and_deterministically_deduplicated(inputs):
    paths,_,_,poses = inputs
    repeated = common_pose_candidates(paths, PROJECT/"outputs/task10d/phase3/search01", 53874)
    assert poses == repeated
    assert 1 <= len(poses) <= 32
    assert sum(len(p["donors"]) for p in poses) == 32
    identities = [tuple(np.round([*p["actual_initial_pelvis_xy"], p["actual_initial_pelvis_yaw_rad"]],10)) for p in poses]
    assert len(set(identities)) == len(poses)


def test_donor_pose_recovers_same_original_grid_placement_and_reach(inputs):
    paths,state,bounds,poses = inputs
    for path in paths:
        pose = next(p for p in poses if dict(primitive_id=path.primitive.primitive_id,geometric_rank=0) in p["donors"])
        result = evaluate_common_pose(path,state,pose,
            nominal_heading_rad=float(state["pelvis_heading_rad"]),bounds=bounds)
        assert result["on_original_yaw_grid"] and result["on_original_residual_grid"]
        assert result["reach_frustum_clearance_allowed"]
        placement = common_start_transform(path, pelvis_xy=pose["actual_initial_pelvis_xy"],
            heading_rad=pose["actual_initial_pelvis_yaw_rad"])
        sample = path.primitive.reference.sample(result["first_active_outward_sample_s"],placement)
        assert np.linalg.norm(sample.glove_pos[path.active_index]-state["head_world"]) <= .14


def test_out_of_envelope_reach_never_becomes_admissible_support(inputs):
    paths,state,_,poses = inputs
    path=paths[0]
    pose = next(p for p in poses if dict(primitive_id=path.primitive.primitive_id,geometric_rank=0) in p["donors"])
    result = evaluate_common_pose(path,state,pose,
        nominal_heading_rad=float(state["pelvis_heading_rad"]),
        bounds=SearchBounds(yaw_degrees=(0.,),residual_xy_m=(0.,)))
    assert result["reach_only"]
    assert not result["reach_within_allowed_envelope"]
    assert not result["reach_frustum_clearance_allowed"]
    assert "residual_xy_envelope" in result["rejection_reasons"]
