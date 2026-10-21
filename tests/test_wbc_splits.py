from dataclasses import replace

import pytest

from human_punching.wbc.splits import (
    HoldoutBlock, SplitAssignment, SplitRecord, assert_no_trajectory_leakage,
    assign_trajectory_groups, partition_records,
)


def blocks():
    return [HoldoutBlock("v", "validation", (.60, .65), (-.2, .2), (.5, .6)),
            HoldoutBlock("t", "test", (.85, .90), (-.2, .2), (.7, .8))]


def test_group_assignment_is_stable_and_not_neighbor_snapshot_randomization():
    first = assign_trajectory_groups(range(100))
    assert first == assign_trajectory_groups(reversed(range(100)))
    assert set(first.values()) == {"train", "validation", "test"}
    assert first != assign_trajectory_groups(range(100), seed=11)


def test_trajectory_replica_and_holdout_geometry_all_constrain_membership():
    records = [SplitRecord("train", 1, 0, .75, 0, .6),
               SplitRecord("neighbor", 1, 0, .66, 0, .6),
               SplitRecord("v", 2, 1, .62, 0, .55),
               SplitRecord("wrong-replica", 2, 0, .62, 0, .55),
               SplitRecord("t", 3, 2, .87, 0, .75),
               SplitRecord("wrong-block", 3, 2, .75, 0, .6)]
    rows = partition_records(records, {1: "train", 2: "validation", 3: "test"}, blocks())
    by_id = {row.record_id: row for row in rows}
    assert by_id["train"].split == "train"
    assert by_id["neighbor"].exclusion_reason == "heldout_buffer"
    assert by_id["v"].split == "validation"
    assert by_id["wrong-replica"].exclusion_reason == "original_replica_split_mismatch"
    assert by_id["t"].split == "test"
    assert by_id["wrong-block"].exclusion_reason == "outside_assigned_holdout"
    assert_no_trajectory_leakage(rows)


def test_leakage_and_missing_holdout_definitions_fail():
    rows = [SplitAssignment("a", "command:1", "train", None),
            SplitAssignment("b", "command:1", "test", None)]
    with pytest.raises(ValueError, match="leaks"):
        assert_no_trajectory_leakage(rows)
    with pytest.raises(ValueError, match="both"):
        partition_records([], {}, blocks()[:1])


def test_cross_holdout_buffers_exclude_ambiguous_records():
    overlapping = blocks() + [replace(blocks()[1], block_id="touching-test", forward_m=(.65, .7),
                                     ttc_s=(.5, .6))]
    rows = partition_records([SplitRecord("v-near-test", 2, 1, .63, 0, .55)],
                             {2: "validation"}, overlapping)
    assert rows[0].exclusion_reason == "other_holdout_buffer"
