"""Trajectory-group splits with explicit buffered target/TTC holdout blocks.

Assign command groups BEFORE expanding snapshots into target/guard/TTC recipes.
All replicas of one collection command share the assignment. Original bank
replica splits remain a second filter; unused rows are recorded as exclusions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping


SPLITS = ("train", "validation", "test")
ORIGINAL_REPLICA_SPLIT = {0: "train", 1: "validation", 2: "test"}


@dataclass(frozen=True)
class SplitRecord:
    record_id: str
    command_id: int
    replica_id: int
    forward_m: float
    lateral_m: float
    ttc_s: float

    def __post_init__(self) -> None:
        if (not self.record_id or self.command_id < 0 or self.replica_id not in ORIGINAL_REPLICA_SPLIT
                or not all(math.isfinite(x) for x in (self.forward_m, self.lateral_m, self.ttc_s))
                or self.ttc_s <= 0):
            raise ValueError("Split records require original bank IDs and finite physical coordinates")

    @property
    def source_group_id(self) -> str:
        return f"command:{self.command_id}"


@dataclass(frozen=True)
class HoldoutBlock:
    block_id: str
    split: str
    forward_m: tuple[float, float]
    lateral_m: tuple[float, float]
    ttc_s: tuple[float, float]
    buffer_forward_m: float = .025
    buffer_lateral_m: float = .025
    buffer_ttc_s: float = .025

    def __post_init__(self) -> None:
        if not self.block_id or self.split not in ("validation", "test"):
            raise ValueError("Holdout blocks belong to validation or test")
        for name in ("forward_m", "lateral_m", "ttc_s"):
            value = tuple(float(item) for item in getattr(self, name))
            if len(value) != 2 or not all(map(math.isfinite, value)) or value[0] >= value[1]:
                raise ValueError(f"{name} requires an increasing finite interval")
            object.__setattr__(self, name, value)
        for value in (self.buffer_forward_m, self.buffer_lateral_m, self.buffer_ttc_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Holdout buffers must be positive")

    def contains(self, record: SplitRecord, *, buffered: bool = False) -> bool:
        values = (record.forward_m, record.lateral_m, record.ttc_s)
        intervals = (self.forward_m, self.lateral_m, self.ttc_s)
        buffers = (self.buffer_forward_m, self.buffer_lateral_m, self.buffer_ttc_s)
        return all(low - (pad if buffered else 0) <= value <= high + (pad if buffered else 0)
                   for value, (low, high), pad in zip(values, intervals, buffers, strict=True))


@dataclass(frozen=True)
class SplitAssignment:
    record_id: str
    source_group_id: str
    split: str | None
    exclusion_reason: str | None
    block_id: str | None = None


def assign_trajectory_groups(command_ids: Iterable[int], *, seed: int = 10,
                             train_fraction: float = .70, validation_fraction: float = .15
                             ) -> dict[int, str]:
    """Stable SHA256 assignment; input ordering and replica IDs cannot change it."""
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Train/validation fractions must leave a positive test fraction")
    result = {}
    for command_id in command_ids:
        if type(command_id) is not int or command_id < 0:
            raise ValueError("Original nonnegative integer command IDs are required")
        digest = hashlib.sha256(f"task10-command-split-v1:{seed}:{command_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        result[command_id] = ("train" if value < train_fraction else
                              "validation" if value < train_fraction + validation_fraction else "test")
    return result


def partition_records(records: Iterable[SplitRecord], group_splits: Mapping[int, str],
                      holdout_blocks: Iterable[HoldoutBlock], *, preserve_replica_split: bool = True
                      ) -> tuple[SplitAssignment, ...]:
    """Keep a recipe only if trajectory, original replica and geometry agree.

    No rejected record is resampled or silently moved to training. Validation/test
    records must be inside their declared blocks; training excludes every block's
    buffer. A validation sample in a test buffer is also excluded, and conversely.
    """
    records, blocks = tuple(records), tuple(holdout_blocks)
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("Duplicate recipe IDs")
    if len({block.block_id for block in blocks}) != len(blocks):
        raise ValueError("Duplicate holdout block IDs")
    if {block.split for block in blocks} != {"validation", "test"}:
        raise ValueError("Declare both validation and test blocks before generating recipes")
    if any(split not in SPLITS for split in group_splits.values()):
        raise ValueError("Unknown trajectory split")
    assigned = []
    for record in records:
        group = group_splits.get(record.command_id)
        reason, block_id = None, None
        if group is None:
            reason = "trajectory_not_assigned"
        elif preserve_replica_split and ORIGINAL_REPLICA_SPLIT[record.replica_id] != group:
            reason = "original_replica_split_mismatch"
        else:
            inside = [block for block in blocks if block.contains(record)]
            near = [block for block in blocks if block.contains(record, buffered=True)]
            if group == "train":
                if near:
                    reason = "heldout_buffer"
            elif any(block.split != group for block in near):
                reason = "other_holdout_buffer"
            else:
                matching = [block for block in inside if block.split == group]
                if not matching:
                    reason = "outside_assigned_holdout"
                else:
                    block_id = matching[0].block_id
        assigned.append(SplitAssignment(record.record_id, record.source_group_id,
                                        group if reason is None else None, reason, block_id))
    return tuple(assigned)


def assert_no_trajectory_leakage(assignments: Iterable[SplitAssignment]) -> None:
    seen = {}
    for row in assignments:
        if row.split is not None:
            previous = seen.setdefault(row.source_group_id, row.split)
            if previous != row.split:
                raise ValueError(f"Trajectory {row.source_group_id} leaks across splits")
