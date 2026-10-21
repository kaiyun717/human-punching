"""Simulator-independent launch contracts for Task 10.

World is right handed, metres/radians/seconds, +z up. Quaternions are wxyz
body-to-world. Launch data is copied to tuples: later simulator writes cannot
redirect a committed punch. Search bounds below are NOT a feasibility claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Iterable

import numpy as np

PHYSICS_DT_S = 0.005
POLICY_DT_S = 0.020
DECIMATION = 4
CONTACT_DISTANCE_M = 0.14
CLEARANCE_DISTANCE_M = 0.16


class PunchFamily(str, Enum):
    JAB = "jab"
    CROSS = "cross"
    LEFT_HOOK = "left_hook"
    RIGHT_HOOK = "right_hook"

    @classmethod
    def coerce(cls, value: "PunchFamily | str") -> "PunchFamily":
        aliases = {"jb": "jab", "cr": "cross", "lh": "left_hook", "rh": "right_hook"}
        return cls(aliases.get(value, value))

    @property
    def active_glove_index(self) -> int:
        """Glove order is always left, right; the stance is orthodox."""
        return 0 if self in (self.JAB, self.LEFT_HOOK) else 1


def _finite_tuple(values: Iterable[float], size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} requires {size} finite values")
    return result


def quaternion_matrix(quaternion: Iterable[float]) -> np.ndarray:
    w, x, y, z = _finite_tuple(quaternion, 4, "quaternion")
    if abs(w * w + x * x + y * y + z * z - 1.0) > 1e-6:
        raise ValueError("Quaternion must be unit length, wxyz")
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


@dataclass(frozen=True)
class TimingContract:
    physics_dt_s: float = PHYSICS_DT_S
    policy_dt_s: float = POLICY_DT_S
    decimation: int = DECIMATION
    integration_substeps: int = 1

    def __post_init__(self) -> None:
        if (self.physics_dt_s, self.policy_dt_s, self.decimation,
                self.integration_substeps) != (.005, .020, 4, 1):
            raise ValueError("Task 10 requires one actual 5 ms step, 20 ms actor, decimation four")


@dataclass(frozen=True)
class LaunchSnapshot:
    """human_origin/quaternion define a fixed ground task frame, not pelvis pose.

    Its axes are forward/left/up. Asset-specific SOMA root rotations belong in
    reference conversion, outside this frame. G1 state is the launch-time state.
    """
    snapshot_id: str
    g1_qpos: tuple[float, ...]
    g1_qvel: tuple[float, ...]
    head_position_w: tuple[float, ...]
    head_velocity_w: tuple[float, ...]
    human_origin_w: tuple[float, ...]
    human_quat_wxyz: tuple[float, ...]
    source_group_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id:
            raise ValueError("A source snapshot identity is required")
        for name, size in (("g1_qpos", 36), ("g1_qvel", 35),
                           ("head_position_w", 3), ("head_velocity_w", 3),
                           ("human_origin_w", 3), ("human_quat_wxyz", 4)):
            object.__setattr__(self, name, _finite_tuple(getattr(self, name), size, name))
        quaternion_matrix(self.g1_qpos[3:7])
        human_rotation = quaternion_matrix(self.human_quat_wxyz)
        if not np.allclose(human_rotation[:, 2], (0, 0, 1), atol=1e-6, rtol=0):
            raise ValueError("Human ground task frame must be level; quaternion describes heading")

    def world_to_human_point(self, point: Iterable[float]) -> tuple[float, ...]:
        point = np.asarray(_finite_tuple(point, 3, "point"))
        return tuple(quaternion_matrix(self.human_quat_wxyz).T @ (point - self.human_origin_w))

    def world_to_human_vector(self, vector: Iterable[float]) -> tuple[float, ...]:
        return tuple(quaternion_matrix(self.human_quat_wxyz).T
                     @ np.asarray(_finite_tuple(vector, 3, "vector")))

    def relative_placement(self) -> tuple[float, float]:
        """Human forward/lateral placement in launch G1 pelvis-yaw frame."""
        rotation = quaternion_matrix(self.g1_qpos[3:7])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        dx, dy = np.asarray(self.human_origin_w[:2]) - self.g1_qpos[:2]
        return (float(math.cos(yaw)*dx + math.sin(yaw)*dy),
                float(-math.sin(yaw)*dx + math.cos(yaw)*dy))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                                        allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PunchRequest:
    family: PunchFamily
    ttc_s: float
    snapshot: LaunchSnapshot
    guard_id: str
    launch_time_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", PunchFamily.coerce(self.family))
        if not math.isfinite(self.ttc_s) or self.ttc_s <= 0:
            raise ValueError("TTC is positive elapsed time from launch to first analytic contact")
        if not math.isfinite(self.launch_time_s) or self.launch_time_s < 0:
            raise ValueError("Launch time must be finite and nonnegative")
        if not isinstance(self.snapshot, LaunchSnapshot) or not self.guard_id:
            raise ValueError("A frozen snapshot and named initial guard are required")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                                        allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class LaunchPlan:
    request: PunchRequest
    reference_id: str
    reference_sha256: str
    target_position_w: tuple[float, ...]

    def __post_init__(self) -> None:
        target = _finite_tuple(self.target_position_w, 3, "target_position_w")
        object.__setattr__(self, "target_position_w", target)
        if target != self.request.snapshot.head_position_w:
            raise ValueError("Committed target must equal the launch snapshot head position")
        if not self.reference_id or len(self.reference_sha256) != 64:
            raise ValueError("An immutable reference identity and SHA256 are required")
        try:
            int(self.reference_sha256, 16)
        except ValueError as error:
            raise ValueError("Reference SHA256 must be hexadecimal") from error


@dataclass(frozen=True)
class SearchBounds:
    forward_m: tuple[float, float] = (.60, .95)
    lateral_m: tuple[float, float] = (-.20, .20)

    def __post_init__(self) -> None:
        for name in ("forward_m", "lateral_m"):
            limits = _finite_tuple(getattr(self, name), 2, name)
            if limits[0] >= limits[1]:
                raise ValueError("Search intervals must increase")
            object.__setattr__(self, name, limits)

    @staticmethod
    def ttc_interval(family: PunchFamily | str) -> tuple[float, float]:
        return {PunchFamily.JAB: (.50, .90), PunchFamily.CROSS: (.50, .90),
                PunchFamily.LEFT_HOOK: (.85, 1.35),
                PunchFamily.RIGHT_HOOK: (.75, 1.20)}[PunchFamily.coerce(family)]

    def contains(self, request: PunchRequest) -> bool:
        forward, lateral = request.snapshot.relative_placement()
        low, high = self.ttc_interval(request.family)
        return (self.forward_m[0] <= forward <= self.forward_m[1]
                and self.lateral_m[0] <= lateral <= self.lateral_m[1]
                and low <= request.ttc_s <= high)
