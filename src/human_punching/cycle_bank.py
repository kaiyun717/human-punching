"""Finite kinematic punch references with boundary-only plan commitment.

This adapter samples reference configurations. It neither steps physics nor
asserts that an external tracker has completed a punch or recovered balance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from .contact import GEOMETRY_VERSION


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    duration_s: float
    diagnostic_ttc_s: float
    head_center_w: tuple[float, float, float]
    trajectory_sha256: str


@dataclass(frozen=True)
class CommittedPlan:
    """Immutable reference choice and world-space head snapshot at launch."""

    variant: str
    family: str
    launch_time_s: float
    duration_s: float
    head_center_w: tuple[float, float, float]
    diagnostic_ttc_s: float


@dataclass(frozen=True)
class ReferenceSample:
    qpos: np.ndarray
    qvel: np.ndarray
    world_time_s: float
    elapsed_s: float
    cycle_complete: bool
    requires_guard_controller: bool
    execution: str = "kinematic reference only; no physical execution implied"


@dataclass(frozen=True)
class GuardTolerances:
    """Independent dimensional tolerances; these are reference admission gates."""

    root_position_m: float = .02
    root_orientation_rad: float = .02
    joint_position_m: float = .02
    joint_angle_rad: float = .02
    root_linear_velocity_mps: float = .10
    root_angular_velocity_radps: float = .10
    joint_linear_velocity_mps: float = .10
    joint_angular_velocity_radps: float = .10


DEFAULT_GUARD_TOLERANCES = GuardTolerances()

def _head(value) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("head_center must be a finite three-vector")
    return tuple(float(x) for x in array)


def _pose(model, value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (model.nq,) or not np.isfinite(pose).all():
        raise ValueError("qpos must match the model and be finite")
    normalized = pose.copy()
    mujoco.mj_normalizeQuat(model, normalized)
    if not np.allclose(pose, normalized, atol=1e-7, rtol=0):
        raise ValueError("qpos must contain normalized quaternions")
    return pose


def interpolate_pose(model, first, second, fraction: float) -> np.ndarray:
    """Interpolate translations/joints and use shortest-arc quaternion motion."""
    first, second = _pose(model, first), _pose(model, second)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Interpolation fraction must be in [0, 1]")
    velocity = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, velocity, 1.0, first, second)
    result = first.copy()
    mujoco.mj_integratePos(model, result, velocity, float(fraction))
    return result


def kinematic_guard_bridge(model, first, second, duration_s: float, control_dt_s: float):
    """Return a quintic, zero-endpoint-speed pose bridge and its limitations.

    This can visualize dissimilar guard poses without teleporting. It does not
    preserve feet, avoid collisions, respect actuator limits, or solve balance.
    Its zero endpoint speed may not match the next clip's nonzero initial speed;
    a physical guard controller must satisfy the adapter's state admission test.
    """
    first, second = _pose(model, first), _pose(model, second)
    if not np.isfinite([duration_s, control_dt_s]).all() or min(duration_s, control_dt_s) <= 0:
        raise ValueError("Bridge duration and control timestep must be positive")
    times = np.unique(np.r_[np.arange(0., duration_s, control_dt_s), duration_s])
    z = times / duration_s
    blend = z**3 * (10 - 15 * z + 6 * z**2)
    derivative = 30 * z**2 * (1 - z)**2 / duration_s
    delta = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, delta, 1., first, second)
    poses = np.stack([interpolate_pose(model, first, second, float(a)) for a in blend])
    return {
        "time_s": times, "qpos": poses, "qvel": derivative[:, None] * delta,
        "execution": "kinematic guard bridge only; dynamic feasibility untested",
        "requires_physical_guard_controller": True,
        "endpoint_velocity_convention": "zero at both ends; next reference may require nonzero velocity",
    }


class CycleBank:
    """Load the generated finite bank and reject changed reference payloads."""

    def __init__(self, manifest_path: Path | str):
        self.path = Path(manifest_path).resolve()
        manifest = json.loads(self.path.read_text())
        if manifest.get("schema") != "human_punching_contact_reference_v1":
            raise ValueError("Unsupported contact-reference bank schema")
        if manifest.get("geometry_version") != GEOMETRY_VERSION:
            raise ValueError("Reference bank does not use accepted Task 08 geometry")
        model_path = Path(manifest["source_model"])
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest["source_model_sha256"]:
            raise ValueError("Reference model XML hash changed")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.variants: dict[str, Variant] = {}
        self._trajectories = {}
        for name, record in manifest["references"].items():
            path = (self.path.parent / name / "human_trajectory.npz").resolve()
            if path.parent.parent != self.path.parent:
                raise ValueError("Reference names must remain inside the bank directory")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record["human_trajectory_sha256"]:
                raise ValueError(f"Reference payload hash changed: {name}")
            with np.load(path, allow_pickle=False) as arrays:
                times, poses, velocities = (np.array(arrays[key], copy=True)
                                           for key in ("time_s", "qpos", "qvel"))
            if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                    or np.any(np.diff(times) <= 0)):
                raise ValueError(f"Invalid reference clock: {name}")
            if poses.shape != (len(times), self.model.nq):
                raise ValueError(f"Reference qpos does not match model: {name}")
            if velocities.shape != (len(times), self.model.nv) or not np.isfinite(velocities).all():
                raise ValueError(f"Invalid reference qvel: {name}")
            for pose in poses:
                _pose(self.model, pose)
            duration = float(times[-1] - times[0])
            ttc = record.get("frozen_head_ttc_from_clip_start_s")
            if not record.get("any_proxy_contact") or ttc is None or not 0 <= ttc <= duration:
                raise ValueError(f"Reference has no accepted diagnostic contact: {name}")
            self.variants[name] = Variant(
                name, str(record["punch"]), duration, float(ttc),
                _head(record["head_center_w"]), digest,
            )
            for array in (times, poses, velocities):
                array.setflags(write=False)
            self._trajectories[name] = times, poses, velocities
        if not self.variants:
            raise ValueError("Reference bank is empty")

    def sample(self, variant: str, elapsed_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Pose interpolation at any reference time; no time scaling is applied."""
        times, poses, velocities = self._trajectories[variant]
        duration = self.variants[variant].duration_s
        if not np.isfinite(elapsed_s) or not 0 <= elapsed_s <= duration:
            raise ValueError("Reference time must lie within its full-cycle duration")
        if elapsed_s == duration:
            return poses[-1].copy(), velocities[-1].copy()
        time = times[0] + elapsed_s
        before = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2))
        after = before + 1
        fraction = float((time - times[before]) / (times[after] - times[before]))
        pose = interpolate_pose(self.model, poses[before], poses[after], fraction)
        velocity = (1 - fraction) * velocities[before] + fraction * velocities[after]
        return pose, velocity

    def valid_plan_mask(self, head_center, *, family=None, max_duration_s=None,
                        diagnostic_ttc_window_s=None) -> dict[str, bool]:
        """Filter fixed references; a different head placement requires retargeting.

        The optional TTC window filters measured clip-start times. It does not
        mean that a requested launch TTC has been synthesized or certified.
        """
        head = _head(head_center)
        if max_duration_s is not None and (not np.isfinite(max_duration_s) or max_duration_s <= 0):
            raise ValueError("max_duration_s must be finite and positive")
        if diagnostic_ttc_window_s is not None:
            lo, hi = diagnostic_ttc_window_s
            if not np.isfinite([lo, hi]).all() or not 0 <= lo <= hi:
                raise ValueError("Invalid diagnostic TTC window")
        return {
            name: bool(np.allclose(head, item.head_center_w, atol=1e-9, rtol=0)
                       and (family is None or item.family == family)
                       and (max_duration_s is None or item.duration_s <= max_duration_s)
                       and (diagnostic_ttc_window_s is None or lo <= item.diagnostic_ttc_s <= hi))
            for name, item in self.variants.items()
        }


class BoundaryCyclePlayer:
    """Commit one complete reference; require explicit guard recovery between cycles."""

    def __init__(self, bank: CycleBank, control_dt_s: float = .02):
        if not np.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError("Control timestep must be finite and positive")
        self.bank, self.control_dt_s = bank, float(control_dt_s)
        self.state = "boundary"
        self.plan: CommittedPlan | None = None
        self._ticks = 0
        self._last_end_time = -np.inf

    def valid_plan_mask(self, head_center, **filters):
        result = self.bank.valid_plan_mask(head_center, **filters)
        return {name: valid and self.state == "boundary" for name, valid in result.items()}

    def guard_errors(self, variant, qpos, qvel) -> dict[str, float]:
        """Separate root and joint pose/velocity errors, with explicit units."""
        expected_q, expected_v = self.bank.sample(variant, 0.)
        actual = _pose(self.bank.model, qpos)
        actual_v = np.asarray(qvel, dtype=float)
        if actual_v.shape != expected_v.shape or not np.isfinite(actual_v).all():
            raise ValueError("Guard qvel must match the model and be finite")
        error = np.empty(self.bank.model.nv)
        mujoco.mj_differentiatePos(self.bank.model, error, 1., actual, expected_q)
        velocity_error = actual_v - expected_v
        result = dict.fromkeys(asdict(GuardTolerances()), 0.)
        for joint, joint_type in enumerate(self.bank.model.jnt_type):
            address = int(self.bank.model.jnt_dofadr[joint])
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                groups = [("root_position_m", "root_linear_velocity_mps", address, 3),
                          ("root_orientation_rad", "root_angular_velocity_radps", address + 3, 3)]
            elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                groups = [("joint_position_m", "joint_linear_velocity_mps", address, 1)]
            else:
                width = 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
                groups = [("joint_angle_rad", "joint_angular_velocity_radps", address, width)]
            for pose_key, velocity_key, first, width in groups:
                result[pose_key] = max(result[pose_key], float(np.linalg.norm(error[first:first+width])))
                result[velocity_key] = max(result[velocity_key], float(np.linalg.norm(velocity_error[first:first+width])))
        return result

    def _check_guard(self, variant, qpos, qvel, tolerances):
        bounds = asdict(tolerances)
        if not np.isfinite(list(bounds.values())).all() or min(bounds.values()) < 0:
            raise ValueError("Guard tolerances must be finite and nonnegative")
        errors = self.guard_errors(variant, qpos, qvel)
        failed = {key: errors[key] for key, bound in bounds.items() if errors[key] > bound}
        if failed:
            raise ValueError(f"Guard does not match next reference; use a guard controller/transition: {failed}")

    def start(self, variant, head_center, launch_time_s, *, current_qpos, current_qvel,
              guard_tolerances: GuardTolerances = DEFAULT_GUARD_TOLERANCES):
        if self.state != "boundary":
            raise RuntimeError("Plan changes require a completed cycle and acknowledged guard boundary")
        if not np.isfinite(launch_time_s) or launch_time_s < self._last_end_time:
            raise ValueError("Launch time must be finite and cannot precede the last cycle end")
        if not self.valid_plan_mask(head_center).get(variant, False):
            raise ValueError("Variant is not legal for this launch head; retarget instead of homing")
        self._check_guard(variant, current_qpos, current_qvel, guard_tolerances)
        item = self.bank.variants[variant]
        self.plan = CommittedPlan(variant, item.family, float(launch_time_s), item.duration_s,
                                  _head(head_center), item.diagnostic_ttc_s)
        self._ticks = 0
        self.state = "active"
        return self.current()

    def current(self) -> ReferenceSample:
        if self.plan is None or self.state == "boundary":
            raise RuntimeError("No current committed cycle")
        elapsed = min(self._ticks * self.control_dt_s, self.plan.duration_s)
        qpos, qvel = self.bank.sample(self.plan.variant, elapsed)
        complete = elapsed >= self.plan.duration_s
        return ReferenceSample(qpos, qvel, self.plan.launch_time_s + self._ticks * self.control_dt_s, elapsed,
                               complete, complete)

    def advance(self) -> ReferenceSample:
        if self.state != "active":
            raise RuntimeError("No active cycle; complete guard recovery before selecting another")
        self._ticks += 1
        sample = self.current()
        if sample.cycle_complete:
            self.state = "waiting_guard"
            self._last_end_time = sample.world_time_s
        return sample

    def acknowledge_guard(self, next_variant, *, current_qpos, current_qvel,
                          guard_tolerances: GuardTolerances = DEFAULT_GUARD_TOLERANCES):
        """External state admission, not a claim of balance or foot-contact validity."""
        if self.state != "waiting_guard":
            raise RuntimeError("Guard acknowledgment is only valid after a complete reference cycle")
        self._check_guard(next_variant, current_qpos, current_qvel, guard_tolerances)
        self.state = "boundary"
        self.plan = None
