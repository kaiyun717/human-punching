"""Single-cycle launch control; runtime integration requires qualification. Pure Torch; never writes simulator state.

Boundary commitment precedes actor observation/inference. after_substep consumes
one real 5 ms endpoint; reference time is zero until an accepted boundary launch.
"""
from dataclasses import dataclass
from enum import IntEnum
import math

import torch

from .event_state import FLAGS, KINEMATIC_WIDTHS, Thresholds, angle


class Phase(IntEnum):
    SETTLING = 0
    STRIKE = 1
    RECOVERY = 2
    COMPLETE = 3
    FAILED = 4
    UNINITIALIZED = 5


class Failure(IntEnum):
    NONE = 0
    REQUEST_REJECTED = 1
    PLANNER_FAILED = 2
    GUARD_TIMEOUT = 3
    PLAN_UNAVAILABLE = 4
    PLAN_CONTEXT_MISMATCH = 5
    INVALID_TELEMETRY = 6
    EXTERNAL_SUPPORT = 7
    FORBIDDEN_COLLISION = 8
    BALANCE = 9
    FOOT_REPOSITION = 10
    JOINT_LIMIT = 11
    EFFORT_LIMIT = 12
    CLOCK = 13
    CYCLE_FAILURE = 14
    RECOVERY_TIMEOUT = 15
    INVALID_PLAN_PACKET = 16
    ENVIRONMENT_TERMINATION = 17
    ENVIRONMENT_TIMEOUT = 18


class PlanStatus(IntEnum):
    PREPARING = 0
    ADMITTED = 1
    REJECTED = 2
    FAILED = 3


@dataclass(frozen=True)
class ReadinessConfig:
    quiet_ticks: int = 40  # proposed 0.200 s launch dwell; recovery remains 1 s
    maximum_settling_ticks: int = 1200  # proposed 6 s, distinct from TTC

    def __post_init__(self):
        for value in (self.quiet_ticks, self.maximum_settling_ticks):
            if type(value) is not int or value <= 0:
                raise ValueError("Positive integer physics-tick budgets required")
        if self.maximum_settling_ticks % 4 or self.quiet_ticks >= self.maximum_settling_ticks:
            raise ValueError("Settling budget must end on a policy boundary after quiet dwell")


class ReadinessClock:
    dt = .005
    decimation = 4

    def __init__(self, n, joint_limits, effort_caps, *, context_width, config=ReadinessConfig(), thresholds=Thresholds()):
        if type(n) is not int or n < 1 or type(context_width) is not int or context_width < 1:
            raise ValueError("Positive exact environment count and G1 context width required")
        if (joint_limits.ndim != 2 or joint_limits.shape[1] != 2 or effort_caps.shape != joint_limits.shape[:1]
                or joint_limits.dtype not in (torch.float32, torch.float64) or effort_caps.dtype != joint_limits.dtype
                or effort_caps.device != joint_limits.device or not torch.isfinite(joint_limits).all()
                or not torch.isfinite(effort_caps).all() or not (effort_caps > 0).all()
                or not (joint_limits[:, 0] < joint_limits[:, 1]).all()):
            raise ValueError("Finite ordered joint limits and positive effort caps required")
        self.n, self.j, self.context_width = n, len(effort_caps), context_width
        self.device, self.dtype = joint_limits.device, joint_limits.dtype
        self.limits, self.caps = joint_limits.clone(), effort_caps.clone()
        self.config, self.thresholds = config, thresholds
        self.phase = self.z(dtype=torch.int64)+int(Phase.UNINITIALIZED)
        self.tick = self.z(dtype=torch.int64)
        self.launch_tick = self.z(dtype=torch.int64)-1
        self.time_origin_s = self.z(dtype=torch.float64)
        self.quiet_intervals = self.z(dtype=torch.int64)
        self.previous_ready = self.z(dtype=torch.bool)
        self.request_id = self.z(dtype=torch.int64)-1
        self.failure = self.z(dtype=torch.int64)
        self.failure_tick = self.z(dtype=torch.int64)-1
        self.logged = self.z(dtype=torch.bool)
        self.environment_terminated = self.z(dtype=torch.bool)
        self.environment_timed_out = self.z(dtype=torch.bool)
        self.attempt_count = self.z(dtype=torch.int64)
        self.launch_count = self.z(dtype=torch.int64)
        self.failure_count = self.z(dtype=torch.int64)
        self.complete_count = self.z(dtype=torch.int64)
        self.nominal = {name: self.z(*shape) for name, shape in KINEMATIC_WIDTHS.items()}
        self.nominal["joint_position"] = self.z(self.j)
        self.launch_guard = {name: value.clone() for name, value in self.nominal.items()}
        self.last_state = None
        self.committed = {
            "target_w": self.z(3), "g1_launch_state": self.z(context_width),
            "requested_ttc_s": self.z(), "outward_end_s": self.z(),
            "family_id": self.z(dtype=torch.int64), "plan_handle": self.z(dtype=torch.int64)-1,
            "plan_digest": self.z(32, dtype=torch.uint8),
        }
        self._steps_seen = 0
        self._last_boundary = -1

    def z(self, *tail, dtype=None):
        return torch.zeros((self.n, *tail), dtype=dtype or self.dtype, device=self.device)

    def _ids(self, ids):
        ids = torch.as_tensor(ids, device=self.device)
        if (ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64) or ids.numel() < 1
                or (ids < 0).any() or (ids >= self.n).any() or ids.unique().numel() != ids.numel()):
            raise ValueError("Unique integer environment IDs required")
        return ids.long()

    def _state_finite(self, state):
        shapes = {**KINEMATIC_WIDTHS, "joint_position": (self.j,), "joint_velocity": (self.j,), "effort": (self.j,),
                  "support_contact": (2,), "support_patch_contact": (2,), **{name: () for name in FLAGS}}
        finite = torch.ones(self.n, dtype=torch.bool, device=self.device)
        for name, tail in shapes.items():
            boolean = name in FLAGS or name in ("support_contact", "support_patch_contact")
            value = state.get(name)
            if (not isinstance(value, torch.Tensor) or value.shape != (self.n, *tail)
                    or value.device != self.device or value.dtype != (torch.bool if boolean else self.dtype)):
                raise ValueError(f"Malformed readiness telemetry: {name}")
            finite &= torch.isfinite(value).reshape(self.n, -1).all(-1)
        for name in ("root_quat_wxyz", "glove_quat_guard_frame"):
            finite &= (torch.linalg.vector_norm(state[name], dim=-1)-1).abs().reshape(self.n, -1).le(1e-5).all(-1)
        return finite

    def start_episode(self, ids, state, *, nominal_guard, request_id, time_origin_s):
        """Call after the one allowed physical initialization; no readiness claim."""
        ids = self._ids(ids)
        if self._steps_seen % 4 or not ((self.phase[ids] == Phase.UNINITIALIZED) | self.logged[ids]).all():
            raise ValueError("Episode start requires a boundary and preserved prior receipt")
        finite = self._state_finite(state)
        # A complete nominal state packet makes its guard-frame conventions explicit.
        nominal_finite = self._state_finite(nominal_guard)
        if not finite[ids].all() or not nominal_finite[ids].all():
            raise ValueError("Finite initial and declared nominal guard states required")
        if (request_id.shape != (self.n,) or request_id.dtype != torch.int64 or request_id.device != self.device
                or time_origin_s.shape != (self.n,) or time_origin_s.dtype != torch.float64
                or time_origin_s.device != self.device or not torch.isfinite(time_origin_s[ids]).all()):
            raise ValueError("Explicit request identities and float64 episode time origins required")
        for name in self.nominal:
            self.nominal[name][ids] = nominal_guard[name][ids].clone()
            self.launch_guard[name][ids] = 0
        self.phase[ids] = Phase.SETTLING
        self.tick[ids], self.launch_tick[ids] = 0, -1
        self.time_origin_s[ids], self.request_id[ids] = time_origin_s[ids].clone(), request_id[ids].clone()
        self.quiet_intervals[ids] = 0
        self.previous_ready[ids] = False
        self.failure[ids], self.failure_tick[ids] = Failure.NONE, -1
        self.logged[ids] = False
        self.environment_terminated[ids] = False
        self.environment_timed_out[ids] = False
        self.attempt_count[ids] += 1
        for value in self.committed.values():
            value[ids] = 0
        self.committed["plan_handle"][ids] = -1
        if self.last_state is None:
            self.last_state = {name: value.clone() for name, value in state.items()}
        else:
            for name in self.last_state:
                self.last_state[name][ids] = state[name][ids].clone()

    def _ready(self, state, target_w):
        t, goal = self.thresholds, self.nominal
        ready = self._state_finite(state)
        ready &= (torch.linalg.vector_norm(state["root_position_w"]-goal["root_position_w"], dim=-1) <= t.root_position_m)
        ready &= angle(state["root_quat_wxyz"], goal["root_quat_wxyz"]) <= t.root_orientation_rad
        ready &= (state["joint_position"]-goal["joint_position"]).abs().amax(-1) <= t.joint_guard_rad
        ready &= (torch.linalg.vector_norm(state["glove_position_guard_frame"]-goal["glove_position_guard_frame"], dim=-1).amax(-1) <= t.glove_position_m)
        ready &= angle(state["glove_quat_guard_frame"], goal["glove_quat_guard_frame"]).amax(-1) <= t.glove_orientation_rad
        ready &= torch.linalg.vector_norm(state["root_linear_velocity_w"], dim=-1) <= t.root_speed
        ready &= torch.linalg.vector_norm(state["root_angular_velocity_w"], dim=-1) <= t.root_angular_speed
        ready &= state["joint_velocity"].abs().amax(-1) <= t.joint_speed
        ready &= torch.linalg.vector_norm(state["glove_velocity_guard_frame"], dim=-1).amax(-1) <= t.glove_speed
        ready &= torch.linalg.vector_norm(state["support_anchor_position_w"][..., :2]-goal["support_anchor_position_w"][..., :2], dim=-1).amax(-1) <= t.footprint_m
        ready &= (torch.linalg.vector_norm(state["glove_position_w"]-target_w[:, None], dim=-1) >= t.clearance_m).all(-1)
        ready &= state["support_contact"].all(-1)
        ready &= ~torch.stack([state[name] for name in FLAGS]).any(0)
        ready &= ((state["joint_position"] >= self.limits[:, 0]) & (state["joint_position"] <= self.limits[:, 1])).all(-1)
        ready &= (state["effort"].abs() <= self.caps+t.effort_excess_nm).all(-1)
        return ready

    def _fail(self, mask, category):
        fresh = mask & (self.phase != Phase.FAILED) & (self.phase != Phase.COMPLETE) & (self.phase != Phase.UNINITIALIZED)
        self.failure = torch.where(fresh, category, self.failure)
        self.failure_tick = torch.where(fresh, self.tick, self.failure_tick)
        self.phase = torch.where(fresh, int(Phase.FAILED), self.phase)
        self.failure_count += fresh.to(torch.int64)

    def after_substep(self, tick, state, *, prospective_target_w, cycle_failed, recovery_confirmed):
        """Actual fresh endpoint; flags/loads belong to the preceding interval."""
        if (tick.shape != (self.n,) or tick.dtype != torch.int64 or tick.device != self.device
                or prospective_target_w.shape != (self.n, 3) or prospective_target_w.dtype != self.dtype
                or prospective_target_w.device != self.device):
            raise ValueError("Exact native tick and prospective world target schema required")
        for value in (cycle_failed, recovery_confirmed):
            if value.shape != (self.n,) or value.dtype != torch.bool or value.device != self.device:
                raise ValueError("Boolean per-environment cycle outcomes required")
        finite = self._state_finite(state)
        clock = tick == self.tick+1
        # Failure timestamps describe the sample actually supplied, including a bad clock.
        self.tick = tick.clone()
        self._steps_seen += 1
        self._fail(~clock, int(Failure.CLOCK))
        self._fail(~finite | state["numerical_failure"] | state["contact_overflow"], int(Failure.INVALID_TELEMETRY))
        self._fail(state["external_support"], int(Failure.EXTERNAL_SUPPORT))
        self._fail(state["forbidden_collision"] | state["human_g1_collision"], int(Failure.FORBIDDEN_COLLISION))
        settling = self.phase == Phase.SETTLING
        q = state["root_quat_wxyz"]
        q = q/torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
        tilt = torch.acos((1-2*(q[:, 1].square()+q[:, 2].square())).clamp(-1, 1))
        # Missing support during the tiny initial drop is NOT a settling fall.
        self._fail(settling & ((state["root_position_w"][:, 2] < self.thresholds.root_height_fraction*self.nominal["root_position_w"][:, 2])
                   | (tilt > self.thresholds.root_tilt_rad)), int(Failure.BALANCE))
        drift = torch.linalg.vector_norm(state["support_anchor_position_w"][..., :2]-self.nominal["support_anchor_position_w"][..., :2], dim=-1).amax(-1)
        self._fail(settling & (drift > self.thresholds.footprint_m), int(Failure.FOOT_REPOSITION))
        excess = torch.maximum(self.limits[:, 0]-state["joint_position"], state["joint_position"]-self.limits[:, 1]).amax(-1)
        self._fail(settling & (excess > self.thresholds.joint_excursion_rad), int(Failure.JOINT_LIMIT))
        self._fail(settling & ((state["effort"].abs()-self.caps).amax(-1) > self.thresholds.effort_excess_nm), int(Failure.EFFORT_LIMIT))
        ready = (self.phase == Phase.SETTLING) & self._ready(state, prospective_target_w) & torch.isfinite(prospective_target_w).all(-1)
        self.quiet_intervals = torch.where(ready & self.previous_ready, self.quiet_intervals+1, torch.zeros_like(tick))
        self.previous_ready = ready
        active = (self.phase == Phase.STRIKE) | (self.phase == Phase.RECOVERY)
        self._fail(active & cycle_failed, int(Failure.CYCLE_FAILURE))
        cycle_time = self.reference_tick.to(self.dtype)*self.dt
        enter_recovery = (self.phase == Phase.STRIKE) & (cycle_time >= self.committed["outward_end_s"])
        self.phase = torch.where(enter_recovery, int(Phase.RECOVERY), self.phase)
        completed = ((self.phase == Phase.RECOVERY) & recovery_confirmed
            & (cycle_time >= self.committed["outward_end_s"]+self.thresholds.dwell_s)
            & (cycle_time <= self.committed["requested_ttc_s"]+4.))
        self.phase = torch.where(completed, int(Phase.COMPLETE), self.phase)
        self.complete_count += completed.to(torch.int64)
        self._fail(((self.phase == Phase.STRIKE) | (self.phase == Phase.RECOVERY))
                   & (cycle_time > self.committed["requested_ttc_s"]+4.), int(Failure.RECOVERY_TIMEOUT))
        self.last_state = {name: value.clone() for name, value in state.items()}

    def boundary(self, packet):
        """Commit before next actor observations; no host tensor reads or planning.

        status/plan_matches_guard_context are verified planner-adapter inputs,
        not a feasibility assertion made by this reducer. A packet's plan handle
        indexes immutable externally hash-verified references.
        """
        if self._steps_seen % 4 or self._last_boundary == self._steps_seen or self.last_state is None:
            raise ValueError("Exactly one command decision per 20 ms boundary required")
        for name, expected in self.committed.items():
            value = packet.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype or value.device != self.device:
                raise ValueError(f"Malformed launch packet {name}")
        for name, dtype in (("status", torch.int64), ("plan_matches_guard_context", torch.bool)):
            value = packet.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != (self.n,) or value.dtype != dtype or value.device != self.device:
                raise ValueError(f"Malformed launch packet {name}")
        self._last_boundary = self._steps_seen
        pending = self.phase == Phase.SETTLING
        status = packet["status"]
        self._fail(pending & (status == PlanStatus.REJECTED), int(Failure.REQUEST_REJECTED))
        self._fail(pending & (status == PlanStatus.FAILED), int(Failure.PLANNER_FAILED))
        finite = torch.isfinite(packet["target_w"]).all(-1) & torch.isfinite(packet["g1_launch_state"]).all(-1)
        valid_packet = (finite & torch.isfinite(packet["requested_ttc_s"]) & torch.isfinite(packet["outward_end_s"])
            & (packet["requested_ttc_s"] > 0) & (packet["outward_end_s"] >= packet["requested_ttc_s"])
            & (packet["plan_handle"] >= 0) & (packet["plan_digest"] != 0).any(-1)
            & (packet["family_id"] >= 0) & (packet["family_id"] < 4))
        self._fail((self.phase == Phase.SETTLING) & (((status < 0) | (status > 3))
                   | ((status == PlanStatus.ADMITTED) & ~valid_packet)), int(Failure.INVALID_PLAN_PACKET))
        ready = (self.quiet_intervals >= self.config.quiet_ticks) & self._ready(self.last_state, packet["target_w"])
        launch = ((self.phase == Phase.SETTLING) & ready & (status == PlanStatus.ADMITTED)
                  & valid_packet & packet["plan_matches_guard_context"])
        for name in self.committed:
            condition = launch.reshape(self.n, *([1]*(self.committed[name].ndim-1)))
            self.committed[name] = torch.where(condition, packet[name], self.committed[name]).clone()
        for name in self.launch_guard:
            condition = launch.reshape(self.n, *([1]*(self.launch_guard[name].ndim-1)))
            self.launch_guard[name] = torch.where(condition, self.last_state[name], self.launch_guard[name]).clone()
        self.launch_tick = torch.where(launch, self.tick, self.launch_tick)
        self.phase = torch.where(launch, int(Phase.STRIKE), self.phase)
        self.launch_count += launch.to(torch.int64)
        expired = (self.phase == Phase.SETTLING) & (self.tick >= self.config.maximum_settling_ticks)
        self._fail(expired & ~ready, int(Failure.GUARD_TIMEOUT))
        self._fail(expired & ready & (status != PlanStatus.ADMITTED), int(Failure.PLAN_UNAVAILABLE))
        self._fail(expired & ready, int(Failure.PLAN_CONTEXT_MISMATCH))
        return launch.clone()

    @property
    def launched(self):
        return self.launch_tick >= 0

    @property
    def reference_tick(self):
        return torch.where(self.launched, self.tick-self.launch_tick, torch.zeros_like(self.tick))

    @property
    def active_cycle(self):
        return (self.phase == Phase.STRIKE) | (self.phase == Phase.RECOVERY)

    def recovery_goal(self):
        """Nominal posture/guard, actual ready footprint; no physical state writes."""
        goal = {name: value.clone() for name, value in self.nominal.items()}
        goal["support_anchor_position_w"] = self.launch_guard["support_anchor_position_w"].clone()
        return goal

    def record_environment_end(self, terminated, timed_out):
        """Count inherited termination before metrics receipt and command reset.

        Pass actual TerminationManager results. This never requests a simulator
        reset and never invents a punch launch for an unready terminated row.
        """
        if self._steps_seen % 4:
            raise ValueError("Environment termination must be recorded at a policy boundary")
        for value in (terminated, timed_out):
            if not isinstance(value, torch.Tensor) or value.shape != (self.n,) or value.dtype != torch.bool or value.device != self.device:
                raise ValueError("Actual Boolean per-row environment outcomes required")
        self.environment_terminated |= terminated
        self.environment_timed_out |= timed_out
        self._fail(terminated, int(Failure.ENVIRONMENT_TERMINATION))
        self._fail(timed_out, int(Failure.ENVIRONMENT_TIMEOUT))

    def snapshot(self, env_ids=None):
        """Selected-row copy; aggregate counters remain explicitly global."""
        ids = slice(None) if env_ids is None else self._ids(env_ids)
        def take(value):
            return value[ids].clone()
        return {"phase": take(self.phase), "request_id": take(self.request_id), "physics_tick": take(self.tick),
            "launch_tick": take(self.launch_tick), "launched": take(self.launched), "reference_tick": take(self.reference_tick),
            "launch_time_s": torch.where(take(self.launched), take(self.time_origin_s)+take(self.launch_tick).double()*self.dt,
                                         torch.full_like(take(self.time_origin_s), torch.nan)),
            "failure_category": take(self.failure), "failure_tick": take(self.failure_tick),
            "environment_terminated": take(self.environment_terminated), "environment_timed_out": take(self.environment_timed_out),
            "attempt_count": take(self.attempt_count), "launch_count": take(self.launch_count),
            "failure_count": take(self.failure_count), "complete_count": take(self.complete_count),
            "counts": {name.lower(): (self.phase == phase).sum() for name, phase in Phase.__members__.items()},
            "failure_counts": {name.lower(): (self.failure == category).sum() for name, category in Failure.__members__.items()},
            "global_cumulative_counts": {"attempts": self.attempt_count.sum(), "launches": self.launch_count.sum(),
                "failures": self.failure_count.sum(), "recoveries": self.complete_count.sum()},
            "commitment": {name: take(value) for name, value in self.committed.items()},
            "nominal_guard": {name: take(value) for name, value in self.nominal.items()},
            "actual_launch_guard": {name: take(value) for name, value in self.launch_guard.items()}}

    def terminal_receipt(self, env_ids=None):
        result = self.snapshot(env_ids)
        ids = slice(None) if env_ids is None else self._ids(env_ids)
        self.logged[ids] |= (self.phase[ids] == Phase.COMPLETE) | (self.phase[ids] == Phase.FAILED)
        return result
