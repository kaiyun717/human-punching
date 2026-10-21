"""Pure-Torch 5 ms event reducer; runtime integration requires qualification. No simulator calls or learning code.

Inputs at tick k: endpoint kinematics; effort/contact flags from [k-1,k].
Launch and reset methods run outside the hot loop. step() has no host tensor
reads. Window and terminal snapshots are copies; reading never consumes events.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Thresholds:
    contact_m: float = .14
    clearance_m: float = .16
    joint_excursion_rad: float = .02
    effort_excess_nm: float = 1e-4
    root_height_fraction: float = .75
    root_tilt_rad: float = .60
    footprint_m: float = .03
    root_position_m: float = .03
    root_orientation_rad: float = .10
    glove_position_m: float = .03
    glove_orientation_rad: float = math.pi/12
    joint_guard_rad: float = .12
    root_speed: float = .05
    root_angular_speed: float = .12
    joint_speed: float = .50
    glove_speed: float = .15
    dwell_s: float = 1.


KINEMATIC_WIDTHS = {
    "glove_position_w": (2, 3), "root_position_w": (3,), "root_quat_wxyz": (4,),
    "root_linear_velocity_w": (3,), "root_angular_velocity_w": (3,),
    "glove_position_guard_frame": (2, 3), "glove_quat_guard_frame": (2, 4),
    "glove_velocity_guard_frame": (2, 3), "support_anchor_position_w": (2, 3),
    "support_anchor_velocity_w": (2, 3),
}
FLAGS = ("external_support", "forbidden_collision", "human_g1_collision", "numerical_failure", "contact_overflow")
# Deterministic priority only breaks simultaneous events. Keep all simultaneous
# categories in first_failure_mask; -1 means no observed failure. Inactive glove
# clearance is diagnostic (contact is fatal), so it is excluded here.
FAILURE_CATEGORIES = ("invalid_input", "clock_fault", "numerical_failure", "contact_overflow",
    "external_support", "forbidden_collision", "human_g1_collision", "inactive_contact",
    "renewed_contact", "fall", "joint_limit", "effort_limit", "step_or_drift")


def angle(q, reference):
    q = q/torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
    reference = reference/torch.linalg.vector_norm(reference, dim=-1, keepdim=True).clamp_min(torch.finfo(reference.dtype).tiny)
    reference = torch.where((q*reference).sum(-1, keepdim=True) < 0, -reference, reference)
    # Unlike acos(dot), this remains exactly zero for equal float32 inputs.
    return 4*torch.atan2(torch.linalg.vector_norm(q-reference, dim=-1), torch.linalg.vector_norm(q+reference, dim=-1))


class PunchEventState:
    dt = .005
    decimation = 4

    def __init__(self, environments, joint_limits, effort_limits, *, thresholds=Thresholds()):
        if isinstance(environments, bool) or not isinstance(environments, int) or environments < 1:
            raise ValueError("Positive exact environment count required")
        if (joint_limits.ndim != 2 or joint_limits.shape[1] != 2 or effort_limits.shape != joint_limits.shape[:1]
                or joint_limits.dtype not in (torch.float32, torch.float64)
                or effort_limits.dtype != joint_limits.dtype or effort_limits.device != joint_limits.device
                or not torch.isfinite(joint_limits).all() or not torch.isfinite(effort_limits).all()
                or not torch.all(joint_limits[:, 0] < joint_limits[:, 1]) or not torch.all(effort_limits > 0)):
            raise ValueError("Finite ordered limits and positive effort caps of matching dtype/device required")
        self.n, self.j = environments, len(effort_limits)
        self.device, self.dtype, self.thresholds = joint_limits.device, joint_limits.dtype, thresholds
        if any(not math.isfinite(value) or value <= 0 for value in vars(thresholds).values()):
            raise ValueError("Positive finite engineering thresholds required")
        if thresholds.clearance_m <= thresholds.contact_m:
            raise ValueError("Clearance must exceed contact radius")
        self.limits, self.caps = joint_limits.clone(), effort_limits.clone()
        self.armed = self.zeros(dtype=torch.bool)
        self.active = self.zeros(dtype=torch.bool)
        self._armed_count = 0  # changed only in host-side launch/reset boundaries
        self.logged = self.zeros(dtype=torch.bool)
        self.tick = self.zeros(dtype=torch.int64)
        self.target = self.zeros(3)
        self.hand = self.zeros(dtype=torch.int64)
        self.ttc = self.zeros()
        self.outward_end = self.zeros()
        self.goal = {name: self.zeros(*shape) for name, shape in KINEMATIC_WIDTHS.items()}
        self.goal["joint_position"] = self.zeros(self.j)
        self.initial_guard = {name: value.clone() for name, value in self.goal.items()}
        self.initial_root_height = self.zeros()
        self.previous_anchor = self.zeros(2, 3)
        self.previous_anchor_velocity = self.zeros(2, 3)
        self.previous_guard = self.zeros(dtype=torch.bool)
        self.previous_recovery = self.zeros(dtype=torch.bool)
        self.first_contact = self.zeros(dtype=torch.int64)-1
        self.withdrawal = self.zeros(dtype=torch.int64)-1
        self.recovery_start = self.zeros(dtype=torch.int64)-1
        self.recovery_tick = self.zeros(dtype=torch.int64)-1
        self.guard_intervals = self.zeros(dtype=torch.int64)
        self.flags = {name: self.zeros(dtype=torch.bool) for name in (*FLAGS, "invalid_input", "clock_fault", "inactive_contact",
            "inactive_clearance", "renewed_contact", "fall", "joint_limit", "effort_limit", "step_or_drift")}
        self.first_flag_tick = {name: self.zeros(dtype=torch.int64)-1 for name in self.flags}
        self.diagnostic_flags = {name: self.zeros(dtype=torch.bool) for name in (
            "invalid_input", "numerical_failure", "contact_overflow")}
        self.first_failure_tick = self.zeros(dtype=torch.int64)-1
        self.first_failure_category = self.zeros(dtype=torch.int64)-1
        self.first_failure_mask = self.zeros(dtype=torch.int64)
        self.episode_joint_peak = self.zeros()
        self.episode_effort_peak = self.zeros()
        self.episode_effort_abs_peak = self.zeros(self.j)
        self.episode_footprint_peak = self.zeros()
        self.episode_loaded_drift_peak = self.zeros()
        self.slip_distance = self.zeros(2)
        self.saturation_seconds = self.zeros(self.j)
        self.window_count = None
        self._clear_window()

    def _ids(self, env_ids):
        original = torch.as_tensor(env_ids, device=self.device)
        if original.dtype not in (torch.int32, torch.int64):
            raise ValueError("Integer environment IDs required")
        ids = original.to(dtype=torch.long)
        if (ids.ndim != 1 or not ids.numel() or torch.any(ids < 0) or torch.any(ids >= self.n)
                or ids.unique().numel() != ids.numel()):
            raise ValueError("Unique in-range one-dimensional environment IDs required")
        return ids

    def zeros(self, *shape, dtype=None):
        return torch.zeros((self.n, *shape), device=self.device, dtype=dtype or self.dtype)

    def _check(self, state):
        shapes = {**KINEMATIC_WIDTHS, "joint_position": (self.j,), "joint_velocity": (self.j,),
                  "effort": (self.j,), "support_contact": (2,), "support_patch_contact": (2,),
                  **{name: () for name in FLAGS}}
        finite = torch.ones(self.n, device=self.device, dtype=torch.bool)
        for name, tail in shapes.items():
            value = state.get(name)
            boolean = name in FLAGS or name in ("support_contact", "support_patch_contact")
            if (not isinstance(value, torch.Tensor) or value.shape != (self.n, *tail)
                    or value.device != self.device or value.dtype != (torch.bool if boolean else self.dtype)):
                raise ValueError(f"Malformed {name}: exact shape, dtype and device required")
            finite &= torch.isfinite(value).reshape(self.n, -1).all(-1)
        for name in ("root_quat_wxyz", "glove_quat_guard_frame"):
            finite &= (torch.linalg.vector_norm(state[name], dim=-1)-1).abs().reshape(self.n, -1).le(1e-5).all(-1)
        return finite

    def _guard(self, state, *, goal=None, target=None):
        t = self.thresholds
        goal = self.goal if goal is None else goal
        target = self.target if target is None else target
        root = state["root_position_w"]
        distance = torch.linalg.vector_norm(state["glove_position_w"]-target[:, None], dim=-1)
        return ((torch.linalg.vector_norm(root-goal["root_position_w"], dim=-1) <= t.root_position_m)
            & (angle(state["root_quat_wxyz"], goal["root_quat_wxyz"]) <= t.root_orientation_rad)
            & ((state["joint_position"]-goal["joint_position"]).abs().amax(-1) <= t.joint_guard_rad)
            & (torch.linalg.vector_norm(state["glove_position_guard_frame"]-goal["glove_position_guard_frame"], dim=-1).amax(-1) <= t.glove_position_m)
            & (angle(state["glove_quat_guard_frame"], goal["glove_quat_guard_frame"]).amax(-1) <= t.glove_orientation_rad)
            & (torch.linalg.vector_norm(state["root_linear_velocity_w"], dim=-1) <= t.root_speed)
            & (torch.linalg.vector_norm(state["root_angular_velocity_w"], dim=-1) <= t.root_angular_speed)
            & (state["joint_velocity"].abs().amax(-1) <= t.joint_speed)
            & (torch.linalg.vector_norm(state["glove_velocity_guard_frame"], dim=-1).amax(-1) <= t.glove_speed)
            & (torch.linalg.vector_norm(state["support_anchor_position_w"][..., :2]-goal["support_anchor_position_w"][..., :2], dim=-1).amax(-1) <= t.footprint_m)
            & (distance >= t.clearance_m).all(-1))

    def launch(self, env_ids, state, *, target_w, active_hand, ttc_s, outward_end_s, nominal_guard=None):
        """Initialize an externally accepted ready launch; never changes physics.

        Command tensors describe all environments; only env_ids are initialized.
        Partial resets must first preserve terminal_summary then reset_after_log.
        """
        ids = self._ids(env_ids)
        valid = self._check(state)
        if self.window_count not in (None, 4) or torch.any(self.armed[ids]):
            raise ValueError("Launch only at a policy boundary into pending environments")
        for name, value, tail in (("target", target_w, (3,)), ("ttc", ttc_s, ()), ("outward", outward_end_s, ())):
            if value.shape != (self.n, *tail) or value.dtype != self.dtype or value.device != self.device or not torch.isfinite(value[ids]).all():
                raise ValueError(f"Invalid launch {name}")
        if (active_hand.shape != (self.n,) or active_hand.dtype != torch.int64 or active_hand.device != self.device
                or not torch.all((active_hand[ids] == 0) | (active_hand[ids] == 1)) or not torch.all(ttc_s[ids] > 0)
                or not torch.all(outward_end_s[ids] >= ttc_s[ids])):
            raise ValueError("Invalid hand/TTC/outward interval")
        candidate_goal = state if nominal_guard is None else nominal_guard
        goal_valid = torch.ones(self.n, device=self.device, dtype=torch.bool)
        for name, expected in self.goal.items():
            value = candidate_goal.get(name)
            if (not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != self.dtype or value.device != self.device):
                raise ValueError(f"Malformed nominal guard {name}")
            goal_valid &= torch.isfinite(value).reshape(self.n, -1).all(-1)
        for name in ("root_quat_wxyz", "glove_quat_guard_frame"):
            goal_valid &= (torch.linalg.vector_norm(candidate_goal[name], dim=-1)-1).abs().reshape(self.n, -1).le(1e-5).all(-1)
        initial = (valid & goal_valid & self._guard(state, goal=candidate_goal, target=target_w) & state["support_contact"].all(-1)
                   & (state["joint_position"] >= self.limits[:, 0]).all(-1)
                   & (state["joint_position"] <= self.limits[:, 1]).all(-1)
                   & (state["effort"].abs() <= self.caps+self.thresholds.effort_excess_nm).all(-1)
                   & (state["root_position_w"][:, 2] > 0)
                   & (self._tilt(state["root_quat_wxyz"]) <= self.thresholds.root_tilt_rad)
                   & ~torch.stack([state[name] for name in FLAGS]).any(0))
        if not initial[ids].all():
            raise ValueError("Launch requires finite quiet supported guard and both gloves clear")
        # Commit only after every selected row passes; rejection changes nothing.
        self.target[ids], self.hand[ids] = target_w[ids].clone(), active_hand[ids].clone()
        self.ttc[ids], self.outward_end[ids] = ttc_s[ids].clone(), outward_end_s[ids].clone()
        for name in self.goal:
            self.goal[name][ids] = candidate_goal[name][ids].clone()
            self.initial_guard[name][ids] = state[name][ids].clone()
        self.goal["support_anchor_position_w"][ids] = state["support_anchor_position_w"][ids].clone()
        self.initial_root_height[ids] = state["root_position_w"][ids, 2]
        self.armed[ids] = True
        self.active[ids] = True
        self._armed_count += ids.numel()
        self.previous_guard[ids] = True
        self.previous_anchor[ids] = state["support_anchor_position_w"][ids]
        self.previous_anchor_velocity[ids] = state["support_anchor_velocity_w"][ids]

    def _clear_window(self):
        self.window_joint_peak = self.zeros()
        self.window_effort_peak = self.zeros()
        self.window_effort_abs_peak = self.zeros(self.j)
        self.window_loaded_slip_speed_sum = self.zeros()
        self.window_new_contact = self.zeros(dtype=torch.bool)
        self.window_new_recovery = self.zeros(dtype=torch.bool)
        self.window_valid = torch.ones(self.n, device=self.device, dtype=torch.bool)
        self.window_active = self.zeros(dtype=torch.bool)
        self.window_diagnostic_valid = torch.ones(self.n, device=self.device, dtype=torch.bool)

    @staticmethod
    def _tilt(q):
        q = q/torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
        return torch.acos((1-2*(q[:, 1].square()+q[:, 2].square())).clamp(-1, 1))

    def _measurement_valid(self):
        return ~torch.stack([self.flags[name] for name in (
            "invalid_input", "clock_fault", "numerical_failure", "contact_overflow")]).any(0)

    def begin_window(self):
        if self.window_count not in (None, 4):
            raise ValueError("Each policy window needs exactly four prior physics ticks")
        self._clear_window()
        self.window_active = self.active.clone()
        self.window_count = 0

    def step(self, tick, state, *, initial_support_grace=None):
        """One post-integration sample; no CPU tensor extraction in this hot path."""
        if self.window_count is None or not 0 <= self.window_count < 4:
            raise ValueError("begin_window and exactly four substep calls required")
        if tick.shape != (self.n,) or tick.dtype != torch.int64 or tick.device != self.device:
            raise ValueError("Integer per-environment physics tick required")
        if initial_support_grace is not None:
            if (initial_support_grace.shape != (self.n,) or initial_support_grace.dtype != torch.bool
                    or initial_support_grace.device != self.device):
                raise ValueError("Explicit per-row boolean training support grace required")
            # Training initialization owns first-support history and eligibility.
            # Hard bound here prevents a caller extending this beyond 20 ms.
            initial_support_grace = initial_support_grace & (tick >= 1) & (tick <= 4)
        finite = self._check(state)
        active_rows = self.window_active & self.active
        clock = tick == self.tick+1
        self.diagnostic_flags["invalid_input"] |= ~finite
        for name in ("numerical_failure", "contact_overflow"):
            self.diagnostic_flags[name] |= state[name]
        self.window_diagnostic_valid &= finite & ~state["numerical_failure"] & ~state["contact_overflow"]
        self.flags["invalid_input"] |= active_rows & ~finite
        self.flags["clock_fault"] |= active_rows & ~clock
        for name in FLAGS:
            self.flags[name] |= active_rows & state[name]
        valid = active_rows & self._measurement_valid()
        self.window_valid &= ~active_rows | valid
        # Arithmetic remains finite when telemetry is invalid. The permanent
        # validity latch prevents sanitized values from creating later success.
        a = {name: torch.nan_to_num(value, nan=0., posinf=0., neginf=0.)
             if value.dtype != torch.bool else value for name, value in state.items() if isinstance(value, torch.Tensor)}
        t = self.thresholds
        distance = torch.linalg.vector_norm(a["glove_position_w"]-self.target[:, None], dim=-1)
        active = distance.gather(1, self.hand[:, None]).squeeze(1)
        inactive = distance.gather(1, (1-self.hand)[:, None]).squeeze(1)
        elapsed = tick.to(self.dtype)*self.dt
        hit = valid & (self.first_contact < 0) & (elapsed <= self.outward_end) & (active <= t.contact_m)
        self.first_contact = torch.where(hit, tick, self.first_contact)
        self.window_new_contact |= hit
        self.flags["inactive_contact"] |= valid & (inactive <= t.contact_m)
        self.flags["inactive_clearance"] |= valid & (inactive < t.clearance_m)
        self.flags["renewed_contact"] |= valid & (self.withdrawal >= 0) & (distance <= t.contact_m).any(-1)
        withdraw = valid & (self.first_contact >= 0) & (tick > self.first_contact) & (self.withdrawal < 0) & (distance >= t.clearance_m).all(-1)
        self.withdrawal = torch.where(withdraw, tick, self.withdrawal)
        excess = torch.maximum(self.limits[:, 0]-a["joint_position"], a["joint_position"]-self.limits[:, 1]).clamp_min(0).amax(-1)
        effort_excess = (a["effort"].abs()-self.caps).clamp_min(0).amax(-1)
        self.window_joint_peak = torch.maximum(self.window_joint_peak, torch.where(active_rows, excess, 0.))
        self.window_effort_peak = torch.maximum(self.window_effort_peak, torch.where(active_rows, effort_excess, 0.))
        self.episode_joint_peak = torch.maximum(self.episode_joint_peak, torch.where(active_rows, excess, 0.))
        self.episode_effort_peak = torch.maximum(self.episode_effort_peak, torch.where(active_rows, effort_excess, 0.))
        effort_abs = torch.where(active_rows[:, None], a["effort"].abs(), 0.)
        self.window_effort_abs_peak = torch.maximum(self.window_effort_abs_peak, effort_abs)
        self.episode_effort_abs_peak = torch.maximum(self.episode_effort_abs_peak, effort_abs)
        self.flags["joint_limit"] |= active_rows & (excess > t.joint_excursion_rad)
        self.flags["effort_limit"] |= active_rows & (effort_excess > t.effort_excess_nm)
        self.saturation_seconds += ((a["effort"].abs() >= .99*self.caps) & active_rows[:, None]).to(self.dtype)*self.dt
        anchor = a["support_anchor_position_w"]
        drift = torch.linalg.vector_norm(anchor[..., :2]-self.goal["support_anchor_position_w"][..., :2], dim=-1)
        loaded = a["support_patch_contact"] & active_rows[:, None]
        self.episode_footprint_peak = torch.maximum(self.episode_footprint_peak, torch.where(active_rows, drift.amax(-1), 0.))
        self.episode_loaded_drift_peak = torch.maximum(self.episode_loaded_drift_peak, torch.where(loaded, drift, 0.).amax(-1))
        self.slip_distance += torch.where(loaded, torch.linalg.vector_norm(anchor[..., :2]-self.previous_anchor[..., :2], dim=-1), 0.)
        self.window_loaded_slip_speed_sum += torch.where(loaded, torch.linalg.vector_norm(self.previous_anchor_velocity[..., :2], dim=-1), 0.).sum(-1)
        self.flags["step_or_drift"] |= active_rows & (drift.amax(-1) > t.footprint_m)
        tilt = self._tilt(a["root_quat_wxyz"])
        no_support = ~a["support_contact"].any(-1)
        if initial_support_grace is not None:
            no_support &= ~initial_support_grace
        self.flags["fall"] |= active_rows & ((a["root_position_w"][:, 2] < t.root_height_fraction*self.initial_root_height)
            | (tilt > t.root_tilt_rad) | no_support)
        now_guard = (valid & self._guard(a) & (tilt <= t.root_tilt_rad)
            & (excess <= t.joint_excursion_rad) & (effort_excess <= t.effort_excess_nm)
            & a["support_contact"].all(-1)
            & ~torch.stack([a[name] for name in FLAGS]).any(0))
        interval_guard = self.previous_guard & now_guard & a["support_contact"].all(-1)
        self.guard_intervals = torch.where(active_rows, torch.where(interval_guard, self.guard_intervals+1, 0), self.guard_intervals)
        eligible = now_guard & (self.withdrawal >= 0) & (elapsed >= self.outward_end)
        continuation = self.previous_recovery & eligible & a["support_contact"].all(-1)
        self.recovery_start = torch.where(active_rows, torch.where(eligible, torch.where(continuation, self.recovery_start, tick), -1), self.recovery_start)
        duration = (tick-self.recovery_start).to(self.dtype)*self.dt
        recovered = (eligible & (self.recovery_tick < 0) & (duration >= t.dwell_s)
                     & (self.recovery_start.to(self.dtype)*self.dt <= self.ttc+3.))
        self.recovery_tick = torch.where(recovered, tick, self.recovery_tick)
        self.window_new_recovery |= recovered
        for name, flag in self.flags.items():
            self.first_flag_tick[name] = torch.where(active_rows & flag & (self.first_flag_tick[name] < 0), tick, self.first_flag_tick[name])
        mask = torch.zeros_like(tick)
        category = torch.full_like(tick, len(FAILURE_CATEGORIES))
        for number, name in enumerate(FAILURE_CATEGORIES):
            mask |= self.flags[name].to(torch.int64) << number
            category = torch.where(self.flags[name] & (category > number), number, category)
        new_failure = active_rows & (self.first_failure_tick < 0) & (mask != 0)
        self.first_failure_tick = torch.where(new_failure, tick, self.first_failure_tick)
        self.first_failure_category = torch.where(new_failure, category, self.first_failure_category)
        self.first_failure_mask = torch.where(new_failure, mask, self.first_failure_mask)
        self.previous_guard = torch.where(active_rows, now_guard, self.previous_guard)
        self.previous_recovery = torch.where(active_rows, eligible, self.previous_recovery)
        self.previous_anchor = torch.where(active_rows[:, None, None], anchor, self.previous_anchor)
        self.previous_anchor_velocity = torch.where(active_rows[:, None, None], a["support_anchor_velocity_w"], self.previous_anchor_velocity)
        self.tick = torch.where(active_rows, tick, self.tick)
        self.window_count += 1

    def window_summary(self):
        if self.window_count != 4:
            raise ValueError("Incomplete four-tick policy window")
        return {"joint_peak": self.window_joint_peak.clone(), "effort_peak": self.window_effort_peak.clone(),
            "effort_abs_peak_nm": self.window_effort_abs_peak.clone(),
            "measurement_valid": (self.window_active & self.window_valid).clone(),
            "cycle_active": self.window_active.clone(), "diagnostic_measurement_valid": self.window_diagnostic_valid.clone(),
            "loaded_slip_speed_rate": self.window_loaded_slip_speed_sum.clone()/4,
            "new_contact": self.window_new_contact.clone(), "new_recovery": self.window_new_recovery.clone(),
            "contact_bonus_rate": (self.window_new_contact & self.window_valid).to(self.dtype)/.020,
            "recovery_bonus_rate": (self.window_new_recovery & self.window_valid).to(self.dtype)/.020}

    def terminal_summary(self, env_ids=None):
        ids = slice(None) if env_ids is None else self._ids(env_ids)
        self.logged[ids] = True
        contact = self.first_contact[ids] >= 0
        return {"first_contact_tick": self.first_contact[ids].clone(), "withdrawal_tick": self.withdrawal[ids].clone(),
            "first_failure_tick": self.first_failure_tick[ids].clone(),
            "first_failure_category": self.first_failure_category[ids].clone(),
            "first_failure_mask": self.first_failure_mask[ids].clone(),
            "measurement_valid": (self.armed & self._measurement_valid())[ids].clone(), "has_contact": contact,
            "cycle_armed": self.armed[ids].clone(), "cycle_active": self.active[ids].clone(),
            "diagnostic_measurement_valid": (~torch.stack(list(self.diagnostic_flags.values())).any(0))[ids].clone(),
            "diagnostic_flags": {name: value[ids].clone() for name, value in self.diagnostic_flags.items()},
            "nominal_recovery_goal": {name: value[ids].clone() for name, value in self.goal.items()},
            "actual_initial_guard": {name: value[ids].clone() for name, value in self.initial_guard.items()},
            "ttc_error_s": torch.where(contact, self.first_contact[ids].to(self.dtype)*self.dt-self.ttc[ids], 0.),
            "recovery_tick": self.recovery_tick[ids].clone(), "guard_dwell_s": self.guard_intervals[ids].to(self.dtype)*self.dt,
            "joint_peak": self.episode_joint_peak[ids].clone(), "effort_peak": self.episode_effort_peak[ids].clone(),
            "effort_abs_peak_nm": self.episode_effort_abs_peak[ids].clone(),
            "footprint_peak": self.episode_footprint_peak[ids].clone(), "loaded_drift_peak": self.episode_loaded_drift_peak[ids].clone(),
            "slip_distance": self.slip_distance[ids].clone(), "saturation_s": self.saturation_seconds[ids].clone(),
            "flags": {name: value[ids].clone() for name, value in self.flags.items()},
            "first_flag_tick": {name: value[ids].clone() for name, value in self.first_flag_tick.items()}}

    def reset_after_log(self, env_ids):
        ids = self._ids(env_ids)
        if self.window_count not in (None, 4) or not self.logged[ids].all():
            raise ValueError("Preserve terminal summary before a boundary reset")
        armed_count = int(self.armed[ids].sum().item())
        for name in ("armed", "active", "logged", "tick", "previous_guard", "previous_recovery", "guard_intervals", "episode_joint_peak", "initial_root_height",
                     "episode_effort_peak", "episode_effort_abs_peak", "episode_footprint_peak", "episode_loaded_drift_peak", "slip_distance", "saturation_seconds", "first_failure_mask"):
            getattr(self, name)[ids] = 0
        for name in ("first_contact", "withdrawal", "recovery_start", "recovery_tick", "first_failure_tick", "first_failure_category"):
            getattr(self, name)[ids] = -1
        for value in self.flags.values():
            value[ids] = False
        for value in self.first_flag_tick.values():
            value[ids] = -1
        for value in self.diagnostic_flags.values():
            value[ids] = False
        for name in ("target", "hand", "ttc", "outward_end", "previous_anchor", "previous_anchor_velocity"):
            getattr(self, name)[ids] = 0
        for collection in (self.goal, self.initial_guard):
            for value in collection.values():
                value[ids] = 0
        self._armed_count -= armed_count

    def finish_cycles(self, mask):
        """Freeze completed/terminated cycle rows at the ordinary policy boundary."""
        if self.window_count not in (None, 4):
            raise ValueError("Cycle termination must wait for the four-tick boundary")
        if not isinstance(mask, torch.Tensor) or mask.shape != (self.n,) or mask.dtype != torch.bool or mask.device != self.device:
            raise ValueError("Boolean per-environment terminal mask required")
        self.active &= ~mask

    def terminal_receipt(self, env_ids=None):
        """Freeze selected rows, preserve copies, and allow later boundary reset.

        Never-launched failed-readiness rows remain explicitly unarmed. Their
        diagnostic input validity is separate from physical-cycle validity.
        """
        ids = torch.arange(self.n, device=self.device) if env_ids is None else self._ids(env_ids)
        mask = self.zeros(dtype=torch.bool)
        mask[ids] = True
        self.finish_cycles(mask)
        return self.terminal_summary(ids)
