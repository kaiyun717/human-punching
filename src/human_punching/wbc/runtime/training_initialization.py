"""Explicit training-only frame-zero exposure, separate from strict readiness.

This module never writes simulator state. The ordinary WBC episode reset has
already selected and written one frame-zero reference. Some training episodes
start following that reference immediately; others use the strict guard clock.
No interior RSI, assistance, mid-cycle replacement or acceptance claim is made.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class TrainingInitializationCfg:
    reference_start_probability: float = 0.0
    initial_support_grace_ticks: int = 0

    def __post_init__(self):
        value = self.reference_start_probability
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("Training reference-start probability must be finite in [0, 1]")
        ticks = self.initial_support_grace_ticks
        if type(ticks) is not int or not 0 <= ticks <= 4 or (ticks and value == 0):
            raise ValueError("Initial support grace requires enabled training starts and at most four physics ticks")

    @property
    def enabled(self):
        return self.reference_start_probability > 0


def require_strict_initialization(config):
    """Acceptance/strict-panel callers must reject an enabled training mode."""
    if not isinstance(config, TrainingInitializationCfg) or config.enabled:
        raise ValueError("Strict evaluation requires default guard-only initialization")


class TrainingInitializer:
    """Per-reset mixture and immutable first-observation command preparation."""

    def __init__(self, n, device, config=TrainingInitializationCfg()):
        if not isinstance(config, TrainingInitializationCfg):
            raise TypeError("Explicit TrainingInitializationCfg required")
        self.config = config
        self.selected = torch.zeros(n, dtype=torch.bool, device=device)
        self.prepared = torch.zeros_like(self.selected)
        self.reference_starts_count = torch.zeros(n, dtype=torch.int64, device=device)
        self.support_seen = torch.zeros_like(self.selected)
        self.grace_ticks = torch.zeros_like(self.reference_starts_count)
        self.grace_total_ticks = torch.zeros_like(self.reference_starts_count)
        self.packet = None

    def prepare(self, ids, packet):
        # Do not consume RNG in the default mode: old strict sampling stays exact.
        probability = self.config.reference_start_probability
        self.selected[ids] = (torch.rand(len(ids), device=self.selected.device) < probability
                              if 0 < probability < 1 else probability == 1)
        self.prepared[ids] = True
        self.support_seen[ids] = False
        self.grace_ticks[ids] = 0
        if self.packet is None:
            self.packet = {name: value.clone() for name, value in packet.items()}
        else:
            if set(packet) != set(self.packet):
                raise ValueError("Prepared training command schema changed")
            for name, value in packet.items():
                self.packet[name][ids] = value[ids].clone()

    def pending_start(self, pending_reset):
        return self.selected & self.prepared & pending_reset

    def freeze_pending(self, packet, pending_reset):
        selected = self.pending_start(pending_reset)
        if self.packet is None:
            return packet
        return {name: torch.where(selected.reshape(-1, *([1]*(value.ndim-1))),
                                  self.packet[name], value).clone()
                for name, value in packet.items()}

    def initialize(self, ids, readiness, events, packet, state, actor_context):
        """Arm diagnostic training histories from the real initialized state.

        Unlike strict ``events.launch``, absent contact during the initial tiny
        floor gap is allowed. All other instantaneous guard checks remain. The
        200 ms quiet dwell is never fabricated. Real subsequent contact, limit,
        balance and recovery checks remain owned by the existing reducers.
        """
        from human_punching.wbc.telemetry.event_state import FLAGS
        from human_punching.wbc.telemetry.readiness_clock import Phase, PlanStatus

        if self.config.enabled and not self.prepared[ids].all():
            raise ValueError("Enabled training initialization must be prepared before the first observation")
        selected = ids[self.selected[ids]]
        if not selected.numel():
            return selected
        if (not self.config.enabled or not self.prepared[selected].all()
                or not (readiness.tick[selected] == 0).all()
                or not (readiness.phase[selected] == Phase.SETTLING).all()
                or events.window_count not in (None, 4) or events.armed[selected].any()):
            raise ValueError("Training reference start requires one prepared ordinary frame-zero reset")
        for name, expected in readiness.committed.items():
            value = packet.get(name)
            if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                    or value.dtype != expected.dtype or value.device != expected.device):
                raise ValueError(f"Malformed training reference command {name}")
        for name, dtype in (("status", torch.int64), ("plan_matches_guard_context", torch.bool)):
            value = packet.get(name)
            if (not isinstance(value, torch.Tensor) or value.shape != (readiness.n,)
                    or value.dtype != dtype or value.device != readiness.device):
                raise ValueError(f"Malformed training admission field {name}")
        valid_packet = ((packet["status"] == PlanStatus.ADMITTED)
            & packet["plan_matches_guard_context"]
            & torch.isfinite(packet["target_w"]).all(-1)
            & torch.isfinite(packet["g1_launch_state"]).all(-1)
            & torch.isfinite(packet["requested_ttc_s"]) & (packet["requested_ttc_s"] > 0)
            & torch.isfinite(packet["outward_end_s"])
            & (packet["outward_end_s"] >= packet["requested_ttc_s"])
            & (packet["family_id"] >= 0) & (packet["family_id"] < 4)
            & (packet["plan_handle"] >= 0) & (packet["plan_digest"] != 0).any(-1))
        valid_state = (events._check(state)
            & events._guard(state, goal=readiness.nominal, target=packet["target_w"])
            & (state["joint_position"] >= readiness.limits[:, 0]).all(-1)
            & (state["joint_position"] <= readiness.limits[:, 1]).all(-1)
            & (state["effort"].abs() <= readiness.caps+events.thresholds.effort_excess_nm).all(-1)
            & (state["root_position_w"][:, 2] > 0)
            & (events._tilt(state["root_quat_wxyz"]) <= events.thresholds.root_tilt_rad)
            & ~torch.stack([state[name] for name in FLAGS]).any(0))
        if not (valid_packet & valid_state)[selected].all():
            raise ValueError("Training start requires an admitted finite clear frame-zero guard")
        # Commit after all selected rows validate. No simulator arrays are touched.
        for name in readiness.committed:
            readiness.committed[name][selected] = packet[name][selected].clone()
        for name in readiness.launch_guard:
            readiness.launch_guard[name][selected] = state[name][selected].clone()
        readiness.launch_tick[selected] = 0
        readiness.phase[selected] = Phase.STRIKE
        readiness.launch_count[selected] += 1
        actor_context[selected] = packet["actor_g1_context"][selected].clone()
        hand = ((packet["family_id"] == 1) | (packet["family_id"] == 3)).long()
        events.target[selected] = packet["target_w"][selected].clone()
        events.hand[selected] = hand[selected]
        events.ttc[selected] = packet["requested_ttc_s"][selected]
        events.outward_end[selected] = packet["outward_end_s"][selected]
        for name in events.goal:
            events.goal[name][selected] = readiness.nominal[name][selected].clone()
            events.initial_guard[name][selected] = state[name][selected].clone()
        events.goal["support_anchor_position_w"][selected] = state["support_anchor_position_w"][selected].clone()
        events.initial_root_height[selected] = state["root_position_w"][selected, 2]
        events.armed[selected] = True
        events.active[selected] = True
        events._armed_count += selected.numel()
        events.previous_guard[selected] = False  # no invented quiet/support history
        events.previous_anchor[selected] = state["support_anchor_position_w"][selected]
        events.previous_anchor_velocity[selected] = state["support_anchor_velocity_w"][selected]
        self.support_seen[selected] = state["support_contact"][selected].any(-1)
        self.reference_starts_count[selected] += 1
        return selected

    def support_grace(self, tick, state, active):
        """Suppress only initial no-support fall, never change observed contacts.

        The concession ends permanently at the first actual supporting foot,
        or after the configured (at most four) native endpoints. Counts measure
        endpoints whose no-support predicate was actually suppressed.
        """
        supported = state["support_contact"].any(-1)
        self.support_seen |= self.selected & active & supported
        grace = (self.selected & active & ~self.support_seen & ~supported
            & (tick >= 1) & (tick <= self.config.initial_support_grace_ticks))
        self.grace_ticks += grace.long()
        self.grace_total_ticks += grace.long()
        return grace

    def receipt(self, ids, readiness):
        selected = self.selected[ids]
        return {"schema": "human_wbc_training_initialization_v1", "config": asdict(self.config),
            "start_type_codes": selected.long().clone(),
            "start_type_names": {"0": "strict_guard_settling", "1": "training_reference_frame_zero"},
            "strict_launch_eligible": (~selected).clone(),
            "strict_ready_launch": (readiness.launched[ids] & ~selected).clone(),
            "training_reference_start": (readiness.launched[ids] & selected).clone(),
            "reference_start_counts": self.reference_starts_count[ids].clone(),
            "strict_ready_launch_counts": (readiness.launch_count-self.reference_starts_count)[ids].clone(),
            "initial_support_grace_ticks_limit": self.config.initial_support_grace_ticks,
            "initial_support_grace_ticks_used": self.grace_ticks[ids].clone(),
            "initial_support_grace_ticks_cumulative": self.grace_total_ticks[ids].clone(),
            "initial_support_seen": self.support_seen[ids].clone(),
            "physical_reset_only": True, "interior_rsi": False}
