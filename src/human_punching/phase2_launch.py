"""Closed-protocol launch admission for an unchanged cached reference.

This helper checks measured continuation of a declared independent start and
prefix. It never changes native state, chooses another command, or requires
physical joints/velocities to equal reference knots. Its admissible context is
the declared initial-state/controller/prefix protocol, not arbitrary live states.
The caller must qualify that protocol and supply every persistent native/adapter
field relevant to planning equality. This is not a saved-state restore adapter.
"""
from __future__ import annotations

import hashlib
import copy
import json
import math
import time

import numpy as np

VERSION = "task10b_closed_prefix_exact_target_launch_v1"
REQUEST_FIELDS = ("trial_id", "request_identity", "family", "requested_ttc_s", "style",
    "candidate_id", "reference_sha256", "checkpoint_sha256", "target_candidate_identity",
    "snapshot_payload_sha256", "snapshot_slot_id", "placement", "head_point_velocity_w", "target_w")
TARGET_FIELDS = ("trial_id", "request_identity", "target_candidate_identity", "snapshot_payload_sha256",
    "snapshot_slot_id", "placement", "head_point_velocity_w", "target_w")
ARRAY_FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied",
    "previous_raw_action", "ticks", "reference_ids", "reset_counts", "support", "time", "phase_denominators_s")
# The original evaluator calls sim.forward before each actor boundary. That
# refresh may replace warmstart; it must remain identical after boundary capture.
FORWARD_REFRESH_FIELDS = frozenset(("qacc_warmstart",))
CONTROLLER_ADVANCE_FIELDS = frozenset(("previous_raw_action", "physics_steps", "actor_updates",
    "ticks", "rejected", "launched"))


class ClosedProtocolViolation(ValueError):
    """The observed trajectory no longer belongs to the declared prefix."""


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _identity(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()).hexdigest()


def _request(row):
    if any(key not in row for key in REQUEST_FIELDS):
        raise ValueError("A complete immutable command/target request is required")
    result = _plain({key: row[key] for key in REQUEST_FIELDS})
    if result["family"] not in ("jab", "cross", "left_hook", "right_hook"):
        raise ValueError("Unknown specialist family")
    if (isinstance(result["requested_ttc_s"], bool) or not isinstance(result["requested_ttc_s"], (int,float))
            or not math.isfinite(result["requested_ttc_s"])
            or result["requested_ttc_s"] < .5):
        raise ValueError("Finite unchanged requested TTC >= .5 s required")
    target = np.asarray(result["target_w"], float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Exact finite target coordinates required")
    for key in ("request_identity", "reference_sha256", "checkpoint_sha256", "target_candidate_identity", "snapshot_payload_sha256"):
        value = result[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Request and payload identities must be lower-case SHA256")
    for key in ("trial_id", "style", "candidate_id"):
        if not isinstance(result[key], str) or not result[key]:
            raise ValueError("Explicit trial, style and candidate identities required")
    if (type(result["snapshot_slot_id"]) is not int or result["snapshot_slot_id"] < 0
            or not isinstance(result["placement"], dict)
            or set(result["placement"]) != {"forward_m", "left_m", "yaw_rad"}):
        raise ValueError("Explicit canonical snapshot slot and fixed placement required")
    placement = result["placement"]
    if (any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v)
            for v in placement.values())
            or abs(placement["forward_m"]) > .05 or abs(placement["left_m"]) > .05
            or abs(placement["yaw_rad"]) > math.radians(5.)):
        raise ValueError("Placement exceeds the initial declared candidate bounds")
    velocity = np.asarray(result["head_point_velocity_w"], float)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError("Measured registered head-point Jacobian velocity required")
    return result


def snapshot_identity(snapshot):
    """Bind all supplied fields, including opaque native/controller extras.

    Additional native state must be supplied when present. Equality here cannot
    certify omitted fields or turn qpos/qvel into a full native restore snapshot.
    """
    digest = hashlib.sha256()
    for key, value in sorted(snapshot.items()):
        digest.update(key.encode())
        if isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode()); digest.update(str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(json.dumps(_plain(value), sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


class ClosedProtocolLaunchGuard:
    def __init__(self, expected_qpos, expected_qvel, requests, launch_ticks,
                 phase_denominators_s, expected_reference_ids=None):
        self._q0, self._v0 = (np.asarray(value, np.float32).copy() for value in (expected_qpos, expected_qvel))
        self.n = len(self._q0)
        if self._q0.shape != (self.n, 55) or self._v0.shape != (self.n, 54) or not self.n:
            raise ValueError("One complete declared initial state per world is required")
        if not np.isfinite(self._q0).all() or not np.isfinite(self._v0).all():
            raise ValueError("Initial state must be finite")
        self._requests = tuple(_request(row) for row in requests)
        if len(self._requests) != self.n or len({row["trial_id"] for row in self._requests}) != self.n:
            raise ValueError("One unique trial request per world required")
        self._launch = np.asarray(launch_ticks).copy()
        self._duration = np.asarray(phase_denominators_s, np.float32).copy()
        self._ids = np.zeros(self.n, np.int64) if expected_reference_ids is None else np.asarray(expected_reference_ids).copy()
        if (self._launch.shape != (self.n,) or self._launch.dtype.kind not in "iu"
                or np.any(self._launch < 0) or np.any(self._launch % 4)
                or self._duration.shape != (self.n,) or not np.isfinite(self._duration).all()
                or np.any(self._duration <= self._launch*.005) or self._ids.shape != (self.n,)
                or self._ids.dtype.kind not in "iu" or np.any(self._ids < 0)):
            raise ValueError("Explicit actor-boundary launch and full reference denominator required")
        self._last = None
        self._steps = 0
        self._initial_time = None
        self._completed_action = np.zeros((self.n, 48), np.float32)
        self._held_action = None
        self._step_open = False
        self._decisions = [None] * self.n
        self._rejected = np.zeros(self.n, bool)
        self._pending = None
        self._boundary_identity = None
        self._boundary_checked_steps = None
        self._snapshot_keys = None

    @property
    def decisions(self):
        return json.loads(json.dumps(self._decisions))

    def assert_requests(self, current_requests):
        """Enforce exact frozen family/TTC/style/checkpoint/target/reference.

        Call at every actor boundary through recovery. The request is never
        replaced by an approximate or nearest cached command.
        """
        try:
            current = tuple(_request(row) for row in current_requests)
        except (TypeError, ValueError) as error:
            raise ClosedProtocolViolation("Committed command or reference identity changed") from error
        if current != self._requests:
            raise ClosedProtocolViolation("Committed command or reference identity changed")
        return True

    def _snapshot(self, source):
        if any(key not in source for key in ARRAY_FIELDS+ ("physics_steps", "actor_updates")):
            raise ClosedProtocolViolation("Incomplete measured native/controller snapshot")
        # Own copies; a caller cannot alter the prior post-state by mutating its input.
        result = copy.deepcopy(source)
        if self._snapshot_keys is not None and set(result) != self._snapshot_keys:
            raise ClosedProtocolViolation("Measured native/controller field inventory changed")
        shapes = {"qpos":(self.n,55), "qvel":(self.n,54), "ctrl":(self.n,48),
            "qacc_warmstart":(self.n,54), "qfrc_applied":(self.n,54),
            "previous_raw_action":(self.n,48), "support":(self.n,2)}
        shapes.update({key:(self.n,) for key in ("ticks", "reference_ids", "reset_counts", "time", "phase_denominators_s")})
        for key, shape in shapes.items():
            result[key] = np.asarray(result[key]).copy()
            if result[key].shape != shape or not np.isfinite(result[key]).all():
                raise ClosedProtocolViolation("Malformed or nonfinite snapshot field: " + key)
        force = np.asarray(result["xfrc_applied"])
        if force.ndim != 3 or force.shape[0] != self.n or force.shape[2] != 6 or not np.isfinite(force).all():
            raise ClosedProtocolViolation("Malformed applied body-force snapshot")
        result["xfrc_applied"] = force.copy()
        for key in ("ticks", "reference_ids", "reset_counts"):
            if result[key].dtype.kind not in "iu":
                raise ClosedProtocolViolation("Integral native/controller counter required: " + key)
        if result["support"].dtype.kind != "b":
            raise ClosedProtocolViolation("Current support flags must be explicit booleans")
        if "act" in result:
            act = np.asarray(result["act"])
            if act.ndim != 2 or act.shape[0] != self.n or not np.isfinite(act).all():
                raise ClosedProtocolViolation("Malformed actuator activation state")
            result["act"] = act.copy()
        for key in ("physics_steps", "actor_updates"):
            if isinstance(result[key], bool) or not isinstance(result[key], (int, np.integer)) or result[key] < 0:
                raise ClosedProtocolViolation("Invalid executed counter: " + key)
            result[key] = int(result[key])
        if (not np.array_equal(result["reference_ids"], self._ids)
                or not np.array_equal(result["phase_denominators_s"].astype(np.float32), self._duration)
                or not np.all(result["reset_counts"] == 1)):
            raise ClosedProtocolViolation("Reference, full phase denominator or reset count changed")
        return result

    def initialize(self, snapshot):
        if self._last is not None:
            raise ClosedProtocolViolation("A closed trial may be initialized only once")
        state = self._snapshot(snapshot)
        if (not np.array_equal(state["qpos"], self._q0) or not np.array_equal(state["qvel"], self._v0)
                or np.any(state["previous_raw_action"]) or np.any(state["ctrl"])
                or np.any(state["ticks"]) or state["physics_steps"] or state["actor_updates"]):
            raise ClosedProtocolViolation("Declared independent initial state/history differs")
        self._last = state
        self._snapshot_keys = set(state)
        self._initial_time = state["time"].copy()
        return dict(version=VERSION, initial_snapshot_sha256=snapshot_identity(state),
            exact_native_float32_initialization=True, full_saved_state_restore=False)

    def _continuity(self, snapshot, *, forward_refresh=False):
        if self._last is None:
            raise ClosedProtocolViolation("Initialize the declared start before observing its prefix")
        state = self._snapshot(snapshot)
        preserved = set(state)-CONTROLLER_ADVANCE_FIELDS
        if forward_refresh:
            preserved -= FORWARD_REFRESH_FIELDS
        for key in preserved:
            if snapshot_identity({key:state[key]}) != snapshot_identity({key:self._last[key]}):
                raise ClosedProtocolViolation("Uninterrupted prefix state changed outside integration: " + key)
        if state["physics_steps"] != self._steps or not np.array_equal(state["ticks"], self._last["ticks"]):
            raise ClosedProtocolViolation("Executed prefix/reference counters changed")
        if (state["actor_updates"] != self._steps//4
                or not np.array_equal(state["previous_raw_action"], self._completed_action)):
            raise ClosedProtocolViolation("Raw action/history or actor counter changed")
        for key, expected in (("rejected", self._rejected),
                ("launched", np.asarray([d is not None and d["accepted"] for d in self._decisions]))):
            if key in state and (np.asarray(state[key]).dtype.kind != "b"
                    or not np.array_equal(state[key], expected)):
                raise ClosedProtocolViolation("Launch disposition changed outside the immutable gate: " + key)
        return state

    def actor_boundary(self, snapshot):
        if self._steps % 4 or self._step_open:
            raise ClosedProtocolViolation("Launch admission must precede a complete actor interval")
        state = self._continuity(snapshot, forward_refresh=True)
        pending = np.array([decision is None for decision in self._decisions]) & (state["ticks"] == self._launch)
        if np.any(np.array([decision is None for decision in self._decisions]) & (state["ticks"] > self._launch)):
            raise ClosedProtocolViolation("A declared launch boundary was missed")
        self._pending = pending
        self._boundary_identity = snapshot_identity(state)
        self._boundary_checked_steps = self._steps
        self._last = state
        return pending.copy()

    def before_step(self, snapshot, action):
        if self._step_open:
            raise ClosedProtocolViolation("Previous physics interval has no post-state")
        state = self._continuity(snapshot)
        action = np.asarray(action)
        if action.shape != (self.n,48) or not np.isfinite(action).all():
            raise ClosedProtocolViolation("One finite raw actor output per world required")
        if self._steps % 4 == 0:
            if self._boundary_checked_steps != self._steps:
                raise ClosedProtocolViolation("The current actor boundary has not been admitted")
            if self._pending is not None and np.any(self._pending):
                raise ClosedProtocolViolation("Outward integration cannot precede final commitment/rejection")
            self._held_action = action.copy()
        elif not np.array_equal(action, self._held_action):
            raise ClosedProtocolViolation("Raw residual changed within its four-step hold")
        self._last = state
        self._step_open = True

    def after_step(self, snapshot, action):
        if not self._step_open:
            raise ClosedProtocolViolation("Completed interval has no matching pre-state")
        state = self._snapshot(snapshot)
        if (not np.array_equal(action, self._held_action) or state["physics_steps"] != self._steps+1
                or state["actor_updates"] != self._steps//4
                or not np.array_equal(state["ticks"], self._last["ticks"]+(~self._rejected).astype(int))
                or not np.allclose(state["time"]-self._last["time"], .005, atol=2.e-6, rtol=0)
                or not np.array_equal(state["previous_raw_action"], self._completed_action)):
            raise ClosedProtocolViolation("Completed 5ms interval/history counters differ")
        self._steps += 1
        self._last = state
        if self._steps % 4 == 0:
            self._completed_action = self._held_action.copy()
        self._step_open = False

    def finalize(self, before, after, receipts, readiness, *, current_requests,
                 planning_started_perf_counter, callback_wall_s, cache_lookup_wall_s=0.):
        """Commit/reject pending worlds without altering any supplied state.

        Callback work is explicitly frozen-snapshot access, not recaptured live
        G1 execution. Planning pause is wall time while simulation time is held.
        """
        if self._pending is None or not np.any(self._pending) or self._steps % 4:
            raise ClosedProtocolViolation("No pending actor-boundary launch request")
        first, last = self._snapshot(before), self._snapshot(after)
        first_id, last_id = snapshot_identity(first), snapshot_identity(last)
        if len(receipts) != self.n or len(current_requests) != self.n:
            raise ValueError("One target receipt/current request per world required")
        ready = np.asarray(readiness)
        if ready.shape != (self.n,) or ready.dtype.kind != "b":
            raise ValueError("Current accepted readiness must be an explicit boolean per world")
        for value in (planning_started_perf_counter, callback_wall_s, cache_lookup_wall_s):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Finite nonnegative measured wall clocks required")
        common = ("planning_started_from_different_boundary_state" if first_id != self._boundary_identity else
                  "physical_or_controller_state_changed_during_planning" if first_id != last_id else None)
        reasons = [None]*self.n
        for world in np.flatnonzero(self._pending):
            expected = self._requests[world]
            reason = common
            try:
                same_request = _request(current_requests[world]) == expected
            except (ValueError, TypeError):
                same_request = False
            target = receipts[world]
            target_matches = (isinstance(target, dict) and
                all(_plain(target.get(key)) == expected[key] for key in TARGET_FIELDS) and
                type(target.get("capture_episode_tick")) is int and
                target.get("capture_episode_tick") == self._steps and
                not isinstance(target.get("input_episode_time_s"), bool) and
                target.get("input_episode_time_s") == self._steps*.005 and
                target.get("head_geometry") == "head_collision" and
                target.get("head_velocity_method") == "jacobian_qvel" and
                target.get("source_kind") == "fixed_canonical_snapshot" and target.get("no_prediction") is True)
            if reason is None:
                reason = ("command_or_reference_identity_changed" if not same_request else
                    "launch_target_input_mismatch_or_stale" if not target_matches else
                    "current_positional_support_readiness_failed" if not ready[world] or not first["support"][world].all() else None)
            reasons[world] = reason
        elapsed = time.perf_counter()-planning_started_perf_counter
        if elapsed < 0 or callback_wall_s+cache_lookup_wall_s > elapsed+1.e-6:
            raise ValueError("Measured component clocks do not fit total finalization time")
        for world in np.flatnonzero(self._pending):
            reason = reasons[world] or ("final_planning_late" if elapsed > .020 else None)
            accepted = reason is None
            self._rejected[world] = not accepted
            self._decisions[world] = dict(version=VERSION, trial_id=self._requests[world]["trial_id"],
                accepted=accepted, reason=reason, actual_launch_tick=self._steps if accepted else None,
                reasons=[] if reason is None else [reason],
                reference_tick=int(first["ticks"][world]), request=self._requests[world].copy(),
                observed_current_request=_plain(current_requests[world]),
                current_positional_support_ready=bool(ready[world]),
                measured_boundary_state_sha256=first_id, after_planning_state_sha256=last_id,
                target_input_receipt=_plain(receipts[world]), planning_wall_s=elapsed,
                target_callback_wall_s=callback_wall_s, cache_lookup_wall_s=cache_lookup_wall_s,
                simulation_pause_wall_s=elapsed, simulated_time_advanced_s=float(last["time"][world]-first["time"][world]),
                same_continuous_cached_reference=True, exact_joint_return_required=False,
                quiet_hook_gate_added=False, live_g1_state_recaptured=False,
                source_head_point_velocity_used_for_prediction=False,
                native_fields_captured=sorted(first),
                forward_refresh_allowance=sorted(FORWARD_REFRESH_FIELDS),
                full_saved_state_restore=False, physical_domain_qualified=False, hardware_real_time_qualified=False)
        self._pending[:] = False
        return self.decisions
