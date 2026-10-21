"""Single-cycle command bridge. No physics steps, state writes, or learning code.

The observer calls begin/endpoint/reset callbacks. The motion command calls
command_boundary BEFORE mjlab computes the next observations. Plan admission,
nominal-guard generation and durable receipt storage are explicit dependencies;
observed reset geometry is never silently promoted to an admitted nominal guard.
Native qualification of this adapter is still required.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from human_punching.wbc.telemetry.event_state import FLAGS, PunchEventState
from human_punching.wbc.telemetry.readiness_clock import Phase, PlanStatus, ReadinessClock, ReadinessConfig
from human_punching.wbc.runtime.training_initialization import TrainingInitializationCfg, TrainingInitializer

ATTRIBUTE = "_human_cycle_runtime"
DERIVATIVES = ("qvel", "body_lin_vel_w", "body_ang_vel_w", "anchor_lin_acc_w", "anchor_ang_acc_w")


def get_cycle_runtime(env):
    bridge = getattr(env, ATTRIBUTE, None)
    if bridge is None:
        raise RuntimeError("The single-cycle readiness bridge was not explicitly installed")
    return bridge


def reference_indices(tick, launched, clip, starts, lengths, offset=0):
    """All look-ahead samples stay at guard frame zero before actual launch."""
    if type(offset) is not int or offset < 0:
        raise ValueError("Reference offset must be a nonnegative physics-tick integer")
    local = torch.where(launched, tick + offset, 0)
    final = lengths[clip]-1
    return starts[clip] + torch.minimum(local, final), (~launched | (local >= final))


class CycleRuntime:
    """Coordinate independent rows without extracting tensors in a 5 ms hook.

    Packet provider returns a full batch at policy boundaries. It must validate
    feasibility, hashes, actual ready-guard compatibility, and frozen G1 context.
    The receipt sink must persist the retained copies or raise before bookkeeping
    is reused (mjlab's metric-reset callback runs after native reset). Launch and
    reset validation may synchronize at 20 ms boundaries;
    after_substep performs tensor operations only.
    """
    def __init__(self, n, limits, caps, *, context_width, packet_provider: Callable,
                 nominal_provider: Callable, request_ids: Callable,
                 reference_gloves: Callable, receipt_sink: Callable,
                 reference_changed: Callable = lambda: None,
                 config=ReadinessConfig(), acceptance_eligible=False,
                 training_initialization=TrainingInitializationCfg()):
        self.readiness = ReadinessClock(n, limits, caps, context_width=context_width, config=config)
        self.events = PunchEventState(n, limits, caps)
        self.packet_provider, self.nominal_provider = packet_provider, nominal_provider
        self.request_ids, self.reference_gloves, self.receipt_sink = request_ids, reference_gloves, receipt_sink
        self.reference_changed = reference_changed
        self.acceptance_eligible = bool(acceptance_eligible)
        self.training_initializer = TrainingInitializer(n, limits.device, training_initialization)
        if self.acceptance_eligible and training_initialization.enabled:
            raise ValueError("Training initialization cannot qualify acceptance attempts")
        self.pending_reset = torch.ones(n, dtype=torch.bool, device=limits.device)
        self.persisted = torch.zeros_like(self.pending_reset)
        self.actor_context = torch.zeros((n, 77), dtype=limits.dtype, device=limits.device)
        self.packet = None
        self._last_command_boundary = -1
        self._receipt_number = 0

    def _state(self, state):
        # Optional observer diagnostics are folded into canonical reducer flags.
        # They may exist only after a native step, not in a fresh reset packet;
        # retaining them would make the next selected-row reset schema-dependent.
        result = dict(state)
        if "root_actuation" in result:
            result["external_support"] = result["external_support"] | result.pop("root_actuation")
        if "clock_failure" in result:
            result["numerical_failure"] = result["numerical_failure"] | result.pop("clock_failure")
        return result

    def _packet(self):
        packet = dict(self.packet_provider())
        frozen = self.readiness.launched & ~self.pending_reset
        # Live caller buffers are prospective inputs only. Even nonfinite later
        # G1 observations cannot interrupt or redirect an already committed row.
        # Immutable reference asset verification remains a separate contract.
        for name, value in self.readiness.committed.items():
            incoming = packet.get(name)
            if (not isinstance(incoming, torch.Tensor) or incoming.shape != value.shape
                    or incoming.dtype != value.dtype or incoming.device != value.device):
                raise ValueError(f"Malformed prospective command field {name}")
            packet[name] = torch.where(frozen.reshape(value.shape[0], *([1]*(value.ndim-1))), value, incoming)
        context = packet.get("actor_g1_context")
        if (not isinstance(context, torch.Tensor) or context.shape != self.actor_context.shape
                or context.dtype != self.actor_context.dtype or context.device != self.actor_context.device):
            raise ValueError("Explicit finite 77-feature G1 actor context required")
        packet["actor_g1_context"] = torch.where(frozen[:, None], self.actor_context, context)
        if not torch.isfinite(packet["actor_g1_context"]).all():
            raise ValueError("Explicit finite 77-feature G1 actor context required")
        # Clone all fields: a caller-owned mutable G1 buffer cannot alter this
        # interval's prospective context, nor any already frozen commitment.
        return self.training_initializer.freeze_pending(
            {name: value.clone() for name, value in packet.items()}, self.pending_reset)

    def prepare_training_reset(self, ids):
        """Called after ordinary command reset, before its first observation."""
        if not self.training_initializer.config.enabled:
            return
        if not self.pending_reset[ids].all():
            raise ValueError("Training initialization may only prepare an ordinary pending reset")
        # Do not reuse a previous episode's pending command for these rows.
        self.training_initializer.prepared[ids] = False
        packet = self._packet()
        self.training_initializer.prepare(ids, packet)
        self.packet = self.training_initializer.freeze_pending(packet, self.pending_reset)

    def after_pending_reset(self, ids, state, tick, time):
        """Observer's first fresh post-initialization state; no readiness credit."""
        state = self._state(state)
        r = self.readiness
        if not (tick[ids] == 0).all():
            raise ValueError("The ordinary reset must initialize the physical interval counter to zero")
        # First initialization has no previous attempt. Later initialization
        # requires the terminal receipt to have been durably accepted by sink.
        previous = ids[r.phase[ids] != Phase.UNINITIALIZED]
        if previous.numel():
            if not self.persisted[previous].all():
                raise RuntimeError("A prior attempt may not disappear during reset")
            self.events.reset_after_log(previous)
        r.start_episode(ids, state, nominal_guard=self.nominal_provider(),
                        request_id=self.request_ids(), time_origin_s=time.to(torch.float64))
        self.persisted[ids] = False
        self.actor_context[ids] = 0
        self.packet = self._packet()
        # A selected training row already exposed frame-zero strike conditioning
        # to its first actor observation. Commit only after fresh reset geometry
        # exists, before integrating that action; strict rows earn no readiness.
        self.training_initializer.initialize(ids, r, self.events, self.packet, state, self.actor_context)
        self.pending_reset[ids] = False

    def begin_window(self):
        self.events.begin_window()

    def after_substep(self, tick, state):
        state = self._state(state)
        r, e = self.readiness, self.events
        if self.packet is None:
            raise RuntimeError("A physical endpoint preceded readiness initialization")
        cycle_tick = torch.where(r.launched, tick-r.launch_tick, 0)
        if self.training_initializer.config.enabled:
            grace = self.training_initializer.support_grace(cycle_tick, state, e.active)
            e.step(cycle_tick, state, initial_support_grace=grace)
        else:
            e.step(cycle_tick, state)
        r.after_substep(tick, state, prospective_target_w=self.packet["target_w"],
                        cycle_failed=e.first_failure_tick >= 0,
                        recovery_confirmed=e.recovery_tick >= 0)
        # Stop event accumulation immediately after a terminal endpoint, even
        # when three more native intervals remain in this policy window.
        e.active &= r.active_cycle
        self.reference_changed()

    def _persist_terminal(self, ids):
        r, e = self.readiness, self.events
        selected = ids[((r.phase[ids] == Phase.COMPLETE) | (r.phase[ids] == Phase.FAILED)) & ~self.persisted[ids]]
        if not selected.numel():
            return
        # terminal_receipt marks logging only after making tensor copies. A
        # failed sink leaves persisted=false and aborts the caller's reset path.
        readiness = r.terminal_receipt(selected)
        events = e.terminal_receipt(selected)
        self.receipt_sink({"schema": "human_wbc_cycle_attempt_v1", "sequence": self._receipt_number,
            "acceptance_eligible": self.acceptance_eligible,
            "initialization": self.training_initializer.receipt(selected, r),
            "requested_family": self.packet["family_id"][selected].clone(),
            "environment_ids": selected.clone(), "readiness": readiness, "events": events})
        self.persisted[selected] = True
        self._receipt_number += 1

    def command_boundary(self):
        """Called by MotionCommand._update_command before actor observations."""
        r, e = self.readiness, self.events
        if r.last_state is None:
            return  # Initial observation; there has been no settling interval.
        if r._steps_seen % 4:
            raise RuntimeError("Command boundary cannot precede all four native endpoints")
        # A manual/reset-only compute can share the same global boundary while
        # selecting a new clip. Refresh pending observations even when no second
        # launch decision is allowed at this boundary.
        self.packet = self._packet()
        if self._last_command_boundary == r._steps_seen:
            return  # CommandManager may update disjoint reset/nonreset row sets.
        self._last_command_boundary = r._steps_seen
        # Pending physical reset rows keep their prior terminal receipt until
        # the action-side observer obtains their new initialized geometry.
        launch = r.boundary(self.packet) & ~self.pending_reset
        ids = torch.nonzero(launch, as_tuple=False).flatten()
        if ids.numel():
            hand = ((r.committed["family_id"] == 1) | (r.committed["family_id"] == 3)).long()
            e.launch(ids, r.last_state, target_w=r.committed["target_w"], active_hand=hand,
                     ttc_s=r.committed["requested_ttc_s"], outward_end_s=r.committed["outward_end_s"],
                     nominal_guard=r.nominal)
            self.actor_context[ids] = self.packet["actor_g1_context"][ids]
        e.finish_cycles((r.phase == Phase.COMPLETE) | (r.phase == Phase.FAILED))
        self._persist_terminal(torch.arange(r.n, device=r.device))
        self.reference_changed()

    def record_reset(self, ids, terminated, timed_out):
        """Retained pre-reset telemetry; physical reset itself belongs to mjlab."""
        r = self.readiness
        initialized = ids[r.phase[ids] != Phase.UNINITIALIZED]
        if initialized.numel():
            selected = torch.zeros_like(terminated)
            selected[initialized] = True
            if not ((terminated | timed_out)[initialized]).all():
                raise RuntimeError("An initialized attempt needs an explicit terminal reset reason")
            r.record_environment_end(terminated & selected, timed_out & selected)
            self._persist_terminal(initialized)
        self.pending_reset[ids] = True

    def reward_context(self):
        r, e = self.readiness, self.events
        if self.packet is None or e.window_count != 4:
            raise RuntimeError("Cycle rewards require one complete physical policy window")
        launched = r.launched & ~self.pending_reset
        command = {name: torch.where(launched.reshape(r.n, *([1]*(value.ndim-1))),
                    value, self.packet[name]) for name, value in r.committed.items()}
        goal = {name: value.clone() for name, value in r.nominal.items()}
        goal["support_anchor_position_w"] = torch.where(launched[:, None, None],
            r.launch_guard["support_anchor_position_w"], goal["support_anchor_position_w"])
        valid_events = e.window_valid & ~self.pending_reset
        hand = ((command["family_id"] == 1) | (command["family_id"] == 3)).long()
        return {"guard_mask": ((r.phase == Phase.SETTLING) | (r.phase == Phase.RECOVERY)) & ~self.pending_reset,
            "strike_mask": (r.phase == Phase.STRIKE) & ~self.pending_reset,
            "active_hand": torch.where(launched, hand, -1), "target_w": command["target_w"],
            "ttc_s": command["requested_ttc_s"], "elapsed_s": r.reference_tick.to(r.dtype)*.005,
            "reference_glove_w": self.reference_gloves(), "goal": goal,
            "new_contact": e.window_new_contact & valid_events,
            "new_recovery": e.window_new_recovery & valid_events,
            "contact_error_s": torch.where(e.first_contact >= 0, e.first_contact.to(r.dtype)*.005-e.ttc, 0.)}


class JsonReceiptSink:
    """Atomic project-local per-boundary receipts; no pickle or NaN timestamps."""
    def __init__(self, output):
        from human_punching.wbc.provenance import writable_path
        self.output = writable_path(output)
        self.output.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _plain(value):
        if isinstance(value, torch.Tensor):
            return JsonReceiptSink._plain(value.detach().cpu().tolist())
        if isinstance(value, dict):
            return {key: JsonReceiptSink._plain(item) for key, item in value.items()}
        if isinstance(value, list):
            return [JsonReceiptSink._plain(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def __call__(self, receipt):
        path = self.output/f"attempt_{receipt['sequence']:08d}.json"
        if path.exists():
            raise FileExistsError("Never overwrite a previous cycle-attempt receipt")
        payload = json.dumps(self._plain(receipt), sort_keys=True, allow_nan=False).encode()
        temp = path.with_suffix(".tmp")
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        temp.replace(path)


def install_callbacks(env, bridge):
    """Explicitly compose existing public observer hooks; never patch upstream."""
    from human_punching.wbc.telemetry.runtime_telemetry import get_observer
    observer = get_observer(env)
    callbacks = ("after_pending_reset", "begin_sink", "endpoint_sink", "reset_sink")
    if any(getattr(observer, name) is not None for name in callbacks) or hasattr(env, ATTRIBUTE):
        raise RuntimeError("Cycle integration requires unoccupied explicit observer callbacks")
    observer.after_pending_reset = bridge.after_pending_reset
    observer.begin_sink = bridge.begin_window
    def endpoint(tick, state):
        state = dict(state)
        state["clock_failure"] = observer.window.values["clock_failure"]
        bridge.after_substep(tick, state)
    observer.endpoint_sink = endpoint
    observer.reset_sink = bridge.record_reset
    setattr(env, ATTRIBUTE, bridge)


def cycle_condition(env):
    """New schema: prior command features + explicit four-way phase (91 dims).

    This changes the actor input schema and requires scratch training/export
    metadata. It must never be substituted under an existing 394-input actor.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse
    cmd = env.command_manager.get_term("motion")
    bridge = getattr(env, ATTRIBUTE, None)
    if bridge is None:
        # Pinned mjlab resolves observation widths before constructing its
        # metrics manager. Construction has no native step and no ready launch.
        # Subsequent observations must use an installed bridge.
        if hasattr(env, "observation_manager"):
            raise RuntimeError("Install the cycle bridge before reset/observations")
        clip = cmd.trajectory_ids
        family = torch.nn.functional.one_hot(cmd.family_ids[clip], 4).float()
        target = quat_apply_inverse(cmd.robot_anchor_quat_w, cmd.target_w-cmd.robot_anchor_pos_w)
        ttc = cmd.ttcs[clip]
        phase = torch.zeros((len(clip), 4), device=ttc.device, dtype=ttc.dtype)
        phase[:, 0] = 1.
        return torch.cat((family, target, torch.stack((ttc, torch.zeros_like(ttc), ttc), -1),
                          cmd.g1_context[clip], phase), -1)
    r = bridge.readiness
    prospective = bridge.packet if bridge.packet is not None else bridge.packet_provider()
    launched = r.launched & ~bridge.pending_reset
    def frozen(name):
        value = r.committed[name]
        return torch.where(launched.reshape(r.n, *([1]*(value.ndim-1))), value, prospective[name])
    family = torch.nn.functional.one_hot(frozen("family_id"), 4).to(r.dtype)
    target = quat_apply_inverse(cmd.robot_anchor_quat_w, frozen("target_w")-cmd.robot_anchor_pos_w)
    elapsed = torch.where(launched, r.reference_tick.to(r.dtype)*.005, 0.)
    ttc = frozen("requested_ttc_s")
    context = torch.where(launched[:, None], bridge.actor_context, prospective["actor_g1_context"])
    phase_id = torch.where(r.phase <= Phase.RECOVERY, r.phase, 3)
    phase_id = torch.where(bridge.pending_reset, 0, phase_id)
    phase_id = torch.where(bridge.training_initializer.pending_start(bridge.pending_reset), int(Phase.STRIKE), phase_id)
    phase = torch.nn.functional.one_hot(phase_id, 4).to(r.dtype)
    return torch.cat((family, target, torch.stack((ttc, elapsed, ttc-elapsed), -1), context, phase), -1)


def cycle_terminal(env):
    phase = get_cycle_runtime(env).readiness.phase
    return (phase == Phase.COMPLETE) | (phase == Phase.FAILED)


from human_punching.wbc.runtime.command import PunchMotionCommand, PunchMotionCommandCfg
from human_punching.wbc.runtime.bundle import load_runtime_bundle, sha256
from human_punching.wbc.telemetry.runtime_telemetry import ObserverAction, ObserverActionCfg


class CycleObserverAction(ObserverAction):
    def process_actions(self, actions):
        get_cycle_runtime(self._env)  # fail before integration if attachment was omitted
        super().process_actions(actions)


@dataclass(kw_only=True)
class CycleObserverActionCfg(ObserverActionCfg):
    def build(self, env):
        return CycleObserverAction(self, env)

class CycleMotionCommand(PunchMotionCommand):
    def __init__(self, cfg, env):
        if (not isinstance(cfg.training_initialization, TrainingInitializationCfg)
                or (cfg.require_physical_qualification and cfg.training_initialization.enabled)):
            raise ValueError("Training reference starts require an explicit unqualified training configuration")
        super().__init__(cfg, env)
        bundle = load_runtime_bundle(cfg.bundle_path)
        if not math.isfinite(cfg.maximum_ttc_s) or (self.ttcs > cfg.maximum_ttc_s).any():
            raise ValueError("Episode deadline must cover every reference TTC")
        if cfg.require_physical_qualification:
            bundle.require_training_qualification()
        # Conversion generates glove_pos_w from named center sites, in
        # left/right order. Bind both the model and each hashed master.
        if any(name not in bundle.body["site_map"] for name in ("LeftHand_center", "RightHand_center")):
            raise ValueError("Named physical glove center sites are required")
        parts = []
        for clip in self.clips:
            path = bundle.path(clip["master_path"])
            if sha256(path) != clip["master_sha256"]:
                raise ValueError("Glove reference master hash mismatch")
            with np.load(path, allow_pickle=False) as data:
                value = data["glove_pos_w"]
                if value.shape != (len(data["time_s"]), 2, 3) or not np.isfinite(value).all():
                    raise ValueError("Finite named left/right glove-center reference arrays required")
                parts.append(value.copy())
        self.glove_master = torch.as_tensor(np.concatenate(parts), dtype=torch.float32, device=self.device)
        self.glove_reference_identity = {"model_sha256": sha256(bundle.model_path),
            "sites": ["LeftHand_center", "RightHand_center"],
            "master_sha256": [clip["master_sha256"] for clip in self.clips]}

    def _cycle_clock(self):
        bridge = getattr(self._env, ATTRIBUTE, None)
        if bridge is None or getattr(self, "_reset_in_progress", False):
            return torch.zeros_like(self.physics_steps), torch.zeros_like(self.physics_steps, dtype=torch.bool)
        pending_start = bridge.training_initializer.pending_start(bridge.pending_reset)
        launched = (bridge.readiness.launched & ~bridge.pending_reset) | pending_start
        # A new strict guard episode can follow a launched training episode.
        # Its first observation must not expose the preceding episode's tick,
        # even while the reset's readiness history awaits its first action.
        return torch.where(bridge.pending_reset, 0, bridge.readiness.reference_tick), launched

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        bridge = getattr(self._env, ATTRIBUTE, None)
        if bridge is not None:
            bridge.prepare_training_reset(env_ids)

    def sample_master(self, name, offset_steps=0):
        tick, launched = self._cycle_clock()
        index, stationary = reference_indices(tick, launched, self.trajectory_ids,
            self.master_starts, self.master_lengths, offset_steps)
        value = self.master[name][index]
        if name in DERIVATIVES:
            value = torch.where(stationary.reshape(-1, *([1]*(value.ndim-1))), 0., value)
        if name == "joint_feedforward":
            # Static joint feedforward is an explicit controller component;
            # no free-base wrench or dynamic frame-zero derivative.
            ff = getattr(self, "settling_feedforward", None)
            if ff is None:
                ff = torch.zeros_like(value)
            value = torch.where(launched[:, None], value, ff)
        return value

    @property
    def elapsed(self):
        tick, launched = self._cycle_clock()
        return torch.where(launched, tick.to(torch.float32)*.005, 0.)

    @property
    def target_w(self):
        bridge = getattr(self._env, ATTRIBUTE, None)
        if bridge is None:
            return super().target_w
        launched = bridge.readiness.launched & ~bridge.pending_reset
        return torch.where(launched[:, None], bridge.readiness.committed["target_w"], super().target_w)

    def reference_gloves(self):
        tick, launched = self._cycle_clock()
        index, _ = reference_indices(tick, launched, self.trajectory_ids, self.master_starts, self.master_lengths)
        return self.glove_master[index] + self._env.scene.env_origins[:, None]

    def _update_command(self, env_ids):
        bridge = getattr(self._env, ATTRIBUTE, None)
        if bridge is not None and not getattr(self, "_reset_in_progress", False):
            bridge.command_boundary()
        tick, _ = self._cycle_clock()
        ids = slice(None) if env_ids is None else env_ids
        clip = self.trajectory_ids[ids]
        frame = torch.minimum(tick[ids]//4, self.motion.segment_length[clip]-1)
        self.time_steps[ids] = self.motion.segment_start_idx[clip]+frame
        self.update_relative_body_poses()

@dataclass(kw_only=True)
class CycleMotionCommandCfg(PunchMotionCommandCfg):
    require_physical_qualification: bool = True
    maximum_ttc_s: float = 1.35
    training_initialization: TrainingInitializationCfg = TrainingInitializationCfg()

    def build(self, env):
        return CycleMotionCommand(self, env)


def configure_cycle_command(cfg, *, maximum_ttc_s, readiness=ReadinessConfig(),
                            require_physical_qualification=True, task_objectives=False,
                            training_initialization=TrainingInitializationCfg()):
    """Opt-in local config replacement; caller separately enables telemetry.

    The default combined actor gains four phase features (394 -> 398). Narrower
    conditioning ablations need explicit corresponding phase-aware functions;
    this baseline refuses to silently overwrite an ablation configuration.
    """
    from human_punching.wbc.runtime import mdp
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    if not math.isfinite(maximum_ttc_s) or maximum_ttc_s <= 0:
        raise ValueError("Maximum qualified TTC must be finite and positive")
    if not isinstance(training_initialization, TrainingInitializationCfg):
        raise TypeError("Explicit training initialization configuration required")
    if require_physical_qualification and training_initialization.enabled:
        raise ValueError("Physically qualified strict tasks cannot enable training initialization")
    if type(cfg.commands.get("motion")) is not PunchMotionCommandCfg:
        raise ValueError("Expected the unmodified human punch command configuration")
    if type(cfg.actions.get("joint_pos")) is not ObserverActionCfg:
        raise ValueError("Enable the native observer before configuring the cycle command")
    for group in ("actor", "critic"):
        term = cfg.observations[group].terms.get("punch_condition")
        if term is None or term.func is not mdp.punch_condition:
            raise ValueError("Explicit combined conditioning is required for this cycle baseline")
    current = cfg.commands["motion"]
    cfg.commands["motion"] = CycleMotionCommandCfg(
        **{field.name: getattr(current, field.name) for field in fields(current)},
        require_physical_qualification=require_physical_qualification,
        maximum_ttc_s=maximum_ttc_s, training_initialization=training_initialization)
    action = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = CycleObserverActionCfg(
        **{field.name: getattr(action, field.name) for field in fields(action) if field.init})
    for group in ("actor", "critic"):
        cfg.observations[group].terms["punch_condition"] = ObservationTermCfg(func=cycle_condition)
    cfg.terminations["cycle_complete"] = TerminationTermCfg(func=cycle_terminal)
    cfg.rewards.pop("timed_target", None)  # endpoint-lagged predecessor; root adds fresh objective separately
    cfg.episode_length_s = max(cfg.episode_length_s,
        math.ceil((readiness.maximum_settling_ticks*.005+maximum_ttc_s+4.)/.020)*.020+.020)
    if task_objectives:
        from human_punching.wbc.runtime.task_reward_adapter import configure_task_rewards
        configure_task_rewards(cfg)
    return {"schema": "human_wbc_cycle_command_v1", "policy_command_dimension": 91,
        "phase_order": ["settling", "strike", "recovery", "terminal"],
        "maximum_ttc_s": maximum_ttc_s, "episode_length_s": cfg.episode_length_s,
        "reference_hold_before_launch": True, "strict_readiness": not training_initialization.enabled,
        "training_initialization": asdict(training_initialization),
        "native_adapter_qualified": False}


def bind_packet_to_command(command, provider):
    """Reject a planner packet that names a different immutable clip or context.

    This adapter currently permits only precomputed snapshot plans. The isolated
    snapshot remains stationary while the human settles. For a future moving G1,
    capture its CURRENT state when ready and produce a new matching frozen plan
    without advancing physics. A reset-time snapshot cannot stand in for that
    later launch state; this provider must be replaced and requalified.
    """
    n, device = command.num_envs, command.device
    digest = torch.tensor([list(bytes.fromhex(c["master_sha256"])) for c in command.clips],
                          dtype=torch.uint8, device=device)
    outward = torch.tensor([c["outward_end_s"] for c in command.clips], dtype=torch.float32, device=device)
    full_context = []
    for c in command.clips:
        snapshot = c["launch_snapshot"]
        full_context.append(sum((list(snapshot[key]) for key in (
            "g1_qpos", "g1_qvel", "head_position_w", "head_velocity_w", "human_origin_w", "human_quat_wxyz")), []))
    context = torch.tensor(full_context, dtype=torch.float32, device=device)
    if context.shape[1] != 84:
        raise ValueError("Expected complete G1 qpos36/qvel35/head/placement launch payload")
    # Context layout is qpos36,qvel35,head_position3,head_velocity3,
    # human_origin3,human_quaternion4. World points translate with scene origins.
    def expected():
        clip, origin = command.trajectory_ids, command._env.scene.env_origins
        snapshot = context[clip].clone()
        for start in (0, 71, 77):
            snapshot[:, start:start+3] += origin
        values = {"target_w": command.targets[clip]+origin, "g1_launch_state": snapshot,
            "requested_ttc_s": command.ttcs[clip], "outward_end_s": outward[clip],
            "family_id": command.family_ids[clip], "plan_handle": clip,
            "plan_digest": digest[clip], "actor_g1_context": command.g1_context[clip]}
        # Plan handles must never expose the command's mutable trajectory_ids
        # storage to a caller preparing or updating prospective packets.
        return {name: value.clone() for name, value in values.items()}
    def checked():
        packet, bound = provider(), expected()
        runtime = getattr(command._env, ATTRIBUTE, None)
        rows = (torch.ones(n, dtype=torch.bool, device=device) if runtime is None else
                (~runtime.readiness.launched | runtime.pending_reset))
        for key, value in bound.items():
            supplied = packet.get(key)
            if (not isinstance(supplied, torch.Tensor) or supplied.shape != value.shape
                    or supplied.dtype != value.dtype or supplied.device != value.device):
                raise ValueError(f"Malformed bound planner field {key}")
            if not torch.equal(supplied[rows], value[rows]):
                raise ValueError(f"Planner field {key} differs from the immutable reference command")
        return packet
    checked.expected = expected
    checked.context_width = 84
    return checked


def attach_cycle_runtime(env, *, nominal_provider, packet_provider, request_ids,
                         receipt_sink, feedforward_artifact=None, readiness=ReadinessConfig()):
    """Attach after environment construction and before its first reset/action.

    Providers are explicit experiment inputs, not generic defaults. In
    particular, neither nominal posture nor admission is inferred from a noisy
    initialized physical state. All reference commands remain precomputed and
    hash-bound in this first implementation.
    """
    from human_punching.wbc.telemetry.runtime_telemetry import get_observer
    observer = get_observer(env)
    command = env.command_manager.get_term("motion")
    if not isinstance(command, CycleMotionCommand) or observer.actual_samples:
        raise ValueError("A fresh cycle-command environment with native telemetry is required")
    bundle = load_runtime_bundle(command.cfg.bundle_path)
    # The qualification flag is deliberately inherited from the constructed
    # command, not a later caller assertion. A diagnostic can never mint a
    # physically qualified panel by passing acceptance_eligible=True here.
    qualified = command.cfg.require_physical_qualification
    if qualified:
        bundle.require_training_qualification()
    limits = torch.tensor(bundle.body["joint_limits_rad"], dtype=torch.float32, device=env.device)
    caps = torch.tensor(bundle.body["effort"], dtype=torch.float32, device=env.device)
    ff = torch.zeros(48, dtype=torch.float32, device=env.device)
    ff_identity = {"mode": "zero_joint_feedforward"}
    if feedforward_artifact is not None:
        from human_punching.wbc.guard_feedforward import load_guard_feedforward
        actual_solver = {"mujoco": asdict(env.cfg.sim.mujoco), "decimation": env.cfg.decimation,
                         "nconmax": env.cfg.sim.nconmax, "njmax": env.cfg.sim.njmax}
        artifact = load_guard_feedforward(feedforward_artifact, bundle, actual_solver_contract=actual_solver)
        ff = torch.tensor(artifact["joint_feedforward_nm"], dtype=torch.float32, device=env.device)
        ff_identity = {"mode": "static_support_lp_joint_feedforward", "artifact_sha256": sha256(Path(feedforward_artifact)),
                       "payload_sha256": artifact["payload_sha256"]}
    checked_provider = bind_packet_to_command(command, packet_provider)
    bridge = CycleRuntime(env.num_envs, limits, caps, context_width=checked_provider.context_width,
        packet_provider=checked_provider, nominal_provider=nominal_provider, request_ids=request_ids,
        reference_gloves=command.reference_gloves, receipt_sink=receipt_sink,
        reference_changed=command.update_relative_body_poses,
        config=readiness, acceptance_eligible=False,
        training_initialization=command.cfg.training_initialization)
    # Even screened references do not qualify this new native launch adapter.
    # Keep acceptance false until its own independent native proof is recorded.
    bridge.reference_qualified = qualified
    bridge.executed_controller_identity = {"body_sha256": sha256(bundle.model_path),
        "base_controller_sha256": bundle.controller_sha256(), "settling_feedforward": ff_identity,
        "strike_feedforward": "immutable_master", "physics_dt_s": .005, "policy_dt_s": .020}
    command.settling_feedforward = ff
    install_callbacks(env, bridge)
    return bridge

