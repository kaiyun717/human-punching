"""DRAFT observer-only native 200 Hz telemetry, disabled unless explicitly added.

Fresh endpoint FK is private. Contacts and effort describe the preceding native
integration interval and are paired with its START material-foot anchors. This
module never invokes forward, collision, integration, reset or physical writes.
It does not declare launch/readiness, redirect references, or alter rewards.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable
import torch

DT = .005
DECIMATION = 4
ATTRIBUTE = "_human_substep_observer"
FLAGS = ("external_support", "root_actuation", "forbidden_collision",
         "human_g1_collision", "numerical_failure", "contact_overflow", "clock_failure")


@dataclass(frozen=True)
class ObserverOptions:
    enabled: bool = False
    prefix: str = "robot/"
    # Analytic isolated-human panel by default. A combined scene must name G1.
    g1_geom_names: tuple[str, ...] = ()
    allowed_self_pairs: tuple[tuple[str, str], ...] = ()


class FourTickWindow:
    """Exact four-substep extrema; stock per_substep metrics average first."""
    def __init__(self, n, limits, caps):
        if (type(n) is not int or n < 1 or limits.shape != (caps.numel(), 2)
                or caps.ndim != 1 or limits.device != caps.device
                or limits.dtype != torch.float32 or caps.dtype != torch.float32
                or not torch.isfinite(limits).all() or not torch.isfinite(caps).all()
                or not (limits[:, 0] < limits[:, 1]).all() or not (caps > 0).all()):
            raise ValueError("Float32 finite ordered joint limits and positive caps required")
        self.n, self.limits, self.caps = n, limits.clone(), caps.clone()
        self.device = limits.device
        self.count = 0
        self.open = False
        self.values = {key: torch.zeros(n, device=self.device) for key in
                       ("joint_limit_excess_rad", "effort_excess_nm", "effort_peak_nm",
                        "clock_error_s", "double_support_fraction", "root_linear_speed_peak_m_s",
                        "root_angular_speed_peak_rad_s", "joint_speed_peak_rad_s",
                        "glove_guard_speed_peak_m_s")}
        self.values.update({key: torch.zeros(n, dtype=torch.bool, device=self.device) for key in FLAGS})
        self.values["loaded_patch_slip_m"] = torch.zeros(n, 2, device=self.device)
        self.values["foot_normal_force_peak_n"] = torch.zeros(n, 2, device=self.device)
        self.previous_anchor = torch.zeros(n, 2, 3, device=self.device)
        self.previous_velocity = torch.zeros_like(self.previous_anchor)
        self.previous_time = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.previous_tick = torch.zeros(n, dtype=torch.int64, device=self.device)

    def _check(self, state, tick, time):
        shape = {"support_anchor_position_w": (self.n, 2, 3),
                 "support_anchor_velocity_w": (self.n, 2, 3)}
        for name, expected in shape.items():
            value = state[name]
            if value.shape != expected or value.device != self.device or value.dtype != torch.float32:
                raise ValueError(f"Malformed {name}")
        if (tick.shape != (self.n,) or tick.dtype != torch.int64 or tick.device != self.device
                or time.shape != (self.n,) or time.dtype not in (torch.float32, torch.float64)
                or time.device != self.device):
            raise ValueError("Exact int64 ticks and floating native time per environment required")

    def begin(self, state, tick, time):
        self._check(state, tick, time)
        if self.open and self.count != DECIMATION:
            raise RuntimeError("Previous policy window lacks four real native samples")
        self.count = 0
        self.open = True
        for value in self.values.values():
            value.zero_()
        self._remember(state, tick, time)

    def _remember(self, state, tick, time):
        # Never retain views into ScratchKinematics' reused endpoint workspace.
        self.previous_anchor.copy_(state["support_anchor_position_w"])
        self.previous_velocity.copy_(state["support_anchor_velocity_w"])
        self.previous_time.copy_(time)
        self.previous_tick.copy_(tick)

    def consume(self, state, tick, time):
        self._check(state, tick, time)
        if not self.open or self.count >= DECIMATION:
            raise RuntimeError("One sample per actual native substep, after begin, required")
        expected = {"joint_position": (self.n, len(self.caps)),
                    "joint_velocity": (self.n, len(self.caps)),
                    "root_linear_velocity_w": (self.n, 3), "root_angular_velocity_w": (self.n, 3),
                    "glove_velocity_guard_frame": (self.n, 2, 3),
                    "effort": (self.n, len(self.caps)), "foot_normal_force_n": (self.n, 2),
                    "support_contact": (self.n, 2), "support_patch_contact": (self.n, 2),
                    **{name: (self.n,) for name in FLAGS if name != "clock_failure"}}
        for name, shape in expected.items():
            boolean = name in FLAGS or name in ("support_contact", "support_patch_contact")
            value = state[name]
            if (value.shape != shape or value.device != self.device
                    or value.dtype != (torch.bool if boolean else torch.float32)):
                raise ValueError(f"Malformed interval telemetry: {name}")
        finite = torch.isfinite(time) & torch.isfinite(self.previous_time)
        for name in ("joint_position", "joint_velocity", "root_linear_velocity_w", "root_angular_velocity_w",
                     "glove_velocity_guard_frame", "effort", "foot_normal_force_n",
                     "support_anchor_position_w", "support_anchor_velocity_w"):
            finite &= torch.isfinite(state[name]).flatten(1).all(-1)
        error = (time-self.previous_time-DT).abs().to(torch.float32)
        clock = (tick != self.previous_tick+1) | ~torch.isfinite(error) | (error > 2.e-6)
        q = state["joint_position"]
        joint = torch.maximum(self.limits[:, 0]-q, q-self.limits[:, 1]).clamp_min(0).amax(-1)
        effort = state["effort"].abs()
        excess = (effort-self.caps).clamp_min(0).amax(-1)
        for name, value in (("joint_limit_excess_rad", joint), ("effort_excess_nm", excess),
                            ("effort_peak_nm", effort.amax(-1)), ("clock_error_s", error),
                            ("root_linear_speed_peak_m_s", torch.linalg.vector_norm(state["root_linear_velocity_w"],dim=-1)),
                            ("root_angular_speed_peak_rad_s", torch.linalg.vector_norm(state["root_angular_velocity_w"],dim=-1)),
                            ("joint_speed_peak_rad_s", state["joint_velocity"].abs().amax(-1)),
                            ("glove_guard_speed_peak_m_s", torch.linalg.vector_norm(state["glove_velocity_guard_frame"],dim=-1).amax(-1))):
            torch.maximum(self.values[name], torch.nan_to_num(value, nan=float("inf"), posinf=float("inf")), out=self.values[name])
        torch.maximum(self.values["foot_normal_force_peak_n"], torch.nan_to_num(state["foot_normal_force_n"], nan=float("inf")), out=self.values["foot_normal_force_peak_n"])
        # Native contact load is from the interval start. Use the same start's
        # velocity, not the new endpoint velocity, for its left-Riemann slip.
        slip = torch.linalg.vector_norm(self.previous_velocity[..., :2], dim=-1)*DT
        self.values["loaded_patch_slip_m"].add_(torch.where(state["support_patch_contact"], slip, 0.))
        self.values["double_support_fraction"].add_(state["support_contact"].all(-1).float()/DECIMATION)
        for flag in FLAGS:
            if flag != "clock_failure":
                self.values[flag] |= state[flag]
        self.values["numerical_failure"] |= ~finite
        self.values["clock_failure"] |= clock
        self._remember(state, tick, time)
        self.count += 1

    def snapshot(self):
        if not self.open or self.count != DECIMATION:
            raise RuntimeError("A reported policy window needs exactly four native intervals")
        return {key: value.clone() for key, value in self.values.items()}


def applied_force_flags(qfrc_applied, xfrc_applied, root_actuator_force):
    """No cancellation between bodies, axes, signs, or generalized forces."""
    n = qfrc_applied.shape[0]
    values = (qfrc_applied, xfrc_applied, root_actuator_force)
    if any(v.shape[0] != n or not v.is_floating_point() for v in values):
        raise ValueError("Batched floating physical force arrays required")
    finite = torch.stack([torch.isfinite(v).flatten(1).all(-1) for v in values]).all(0)
    external = ((qfrc_applied.abs().flatten(1).amax(-1) > 1e-9)
                | (xfrc_applied.abs().flatten(1).amax(-1) > 1e-9))
    root = root_actuator_force.abs().flatten(1).amax(-1) > 1e-9
    return external, root, ~finite


class ObserverCore:
    """Backend-injected sequencing, also testable without any GPU integration."""
    def __init__(self, window, *, fresh_kinematics: Callable, interval_contacts: Callable,
                 force_state: Callable, clock_state: Callable, endpoint_sink=None):
        self.window = window
        self.fresh_kinematics = fresh_kinematics
        self.interval_contacts = interval_contacts
        self.force_state = force_state
        self.clock_state = clock_state
        self.endpoint_sink = endpoint_sink
        self.after_pending_reset = None
        self.begin_sink = None
        self.reset_sink = None
        self.actual_samples = 0
        self.last_state = None
        self.nominal_geometry = None
        self.pending_initialization = torch.ones(window.n, dtype=torch.bool, device=window.device)
        self.reset_count = torch.zeros(window.n, dtype=torch.int64, device=window.device)
        self.reset_terminated = torch.zeros(window.n, dtype=torch.bool, device=window.device)
        self.reset_timed_out = torch.zeros_like(self.reset_terminated)

    def begin_policy_window(self):
        state = self.fresh_kinematics()
        if self.nominal_geometry is None:
            self.nominal_geometry = {name: value.clone() for name, value in state.items()}
        else:
            for name, value in state.items():
                self.nominal_geometry[name][self.pending_initialization] = value[self.pending_initialization]
        tick, time = self.clock_state()
        if self.after_pending_reset is not None:
            ids = torch.nonzero(self.pending_initialization, as_tuple=False).flatten()
            if ids.numel():
                initial = dict(state)
                initial.update(self.interval_contacts(state["support_anchor_position_w"]))
                initial.update(self.force_state())
                initial["numerical_failure"] = (initial["numerical_failure"]
                    | ~initial["finite_kinematics"] | initial.pop("nonfinite_force"))
                self.after_pending_reset(ids, initial, tick, time)
        self.pending_initialization.zero_()
        self.window.begin(state, tick, time)
        if self.begin_sink is not None:
            self.begin_sink()

    def after_native_substep(self):
        # Contacts first: the next FK update may overwrite reused private state.
        contact = self.interval_contacts(self.window.previous_anchor)
        state = dict(self.fresh_kinematics())
        state.update(contact)
        state.update(self.force_state())
        state["numerical_failure"] = (state["numerical_failure"]
            | ~state["finite_kinematics"] | state.pop("nonfinite_force"))
        tick, time = self.clock_state()
        self.window.consume(state, tick, time)
        self.actual_samples += 1
        self.last_state = state
        if self.endpoint_sink is not None:
            # Sink must clone retained values. Observer owns no reference command,
            # admission, launch clock or cycle success state in this first draft.
            self.endpoint_sink(tick, state)
        return ~(state["numerical_failure"] | self.window.values["clock_failure"])

    def record_reset(self, ids, *, terminated, timed_out):
        # Called by public metric reset after native reset. No FK here: prior
        # endpoint data are retained, and next process_actions samples new state.
        ids = torch.arange(self.window.n, device=self.window.device)[ids]
        if self.window.open and self.window.count != DECIMATION:
            raise RuntimeError("Physical reset arrived inside a four-tick window")
        if self.reset_sink is not None:
            self.reset_sink(ids, terminated, timed_out)
        self.pending_initialization[ids] = True
        self.reset_count[ids] += 1
        self.reset_terminated[ids] = terminated[ids]
        self.reset_timed_out[ids] = timed_out[ids]


    def endpoint_state(self):
        """Read-only views of the most recent fresh endpoint; no implicit FK."""
        if self.last_state is None:
            raise RuntimeError("No physical endpoint has been observed yet")
        return self.last_state

    def nominal_guard_state(self):
        """Read-only initialization geometry, not an admitted/ready guard claim."""
        if self.nominal_geometry is None:
            raise RuntimeError("No completed initialization has been sampled yet")
        return self.nominal_geometry

    def window_summary(self):
        if not self.window.open or self.window.count != DECIMATION:
            raise RuntimeError("Window requires four actual physics intervals")
        return self.window.values


def get_observer(env):
    observer = getattr(env, ATTRIBUTE, None)
    if observer is None:
        raise RuntimeError("Native telemetry was not explicitly enabled")
    return observer


def native_tensor(value):
    """Read-only unwrapping of pinned mjlab TorchArray; no copy or cast."""
    tensor = getattr(value, "_tensor", value)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Native telemetry requires tensor-backed state arrays")
    return tensor


class NativeObserver(ObserverCore):
    def __init__(self, env, options):
        from .kinematics import EntityBindings, ScratchKinematics
        from .contacts import NativeContactTelemetry
        from human_punching.wbc.cycle_trace import TraceBindings
        from human_punching.wbc.runtime.bundle import load_runtime_bundle
        self.env = env
        model = env.sim.mj_model
        if abs(float(model.opt.timestep)-DT) > 1e-12 or env.cfg.decimation != DECIMATION:
            raise ValueError("Native observer requires actual 5 ms integration and decimation four")
        cmd = env.command_manager.get_term("motion")
        bundle = load_runtime_bundle(cmd.cfg.bundle_path)
        device = torch.device(env.device)
        self.fk = ScratchKinematics(model, env.sim.wp_model,
            EntityBindings(options.prefix, tuple(bundle.joint_names)), env.num_envs,
            device, live_data=env.sim.wp_data)
        contact_bindings = TraceBindings(prefix=options.prefix,
            human_geom_prefix=options.prefix+"physical/", g1_geom_names=options.g1_geom_names,
            allowed_self_pairs=options.allowed_self_pairs)
        self.contact = NativeContactTelemetry(model, env.sim.wp_model, env.sim.wp_data,
                                              contact_bindings, torch_device=device)
        actuator_ids = tuple(int(model.actuator(options.prefix+n+"_native_pd").id) for n in bundle.joint_names)
        root_body = int(model.body(options.prefix+"Hips").id)
        root_dof = int(model.jnt_dofadr[int(model.body_jntadr[root_body])])
        self.actuator_ids = torch.tensor(actuator_ids, dtype=torch.int64, device=device)
        limits = torch.tensor(bundle.body["joint_limits_rad"], dtype=torch.float32, device=device)
        caps = torch.tensor(bundle.body["effort"], dtype=torch.float32, device=device)
        data = env.sim.data
        def fresh():
            kwargs = {name: native_tensor(getattr(data, name)) for name in ("mocap_pos", "mocap_quat")} if model.nmocap else {}
            return self.fk.update(native_tensor(data.qpos), native_tensor(data.qvel), **kwargs)
        def force():
            external, root, nonfinite = applied_force_flags(native_tensor(data.qfrc_applied),
                native_tensor(data.xfrc_applied), native_tensor(data.qfrc_actuator)[:, root_dof:root_dof+6])
            return dict(effort=data.actuator_force.index_select(1, self.actuator_ids),
                        external_support=external, root_actuation=root, nonfinite_force=nonfinite)
        super().__init__(FourTickWindow(env.num_envs, limits, caps), fresh_kinematics=fresh,
                         interval_contacts=self.contact.sample, force_state=force,
                         clock_state=lambda: (cmd.physics_steps, data.time.reshape(env.num_envs)))
        self.metadata = dict(schema="human_wbc_observer_draft_v1", physics_dt_s=DT,
            decimation=DECIMATION, launch_integration_supported=False,
            acceptance_qualified=False, collision_panel="human_g1" if options.g1_geom_names else "analytic_no_g1",
            kinematics="fresh_endpoint_private_fk", contacts="preceding_integration_interval",
            source_of_slip_velocity="preceding_interval_start_material_anchor",
            private_fk_memory=self.fk.memory)


class TelemetryMetric:
    """Published mjlab class-term hook, constructed once per environment."""
    def __init__(self, cfg, env):
        if hasattr(env, ATTRIBUTE):
            raise ValueError("Only one native substep observer may own this environment")
        self.env = env
        self.observer = NativeObserver(env, cfg.params["options"])
        setattr(env, ATTRIBUTE, self.observer)

    def __call__(self, env, options):
        del env, options
        return self.observer.after_native_substep().float()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.observer.record_reset(ids, terminated=self.env.termination_manager.terminated,
                                   timed_out=self.env.termination_manager.time_outs)


def window_metric(env, name):
    window = getattr(env, ATTRIBUTE).window
    if window.count != DECIMATION:
        raise RuntimeError("Window metric called before all four real integration steps")
    value = window.values[name]
    return value.float() if value.ndim == 1 else value.amax(-1)



from human_punching.wbc.runtime.action import PunchJointPositionAction, PunchJointPositionActionCfg


class ObserverAction(PunchJointPositionAction):
    """Same native drive law; only samples the pre-integration anchor state."""
    def process_actions(self, actions):
        get_observer(self._env).begin_policy_window()
        super().process_actions(actions)


@dataclass(kw_only=True)
class ObserverActionCfg(PunchJointPositionActionCfg):
    def build(self, env):
        return ObserverAction(self, env)


def configure_observer(cfg, options=ObserverOptions()):
    """Explicit opt-in using public cfg.build/class-metric extension points."""
    if not options.enabled:
        return cfg
    from human_punching.wbc.runtime.action import PunchJointPositionAction, PunchJointPositionActionCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    old = cfg.actions["joint_pos"]
    if type(old) is not PunchJointPositionActionCfg:
        raise ValueError("Observer draft only wraps the reviewed native human action config")
    if "human_native_substep_valid" in cfg.metrics:
        raise ValueError("Duplicate observer metric registration")
    cfg.actions["joint_pos"] = ObserverActionCfg(**{f.name: getattr(old, f.name) for f in fields(old) if f.init})
    cfg.metrics["human_native_substep_valid"] = MetricsTermCfg(func=TelemetryMetric,
        params={"options": options}, per_substep=True, reduce="mean")
    for name in ("joint_limit_excess_rad", "effort_excess_nm", *FLAGS):
        cfg.metrics["human_native_"+name] = MetricsTermCfg(func=window_metric,
            params={"name": name}, per_substep=False, reduce="max")
    return cfg
