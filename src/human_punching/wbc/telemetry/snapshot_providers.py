"""Explicit immutable-snapshot engineering providers; no physical admission claim."""
from __future__ import annotations

import numpy as np
import torch

from human_punching.wbc.cycle_trace import TraceBindings, derive_kinematics
from human_punching.wbc.runtime.bundle import load_runtime_bundle
from .cycle_runtime import ATTRIBUTE, bind_packet_to_command
from .event_state import FLAGS, KINEMATIC_WIDTHS
from .readiness_clock import PlanStatus
from .runtime_telemetry import get_observer


def declared_guard_table(model, bundle, clips, *, bindings=TraceBindings()):
    """Offline named FK of each declared master-zero guard; never current pose.

    Nominal derivatives/effort and nominal support flags are zero. In particular,
    support is never inferred from sole height; actual readiness needs real
    post-integration forces from the observer.
    """
    if model.nq != 55 or model.nv != 54:
        raise ValueError('This isolated human provider requires named 55/54 state ownership')
    rows = []
    for clip in clips:
        with np.load(bundle.path(clip['master_path']), allow_pickle=False) as master:
            q = master['qpos'][:1].copy()
        result = {name: value[0] for name, value in derive_kinematics(model, q, np.zeros((1, 54)), bindings).items()}
        result.update(joint_position=q[0, 7:].copy(), joint_velocity=np.zeros(48), effort=np.zeros(48),
            support_anchor_velocity_w=np.zeros((2, 3)), support_contact=np.zeros(2, bool),
            support_patch_contact=np.zeros(2, bool), **{name: np.asarray(False) for name in FLAGS})
        if not set(KINEMATIC_WIDTHS) <= result.keys():
            raise ValueError('Declared guard geometry is incomplete')
        rows.append(result)
    return {name: np.stack([row[name] for row in rows]) for name in rows[0]}


def make_snapshot_providers(env, *, engineering_admitted: bool, perturb_after_launch=False):
    """Precomputed isolated snapshot only; admission is an explicit experiment input.

    A caller must separately bind engineering/reference-screening evidence.
    ``engineering_admitted=True`` requests native testing, not production
    acceptance or a feasibility certificate. A future moving G1 must replace
    this provider with an actual ready-time snapshot and matching frozen plan.
    """
    if engineering_admitted is not True:
        raise ValueError('An explicit engineering admission decision is required')
    command = env.command_manager.get_term('motion')
    bundle = load_runtime_bundle(command.cfg.bundle_path)
    tables = declared_guard_table(env.sim.mj_model, bundle, command.clips)
    tensors = {name: torch.tensor(value, device=env.device,
        dtype=torch.bool if value.dtype == bool else torch.float32) for name, value in tables.items()}
    def nominal():
        result = {name: value[command.trajectory_ids].clone() for name, value in tensors.items()}
        for name in ('root_position_w', 'glove_position_w', 'support_anchor_position_w'):
            origin = env.scene.env_origins
            result[name] += origin[:, None] if result[name].ndim == 3 else origin
        return result
    binder = bind_packet_to_command(command, lambda: None)
    def packet():
        result = binder.expected()
        result['status'] = torch.full((env.num_envs,), PlanStatus.ADMITTED, dtype=torch.int64, device=env.device)
        result['plan_matches_guard_context'] = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        # Diagnostic only: prove later invalid live metadata cannot redirect a
        # committed snapshot. Pending reset rows receive the original plan.
        bridge = getattr(env, ATTRIBUTE, None)
        if perturb_after_launch and bridge is not None:
            selected = bridge.readiness.launched & ~bridge.pending_reset
            result['target_w'][selected] = float('nan')
            result['g1_launch_state'][selected] = float('nan')
            result['actor_g1_context'][selected] = float('nan')
        return result
    def identities():
        observer = get_observer(env)
        return observer.reset_count*env.num_envs+torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    return dict(nominal_provider=nominal, packet_provider=packet, request_ids=identities)
