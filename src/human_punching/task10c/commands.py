"""Stored-reference command, adapted from TaskNPoint's motion command structure.

TaskNPoint b8faaf284e0cef8021b94e2219492dbd54efba0c supplies the six-array
motion-loader/command pattern. Task10C adds named maps, fractional sampling,
committed timing, complete recovery and unknown-history snippet initialization.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .events import PunchState, reference_clock, scheduled_normalizers

FAMILY_ACTIVE_HAND = {'jab': 'left', 'cross': 'right', 'left_hook': 'left', 'right_hook': 'right'}


class TorchReference:
    def __init__(self, path, device, joint_names, family=None):
        path = Path(path)
        self.metadata = json.loads(path.with_suffix('.json').read_text())
        if family is not None:
            if family not in FAMILY_ACTIVE_HAND:
                raise ValueError(f'Unknown punch family: {family!r}')
            if self.metadata.get('family') != family:
                raise ValueError(f'Reference family {self.metadata.get("family")!r} does not match selected {family!r}')
            expected_hand = FAMILY_ACTIVE_HAND[family]
            if self.metadata.get('active_hand') != expected_hand:
                raise ValueError(f'Reference active_hand must be {expected_hand!r} for {family!r}')
        with np.load(path, allow_pickle=False) as data:
            self.joint_names = tuple(map(str, data['joint_names']))
            self.body_names = tuple(map(str, data['body_names']))
            if (len(self.joint_names) != 31 or len(joint_names) != 31
                    or len(set(self.joint_names)) != 31 or set(self.joint_names) != set(joint_names)):
                raise ValueError('Reference must match all 31 selected H2 joints by name')
            order = [self.joint_names.index(name) for name in joint_names]
            self.arrays = {key: torch.as_tensor(data[key].copy(), dtype=torch.float32, device=device)
                           for key in data.files if data[key].dtype.kind in 'fiu'}
            self.arrays['joint_pos'] = self.arrays['joint_pos'][:, order]
            self.arrays['joint_vel'] = self.arrays['joint_vel'][:, order]
        self.time = self.arrays['time_s']
        if not bool(torch.all(self.time[1:] > self.time[:-1])):
            raise ValueError('Reference timestamps must strictly increase')
        self.strike_time = float(self.metadata['strike_time_s'])
        self.end_time = float(self.time[-1])
        self.recovery = max(0.6, float(self.metadata.get('recovery_duration_s', self.end_time - self.strike_time)))
        if not 0 < self.strike_time < self.end_time:
            raise ValueError('Strike annotation must lie strictly inside reference')

    def sample(self, source_time, rate):
        hi = torch.searchsorted(self.time, source_time.contiguous(), right=True).clamp(1, len(self.time) - 1)
        lo = hi - 1
        alpha = ((source_time - self.time[lo]) / (self.time[hi] - self.time[lo])).clamp(0.0, 1.0)
        result = {}
        for key, array in self.arrays.items():
            if key in ('time_s', 'source_frames') or array.ndim < 2:
                continue
            a, b = array[lo], array[hi]
            fraction = alpha.reshape((-1,) + (1,) * (array.ndim - 1))
            if 'quat' in key:
                # Short-arc SLERP with a stable normalized-linear small-angle branch.
                dot = (a * b).sum(-1, keepdim=True)
                b = torch.where(dot < 0, -b, b)
                dot = dot.abs().clamp(0, 1)
                angle = torch.acos(dot.clamp(max=1 - 1e-7))
                sin_angle = torch.sin(angle)
                curved = (torch.sin((1 - fraction) * angle) * a + torch.sin(fraction * angle) * b) / sin_angle
                linear = a + fraction * (b - a)
                value = torch.where(dot > 0.9995, linear, curved)
                value = torch.nn.functional.normalize(value, dim=-1)
            else:
                value = a + fraction * (b - a)
            if 'vel' in key:
                value = value * rate.reshape((-1,) + (1,) * (array.ndim - 1))
            result[key] = value
        # Full reset qpos must use the same shortest-arc quaternion interpolation.
        if 'qpos' in result:
            result['qpos'][:, 3:7] = result['root_quat_w']
        return result


@dataclass(kw_only=True)
class PunchCommandCfg(CommandTermCfg):
    reference_path: str
    target_pool_path: str
    family: str = 'jab'
    entity_name: str = 'robot'
    recovery_fraction: float = 0.2
    target_split: str = 'train'
    ttc_range: tuple[float, float] | None = None
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    debug_vis: bool = True

    def __post_init__(self):
        if self.family not in FAMILY_ACTIVE_HAND:
            raise ValueError(f'Unknown punch family: {self.family!r}')

    def build(self, env):
        return PunchCommand(self, env)


class PunchCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from .robot import GLOVE_SITE_NAMES
        from .targets import TargetPool
        self.robot = env.scene[cfg.entity_name]
        self.reference = TorchReference(cfg.reference_path, self.device, self.robot.joint_names, family=cfg.family)
        self.pool = TargetPool.load(cfg.target_pool_path)
        self.targets = torch.as_tensor(self.pool.points(cfg.target_split, cfg.family), dtype=torch.float32, device=self.device)
        if self.targets.ndim != 2 or self.targets.shape[1] != 3 or len(self.targets) == 0:
            raise ValueError('Target pool must supply nonempty [N,3] real snapshot points')
        self.site_ids = self.robot.find_sites(GLOVE_SITE_NAMES, preserve_order=True)[0]
        self.active_index = 0 if FAMILY_ACTIVE_HAND[cfg.family] == 'left' else 1
        self.root_index = self.reference.body_names.index('pelvis')
        self.torso_index = self.robot.body_names.index('torso_link')
        self.reference_torso_index = self.reference.body_names.index('torso_link')
        self.state = PunchState(self.num_envs, self.device, env.physics_dt)
        self.target_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.horizon = torch.ones(self.num_envs, device=self.device)
        self.interaction_z = torch.ones_like(self.horizon)
        self.guard_z = torch.ones_like(self.horizon)
        self.source_time = torch.zeros_like(self.horizon)
        self.source_rate = torch.zeros_like(self.horizon)
        self.ref = self.reference.sample(self.source_time, self.source_rate)
        self.terminal = self.reference.sample(torch.full_like(self.horizon, self.reference.end_time), torch.zeros_like(self.horizon))
        self.glove_velocity = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.previous_glove = torch.zeros_like(self.glove_velocity)
        self.pending_launch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.component_totals = {}
        self.reset_counts = {'full': 0, 'snippet': 0}
        self.transition_counts = {'full': 0, 'snippet': 0}
        self.ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.steps_seen = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.settled_duration = torch.zeros_like(self.horizon)
        self.first_settled_elapsed = torch.full_like(self.horizon, float('nan'))

    @property
    def command(self):
        return torch.cat((self.ref['joint_pos'], self.ref['joint_vel']), dim=-1)

    @property
    def target_relative(self):
        return quat_apply_inverse(self.robot.data.root_link_quat_w,
                                  self.state.target_world - self.robot.data.root_link_pos_w)

    def refresh_reference(self):
        self.source_time, self.source_rate = reference_clock(
            self.state.elapsed, self.state.requested_ttc, self.reference.strike_time,
            self.reference.end_time, self.reference.recovery)
        self.ref = self.reference.sample(self.source_time, self.source_rate)

    def _resample_command(self, env_ids):
        n = len(env_ids)
        t0 = max(0.5, self.reference.strike_time)
        bounds = self.cfg.ttc_range or (max(0.5, 0.9 * t0), 1.1 * t0)
        requested = torch.empty(n, device=self.device).uniform_(*bounds)
        known = torch.rand(n, device=self.device) >= self.cfg.recovery_fraction
        horizon, zi, zg = scheduled_normalizers(requested, self.reference.recovery, self._env.step_dt)
        # Quantized command ages preserve the original attempt's actor grid.
        first_recovery_step = torch.floor((requested + 0.10) / self._env.step_dt).long() + 1
        last_recovery_step = torch.floor((requested + self.reference.recovery) / self._env.step_dt).long()
        sample_step = first_recovery_step + (torch.rand(n, device=self.device) * (last_recovery_step - first_recovery_step + 1)).long()
        elapsed = torch.where(known, torch.zeros_like(requested), sample_step * self._env.step_dt)
        indices = torch.randint(len(self.targets), (n,), device=self.device)
        self.target_index[env_ids] = indices
        target = self.targets[indices] + self._env.scene.env_origins[env_ids]
        self.state.reset(env_ids, target, requested, elapsed, known)
        self.horizon[env_ids], self.interaction_z[env_ids], self.guard_z[env_ids] = horizon, zi, zg
        self.refresh_reference()
        root_pos = self.ref['root_pos_w'][env_ids] + self._env.scene.env_origins[env_ids]
        root_quat = self.ref['root_quat_w'][env_ids]
        root_vel = torch.cat((self.ref['body_lin_vel_w'][env_ids, self.root_index],
                              self.ref['body_ang_vel_w'][env_ids, self.root_index]), dim=-1)
        root_vel[known] = 0.0
        joint_vel = self.ref['joint_vel'][env_ids].clone()
        joint_vel[known] = 0.0
        self.robot.write_root_state_to_sim(torch.cat((root_pos, root_quat, root_vel), dim=-1), env_ids)
        self.robot.write_joint_state_to_sim(self.ref['joint_pos'][env_ids], joint_vel, env_ids=env_ids)
        # Match previous action to initialized posture under q_default + scale*a.
        term = self._env.action_manager.get_term('joint_pos')
        previous = (self.ref['joint_pos'][env_ids][:, term.target_ids] - term.offset[env_ids]) / term.scale[env_ids]
        for action in (self._env.action_manager.action, self._env.action_manager.prev_action,
                       self._env.action_manager.prev_prev_action):
            action[env_ids] = previous
        term._raw_actions[env_ids] = previous
        self.robot.set_joint_position_target(self.ref['joint_pos'][env_ids], env_ids=env_ids)
        self.pending_launch[env_ids] = True
        self.ready[env_ids] = True
        self.steps_seen[env_ids] = 0
        self.settled_duration[env_ids] = 0
        self.first_settled_elapsed[env_ids] = float('nan')
        for values in self.component_totals.values():
            values[env_ids] = 0.0
        self.reset_counts['full'] += int(known.sum())
        self.reset_counts['snippet'] += int((~known).sum())

    def _update_command(self, env_ids=None):
        ids = self.pending_launch.nonzero().flatten()
        if len(ids):
            glove = self.robot.data.site_pos_w[ids][:, self.site_ids]
            self.previous_glove[ids] = glove
            self.glove_velocity[ids] = self.ref['glove_lin_vel_w'][ids]
            self.glove_velocity[ids[self.state.history_known[ids]]] = 0
            self.state.observe(glove[:, self.active_index], glove[:, 1 - self.active_index], ids=ids, launch=True)
            self.pending_launch[ids] = False
        self.refresh_reference()

    def _update_metrics(self):
        pass

    def accumulate(self, name, rate):
        if not bool(torch.isfinite(rate).all()):
            raise FloatingPointError(f'Nonfinite {name} reward before RewardManager sanitization')
        if name not in self.component_totals:
            # This lazy allocation may first occur in PPO's inference context;
            # episode totals must remain mutable during an ordinary explicit reset.
            with torch.inference_mode(False):
                self.component_totals[name] = torch.zeros(self.num_envs, device=self.device)
        self.component_totals[name] += rate.detach() * self._env.step_dt
        return rate

    def _debug_vis_impl(self, visualizer):
        idx = visualizer.env_idx
        visualizer.add_sphere(self.state.target_world[idx].detach().cpu().numpy(), 0.06,
                              (0.95, 0.2, 0.2, 0.35), label='frozen non-solid head')
