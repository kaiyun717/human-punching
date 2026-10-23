"""Stored-reference command, adapted from TaskNPoint's motion command structure.

TaskNPoint b8faaf284e0cef8021b94e2219492dbd54efba0c supplies the six-array
motion-loader/command pattern. Task10C adds named maps, fractional sampling,
committed timing, complete recovery and unknown-history snippet initialization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .events import PunchState, reference_clock, scheduled_normalizers

FAMILY_ACTIVE_HAND = {'jab': 'left', 'cross': 'right', 'left_hook': 'left', 'right_hook': 'right'}


class TorchReference:
    def __init__(self, path, device, joint_names, family=None, *, clock_mode='legacy',
                 ttc_anchor_time=None, post_entry_duration=None):
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
        self.clock_mode = clock_mode
        self.ttc_anchor_time = self.strike_time if ttc_anchor_time is None else float(ttc_anchor_time)
        if post_entry_duration is not None:
            self.recovery = float(post_entry_duration)
        if clock_mode not in ('legacy', 'smooth_entry'):
            raise ValueError('Unknown reference clock mode')
        if (not 0 < self.ttc_anchor_time < self.end_time
                or not np.isfinite(self.recovery) or self.recovery <= 0.1):
            raise ValueError('Invalid reference encounter anchor or post-entry duration')
        if clock_mode == 'smooth_entry':
            nominal_entry = self.metadata.get('nominal_first_entry_s')
            if (ttc_anchor_time is None or post_entry_duration is None or nominal_entry is None
                    or not np.isclose(self.ttc_anchor_time, nominal_entry, atol=1e-8, rtol=0)):
                raise ValueError('Smooth entry clock requires an explicit measured first-entry anchor and duration')

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
    target_fraction: float = 1.0
    easy_fraction: float = 0.5
    easy_ttc_range: tuple[float, float] | None = None
    clock_mode: str = 'legacy'
    ttc_anchor_time: float | None = None
    post_entry_duration: float | None = None
    reward_scales: dict[str, float] = field(default_factory=lambda: {
        'balance': 1.0, 'guard': 1.0, 'style': 1.0})
    guard_speed_profile: str = 'gaussian'
    reward_mode: str = 'whole_body'
    imitation_budget: float = 7.0
    recovery_budget: float = 6.0
    command_excess_weight: float = 1.0
    imitation_frame: str = 'relative'
    task_rewards_enabled: bool = True
    fixed_target_snapshot_id: int | None = None
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    debug_vis: bool = True

    def __post_init__(self):
        if self.family not in FAMILY_ACTIVE_HAND:
            raise ValueError(f'Unknown punch family: {self.family!r}')
        if self.clock_mode not in ('legacy', 'smooth_entry'):
            raise ValueError('Unknown reference clock mode')
        if self.easy_ttc_range is not None and (len(self.easy_ttc_range) != 2
                or not all(np.isfinite(x) for x in self.easy_ttc_range)
                or not 0 < self.easy_ttc_range[0] <= self.easy_ttc_range[1]):
            raise ValueError('Explicit easy TTC range must be positive, finite and ordered')
        if (set(self.reward_scales) - {'balance', 'guard', 'style'}
                or any(not np.isfinite(value) or value < 0 for value in self.reward_scales.values())):
            raise ValueError('Reward budget scales must be finite, nonnegative balance/guard/style values')
        if self.guard_speed_profile not in ('gaussian', 'rational'):
            raise ValueError('Guard speed profile must be gaussian or rational')
        if self.reward_mode not in ('whole_body', 'legacy'):
            raise ValueError('Reward mode must be whole_body or legacy')
        if self.imitation_frame not in ('relative', 'launch_registered'):
            raise ValueError('Imitation frame must be relative or launch_registered')
        if self.imitation_frame == 'launch_registered' and self.reward_mode != 'whole_body':
            raise ValueError('Launch registration requires the six-term whole-body reward')
        if not isinstance(self.task_rewards_enabled, bool):
            raise ValueError('task_rewards_enabled must be a boolean')
        if (self.fixed_target_snapshot_id is not None
                and (isinstance(self.fixed_target_snapshot_id, bool)
                     or not isinstance(self.fixed_target_snapshot_id, int)
                     or self.fixed_target_snapshot_id < 0)):
            raise ValueError('Fixed target snapshot ID must be a nonnegative integer')
        if any(not np.isfinite(value) or value < 0 for value in
               (self.imitation_budget, self.recovery_budget, self.command_excess_weight)):
            raise ValueError('Whole-body reward budgets and command cost must be finite and nonnegative')

    def build(self, env):
        return PunchCommand(self, env)


class PunchCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from .robot import GLOVE_SITE_NAMES
        from .targets import TargetPool
        self.robot = env.scene[cfg.entity_name]
        # Registration is the existing scene translation with identity rotation.
        # Plane origins are fixed for the scene lifetime. Capture a separate copy
        # so neither reset nor imitation ever derives placement from a learned
        # pelvis. The source root trajectory itself remains intact in both arms.
        self.reference_origin_w = env.scene.env_origins.clone()
        self.reference = TorchReference(cfg.reference_path, self.device, self.robot.joint_names, family=cfg.family,
                                        clock_mode=cfg.clock_mode, ttc_anchor_time=cfg.ttc_anchor_time,
                                        post_entry_duration=cfg.post_entry_duration)
        self.pool = TargetPool.load(cfg.target_pool_path)
        self.targets = torch.as_tensor(self.pool.points(cfg.target_split, cfg.family), dtype=torch.float32, device=self.device)
        if self.targets.ndim != 2 or self.targets.shape[1] != 3 or len(self.targets) == 0:
            raise ValueError('Target pool must supply nonempty [N,3] real snapshot points')
        self.fixed_target_index = None
        if cfg.fixed_target_snapshot_id is not None:
            matches = np.flatnonzero(self.pool.snapshot_ids(cfg.target_split) == cfg.fixed_target_snapshot_id)
            if len(matches) != 1:
                raise ValueError('Fixed target snapshot must exist exactly once in the selected cached split')
            self.fixed_target_index = int(matches[0])
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
        self.evaluation_recipes = None
        self.evaluation_recipe_ids = []
        self.set_curriculum({'stage': 0, 'target_fraction': cfg.target_fraction,
                             'ttc_range': cfg.ttc_range, 'easy_fraction': cfg.easy_fraction})

    def set_curriculum(self, state):
        """Change future resets only; active commands remain committed."""
        from .targets import target_subset_indices
        t0 = max(0.5, self.reference.strike_time)
        initial_ttc = tuple(self.cfg.easy_ttc_range or (max(0.5, 0.9 * t0), 1.1 * t0))
        bounds = tuple(state.get('ttc_range') or initial_ttc)
        fraction = float(state.get('target_fraction', 1.0))
        easy_fraction = float(state.get('easy_fraction', 0.5))
        if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1] or not 0 <= easy_fraction <= 1:
            raise ValueError('Invalid curriculum timing or easy-case fraction')
        points = self.pool.points(self.cfg.target_split, self.cfg.family)
        self.curriculum = {**state, 'ttc_range': list(bounds), 'target_fraction': fraction,
                           'easy_fraction': easy_fraction, 'easy_ttc_range': list(initial_ttc)}
        if (self.reference.clock_mode == 'smooth_entry'
                and self.reference.ttc_anchor_time / bounds[0]
                > 3 * (self.reference.end_time - self.reference.ttc_anchor_time) / self.reference.recovery):
            raise ValueError('Requested TTC is outside the monotone Hermite clock domain')
        self.target_candidates = torch.as_tensor(target_subset_indices(points, fraction), device=self.device)
        self.easy_target_candidates = torch.as_tensor(target_subset_indices(points, min(0.25, fraction)), device=self.device)
        if self.fixed_target_index is not None:
            # Select the real cached point, without changing its coordinates or
            # the full/easy RNG branches and the full/snippet reset mixture.
            self.target_candidates = torch.tensor([self.fixed_target_index], device=self.device)
            self.easy_target_candidates = self.target_candidates.clone()
        self.initial_ttc_range = initial_ttc
        self.cfg.ttc_range = bounds
        self.cfg.target_fraction = fraction
        self.cfg.easy_fraction = easy_fraction

    def set_evaluation_recipes(self, recipes):
        """Install deterministic guard-start recipes, mapped by environment index."""
        rows = recipes['rows']
        if not rows or recipes.get('split', self.cfg.target_split) != self.cfg.target_split:
            raise ValueError('Evaluation recipes must match the nonempty target split')
        indices = [int(row['target_index']) for row in rows]
        requested = [float(row['requested_ttc']) for row in rows]
        if any(i < 0 or i >= len(self.targets) for i in indices) or any(t <= 0 or not np.isfinite(t) for t in requested):
            raise ValueError('Invalid evaluation target index or TTC')
        if self.fixed_target_index is not None and any(i != self.fixed_target_index for i in indices):
            raise ValueError('Evaluation recipes must preserve the configured fixed snapshot')
        if (self.reference.clock_mode == 'smooth_entry'
                and any(self.reference.ttc_anchor_time / t
                        > 3 * (self.reference.end_time - self.reference.ttc_anchor_time) / self.reference.recovery
                        for t in requested)):
            raise ValueError('Evaluation TTC is outside the monotone Hermite clock domain')
        self.evaluation_recipes = (
            torch.tensor(indices, dtype=torch.long, device=self.device),
            torch.tensor(requested, dtype=torch.float32, device=self.device))
        self.evaluation_recipe_ids = [row['recipe_id'] for row in rows]

    @property
    def command(self):
        return torch.cat((self.ref['joint_pos'], self.ref['joint_vel']), dim=-1)

    @property
    def target_relative(self):
        return quat_apply_inverse(self.robot.data.root_link_quat_w,
                                  self.state.target_world - self.robot.data.root_link_pos_w)

    def refresh_reference(self):
        self.source_time, self.source_rate = reference_clock(
            self.state.elapsed, self.state.requested_ttc, self.reference.ttc_anchor_time,
            self.reference.end_time, self.reference.recovery, mode=self.reference.clock_mode)
        self.ref = self.reference.sample(self.source_time, self.source_rate)

    def _resample_command(self, env_ids):
        n = len(env_ids)
        if self.evaluation_recipes is not None:
            target_indices, requests = self.evaluation_recipes
            rows = env_ids % len(requests)
            requested = requests[rows]
            indices = target_indices[rows]
            known = torch.ones(n, device=self.device, dtype=torch.bool)
        else:
            requested = torch.empty(n, device=self.device).uniform_(*self.cfg.ttc_range)
            indices = self.target_candidates[torch.randint(len(self.target_candidates), (n,), device=self.device)]
            easy = torch.rand(n, device=self.device) < self.cfg.easy_fraction
            # Broader domains retain the original easy targets and their timing.
            easy_requested = torch.empty(n, device=self.device).uniform_(*self.initial_ttc_range)
            if self.cfg.ttc_range[0] <= self.initial_ttc_range[0] and self.cfg.ttc_range[1] >= self.initial_ttc_range[1]:
                requested = torch.where(easy, easy_requested, requested)
            easy_indices = self.easy_target_candidates[torch.randint(len(self.easy_target_candidates), (n,), device=self.device)]
            indices = torch.where(easy, easy_indices, indices)
            known = torch.rand(n, device=self.device) >= self.cfg.recovery_fraction
        horizon, zi, zg = scheduled_normalizers(requested, self.reference.recovery, self._env.step_dt)
        # Quantized command ages preserve the original attempt's actor grid.
        first_recovery_step = torch.floor((requested + 0.10) / self._env.step_dt).long() + 1
        last_recovery_step = torch.floor((requested + self.reference.recovery) / self._env.step_dt).long()
        sample_step = first_recovery_step + (torch.rand(n, device=self.device) * (last_recovery_step - first_recovery_step + 1)).long()
        elapsed = torch.where(known, torch.zeros_like(requested), sample_step * self._env.step_dt)
        self.target_index[env_ids] = indices
        target = self.targets[indices] + self._env.scene.env_origins[env_ids]
        self.state.reset(env_ids, target, requested, elapsed, known)
        self.horizon[env_ids], self.interaction_z[env_ids], self.guard_z[env_ids] = horizon, zi, zg
        self.refresh_reference()
        root_pos = self.ref['root_pos_w'][env_ids] + self.reference_origin_w[env_ids]
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
        # Budget scaling occurs before both accounting and RewardManager's dt
        # integration; the recorded component is the actual learner contribution.
        rate = rate * self.cfg.reward_scales.get(name, 1.0)
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
