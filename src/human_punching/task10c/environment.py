"""Thin mjlab environment adapter for true episode totals and stream ordering."""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from . import mdp


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


class PunchEnv(ManagerBasedRlEnv):
    """Use the existing manager step/physics loop and PPO-compatible interface."""
    def __init__(self, *args, **kwargs):
        self.episode_records = []
        super().__init__(*args, **kwargs)
        self.command_manager.get_term('punch').episode_records = self.episode_records

    @contextmanager
    def _stream(self):
        # mjlab's WarpBridge writes on Warp's stream. Running torch callbacks on
        # this same stream makes kinematics -> site read -> latch ordering explicit.
        if str(self.device).startswith('cuda'):
            caller = torch.cuda.current_stream(self.device)
            stream = torch.cuda.ExternalStream(wp.get_stream(self.sim.wp_device).cuda_stream,
                                                device=self.device)
            stream.wait_stream(caller)
            try:
                with torch.cuda.stream(stream):
                    yield
            finally:
                caller.wait_stream(stream)
        else:
            yield

    def reset(self, **kwargs):
        with self._stream():
            return super().reset(**kwargs)

    def step(self, action):
        with self._stream():
            if not bool(torch.isfinite(action).all()):
                raise FloatingPointError('Nonfinite policy action before physics; inspect last setup checkpoint')
            c = self.command_manager.get_term('punch')
            c.steps_seen += 1
            c.transition_counts['full'] += int(c.state.history_known.sum())
            c.transition_counts['snippet'] += int((~c.state.history_known).sum())
            return super().step(action)

    def _reset_idx(self, env_ids=None):
        c = self.command_manager.get_term('punch')
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        completed = ids[c.ready[ids] & (c.steps_seen[ids] > 0)]
        if len(completed):
            error, speed = mdp.guard_errors(self)
            falls = mdp.fallen(self)
            # Transfer a compact batch once, rather than synchronizing CUDA for
            # every scalar of every finished episode at substantive PPO scale.
            names = ('known', 'hit', 'actual', 'target_index', 'requested', 'start',
                     'end', 'phase', 'horizon', 'steps', 'launch', 'overlap', 'inactive',
                     'inactive_time', 'fall', 'complete', 'guard_error', 'guard_speed', 'settled')
            values = (c.state.history_known, c.state.encountered, c.state.first_encounter_elapsed,
                      c.target_index, c.state.requested_ttc, c.state.start_elapsed,
                      c.state.elapsed, c.source_time, c.horizon, c.steps_seen, c.state.launch_time,
                      c.state.launch_overlap, c.state.inactive_encountered, c.state.first_inactive_elapsed,
                      falls, c.state.elapsed >= c.horizon - 1e-6, error, speed, c.first_settled_elapsed)
            component_names = tuple(c.component_totals)
            columns = torch.stack([v[completed].float() for v in values]
                                  + [c.component_totals[n][completed] for n in component_names], dim=1).cpu().tolist()
            targets = c.state.target_world[completed].cpu().tolist()
            snapshot_ids = c.pool.snapshot_ids(c.cfg.target_split)
            for idx, column, target in zip(completed.tolist(), columns, targets):
                row = dict(zip(names, column))
                known = bool(row['known'])
                hit = bool(row['hit']) if known else None
                actual = _number(row['actual']) if known else None
                self.episode_records.append({
                    'family': c.cfg.family, 'environment': idx,
                    'reset_kind': 'full' if known else 'recovery_snippet',
                    'snapshot_id': int(snapshot_ids[int(row['target_index'])]),
                    'target_world': target,
                    'requested_ttc_s': row['requested'],
                    'elapsed_start_s': row['start'],
                    'elapsed_end_s': row['end'],
                    'source_phase_end_s': row['phase'],
                    'horizon_s': row['horizon'], 'actor_steps': int(row['steps']),
                    'hit': hit, 'actual_ttc_s': actual,
                    'launch_local_sim_time_s': row['launch'],
                    'first_encounter_local_sim_time_s': None if actual is None else actual + row['launch'],
                    'signed_timing_error_s': None if actual is None else actual - row['requested'],
                    'launch_overlap': bool(row['overlap']),
                    'inactive_encounter': bool(row['inactive']),
                    'first_inactive_elapsed_s': _number(row['inactive_time']) if known else None,
                    'snippet_observed_encounter': bool(row['hit']) if not known else None,
                    'fall': bool(row['fall']), 'complete_horizon': bool(row['complete']),
                    'guard_glove_error_m': row['guard_error'], 'guard_glove_speed_m_s': row['guard_speed'],
                    'settled_guard_elapsed_s': _number(row['settled']),
                    'raw_component_totals': dict(zip(component_names, column[len(names):])),
                })
        super()._reset_idx(env_ids)
