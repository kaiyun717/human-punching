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
            for idx in completed.tolist():
                known = bool(c.state.history_known[idx])
                hit = bool(c.state.encountered[idx]) if known else None
                actual = _number(c.state.first_encounter_elapsed[idx]) if known else None
                self.episode_records.append({
                    'family': c.cfg.family, 'environment': idx,
                    'reset_kind': 'full' if known else 'recovery_snippet',
                    'snapshot_id': int(c.pool.snapshot_ids(c.cfg.target_split)[int(c.target_index[idx])]),
                    'target_world': c.state.target_world[idx].cpu().tolist(),
                    'requested_ttc_s': float(c.state.requested_ttc[idx]),
                    'elapsed_start_s': float(c.state.start_elapsed[idx]),
                    'elapsed_end_s': float(c.state.elapsed[idx]),
                    'source_phase_end_s': float(c.source_time[idx]),
                    'horizon_s': float(c.horizon[idx]), 'actor_steps': int(c.steps_seen[idx]),
                    'hit': hit, 'actual_ttc_s': actual,
                    'launch_local_sim_time_s': float(c.state.launch_time[idx]),
                    'first_encounter_local_sim_time_s': None if actual is None else actual + float(c.state.launch_time[idx]),
                    'signed_timing_error_s': None if actual is None else actual - float(c.state.requested_ttc[idx]),
                    'launch_overlap': bool(c.state.launch_overlap[idx]),
                    'inactive_encounter': bool(c.state.inactive_encountered[idx]),
                    'snippet_observed_encounter': bool(c.state.encountered[idx]) if not known else None,
                    'fall': bool(falls[idx]), 'complete_horizon': bool(c.state.elapsed[idx] >= c.horizon[idx] - 1e-6),
                    'guard_glove_error_m': float(error[idx]), 'guard_glove_speed_m_s': float(speed[idx]),
                    'settled_guard_elapsed_s': _number(c.first_settled_elapsed[idx]),
                    'raw_component_totals': {name: float(values[idx]) for name, values in c.component_totals.items()},
                })
        super()._reset_idx(env_ids)
