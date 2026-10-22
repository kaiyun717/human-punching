"""Task10C clocks and first-encounter state; target geometry is never solid."""
from __future__ import annotations

import torch


def reference_clock(elapsed, requested_ttc, strike_time, end_time, recovery):
    """Return source seconds and ds/dt, including a zero-velocity terminal hold."""
    attack = elapsed <= requested_ttc
    hold = elapsed >= requested_ttc + recovery - 1e-6
    attack_rate = strike_time / requested_ttc
    recovery_rate = (end_time - strike_time) / recovery
    phase = torch.where(attack, elapsed * attack_rate,
                        strike_time + (elapsed - requested_ttc) * recovery_rate)
    phase = torch.where(hold, torch.full_like(phase, end_time), phase)
    rate = torch.where(hold, torch.zeros_like(elapsed),
                       torch.where(attack, attack_rate, recovery_rate))
    return phase.clamp(0.0, end_time), rate


def interaction_weight(elapsed, requested_ttc):
    return (torch.exp(-0.5 * ((elapsed - requested_ttc) / 0.15) ** 2)
            * (elapsed >= (requested_ttc - 0.30).clamp_min(0.0))
            * (elapsed <= requested_ttc + 0.10))


def guard_weight(elapsed, requested_ttc, recovery):
    return ((elapsed - requested_ttc - 0.10) / (recovery - 0.10)).clamp(0.0, 1.0)


def scheduled_normalizers(requested_ttc, recovery, step_dt, hold=0.4):
    """Normalize once on the full-attempt actor grid, before hit/snippet masking."""
    horizon = torch.ceil((requested_ttc + recovery + hold) / step_dt) * step_dt
    count = int(torch.ceil(horizon.max() / step_dt).item())
    grid = torch.arange(1, count + 1, device=requested_ttc.device) * step_dt
    valid = grid[None, :] <= horizon[:, None] + 1e-6
    interaction = interaction_weight(grid[None, :], requested_ttc[:, None])
    guard = guard_weight(grid[None, :], requested_ttc[:, None], recovery)
    return horizon, (interaction * valid).sum(-1) * step_dt, (guard * valid).sum(-1) * step_dt


class PunchState:
    """Independent per-env state with a pulse that survives all four substeps."""
    encounter_radius = 0.14

    def __init__(self, num_envs, device, physics_dt=0.005):
        self.physics_dt = physics_dt
        self.target_world = torch.zeros(num_envs, 3, device=device)
        self.requested_ttc = torch.ones(num_envs, device=device)
        self.start_elapsed = torch.zeros(num_envs, device=device)
        self.physics_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.history_known = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.encountered = torch.zeros_like(self.history_known)
        self.inactive_encountered = torch.zeros_like(self.history_known)
        self.pending = torch.zeros_like(self.history_known)
        self.launch_overlap = torch.zeros_like(self.history_known)
        self.first_encounter_elapsed = torch.full((num_envs,), float('nan'), device=device)
        self.first_inactive_elapsed = torch.full_like(self.first_encounter_elapsed, float('nan'))

    @property
    def elapsed(self):
        return self.start_elapsed + self.physics_steps * self.physics_dt

    @property
    def local_sim_time(self):
        return self.physics_steps * self.physics_dt

    @property
    def launch_time(self):
        # Each reset starts a local simulator clock. A recovery snippet's
        # conceptual command launch predates that reset; its history is unknown.
        return -self.start_elapsed

    @property
    def remaining(self):
        return self.requested_ttc - self.elapsed

    def reset(self, ids, target_world, requested_ttc, start_elapsed, history_known):
        self.target_world[ids] = target_world
        self.requested_ttc[ids] = requested_ttc
        self.start_elapsed[ids] = start_elapsed
        self.physics_steps[ids] = 0
        self.history_known[ids] = history_known
        for value in (self.encountered, self.inactive_encountered, self.pending, self.launch_overlap):
            value[ids] = False
        self.first_encounter_elapsed[ids] = float('nan')
        self.first_inactive_elapsed[ids] = float('nan')

    def observe(self, active_glove, inactive_glove, ids=None, launch=False):
        if ids is None:
            ids = torch.arange(len(self.requested_ttc), device=self.requested_ttc.device)
        active_inside = torch.linalg.vector_norm(active_glove - self.target_world[ids], dim=-1) <= self.encounter_radius
        inactive_inside = torch.linalg.vector_norm(inactive_glove - self.target_world[ids], dim=-1) <= self.encounter_radius
        new = active_inside & ~self.encountered[ids]
        new_other = inactive_inside & ~self.inactive_encountered[ids]
        # Snippets may log their first *observed* entry, but cannot invent actual TTC.
        self.first_encounter_elapsed[ids] = torch.where(new & self.history_known[ids], self.elapsed[ids], self.first_encounter_elapsed[ids])
        self.first_inactive_elapsed[ids] = torch.where(new_other & self.history_known[ids], self.elapsed[ids], self.first_inactive_elapsed[ids])
        self.encountered[ids] |= active_inside
        self.inactive_encountered[ids] |= inactive_inside
        self.pending[ids] |= new & self.history_known[ids]
        if launch:
            self.launch_overlap[ids] = active_inside & self.history_known[ids]

    def event_bonus(self):
        time = torch.nan_to_num(self.first_encounter_elapsed)
        eligible = self.pending & self.history_known & (time <= self.requested_ttc + 0.10)
        bonus = (2.0 + 8.0 * torch.exp(-0.5 * ((time - self.requested_ttc) / 0.10) ** 2)) * eligible
        self.pending.zero_()
        return bonus


def physics_event_callback(env):
    """Refresh only kinematics at integrated post-step qpos; never rerun contacts."""
    import mujoco_warp as mjwarp
    import warp as wp

    command = env.command_manager.get_term('punch')
    if env.sim.nan_guard.enabled and env.sim.nan_guard._dumped:
        raise FloatingPointError('Nonfinite physics state; numerical_diagnostics contains native state dump')
    with wp.ScopedDevice(env.sim.wp_device):
        mjwarp.kinematics(env.sim.wp_model, env.sim.wp_data)
    command.state.physics_steps += 1
    glove = command.robot.data.site_pos_w[:, command.site_ids]
    if not bool(torch.isfinite(glove).all() & torch.isfinite(env.sim.data.qpos[:]).all()
                & torch.isfinite(env.sim.data.qvel[:]).all()):
        raise FloatingPointError('Nonfinite integrated state/glove geometry before event/reward computation')
    command.glove_velocity.copy_((glove - command.previous_glove) / env.physics_dt)
    command.previous_glove.copy_(glove)
    command.state.observe(glove[:, command.active_index], glove[:, 1 - command.active_index])
    # Refreshed post-step geometry and scoring share this same authoritative clock.
    command.refresh_reference()
    return command.state.encountered.float()
