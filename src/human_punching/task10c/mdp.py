"""TaskNPoint-style observations/rewards specialized to a complete H2 punch.

All dense terms are rates. RewardManager applies exactly one actor-dt factor;
the encounter pulse alone is divided by actor dt. No target contact forces exist.
"""
from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_inv, quat_mul
from .events import guard_weight, interaction_weight
from .imitation import (
    H2_TRACKED_BODY_NAMES, command_excess_loss, imitation_rates,
    named_joint_groups, whole_body_scores,
)

OBSERVATION_SCHEMA = (
    ('joint_position_rad', 31), ('joint_velocity_rad_s', 31),
    ('anchor_linear_velocity_m_s', 3), ('anchor_angular_velocity_rad_s', 3),
    ('projected_gravity', 3), ('base_height_m', 1), ('previous_action', 31),
    ('reference_joint_position_rad', 31), ('retimed_reference_joint_velocity_rad_s', 31),
    ('reference_anchor_orientation_wxyz_relative_to_h2', 4),
    ('frozen_target_relative_to_h2_m', 3), ('requested_ttc_s', 1),
    ('signed_time_remaining_s', 1), ('normalized_source_phase', 1),
    ('terminal_hold', 1), ('encounter_history_known', 1), ('first_encounter_latched', 1),
)


def command(env):
    return env.command_manager.get_term('punch')


def observations(env):
    c = command(env)
    d = c.robot.data
    # Launch registration has identity rotation: source orientation and world
    # velocities are already in the registered scene frame in both controls.
    relative_orientation = quat_mul(quat_inv(d.root_link_quat_w), c.ref['root_quat_w'])
    orientation_sign = torch.where(relative_orientation[:, :1] < 0, -1.0, 1.0)
    temporal = torch.stack((c.state.requested_ttc, c.state.remaining,
                            c.source_time / c.reference.end_time,
                            (c.state.elapsed >= c.state.requested_ttc + c.reference.recovery - 1e-6).float(),
                            c.state.history_known.float(), c.state.encountered.float()), dim=-1)
    return torch.cat((d.joint_pos, d.joint_vel, d.root_link_lin_vel_b, d.root_link_ang_vel_b,
                      d.projected_gravity_b, d.root_link_pos_w[:, 2:3], env.action_manager.action,
                      c.ref['joint_pos'], c.ref['joint_vel'], relative_orientation * orientation_sign,
                      c.target_relative, temporal), dim=-1)


def encounter(env):
    c = command(env)
    return c.accumulate('encounter', c.state.event_bonus() / env.step_dt)


def interaction(env):
    c = command(env)
    glove = c.robot.data.site_pos_w[:, c.site_ids[c.active_index]]
    distance = (torch.linalg.vector_norm(glove - c.state.target_world, dim=-1) - 0.14).clamp_min(0)
    proximity = torch.exp(-0.5 * (distance / 0.15) ** 2)
    velocity = c.glove_velocity[:, c.active_index]
    reference_velocity = c.ref['glove_lin_vel_w'][:, c.active_index]
    direction = ((torch.nn.functional.normalize(velocity, dim=-1)
                  * torch.nn.functional.normalize(reference_velocity, dim=-1)).sum(-1) + 1) / 2
    speed = torch.exp(-0.5 * ((velocity.norm(dim=-1) - reference_velocity.norm(dim=-1)) / 2.0) ** 2)
    quat = c.robot.data.site_quat_w[:, c.site_ids[c.active_index]]
    orientation = (quat * c.ref['glove_quat_w'][:, c.active_index]).sum(-1).square().clamp(0, 1)
    quality = proximity * (0.8 + 0.2 * direction * orientation * speed)
    credit = c.state.history_known & ~c.state.encountered
    value = 6 * interaction_weight(c.state.elapsed, c.state.requested_ttc) * quality * credit / c.interaction_z
    return c.accumulate('interaction', value)


def _relative_positions(position, anchor_position, anchor_quat):
    return quat_apply_inverse(anchor_quat[:, None, :].expand(-1, position.shape[1], -1),
                              position - anchor_position[:, None, :])


def _whole_body_rates(env):
    """Gather and score once per actor transition, then log six components.

    The reference object is included in the cache key so an explicit reset or
    reference refresh at the same global step cannot reuse stale scores.
    """
    c = command(env)
    if (getattr(c, '_imitation_step', None) == env.common_step_counter
            and getattr(c, '_imitation_reference', None) is c.ref):
        return c._imitation_rates
    if not hasattr(c, '_imitation_robot_ids'):
        c._imitation_robot_ids = [c.robot.body_names.index(name) for name in H2_TRACKED_BODY_NAMES]
        c._imitation_reference_ids = [c.reference.body_names.index(name) for name in H2_TRACKED_BODY_NAMES]
    d = c.robot.data
    actual_ids, reference_ids = c._imitation_robot_ids, c._imitation_reference_ids
    scores = whole_body_scores(
        d.body_link_pos_w[:, actual_ids], d.body_link_quat_w[:, actual_ids],
        d.body_link_lin_vel_w[:, actual_ids], d.body_link_ang_vel_w[:, actual_ids],
        c.ref['body_pos_w'][:, reference_ids] + c.reference_origin_w[:, None, :],
        c.ref['body_quat_w'][:, reference_ids],
        c.ref['body_lin_vel_w'][:, reference_ids], c.ref['body_ang_vel_w'][:, reference_ids],
        imitation_frame=c.cfg.imitation_frame)
    rates = imitation_rates(
        scores, c.horizon,
        guard_weight(c.state.elapsed, c.state.requested_ttc, c.reference.recovery),
        c.guard_z, c.cfg.imitation_budget, c.cfg.recovery_budget)
    update_hand_settling_diagnostic(env)
    c._imitation_scores = scores
    c._imitation_rates = rates
    c._imitation_step = env.common_step_counter
    c._imitation_reference = c.ref
    return rates


def anchor_height(env):
    return command(env).accumulate('anchor_height', _whole_body_rates(env)['anchor_height'])


def anchor_orientation(env):
    return command(env).accumulate('anchor_orientation', _whole_body_rates(env)['anchor_orientation'])


def body_position(env):
    return command(env).accumulate('body_position', _whole_body_rates(env)['body_position'])


def body_orientation(env):
    return command(env).accumulate('body_orientation', _whole_body_rates(env)['body_orientation'])


def body_linear_velocity(env):
    return command(env).accumulate('body_linear_velocity', _whole_body_rates(env)['body_linear_velocity'])


def body_angular_velocity(env):
    return command(env).accumulate('body_angular_velocity', _whole_body_rates(env)['body_angular_velocity'])


def command_excess(env):
    """All-joint physical target excess against named native hard limits."""
    c = command(env)
    if not hasattr(c, '_command_excess_groups'):
        c._command_excess_groups = named_joint_groups(c.robot.joint_names)
        model = env.sim.mj_model
        native_ids = [model.joint(f'{c.cfg.entity_name}/{name}').id for name in c.robot.joint_names]
        limits = model.jnt_range[native_ids]
        c._command_excess_hard_limits = torch.as_tensor(limits, device=env.device, dtype=torch.float32)
        if not bool(torch.isfinite(c._command_excess_hard_limits).all()
                    & (c._command_excess_hard_limits[:, 1] > c._command_excess_hard_limits[:, 0]).all()):
            raise ValueError('Each named H2 command needs finite, ordered native hard limits')
    # joint_pos_target is the actual physical PD target after the unchanged
    # affine action map and encoder-bias subtraction, in entity joint order.
    loss = command_excess_loss(c.robot.data.joint_pos_target,
                              c._command_excess_hard_limits, c._command_excess_groups)
    return c.accumulate('command_excess', -c.cfg.command_excess_weight * loss / c.horizon)


def style(env):
    """Historical six-origin surrogate, enabled only by legacy reward mode."""
    c = command(env)
    # Soft torso-relative limb style, with no global-root or exact-foot orientation loss.
    if not hasattr(c, 'style_robot_ids'):
        names = ('left_elbow_link', 'right_elbow_link', 'left_wrist_yaw_link',
                 'right_wrist_yaw_link', 'left_knee_link', 'right_knee_link')
        c.style_robot_ids = [c.robot.body_names.index(name) for name in names]
        c.style_ref_ids = [c.reference.body_names.index(name) for name in names]
    d = c.robot.data
    actual = _relative_positions(d.body_link_pos_w[:, c.style_robot_ids],
                                  d.body_link_pos_w[:, c.torso_index], d.body_link_quat_w[:, c.torso_index])
    target = _relative_positions(c.ref['body_pos_w'][:, c.style_ref_ids],
                                 c.ref['body_pos_w'][:, c.reference_torso_index], c.ref['body_quat_w'][:, c.reference_torso_index])
    pose_score = torch.exp(-((actual - target).square().sum(-1)).mean(-1) / 0.3 ** 2)
    actual_vel = d.body_link_lin_vel_w[:, c.style_robot_ids] - d.root_link_lin_vel_w[:, None, :]
    target_vel = c.ref['body_lin_vel_w'][:, c.style_ref_ids] - c.ref['body_lin_vel_w'][:, c.root_index, None, :]
    velocity_score = torch.exp(-(actual_vel - target_vel).square().mean((-1, -2)) / 1.5 ** 2)
    return c.accumulate('style', 2 * (0.75 * pose_score + 0.25 * velocity_score) / c.horizon)


def balance(env):
    """Historical upright/height surrogate, enabled only by legacy mode."""
    c = command(env)
    upright = (-c.robot.data.projected_gravity_b[:, 2]).clamp(0, 1)
    target_height = c.ref['root_pos_w'][:, 2]
    height = torch.exp(-0.5 * ((c.robot.data.root_link_pos_w[:, 2] - target_height) / 0.2) ** 2)
    motion = torch.exp(-0.1 * c.robot.data.root_link_ang_vel_b.square().sum(-1))
    return c.accumulate('balance', upright * height * motion / c.horizon)


def guard_errors(env):
    c = command(env)
    d = c.robot.data
    actual = _relative_positions(d.site_pos_w[:, c.site_ids], d.body_link_pos_w[:, c.torso_index],
                                  d.body_link_quat_w[:, c.torso_index])
    desired = _relative_positions(c.terminal['glove_pos_w'], c.terminal['body_pos_w'][:, c.reference_torso_index],
                                   c.terminal['body_quat_w'][:, c.reference_torso_index])
    error = (actual - desired).norm(dim=-1).mean(-1)
    speed = c.glove_velocity.norm(dim=-1).mean(-1)
    return error, speed


def guard_speed_score(speed: torch.Tensor, profile: str = 'gaussian') -> torch.Tensor:
    """Soft recovery-speed preference with the existing 0.5 m/s scale.

    Rational decay retains useful reward contrast at larger observed speeds;
    both profiles favor zero speed and leave diagnostic tolerances unchanged.
    """
    squared_speed = (speed / 0.5) ** 2
    if profile == 'gaussian':
        return torch.exp(-0.5 * squared_speed)
    if profile == 'rational':
        return 1.0 / (1.0 + squared_speed)
    raise ValueError(f'Unknown guard speed profile: {profile!r}')


def update_hand_settling_diagnostic(env, error=None, speed=None):
    """Preserve the old hand-settling statistic without a corrective reward.

    This is intentionally only a diagnostic, not whole-body recovery success.
    The global actor-step key makes repeated component/metric calls idempotent.
    """
    c = command(env)
    if getattr(c, '_hand_settling_step', None) == env.common_step_counter:
        return
    if error is None or speed is None:
        error, speed = guard_errors(env)
    upright = (-c.robot.data.projected_gravity_b[:, 2]).clamp(0, 1)
    after = c.state.elapsed > c.state.requested_ttc + 0.10
    settled = after & (error <= 0.12) & (speed <= 0.30) & (upright >= 0.8)
    # Keep the ordinary tensors mutable during resets outside inference_mode.
    c.settled_duration.copy_(torch.where(settled, c.settled_duration + env.step_dt, 0.0))
    newly_settled = (c.settled_duration >= 0.2 - 1e-6) & torch.isnan(c.first_settled_elapsed)
    c.first_settled_elapsed[newly_settled] = c.state.elapsed[newly_settled]
    c._hand_settling_step = env.common_step_counter


def guard(env):
    """Historical glove-only reward, enabled only by legacy reward mode."""
    c = command(env)
    error, speed = guard_errors(env)
    d = c.robot.data
    torso_quat = d.body_link_quat_w[:, c.torso_index, None, :].expand(-1, 2, -1)
    ref_torso_quat = c.terminal['body_quat_w'][:, c.reference_torso_index, None, :].expand(-1, 2, -1)
    actual_quat = quat_mul(quat_inv(torso_quat), d.site_quat_w[:, c.site_ids])
    desired_quat = quat_mul(quat_inv(ref_torso_quat), c.terminal['glove_quat_w'])
    orientation = (actual_quat * desired_quat).sum(-1).square().mean(-1).clamp(0, 1)
    upright = (-d.projected_gravity_b[:, 2]).clamp(0, 1)
    quality = torch.exp(-0.5 * (error / 0.15) ** 2) * guard_speed_score(speed, c.cfg.guard_speed_profile)
    quality *= torch.exp(-0.5 * d.root_link_lin_vel_b.square().sum(-1)) * upright
    quality *= 0.8 + 0.2 * orientation
    value = 2 * guard_weight(c.state.elapsed, c.state.requested_ttc, c.reference.recovery) * quality / c.guard_z
    update_hand_settling_diagnostic(env, error, speed)
    return c.accumulate('guard', value)


def action_rate(env):
    value = -0.01 * (env.action_manager.action - env.action_manager.prev_action).square().mean(-1)
    return command(env).accumulate('action_rate', value)


def effort(env):
    # Compiled native motor force ranges are the selected controller's real limits.
    c = command(env)
    if not hasattr(c, 'effort_limits'):
        limits = abs(env.sim.mj_model.actuator_forcerange).max(-1)
        c.effort_limits = torch.as_tensor(limits, device=env.device, dtype=torch.float32).clamp_min(1e-6)
    force = env.sim.data.actuator_force[:]
    return c.accumulate('effort', -0.01 * (force / c.effort_limits).square().mean(-1))


def joint_margin(env):
    c = command(env)
    limits = c.robot.data.soft_joint_pos_limits
    violation = (limits[..., 0] - c.robot.data.joint_pos).clamp_min(0) + (c.robot.data.joint_pos - limits[..., 1]).clamp_min(0)
    return c.accumulate('joint_margin', -0.1 * violation.square().mean(-1))


def complete(env):
    c = command(env)
    return c.state.elapsed >= c.horizon - 1e-6


def fallen(env):
    d = command(env).robot.data
    return (d.root_link_pos_w[:, 2] < 0.55) | (-d.projected_gravity_b[:, 2] < 0.2)
