"""Human task observations/objectives; acceptance is an independent trace audit."""

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse


def punch_condition(env):
    cmd = env.command_manager.get_term("motion")
    clip = cmd.trajectory_ids
    family = torch.nn.functional.one_hot(cmd.family_ids[clip], num_classes=4).float()
    target = quat_apply_inverse(cmd.robot_anchor_quat_w, cmd.target_w - cmd.robot_anchor_pos_w)
    launch_elapsed = cmd.elapsed - cmd.launch_times[clip]
    timing = torch.stack((cmd.ttcs[clip], launch_elapsed, cmd.ttcs[clip] - launch_elapsed), -1)
    return torch.cat((family, target, timing, cmd.g1_context[clip]), dim=-1)


def future_joint_reference(env):
    cmd = env.command_manager.get_term("motion")
    return torch.cat([cmd.sample_master("qpos", offset)[:, 7:] - cmd.robot_joint_pos
                      for offset in (12, 24)], dim=-1)


def cycle_finished(env):
    cmd = env.command_manager.get_term("motion")
    end = (cmd.master_lengths[cmd.trajectory_ids] - 1) * .005 + cmd.cfg.final_guard_hold_s
    return cmd.elapsed >= end


def timed_target_reward(env):
    """Targeting shaping aligned with mjlab's pre-integration geometry timestamp.

    The accepted mjlab step computes rewards before its final forward refresh.
    The glove positions therefore describe the beginning of the last 5 ms
    interval, while the command counter describes its end. Align the timing
    kernel to that preceding geometry sample. This is shaping only; independent
    acceptance still needs fresh geometry and first-event measurements at every
    physics sample.
    """
    cmd = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    site_ids, _ = robot.find_sites(("LeftHand_center", "RightHand_center"), preserve_order=True)
    points = robot.data.site_pos_w[:, site_ids]
    family = cmd.family_ids[cmd.trajectory_ids]
    hand = ((family == 1) | (family == 3)).long()
    glove = points[torch.arange(env.num_envs, device=env.device), hand]
    error_sq = (glove - cmd.target_w).square().sum(-1)
    geometry_elapsed = torch.clamp(cmd.elapsed - env.physics_dt, min=0.)
    time_error = geometry_elapsed - cmd.launch_times[cmd.trajectory_ids] - cmd.ttcs[cmd.trajectory_ids]
    return torch.exp(-error_sq / .14**2) * torch.exp(-time_error.square() / .04**2)
