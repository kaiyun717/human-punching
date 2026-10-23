"""Evaluation-only whole-body and native solver telemetry.

Contact forces are read before mjlab's post-step forward call, from the actual
solver's last pre-integration state. No sensor, dynamics or policy state changes.
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp
from mujoco_warp._src.support import contact_force

from .imitation import (
    H2_TRACKED_BODY_NAMES, IMITATION_STDS, IMITATION_WEIGHTS,
    align_reference_poses, whole_body_errors,
)


def solver_observer(env):
    """Installed only in evaluation, after the encounter/timer callback."""
    recorder = getattr(env, "whole_body_metrics", None)
    if recorder is not None:
        recorder.capture_solver()
    return torch.zeros(env.num_envs, device=env.device)


class WholeBodyMetrics:
    def __init__(self, env):
        self.env = env
        self.c = env.command_manager.get_term("punch")
        self.robot = self.c.robot
        self.body_ids = [self.robot.body_names.index(name) for name in H2_TRACKED_BODY_NAMES]
        self.reference_ids = [self.c.reference.body_names.index(name) for name in H2_TRACKED_BODY_NAMES]
        self.foot_ids = [self.robot.body_names.index(f"{side}_ankle_pitch_link") for side in ("left", "right")]
        self.orientation_names = ("pelvis", "torso_link", "head_pitch_link", "head_yaw_link",
                                  "left_ankle_pitch_link", "right_ankle_pitch_link")
        self.orientation_ids = [self.robot.body_names.index(name) for name in self.orientation_names]
        model, device, n = env.sim.mj_model, env.device, env.num_envs
        self.limits = torch.as_tensor(np.asarray([
            model.jnt_range[model.joint("robot/" + name).id] for name in self.robot.joint_names]),
            dtype=torch.float32, device=device)
        foot_geom = np.full(model.ngeom, -1, dtype=np.int64)
        for side, name in enumerate(("left_ankle_pitch_link", "right_ankle_pitch_link")):
            foot_geom[model.geom_bodyid == model.body("robot/" + name).id] = side
        self.foot_geom = torch.as_tensor(foot_geom, device=device)
        self.ground_geom = torch.as_tensor(np.asarray(model.geom_type) == 0, device=device)
        if not bool(self.ground_geom.any()) or not all(np.any(foot_geom == side) for side in (0, 1)):
            raise RuntimeError("Native foot/ground geometry map unavailable")
        robot_mass = sum(float(model.body_mass[i]) for i in range(model.nbody)
                         if model.body(i).name.startswith("robot/"))
        self.weight = robot_mass * float(np.linalg.norm(model.opt.gravity))
        self.capacity = env.sim.wp_data.contact.geom.shape[0]
        self.contact_ids = torch.arange(self.capacity, dtype=torch.int32, device=device)
        self.contact_ids_wp = wp.from_torch(self.contact_ids)
        self.forces_wp = wp.zeros(self.capacity, dtype=wp.spatial_vector, device=env.sim.wp_device)
        self.forces = wp.to_torch(self.forces_wp)
        self.active = torch.zeros(n, dtype=torch.bool, device=device)
        self.max_command_excess = torch.zeros(n, len(self.robot.joint_names), device=device)
        self.max_abs_command = torch.zeros_like(self.max_command_excess)
        self.command_excess_steps = torch.zeros_like(self.max_command_excess)
        self.actor_samples = torch.zeros(n, dtype=torch.long, device=device)
        self.last_load = torch.zeros(n, 2, 3, device=device)
        self.last_contacts = torch.zeros(n, 2, device=device)
        self.last_solver_elapsed = torch.full((n,), -1., device=device)
        self.hold_count = torch.zeros(n, dtype=torch.long, device=device)
        self.hold_first_solver_elapsed = torch.full((n,), -1., device=device)
        self.hold_last_solver_elapsed = torch.full((n,), -1., device=device)
        self.hold_load_sum = torch.zeros_like(self.last_load)
        self.hold_load_min = torch.full_like(self.last_load, float("inf"))
        self.hold_load_max = torch.full_like(self.last_load, -float("inf"))
        self.hold_positive_load_count = torch.zeros(n, 2, device=device)
        self.hold_contact_count = torch.zeros(n, 2, device=device)
        self.hold_support_quality_sum = torch.zeros(n, device=device)

    def begin_batch(self, count):
        self.active.zero_()
        self.active[:count] = True
        for value in (self.max_command_excess, self.max_abs_command, self.command_excess_steps,
                      self.actor_samples, self.last_load, self.last_contacts, self.hold_count,
                      self.hold_load_sum, self.hold_positive_load_count, self.hold_contact_count,
                      self.hold_support_quality_sum):
            value.zero_()
        self.last_solver_elapsed.fill_(-1.)
        self.hold_first_solver_elapsed.fill_(-1.)
        self.hold_last_solver_elapsed.fill_(-1.)
        self.hold_load_min.fill_(float("inf"))
        self.hold_load_max.fill_(-float("inf"))

    def capture_actor(self):
        targets = self.robot.data.joint_pos_target
        excess = ((self.limits[:, 0] - targets).clamp_min(0)
                  + (targets - self.limits[:, 1]).clamp_min(0))
        active = self.active[:, None]
        self.max_command_excess.copy_(torch.where(active, torch.maximum(self.max_command_excess, excess),
                                                 self.max_command_excess))
        self.max_abs_command.copy_(torch.where(active, torch.maximum(self.max_abs_command, targets.abs()),
                                              self.max_abs_command))
        self.command_excess_steps.add_(active & (excess > 0))
        self.actor_samples.add_(self.active)

    def capture_solver(self):
        """Reduce native forces on GPU; no host copy or new forward/solve."""
        env, native = self.env, self.env.sim.wp_data
        with wp.ScopedDevice(env.sim.wp_device):
            contact_force(env.sim.wp_model, native, self.contact_ids_wp, True, self.forces_wp)
        valid = self.contact_ids < wp.to_torch(native.nacon)[0]
        geoms = wp.to_torch(native.contact.geom).long().clamp(0, len(self.foot_geom) - 1)
        worlds = wp.to_torch(native.contact.worldid).long().clamp(0, env.num_envs - 1)
        foot0, foot1 = self.foot_geom[geoms[:, 0]], self.foot_geom[geoms[:, 1]]
        forward = self.ground_geom[geoms[:, 0]] & (foot1 >= 0)
        backward = self.ground_geom[geoms[:, 1]] & (foot0 >= 0)
        selected = valid & (forward | backward)
        side = torch.where(forward, foot1, foot0).clamp_min(0)
        indices = worlds * 2 + side
        sign = torch.where(forward, 1., -1.)
        force = torch.where(selected[:, None], self.forces[:, :3] * sign[:, None], 0.)
        loads = torch.zeros(env.num_envs * 2, 3, device=env.device).index_add_(0, indices, force).reshape(-1, 2, 3)
        counts = torch.zeros(env.num_envs * 2, device=env.device).index_add_(0, indices, selected.float()).reshape(-1, 2)
        self.last_load.copy_(torch.where(self.active[:, None, None], loads, self.last_load))
        self.last_contacts.copy_(torch.where(self.active[:, None], counts, self.last_contacts))
        # The preceding encounter callback advances elapsed after integration.
        elapsed = self.c.state.elapsed - env.physics_dt
        self.last_solver_elapsed.copy_(torch.where(self.active, elapsed, self.last_solver_elapsed))
        hold = self.active & (elapsed >= self.c.state.requested_ttc + self.c.reference.recovery - 1e-6)
        self.hold_first_solver_elapsed.copy_(torch.where(hold & (self.hold_count == 0), elapsed, self.hold_first_solver_elapsed))
        self.hold_last_solver_elapsed.copy_(torch.where(hold, elapsed, self.hold_last_solver_elapsed))
        self.hold_count.add_(hold)
        self.hold_load_sum.add_(torch.where(hold[:, None, None], loads, 0.))
        self.hold_load_min.copy_(torch.where(hold[:, None, None], torch.minimum(self.hold_load_min, loads), self.hold_load_min))
        self.hold_load_max.copy_(torch.where(hold[:, None, None], torch.maximum(self.hold_load_max, loads), self.hold_load_max))
        self.hold_positive_load_count.add_(hold[:, None] & (loads[..., 2] > 0))
        self.hold_contact_count.add_(torch.where(hold[:, None], counts, 0.))
        speed2 = self.robot.data.body_link_lin_vel_w[:, self.foot_ids].square().sum(-1).mean(-1)
        total_load_score = (loads[..., 2].clamp_min(0).sum(-1) / self.weight).clamp(0, 1)
        quality = total_load_score * torch.exp(-speed2 / .25 ** 2)
        self.hold_support_quality_sum.add_(torch.where(hold, quality, 0.))

    def _tracking(self):
        d, ref = self.robot.data, self.c.ref
        arrays = (d.body_link_pos_w[:, self.body_ids], d.body_link_quat_w[:, self.body_ids],
                  d.body_link_lin_vel_w[:, self.body_ids], d.body_link_ang_vel_w[:, self.body_ids],
                  ref['body_pos_w'][:, self.reference_ids] + self.c.reference_origin_w[:, None, :],
                  ref['body_quat_w'][:, self.reference_ids],
                  ref['body_lin_vel_w'][:, self.reference_ids], ref['body_ang_vel_w'][:, self.reference_ids])
        errors = whole_body_errors(*arrays, imitation_frame=self.c.cfg.imitation_frame)
        scores = {name: torch.exp(-error / IMITATION_STDS[name] ** 2) for name, error in errors.items()}
        return errors, scores

    def terminal(self, index):
        from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude
        errors, scores = self._tracking()
        d = self.robot.data
        body_position = d.body_link_pos_w[:, self.body_ids]
        body_quaternion = d.body_link_quat_w[:, self.body_ids]
        reference_position = (self.c.ref['body_pos_w'][:, self.reference_ids]
                              + self.c.reference_origin_w[:, None, :])
        reference_quaternion = self.c.ref['body_quat_w'][:, self.reference_ids]
        if self.c.cfg.imitation_frame == 'relative':
            aligned_position, aligned_quaternion = align_reference_poses(
                body_position, body_quaternion, reference_position, reference_quaternion)
        else:
            aligned_position, aligned_quaternion = reference_position, reference_quaternion
        local_up = torch.zeros(self.env.num_envs, len(self.orientation_ids), 3, device=self.env.device)
        local_up[..., 2] = 1
        upright = quat_apply(d.body_link_quat_w[:, self.orientation_ids], local_up)[index, :, 2].clamp(-1, 1)
        count = int(self.hold_count[index])
        targets = d.joint_pos_target[index]
        excess = ((self.limits[:, 0] - targets).clamp_min(0)
                  + (targets - self.limits[:, 1]).clamp_min(0))
        weighted_score = sum(IMITATION_WEIGHTS[name] * scores[name][index] for name in scores) / sum(IMITATION_WEIGHTS.values())
        return {
            "reward_mode": getattr(self.c.cfg, "reward_mode", "legacy"),
            "imitation_frame": self.c.cfg.imitation_frame,
            "terminal_whole_body_score": float(weighted_score),
            "terminal_tracking_squared_errors": {name: float(value[index]) for name, value in errors.items()},
            "terminal_tracking_scores": {name: float(value[index]) for name, value in scores.items()},
            "terminal_body_position_error_m": (body_position - aligned_position).norm(dim=-1)[index].cpu().tolist(),
            "terminal_body_orientation_error_rad": quat_error_magnitude(body_quaternion, aligned_quaternion)[index].cpu().tolist(),
            "terminal_body_tilt_degrees": dict(zip(self.orientation_names, torch.rad2deg(torch.acos(upright)).cpu().tolist())),
            "terminal_foot_position_world_m": (d.body_link_pos_w[index, self.foot_ids] - self.env.scene.env_origins[index]).cpu().tolist(),
            "terminal_foot_linear_velocity_world_m_s": d.body_link_lin_vel_w[index, self.foot_ids].cpu().tolist(),
            "terminal_foot_angular_velocity_world_rad_s": d.body_link_ang_vel_w[index, self.foot_ids].cpu().tolist(),
            "terminal_joint_position_rad": d.joint_pos[index].cpu().tolist(),
            "terminal_joint_velocity_rad_s": d.joint_vel[index].cpu().tolist(),
            "terminal_commanded_target_rad": targets.cpu().tolist(),
            "terminal_command_excess_rad": excess.cpu().tolist(),
            "maximum_command_excess_rad_by_joint": self.max_command_excess[index].cpu().tolist(),
            "maximum_command_excess_rad": float(self.max_command_excess[index].max()),
            "maximum_abs_commanded_target_rad_by_joint": self.max_abs_command[index].cpu().tolist(),
            "command_excess_actor_samples_by_joint": self.command_excess_steps[index].cpu().tolist(),
            "command_actor_samples": int(self.actor_samples[index]),
            "terminal_solver_elapsed_s": float(self.last_solver_elapsed[index]),
            "terminal_foot_ground_force_world_N": self.last_load[index].cpu().tolist(),
            "terminal_foot_ground_contact_count": self.last_contacts[index].int().cpu().tolist(),
            "terminal_hold_solver_samples": count,
            "terminal_hold_first_solver_elapsed_s": float(self.hold_first_solver_elapsed[index]) if count else None,
            "terminal_hold_last_solver_elapsed_s": float(self.hold_last_solver_elapsed[index]) if count else None,
            "terminal_hold_duration_sampled_s": count * self.env.physics_dt,
            "terminal_hold_foot_force_world_mean_N": (self.hold_load_sum[index] / count).cpu().tolist() if count else None,
            "terminal_hold_foot_force_world_min_N": self.hold_load_min[index].cpu().tolist() if count else None,
            "terminal_hold_foot_force_world_max_N": self.hold_load_max[index].cpu().tolist() if count else None,
            "terminal_hold_foot_positive_vertical_load_duration_s": (self.hold_positive_load_count[index] * self.env.physics_dt).cpu().tolist(),
            "terminal_hold_foot_ground_contact_mean_count": (self.hold_contact_count[index] / count).cpu().tolist() if count else None,
            "terminal_hold_support_quality": float(self.hold_support_quality_sum[index] / count) if count else 0.,
        }

    def trace_arrays(self, count):
        d, ref = self.robot.data, self.c.ref
        qpos = self.env.sim.data.qpos[:count].clone()
        qpos[:, :3] -= self.env.scene.env_origins[:count]
        arrays = {
            'qpos': qpos, 'qvel': self.env.sim.data.qvel[:count],
            'elapsed_s': self.c.state.elapsed[:count], 'source_time_s': self.c.source_time[:count],
            'raw_action': self.env.action_manager.action[:count],
            'joint_pos': d.joint_pos[:count], 'joint_vel': d.joint_vel[:count],
            'commanded_target_rad': d.joint_pos_target[:count],
            'body_pos_w': d.body_link_pos_w[:count, self.body_ids] - self.env.scene.env_origins[:count, None, :],
            'body_quat_w': d.body_link_quat_w[:count, self.body_ids],
            'body_lin_vel_w': d.body_link_lin_vel_w[:count, self.body_ids],
            'body_ang_vel_w': d.body_link_ang_vel_w[:count, self.body_ids],
            'solver_elapsed_s': self.last_solver_elapsed[:count],
            'foot_ground_force_world_N': self.last_load[:count],
            'foot_ground_contact_count': self.last_contacts[:count],
        }
        for key in ('body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'):
            arrays['reference_' + key] = ref[key][:count, self.reference_ids]
        return {name: value.detach().cpu().numpy().copy() for name, value in arrays.items()}

    def metadata(self):
        return {
            "imitation_frame": self.c.cfg.imitation_frame,
            "reference_registration": "stored source trajectory plus frozen scene origin; identity rotation; relative mode additionally aligns body poses to current pelvis XY/yaw",
            "joint_names": list(self.robot.joint_names), "body_names": list(H2_TRACKED_BODY_NAMES),
            "foot_order": ["left", "right"], "hard_joint_limits_rad": self.limits.cpu().tolist(),
            "robot_weight_N": self.weight, "physics_dt_s": self.env.physics_dt,
            "contact_semantics": "Actual native efc contact forces in world frame; force on foot, decoded before post-step forward. Solver time is elapsed_end minus physics_dt. No contact recomputation or prescribed force sharing.",
            "hold_window": "solver_elapsed >= requested_ttc + reference.recovery, through termination; zero samples if fallen before hold",
            "support_quality": "mean_hold(min(sum_positive_foot_vertical_force/robot_weight,1) * exp(-mean_foot_linear_speed_squared/0.25^2)); engineering retention diagnostic, no acceptance threshold",
            "positive_load_duration": "strictly positive native vertical load, without a minimum-load threshold",
            "trajectory_semantics": "qpos/kinematics at actor endpoint; support arrays are the preceding solver state; initial solver_elapsed=-1 means no solver sample yet",
        }
