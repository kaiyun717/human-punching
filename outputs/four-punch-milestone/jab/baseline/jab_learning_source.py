"""One-clip, residual-PD PPO imitation pilot on pinned mjlab/MuJoCo Warp.

The policy observes the elapsed reference phase, proprioception, and committed
reference samples. Physics has a free root and receives only bounded joint motor
torques. State writes occur at episode resets, never during an evaluation cycle.
This deliberately small trainer has no dependency on a second robotics framework.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from human_punching.scalar_control import (
    actuator_torque_ranges,
    configure_implicit_pd,
    export_runtime_model,
    native_pd_drive,
    scalar_actuator_layout,
)


def tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def maximum_joint_limit_excursion(joint_position, low, high):
    """Worst physical soft-stop excursion per world, in radians."""
    return torch.maximum(low-joint_position, joint_position-high).clamp_min(0).amax(-1)


def points_in_body_frame(points, origin, rotation):
    """Express batched world-space points in a body's local coordinate frame."""
    return torch.einsum('bji,bhj->bhi', rotation, points-origin[:, None])


def interpolate_reference(times, values, query):
    """Linear interpolation with clamped endpoints; supports arbitrary batches."""
    clipped = query.clamp(times[0], times[-1])
    upper = torch.searchsorted(times, clipped.contiguous(), right=True).clamp(1, len(times)-1)
    lower = upper-1
    alpha = (clipped-times[lower])/(times[upper]-times[lower])
    return values[lower] + alpha.reshape(*alpha.shape, *([1]*(values.ndim-1))) * (
        values[upper]-values[lower])


def quat_up(quaternion):
    """World z component of local z, for MuJoCo wxyz quaternions."""
    return 1-2*(quaternion[..., 1].square()+quaternion[..., 2].square())


def quaternion_error(current, desired):
    """Shortest local rotation vector from current to desired, wxyz convention."""
    aw, av = current[..., :1], -current[..., 1:]
    bw, bv = desired[..., :1], desired[..., 1:]
    w = aw*bw-(av*bv).sum(-1, keepdim=True)
    vector = aw*bv+bw*av+torch.linalg.cross(av, bv, dim=-1)
    sign = torch.where(w < 0, -1., 1.)
    w, vector = w*sign, vector*sign
    magnitude = vector.norm(dim=-1, keepdim=True)
    angle = 2*torch.atan2(magnitude, w.clamp_min(1e-8))
    return vector*torch.where(magnitude > 1e-7, angle/magnitude.clamp_min(1e-8), 2.)


def actuator_layout(model):
    """Reject incompatible rigs rather than silently actuating root coordinates."""
    if np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE):
        result = scalar_actuator_layout(model)
        actuator_torque_ranges(model)
        return result
    ids = model.actuator_trnid[:, 0].astype(int)
    if not np.all(model.jnt_type[ids] == mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError('The imitation pilot requires scalar hinge motor actuators.')
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('Each hinge must have exactly one motor.')
    if not np.all(model.actuator_ctrllimited):
        raise ValueError('Every motor must have an explicit bounded ctrlrange.')
    if not np.allclose(model.actuator_gear[:, 0], 1):
        raise ValueError('Expected gear=1 so ctrl is an auditable joint torque.')
    qadr, vadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    if np.any(qadr < 7) or np.any(vadr < 6):
        raise ValueError('The free root must be unactuated.')
    return ids, qadr, vadr


def default_gains(model, ids):
    kp, kd = [], []
    for joint in ids:
        name = model.joint(int(joint)).name.lower()
        if any(part in name for part in ('upleg', 'thigh', 'leg', 'shin')):
            p, d = 300., 22.
        elif any(part in name for part in ('foot', 'ankle')):
            p, d = 180., 14.
        elif any(part in name for part in ('spine', 'chest', 'torso')):
            p, d = 180., 16.
        elif 'forearm' in name:
            p, d = 100., 8.
        elif 'hand' in name:
            p, d = 35., 3.
        elif 'arm' in name:
            p, d = 120., 10.
        else:
            p, d = 45., 4.
        kp.append(p)
        kd.append(d)
    return np.asarray(kp), np.asarray(kd)


def load_reference(model, path, speed=1.):
    source = np.load(path, allow_pickle=False)
    times = np.asarray(source['t'] if 't' in source else source['time_s'], dtype=float)
    qpos = np.asarray(source['qpos'], dtype=float)
    qvel = np.asarray(source['qvel'], dtype=float)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Reference times must be strictly increasing.')
    if qpos.shape != (len(times), model.nq) or qvel.shape != (len(times), model.nv):
        raise ValueError('Reference dimensions do not match model.')
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError('Reference contains nonfinite state.')
    if speed <= 0:
        raise ValueError('Reference playback speed must be positive.')
    times = (times-times[0])/speed
    qvel = qvel*speed
    # Align quaternion signs before nlerp; scalar joint angles need no conversion.
    qpos = qpos.copy()
    for frame in range(1, len(qpos)):
        if np.dot(qpos[frame-1, 3:7], qpos[frame, 3:7]) < 0:
            qpos[frame, 3:7] *= -1
    data = mujoco.MjData(model)
    body, gloves, glove_rotation = [], [], []
    hands = [model.site(f'{side}Hand_center').id for side in ('Left', 'Right')]
    chest = model.body('Chest').id
    guard_glove_chest_pos = None
    for pose, velocity in zip(qpos, qvel, strict=True):
        data.qpos[:], data.qvel[:] = pose, velocity
        mujoco.mj_forward(model, data)
        body.append(data.xpos.copy())
        gloves.append(data.site_xpos[hands].copy())
        glove_rotation.append(data.site_xmat[hands].copy().reshape(2, 9))
        if guard_glove_chest_pos is None:
            guard_glove_chest_pos = (gloves[-1]-data.xpos[chest])@data.xmat[chest].reshape(3, 3)
    return dict(t=times, qpos=qpos, qvel=qvel, body_pos=np.asarray(body),
                glove_pos=np.asarray(gloves), glove_rotation=np.asarray(glove_rotation),
                guard_glove_chest_pos=guard_glove_chest_pos,
                guard_hold_start=float(source['guard_hold_start'])/speed
                if 'guard_hold_start' in source else times[-1],
                active_hand_index=int(source['active_hand_index'])
                if 'active_hand_index' in source else 0,
                head_center=source['head_center'] if 'head_center' in source else None)


def support_feedforward(model, reference, vadr):
    """Offline gravity/velocity compensation using assumed double-foot support.

    These contact forces are an inverse-statics calculation only. They are never
    applied to the simulator. Only its bounded joint-torque result is used.
    Actual contact/balance is left to physics and the learned feedback policy.
    """
    data = mujoco.MjData(model)
    sites = [i for i in range(model.nsite)
             if 'sole' in (model.site(i).name or '').lower()]
    if not sites:
        return np.zeros((len(reference['t']), len(vadr)))
    torque = []
    for pose, velocity in zip(reference['qpos'], reference['qvel'], strict=True):
        data.qpos[:], data.qvel[:] = pose, velocity
        mujoco.mj_forward(model, data)
        jacobians = []
        for site in sites:
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac, None, site)
            jacobians.append(jac)
        jac = np.concatenate(jacobians)
        forces = np.linalg.lstsq(jac[:, :6].T, data.qfrc_bias[:6], rcond=1e-5)[0]
        forces = forces.reshape(-1, 3)
        forces[:, 2] = np.maximum(forces[:, 2], 0)
        forces[:, :2] = np.clip(forces[:, :2], -.8*forces[:, 2:3], .8*forces[:, 2:3])
        torque.append((data.qfrc_bias-jac.T@forces.ravel())[vadr])
    limits = actuator_torque_ranges(model)
    return np.clip(torque, limits[:, 0], limits[:, 1])


def support_balance_map(model, reference, vadr):
    """Reference linearization mapping desired root acceleration to joint torque.

    Contact support is assumed, not imposed. Matrix rows are actuated hinge DOFs;
    there are deliberately no output rows for the free-root force coordinates.
    """
    sites = [i for i in range(model.nsite)
             if 'sole' in (model.site(i).name or '').lower()]
    if not sites:
        raise ValueError('Balance feedback requires planted sole reference sites.')
    data = mujoco.MjData(model)
    mass = np.zeros((model.nv, model.nv))
    matrices = []
    for pose, velocity in zip(reference['qpos'], reference['qvel'], strict=True):
        data.qpos[:], data.qvel[:] = pose, velocity
        mujoco.mj_forward(model, data)
        jacobians = []
        for site in sites:
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac, None, site)
            jacobians.append(jac)
        jac = np.concatenate(jacobians)
        mujoco.mj_fullM(model, data, mass)
        matrix = -jac[:, vadr].T@np.linalg.pinv(jac[:, :6].T, rcond=1e-5)@mass[:6, :6]
        matrices.append(matrix)
    return np.asarray(matrices)


class ActorCritic(nn.Module):
    def __init__(self, observations, actions):
        super().__init__()
        def mlp(outputs):
            return nn.Sequential(nn.Linear(observations, 256), nn.ELU(),
                                 nn.Linear(256, 256), nn.ELU(), nn.Linear(256, outputs))
        self.actor = mlp(actions)
        self.critic = mlp(1)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((actions,), -2.0))

    def distribution(self, observations):
        return Normal(self.actor(observations), self.log_std.clamp(-4, 0).exp())


class JabEnv:
    def __init__(self, args):
        import warp as wp
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
        self.args, self.device, self.n = args, 'cuda:0', args.num_envs
        self.model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
        ids, qadr, vadr = actuator_layout(self.model)
        self.ids, self.qadr, self.vadr = ids, qadr, vadr
        self.dt = args.timestep
        self.decimation = round(args.policy_dt/self.dt)
        if self.decimation < 1 or abs(self.decimation*self.dt-args.policy_dt) > 1e-8:
            raise ValueError('policy-dt must be an integer multiple of timestep.')
        self.model.opt.timestep = self.dt
        self.ref_np = load_reference(self.model, args.reference, args.speed)
        self.active_hand_index = self.ref_np['active_hand_index']
        self.duration = float(self.ref_np['t'][-1])+args.hold
        self.reference = {k: tensor(v, self.device) for k, v in self.ref_np.items()
                          if v is not None}
        self.reference['torque'] = tensor(
            support_feedforward(self.model, self.ref_np, vadr), self.device)
        if args.balance_feedback:
            self.reference['balance_map'] = tensor(
                support_balance_map(self.model, self.ref_np, vadr), self.device)
        kp, kd = default_gains(self.model, ids)
        if args.body_metadata:
            meta = json.loads(args.body_metadata.read_text())
            if 'kp' in meta and 'kd' in meta:
                kp, kd = np.asarray(meta['kp']), np.asarray(meta['kd'])
            elif 'joints' in meta:
                rows = {row['name']: row for row in meta['joints']}
                kp = np.asarray([rows[self.model.joint(int(j)).name]['kp'] for j in ids])
                kd = np.asarray([rows[self.model.joint(int(j)).name]['kd'] for j in ids])
            if kp.shape != (self.model.nu,) or kd.shape != kp.shape:
                raise ValueError('Metadata kp/kd must follow actuator order.')
        kp, kd = kp*args.gain_scale, kd*np.sqrt(args.gain_scale)
        self.controller_mode = args.controller
        if self.controller_mode == 'implicit_pd':
            configure_implicit_pd(self.model, kp, kd)
        elif np.any(self.model.actuator_biastype != mujoco.mjtBias.mjBIAS_NONE):
            raise ValueError('Explicit torque PD requires bias-free motor actuators.')
        self.kp, self.kd = tensor(kp, self.device), tensor(kd, self.device)
        self.torque_limits = actuator_torque_ranges(self.model)
        self.low = tensor(self.torque_limits[:, 0], self.device)
        self.high = tensor(self.torque_limits[:, 1], self.device)
        self.joint_low = tensor(self.model.jnt_range[ids, 0], self.device)
        self.joint_high = tensor(self.model.jnt_range[ids, 1], self.device)
        wp.init()
        wp.config.kernel_cache_dir = str(Path('.cache/physics/warp').resolve())
        cfg = SimulationCfg(nconmax=96, njmax=256,
                            mujoco=MujocoCfg(timestep=self.dt, cone='elliptic',
                                            iterations=30, ls_iterations=15, tolerance=1e-7))
        self.sim = Simulation(self.n, cfg, self.model, self.device)
        with wp.ScopedDevice(self.sim.wp_device):
            self.stream = torch.cuda.ExternalStream(wp.get_stream().cuda_stream)
        self.stream.wait_stream(torch.cuda.current_stream())
        self.clock = torch.zeros(self.n, device=self.device)
        self.last_action = torch.zeros((self.n, self.model.nu), device=self.device)
        self.hands = [self.model.site(f'{side}Hand_center').id for side in ('Left', 'Right')]
        self.foot_geoms = [self.model.geom(f'physical/{side}Foot').id for side in ('Left', 'Right')]
        self.floor = self.model.geom('floor').id
        self.hips = self.model.body('Hips').id
        self.chest = self.model.body('Chest').id
        self.completed = 0
        self.fallen = 0

    def schema(self):
        return dict(version=1, action='scalar PD angle residual, clipped to [-1,1]',
                    action_scale_rad=self.args.action_scale,
                    observations=['root height+quaternion', 'root velocity*.2', 'joint angles',
                                  'joint velocities*.1', 'reference root translation error*5',
                                  'reference root quaternion', 'joint reference error',
                                  'joint velocity reference error*.1', 'phase,sin,cos',
                                  'previous raw action', 'future .1s root/joint error',
                                  'future .25s root/joint error', 'both glove position errors*5'],
                    observation_clip=20, phase_clock='elapsed from reset/reference sample',
                    joint_names=[self.model.joint(int(j)).name for j in self.ids],
                    qpos_addresses=self.qadr.tolist(), dof_addresses=self.vadr.tolist(),
                    kp=self.kp.cpu().tolist(), kd=self.kd.cpu().tolist(),
                    torque_limits=self.torque_limits.tolist(),
                    controller_mode=self.controller_mode,
                    control_input='PD drive; native affine feedback; output force clamp'
                    if self.controller_mode == 'implicit_pd' else 'bounded motor torque',
                    refresh_kinematics=self.args.refresh_kinematics,
                    physics_dt=self.dt, actor_dt=self.args.policy_dt,
                    speed=self.args.speed, guard_hold_s=self.args.hold,
                    feedforward='none' if self.args.no_feedforward else 'offline inverse statics joint torques',
                    balance_feedback=self.args.balance_feedback,
                    balance_feedback_gains={'kp': 12., 'kd': 5., 'acceleration_clip': 3.},
                    model_sha256=sha256(self.args.model), reference_sha256=sha256(self.args.reference))

    def support_contact(self):
        """Positive actual foot-floor normal forces from elliptic Warp constraints."""
        contact = self.sim.data.contact
        world = contact.worldid[:].long()
        address = contact.efc_address[:, 0].long()
        valid = ((torch.arange(len(world), device=self.device) < self.sim.data.nacon[0])
                 & (world >= 0) & (world < self.n) & (address >= 0))
        force = self.sim.data.efc.force[world.clamp(0, self.n-1), address.clamp(
            0, self.sim.data.efc.force.shape[1]-1)]
        geoms = contact.geom[:]
        valid &= (force > 1e-6) & ((geoms == self.floor).any(-1))
        support = torch.zeros((self.n, 2), device=self.device)
        for side, geom in enumerate(self.foot_geoms):
            values = (valid & (geoms == geom).any(-1)).float()
            support[:, side].scatter_add_(0, world.clamp(0, self.n-1), values)
        return support > 0

    def ref(self, key, offset=0.):
        result = interpolate_reference(self.reference['t'], self.reference[key], self.clock+offset)
        if key == 'qpos':
            result[:, 3:7] /= result[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return result

    def reset(self, env_ids=None, *, random_phase=True, perturb=True):
        if env_ids is None:
            env_ids = torch.arange(self.n, device=self.device)
        self.sim.reset(env_ids)
        self.clock[env_ids] = 0
        if random_phase:
            starts = torch.rand(len(env_ids), device=self.device)
            # Half of episodes begin at guard; the rest sample reference phases.
            self.clock[env_ids] = torch.where(starts < .5, 0.,
                                              (starts-.5)*1.6*self.reference['t'][-1])
        pose, velocity = self.ref('qpos')[env_ids], self.ref('qvel')[env_ids]
        if perturb:
            pose[:, self.qadr] += .005*torch.randn((len(env_ids), self.model.nu), device=self.device)
            pose[:, self.qadr] = pose[:, self.qadr].clamp(self.joint_low, self.joint_high)
            velocity += .02*torch.randn_like(velocity)
        self.sim.data.qpos[env_ids] = pose
        self.sim.data.qvel[env_ids] = velocity
        self.last_action[env_ids] = 0
        self.sim.forward()
        return self.observe()

    def observe(self):
        q, v = self.sim.data.qpos[:], self.sim.data.qvel[:]
        desired = self.ref('qpos')
        desired_v = self.ref('qvel')
        phase = self.clock/self.duration
        parts = [q[:, 2:7], v[:, :6]*.2, q[:, self.qadr], v[:, self.vadr]*.1,
                 (desired[:, :3]-q[:, :3])*5, desired[:, 3:7],
                 desired[:, self.qadr]-q[:, self.qadr],
                 (desired_v[:, self.vadr]-v[:, self.vadr])*.1,
                 torch.stack((phase, torch.sin(phase*2*np.pi), torch.cos(phase*2*np.pi)), -1),
                 self.last_action]
        for offset in (.1, .25):
            future = self.ref('qpos', offset)
            parts.extend(((future[:, :3]-q[:, :3])*5,
                          future[:, self.qadr]-q[:, self.qadr]))
        parts.append((self.ref('glove_pos')-self.sim.data.site_xpos[:, self.hands]).flatten(1)*5)
        return torch.nan_to_num(torch.cat(parts, -1)).clamp(-20, 20)

    def control(self, action):
        desired, desired_v = self.ref('qpos'), self.ref('qvel')
        target = (desired[:, self.qadr]+action.clamp(-1, 1)*self.args.action_scale).clamp(
            self.joint_low, self.joint_high)
        q, v = self.sim.data.qpos[:, self.qadr], self.sim.data.qvel[:, self.vadr]
        # Derivative feedback is joint damping. No free-root wrench is generated.
        if self.controller_mode == 'implicit_pd':
            torque = native_pd_drive(target, desired_v[:, self.vadr], self.kp, self.kd)
        else:
            torque = self.kp*(target-q)+self.kd*(desired_v[:, self.vadr]-v)
        if not self.args.no_feedforward:
            torque += self.ref('torque')
        if self.args.balance_feedback:
            root_error = torch.cat((desired[:, :3]-self.sim.data.qpos[:, :3],
                                    quaternion_error(self.sim.data.qpos[:, 3:7], desired[:, 3:7])), -1)
            acceleration = (12*root_error-5*(self.sim.data.qvel[:, :6]-desired_v[:, :6])).clamp(-3, 3)
            torque += torch.einsum('bij,bj->bi', self.ref('balance_map'), acceleration)
        # Native force limits apply after position/velocity feedback. Clipping the
        # drive here would destroy the intended force law when kp*q is large.
        self.sim.data.ctrl[:] = torque if self.controller_mode == 'implicit_pd' else torque.clamp(
            self.low, self.high)

    def advance_substep(self, action):
        self.control(action)
        self.sim.step()
        self.clock += self.dt

    def reward(self, action, peak_limit_excursion=None):
        q, v = self.sim.data.qpos[:], self.sim.data.qvel[:]
        desired, desired_v = self.ref('qpos'), self.ref('qvel')
        joint_error = (q[:, self.qadr]-desired[:, self.qadr]).square().mean(-1)
        velocity_error = (v[:, self.vadr]-desired_v[:, self.vadr]).square().mean(-1)
        root_error = (q[:, :3]-desired[:, :3]).square().sum(-1)
        root_rotation = 1-(q[:, 3:7]*desired[:, 3:7]).sum(-1).square().clamp(0, 1)
        body_error = (self.sim.data.xpos[:, 1:]-self.ref('body_pos')[:, 1:]).square().sum(-1).mean(-1)
        glove_errors = (self.sim.data.site_xpos[:, self.hands]-self.ref('glove_pos')).square().sum(-1)
        glove_error = glove_errors.mean(-1)
        rotation = self.sim.data.site_xmat[:, self.hands].reshape(self.n, 2, 9)
        glove_rotation = (rotation-self.ref('glove_rotation')).square().mean((-1, -2))
        if self.args.reward_profile == 'precision':
            reward = (.15*torch.exp(-12*joint_error)+.05*torch.exp(-.05*velocity_error)
                      +.10*torch.exp(-60*body_error)+.20*torch.exp(-60*root_error)
                      +.10*torch.exp(-12*root_rotation)
                      +.35*torch.exp(-self.args.glove_sharpness*glove_error)
                      +.05*torch.exp(-6*glove_rotation))
        else:
            reward = (.25*torch.exp(-10*joint_error)+.10*torch.exp(-.05*velocity_error)
                      +.10*torch.exp(-40*body_error)+.25*torch.exp(-80*root_error)
                      +.10*torch.exp(-10*root_rotation)+.15*torch.exp(-100*glove_error)
                      +.05*torch.exp(-5*glove_rotation))
        if self.args.attack_tracking_weight:
            start, end = np.asarray(self.args.attack_window)/self.args.speed
            attack = ((self.clock >= start) & (self.clock <= end)).float()
            weight = self.args.attack_tracking_weight*attack
            # Rational shaping retains a useful distance signal for initially
            # missed hooks, unlike a very sharp exponential near zero reward.
            active_tracking = 1/(1+glove_errors[:, self.active_hand_index]
                                 /self.args.attack_tracking_scale**2)
            reward = (1-weight)*reward+weight*active_tracking
        guard = self.clock >= self.reference['guard_hold_start']
        if self.args.guard_relative_weight:
            chest_rotation = self.sim.data.xmat[:, self.chest].reshape(self.n, 3, 3)
            relative = points_in_body_frame(self.sim.data.site_xpos[:, self.hands],
                                            self.sim.data.xpos[:, self.chest], chest_rotation)
            relative_error = (relative-self.reference['guard_glove_chest_pos']).square().sum(-1).mean(-1)
            weight = self.args.guard_relative_weight*guard.float()
            reward = (1-weight)*reward+weight/(1+relative_error/.02**2)
        reward -= .05*guard.float()*(1-torch.exp(-.2*v.square().mean(-1)))
        reward -= .001*action.square().mean(-1)
        reward -= .002*(action-self.last_action).square().mean(-1)
        if self.args.joint_limit_penalty:
            if peak_limit_excursion is None:
                peak_limit_excursion = maximum_joint_limit_excursion(
                    q[:, self.qadr], self.joint_low, self.joint_high)
            # Normalize by an angular scale of 0.01 rad. This is
            # a training cost, not a change to the physical joint stops or gate.
            reward -= self.args.joint_limit_penalty*(peak_limit_excursion/.01).square()
        finite = torch.isfinite(q).all(-1) & torch.isfinite(v).all(-1)
        fallen = (~finite | (q[:, 2] < self.reference['qpos'][0, 2]*.65)
                  | (quat_up(q[:, 3:7]) < .3) | (v.abs().amax(-1) > 100))
        reward = torch.nan_to_num(reward)-2*fallen.float()
        completed = self.clock >= self.duration
        return reward, fallen | completed, fallen, completed

    def step(self, action):
        peak_limit_excursion = torch.zeros(self.n, device=self.device)
        for _ in range(self.decimation):
            self.advance_substep(action)
            if self.args.joint_limit_penalty:
                peak_limit_excursion = torch.maximum(peak_limit_excursion,
                    maximum_joint_limit_excursion(self.sim.data.qpos[:, self.qadr],
                                                 self.joint_low, self.joint_high))
        if self.args.refresh_kinematics:
            # Recompute derived state, without integrating or replacing qpos/qvel.
            self.sim.forward()
        reward, done, fallen, completed = self.reward(action, peak_limit_excursion)
        self.last_action[:] = action
        self.fallen += int(fallen.sum())
        self.completed += int((completed & ~fallen).sum())
        ids = done.nonzero(as_tuple=False).flatten()
        if len(ids):
            self.reset(ids)
        return self.observe(), reward, done


def advantages(rewards, values, dones, last_value, gamma=.99, lam=.95):
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(last_value)
    for step in reversed(range(len(rewards))):
        nxt = last_value if step == len(rewards)-1 else values[step+1]
        alive = (~dones[step]).float()
        delta = rewards[step]+gamma*nxt*alive-values[step]
        carry = delta+gamma*lam*alive*carry
        advantage[step] = carry
    return advantage, advantage+values


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_controller_schema(expected, actual):
    """Prevent policy weights being silently reinterpreted by a different PD/task API."""
    if not expected:
        raise ValueError('Checkpoint lacks a controller schema; cannot verify its control contract.')
    numeric = ('kp', 'kd', 'qpos_addresses', 'dof_addresses', 'torque_limits',
               'action_scale_rad', 'observation_clip', 'physics_dt', 'actor_dt', 'speed', 'guard_hold_s')
    for key in numeric:
        if key not in expected or np.shape(expected[key]) != np.shape(actual[key]):
            raise ValueError(f'Checkpoint controller mismatch: {key}.')
        if not np.allclose(expected[key], actual[key], rtol=1e-6, atol=1e-7):
            raise ValueError(f'Checkpoint controller mismatch: {key}.')
    for key in ('version', 'action', 'observations', 'joint_names', 'phase_clock', 'feedforward'):
        if expected.get(key) != actual[key]:
            raise ValueError(f'Checkpoint controller mismatch: {key}.')
    if expected.get('balance_feedback', False) != actual['balance_feedback']:
        raise ValueError('Checkpoint controller mismatch: balance_feedback.')
    for key, default in (('controller_mode', 'explicit_pd'), ('refresh_kinematics', False)):
        if expected.get(key, default) != actual.get(key, default):
            raise ValueError(f'Checkpoint controller mismatch: {key}.')


def save_checkpoint(path, policy, optimizer, args, update, observations, schema):
    torch.save(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                    update=update, observations=observations, actions=policy.log_std.numel(),
                    controller_schema=schema,
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    model_sha256=sha256(args.model), reference_sha256=sha256(args.reference)), path)


@torch.no_grad()
def evaluate(env, policy, output, *, label, recorded=4, perturb=True):
    """Guard-start actual physics; exactly one reset per world for the whole cycle."""
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256(env.args.checkpoint) if policy is not None else ''
    torch.manual_seed(env.args.seed+1000)
    bank = np.load(env.args.evaluation_bank, allow_pickle=False) if env.args.evaluation_bank else None
    env.reset(random_phase=False, perturb=perturb and bank is None)
    # First world is the exact guard. Other worlds use reproducible tiny disturbances.
    if bank is None:
        env.sim.data.qpos[0] = env.reference['qpos'][0]
        env.sim.data.qvel[0] = env.reference['qvel'][0]
    else:
        if bank['qpos'].shape != (env.n, env.model.nq) or bank['qvel'].shape != (env.n, env.model.nv):
            raise ValueError('Evaluation-bank state dimensions differ from this model.')
        if not np.isfinite(bank['qpos']).all() or not np.isfinite(bank['qvel']).all():
            raise ValueError('Evaluation bank contains nonfinite states.')
        for key, path in (('model_sha256', env.args.model), ('reference_sha256', env.args.reference)):
            if key in bank and str(bank[key].item()) != sha256(path):
                raise ValueError(f'Evaluation bank {key} differs from the supplied experiment.')
        env.sim.data.qpos[:] = tensor(bank['qpos'], env.device)
        env.sim.data.qvel[:] = tensor(bank['qvel'], env.device)
    env.sim.forward()
    recorded = min(recorded, env.n)
    storage = {k: [] for k in ('t', 'qpos', 'qvel', 'ctrl', 'actuator_force', 'qfrc_actuator',
                              'qfrc_applied', 'xfrc_applied')}
    simulator_time = []
    support = []
    rewards, failures = [], torch.zeros(env.n, dtype=torch.bool, device=env.device)
    minimum = torch.full((env.n,), float('inf'), device=env.device)
    attack_minimum = torch.full_like(minimum, float('inf'))
    action = torch.zeros_like(env.last_action)
    steps = int(np.ceil(env.duration/env.dt))
    for step in range(steps+1):
        storage['t'].append(step*env.dt)
        simulator_time.append(env.sim.data.time[:recorded].cpu().numpy().copy())
        for key in storage:
            if key != 't':
                storage[key].append(getattr(env.sim.data, key)[:recorded].cpu().numpy().copy())
        support.append(env.support_contact()[:recorded].cpu().numpy())
        if 'head_center' in env.reference:
            distance = (env.sim.data.site_xpos[:, env.hands]-env.reference['head_center']).norm(dim=-1)
            minimum = torch.minimum(minimum, distance.amin(-1))
            if env.args.attack_window is not None:
                start, end = np.asarray(env.args.attack_window)/env.args.speed
                if start <= step*env.dt <= end:
                    attack_minimum = torch.minimum(attack_minimum, distance.amin(-1))
        if step % env.decimation == 0 and step < steps:
            if env.args.refresh_kinematics:
                env.sim.forward()
            action = torch.zeros_like(action) if policy is None else policy.actor(env.observe())
            reward, _, fallen, _ = env.reward(action)
            failures |= fallen
            rewards.append(reward.cpu().numpy())
            env.last_action[:] = action
        if step < steps:
            env.advance_substep(action)
    arrays = {k: np.asarray(v) for k, v in storage.items()}
    guard_error = (env.sim.data.qpos[:, env.qadr]-env.reference['qpos'][0, env.qadr]).abs().amax(-1)
    guard_root_error = (env.sim.data.qpos[:, :3]-env.reference['qpos'][0, :3]).norm(dim=-1)
    results = dict(kind=label, phase_clock='elapsed since guard reset, not outward-launch TTC',
                   worlds=env.n, timestep=env.dt, policy_dt=env.args.policy_dt,
                   checkpoint_sha256=checkpoint_hash,
                   duration=env.duration, seed=env.args.seed+1000,
                   evaluation_bank_sha256=sha256(env.args.evaluation_bank) if bank is not None else None,
                   initial_state_seed=int(bank['seed']) if bank is not None and 'seed' in bank else env.args.seed+1000,
                   mean_reward=float(np.mean(rewards)), fallen_fraction=float(failures.float().mean()),
                   minimum_glove_head_m=minimum.cpu().tolist(),
                   final_guard_joint_max_rad=guard_error.cpu().tolist(),
                   final_guard_root_m=guard_root_error.cpu().tolist(),
                   contact_fraction=float((minimum <= .14).float().mean()),
                   attack_window_reference_s=env.args.attack_window,
                   attack_contact_fraction=float((attack_minimum <= .14).float().mean())
                   if env.args.attack_window is not None else None,
                   attack_minimum_glove_head_m=attack_minimum.cpu().tolist()
                   if env.args.attack_window is not None else None,
                   learned_policy=policy is not None, external_assistance=False)
    for world in range(recorded):
        path = output/f'rollout_{world:02d}.npz'
        payload = {k: v if k == 't' else v[:, world] for k, v in arrays.items()}
        np.savez_compressed(path, **payload, time_s=payload['t'], state_reset_count=1,
                            simulation_time_s=np.asarray(simulator_time)[:, world],
                            support_contact=np.asarray(support)[:, world],
                            physics_timestep_s=env.dt, policy_timestep_s=env.args.policy_dt,
                            policy_update_step_indices=np.arange(0, steps, env.decimation),
                            controller_mode=env.controller_mode,
                            runtime_model_sha256=env.runtime_model_sha256,
                            reference_speed=env.args.speed,
                            evaluation_bank_sha256=sha256(env.args.evaluation_bank) if bank is not None else '',
                            evaluation_bank_world=world,
                            initial_state_seed=int(bank['seed']) if bank is not None and 'seed' in bank else env.args.seed+1000,
                            trial_id=str(bank['trial_ids'][world]) if bank is not None and 'trial_ids' in bank else str(world),
                            physics_step_count=steps, evaluation_kind=label,
                            seed=env.args.seed+1000, model_sha256=sha256(env.args.model),
                            reference_sha256=sha256(env.args.reference),
                            checkpoint=str(env.args.checkpoint or '') if policy is not None else '',
                            checkpoint_sha256=checkpoint_hash)
    (output/'metrics.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({k: v for k, v in results.items() if not isinstance(v, list)}), flush=True)
    return results


def train(env, policy, args, output):
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    start_update = 0
    if args.initialize_policy:
        saved = torch.load(args.initialize_policy, map_location=env.device, weights_only=False)
        if saved['actions'] != env.model.nu or saved['observations'] != env.observe().shape[-1]:
            raise ValueError('Warm-start policy dimensions must match this controller.')
        if saved['controller_schema']['joint_names'] != env.schema()['joint_names']:
            raise ValueError('Warm-start joint order must match this body.')
        # Explicit transfer, never a resume claim: fresh optimizer and update count.
        # Evaluation still strictly requires the final trained model/reference/schema.
        policy.load_state_dict(saved['policy'])
        (output/'initialization.json').write_text(json.dumps(dict(
            kind='explicit policy-weight transfer; new optimizer and task',
            checkpoint=str(args.initialize_policy.resolve()),
            checkpoint_sha256=sha256(args.initialize_policy),
            source_schema=saved['controller_schema'], target_schema=env.schema()), indent=2)+'\n')
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
        if saved['model_sha256'] != sha256(args.model):
            raise ValueError('Checkpoint/model mismatch; a new body requires a new controller.')
        if saved['reference_sha256'] != sha256(args.reference):
            raise ValueError('Resume requires the same committed reference.')
        validate_controller_schema(saved.get('controller_schema', {}), env.schema())
        policy.load_state_dict(saved['policy'])
        optimizer.load_state_dict(saved['optimizer'])
        start_update = int(saved['update'])
    baseline = evaluate(env, None, output/'baseline', label='reference_pd', recorded=1)
    observation = env.reset()
    obs_size = observation.shape[-1]
    started = time.monotonic()
    metrics_path = output/'training.jsonl'
    with metrics_path.open('a') as log:
        for update in range(start_update+1, start_update+args.updates+1):
            history = {k: [] for k in ('obs', 'action', 'logp', 'value', 'reward', 'done')}
            with torch.no_grad():
                for _ in range(args.horizon):
                    dist = policy.distribution(observation)
                    action = dist.sample()
                    value = policy.critic(observation).squeeze(-1)
                    next_observation, reward, done = env.step(action)
                    for key, val in (('obs', observation), ('action', action),
                                     ('logp', dist.log_prob(action).sum(-1)), ('value', value),
                                     ('reward', reward), ('done', done)):
                        history[key].append(val)
                    observation = next_observation
                data = {k: torch.stack(v) for k, v in history.items()}
                advantage, returns = advantages(data['reward'], data['value'], data['done'],
                                                policy.critic(observation).squeeze(-1))
                advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
            flat = {k: v.flatten(0, 1) for k, v in data.items()}
            adv, target = advantage.flatten(), returns.flatten()
            losses, kls = [], []
            early_stopped = False
            for _ in range(args.epochs):
                for indices in torch.randperm(len(adv), device=env.device).split(args.minibatch):
                    dist = policy.distribution(flat['obs'][indices])
                    logp = dist.log_prob(flat['action'][indices]).sum(-1)
                    log_ratio = logp-flat['logp'][indices]
                    ratio = log_ratio.exp()
                    # Nonnegative sample estimate; stop before another optimizer
                    # step can push this rollout's policy further from its origin.
                    kl = float(((ratio-1)-log_ratio).mean().detach())
                    if args.target_kl and kl > 1.5*args.target_kl:
                        early_stopped = True
                        break
                    actor = -torch.minimum(ratio*adv[indices],
                                           ratio.clamp(.8, 1.2)*adv[indices]).mean()
                    predicted = policy.critic(flat['obs'][indices]).squeeze(-1)
                    value_loss = (predicted-target[indices]).square().mean()
                    loss = actor+.5*value_loss-.001*dist.entropy().sum(-1).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    kls.append(kl)
                if early_stopped:
                    break
            metrics = dict(update=update, transitions=update*args.horizon*env.n,
                           mean_reward=float(data['reward'].mean()), loss=float(np.mean(losses)),
                           approximate_kl=float(np.mean(kls)), completed=env.completed, fallen=env.fallen,
                           optimizer_steps=len(losses), kl_early_stopped=early_stopped,
                           action_std=float(policy.log_std.detach().exp().mean()),
                           elapsed_s=time.monotonic()-started)
            log.write(json.dumps(metrics)+'\n')
            log.flush()
            if update % args.log_every == 0 or update == start_update+1:
                print(json.dumps(metrics), flush=True)
            if update % args.save_every == 0:
                save_checkpoint(output/f'checkpoint_{update:05d}.pt', policy, optimizer,
                                args, update, obs_size, env.schema())
    save_checkpoint(output/'checkpoint.pt', policy, optimizer, args, update, obs_size, env.schema())
    args.checkpoint = output/'checkpoint.pt'
    learned = evaluate(env, policy, output/'learned', label='learned_policy', recorded=args.recorded_worlds)
    (output/'comparison.json').write_text(json.dumps(
        dict(baseline=baseline, learned=learned, improvement=learned['mean_reward']-baseline['mean_reward']),
        indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train', 'evaluate'), default='train')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--body-metadata', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--initialize-policy', type=Path,
                        help='Explicit weight transfer to a changed task/controller; fresh optimizer.')
    parser.add_argument('--controller', choices=('explicit_pd', 'implicit_pd'), default='explicit_pd')
    parser.add_argument('--refresh-kinematics', action='store_true',
                        help='Refresh derived FK at policy boundaries without advancing physics.')
    parser.add_argument('--reward-profile', choices=('legacy', 'precision'), default='legacy')
    parser.add_argument('--glove-sharpness', type=float, default=600.)
    parser.add_argument('--joint-limit-penalty', type=float, default=0.,
                        help='Cost for worst actual joint excursion across every physics step; normalized by .01 rad.')
    parser.add_argument('--attack-tracking-weight', type=float, default=0.,
                        help='Mix separate active-hand tracking into the reward during the annotated attack.')
    parser.add_argument('--attack-tracking-scale', type=float, default=.06,
                        help='Distance scale in metres for rational active-hand tracking reward.')
    parser.add_argument('--guard-relative-weight', type=float, default=0.,
                        help='Mix Chest-relative glove return tracking into final-guard reward.')
    parser.add_argument('--require-five-ms', action='store_true',
                        help='Reject any physical timestep other than .005 and policy period .02.')
    parser.add_argument('--evaluation-bank', type=Path,
                        help='Frozen guard qpos/qvel NPZ; evaluation mode uses its world count.')
    parser.add_argument('--recorded-worlds', type=int, default=4)
    parser.add_argument('--attack-window', type=float, nargs=2,
                        help='Reference-time strike window for separately reported contact metrics.')
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--horizon', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--minibatch', type=int, default=2048)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--target-kl', type=float, default=0.,
                        help='Optional PPO early-stop target; zero preserves legacy fixed epochs.')
    parser.add_argument('--timestep', type=float, default=.002)
    parser.add_argument('--policy-dt', type=float, default=.02)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--hold', type=float, default=1.)
    parser.add_argument('--gain-scale', type=float, default=1.)
    parser.add_argument('--action-scale', type=float, default=.25)
    parser.add_argument('--no-feedforward', action='store_true')
    parser.add_argument('--balance-feedback', action='store_true',
                        help='Reference-linearized root feedback through joint torques only.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--save-every', type=int, default=50)
    args = parser.parse_args()
    if args.checkpoint and args.initialize_policy:
        raise ValueError('Choose exact resume/checkpoint evaluation or explicit weight transfer, not both.')
    if args.joint_limit_penalty < 0 or args.target_kl < 0:
        raise ValueError('Joint-limit penalty and target KL must be nonnegative.')
    if args.attack_window is not None and not 0 <= args.attack_window[0] < args.attack_window[1]:
        raise ValueError('Attack window must contain two ordered nonnegative reference times.')
    if args.attack_window is None:
        with np.load(args.reference, allow_pickle=False) as source:
            if 'attack_window_reference_s' in source:
                args.attack_window = source['attack_window_reference_s'].astype(float).tolist()
    if not 0 <= args.attack_tracking_weight <= 1 or args.attack_tracking_scale <= 0:
        raise ValueError('Attack tracking requires weight in [0,1] and positive distance scale.')
    if not 0 <= args.guard_relative_weight <= 1:
        raise ValueError('Guard-relative reward weight must lie in [0,1].')
    if args.attack_tracking_weight and args.attack_window is None:
        raise ValueError('Attack-specific tracking requires an annotated attack window.')
    if args.evaluation_bank and args.mode == 'evaluate':
        args.num_envs = len(np.load(args.evaluation_bank, allow_pickle=False)['qpos'])
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run this trainer inside an allocated Slurm CUDA GPU job.')
    if min(args.num_envs, args.updates, args.horizon, args.epochs, args.minibatch) < 1:
        raise ValueError('Training sizes must be positive.')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.mode == 'evaluate' and args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        # Evaluation restores the controller's timing/scales instead of silently
        # reinterpreting a checkpoint with this CLI's default hyperparameters.
        for key in ('timestep', 'policy_dt', 'speed', 'hold', 'gain_scale',
                    'action_scale', 'no_feedforward', 'balance_feedback'):
            setattr(args, key, saved['arguments'].get(key, False))
        for key, default in (('controller', 'explicit_pd'), ('refresh_kinematics', False),
                             ('reward_profile', 'legacy'), ('glove_sharpness', 600.),
                             ('joint_limit_penalty', 0.), ('attack_tracking_weight', 0.),
                             ('attack_tracking_scale', .06), ('guard_relative_weight', 0.)):
            setattr(args, key, saved['arguments'].get(key, default))
        if saved['reference_sha256'] != sha256(args.reference):
            raise ValueError('Checkpoint/reference mismatch; evaluate the committed training clip.')
    if args.require_five_ms and (abs(args.timestep-.005) > 1e-12 or abs(args.policy_dt-.02) > 1e-12):
        raise ValueError('This milestone requires genuine 5 ms physics and 20 ms policy updates.')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Freeze the small self-contained generated rig and motion before experiments.
    model_destination = output/'model'
    if args.model.parent.resolve() != model_destination.resolve():
        shutil.copytree(args.model.parent, model_destination, dirs_exist_ok=True)
    args.model = model_destination/args.model.name
    reference_destination = output/'reference.npz'
    if args.reference.resolve() != reference_destination.resolve():
        shutil.copy2(args.reference, reference_destination)
    args.reference = reference_destination
    if args.body_metadata:
        metadata_destination = output/'body_metadata.json'
        if args.body_metadata.resolve() != metadata_destination.resolve():
            shutil.copy2(args.body_metadata, metadata_destination)
        args.body_metadata = metadata_destination
    if args.evaluation_bank:
        bank_destination = output/'evaluation_bank.npz'
        if args.evaluation_bank.resolve() != bank_destination.resolve():
            shutil.copy2(args.evaluation_bank, bank_destination)
        args.evaluation_bank = bank_destination
    shutil.copy2(Path(__file__), output/'jab_learning_source.py')
    shutil.copy2(Path(__file__).resolve().parents[0]/'scalar_control.py', output/'scalar_control_source.py')
    (output/'arguments.json').write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2)+'\n')
    env = JabEnv(args)
    export_runtime_model(args.model, output/'model'/'human_runtime.xml', env.model)
    env.runtime_model_sha256 = sha256(output/'model'/'human_runtime.xml')
    with torch.cuda.stream(env.stream):
        (output/'controller_schema.json').write_text(json.dumps(env.schema(), indent=2)+'\n')
        observation = env.reset()
        policy = ActorCritic(observation.shape[-1], env.model.nu).to(env.device)
        if args.mode == 'train':
            train(env, policy, args, output)
        else:
            if args.checkpoint:
                checkpoint = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
                if checkpoint['model_sha256'] != sha256(args.model):
                    raise ValueError('Checkpoint/model hash mismatch; evaluate its original body.')
                validate_controller_schema(checkpoint.get('controller_schema', {}), env.schema())
                policy.load_state_dict(checkpoint['policy'])
            evaluate(env, policy if args.checkpoint else None, output,
                     label='learned_policy' if args.checkpoint else 'reference_pd',
                     recorded=args.recorded_worlds)
    torch.cuda.current_stream().wait_stream(env.stream)


if __name__ == '__main__':
    main()
