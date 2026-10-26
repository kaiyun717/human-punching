"""Bounded contact inverse-dynamics control and free-base forward simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from qpsolvers import solve_qp
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from .ducking_inverse import ROOT, MODEL_XML, build_g1_model, tangent_derivatives


class Reference:
    def __init__(self, model, path):
        archive = np.load(path)
        self.times = archive['time_s']
        self.poses = archive['qpos']
        self.duration = float(self.times[-1])
        self.model = model
        self.scalar = CubicSpline(self.times, self.poses)
        self.rot = RotationSpline(self.times, Rotation.from_quat(self.poses[:, [4, 5, 6, 3]]))
        vel, acc = tangent_derivatives(model, self.poses, self.times)
        self.vel = CubicSpline(self.times, vel)
        self.acc = CubicSpline(self.times, acc)

    def at(self, t):
        t = float(np.clip(t, self.times[0], self.duration))
        q = self.scalar(t)
        q[3:7] = self.rot(t).as_quat(scalar_first=True)
        return q, self.vel(t), self.acc(t)


class ContactInverseController:
    """QP for joint torques; predicted support forces are NEVER applied externally.

    The six free-base dynamics equations are exact constraints. Joint torque,
    unilateral normal-force and an inscribed friction cone are bounded. Foot
    pose feedback is a task, allowing compliant contact and lift when needed.
    """
    def __init__(self, model, motors, *, friction=.6, gain=1., foot_weight=30.):
        self.model, self.motors = model, motors
        self.friction, self.gain, self.foot_weight = friction, gain, foot_weight
        self.foot_ids = [model.body(s+'_ankle_roll_link').id for s in ('left', 'right')]
        self.local_points = np.array([[-.044, -.018, -.035], [-.044, .018, -.035],
                                      [.123, -.018, -.035], [.123, .018, -.035]])
        self.nf, self.nv = 24, model.nv
        self.n = self.nv+self.nf
        self.weight = float(np.sum(model.body_mass)*9.81)
        self.target_data = mujoco.MjData(model)

    def solve(self, data, qref, vref, aref):
        model = self.model
        target = self.target_data
        target.qpos[:], target.qvel[:] = qref, vref
        mujoco.mj_forward(model, target)
        M = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, data, M)
        h = data.qfrc_bias-data.qfrc_passive
        J = np.empty((24, model.nv))
        active = []
        for k, body in enumerate(self.foot_ids):
            R = data.xmat[body].reshape(3, 3)
            for j, local in enumerate(self.local_points):
                point = R@local+data.xpos[body]
                mujoco.mj_jac(model, data, J[3*(4*k+j):3*(4*k+j+1)], None, point, body)
                reference_point = target.xmat[body].reshape(3,3)@local+target.xpos[body]
                active.append(bool(point[2] <= .0015 and reference_point[2] <= .0015))
        B = J.T*self.weight
        error = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, error, 1., data.qpos, qref)
        kp = np.r_[np.full(3, 80.), np.full(3, 120.), np.full(29, 100.)]*self.gain
        kd = 2*np.sqrt(kp)
        desired = aref+kp*error+kd*(vref-data.qvel)
        weights = np.r_[np.full(3, 3.), np.full(3, 5.), np.ones(29)]
        H = np.eye(self.n)*1e-6
        H[self.nv:,self.nv:] = .1*np.eye(self.nf)
        H[:self.nv, :self.nv] += np.diag(weights)
        g = np.zeros(self.n)
        g[:self.nv] -= weights*desired
        # Track the desired foot acceleration and pose. This does not weld feet.
        # A reference with a lifted foot disables that foot's ground-force cone.
        for body in self.foot_ids:
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            dp, dr = np.zeros_like(jp), np.zeros_like(jr)
            mujoco.mj_jacBody(model, data, jp, jr, body)
            mujoco.mj_jacDot(model, data, dp, dr, data.xpos[body], body)
            Jf, Jdot = np.vstack((jp, jr)), np.vstack((dp, dr))
            rerror = Rotation.from_matrix(target.xmat[body].reshape(3, 3) @
                                         data.xmat[body].reshape(3, 3).T).as_rotvec()
            pe = np.r_[target.xpos[body]-data.xpos[body], rerror]
            tjp, tjr = np.zeros_like(jp), np.zeros_like(jr)
            tdp, tdr = np.zeros_like(jp), np.zeros_like(jr)
            mujoco.mj_jacBody(model, target, tjp, tjr, body)
            mujoco.mj_jacDot(model, target, tdp, tdr, target.xpos[body], body)
            Jt = np.vstack((tjp, tjr))
            ades = Jt@aref+np.vstack((tdp, tdr))@vref + 200*pe + 28*(Jt@vref-Jf@data.qvel)
            rhs = ades-Jdot@data.qvel
            H[:self.nv, :self.nv] += self.foot_weight*(Jf.T@Jf)
            g[:self.nv] -= self.foot_weight*(Jf.T@rhs)
        A = np.c_[M[:6], -B[:6]]
        b = -h[:6]
        # Normalized torque regularizer plus exact torque inequalities.
        T = np.c_[M[6:], -B[6:]]
        effort = self.motors['effort_limit']
        Tn = T/effort[:, None]
        H += .015*(Tn.T@Tn)
        g += .015*Tn.T@(h[6:]/effort)
        C = [T]
        low, high = [-effort-h[6:]], [effort-h[6:]]
        for k in range(8):
            cols = self.nv+3*k
            for j in range(8):
                row = np.zeros(self.n)
                theta = j*np.pi/4
                row[cols:cols+3] = [np.cos(theta), np.sin(theta), -self.friction*np.cos(np.pi/8)]
                C.append(row[None]); low.append(np.array([-1e6])); high.append(np.array([0.]))
            row = np.zeros(self.n); row[cols+2] = 1
            C.append(row[None]); low.append(np.array([0.])); high.append(np.array([2. if active[k] else 0.]))
        row = np.zeros((29, self.n)); row[:, 6:35] = np.eye(29)
        speed = self.motors['velocity_limit']
        # One-control-step speed bounds, in addition to measured speed checks.
        horizon = model.opt.timestep
        C.append(row)
        low.append((-speed-data.qvel[6:])/horizon)
        high.append((speed-data.qvel[6:])/horizon)
        C, low, high = np.vstack(C), np.concatenate(low), np.concatenate(high)
        # Normalize constraint rows and use a dense active-set solve. Motor
        # acceleration, newtons and normalized contact forces have different
        # scales; the physical residuals below remain in original units.
        row_scale = np.maximum(np.linalg.norm(C, axis=1), 1e-8)
        cn, ln, un = C/row_scale[:,None], low/row_scale, high/row_scale
        finite_lower = low > -1e5
        G = np.vstack((cn, -cn[finite_lower]))
        bounds = np.r_[un, -ln[finite_lower]]
        eq_scale = np.maximum(np.linalg.norm(A, axis=1), 1e-8)
        # Remove inactive force variables exactly, avoiding a degenerate cone
        # collapsed to its apex (many linearly dependent active inequalities).
        keep = np.r_[np.arange(self.nv), self.nv+np.flatnonzero(np.repeat(active,3))]
        gs = G[:,keep]
        nonzero = np.linalg.norm(gs,axis=1)>1e-12
        solved = solve_qp(H[np.ix_(keep,keep)], g[keep], G=gs[nonzero], h=bounds[nonzero],
                          A=(A/eq_scale[:,None])[:,keep], b=b/eq_scale,
                          solver='daqp', primal_tol=1e-8, dual_tol=1e-8)
        if solved is None:
            raise RuntimeError(f'Dense inverse-dynamics QP failed active={active}')
        x = np.zeros(self.n); x[keep] = solved
        equality = float(np.max(abs(A@x-b)))
        inequality = float(max(0., np.max(C@x-high), np.max(low-C@x)))
        cold_restart = False
        tau = T@x+h[6:]
        if not np.isfinite(x).all() or equality > 1e-3 or inequality > 1e-3:
            raise RuntimeError(f'Inverse QP residual failed, '
                               f'equality={equality:g}, inequality={inequality:g}, active={active}')
        return tau, dict(equality=equality, inequality=inequality, forces=x[self.nv:].reshape(8, 3)*self.weight,
                         active=active, qacc=x[:self.nv].copy(), cold_restart=cold_restart)


def run(reference_path, output_dir, *, gain=1., foot_weight=30., friction=.6,
        timestep=.002, joint_feedback=1.):
    began = time.monotonic()
    output = Path(output_dir).resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError('Outputs must stay inside human-punching')
    output.mkdir(parents=True, exist_ok=True)
    model, motors = build_g1_model(timestep=timestep, friction=friction)
    reference = Reference(model, reference_path)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], _ = reference.at(0.)
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    controller = ContactInverseController(model, motors, friction=friction, gain=gain,
                                           foot_weight=foot_weight)
    # Torque-level posture feedback compensates the difference between the
    # optimized support-force distribution and MuJoCo's compliant contacts.
    posture_kp, posture_kd = [], []
    for name in motors['joint_names']:
        if 'ankle' in name:
            kp,kd=120.,3.
        elif 'hip' in name or 'knee' in name:
            kp,kd=80.,3.
        elif 'waist' in name:
            kp,kd=50.,2.
        elif 'wrist_pitch' in name or 'wrist_yaw' in name:
            kp,kd=5.,.3
        else:
            kp,kd=20.,1.
        posture_kp.append(kp*joint_feedback); posture_kd.append(kd*joint_feedback)
    posture_kp,posture_kd=np.asarray(posture_kp),np.asarray(posture_kd)
    trace = {key: [] for key in ('time_s','qpos','qvel','qpos_reference','torque','contact_forces',
                                 'qp_equality','qp_inequality','contact_count','base_applied',
                                 'joint_error','foot_positions','com','min_nonfoot_contact_distance',
                                 'measured_foot_forces','joint_limit_violation','qp_cold_restart',
                                 'qacc','actual_torque','qfrc_constraint','dynamics_residual',
                                 'joint_limit_constraints','head_height','torso_tilt_deg',
                                 'inverse_torque','posture_feedback_torque','torque_clip')}
    stop_reason = 'completed'
    foot_ids = set(controller.foot_ids)
    floor_id = model.geom('ducking_ground').id
    nsteps = int(round(reference.duration/timestep))
    for step in range(nsteps+1):
        t = float(data.time)
        qref, vref, aref = reference.at(t)
        mujoco.mj_forward(model, data)
        try:
            inverse_torque, qp = controller.solve(data, qref, vref, aref)
        except RuntimeError as exc:
            stop_reason = str(exc)
            np.savez(output/'failure_state.npz',qpos=data.qpos,qvel=data.qvel,
                     qref=qref,vref=vref,aref=aref,time_s=data.time)
            break
        feedback = posture_kp*(qref[7:]-data.qpos[7:])+posture_kd*(vref[6:]-data.qvel[6:])
        raw_torque=inverse_torque+feedback
        torque=np.clip(raw_torque,-motors['effort_limit'],motors['effort_limit'])
        data.ctrl[:] = torque
        # Recompute contact forces for this control, before recording them.
        mujoco.mj_forward(model, data)
        # Gravity and MuJoCo contact alone support the base. Never reset poses
        # after the initial condition or inject the optimized contact forces.
        assert not data.qfrc_applied.any() and not data.xfrc_applied.any()
        nonfoot = []
        measured_forces = np.zeros((2, 3))
        for ci, contact in enumerate(data.contact):
            a,b = int(contact.geom1),int(contact.geom2)
            if floor_id in (a,b):
                other = b if a == floor_id else a
                if int(model.geom_bodyid[other]) not in foot_ids:
                    nonfoot.append(float(contact.dist))
                else:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, force)
                    world_force = contact.frame.reshape(3, 3).T @ force[:3]
                    side = controller.foot_ids.index(int(model.geom_bodyid[other]))
                    measured_forces[side] += world_force * (1 if b == other else -1)
            elif contact.dist < -.002:
                nonfoot.append(float(contact.dist))
        trace['time_s'].append(t); trace['qpos'].append(data.qpos.copy())
        trace['qvel'].append(data.qvel.copy()); trace['qpos_reference'].append(qref)
        trace['torque'].append(torque); trace['contact_forces'].append(qp['forces'])
        trace['inverse_torque'].append(inverse_torque)
        trace['posture_feedback_torque'].append(feedback)
        trace['torque_clip'].append(raw_torque-torque)
        trace['qp_equality'].append(qp['equality']); trace['qp_inequality'].append(qp['inequality'])
        trace['contact_count'].append(data.ncon); trace['base_applied'].append(data.qfrc_applied[:6].copy())
        trace['qp_cold_restart'].append(qp['cold_restart'])
        trace['joint_error'].append(data.qpos[7:]-qref[7:])
        trace['foot_positions'].append(data.xpos[list(controller.foot_ids)].copy())
        trace['com'].append(data.subtree_com[model.body('pelvis').id].copy())
        trace['min_nonfoot_contact_distance'].append(min(nonfoot, default=0.))
        trace['measured_foot_forces'].append(measured_forces)
        limits = model.jnt_range[motors['joint_ids']]
        trace['joint_limit_violation'].append(max(0.,np.max(limits[:,0]-data.qpos[7:]),
                                                  np.max(data.qpos[7:]-limits[:,1])))
        trace['qacc'].append(data.qacc.copy())
        trace['actual_torque'].append(data.actuator_force.copy())
        trace['qfrc_constraint'].append(data.qfrc_constraint.copy())
        mass = np.empty((model.nv,model.nv))
        mujoco.mj_fullM(model,data,mass)
        residual = (mass@data.qacc+data.qfrc_bias-data.qfrc_passive-
                    data.qfrc_actuator-data.qfrc_constraint)
        trace['dynamics_residual'].append(float(np.max(abs(residual))))
        trace['joint_limit_constraints'].append(int(np.sum(
            data.efc_type==mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)))
        trace['head_height'].append(float(data.geom_xpos[model.geom('head_collision').id,2]))
        zaxis=data.xmat[model.body('torso_link').id].reshape(3,3)[:,2]
        trace['torso_tilt_deg'].append(float(np.rad2deg(np.arctan2(np.linalg.norm(zaxis[:2]),zaxis[2]))))
        if not np.isfinite(data.qpos).all() or data.qpos[2] < .25:
            stop_reason = 'fall_or_nonfinite'
            break
        if step < nsteps:
            mujoco.mj_step(model, data)
    arrays = {key: np.asarray(value) for key,value in trace.items()}
    np.savez_compressed(output/'simulation.npz', **arrays,
                        joint_names=np.asarray(motors['joint_names']), effort_limits=motors['effort_limit'],
                        velocity_limits=motors['velocity_limit'])
    if len(arrays['time_s']):
        q, qr, v = arrays['qpos'], arrays['qpos_reference'], arrays['qvel']
        metrics = dict(samples=len(q), simulated_seconds=float(arrays['time_s'][-1]),
            joint_rms_error_rad=float(np.sqrt(np.mean((q[:,7:]-qr[:,7:])**2))),
            joint_max_error_rad=float(np.max(abs(q[:,7:]-qr[:,7:]))),
            pelvis_rms_error_m=float(np.sqrt(np.mean(np.sum((q[:,:3]-qr[:,:3])**2, axis=1)))),
            pelvis_max_error_m=float(np.max(np.linalg.norm(q[:,:3]-qr[:,:3],axis=1))),
            pelvis_drop_m=float(np.ptp(q[:,2])),
            max_torque_fraction=float(np.max(abs(arrays['torque'])/motors['effort_limit'])),
            max_speed_fraction=float(np.max(abs(v[:,6:])/motors['velocity_limit'])),
            final_joint_rms_error_rad=float(np.sqrt(np.mean((q[-1,7:]-qr[-1,7:])**2))),
            final_base_speed_mps=float(np.linalg.norm(v[-1,:3])),
            min_nonfoot_contact_distance_m=float(np.min(arrays['min_nonfoot_contact_distance'])),
            max_joint_limit_violation_rad=float(np.max(arrays['joint_limit_violation'])),
            minimum_total_measured_normal_force_n=float(np.min(arrays['measured_foot_forces'][:,:,2].sum(1))),
            maximum_total_measured_normal_force_n=float(np.max(arrays['measured_foot_forces'][:,:,2].sum(1))),
            maximum_qp_equality_residual=float(np.max(arrays['qp_equality'])),
            qp_cold_restarts=int(np.sum(arrays['qp_cold_restart'])),
            maximum_forward_dynamics_residual=float(np.max(arrays['dynamics_residual'])),
            maximum_actual_motor_fraction=float(np.max(abs(arrays['actual_torque'])/motors['effort_limit'])),
            maximum_torque_clip_nm=float(np.max(abs(arrays['torque_clip']))),
            maximum_joint_limit_constraints=int(np.max(arrays['joint_limit_constraints'])),
            head_drop_m=float(np.ptp(arrays['head_height'])),
            maximum_torso_tilt_deg=float(np.max(arrays['torso_tilt_deg'])),
            external_base_force_max=float(np.max(abs(arrays['base_applied']))))
    else:
        metrics = dict(samples=0)
    result = dict(status=stop_reason, reference_path=str(Path(reference_path).resolve()),
                  trajectory_path=str(output/'simulation.npz'), model_xml_path=str(MODEL_XML),
                  free_floating_base=True, actuated_joints_only=True, pose_resets_after_initialization=0,
                  predicted_contact_forces_applied=False, friction=friction, timestep_s=timestep,
                  gain=gain, foot_weight=foot_weight, joint_feedback=joint_feedback, metrics=metrics,
                  controller='contact inverse-dynamics QP plus bounded joint posture feedback',
                  controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_builder_sha256=hashlib.sha256((ROOT/'src/human_punching/ducking_inverse.py').read_bytes()).hexdigest(),
                  source_model_sha256=hashlib.sha256(MODEL_XML.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
                  wall_seconds=time.monotonic()-began)
    (output/'simulation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gain',type=float,default=1.)
    parser.add_argument('--foot-weight',type=float,default=30.)
    parser.add_argument('--timestep',type=float,default=.002)
    parser.add_argument('--joint-feedback',type=float,default=1.)
    args=parser.parse_args()
    result=run(args.reference,args.output,gain=args.gain,foot_weight=args.foot_weight,
               timestep=args.timestep,joint_feedback=args.joint_feedback)
    raise SystemExit(0 if result['status']=='completed' else 2)
