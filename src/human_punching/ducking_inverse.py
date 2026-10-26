"""Contact-aware inverse dynamics for the recorded G1 duck; no base actuation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
MODEL_XML = ROOT / '.venv-task10c/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
CONSTANTS_SOURCE = MODEL_XML.parent.parent / 'g1_constants.py'
DEFAULT_TRAJECTORY = ROOT / 'outputs/ducking/g1_ducking.npz'


def _armature(rotors, gears):
    return rotors[0] * (gears[1] * gears[2])**2 + rotors[1] * gears[2]**2 + rotors[2]


def motor_properties(model: mujoco.MjModel) -> dict:
    """Installed mjlab G1 motor limits and nominal reflected inertias, joint order.

    Parallel ankle/waist linkage uses mjlab's documented nominal 1:1 model;
    these constants are model assumptions, not a measured hardware envelope.
    """
    a5020 = _armature((.139e-4, .017e-4, .169e-4), (1, 1+46/18, 1+56/16))
    a752014 = _armature((.489e-4, .098e-4, .533e-4), (1, 4.5, 1+48/22))
    a752022 = _armature((.489e-4, .109e-4, .738e-4), (1, 4.5, 5))
    a4010 = _armature((.068e-4, 0, 0), (1, 5, 5))
    names, ids, effort, speed, armature = [], [], [], [], []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = model.joint(j).name
        if any(s in name for s in ('ankle_pitch', 'ankle_roll', 'waist_pitch', 'waist_roll')):
            e, v, a = 50., 37., 2*a5020
        elif any(s in name for s in ('hip_pitch', 'hip_yaw', 'waist_yaw')):
            e, v, a = 88., 32., a752014
        elif any(s in name for s in ('hip_roll', 'knee')):
            e, v, a = 139., 20., a752022
        elif any(s in name for s in ('wrist_pitch', 'wrist_yaw')):
            e, v, a = 5., 22., a4010
        elif any(s in name for s in ('elbow', 'shoulder', 'wrist_roll')):
            e, v, a = 25., 37., a5020
        else:
            raise ValueError(f'Unknown G1 motor {name}')
        names.append(name); ids.append(j); effort.append(e); speed.append(v); armature.append(a)
    ids = np.asarray(ids, dtype=int)
    return dict(joint_names=names, joint_ids=ids, qpos_indices=model.jnt_qposadr[ids].copy(),
                dof_indices=model.jnt_dofadr[ids].copy(), effort_limit=np.asarray(effort),
                velocity_limit=np.asarray(speed), armature=np.asarray(armature),
                constants_source=str(CONSTANTS_SOURCE))


def build_g1_model(timestep: float = .002, friction: float = .6):
    """Native floating G1, native collision capsules, bounded direct torque motors."""
    spec = mujoco.MjSpec.from_file(str(MODEL_XML))
    base = spec.compile()
    motors = motor_properties(base)
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.gravity = [0, 0, -9.81]
    spec.option.iterations = 100
    spec.option.tolerance = 1e-10
    spec.worldbody.add_geom(name='ducking_ground', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[10, 10, .1], rgba=[.22, .25, .29, 1],
                            contype=1, conaffinity=1, condim=3,
                            friction=[friction, .005, .0001])
    for geom in spec.geoms:
        name = geom.name or ''
        if name.endswith('_collision'):
            geom.contype = geom.conaffinity = 1
            geom.condim = 3 if '_foot' in name else 1
            if '_foot' in name:
                geom.priority = 1
                geom.friction = [friction, .005, .0001]
    for name, armature, effort in zip(motors['joint_names'], motors['armature'], motors['effort_limit']):
        spec.joint(name).armature = float(armature)
        spec.add_actuator(name='torque_'+name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gear=[1, 0, 0, 0, 0, 0], gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                          gainprm=[1] + [0]*9, biastype=mujoco.mjtBias.mjBIAS_NONE,
                          ctrllimited=True, ctrlrange=[-float(effort), float(effort)],
                          forcelimited=True, forcerange=[-float(effort), float(effort)])
    model = spec.compile()
    assert (model.nq, model.nv, model.nu) == (36, 35, 29)
    np.testing.assert_array_equal(model.actuator_trnid[:, 0], motors['joint_ids'])
    np.testing.assert_allclose(model.dof_armature[motors['dof_indices']], motors['armature'])
    return model, motors


def tangent_derivatives(model, qpos, times):
    """Differentiate poses on MuJoCo's tangent space, never quaternion components."""
    qpos, times = np.asarray(qpos), np.asarray(times)
    if len(qpos) < 3 or len(times) != len(qpos) or np.any(np.diff(times) <= 0):
        raise ValueError('Need three or more poses and strictly increasing times')
    velocity = np.zeros((len(qpos), model.nv))
    for i in range(len(qpos)):
        lo, hi = max(0, i-1), min(len(qpos)-1, i+1)
        mujoco.mj_differentiatePos(model, velocity[i], times[hi]-times[lo], qpos[lo], qpos[hi])
    acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
    return velocity, acceleration


def smooth_trajectory(qpos, times, sigma_s=.04):
    result = np.asarray(qpos).copy()
    if sigma_s <= 0:
        return result
    sigma = sigma_s / np.mean(np.diff(times))
    result = gaussian_filter1d(result, sigma, axis=0, mode='nearest')
    rotation = Rotation.from_quat(np.asarray(qpos)[:, [4, 5, 6, 3]])
    relative = (rotation[0].inv() * rotation).as_rotvec()
    if np.max(np.linalg.norm(relative, axis=-1)) > np.pi-.05:
        raise ValueError('Root motion leaves the local rotation smoothing chart')
    filtered = rotation[0] * Rotation.from_rotvec(gaussian_filter1d(relative, sigma, axis=0, mode='nearest'))
    result[:, 3:7] = filtered.as_quat()[:, [3, 0, 1, 2]]
    return result


def contact_points_and_jacobians(model, data):
    """Sphere-bottom points at both endpoints of every native foot capsule.

    Points are actual world positions, not projected onto an invented floor.
    A flat capsule has a line of contact whose wrench cone is spanned by its
    endpoint contacts. A tilted capsule only admits its low endpoint(s).
    """
    points, jacobians, labels, bodies = [], [], [], []
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if not name or '_foot' not in name or not name.endswith('_collision'):
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            raise ValueError('Expected native G1 foot capsules')
        radius, half = model.geom_size[gid, :2]
        axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
        body = int(model.geom_bodyid[gid])
        for end in (-1, 1):
            point = data.geom_xpos[gid] + end*half*axis - [0, 0, radius]
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jp, jr, point, body)
            points.append(point); jacobians.append(jp); bodies.append(body); labels.append(f'{name}:{end}')
    return dict(points=np.asarray(points), jacobians=np.asarray(jacobians),
                body_ids=np.asarray(bodies), names=labels)


def bounded_inverse(model, data, acceleration, motors, *, friction=.6, max_gap=.005,
                    contact_mode='near'):
    """Feasible foot-force distribution, with unactuated base and motor bounds.

    'optimistic' permits all foot points independent of gap; this is a wrench
    bound only. 'near' permits points within max_gap of the ground, whose drift
    still must be checked separately. Neither mode certifies a trajectory.
    """
    contact = contact_points_and_jacobians(model, data)
    allowed = np.ones(len(contact['points']), dtype=bool) if contact_mode == 'optimistic' else contact['points'][:, 2] <= max_gap
    if contact_mode not in ('near', 'optimistic'):
        raise ValueError(contact_mode)
    J = contact['jacobians'][allowed].reshape(-1, model.nv)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    required = M @ acceleration + data.qfrc_bias - data.qfrc_passive
    n = len(J)
    result = dict(feasible=False, required=required, allowed=allowed,
                  torque=np.full(model.nu, np.nan), forces=np.zeros((len(allowed), 3)),
                  equality_residual=np.inf, max_gap_m=float(np.max(contact['points'][:, 2])),
                  minimum_gap_m=float(np.min(contact['points'][:, 2])),
                  max_point_speed_mps=float(np.max(np.linalg.norm(contact['jacobians'] @ data.qvel, axis=-1))),
                  left_available=int(sum(allowed[k] and name.startswith('left') for k, name in enumerate(contact['names']))),
                  right_available=int(sum(allowed[k] and name.startswith('right') for k, name in enumerate(contact['names']))),
                  loaded_contact_max_speed_mps=np.nan, loaded_contact_max_gap_m=np.nan)
    if n == 0:
        torque = required[motors['dof_indices']]
        residual = float(np.max(abs(required[:6])))
        result['feasible'] = bool(residual < 1e-6 and np.all(abs(torque) <= motors['effort_limit']))
        result['reason'] = 'unactuated_flight_balance' if result['feasible'] else 'unsupported_nonballistic_motion'
        result['equality_residual'] = residual
        if result['feasible']:
            result['torque'] = torque
        return result
    # Inscribed eight-sided friction cone, conservatively inside circular cone.
    cone = np.zeros((n//3*8, n))
    for k in range(n//3):
        for a in range(8):
            theta = a*np.pi/4
            cone[8*k+a, 3*k:3*k+3] = [np.cos(theta), np.sin(theta), -friction*np.cos(np.pi/8)]
    act = motors['dof_indices']
    L = J.T[act]
    limit = motors['effort_limit']
    # Minimize the largest normalized motor torque, rather than returning an
    # arbitrary force distribution at the motor bounds. Last variable is rho.
    A = np.vstack((np.c_[cone, np.zeros(len(cone))],
                   np.c_[L, -limit], np.c_[-L, -limit]))
    b = np.r_[np.zeros(len(cone)), required[act], -required[act]]
    bounds = [(None, None), (None, None), (0, None)] * (n//3) + [(0, 1)]
    lp = linprog(np.r_[np.tile([0., 0., 1e-7], n//3), 1.], A_ub=A, b_ub=b,
                 A_eq=np.c_[J.T[:6], np.zeros(6)], b_eq=required[:6], bounds=bounds, method='highs')
    result['reason'] = lp.message
    if lp.success:
        force = lp.x[:-1].reshape(-1, 3)
        torque = required[act] - J.T[act] @ lp.x[:-1]
        residual = required - J.T @ lp.x[:-1]
        residual[act] -= torque
        if (np.max(abs(residual)) > 1e-5 or np.max(A @ lp.x - b) > 1e-5
                or np.max(np.abs(torque) - limit) > 1e-5):
            raise RuntimeError('Inverse-dynamics LP returned inconsistent forces or torques')
        result.update(feasible=True, torque=torque, equality_residual=float(np.max(abs(residual))))
        result['forces'][allowed] = force
        loaded = result['forces'][:, 2] > 1.
        if loaded.any():
            result['loaded_contact_max_speed_mps'] = float(np.max(np.linalg.norm((contact['jacobians'] @ data.qvel)[loaded], axis=-1)))
            result['loaded_contact_max_gap_m'] = float(np.max(np.abs(contact['points'][loaded, 2])))
    return result


def audit_trajectory(model, qpos, times, *, friction=.6, contact_mode='near', max_gap=.005,
                     stride=1):
    motors = motor_properties(model)
    velocity, acceleration = tangent_derivatives(model, qpos, times)
    data = mujoco.MjData(model)
    indices = np.arange(0, len(qpos), stride)
    arrays = {'feasible': [], 'torque': [], 'base_required': [], 'force': [],
              'minimum_foot_gap': [], 'maximum_foot_gap': [], 'foot_point_speed': [],
              'equality_residual': [], 'contact_points': [], 'left_available': [],
              'right_available': [], 'loaded_contact_speed': [], 'loaded_contact_gap': []}
    for i in indices:
        data.qpos[:] = qpos[i]; data.qvel[:] = velocity[i]
        mujoco.mj_forward(model, data)
        inverse = bounded_inverse(model, data, acceleration[i], motors, friction=friction,
                                  contact_mode=contact_mode, max_gap=max_gap)
        arrays['feasible'].append(inverse['feasible'])
        arrays['torque'].append(inverse['torque'])
        arrays['force'].append(inverse['forces'])
        arrays['base_required'].append(inverse['required'][:6])
        arrays['minimum_foot_gap'].append(inverse['minimum_gap_m'])
        arrays['maximum_foot_gap'].append(inverse['max_gap_m'])
        arrays['foot_point_speed'].append(inverse['max_point_speed_mps'])
        arrays['equality_residual'].append(inverse['equality_residual'])
        arrays['contact_points'].append(int(inverse['allowed'].sum()))
        arrays['left_available'].append(inverse['left_available'])
        arrays['right_available'].append(inverse['right_available'])
        arrays['loaded_contact_speed'].append(inverse['loaded_contact_max_speed_mps'])
        arrays['loaded_contact_gap'].append(inverse['loaded_contact_max_gap_m'])
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    ratio = np.abs(arrays['torque']) / motors['effort_limit']
    finite_ratio = ratio[np.isfinite(ratio)]
    summary = dict(frames_audited=len(indices), feasible_wrench_frames=int(arrays['feasible'].sum()),
                   fraction_feasible_wrench=float(arrays['feasible'].mean()),
                   contact_mode=contact_mode, friction=friction, max_gap_m=max_gap,
                   max_torque_fraction_on_feasible_frames=float(finite_ratio.max()) if len(finite_ratio) else None,
                   max_joint_speed_fraction=float(np.max(np.abs(velocity[:, motors['dof_indices']]) / motors['velocity_limit'])),
                   minimum_foot_gap_m=float(arrays['minimum_foot_gap'].min()),
                   maximum_foot_gap_m=float(arrays['maximum_foot_gap'].max()),
                   maximum_foot_point_speed_mps=float(arrays['foot_point_speed'].max()),
                   frames_neither_foot_available=int(np.sum((arrays['left_available']==0)&(arrays['right_available']==0))),
                   frames_only_left_available=int(np.sum((arrays['left_available']>0)&(arrays['right_available']==0))),
                   frames_only_right_available=int(np.sum((arrays['left_available']==0)&(arrays['right_available']>0))),
                   frames_both_available=int(np.sum((arrays['left_available']>0)&(arrays['right_available']>0))),
                   floating_base_actuation=False,
                   qualification='Instantaneous force balance only; contact gaps, slip and acceleration consistency remain separate requirements.')
    arrays.update(indices=indices, qvel=velocity, qacc=acceleration)
    return summary, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/ducking/inverse')
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError('Output must stay inside human-punching')
    out.mkdir(parents=True, exist_ok=True)
    packet = np.load(args.trajectory)
    model, motors = build_g1_model()
    baseq, baset = packet['qpos'], packet['time_s']
    report = dict(model=str(MODEL_XML), motor_source=str(CONSTANTS_SOURCE), variants={})
    for label, sigma, stretch in [('raw_1x', 0., 1.), ('smooth_1x', .04, 1.), ('smooth_0.5x', .04, 2.), ('smooth_0.25x', .04, 4.)]:
        q = smooth_trajectory(baseq, baset, sigma)
        t = baset*stretch
        for mode in ('near', 'optimistic'):
            name = label+'_'+mode
            summary, arrays = audit_trajectory(model, q, t, contact_mode=mode)
            report['variants'][name] = summary
            np.savez_compressed(out/(name+'.npz'), qpos=q, time_s=t, **arrays)
            print(name, json.dumps(summary), flush=True)
    (out/'inverse_audit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
