"""Condition the GMR duck for fixed, level foot contacts without dynamic claims."""
from __future__ import annotations

import json
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, RotationSpline

from human_punching.ducking import DEFAULT_MODEL, ROOT


def _ease_clock(clock, duration, edge=.2):
    """C2 onset/offset, identity clock away from the endpoint windows."""
    result = np.asarray(clock, dtype=float).copy()
    for reverse in (False, True):
        value = duration - clock if reverse else clock
        mask = value < edge
        u = value[mask] / edge
        mapped = edge * (6*u**3 - 8*u**4 + 3*u**5)
        result[mask] = duration - mapped if reverse else mapped
    return result


def condition_reference(output_dir, source_path=None, *, duration_s=4., hold_s=.75,
                        final_hold_s=1., root_lowering_m=.08, smooth_sigma_s=.04,
                        sample_hz=120., stance_width_scale=1., joint_margin_rad=.03,
                        com_weight=30.):
    """Save a slow or original-duration duck with finite planted-foot geometry.

    Torso orientation and arms track the smoothed recorded shape. Root pose and
    leg articulation may change to accommodate fixed soles and joint limits.
    No state is changed during simulation here.
    """
    output = Path(output_dir).resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError('Outputs must stay within human-punching')
    source_path = Path(source_path or ROOT/'outputs/ducking/g1_ducking.npz').resolve()
    original = np.load(source_path)
    times, poses = original['time_s'], original['qpos']
    if min(duration_s, sample_hz) <= 0 or min(hold_s, final_hold_s, root_lowering_m, smooth_sigma_s) < 0:
        raise ValueError('Invalid reference conditioning parameters')
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
    data = mujoco.MjData(model)
    # Smooth targets before fitting contacts. Quaternion signs are handled in a
    # local rotation chart, rather than averaging quaternion coordinates.
    sigma = smooth_sigma_s / np.median(np.diff(times))
    smooth = gaussian_filter1d(poses, sigma, axis=0, mode='nearest') if sigma else poses.copy()
    rotations = Rotation.from_quat(poses[:, [4, 5, 6, 3]])
    first = rotations[0]
    rv = (rotations * first.inv()).as_rotvec()
    if sigma:
        rv = gaussian_filter1d(rv, sigma, axis=0, mode='nearest')
    rotations = Rotation.from_rotvec(rv) * first
    smooth[:, 3:7] = rotations.as_quat()[:, [3, 0, 1, 2]]
    # Feet use one constant representative guard anchor and yaw per side.
    guards = np.r_[np.arange(min(25, len(poses))), np.arange(max(0, len(poses)-25), len(poses))]
    samples = {side: [] for side in ('left', 'right')}
    for index in guards:
        data.qpos[:] = smooth[index]
        mujoco.mj_forward(model, data)
        for side in samples:
            body = data.body(side+'_ankle_roll_link')
            rot = body.xmat.reshape(3, 3)
            samples[side].append(np.r_[body.xpos[:2], np.arctan2(rot[1, 0], rot[0, 0])])
    anchors = {}
    center_y = float(np.mean([np.median(values,axis=0)[1] for values in samples.values()]))
    for side, values in samples.items():
        median = np.median(values, axis=0)
        median[1] = center_y + stance_width_scale * (median[1]-center_y)
        position = np.r_[median[:2], .035]
        rotation = Rotation.from_euler('z', median[2]).as_matrix()
        anchors[side] = {'position': position.tolist(), 'rotation': rotation.tolist()}
    motion_time = np.linspace(0, duration_s, int(round(duration_s*sample_hz))+1)
    source_clock = _ease_clock(motion_time / duration_s * times[-1], times[-1])
    motion = CubicSpline(times, smooth)(source_clock)
    motion[:, 3:7] = RotationSpline(times, rotations)(source_clock).as_quat()[:, [3, 0, 1, 2]]
    motion[:, 2] -= root_lowering_m
    targets = motion.copy()
    configuration = mink.Configuration(model)
    configuration.update(targets[0])
    foot_tasks = []
    for side, anchor in anchors.items():
        task = mink.FrameTask(side+'_ankle_roll_link', 'body', 1500., 1500., lm_damping=0.1)
        quat = Rotation.from_matrix(anchor['rotation']).as_quat(scalar_first=True)
        task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(quat), np.array(anchor['position'])))
        foot_tasks.append(task)
    root_task = mink.FrameTask('pelvis', 'body', [20.,20.,100.], 5., lm_damping=.1)
    torso_task = mink.FrameTask('torso_link', 'body', 0., 70., lm_damping=.1)
    weights = np.r_[np.zeros(6), np.ones(model.nv-6)]
    for joint in range(1,model.njnt):
        name = model.joint(joint).name
        if any(part in name for part in ('shoulder','elbow','wrist')):
            weights[model.jnt_dofadr[joint]] = 5.
    posture = mink.PostureTask(model, cost=weights)
    com_task = mink.ComTask(cost=[com_weight,com_weight,0.])
    center = np.mean([anchor['position'] for anchor in anchors.values()],axis=0)
    center[0] += .035
    com_task.set_target(center)
    tasks = [root_task,torso_task,posture,com_task]
    limits = [mink.ConfigurationLimit(model,min_distance_from_limits=joint_margin_rad)]
    errors = []
    for index, target in enumerate(motion):
        data.qpos[:] = target
        mujoco.mj_forward(model,data)
        for task,name in ((root_task,'pelvis'),(torso_task,'torso_link')):
            body = data.body(name)
            quat = Rotation.from_matrix(body.xmat.reshape(3,3)).as_quat(scalar_first=True)
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(quat),body.xpos.copy()))
        posture.set_target(target)
        for _ in range(100 if index==0 else 35):
            velocity = mink.solve_ik(configuration,tasks,.01,solver='proxqp',damping=.001,
                                     limits=limits,constraints=foot_tasks,eps_abs=1e-9)
            factor = min(.5, .05/max(np.max(abs(velocity))*.01,1e-12))
            velocity *= factor
            configuration.integrate_inplace(velocity,.01)
            if np.linalg.norm(velocity)*.01 < 1e-8:
                break
        motion[index] = configuration.q
        error = []
        for task in foot_tasks:
            residual = task.compute_error(configuration)
            error.extend((np.linalg.norm(residual[:3]),np.linalg.norm(residual[3:])))
        errors.append(error)
    errors = np.array(errors)
    if errors[:, [0, 2]].max() > 5e-4 or errors[:, [1, 3]].max() > .005:
        raise ValueError(f'Cannot level and plant feet with requested root lowering: '
                         f'position={errors[:, [0, 2]].max():.6f} m, '
                         f'orientation={errors[:, [1, 3]].max():.6f} rad; '
                         f'position worst frame={np.argmax(np.max(errors[:,[0,2]],axis=1))}, '
                         f'initial={errors[0].tolist()}, final={errors[-1].tolist()}')
    # Duplicate endpoint poses give actual zero-velocity hold intervals. No
    # smoothing follows foot IK, so all samples retain their planted soles.
    begin = np.repeat(motion[:1], int(round(hold_s*sample_hz)), axis=0)
    end = np.repeat(motion[-1:], int(round(final_hold_s*sample_hz)), axis=0)
    qpos = np.concatenate((begin, motion, end))
    final_time = np.arange(len(qpos))/sample_hz
    clock = np.r_[np.zeros(len(begin)), source_clock, np.full(len(end), times[-1])]
    com, tilt = [], []
    corners = []
    for anchor in anchors.values():
        r, p = np.array(anchor['rotation']), np.array(anchor['position'])
        # Conservative rectangle inside the native capsule support footprint.
        corners.extend(np.array([[-.04,-.02,-.035],[-.04,.02,-.035],
                                 [.11,-.02,-.035],[.11,.02,-.035]]) @ r.T+p)
    hull = ConvexHull(np.asarray(corners)[:, :2])
    margins = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        com.append(data.subtree_com[1].copy())
        margins.append(np.min(-(hull.equations[:,:2] @ com[-1][:2] + hull.equations[:,2])))
        axis = data.body('torso_link').xmat.reshape(3, 3)[:, 2]
        tilt.append(np.rad2deg(np.arctan2(np.linalg.norm(axis[:2]),axis[2])))
    output.mkdir(parents=True, exist_ok=True)
    trajectory = output/'reference.npz'
    np.savez_compressed(trajectory, qpos=qpos, time_s=final_time, source_time_s=clock,
                        com_positions=np.asarray(com), torso_tilt_deg=np.asarray(tilt),
                        foot_anchor_positions=np.asarray([a['position'] for a in anchors.values()]),
                        foot_anchor_rotations=np.asarray([a['rotation'] for a in anchors.values()]),
                        source_frames=np.interp(clock, times, original['source_frames']))
    result = {'trajectory_path': str(trajectory), 'model_xml_path': str(DEFAULT_MODEL.resolve()),
              'source_path': str(source_path), 'frames': len(qpos), 'sample_hz': sample_hz,
              'motion_duration_s': duration_s, 'initial_hold_s': hold_s, 'final_hold_s': final_hold_s,
              'duration_s': float(final_time[-1]), 'source_duration_s': float(times[-1]),
              'smoothing_sigma_s': smooth_sigma_s, 'root_lowering_m': root_lowering_m,
              'stance_width_scale': stance_width_scale,
              'joint_limit_margin_rad':joint_margin_rad,
              'com_centering_weight':com_weight,
              'anchors': anchors,
              'metrics': {'max_foot_position_error_m': float(errors[:,[0,2]].max()),
                          'max_foot_orientation_error_rad': float(errors[:,[1,3]].max()),
                          'minimum_projected_com_margin_m': float(min(margins)),
                          'pelvis_drop_m': float(np.ptp(qpos[:,2])),
                          'maximum_root_position_change_m':float(np.max(np.linalg.norm(motion[:,:3]-targets[:,:3],axis=1))),
                          'maximum_torso_tilt_deg': float(max(tilt))},
              'dynamics_claim': 'Geometric conditioning only; requires contact-aware inverse dynamics and forward simulation.'}
    (output/'reference.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def smooth_conditioned_reference(source_path, output_dir, *, sigma_s=.03):
    """Remove small IK acceleration spikes, measuring the resulting foot drift.

    This explicitly relaxes exact foot planting by less than a millimetre in the
    selected candidate. Instantaneous inverse force balance is audited, while
    execution and contact consistency still require forward simulation.
    """
    from human_punching.ducking_inverse import (
        audit_trajectory, build_g1_model, smooth_trajectory,
    )
    source, output = Path(source_path).resolve(), Path(output_dir).resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError('Outputs must stay inside human-punching')
    packet = np.load(source)
    arrays = {name: packet[name].copy() for name in packet.files}
    arrays['qpos'] = smooth_trajectory(arrays['qpos'], arrays['time_s'], sigma_s)
    model, _ = build_g1_model()
    data = mujoco.MjData(model)
    foot_ids = [model.body(side+'_ankle_roll_link').id for side in ('left','right')]
    feet, com, tilt = [], [], []
    for q in arrays['qpos']:
        data.qpos[:] = q
        mujoco.mj_forward(model,data)
        feet.append(data.xpos[foot_ids].copy())
        com.append(data.subtree_com[model.body('pelvis').id].copy())
        axis = data.body('torso_link').xmat.reshape(3,3)[:,2]
        tilt.append(np.rad2deg(np.arctan2(np.linalg.norm(axis[:2]),axis[2])))
    foot_drift = float(np.max(np.linalg.norm(np.asarray(feet)-arrays['foot_anchor_positions'],axis=-1)))
    if foot_drift > .001:
        raise ValueError(f'Postfilter foot drift {foot_drift} m exceeds the chosen 1 mm conditioning bound')
    arrays['com_positions'] = np.asarray(com)
    arrays['torso_tilt_deg'] = np.asarray(tilt)
    summary, inverse = audit_trajectory(model, arrays['qpos'], arrays['time_s'],stride=1)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'reference.npz',**arrays)
    np.savez_compressed(output/'inverse_samples.npz',**inverse)
    result = {'trajectory_path': str(output/'reference.npz'), 'source_path': str(source),
              'model_xml_path': str(DEFAULT_MODEL.resolve()), 'smoothing_sigma_s':sigma_s,
              'maximum_foot_center_drift_m':foot_drift,
              'maximum_loaded_contact_gap_m':float(np.nanmax(inverse['loaded_contact_gap'])),
              'maximum_loaded_contact_speed_mps':float(np.nanmax(inverse['loaded_contact_speed'])),
              'inverse_summary':summary,
              'qualification':'Force balance only; the postfilter introduces measured submillimetre foot drift. Forward dynamics remains required.'}
    source_metadata = source.parent/'reference.json'
    if source_metadata.is_file():
        metadata = json.loads(source_metadata.read_text())
        result['conditioning_joint_limit_margin_rad'] = metadata.get('joint_limit_margin_rad')
    joint_margins = []
    for joint in range(1,model.njnt):
        values = arrays['qpos'][:,model.jnt_qposadr[joint]]
        lower,upper = model.jnt_range[joint]
        joint_margins.append(min(float(np.min(values-lower)),float(np.min(upper-values))))
    result['minimum_joint_limit_margin_rad'] = min(joint_margins)
    (output/'reference.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=4.)
    parser.add_argument('--root-lowering', type=float, default=.08)
    parser.add_argument('--joint-margin',type=float,default=.03)
    args = parser.parse_args()
    print(json.dumps(condition_reference(args.output, duration_s=args.duration,
                                         root_lowering_m=args.root_lowering,
                                         joint_margin_rad=args.joint_margin), indent=2))
