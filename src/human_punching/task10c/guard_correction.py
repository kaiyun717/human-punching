"""Bounded guard endpoint adjustments; no change to the central punch solve.

Each guard retains its own source pelvis orientation, foot placement/yaw and
upper-body posture as the starting point. Optional hook terminal waist correction
matches the recorded source torso before the guard fit. A small endpoint-only
least-squares adjustment places supporting sole boxes on the floor and centers
the mass between them. Glove clearances use the actual training primitives.
This is a pose correction, not a trajectory optimizer or a balance controller.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

from .env_cfg import ROOT, make_env_cfg
from .references import StoredReference, arrays_from_qpos
from .robot import JOINT_NAMES


def training_model():
    from mjlab.scene import Scene
    model = Scene(make_env_cfg(num_envs=1).scene, device='cpu').compile()
    model.opt.timestep = .005
    return model


def _name(model, bare, kind):
    lookup = getattr(model, kind)
    try:
        return lookup('robot/' + bare).id
    except KeyError:
        return lookup(bare).id


def sole_vertices(model, data, geom_id):
    local = np.array(list(itertools.product((-1., 1.), repeat=3))) * model.geom_size[geom_id]
    return local @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]


def pose_diagnostics(model, qpos):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    pelvis = _name(model, 'pelvis', 'body')
    feet, support = {}, []
    for side in ('left', 'right'):
        gid = _name(model, side + '_foot_sole_collision', 'geom')
        points = sole_vertices(model, data, gid)
        active = points[points[:, 2] <= .006]
        support.extend(active[:, :2])
        feet[side] = {'center_world': data.geom_xpos[gid].tolist(),
                      'rotation_xyz_rad': Rotation.from_matrix(data.geom_xmat[gid].reshape(3, 3)).as_euler('xyz').tolist(),
                      'lowest_vertex_m': float(points[:, 2].min()),
                      'near_ground_vertices': len(active)}
    outside = None
    if len(support) >= 3:
        points = np.unique(np.asarray(support), axis=0)
        if np.linalg.matrix_rank(points - points[0]) == 2:
            hull = ConvexHull(points)
            outside = float(np.max(hull.equations[:, :2] @ data.subtree_com[pelvis, :2] + hull.equations[:, 2]))
    overlaps = []
    for contact in data.contact[:data.ncon]:
        if contact.dist < -.001:
            overlaps.append({'geoms': [model.geom(int(contact.geom1)).name, model.geom(int(contact.geom2)).name],
                             'penetration_m': float(-contact.dist)})
    return {'pelvis_world': data.qpos[:3].tolist(), 'pelvis_quaternion_wxyz': data.qpos[3:7].tolist(),
            'center_of_mass_world': data.subtree_com[pelvis].tolist(), 'feet': feet,
            'com_outside_near_ground_support_m': outside,
            'physical_contact_penetrations_over_1mm': overlaps,
            'adjacent_geometry_policy': 'MuJoCo training contact exclusions/parent filtering retained; no all-pairs rejection'}


def fit_guard(model, original):
    """Fit one endpoint, preserving source pelvis rotation and waist coordinates."""
    q = np.asarray(original, dtype=float).copy()
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    feet = [_name(model, side + '_foot_sole_collision', 'geom') for side in ('left', 'right')]
    pelvis = _name(model, 'pelvis', 'body')
    torso = _name(model, 'torso_link', 'body')
    gloves = [_name(model, side + '_glove_center', 'site') for side in ('left', 'right')]
    foot_targets = data.geom_xpos[feet].copy()
    foot_targets[:, 2] = model.geom_size[feet, 2] + .001
    foot_rotations = []
    for gid in feet:
        mat = data.geom_xmat[gid].reshape(3, 3)
        foot_rotations.append(Rotation.from_euler('z', np.arctan2(mat[1, 0], mat[0, 0])))
    com_target = foot_targets[:, :2].mean(axis=0)
    leg_names = [name for name in JOINT_NAMES if any(part in name for part in ('hip_', 'knee_', 'ankle_'))]
    arm_names = [name for name in JOINT_NAMES if any(part in name for part in ('shoulder_', 'elbow_', 'wrist_'))]
    leg_joints = [_name(model, name, 'joint') for name in leg_names]
    arm_joints = [_name(model, name, 'joint') for name in arm_names]
    leg_qa = model.jnt_qposadr[leg_joints]
    arm_qa = model.jnt_qposadr[arm_joints]

    def fit_stance(current):
        indices = np.r_[np.arange(3), leg_qa]
        initial = original[indices].copy()
        bounds = np.r_[np.array([.16, .16, .24]), np.full(len(leg_qa), .75)]
        lower, upper = initial - bounds, initial + bounds
        lower[3:] = np.maximum(lower[3:], model.jnt_range[leg_joints, 0] + .001)
        upper[3:] = np.minimum(upper[3:], model.jnt_range[leg_joints, 1] - .001)

        def residual(value):
            test = current.copy()
            test[indices] = value
            data.qpos[:] = test
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            parts = [(data.geom_xpos[feet] - foot_targets).ravel() / .003,
                     (data.subtree_com[pelvis, :2] - com_target) / .04,
                     .3 * (value - initial) / bounds]
            for gid, desired in zip(feet, foot_rotations):
                actual = Rotation.from_matrix(data.geom_xmat[gid].reshape(3, 3))
                parts.append((desired.inv() * actual).as_rotvec() / .03)
            return np.concatenate(parts)

        fitted = least_squares(residual, np.clip(current[indices], lower + 1e-8, upper - 1e-8),
                               bounds=(lower, upper), max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        result = current.copy()
        result[indices] = fitted.x
        return result, {'function_evaluations': fitted.nfev, 'cost': float(fitted.cost)}

    q, stance_fit = fit_stance(q)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    glove_target = data.site_xpos[gloves].copy()
    torso_rotation = data.xmat[torso].reshape(3, 3)
    head_id = _name(model, 'head_collision', 'geom')
    head = data.geom_xpos[head_id].copy()
    # Preserve each family's initial hint, projecting only insufficient space.
    # The radii here come from the actual compiled head/glove primitives.
    head_clearance = float(model.geom_size[head_id, 0]) + .08 + .015
    for _ in range(3):
        separation = np.dot(glove_target[0] - glove_target[1], torso_rotation[:, 1])
        spread = max(.18 - separation, 0) / 2
        glove_target[0] += spread * torso_rotation[:, 1]
        glove_target[1] -= spread * torso_rotation[:, 1]
        for i in range(2):
            vector = glove_target[i] - head
            length = np.linalg.norm(vector)
            if length < head_clearance:
                glove_target[i] = head + vector * head_clearance / max(length, 1e-6)
    glove_rotations = [Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)) for site in gloves]
    pairs = [('left_glove_collision', 'right_glove_collision'),
             ('left_wrist_collision', 'right_wrist_collision')]
    for side, other in (('left', 'right'), ('right', 'left')):
        pairs.extend([(side + '_glove_collision', name) for name in
                      ('head_collision', 'torso_collision', other + '_wrist_collision', other + '_elbow_collision')])
    pairs = [(_name(model, a, 'geom'), _name(model, b, 'geom')) for a, b in pairs]
    arm_initial = q[arm_qa].copy()
    lower = np.maximum(arm_initial - .55, model.jnt_range[arm_joints, 0] + .001)
    upper = np.minimum(arm_initial + .55, model.jnt_range[arm_joints, 1] - .001)

    def arm_residual(value):
        test = q.copy()
        test[arm_qa] = value
        data.qpos[:] = test
        mujoco.mj_kinematics(model, data)
        parts = [(data.site_xpos[gloves] - glove_target).ravel() / .025,
                 .4 * (value - arm_initial) / .55]
        for site, desired in zip(gloves, glove_rotations):
            actual = Rotation.from_matrix(data.site_xmat[site].reshape(3, 3))
            parts.append((desired.inv() * actual).as_rotvec() / .6)
        distances = [mujoco.mj_geomDistance(model, data, a, b, .1, None) for a, b in pairs]
        parts.append(np.maximum(.006 - np.asarray(distances), 0) / .008)
        return np.concatenate(parts)

    fitted = least_squares(arm_residual, np.clip(arm_initial, lower + 1e-8, upper - 1e-8),
                           bounds=(lower, upper), max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    q[arm_qa] = fitted.x
    q, final_stance = fit_stance(q)
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    floor_lift = max(.001 - min(sole_vertices(model, data, gid)[:, 2].min() for gid in feet), 0.)
    q[2] += floor_lift
    # No head/waist/pelvis-orientation substitution, including source yaw/lean.
    return q, {'stance_fit': stance_fit, 'arm_clearance_fit': {'function_evaluations': fitted.nfev,
                  'cost': float(fitted.cost)}, 'final_stance_fit': final_stance,
               'maximum_joint_adjustment_rad': float(np.max(np.abs(q[7:] - original[7:]))),
               'maximum_leg_adjustment_rad': float(np.max(np.abs(q[leg_qa] - original[leg_qa]))),
               'maximum_arm_adjustment_rad': float(np.max(np.abs(q[arm_qa] - original[arm_qa]))),
               'final_uniform_floor_lift_m': float(floor_lift),
               'root_position_adjustment_m': (q[:3] - original[:3]).tolist()}


def _blend(a, b, fraction):
    fraction = float(np.clip(fraction, 0., 1.))
    result = (1 - fraction) * a + fraction * b
    result[3:7] = Slerp([0, 1], Rotation.from_quat(np.stack([a[3:7], b[3:7]]), scalar_first=True))([fraction]).as_quat(scalar_first=True)[0]
    return result


def source_terminal_torso_seed(reference, model):
    """Match the recorded hook terminal torso with a fixed-pelvis ZXY waist solve."""
    from human_punching.motion import JOINT_NAMES as SOURCE_NAMES, load_crop
    from .retargeting import BIND_TO_ROBOT

    metadata = reference.metadata
    if metadata['family'] not in ('left_hook', 'right_hook'):
        raise ValueError('Source terminal torso correction is limited to hook references')
    source = metadata['source']
    clip = load_crop(Path(source['source_path']).parents[1], metadata['family'],
                     padding_s=source['padding_requested_s'])
    if clip.metadata['source_sha256'] != source['source_sha256']:
        raise ValueError('Terminal torso correction source identity differs from the reference')
    frame = float(reference.arrays['source_frames'][-1])
    scene = np.asarray(metadata['adapter']['source_to_reference_rotation'])
    source_rotations = scene @ clip.global_rotations[:, SOURCE_NAMES.index('Spine1')] @ BIND_TO_ROBOT.T
    desired = Slerp(clip.source_frames, Rotation.from_matrix(source_rotations))([frame])[0]
    original = reference.arrays['qpos'][-1]
    pelvis = Rotation.from_quat(original[3:7], scalar_first=True)
    waist = (pelvis.inv() * desired).as_euler('ZXY')
    names = ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint')
    joints = [_name(model, name, 'joint') for name in names]
    addresses = model.jnt_qposadr[joints]
    limits = model.jnt_range[joints]
    if np.any(waist <= limits[:, 0] + .001) or np.any(waist >= limits[:, 1] - .001):
        raise ValueError('Recorded terminal torso cannot be matched within selected-H2 waist limits')
    seed = original.copy()
    seed[addresses] = waist
    data = mujoco.MjData(model)
    data.qpos[:] = seed
    mujoco.mj_kinematics(model, data)
    actual = Rotation.from_matrix(data.xmat[_name(model, 'torso_link', 'body')].reshape(3, 3))
    error = float(np.rad2deg((desired.inv() * actual).magnitude()))
    if error > 1e-7:
        raise ValueError(f'Unexpected selected-H2 waist chain: terminal torso error {error} degrees')
    return seed, {
        'method': 'analytic fixed-pelvis ZXY waist match to recorded Spine1 with existing bind correction',
        'source_frame': frame, 'source_sha256': source['source_sha256'],
        'waist_joint_names': list(names), 'waist_before_rad': original[addresses].tolist(),
        'waist_after_rad': waist.tolist(), 'waist_limits_rad': limits.tolist(),
        'maximum_waist_adjustment_rad': float(np.abs(seed[addresses] - original[addresses]).max()),
        'target_torso_world_quaternion_wxyz': desired.as_quat(scalar_first=True).tolist(),
        'source_torso_tilt_deg': float(np.rad2deg(np.arccos(np.clip(desired.as_matrix()[2, 2], -1, 1)))),
        'analytic_orientation_error_deg': error,
        'scope': 'hook terminal guard and existing final recovery blend only; pelvis and central punch unchanged',
    }


def correct_reference(input_npz, output_npz, *, attack_blend_s=.24, recovery_blend_s=.30,
                      source_terminal_torso=False):
    input_npz, output_npz = Path(input_npz), Path(output_npz).resolve()
    output_npz.relative_to(ROOT / 'outputs/task10c')
    reference = StoredReference.load(input_npz)
    original = reference.arrays['qpos']
    model = training_model()
    launch, launch_fit = fit_guard(model, original[0])
    terminal_seed, torso_details = original[-1], None
    if source_terminal_torso and reference.metadata['family'] in ('left_hook', 'right_hook'):
        terminal_seed, torso_details = source_terminal_torso_seed(reference, model)
    terminal, terminal_fit = fit_guard(model, terminal_seed)
    if torso_details is not None:
        terminal_fit['maximum_joint_adjustment_rad'] = float(np.abs(terminal[7:] - original[-1, 7:]).max())
        data = mujoco.MjData(model)
        data.qpos[:] = terminal
        mujoco.mj_kinematics(model, data)
        final_rotation = Rotation.from_matrix(data.xmat[_name(model, 'torso_link', 'body')].reshape(3, 3))
        desired = Rotation.from_quat(torso_details['target_torso_world_quaternion_wxyz'], scalar_first=True)
        torso_details['final_orientation_error_deg'] = float(np.rad2deg((desired.inv() * final_rotation).magnitude()))
    qpos = original.copy()
    times = reference.time_s
    for i, time in enumerate(times):
        if time < attack_blend_s:
            phase = np.clip((time - .02) / (attack_blend_s - .02), 0, 1)
            weight = phase ** 3 * (10 - 15 * phase + 6 * phase ** 2)
            qpos[i] = _blend(launch, original[i], weight)
        if time > times[-1] - recovery_blend_s:
            phase = np.clip((time - times[-1] + recovery_blend_s) / (recovery_blend_s - .04), 0, 1)
            weight = phase ** 3 * (10 - 15 * phase + 6 * phase ** 2)
            qpos[i] = _blend(original[i], terminal, weight)
    arrays = arrays_from_qpos(times, qpos)
    arrays['source_frames'] = reference.arrays['source_frames'].copy()
    details = {'method': 'bounded independent guard-pose correction and endpoint quintic blending',
               'central_punch_unchanged_interval_s': [attack_blend_s, float(times[-1] - recovery_blend_s)],
               'launch_guard_reference_hold_s': .02, 'terminal_guard_reference_hold_s': .04,
               'launch_clock': 'unchanged t=0; every initial hold/blend millisecond counts toward commanded TTC',
               'preserved_endpoint_coordinates': 'pelvis orientation including lean/yaw, waist, head; family foot XY/yaw used as fit targets',
               'source_frame_annotation': 'source_frames remain provenance; endpoint poses are intentionally modified, not exact source samples',
               'launch': {'before': pose_diagnostics(model, original[0]), 'after': pose_diagnostics(model, qpos[0]), 'fit': launch_fit},
               'recovery': {'before': pose_diagnostics(model, original[-1]), 'after': pose_diagnostics(model, qpos[-1]), 'fit': terminal_fit}}
    if torso_details is not None:
        details['source_terminal_torso'] = torso_details
        details['preserved_endpoint_coordinates'] = ('pelvis orientation including lean/yaw and head; launch waist unchanged; '
            'hook terminal waist matches recorded source torso; family foot XY/yaw used as fit targets')
    metadata = {**reference.metadata, 'guard_correction': details,
                'terminal_guard_interval_s': [float(times[-1] - .04), float(times[-1])],
                'guard_parent_npz': str(input_npz.resolve()), 'guard_parent_sha256': hashlib.sha256(input_npz.read_bytes()).hexdigest()}
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    output_npz.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-terminal-torso', action='store_true',
                        help='For hooks only, match the recorded terminal torso through the selected-H2 waist before fitting the guard.')
    args = parser.parse_args()
    print(json.dumps(correct_reference(args.input, args.output,
                                      source_terminal_torso=args.source_terminal_torso), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
