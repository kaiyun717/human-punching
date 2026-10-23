"""One offline contact-directed H2 jab adaptation; no policy or runtime changes.

Reuse Task09 targeting's quintic Cartesian deformation and scalar_reference's
bounded Jacobian least-squares idea. All solves/FK use the selected H2 model.
Mannequin coordinates, controller values and policies are never transferred.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import BPoly
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation, Slerp

from human_punching.targeting import _smoothstep
from .references import StoredReference, arrays_from_qpos
from .robot import H2_XML, JOINT_NAMES, ROOT
from .targets import TargetPool, sha256


def write_json(path, value):
    path = Path(path).resolve()
    path.relative_to(ROOT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def representative_target(pool_path):
    pool = TargetPool.load(pool_path)
    ids, points = pool.snapshot_ids('train'), pool.points('train', 'jab')
    median = np.median(points, axis=0)
    index = int(np.argmin(np.linalg.norm(points - median, axis=1)))
    return {'snapshot_id': int(ids[index]), 'target_world': points[index].tolist(),
            'split': 'train', 'selection': 'actual training snapshot nearest XYZ median',
            'training_median_world': median.tolist(), 'training_count': len(ids),
            'median_distance_m': float(np.linalg.norm(points[index] - median)),
            'pool_sha256': sha256(Path(pool_path)), 'height_adjustment_m': 0.,
            'head_radius_m': .06, 'glove_radius_m': .08, 'encounter_radius_m': .14,
            'target_is_non_solid': True}


def native_paths(model, poses):
    data = mujoco.MjData(model)
    result = {key: [] for key in ('left', 'right', 'left_rotation', 'right_rotation',
                                  'left_elbow', 'left_shoulder', 'left_wrist', 'head', 'head_rotation')}
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        for side in ('left', 'right'):
            result[side].append(data.site(f'{side}_glove_center').xpos.copy())
            result[side + '_rotation'].append(data.site(f'{side}_glove_center').xmat.reshape(3, 3).copy())
        result['left_elbow'].append(data.body('left_elbow_link').xpos.copy())
        result['left_shoulder'].append(data.body('left_shoulder_yaw_link').xpos.copy())
        result['left_wrist'].append(data.body('left_wrist_yaw_link').xpos.copy())
        result['head'].append(data.geom('head_collision').xpos.copy())
        result['head_rotation'].append(data.body('head_yaw_link').xmat.reshape(3, 3).copy())
    return {key: np.asarray(value) for key, value in result.items()}


def axis_rotation(source, direction, strength):
    """Rotate hand continuation +X toward desired forearm; preserve source roll."""
    a = source[:, 0]
    b = direction / max(np.linalg.norm(direction), 1e-12)
    cross = np.cross(a, b)
    angle = np.arctan2(np.linalg.norm(cross), np.dot(a, b))
    vector = cross / max(np.linalg.norm(cross), 1e-12) * angle * strength
    return Rotation.from_rotvec(vector).as_matrix() @ source


def solve_reference(model, ref, target):
    times, source = ref.time_s, ref.arrays['qpos']
    paths = native_paths(model, source)
    annotated = ref.metadata['strike_time_s']
    window = np.flatnonzero((times >= annotated - .16) & (times <= annotated + .16))
    anchor = int(window[np.argmax(paths['left'][window, 0])])
    center_time = float(times[anchor])
    # Deform the entire approach and withdrawal, retaining this jab's guards.
    start, end = .12, 1.22
    envelope = np.where(times <= center_time,
                        _smoothstep((times - start) / (center_time - start)),
                        _smoothstep((end - times) / (end - center_time)))
    direction = np.asarray(target) - paths['left'][0]
    direction[2] = 0.
    direction /= np.linalg.norm(direction)
    # Inside the unchanged encounter sphere; not Task09's stopping-short mode.
    radius = .04
    contact_point = np.asarray(target) - radius * direction
    correction = contact_point - paths['left'][anchor]
    # Direct translation retained the source's dip/bump, producing a downward
    # entry and a bent reaching arm. One local refinement replaces that middle
    # path by C2 quintic Hermite approach/withdrawal segments, matching the H2
    # source position/velocity/acceleration at both ends. The extension center
    # has zero velocity/acceleration, a turnaround rather than a hold plateau.
    desired = paths['left'].copy()
    velocity = np.gradient(paths['left'], times, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
    begin, finish = int(np.argmin(abs(times-start))), int(np.argmin(abs(times-end)))
    curve = BPoly.from_derivatives([times[begin], center_time, times[finish]],
        [[paths['left'][begin], velocity[begin], acceleration[begin]],
         [contact_point, np.zeros(3), np.zeros(3)],
         [paths['left'][finish], velocity[finish], acceleration[finish]]])
    desired[begin:finish+1] = curve(times[begin:finish+1])
    reach = desired - paths['left_shoulder']
    reach /= np.maximum(np.linalg.norm(reach, axis=1, keepdims=True), 1e-12)
    upper_length = np.linalg.norm(paths['left_elbow']-paths['left_shoulder'], axis=1)
    extended_elbow = paths['left_shoulder'] + upper_length[:, None] * reach
    elbow_desired = paths['left_elbow'] + envelope[:, None] * (extended_elbow-paths['left_elbow'])
    desired_rotation = np.stack([axis_rotation(r, p-e, w) for r, p, e, w in
                                 zip(paths['left_rotation'], desired, elbow_desired, envelope)])
    # Guard is expressed in the moving H2 head frame, not fixed world space.
    offsets = np.einsum('nij,nj->ni', paths['head_rotation'].transpose(0, 2, 1),
                        paths['right'] - paths['head'])
    guard_blend = _smoothstep(times / times[-1])
    guard_offset = offsets[0] + guard_blend[:, None] * (offsets[-1] - offsets[0])
    guard_position = paths['head'] + np.einsum('nij,nj->ni', paths['head_rotation'], guard_offset)
    right_desired = paths['right'] + envelope[:, None] * (guard_position - paths['right'])
    # The existing source guard rotations remain the soft right-fist target.
    joint_names = [n for n in JOINT_NAMES if any(x in n for x in ('shoulder', 'elbow', 'wrist'))]
    joint_ids = np.array([model.joint(n).id for n in joint_names])
    qa, dofs = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    lower, upper = model.jnt_range[joint_ids].T
    result, previous = source.copy(), np.zeros(len(qa))
    data = mujoco.MjData(model)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    iterations = []
    for i in range(len(times)):
        if envelope[i] == 0:
            previous[:] = 0
            iterations.append(0)
            continue
        pose = source[i].copy()
        pose[qa] = np.clip(pose[qa] + previous, lower, upper)
        for iteration in range(80):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            matrices, errors = [], []
            for side, goal, rotation in (('left', desired[i], desired_rotation[i]),
                                         ('right', right_desired[i], paths['right_rotation'][i])):
                site = model.site(f'{side}_glove_center').id
                mujoco.mj_jacSite(model, data, jp, jr, site)
                matrices.extend([jp[:, dofs].copy(), .12 * jr[:, dofs].copy()])
                errors.extend([goal - data.site_xpos[site], .12 * Rotation.from_matrix(
                    rotation @ data.site_xmat[site].reshape(3, 3).T).as_rotvec()])
            elbow = model.body('left_elbow_link').id
            mujoco.mj_jacBody(model, data, jp, jr, elbow)
            matrices.append(.30 * jp[:, dofs].copy())
            errors.append(.30 * (elbow_desired[i] - data.xpos[elbow]))
            elbow_jac = jp[:, dofs].copy()
            wrist = model.body('left_wrist_yaw_link').id
            mujoco.mj_jacBody(model, data, jp, jr, wrist)
            forearm = data.xpos[wrist] - data.xpos[elbow]
            length = np.linalg.norm(forearm)
            forearm /= length
            fist = data.xmat[wrist].reshape(3, 3)[:, 0]
            skew = np.array([[0., -fist[2], fist[1]], [fist[2], 0., -fist[0]],
                             [-fist[1], fist[0], 0.]])
            direction_jac = (np.eye(3) - np.outer(forearm, forearm)) @ (jp[:, dofs] - elbow_jac) / length
            # Soft geometric hand/forearm alignment addresses the observed
            # lateral wrist kink; no copied joint-angle prescription or gate.
            align_weight = .15 * envelope[i]
            matrices.append(align_weight * (-skew @ jr[:, dofs] - direction_jac))
            errors.append(align_weight * (forearm - fist))
            matrices.extend([.025 * np.eye(len(qa)), .035 * np.eye(len(qa)), .025 * np.eye(len(qa))])
            errors.extend([.025 * (source[i, qa] - pose[qa]),
                           .035 * (source[i, qa] + previous - pose[qa]), np.zeros(len(qa))])
            # Bounds are inside the solve: never solve for an impossible step
            # and then clip all linked tasks at a saturated joint.
            low = np.maximum(-.12, lower - pose[qa])
            high = np.minimum(.12, upper - pose[qa])
            delta = lsq_linear(np.vstack(matrices), np.concatenate(errors),
                               bounds=(low, high), method='bvls', max_iter=50, tol=1e-8).x
            pose[qa] += delta
            if np.max(np.abs(delta)) < 2e-6:
                break
        result[i, qa] = pose[qa]
        previous = pose[qa] - source[i, qa]
        iterations.append(iteration + 1)
    design = {'method': 'Task09 smooth Cartesian deformation/IK adapted to H2, with C2 quintic Hermite approach and withdrawal',
              'source_annotation_time_s': annotated, 'strike_center_time_s': center_time,
              'deformation_interval_s': [start, end], 'contact_directed_center_radius_m': radius,
              'contact_point_world': contact_point.tolist(), 'path_correction_world_m': correction.tolist(),
              'modified_joint_names': joint_names, 'root_waist_legs_head': 'source trajectory unchanged',
              'inactive_guard': 'soft H2 head-relative guard position, source fist orientation',
              'orientation': 'soft local +X continuation aligned toward desired forearm; source roll retained',
              'solver_weights': {'glove_position': 1., 'glove_orientation': .12, 'elbow_position': .30,
                                 'source_posture': .025, 'previous_correction': .035, 'step_damping': .025,
                                 'fist_forearm_direction_alignment': .15},
              'step_bound_rad': .12, 'max_iterations': max(iterations),
              'focused_refinement_reason': 'Draft translated path had a downward dip and reduced elbow extension; smooth the path, align elbow with reach and reduce measured lateral wrist kink with a soft native geometric alignment objective',
              'no_speed_or_impact_objective': True}
    return result, design, {'active_glove_target': desired, 'inactive_glove_target': right_desired,
                            'active_elbow_target': elbow_desired, 'deformation_envelope': envelope}


def comparison_metrics(ref, target, model):
    times = np.arange(0, ref.time_s[-1] + 1e-9, .005)
    q = np.stack([np.interp(times, ref.time_s, col) for col in ref.arrays['qpos'].T], axis=-1)
    q[:, 3:7] = Slerp(ref.time_s, Rotation.from_quat(ref.arrays['qpos'][:, 3:7], scalar_first=True))(times).as_quat(scalar_first=True)
    paths = native_paths(model, q)
    distance = np.linalg.norm(paths['left'] - target, axis=1)
    entries = np.flatnonzero(distance <= .14)
    first = int(entries[0]) if len(entries) else None
    closest = int(np.argmin(distance))
    velocity = np.gradient(paths['left'], times, axis=0, edge_order=2)
    head_gap = np.linalg.norm(paths['right'] - paths['head'], axis=1)
    at = first if first is not None else closest
    fist = paths['left_rotation'][at, :, 0]
    forearm = paths['left_wrist'][at] - paths['left_elbow'][at]
    forearm /= np.linalg.norm(forearm)
    ray = target - paths['left'][at]
    ray /= np.linalg.norm(ray)
    angle = lambda a, b: float(np.degrees(np.arccos(np.clip(np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12), -1, 1))))
    hinge = np.array([model.joint(n).id for n in JOINT_NAMES])
    qa = model.jnt_qposadr[hinge]
    lower, upper = model.jnt_range[hinge].T
    return {'first_entry_s': None if first is None else float(times[first]),
            'closest_approach_s': float(times[closest]), 'closest_center_distance_m': float(distance[closest]),
            'strike_center_time_s': ref.metadata['strike_time_s'],
            'first_entry_sampling_s': .005, 'sampling_note': 'native H2 FK of interpolated stored poses, not dynamics',
            'time_inside_encounter_region_s': float(len(entries) * .005),
            'event_or_closest_glove_position': paths['left'][at].tolist(),
            'event_or_closest_glove_speed_m_s': float(np.linalg.norm(velocity[at])),
            'event_or_closest_approach_to_target_angle_deg': angle(velocity[at], ray),
            'event_or_closest_fist_to_forearm_angle_deg': angle(fist, forearm),
            'event_or_closest_fist_to_target_angle_deg': angle(fist, ray),
            'inactive_glove_to_head_center_range_m': [float(head_gap.min()), float(head_gap.max())],
            'maximum_joint_limit_excess_rad': float(np.maximum(np.maximum(lower-q[:, qa], q[:, qa]-upper), 0).max()),
            'maximum_stored_joint_speed_rad_s': float(np.abs(ref.arrays['joint_vel']).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/task10c/references/jab.npz')
    parser.add_argument('--pool', type=Path, default=ROOT/'outputs/task10c/setup/target_pool.npz')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.relative_to(ROOT.resolve())
    if (out/'candidate/jab.npz').exists():
        raise FileExistsError('Preserve prior candidate; choose a new explicit output directory')
    ref = StoredReference.load(args.source)
    target = representative_target(args.pool)
    write_json(out/'target.json', target)
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    poses, design, goals = solve_reference(model, ref, np.array(target['target_world']))
    arrays = arrays_from_qpos(ref.time_s.copy(), poses)
    arrays['source_frames'] = ref.arrays['source_frames'].copy()
    metadata = deepcopy(ref.metadata)
    metadata['strike_time_s'] = design['strike_center_time_s']
    metadata['strike_center_time_s'] = design['strike_center_time_s']
    metadata['source_strike_annotation_time_s'] = design['source_annotation_time_s']
    metadata['annotation_rationale'] = ('Source frames remain lineage, not exact adapted poses. '
        'strike_time_s/strike_center_time_s identify the nominal extension center; '
        'nominal_first_entry_s is measured separately by native H2 FK at 5 ms resolution.')
    metadata['recovery_duration_s'] = float(ref.time_s[-1] - metadata['strike_time_s'])
    metadata['withdrawal_interval_s'] = [metadata['strike_time_s'], float(ref.time_s[-1])]
    metadata['nominal_target_adaptation'] = design | {'target': target, 'source_npz_sha256': sha256(args.source),
        'source_npz': str(args.source.resolve()), 'asset_sha256': sha256(H2_XML),
        'kinematic_only': True, 'reference_not_selected_for_training': True,
        'dependent_arrays': 'recomputed from native H2 qpos via arrays_from_qpos; no mannequin coordinates'}
    candidate = StoredReference(arrays, metadata)
    before = comparison_metrics(ref, np.asarray(target['target_world']), model)
    after = comparison_metrics(candidate, np.asarray(target['target_world']), model)
    metadata['nominal_first_entry_s'] = after['first_entry_s']
    metadata['nominal_closest_approach_s'] = after['closest_approach_s']
    metadata['post_entry_motion_duration_s'] = (None if after['first_entry_s'] is None
        else float(ref.time_s[-1] - after['first_entry_s']))
    metadata['timing_anchor_status'] = 'strike_time_s remains strike-center annotation; nominal_first_entry_s is separate. Runtime timing-anchor adoption requires an explicit next experiment.'
    candidate_dir = out/'candidate'; candidate_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(candidate_dir/'jab.npz', **arrays)
    write_json(candidate_dir/'jab.json', metadata)
    np.savez_compressed(out/'design_targets.npz', time_s=ref.time_s, **goals)
    nonarm = [i for i, n in enumerate(JOINT_NAMES) if n not in design['modified_joint_names']]
    comparison = {'before': before, 'after': after, 'design': design,
        'target': target, 'root_qpos_unchanged': bool(np.array_equal(poses[:, :7], ref.arrays['qpos'][:, :7])),
        'nonarm_joint_poses_unchanged': bool(np.array_equal(arrays['joint_pos'][:, nonarm], ref.arrays['joint_pos'][:, nonarm])),
        'launch_pose_unchanged': bool(np.array_equal(poses[0], ref.arrays['qpos'][0])),
        'terminal_pose_unchanged': bool(np.array_equal(poses[-1], ref.arrays['qpos'][-1])),
        'candidate_sha256': sha256(candidate_dir/'jab.npz'), 'source_sha256': sha256(args.source),
        'generator_sha256': sha256(Path(__file__)), 'runtime_or_reward_changes': False,
        'training_updates': 0, 'physics_steps': 0}
    write_json(out/'comparison.json', comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
