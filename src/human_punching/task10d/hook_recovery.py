"""One source-specific right-hook recovery repair; no physical H2 controller.

The recorded strike and early withdrawal stay intact. A bounded native pose
fit constructs a source-related terminal guard, then one coupled trajectory
fit changes only the right arm after the declared source-time boundary.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from human_punching.targeting import _smoothstep
from human_punching.task10c.references import StoredReference, arrays_from_qpos
from human_punching.task10c.robot import H2_XML, JOINT_NAMES, ROOT
from human_punching.task10c.targets import sha256


def write_json(path, value):
    path = Path(path).resolve()
    path.relative_to((ROOT/'outputs/task10d/phase2/variants/right_hook_repair01/recovery01').resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def arm_map(model):
    names = [n for n in JOINT_NAMES if n.startswith('right_') and any(
        x in n for x in ('shoulder', 'elbow', 'wrist'))]
    ids = np.array([model.joint(n).id for n in names])
    return names, model.jnt_qposadr[ids], model.jnt_dofadr[ids], model.jnt_range[ids, 0], model.jnt_range[ids, 1]


def geometry(model, qposes):
    data = mujoco.MjData(model)
    out = {k: [] for k in ('glove', 'inactive_glove', 'elbow', 'head', 'head_rotation', 'fist_forearm_angle_deg')}
    for q in qposes:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        glove = data.site('right_glove_center').xpos.copy()
        out['glove'].append(glove)
        out['inactive_glove'].append(data.site('left_glove_center').xpos.copy())
        out['elbow'].append(data.body('right_elbow_link').xpos.copy())
        out['head'].append(data.geom('head_collision').xpos.copy())
        out['head_rotation'].append(data.body('head_yaw_link').xmat.reshape(3, 3).copy())
        v = data.body('right_wrist_yaw_link').xpos-data.body('right_elbow_link').xpos
        v /= np.linalg.norm(v)
        axis = data.body('right_wrist_yaw_link').xmat.reshape(3, 3)[:, 0]
        out['fist_forearm_angle_deg'].append(float(np.degrees(np.arccos(np.clip(np.dot(v, axis), -1, 1)))))
    return {k: np.asarray(v) for k, v in out.items()}


def pose_tasks(model, data, dofs, glove_target, elbow_target, config, alignment_weight):
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    site = model.site('right_glove_center').id
    elbow = model.body('right_elbow_link').id
    wrist = model.body('right_wrist_yaw_link').id
    mujoco.mj_jacSite(model, data, jp, jr, site)
    glove_jac = jp[:, dofs].copy()
    matrices, errors = [glove_jac], [glove_target-data.site_xpos[site]]
    mujoco.mj_jacBody(model, data, jp, jr, elbow)
    elbow_jac = jp[:, dofs].copy()
    matrices.append(.14*elbow_jac)
    errors.append(.14*(elbow_target-data.xpos[elbow]))
    mujoco.mj_jacBody(model, data, jp, jr, wrist)
    forearm = data.xpos[wrist]-data.xpos[elbow]
    length = np.linalg.norm(forearm)
    forearm /= length
    fist = data.xmat[wrist].reshape(3, 3)[:, 0]
    skew = np.array([[0., -fist[2], fist[1]], [fist[2], 0., -fist[0]], [-fist[1], fist[0], 0.]])
    direction_jac = (np.eye(3)-np.outer(forearm, forearm))@(jp[:, dofs]-elbow_jac)/length
    matrices.append(alignment_weight*(-skew@jr[:, dofs]-direction_jac))
    errors.append(alignment_weight*(forearm-fist))
    # Two explicit geometric preferences, not a generic collision framework.
    for center, margin in ((data.geom('head_collision').xpos, config['head_glove_center_preference_m']),
                           (data.site('left_glove_center').xpos, config['glove_pair_center_preference_m'])):
        ray = data.site_xpos[site]-center
        distance = np.linalg.norm(ray)
        if distance < margin:
            matrices.append(1.5*(ray/max(distance, 1e-12))[None, :]@glove_jac)
            errors.append(np.array([1.5*(margin-distance)]))
        else:
            matrices.append(np.zeros((1, len(dofs))))
            errors.append(np.zeros(1))
    return np.vstack(matrices), np.concatenate(errors)


def fit_terminal_guard(model, source, qa, dofs, lower, upper, target, config):
    data = mujoco.MjData(model)
    pose = source[-1].copy()
    # Compare source-related seed choices explicitly; the raw final shoulder
    # lies near three limits, but may admit a local wrist/arm correction.
    seed_choice = config.get('terminal_seed', 'event_initial')
    if seed_choice == 'event_initial':
        pose[qa] = source[0, qa]
    elif seed_choice != 'raw_end':
        raise ValueError(seed_choice)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    elbow_goal = data.body('right_elbow_link').xpos.copy()
    seed = pose[qa].copy()
    for iteration in range(100):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        matrix, error = pose_tasks(model, data, dofs, target, elbow_goal, config,
                                   config['fist_forearm_alignment_weight'])
        matrix = np.vstack((matrix, .025*np.eye(len(qa)), .025*np.eye(len(qa))))
        error = np.concatenate((error, .025*(seed-pose[qa]), np.zeros(len(qa))))
        delta = lsq_linear(matrix, error, bounds=(np.maximum(-.12, lower-pose[qa]),
                           np.minimum(.12, upper-pose[qa])), method='bvls', tol=1e-8, max_iter=50).x
        pose[qa] += delta
        if np.max(abs(delta)) < 2e-6:
            break
    return pose, {'seed': seed_choice,
        'iterations': iteration+1, 'last_step_rad': float(np.max(abs(delta))),
        'terminal_arm_delta_from_raw_rad': (pose[qa]-source[-1, qa]).tolist()}


def repair(model, ref, config):
    source, times = ref.arrays['qpos'], ref.time_s
    names, qa, dofs, lower, upper = arm_map(model)
    begin = int(np.argmin(abs(times-config['preserve_through_s'])))
    original = geometry(model, source)
    ramp = _smoothstep((times-times[begin])/(config['offset_ramp_complete_s']-times[begin]))
    offset = np.einsum('nij,j->ni', original['head_rotation'], np.asarray(config['glove_offset_head_frame_m']))
    desired = original['glove']+ramp[:, None]*offset
    endpoint, endpoint_info = fit_terminal_guard(model, source, qa, dofs, lower, upper, desired[-1], config)
    end_geometry = geometry(model, endpoint[None, :])
    whole_ramp = _smoothstep((times-times[begin])/(times[-1]-times[begin]))
    # Elbow recovery also unwinds toward the derived end guard; preserve its
    # source route early instead of forcing all arm coordinates to one pose.
    elbow_desired = original['elbow']+.35*ramp[:, None]*offset
    elbow_desired += whole_ramp[:, None]*(end_geometry['elbow'][0]-elbow_desired[-1])
    indices = np.arange(begin+1, len(times))
    n, width = len(indices), len(qa)
    endpoint_delta = endpoint[qa]-source[-1, qa]
    x = whole_ramp[indices, None]*endpoint_delta
    lo = np.maximum(lower-source[indices][:, qa], -config['maximum_correction_rad']*whole_ramp[indices, None])
    hi = np.minimum(upper-source[indices][:, qa], config['maximum_correction_rad']*whole_ramp[indices, None])
    x = np.clip(x, lo, hi)
    # Two fixed zero-correction prefix samples enforce the departure shape;
    # the endpoint is free to retain its corrected guard.
    eye = np.eye(n+2)
    d1 = sparse.kron(sparse.csr_matrix(np.diff(eye, axis=0)[:, 2:]), sparse.eye(width))
    d2 = sparse.kron(sparse.csr_matrix(np.diff(eye, n=2, axis=0)[:, 2:]), sparse.eye(width))
    identity = sparse.eye(n*width)
    target_delta = whole_ramp[indices, None]*endpoint_delta
    terminal_selector = sparse.csr_matrix((np.ones(width), (np.arange(width), np.arange((n-1)*width, n*width))), shape=(width, n*width))
    data = mujoco.MjData(model)
    history = []
    for iteration in range(config['max_trajectory_iterations']):
        blocks, error_blocks = [], []
        for j, i in enumerate(indices):
            data.qpos[:] = source[i]
            data.qpos[qa] += x[j]
            mujoco.mj_forward(model, data)
            matrix, error = pose_tasks(model, data, dofs, desired[i], elbow_desired[i], config,
                config['fist_forearm_alignment_weight']*ramp[i])
            blocks.append(sparse.csr_matrix(matrix))
            error_blocks.append(error)
        flat = x.ravel()
        w1, w2 = config['correction_first_difference_weight'], config['correction_second_difference_weight']
        matrix = sparse.vstack([sparse.block_diag(blocks), .02*identity, w1*d1, w2*d2,
            .15*terminal_selector, .025*identity], format='csr')
        error = np.concatenate([*error_blocks, .02*(target_delta.ravel()-flat),
            -w1*(d1@flat), -w2*(d2@flat), .15*(endpoint_delta-x[-1]), np.zeros_like(flat)])
        delta = lsq_linear(matrix, error, bounds=(np.maximum(-.10, lo.ravel()-flat),
            np.minimum(.10, hi.ravel()-flat)), method='trf', lsq_solver='lsmr',
            lsmr_tol=1e-6, max_iter=60, tol=1e-7).x
        x += delta.reshape(n, width)
        step = float(np.max(abs(delta)))
        history.append({'iteration': iteration+1, 'maximum_step_rad': step,
                        'linearized_objective_squared': float(np.dot(error, error))})
        if step < 2e-5:
            break
    result = source.copy()
    result[np.ix_(indices, qa)] += x
    return result, {'method': 'source-specific bounded terminal guard and coupled right-arm recovery fit',
        'preserved_through_source_s': float(times[begin]), 'right_arm_joint_names': names,
        'head_frame_glove_offset_m': config['glove_offset_head_frame_m'],
        'offset_ramp_complete_s': config['offset_ramp_complete_s'],
        'endpoint_fit': endpoint_info, 'trajectory_iterations': len(history),
        'trajectory_last_step_rad': history[-1]['maximum_step_rad'],
        'termination': 'small_step' if history[-1]['maximum_step_rad']<2e-5 else 'iteration_cap',
        'maximum_arm_correction_rad': float(abs(x).max()),
        'final_arm_correction_rad': x[-1].tolist(),
        'terminal_guard': 'derived source-related posture; correction remains nonzero through final guard',
        'source_frames_role': 'original recorded event/frame lineage retained; modified recovery is derived, not a newly recorded event'}, history


def review(model, reference, boundary):
    times = np.arange(0., float(reference.time_s[-1])+1e-9, .005)
    poses = reference.sample(times)['qpos']
    g = geometry(model, poses)
    gap = np.linalg.norm(g['glove']-g['head'], axis=1)-.15
    pair = np.linalg.norm(g['glove']-g['inactive_glove'], axis=1)-.16
    data = mujoco.MjData(model)
    contacts, sole = {}, 1e9
    for t, q in zip(times, poses):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for side in ('left', 'right'):
            gid = model.geom(side+'_foot_sole_collision').id
            sole = min(sole, float(data.geom_xpos[gid, 2]-np.dot(abs(data.geom_xmat[gid].reshape(3,3)[2]), model.geom_size[gid])))
        for c in data.contact[:data.ncon]:
            if c.dist < -.001:
                key = '|'.join(sorted([model.geom(c.geom1).name, model.geom(c.geom2).name]))
                if c.dist < contacts.get(key, {'distance_m': 0.})['distance_m']:
                    contacts[key] = {'distance_m': float(c.dist), 'time_s': float(t)}
    ids = np.array([model.joint(n).id for n in JOINT_NAMES])
    qp = reference.arrays['joint_pos']
    lo, hi = model.jnt_range[ids].T
    recovery = times>boundary+1e-9
    speed = np.gradient(g['glove'], times, axis=0, edge_order=2)
    acceleration = np.gradient(speed, times, axis=0, edge_order=2)
    stored_recovery = reference.time_s > boundary+1e-9
    stored_acceleration = np.gradient(reference.arrays['joint_vel'], reference.time_s, axis=0, edge_order=2)
    _, qa, _, _, _ = arm_map(model)
    arm_columns = qa-7
    recovery_metrics = {
        'interval_s': [boundary, float(reference.time_s[-1])],
        'maximum_joint_step_rad': float(abs(np.diff(qp, axis=0)[stored_recovery[1:]]).max()),
        'maximum_joint_speed_rad_s': float(abs(reference.arrays['joint_vel'][stored_recovery]).max()),
        'maximum_joint_acceleration_rad_s2': float(abs(stored_acceleration[stored_recovery]).max()),
        'maximum_active_arm_joint_step_rad': float(abs(np.diff(qp[:, arm_columns], axis=0)[stored_recovery[1:]]).max()),
        'maximum_active_arm_joint_speed_rad_s': float(abs(reference.arrays['joint_vel'][stored_recovery][:, arm_columns]).max()),
        'maximum_active_arm_joint_acceleration_rad_s2': float(abs(stored_acceleration[stored_recovery][:, arm_columns]).max()),
        'maximum_active_glove_speed_m_s': float(np.linalg.norm(speed[recovery], axis=1).max()),
        'maximum_active_glove_acceleration_m_s2': float(np.linalg.norm(acceleration[recovery], axis=1).max()),
        'active_arm_joint_path_length_rad': float(np.linalg.norm(np.diff(qp[:, arm_columns], axis=0)[stored_recovery[1:]],axis=1).sum()),
    }
    return {'sample_dt_s': .005, 'sample_count': len(times),
        'minimum_active_glove_own_head_surface_gap_m': float(gap.min()),
        'minimum_active_glove_own_head_gap_time_s': float(times[np.argmin(gap)]),
        'terminal_active_glove_own_head_surface_gap_m': float(gap[-1]),
        'minimum_glove_pair_surface_gap_m': float(pair.min()),
        'maximum_recovery_fist_forearm_angle_deg': float(g['fist_forearm_angle_deg'][recovery].max()),
        'terminal_fist_forearm_angle_deg': float(g['fist_forearm_angle_deg'][-1]),
        'angle_at_boundary_deg': float(g['fist_forearm_angle_deg'][np.argmin(abs(times-boundary))]),
        'maximum_joint_limit_excess_rad': float(np.maximum(np.maximum(lo-qp, qp-hi), 0).max()),
        'maximum_stored_joint_step_rad': float(abs(np.diff(qp, axis=0)).max()),
        'maximum_stored_joint_speed_rad_s': float(abs(reference.arrays['joint_vel']).max()),
        'maximum_stored_joint_acceleration_rad_s2': float(abs(np.gradient(reference.arrays['joint_vel'], reference.time_s, axis=0, edge_order=2)).max()),
        'maximum_active_glove_speed_m_s': float(np.linalg.norm(speed, axis=1).max()),
        'maximum_active_glove_acceleration_m_s2': float(np.linalg.norm(acceleration, axis=1).max()),
        'recovery_only_metrics': recovery_metrics,
        'native_filtered_self_penetrations_over_1mm': contacts, 'lowest_sole_vertex_m': sole,
        'own_head_radius_m': .07, 'glove_radius_m': .08,
        'note': 'Native geometric diagnostics only; prescribed H2 has no contacts or physical control.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/task10d/phase2_right_hook_recovery.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = (args.output or ROOT/config['output']).resolve()
    output.relative_to((ROOT/'outputs/task10d/phase2/variants/right_hook_repair01/recovery01').resolve())
    if output.exists():
        raise FileExistsError('Preserve previous repair attempts; choose a new output path')
    path = ROOT/config['source']
    if sha256(path) != config['source_sha256']:
        raise ValueError('Raw event identity differs')
    ref = StoredReference.load(path)
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    qpos, design, history = repair(model, ref, config)
    arrays = arrays_from_qpos(ref.time_s.copy(), qpos)
    arrays['source_frames'] = ref.arrays['source_frames'].copy()
    metadata = deepcopy(ref.metadata)
    metadata['derived_right_arm_recovery'] = design
    metadata['raw_reference'] = {'path': str(path), 'npz_sha256': sha256(path), 'json_sha256': sha256(path.with_suffix('.json'))}
    metadata['annotation_rationale'] += ' Right-arm recovery after0.90s and terminal guard are now derived; original source frame lineage/time remains unchanged.'
    candidate = StoredReference(arrays, metadata)
    before, after = review(model, ref, config['preserve_through_s']), review(model, candidate, config['preserve_through_s'])
    names, qa, _, _, _ = arm_map(model)
    other = [i for i in range(model.nq) if i not in qa]
    prefix = ref.time_s <= config['preserve_through_s']+1e-9
    rebuilt = arrays_from_qpos(arrays['time_s'], arrays['qpos'])
    verification = {'source_clock_exact': bool(np.array_equal(arrays['time_s'], ref.time_s)),
        'source_frame_lineage_exact': bool(np.array_equal(arrays['source_frames'], ref.arrays['source_frames'])),
        'prefix_qpos_exact': bool(np.array_equal(qpos[prefix], ref.arrays['qpos'][prefix])),
        'all_nonactive_qpos_exact': bool(np.array_equal(qpos[:, other], ref.arrays['qpos'][:, other])),
        'maximum_nonactive_qpos_difference': float(abs(qpos[:, other]-ref.arrays['qpos'][:, other]).max()),
        'dependent_array_regeneration_max_errors': {k:float(np.max(abs(v-arrays[k]))) for k,v in rebuilt.items() if v.dtype.kind in 'fc'}}
    output.mkdir(parents=True)
    np.savez_compressed(output/'right_hook.npz', **arrays)
    write_json(output/'right_hook.json', metadata)
    write_json(output/'review.json', {'raw':before, 'candidate':after, 'verification':verification, 'design':design})
    write_json(output/'generation.json', {'command':'source scripts/task10d_env.sh && .venv-task10c/bin/python -B -m human_punching.task10d.hook_recovery'+(' --output '+str(args.output) if args.output else ''),
        'config':config, 'config_sha256':sha256(args.config), 'generator_sha256':sha256(Path(__file__)),
        'source_npz_sha256':sha256(path), 'source_json_sha256':sha256(path.with_suffix('.json')),
        'h2_xml_sha256':sha256(H2_XML), 'candidate_npz_sha256':sha256(output/'right_hook.npz'),
        'candidate_json_sha256':sha256(output/'right_hook.json'), 'training_updates':0, 'physics_steps':0})
    write_json(output/'optimization_trace.json', history)
    print(json.dumps({'path':str(output/'right_hook.npz'), 'sha256':sha256(output/'right_hook.npz'),
        'candidate':after, 'verification':verification, 'design':design},indent=2))


if __name__ == '__main__':
    main()
