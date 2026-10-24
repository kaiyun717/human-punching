"""Explicit family-aware offline kinematic adaptation; no dynamic H2 control.

The existing nominal-jab deformation and bounded native IK are reused as a
method, with independent family windows, active hand, and source curved paths.
Planar reset placement is baked into qpos; runtime placement is identity.
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
from scipy import sparse
from scipy.spatial.transform import Rotation

from human_punching.targeting import _smoothstep
from human_punching.task10c.nominal_jab import axis_rotation, write_json
from human_punching.task10c.references import StoredReference, arrays_from_qpos
from human_punching.task10c.robot import H2_XML, JOINT_NAMES, ROOT
from human_punching.task10c.targets import sha256
from human_punching.task10d.replay import APPROVED_JAB, APPROVED_JAB_SHA256


def native_paths(model: mujoco.MjModel, poses: np.ndarray) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    result = {k: [] for side in ('left', 'right') for k in
              (side, side+'_rotation', side+'_shoulder', side+'_elbow', side+'_wrist')}
    result['head'] = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        for side in ('left', 'right'):
            site = data.site(side+'_glove_center')
            result[side].append(site.xpos.copy())
            result[side+'_rotation'].append(site.xmat.reshape(3, 3).copy())
            for part, link in (('shoulder', 'shoulder_yaw'), ('elbow', 'elbow'), ('wrist', 'wrist_yaw')):
                result[side+'_'+part].append(data.body(side+'_'+link+'_link').xpos.copy())
        result['head'].append(data.geom('head_collision').xpos.copy())
    return {k: np.asarray(v) for k, v in result.items()}


def place_source(model, ref, target, cfg, center_offset):
    """Choose XY registration before arm deformation; root height is untouched."""
    poses = ref.arrays['qpos'].copy()
    yaw = float(cfg['placement_yaw_rad'])
    rot = Rotation.from_rotvec([0., 0., yaw])
    poses[:, :3] = rot.apply(poses[:, :3])
    poses[:, 3:7] = (rot * Rotation.from_quat(poses[:, 3:7], scalar_first=True)).as_quat(scalar_first=True)
    paths = native_paths(model, poses)
    side, times = cfg['active_hand'], ref.time_s
    lo, hi = cfg['anchor_window_s']
    window = np.flatnonzero((times >= lo) & (times <= hi))
    # A hook's furthest-forward sweep center is different from its inherited
    # outward-completion frame; retain both annotations explicitly.
    direction = rot.apply([1., 0., 0.])
    anchor = int(window[np.argmax(paths[side][window] @ direction)])
    contact = target - center_offset*direction
    shift = contact[:2]-paths[side][anchor, :2]
    # A lower head should not force the source's maximum horizontal reach.
    # This isolated reset moves H2 closer, leaving room for a bent hook arm.
    shift += float(cfg.get('closer_reset_m', 0.0))*direction[:2]
    poses[:, :2] += shift
    return poses, anchor, contact, {'xy': shift.tolist(), 'yaw': yaw,
        'z': 0.0, 'baked_into_reference_qpos': True,
        'method': 'align selected native forward-sweep center in XY, then declared closer planar reset before adaptation',
        'closer_reset_m': float(cfg.get('closer_reset_m', 0.0)),
        'runtime_placement': {'xy': [0., 0.], 'yaw': 0.}}


def coupled_hook_solve(model, source, qa, dofs, lower, upper, begin, finish,
                       envelope, side, desired, rotations, elbow, alignment_weight=0.):
    """Fit one smooth arm correction trajectory, preventing framewise branches."""
    indices = np.arange(begin+1, finish)
    n, width = len(indices), len(qa)
    x = np.zeros((n, width))
    data = mujoco.MjData(model)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    site, elbow_id = model.site(side+'_glove_center').id, model.body(side+'_elbow_link').id
    # Endpoint corrections are identically zero; include these fixed samples
    # in the first/second difference rows so the entire correction returns.
    eye = np.eye(n+2)
    d1 = sparse.kron(sparse.csr_matrix(np.diff(eye, axis=0)[:, 1:-1]), sparse.eye(width))
    d2 = sparse.kron(sparse.csr_matrix(np.diff(eye, n=2, axis=0)[:, 1:-1]), sparse.eye(width))
    identity = sparse.eye(n*width)
    lo = np.maximum(lower-source[indices][:, qa], -1.25*envelope[indices, None]).ravel()
    hi = np.minimum(upper-source[indices][:, qa], 1.25*envelope[indices, None]).ravel()
    final_step = 0.
    for iteration in range(65):
        blocks, error_blocks = [], []
        for j, i in enumerate(indices):
            data.qpos[:] = source[i]
            data.qpos[qa] += x[j]
            mujoco.mj_forward(model, data)
            mujoco.mj_jacSite(model, data, jp, jr, site)
            matrices = [jp[:, dofs].copy(), .06*jr[:, dofs].copy()]
            errors = [desired[i]-data.site_xpos[site], .06*Rotation.from_matrix(
                rotations[i]@data.site_xmat[site].reshape(3, 3).T).as_rotvec()]
            mujoco.mj_jacBody(model, data, jp, jr, elbow_id)
            matrices.append(.18*jp[:, dofs].copy())
            errors.append(.18*(elbow[i]-data.xpos[elbow_id]))
            if alignment_weight:
                # Opt-in geometric wrist correction, evaluated against the
                # actual current forearm rather than the desired elbow path.
                # The full source guard and window endpoints remain unchanged.
                elbow_jac = jp[:, dofs].copy()
                wrist = model.body(side+'_wrist_yaw_link').id
                mujoco.mj_jacBody(model, data, jp, jr, wrist)
                forearm = data.xpos[wrist]-data.xpos[elbow_id]
                length = np.linalg.norm(forearm)
                forearm /= length
                fist = data.xmat[wrist].reshape(3, 3)[:, 0]
                skew = np.array([[0., -fist[2], fist[1]], [fist[2], 0., -fist[0]],
                                 [-fist[1], fist[0], 0.]])
                direction_jac = (np.eye(3)-np.outer(forearm, forearm)) @ (jp[:, dofs]-elbow_jac)/length
                weight = alignment_weight*envelope[i]
                matrices.append(weight*(-skew@jr[:, dofs]-direction_jac))
                errors.append(weight*(forearm-fist))
            blocks.append(sparse.csr_matrix(np.vstack(matrices)))
            error_blocks.append(np.concatenate(errors))
        flat = x.ravel()
        matrix = sparse.vstack([sparse.block_diag(blocks), .035*identity, .30*d1, 1.0*d2, .025*identity], format='csr')
        error = np.concatenate([*error_blocks, -.035*flat, -.30*(d1@flat), -1.0*(d2@flat), np.zeros_like(flat)])
        delta = lsq_linear(matrix, error, bounds=(np.maximum(-.12, lo-flat), np.minimum(.12, hi-flat)),
            method='trf', lsq_solver='lsmr', lsmr_tol=1e-6, max_iter=50, tol=1e-7).x
        x += delta.reshape(n, width)
        final_step = float(np.max(abs(delta)))
        if final_step < 2e-5:
            break
    result = source.copy()
    result[np.ix_(indices, qa)] += x
    return result, iteration+1, final_step


def solve_family(model, ref, cfg, target, center_offset):
    source, anchor, contact, placement = place_source(model, ref, target, cfg, center_offset)
    paths = native_paths(model, source)
    side, times = cfg['active_hand'], ref.time_s
    begin, finish = [int(np.argmin(abs(times-x))) for x in cfg['deformation_interval_s']]
    center = float(times[anchor])
    if not times[begin] < center < times[finish]:
        raise ValueError('Family deformation window must contain its own center')
    envelope = np.where(times <= center,
        _smoothstep((times-times[begin])/(center-times[begin])),
        _smoothstep((times[finish]-times)/(times[finish]-center)))
    correction = contact-paths[side][anchor]
    desired = paths[side] + envelope[:, None]*correction
    if cfg['path'] == 'straight_c2':
        velocity = np.gradient(paths[side], times, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
        curve = BPoly.from_derivatives([times[begin], center, times[finish]],
            [[paths[side][begin], velocity[begin], acceleration[begin]],
             [contact, np.zeros(3), np.zeros(3)],
             [paths[side][finish], velocity[finish], acceleration[finish]]])
        desired[begin:finish+1] = curve(times[begin:finish+1])
    elif cfg['path'] not in ('curved_source_offset', 'source_path_offset'):
        raise ValueError('Unknown family path method')
    elbow = paths[side+'_elbow'] + cfg['elbow_translation_fraction']*(desired-paths[side])
    if cfg['path'] == 'straight_c2':
        reach = desired-paths[side+'_shoulder']
        reach /= np.maximum(np.linalg.norm(reach, axis=1, keepdims=True), 1e-12)
        length = np.linalg.norm(paths[side+'_elbow']-paths[side+'_shoulder'], axis=1)
        extended = paths[side+'_shoulder']+length[:, None]*reach
        elbow += envelope[:, None]*(extended-elbow)
    rotations = np.stack([axis_rotation(r, p-e, .75*w) for r, p, e, w in
        zip(paths[side+'_rotation'], desired, elbow, envelope)])
    names = [n for n in JOINT_NAMES if n.startswith(side+'_') and any(
        x in n for x in ('shoulder', 'elbow', 'wrist'))]
    ids = np.array([model.joint(n).id for n in names])
    qa, dofs = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    lower, upper = model.jnt_range[ids].T
    result, previous = source.copy(), np.zeros(len(qa))
    data = mujoco.MjData(model)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    site, elbow_id = model.site(side+'_glove_center').id, model.body(side+'_elbow_link').id
    iterations = []
    for i in range(len(times)) if cfg['path']=='straight_c2' else []:
        if envelope[i] == 0:
            previous[:] = 0
            iterations.append(0)
            continue
        pose = source[i].copy()
        # Bounds shrink with the same C2 path envelope. This prevents a
        # singular shoulder branch from persisting into an unchanged edge.
        # The bound is an offline shape preference, not a feasibility gate.
        correction_bound = 1.20 if cfg['path'] == 'straight_c2' else .90
        frame_lower = np.maximum(lower, source[i, qa]-correction_bound*envelope[i])
        frame_upper = np.minimum(upper, source[i, qa]+correction_bound*envelope[i])
        pose[qa] = np.clip(pose[qa]+previous, frame_lower, frame_upper)
        for iteration in range(80):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            mujoco.mj_jacSite(model, data, jp, jr, site)
            matrices = [jp[:, dofs].copy(), .08*jr[:, dofs].copy()]
            errors = [desired[i]-data.site_xpos[site], .08*Rotation.from_matrix(
                rotations[i] @ data.site_xmat[site].reshape(3, 3).T).as_rotvec()]
            mujoco.mj_jacBody(model, data, jp, jr, elbow_id)
            matrices.append(.22*jp[:, dofs].copy())
            errors.append(.22*(elbow[i]-data.xpos[elbow_id]))
            source_weight = .025 + .15*(1-envelope[i])**2
            matrices.extend([source_weight*np.eye(len(qa)), .035*np.eye(len(qa)), .025*np.eye(len(qa))])
            errors.extend([source_weight*(source[i, qa]-pose[qa]),
                .035*(source[i, qa]+previous-pose[qa]), np.zeros(len(qa))])
            delta = lsq_linear(np.vstack(matrices), np.concatenate(errors),
                bounds=(np.maximum(-.12, frame_lower-pose[qa]), np.minimum(.12, frame_upper-pose[qa])),
                method='bvls', max_iter=50, tol=1e-8).x
            pose[qa] += delta
            if np.max(abs(delta)) < 2e-6:
                break
        result[i, qa] = pose[qa]
        previous = pose[qa]-source[i, qa]
        iterations.append(iteration+1)
    coupled_final_step = None
    if cfg['path'] in ('curved_source_offset', 'source_path_offset'):
        result, count, coupled_final_step = coupled_hook_solve(model, source, qa, dofs, lower, upper,
            begin, finish, envelope, side, desired, rotations, elbow,
            alignment_weight=float(cfg.get('fist_forearm_alignment_weight', 0.)))
        iterations.append(count)
        correction_bound = 1.25
    design = {'method': 'family-aware smooth Cartesian deformation and native bounded arm IK',
        'path_method': cfg['path'], 'active_hand': side, 'source_annotation_time_s': ref.metadata['strike_time_s'],
        'strike_center_time_s': center, 'deformation_interval_s': [float(times[begin]), float(times[finish])],
        'contact_point_world': contact.tolist(), 'peak_path_correction_world_m': correction.tolist(),
        'source_to_world_placement': placement, 'modified_joint_names': names,
        'maximum_joint_adjustment_rad': float(np.max(abs(result[:, qa]-source[:, qa]))),
        'max_iterations': max(iterations), 'step_bound_rad': .12,
        'correction_bound_rad_times_envelope': correction_bound,
        'source_posture_weight': ('.025 + .15*(1-envelope)^2; stronger source attraction at adaptation edges'
                                 if cfg['path']=='straight_c2' else '.035; coupled whole-interval correction'),
        'hook_temporal_fit': None if coupled_final_step is None else {
            'method': 'one sparse bounded fit of complete arm correction trajectory',
            'correction_first_difference_weight': .30, 'correction_second_difference_weight': 1.0,
            'source_posture_weight': .035, 'final_optimization_step_rad': coupled_final_step,
            'termination': 'small_step' if coupled_final_step < 2e-5 else '65_iteration_cap',
            'convergence_claim': bool(coupled_final_step < 2e-5),
            'endpoint_corrections': 'identically zero; both fixed edges included in derivative residuals'},
        'unchanged_after_reset_placement': 'root trajectory, legs, waist, head, inactive arm and endpoints',
        'source_curvature': 'native source curve retained with smooth target offset' if cfg['path']!='straight_c2' else 'C2 approach/withdrawal replaces central Cartesian segment',
        'orientation': 'source roll retained with soft fist continuation toward adapted forearm',
        'explicit_fist_forearm_alignment_weight': float(cfg.get('fist_forearm_alignment_weight', 0.)),
        'source_frame_role': 'original event lineage; adapted samples are not exact recorded poses',
        'no_dynamic_feasibility_claim': True}
    return result, source, design


def diagnostics(model, arrays, target, side):
    ref = StoredReference(arrays, {})
    times = np.arange(0., float(ref.time_s[-1])+1e-9, .005)
    poses = ref.sample(times)['qpos']
    paths = native_paths(model, poses)
    distances = np.stack([np.linalg.norm(paths[s]-target, axis=1) for s in ('left', 'right')], axis=1)
    hits = np.flatnonzero(distances.min(axis=1) <= .14)
    active = 0 if side == 'left' else 1
    nearest = int(np.argmin(distances[:, active]))
    jp = arrays['joint_pos']
    velocity = np.gradient(paths[side], times, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, times, axis=0, edge_order=2)
    ids = np.array([model.joint(n).id for n in JOINT_NAMES])
    lower, upper = model.jnt_range[ids].T
    data = mujoco.MjData(model)
    deepest = {}
    sole_min = 1e9
    glove_self_min = 1e9
    for pose in arrays['qpos']:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for s in ('left', 'right'):
            geom = model.geom(s+'_foot_sole_collision').id
            rotation = data.geom_xmat[geom].reshape(3, 3)
            sole_min = min(sole_min, float(data.geom_xpos[geom, 2]-np.dot(abs(rotation[2]), model.geom_size[geom])))
        glove_self_min = min(glove_self_min, float(np.linalg.norm(data.site('left_glove_center').xpos-data.site('right_glove_center').xpos)))
        for c in data.contact[:data.ncon]:
            if c.dist < -.001:
                pair = '|'.join(sorted([model.geom(c.geom1).name, model.geom(c.geom2).name]))
                deepest[pair] = min(deepest.get(pair, 0.), float(c.dist))
    return {'first_entry_s': None if not len(hits) else float(times[hits[0]]),
        'first_entry_hand': None if not len(hits) else ('left','right')[int(np.argmin(distances[hits[0]]))],
        'first_entry_sampling_s': .005, 'minimum_both_glove_distance_m': float(distances.min()),
        'active_closest_center_distance_m': float(distances[nearest, active]),
        'active_closest_approach_s': float(times[nearest]),
        'initial_both_glove_distances_m': distances[0].tolist(),
        'terminal_both_glove_distances_m': distances[-1].tolist(),
        'maximum_active_glove_speed_m_s': float(np.linalg.norm(velocity, axis=1).max()),
        'maximum_active_glove_acceleration_m_s2': float(np.linalg.norm(acceleration, axis=1).max()),
        'maximum_stored_joint_speed_rad_s': float(abs(arrays['joint_vel']).max()),
        'maximum_stored_joint_step_rad': float(abs(np.diff(jp, axis=0)).max()),
        'maximum_stored_joint_acceleration_rad_s2': float(abs(np.gradient(arrays['joint_vel'], arrays['time_s'], axis=0, edge_order=2)).max()),
        'minimum_each_glove_to_own_head_center_m': [float(np.linalg.norm(paths[s]-paths['head'], axis=1).min()) for s in ('left','right')],
        'maximum_joint_limit_excess_rad': float(np.maximum(np.maximum(lower-jp, jp-upper), 0).max()),
        'lowest_sole_vertex_m': sole_min, 'minimum_glove_to_glove_center_distance_m': glove_self_min,
        'native_filtered_self_penetrations_over_1mm': deepest,
        'contact_diagnostic_note': 'native geometry with original pair filtering; prescribed replay disables all contacts',
        'acceleration_note': '5 ms FK finite differences of linearly interpolated 20 ms poses; interpolation corners included'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/task10d/phase2_nominal.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--families', nargs='+', choices=['cross', 'left_hook', 'right_hook'])
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    output = (args.output or ROOT/cfg['output']).resolve()
    output.relative_to(ROOT.resolve())
    if (output/'nominal_manifest.json').exists():
        raise FileExistsError('Use a new candidate output; preserve previous evidence')
    output.mkdir(parents=True, exist_ok=True)
    target = np.asarray(cfg['target']['head_world'])
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    primitives, comparisons = [], {}
    if sha256(APPROVED_JAB) != APPROVED_JAB_SHA256:
        raise ValueError('Approved jab changed')
    jab = StoredReference.load(APPROVED_JAB)
    primitives.append({'primitive_id': 'jab_nominal', 'family': 'jab', 'active_hand': 'left', 'role': 'nominal',
        'reference_npz': str(APPROVED_JAB.relative_to(ROOT)), 'reference_sha256': APPROVED_JAB_SHA256,
        'reference_json_sha256': sha256(APPROVED_JAB.with_suffix('.json')),
        'placement': {'xy': [0., 0.], 'yaw': 0.}, 'target': cfg['target'],
        'annotations': {'extension_center_s': .68, 'nominal_first_entry_s': .51, 'withdrawal_start_s': .68,
                        'terminal_guard_start_s': 1.38, 'end_s': 1.42},
        'provenance': 'approved derived-spatial jab retained byte-for-byte', 'source': jab.metadata['source']})
    for family, family_cfg in cfg['families'].items():
        if args.families and family not in args.families:
            continue
        source_path = ROOT/family_cfg['reference']
        if sha256(source_path) != family_cfg['sha256']:
            raise ValueError(f'Selected {family} identity differs')
        ref = StoredReference.load(source_path)
        if (ref.metadata['family'], ref.metadata['active_hand']) != (family, family_cfg['active_hand']):
            raise ValueError('Family/hand annotation mismatch')
        poses, placed, design = solve_family(model, ref, family_cfg, target, cfg['contact_center_offset_m'])
        arrays = arrays_from_qpos(ref.time_s.copy(), poses)
        arrays['source_frames'] = ref.arrays['source_frames'].copy()
        before = diagnostics(model, arrays_from_qpos(ref.time_s, placed), target, family_cfg['active_hand'])
        after = diagnostics(model, arrays, target, family_cfg['active_hand'])
        rebuilt = arrays_from_qpos(arrays['time_s'], arrays['qpos'])
        regeneration = {k: float(np.max(abs(v-arrays[k]))) for k, v in rebuilt.items()
                        if v.dtype.kind in 'fc'}
        metadata = deepcopy(ref.metadata)
        metadata['historical_source_ttc_range_s'] = metadata.pop('ttc_range_s', None)
        metadata['timing_range_status'] = 'Historical Task10C range is provenance only; no supported TTC range established for this kinematic primitive.'
        metadata.update({'schema': 'task10d_h2_reference_v1', 'strike_time_s': design['strike_center_time_s'],
            'strike_center_time_s': design['strike_center_time_s'],
            'source_strike_annotation_time_s': ref.metadata['strike_time_s'],
            'nominal_first_entry_s': after['first_entry_s'],
            'nominal_closest_approach_s': after['active_closest_approach_s'],
            'nominal_target_adaptation': design | {'target': cfg['target'], 'source_npz': str(source_path),
                'source_npz_sha256': sha256(source_path), 'source_json_sha256': sha256(source_path.with_suffix('.json'))},
            'annotation_rationale': 'Own-family native sweep/extension center and measured 5 ms first entry are distinct; source frames remain original event lineage.',
            'strike_center_semantics': 'Design aim center selected by source forward projection; not necessarily adapted closest approach or maximum elbow extension.',
            'recovery_duration_s': float(ref.time_s[-1]-design['strike_center_time_s']),
            'withdrawal_interval_s': [design['strike_center_time_s'], ref.metadata['withdrawal_interval_s'][-1]],
            'runtime_placement': {'xy': [0., 0.], 'yaw': 0.},
            'dependent_arrays': 'Recomputed from placed/adapted native qpos using arrays_from_qpos'})
        path = output/(family+'.npz')
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, **arrays)
        write_json(path.with_suffix('.json'), metadata)
        comparisons[family] = {'placed_source': before, 'candidate': after, 'design': design,
            'dependent_array_regeneration_max_absolute_errors': regeneration,
            'source_sha256': sha256(source_path), 'candidate_sha256': sha256(path),
            'launch_qpos_matches_placed_source': bool(np.array_equal(poses[0], placed[0])),
            'terminal_qpos_matches_placed_source': bool(np.array_equal(poses[-1], placed[-1])),
            'root_qpos_matches_placed_source': bool(np.array_equal(poses[:, :7], placed[:, :7]))}
        primitives.append({'primitive_id': family+'_nominal', 'family': family,
            'active_hand': family_cfg['active_hand'], 'role': 'nominal',
            'reference_npz': str(path.relative_to(ROOT)), 'reference_sha256': sha256(path),
            'reference_json_sha256': sha256(path.with_suffix('.json')),
            'placement': {'xy': [0., 0.], 'yaw': 0.}, 'source_to_world_placement': design['source_to_world_placement'],
            'target': cfg['target'], 'annotations': {'extension_center_s': design['strike_center_time_s'],
                'nominal_first_entry_s': after['first_entry_s'], 'withdrawal_start_s': design['strike_center_time_s'],
                'terminal_guard_start_s': ref.metadata['terminal_guard_interval_s'][0], 'end_s': float(ref.time_s[-1])},
            'provenance': 'derived-spatial adaptation of selected corrected recorded event', 'source': ref.metadata['source'],
            'diagnostics': after})
        print(json.dumps({'family': family, 'candidate': str(path), 'diagnostics': after}), flush=True)
    write_json(output/'diagnostics.json', comparisons)
    write_json(output/'generation.json', {'config': cfg, 'config_path': str(args.config.resolve()),
        'processed_families': args.families or list(cfg['families']),
        'config_sha256': sha256(args.config), 'generator_sha256': sha256(Path(__file__)),
        'h2_xml_sha256': sha256(H2_XML), 'training_updates': 0, 'physics_steps': 0})
    write_json(output/'nominal_manifest.json', {'schema': 'task10d_phase2_primitives_v1',
        'primitives': primitives, 'scope': 'isolated complete reset scenarios; no live family switching'})


if __name__ == '__main__':
    main()
