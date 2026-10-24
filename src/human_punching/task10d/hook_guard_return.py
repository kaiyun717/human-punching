"""Focused same-event return to the exact initial poses of both H2 arms.

Only the fourteen shoulder/elbow/wrist coordinates change. A smooth blend
retains the input motion through the stated withdrawal boundary and finishes
at its own initial arm configuration on the retained current body and stance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import BPoly

from human_punching.task10c.references import StoredReference, arrays_from_qpos
from human_punching.task10c.robot import H2_XML, JOINT_NAMES, ROOT, joint_addresses
from human_punching.task10c.targets import sha256
from human_punching.task10d.adaptation import diagnostics


OUTPUT_ROOT = ROOT/'outputs/task10d/phase2/variants/right_hook_guard_revision01'


def write_json(path: Path, value: dict) -> None:
    path = path.resolve()
    path.relative_to(OUTPUT_ROOT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def arm_coordinates(model: mujoco.MjModel):
    names = [n for n in JOINT_NAMES if n.startswith(('left_', 'right_'))
             and any(part in n for part in ('shoulder', 'elbow', 'wrist'))]
    assert len(names) == 14
    ids = np.asarray([model.joint(name).id for name in names])
    return names, model.jnt_qposadr[ids], model.jnt_dofadr[ids]


def blend_return(model: mujoco.MjModel, ref: StoredReference, config: dict):
    names, qa, _ = arm_coordinates(model)
    start, finish = config['blend_start_s'], config['guard_complete_s']
    if not (0.60 <= start < finish <= ref.time_s[-1]):
        raise ValueError('Guard blend must preserve the strike and fit the existing clock')
    if config['method'] not in ('moving_source_quintic_blend', 'fixed_right_waypoint_and_left_blend'):
        raise ValueError(config['method'])
    u = np.clip((ref.time_s-start)/(finish-start), 0., 1.)
    w = 10*u**3-15*u**4+6*u**5
    q = ref.arrays['qpos'].copy()
    q[:, qa] += w[:, None]*(q[0, qa]-q[:, qa])
    fixed_details = None
    if config['method'] == 'fixed_right_waypoint_and_left_blend':
        right_names = [name for name in names if name.startswith('right_')]
        ids = np.asarray([model.joint(name).id for name in right_names])
        right_qa, right_va = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
        anchor = ref.sample(start)
        start_pose = anchor['qpos'][right_qa]
        start_velocity = anchor['qvel'][right_va]
        guard = ref.arrays['qpos'][0, right_qa]
        middle = config['right_waypoint_s']
        waypoint_velocity = (guard-start_pose)/(finish-start)
        waypoint = start_pose+(middle-start)*waypoint_velocity
        waypoint[:4] = config['right_waypoint_shoulder_elbow_rad']
        curve = BPoly.from_derivatives([start, middle, finish],
            [[start_pose, start_velocity, np.zeros(7)],
             [waypoint, waypoint_velocity, np.zeros(7)],
             [guard, np.zeros(7), np.zeros(7)]])
        edited = (ref.time_s > start+1e-9)&(ref.time_s < finish-1e-9)
        q[np.ix_(np.flatnonzero(edited), right_qa)] = curve(ref.time_s[edited])
        fixed_details = {'right_waypoint_s': middle, 'right_waypoint_joint_names': right_names,
            'right_waypoint_joint_pose_rad':waypoint.tolist(), 'right_waypoint_velocity_rad_s':waypoint_velocity.tolist(),
            'anchor_velocity_rad_s':start_velocity.tolist(),
            'construction':'Two quintic Hermite spans; source position/velocity at anchor, shared waypoint position/velocity and zero acceleration, exact initial guard with zero endpoint velocity/acceleration.',
            'coordination':'Shoulder pitch reaches zero as roll crosses -pi/2; elbow folds during withdrawal. Left arm retains moving-source guard blend.'}
    # Exact endpoints are the requirement, not a soft posture objective.
    q[ref.time_s <= start+1e-9] = ref.arrays['qpos'][ref.time_s <= start+1e-9]
    q[np.ix_(np.flatnonzero(ref.time_s >= finish-1e-9), qa)] = q[0, qa]
    design = {'method': 'moving-source quintic joint blend to exact initial arm poses',
        'formula': 'q_arm(t)=(1-w(t))*q_input_arm(t)+w(t)*q_input_arm(0); w=10u^3-15u^4+6u^5',
        'blend_interval_s': [start, finish], 'preserved_through_s': start,
        'modified_joint_names': names, 'endpoint_constraint': 'Exact initial values for all fourteen arm joints',
        'guard_coordinate_frame': 'Arm joint coordinates relative to the retained torso, not the original world/root pose',
        'unchanged': 'Root, legs, waist, head, processed timestamps, source-frame lineage and original source query array',
        'orientation': 'No uniform alignment constraint; initial guard wrist flexion and arm posture restored exactly',
        'provenance': 'Derived two-arm withdrawal and guard revision of the same recorded event; no new style recording',
        'historical_recipe': 'Task10C hook_guard_return.correct_guard_return quintic arm blend; no old output path or task obligations reused'}
    if fixed_details:
        design['method'] = 'Fixed right-arm Hermite withdrawal through one coordinated waypoint, left-arm quintic source blend'
        design['formula'] = 'Right: fixed quintic Hermite spans; left: q=(1-w)*q_input+w*q_initial'
        design['fixed_right_return'] = fixed_details
        design['historical_recipe'] = 'Task10C compact_right_return fixed waypoint construction; event-specific times/anchor/guard, no prior output path or task obligations reused'
    return q, design


def native_review(model: mujoco.MjModel, ref: StoredReference, boundary: float):
    times = np.arange(0., float(ref.time_s[-1])+1e-9, .005)
    poses = ref.sample(times)['qpos']
    data = mujoco.MjData(model)
    gloves, heads, angles, elbows, torso_elbows, contacts = [], [], [], [], [], {}
    sole_min = 1e9
    for time, q in zip(times, poses):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        gloves.append([data.site(side+'_glove_center').xpos.copy() for side in ('left', 'right')])
        heads.append(data.geom('head_collision').xpos.copy())
        angle, elbow_angles, local_elbows = [], [], []
        for side in ('left', 'right'):
            wrist = data.body(side+'_wrist_yaw_link')
            elbow = data.body(side+'_elbow_link').xpos
            shoulder = data.body(side+'_shoulder_yaw_link').xpos
            torso = data.body('torso_link')
            local_elbows.append(torso.xmat.reshape(3,3).T@(elbow-torso.xpos))
            v = wrist.xpos-elbow
            v /= np.linalg.norm(v)
            fist = wrist.xmat.reshape(3, 3)[:, 0]
            angle.append(float(np.degrees(np.arccos(np.clip(np.dot(v, fist), -1., 1.)))))
            proximal = shoulder-elbow
            proximal /= np.linalg.norm(proximal)
            elbow_angles.append(float(np.degrees(np.arccos(np.clip(np.dot(proximal, v), -1., 1.)))))
            gid = model.geom(side+'_foot_sole_collision').id
            sole_min = min(sole_min, float(data.geom_xpos[gid, 2]-np.dot(
                abs(data.geom_xmat[gid].reshape(3,3)[2]), model.geom_size[gid])))
        angles.append(angle)
        elbows.append(elbow_angles)
        torso_elbows.append(local_elbows)
        for contact in data.contact[:data.ncon]:
            if contact.dist < -.001:
                key = '|'.join(sorted([model.geom(contact.geom1).name, model.geom(contact.geom2).name]))
                if contact.dist < contacts.get(key, {'distance_m': 0.})['distance_m']:
                    contacts[key] = {'distance_m': float(contact.dist), 'time_s': float(time)}
    gloves, heads, angles, elbows, torso_elbows = map(np.asarray, (gloves, heads, angles, elbows, torso_elbows))
    gaps = np.linalg.norm(gloves-heads[:, None], axis=2)-.15
    pair = np.linalg.norm(gloves[:, 0]-gloves[:, 1], axis=1)-.16
    stored = ref.time_s > boundary+1e-9
    recovery = times > boundary+1e-9
    names, qa, va = arm_coordinates(model)
    arm_step = np.diff(ref.arrays['qpos'][:, qa], axis=0)[stored[1:]]
    arm_vel = ref.arrays['qvel'][:, va]
    arm_accel = np.gradient(arm_vel, ref.time_s, axis=0, edge_order=2)
    glove_vel = np.gradient(gloves, times, axis=0, edge_order=2)
    glove_accel = np.gradient(glove_vel, times, axis=0, edge_order=2)
    ids = np.asarray([model.joint(name).id for name in names])
    lower, upper = model.jnt_range[ids].T
    snapshot_times = [.0, boundary, .70, .80, 1., 1.20, 1.40, 1.60, float(ref.time_s[-1])]
    indices = [int(np.argmin(abs(times-t))) for t in snapshot_times]
    return {'sample_dt_s': .005, 'sample_count': len(times), 'hand_order': ['left', 'right'],
        'minimum_glove_own_head_surface_gap_m': gaps.min(axis=0).tolist(),
        'minimum_glove_own_head_gap_time_s': times[np.argmin(gaps, axis=0)].tolist(),
        'minimum_recovery_glove_own_head_surface_gap_m': gaps[recovery].min(axis=0).tolist(),
        'terminal_glove_own_head_surface_gap_m': gaps[-1].tolist(),
        'minimum_glove_pair_surface_gap_m': float(pair.min()),
        'minimum_recovery_glove_pair_surface_gap_m': float(pair[recovery].min()),
        'terminal_glove_pair_surface_gap_m': float(pair[-1]),
        'native_filtered_self_penetrations_over_1mm': contacts,
        'lowest_sole_vertex_m': sole_min,
        'maximum_arm_joint_limit_excess_rad': float(np.maximum(np.maximum(lower-ref.arrays['qpos'][:, qa],ref.arrays['qpos'][:, qa]-upper),0).max()),
        'recovery_metrics': {'interval_s': [boundary, float(ref.time_s[-1])],
            'maximum_arm_joint_step_rad': float(abs(arm_step).max()),
            'maximum_arm_joint_speed_rad_s': float(abs(arm_vel[stored]).max()),
            'maximum_arm_joint_acceleration_rad_s2': float(abs(arm_accel[stored]).max()),
            'arm_joint_path_length_rad': float(np.linalg.norm(arm_step,axis=1).sum()),
            'maximum_each_glove_speed_m_s': np.linalg.norm(glove_vel[recovery],axis=2).max(axis=0).tolist(),
            'maximum_each_glove_acceleration_m_s2': np.linalg.norm(glove_accel[recovery],axis=2).max(axis=0).tolist(),
            'maximum_each_fist_forearm_angle_deg': angles[recovery].max(axis=0).tolist()},
        'recovery_torso_frame_elbows':{
            'minimum_xyz_m':torso_elbows[recovery].min(axis=0).tolist(),
            'maximum_xyz_m':torso_elbows[recovery].max(axis=0).tolist(),
            'path_length_m':np.linalg.norm(np.diff(torso_elbows[times>=boundary-1e-9],axis=0),axis=2).sum(axis=0).tolist()},
        'snapshots': {'time_s': times[indices].tolist(), 'fist_forearm_angle_deg': angles[indices].tolist(),
            'elbow_interior_angle_deg': elbows[indices].tolist(), 'glove_own_head_surface_gap_m': gaps[indices].tolist(),
            'torso_frame_elbow_xyz_m':torso_elbows[indices].tolist()},
        'terminal_arm_joint_speed_rad_s': arm_vel[-1].tolist(),
        'note': 'Native FK geometry only; 5 ms interpolation corners are included in finite-difference acceleration. Own H2 head radius .07 m differs from frozen G1 target radius .06 m.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/task10d/phase2_right_hook_guard_return.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = (args.output or ROOT/config['output']).resolve()
    output.relative_to(OUTPUT_ROOT.resolve())
    if output.exists():
        raise FileExistsError('Preserve previous candidates; choose a new output')
    source = (ROOT/config['source']).resolve()
    if sha256(source) != config['source_sha256']:
        raise ValueError('Input reference hash differs')
    ref = StoredReference.load(source)
    manifest_path = ROOT/config['source_manifest']
    manifest = json.loads(manifest_path.read_text())
    primitive = deepcopy(manifest['primitives'][0])
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    qpos, design = blend_return(model, ref, config)
    arrays = arrays_from_qpos(ref.time_s.copy(), qpos)
    for key, value in ref.arrays.items():
        if key not in arrays:
            arrays[key] = value.copy()
    metadata = deepcopy(ref.metadata)
    prior = {'path': str(source.relative_to(ROOT)), 'npz_sha256':sha256(source),
             'json_sha256':sha256(source.with_suffix('.json')), 'manifest_sha256':sha256(manifest_path)}
    metadata['prior_candidate_reference'] = prior
    metadata['prior_derived_recovery_repair'] = metadata.pop('derived_recovery_repair', None)
    metadata['derived_bilateral_start_guard_return'] = design
    metadata['terminal_guard_hold_s'] = float(ref.time_s[-1]-config['guard_complete_s'])
    metadata['annotation_rationale'] = ('Strike/entry and source-event annotations remain distinct and unchanged. '
        'Both arm joint poses now return exactly to their initial values at the terminal guard, '
        'while current nonarm motion and terminal body/stance are retained.')
    metadata['derived_terminal_timing']['endpoint_scope'] = 'Both arm endpoint poses replaced with exact initial arm coordinates; current nonarm endpoint retained.'
    metadata['derived_terminal_timing']['repaired_final_qpos_preserved_by_terminal_easing'] = False
    metadata['derived_terminal_timing']['current_processed_clock_preserved_exactly'] = True
    metadata['terminal_guard_pose_start_s'] = config['guard_complete_s']
    metadata['terminal_zero_stored_velocity_start_s'] = float(ref.time_s[np.flatnonzero(ref.time_s>config['guard_complete_s']+1e-9)[0]])
    candidate = StoredReference(arrays, metadata)
    target = np.asarray(primitive['target']['head_world'])
    nominal = diagnostics(model, arrays, target, 'right')
    before = native_review(model, ref, config['blend_start_s'])
    after = native_review(model, candidate, config['blend_start_s'])
    names, qa, _ = arm_coordinates(model)
    nonarm = np.asarray([i for i in range(model.nq) if i not in qa])
    prefix = ref.time_s <= config['blend_start_s']+1e-9
    terminal = ref.time_s >= config['guard_complete_s']-1e-9
    rebuilt = arrays_from_qpos(arrays['time_s'], arrays['qpos'])
    checks = {'prefix_poses_exact':bool(np.array_equal(qpos[prefix],ref.arrays['qpos'][prefix])),
        'all_nonarm_poses_exact':bool(np.array_equal(qpos[:,nonarm],ref.arrays['qpos'][:,nonarm])),
        'both_arms_start_equals_end_exact':bool(np.array_equal(qpos[0,qa],qpos[-1,qa])),
        'entire_terminal_guard_arm_pose_exact':bool(np.array_equal(qpos[terminal][:,qa],np.broadcast_to(qpos[0,qa],(int(terminal.sum()),len(qa))))),
        'timestamps_exact':bool(np.array_equal(arrays['time_s'],ref.time_s)),
        'source_frames_exact':bool(np.array_equal(arrays['source_frames'],ref.arrays['source_frames'])),
        'source_query_exact':bool(np.array_equal(arrays['pre_easing_reference_time_s'],ref.arrays['pre_easing_reference_time_s'])),
        'maximum_terminal_arm_difference_rad':float(abs(qpos[-1,qa]-qpos[0,qa]).max()),
        'dependent_array_regeneration_max_errors':{k:float(np.max(abs(v-arrays[k]))) for k,v in rebuilt.items() if v.dtype.kind in 'fc'}}
    if not all(v for k,v in checks.items() if isinstance(v,bool)):
        raise AssertionError(checks)
    output.mkdir(parents=True)
    np.savez_compressed(output/'right_hook.npz', **arrays)
    write_json(output/'right_hook.json', metadata)
    primitive.update(reference_npz=str((output/'right_hook.npz').relative_to(ROOT)),
        reference_sha256=sha256(output/'right_hook.npz'),reference_json_sha256=sha256(output/'right_hook.json'),
        diagnostics=nominal,role='candidate',review_status='CPU checks complete; visual/native preview pending',
        prior_candidate_reference=prior,derived_bilateral_start_guard_return=design,
        derived_terminal_timing=metadata['derived_terminal_timing'],previews={})
    primitive['prior_derived_recovery_repair'] = primitive.pop('derived_recovery_repair',None)
    primitive['annotations']['terminal_guard_pose_start_s'] = metadata['terminal_guard_pose_start_s']
    primitive['annotations']['terminal_zero_stored_velocity_start_s'] = metadata['terminal_zero_stored_velocity_start_s']
    primitive.pop('visual_review',None)
    primitive['limitations'] = ['Prescribed kinematic H2; no impact, torque, dynamic balance or avoidance claim.',
        'Derived two-arm return restores exact initial arm coordinates on the retained current body/stance; not the original recorded recovery.',
        'Full cycle visual preview and user disposition remain pending.',
        'One isolated reset and real target; no coverage or supported TTC-range claim.']
    write_json(output/'candidate_manifest.json',{'schema':'task10d_phase2_primitives_v1',
        'status':'Candidate only; no acceptance or accepted-library change','primitives':[primitive]})
    write_json(output/'review.json',{'input':before,'candidate':after,'checks':checks,'nominal':nominal,'design':design})
    write_json(output/'generation.json',{'command':'source scripts/task10d_env.sh && .venv-task10c/bin/python -B -m human_punching.task10d.hook_guard_return'+(' --output '+str(args.output) if args.output else ''),
        'config':config,'config_sha256':sha256(args.config),'generator_sha256':sha256(Path(__file__)),
        'input':prior,'h2_xml_sha256':sha256(H2_XML),'candidate_npz_sha256':sha256(output/'right_hook.npz'),
        'candidate_json_sha256':sha256(output/'right_hook.json'),'training_updates':0,'physics_steps':0})
    print(json.dumps({'path':str(output/'right_hook.npz'),'sha256':sha256(output/'right_hook.npz'),
        'checks':checks,'candidate':after,'nominal':nominal},indent=2))


if __name__ == '__main__':
    main()
