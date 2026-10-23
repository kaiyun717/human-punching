"""Posthoc tracking and fixed-launch media for the two nominal jab controls.

No simulation stepping, policy inference, reward edits or checkpoint selection.
Actor endpoint kinematics are distinct from the original 5 ms encounter latch
and solver telemetry. Repeated nominal recipes are not independent samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mjlab.scene import Scene
from mjlab.utils.lab_api.math import (
    quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat,
)

from .commands import TorchReference
from .env_cfg import make_env_cfg
from .evaluate import reference_env_kwargs, verify_reference_contract
from .events import reference_clock
from .imitation import IMITATION_STDS, IMITATION_WEIGHTS, align_reference_poses
from .robot import JOINT_NAMES, ROOT


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _stats(value):
    value = np.asarray(value)
    if not value.size:
        return None
    return {'mean': float(value.mean()), 'rms': float(np.sqrt(np.mean(value**2))),
            'minimum': float(value.min()), 'maximum': float(value.max())}


def _yaw(quat):
    w, x, y, z = np.asarray(quat).T
    return np.arctan2(2*(x*y+w*z), 1-2*(y*y+z*z))


def _wrapped(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _angle(a, b):
    cosine = np.sum(a*b, axis=-1) / np.maximum(
        np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1), 1e-12)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def _fk(model, qpos, body_names):
    data = mujoco.MjData(model)
    body_ids = [model.body('robot/'+name).id for name in body_names]
    sites = [model.site('robot/'+side+'_glove_center').id for side in ('left', 'right')]
    extra_names = [side+'_'+part+'_link' for side in ('left', 'right')
                   for part in ('shoulder_yaw', 'elbow', 'wrist_yaw')]
    extra_ids = [model.body('robot/'+name).id for name in extra_names]
    head = model.geom('robot/head_collision').id
    result = {name: [] for name in ('glove', 'glove_rotation', 'body_position',
                                   'body_quaternion', 'arm_origins', 'head')}
    for pose in qpos:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        for name, value in (
            ('glove', data.site_xpos[sites]),
            ('glove_rotation', data.site_xmat[sites].reshape(2, 3, 3)),
            ('body_position', data.xpos[body_ids]), ('body_quaternion', data.xquat[body_ids]),
            ('arm_origins', data.xpos[extra_ids].reshape(2, 3, 3)),
            ('head', data.geom_xpos[head]),
        ):
            result[name].append(value.copy())
    return {name: np.asarray(value) for name, value in result.items()}


def _reference_at(reference, model, elapsed, requested):
    elapsed = torch.as_tensor(np.asarray(elapsed), dtype=torch.float32)
    source, rate = reference_clock(elapsed, torch.full_like(elapsed, requested),
        reference.ttc_anchor_time, reference.end_time, reference.recovery,
        mode=reference.clock_mode)
    sample = reference.sample(source, rate)
    qpos = np.repeat(model.qpos0[None], len(elapsed), axis=0)
    qpos[:, :3] = sample['root_pos_w'].numpy()
    qpos[:, 3:7] = sample['root_quat_w'].numpy()
    addresses = [model.jnt_qposadr[model.joint('robot/'+name).id] for name in JOINT_NAMES]
    qpos[:, addresses] = sample['joint_pos'].numpy()
    return source.numpy(), sample, qpos


def _tracking(trace, sample, reference_ids):
    actual_position = torch.as_tensor(trace['body_pos_w'])
    actual_quaternion = torch.as_tensor(trace['body_quat_w'])
    reference_position = sample['body_pos_w'][:, reference_ids]
    reference_quaternion = sample['body_quat_w'][:, reference_ids]
    relative_position, relative_quaternion = align_reference_poses(
        actual_position, actual_quaternion, reference_position, reference_quaternion)
    common = {
        'anchor_height': (reference_position[:, 0, 2]-actual_position[:, 0, 2]).square(),
        'anchor_orientation': quat_error_magnitude(reference_quaternion[:, 0], actual_quaternion[:, 0]).square(),
        'body_linear_velocity': (sample['body_lin_vel_w'][:, reference_ids]-torch.as_tensor(trace['body_lin_vel_w'])).square().sum(-1).mean(-1),
        'body_angular_velocity': (sample['body_ang_vel_w'][:, reference_ids]-torch.as_tensor(trace['body_ang_vel_w'])).square().sum(-1).mean(-1),
    }
    results = {}
    for name, position, quaternion in (
        ('launch_registered', reference_position, reference_quaternion),
        ('relative', relative_position, relative_quaternion),
    ):
        position_error = (position-actual_position).norm(dim=-1)
        orientation_error = quat_error_magnitude(quaternion, actual_quaternion)
        errors = {**common, 'body_position': position_error.square().mean(-1),
                  'body_orientation': orientation_error.square().mean(-1)}
        scores = {key: torch.exp(-value/IMITATION_STDS[key]**2) for key, value in errors.items()}
        results[name] = {'body_position_error_m': position_error.numpy(),
            'body_orientation_error_rad': orientation_error.numpy(),
            'squared_errors': {key: value.numpy() for key, value in errors.items()},
            'scores': {key: value.numpy() for key, value in scores.items()},
            'weighted_score': (sum(IMITATION_WEIGHTS[key]*value for key, value in scores.items())
                               / sum(IMITATION_WEIGHTS.values())).numpy()}
    return results


def _phase_tracking(tracking, mask, groups):
    if not mask.any():
        return None
    return {'six_term_weighted_score': _stats(tracking['weighted_score'][mask]),
        'component_score_means': {key: float(value[mask].mean()) for key, value in tracking['scores'].items()},
        'component_squared_error_means': {key: float(value[mask].mean()) for key, value in tracking['squared_errors'].items()},
        'body_groups': {group: {
            'position_error_m': _stats(tracking['body_position_error_m'][mask][:, indices]),
            'orientation_error_deg': _stats(np.degrees(tracking['body_orientation_error_rad'][mask][:, indices])),
        } for group, indices in groups.items()}}


def _plot(output, series, requested, entry, strike_range):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = series['elapsed_s']
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    for frame in ('launch_registered', 'relative'):
        axes[0, 0].plot(t, series[frame+'_body_position_rmse_m'], label=frame)
    axes[0, 0].set_ylabel('Body-position RMSE (m)')
    for name, style in (('actual', '-'), ('reference', '--')):
        axes[0, 1].plot(t, series[name+'_left_elbow_extension_deg'], style, label=name)
        axes[1, 0].plot(t, series[name+'_torso_tilt_deg'], style, label=name+' torso')
        axes[1, 0].plot(t, series[name+'_head_tilt_deg'], style, label=name+' head')
    axes[0, 1].set_ylabel('Left elbow angle (deg; 180 straight)')
    axes[1, 0].set_ylabel('Tilt from vertical (deg)')
    axes[1, 1].plot(t, series['pelvis_xy_reference_error_m'], label='XY error (m)')
    axes[1, 1].plot(t, np.radians(series['pelvis_yaw_reference_error_deg']), label='yaw error (rad)')
    axes[1, 1].set_ylabel('Pelvis error to fixed reference')
    for side, i in (('left', 0), ('right', 1)):
        valid = series['solver_elapsed_s'] >= 0
        axes[2, 0].plot(series['solver_elapsed_s'][valid], series['foot_vertical_force_N'][valid, i], label=side)
    axes[2, 0].set_ylabel('Actual sampled foot load (N)')
    axes[2, 1].plot(t, series['command_excess_rad'].max(-1), label='max named joint excess')
    axes[2, 1].set_ylabel('Command excess (rad)')
    for ax in axes.flat:
        ax.axvline(requested, color='black', lw=.8, label='requested TTC' if ax is axes[0, 0] else None)
        if entry is not None:
            ax.axvline(entry, color='tab:red', lw=.8, ls=':')
        if strike_range is not None:
            ax.axvspan(*strike_range, color='gold', alpha=.12)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel('Elapsed from full-guard reset (s)')
    fig.suptitle('Nominal control: strike-window tracking, motion, support and commands\n'
                 'Gold: source 0.30–0.90 s; red: actual 5 ms first encounter; no acceptance thresholds')
    fig.savefig(output/'summary.png', dpi=150)
    plt.close(fig)


def _render(output, model, actual_qpos, reference_qpos, elapsed, source, target, row, metrics):
    if not torch.cuda.is_available():
        raise RuntimeError('Reference/actual rendering requires the authorized allocated CUDA GPU')
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    launch = reference_qpos[0, :3].copy()
    launch[2] = 0
    yaw = float(_yaw(reference_qpos[0:1, 3:7])[0])
    inverse = Rotation.from_euler('z', -yaw)
    def fixed_launch(qpos):
        qpos = qpos.copy()
        qpos[:, :3] = inverse.apply(qpos[:, :3]-launch)
        qpos[:, 3:7] = (inverse*Rotation.from_quat(qpos[:, 3:7], scalar_first=True)).as_quat(scalar_first=True)
        return qpos
    reference_qpos, actual_qpos = fixed_launch(reference_qpos), fixed_launch(actual_qpos)
    target = inverse.apply(target-launch)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=480)
    cameras = []
    for azimuth in (90, 0):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [.35, 0, .90]
        camera.distance, camera.azimuth, camera.elevation = 3.25, azimuth, -8
        cameras.append(camera)
    path = output/'reference_vs_actual.mp4'
    try:
        with imageio.get_writer(path, fps=50, macro_block_size=16) as writer:
            for index, time in enumerate(elapsed):
                columns = []
                for title, poses in (('KINEMATIC REFERENCE', reference_qpos), ('ACTUAL POLICY', actual_qpos)):
                    data.qpos[:] = poses[index]
                    mujoco.mj_forward(model, data)
                    views = []
                    for camera in cameras:
                        renderer.update_scene(data, camera)
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.full(3, .06), target, np.eye(3).ravel(),
                            np.array([1., .2, .2, .4], dtype=np.float32))
                        renderer.scene.ngeom += 1
                        views.append(renderer.render().copy())
                    panel = Image.fromarray(np.concatenate(views, axis=0))
                    ImageDraw.Draw(panel).text((8, 8), title+' | fixed launch frame', fill='white', stroke_width=1, stroke_fill='black')
                    columns.append(np.asarray(panel))
                frame = Image.fromarray(np.concatenate(columns, axis=1))
                actual = row.get('actual_ttc_s')
                event = 'none yet' if actual is None or time < actual else f'{actual:.3f}s'
                text = (f"t={time:.3f}s | source={source[index]:.3f}s | requested={row['requested_ttc_s']:.3f}s | actual first encounter={event}\n"
                        f"{metrics['imitation_frame']} | non-solid frozen target | side / front | source extension center=0.680s")
                ImageDraw.Draw(frame).text((8, 675), text, fill='white', stroke_width=1, stroke_fill='black')
                writer.append_data(np.asarray(frame))
    finally:
        renderer.close()
    return {'path': str(path), 'frames': len(elapsed), 'fps': 50,
            'first_elapsed_s': float(elapsed[0]), 'last_elapsed_s': float(elapsed[-1]),
            'reference_panel': 'Native compiled-scene FK; not dynamically executed',
            'actual_panel': 'Unmodified saved policy qpos; no policy or dynamics replay',
            'placement': 'Same fixed inverse launch yaw/ground-projection transform on both motions and target; no ongoing alignment',
            'views': ['side', 'front']}


def analyze_evaluation(evaluation_dir, output_dir=None, render=False) -> dict:
    """Analyze saved recipe 0, returning JSON-ready metrics for lightweight review.

    Uses summary.reference_contract and the existing compiled training Scene.
    Both common fixed-world and current-pelvis-relative tracking are reported
    irrespective of the condition's reward frame. Never selects a checkpoint.
    """
    evaluation = Path(evaluation_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else evaluation/'nominal_control_analysis'
    output.relative_to((ROOT/'outputs/task10c').resolve())
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((evaluation/'summary.json').read_text())
    contract = verify_reference_contract(summary['reference_contract'])
    row = next(row for row in json.loads((evaluation/'attempts.json').read_text()) if row['recipe_id'] == 0)
    trace_path = evaluation/'videos/recipe_0000.npz'
    with np.load(trace_path, allow_pickle=False) as loaded:
        trace = {name: loaded[name].copy() for name in loaded.files}
    elapsed = trace['elapsed_s'].astype(float)
    if elapsed[0] != 0 or row.get('reset_kind') != 'full':
        raise ValueError('Nominal control analysis requires a saved full-guard recipe 0')
    cfg = make_env_cfg(num_envs=1, **reference_env_kwargs(contract))
    model = Scene(cfg.scene, device='cpu').compile()
    model.opt.timestep = cfg.sim.mujoco.timestep
    reference = TorchReference(ROOT/contract['reference_path'], 'cpu', JOINT_NAMES, family='jab',
        clock_mode=contract['clock_mode'], ttc_anchor_time=contract['ttc_anchor_time'],
        post_entry_duration=contract['post_entry_duration'])
    requested = float(row['requested_ttc_s'])
    source, sample, reference_qpos = _reference_at(reference, model, elapsed, requested)
    body_names = list(map(str, trace['body_names']))
    if body_names[0] != 'pelvis':
        raise ValueError('Expected pelvis anchor first in the named tracking map')
    reference_ids = [reference.body_names.index(name) for name in body_names]
    actual = _fk(model, trace['qpos'], body_names)
    desired = _fk(model, reference_qpos, body_names)
    tracking = _tracking(trace, sample, reference_ids)
    frame = contract.get('imitation_frame', 'relative')
    groups = {'whole_body': list(range(len(body_names))),
        'active_arm': [i for i, name in enumerate(body_names) if name.startswith('left_') and any(part in name for part in ('shoulder', 'elbow', 'wrist'))],
        'inactive_arm': [i for i, name in enumerate(body_names) if name.startswith('right_') and any(part in name for part in ('shoulder', 'elbow', 'wrist'))],
        'head': [i for i, name in enumerate(body_names) if name.startswith('head_')],
        'pelvis_torso': [body_names.index(name) for name in ('pelvis', 'torso_link')],
        'legs_feet': [i for i, name in enumerate(body_names) if any(part in name for part in ('hip', 'knee', 'ankle'))]}
    phase_masks = {'whole_cycle': np.ones(len(source), dtype=bool),
        'strike_window': (source >= .30) & (source <= .90),
        'approach': (source >= .12) & (source <= .510),
        'extension_after_entry_anchor': (source > .510) & (source <= .680),
        'withdrawal': (source > .680) & (elapsed < requested+reference.recovery),
        'terminal_hold': elapsed >= requested+reference.recovery}
    limits = trace['hard_joint_limits_rad']
    commands = trace['commanded_target_rad']
    excess = np.maximum(limits[:, 0]-commands, 0)+np.maximum(commands-limits[:, 1], 0)
    pelvis_error = trace['qpos'][:, :3]-reference_qpos[:, :3]
    actual_yaw = np.unwrap(_yaw(trace['qpos'][:, 3:7]))
    reference_yaw = np.unwrap(_yaw(reference_qpos[:, 3:7]))
    series = {'elapsed_s': elapsed, 'source_time_s': source,
        'pelvis_xy_reference_error_m': np.linalg.norm(pelvis_error[:, :2], axis=-1),
        'pelvis_yaw_reference_error_deg': np.degrees(_wrapped(actual_yaw-reference_yaw)),
        'pelvis_height_reference_error_m': pelvis_error[:, 2],
        'actual_pelvis_xy_drift_from_launch_m': np.linalg.norm(trace['qpos'][:, :2]-trace['qpos'][0, :2], axis=-1),
        'actual_pelvis_yaw_drift_from_launch_deg': np.degrees(actual_yaw-actual_yaw[0]),
        'solver_elapsed_s': trace['solver_elapsed_s'],
        'foot_vertical_force_N': trace['foot_ground_force_world_N'][..., 2],
        'command_excess_rad': excess, 'actual_glove_pos_w': actual['glove'],
        'reference_glove_pos_w': desired['glove']}
    for key, value in tracking.items():
        series[key+'_body_position_rmse_m'] = np.sqrt(np.mean(value['body_position_error_m']**2, axis=-1))
        series[key+'_body_position_error_m'] = value['body_position_error_m']
        series[key+'_body_orientation_error_rad'] = value['body_orientation_error_rad']
        series[key+'_weighted_score'] = value['weighted_score']
    for name, values in (('actual', actual), ('reference', desired)):
        arm = values['arm_origins'][:, 0]
        series[name+'_left_elbow_extension_deg'] = _angle(arm[:, 0]-arm[:, 1], arm[:, 2]-arm[:, 1])
        series[name+'_inactive_glove_head_distance_m'] = np.linalg.norm(values['glove'][:, 1]-values['head'], axis=-1)
        for part, body in (('torso', 'torso_link'), ('head', 'head_yaw_link')):
            quat = values['body_quaternion'][:, body_names.index(body)]
            up = Rotation.from_quat(quat, scalar_first=True).apply(np.tile([0., 0., 1.], (len(quat), 1)))
            series[name+'_'+part+'_tilt_deg'] = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
    delta_quat = yaw_quat(quat_mul(torch.as_tensor(trace['body_quat_w'][:, 0]),
        quat_inv(sample['body_quat_w'][:, reference_ids[0]])))
    aligned_anchor = torch.as_tensor(trace['body_pos_w'][:, 0]).clone()
    aligned_anchor[:, 2] = sample['body_pos_w'][:, reference_ids[0], 2]
    reference_glove_quat = torch.as_tensor(Rotation.from_matrix(
        desired['glove_rotation'].reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(-1, 2, 4),
        dtype=torch.float32)
    actual_glove_quat = torch.as_tensor(Rotation.from_matrix(
        actual['glove_rotation'].reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(-1, 2, 4),
        dtype=torch.float32)
    for side, index in (('active', 0), ('inactive', 1)):
        relative_glove = aligned_anchor + quat_apply(delta_quat,
            torch.as_tensor(desired['glove'][:, index], dtype=torch.float32)
            - sample['body_pos_w'][:, reference_ids[0]])
        series['launch_registered_'+side+'_glove_error_m'] = np.linalg.norm(
            actual['glove'][:, index]-desired['glove'][:, index], axis=-1)
        series['relative_'+side+'_glove_error_m'] = np.linalg.norm(
            actual['glove'][:, index]-relative_glove.numpy(), axis=-1)
        for label, quat in (('launch_registered', reference_glove_quat[:, index]),
                            ('relative', quat_mul(delta_quat, reference_glove_quat[:, index]))):
            series[label+'_'+side+'_glove_orientation_error_deg'] = np.degrees(
                quat_error_magnitude(quat, actual_glove_quat[:, index]).numpy())
    phases = {}
    for name, mask in phase_masks.items():
        valid_solver = mask & (trace['solver_elapsed_s'] >= 0)
        phases[name] = {'actor_samples': int(mask.sum()),
            'elapsed_range_s': [float(elapsed[mask][0]), float(elapsed[mask][-1])] if mask.any() else None,
            'source_range_s': [float(source[mask][0]), float(source[mask][-1])] if mask.any() else None,
            'tracking': {key: _phase_tracking(value, mask, groups) for key, value in tracking.items()},
            'glove_tracking': {key: _stats(value[mask]) for key, value in series.items()
                               if key.startswith(('launch_registered_', 'relative_')) and '_glove_' in key},
            'motion': {key: _stats(value[mask]) for key, value in series.items() if
                key.startswith(('pelvis_', 'actual_pelvis_', 'actual_left_elbow_', 'reference_left_elbow_',
                                'actual_inactive_', 'reference_inactive_', 'actual_torso_', 'reference_torso_', 'actual_head_', 'reference_head_'))},
            'sampled_foot_vertical_load_N': {side: _stats(series['foot_vertical_force_N'][valid_solver, index])
                                           for index, side in enumerate(('left', 'right'))},
            'command_excess_rad': _stats(excess[mask]),
            'maximum_command_excess_rad_by_joint': excess[mask].max(0).tolist() if mask.any() else None}
    outward = source <= .90
    launch_yaw = reference_yaw[0]
    forward = np.array([np.cos(launch_yaw), np.sin(launch_yaw), 0.])
    amplitude = {}
    for name, values in (('actual', actual), ('reference', desired)):
        glove = values['glove'][:, 0]
        shoulder = values['arm_origins'][:, 0, 0]
        shoulder_reach = (glove-shoulder)@forward
        elbow = series[name+'_left_elbow_extension_deg']
        amplitude[name] = {
            'active_glove_forward_excursion_from_launch_m': float(((glove-glove[0])@forward)[outward].max()),
            'active_glove_relative_shoulder_forward_excursion_m': float((shoulder_reach-shoulder_reach[0])[outward].max()),
            'left_elbow_max_extension_deg': float(elbow[outward].max()),
            'left_elbow_extension_change_from_launch_deg': float((elbow-elbow[0])[outward].max()),
            'active_glove_path_length_through_source_0_90_m': float(np.linalg.norm(np.diff(glove[outward], axis=0), axis=-1).sum())}
    target = np.asarray(row['target_world'])
    dense_elapsed = np.arange(0, max(requested+reference.recovery, elapsed[-1])+.0001, .005)
    dense_source, _, dense_qpos = _reference_at(reference, model, dense_elapsed, requested)
    dense_reference = _fk(model, dense_qpos, body_names)
    dense_glove = dense_reference['glove'][:, 0]
    # A fallen policy must not shorten the expected nominal extension amplitude.
    dense_outward = dense_source <= .90
    arm = dense_reference['arm_origins'][:, 0]
    elbow = _angle(arm[:, 0]-arm[:, 1], arm[:, 2]-arm[:, 1])
    shoulder_reach = (dense_glove-arm[:, 0])@forward
    amplitude['reference'] = {
        'active_glove_forward_excursion_from_launch_m': float(((dense_glove-dense_glove[0])@forward)[dense_outward].max()),
        'active_glove_relative_shoulder_forward_excursion_m': float((shoulder_reach-shoulder_reach[0])[dense_outward].max()),
        'left_elbow_max_extension_deg': float(elbow[dense_outward].max()),
        'left_elbow_extension_change_from_launch_deg': float((elbow-elbow[0])[dense_outward].max()),
        'active_glove_path_length_through_source_0_90_m': float(np.linalg.norm(np.diff(dense_glove[dense_outward], axis=0), axis=-1).sum()),
        'sample_interval_s': .005, 'coverage': 'Full nominal interval even if policy falls early'}
    amplitude['actual']['sample_interval_s'] = .020
    amplitude['actual']['observed_last_source_s'] = float(source[-1])
    amplitude['actual']['observed_through_source_0_90'] = bool(source[-1] >= .90)
    dense_distance = np.linalg.norm(dense_glove-target, axis=-1)
    entries = np.flatnonzero(dense_distance <= .14)
    _, _, exact_qpos = _reference_at(reference, model, [requested], requested)
    at_t = _fk(model, exact_qpos, body_names)['glove'][0, 0]
    actual_distance = np.linalg.norm(actual['glove'][:, 0]-target, axis=-1)
    np.savez_compressed(output/'tracking_series.npz', **series, body_names=np.asarray(body_names),
                        joint_names=trace['joint_names'], target_world=target)
    metrics = {'schema': 'task10c_nominal_control_tracking_v1', 'evaluation_directory': str(evaluation),
        'trace_path': str(trace_path), 'recipe_id': 0, 'snapshot_id': row['snapshot_id'],
        'requested_ttc_s': requested, 'reference_contract': contract, 'imitation_frame': frame,
        'task_rewards_enabled': contract.get('task_rewards_enabled', True),
        'counterfactual_tracking_frames': ['launch_registered', 'relative'],
        'reward_tracking_frame': frame, 'tracking_windows': phases,
        'extension_amplitude_source_0_to_0_90': amplitude,
        'actual_outcome': {key: row.get(key) for key in ('hit', 'actual_ttc_s', 'signed_timing_error_s',
            'fall', 'complete_horizon', 'inactive_encounter', 'inactive_first_encounter_ttc_s')},
        'reference_geometric_encounter': {'physics_grid_s': .005,
            'first_entry_elapsed_s': float(dense_elapsed[entries[0]]) if len(entries) else None,
            'minimum_distance_m': float(dense_distance.min()),
            'closest_elapsed_s': float(dense_elapsed[dense_distance.argmin()]),
            'closest_source_s': float(dense_source[dense_distance.argmin()]),
            'distance_at_requested_ttc_m': float(np.linalg.norm(at_t-target)),
            'note': 'Counterfactual kinematic reference on 5 ms elapsed grid; not actual encounter'},
        'actual_actor_sample_geometry': {'minimum_distance_m': float(actual_distance.min()),
            'distance_at_requested_ttc_linear_interpolation_m': (float(np.interp(requested, elapsed, actual_distance))
                if elapsed[0] <= requested <= elapsed[-1] else None),
            'note': '20 ms actor samples; actual TTC comes only from original 5 ms physics latch. Requested-time distance is null when outside the saved episode; no post-fall extrapolation.'},
        'original_terminal_support_and_commands': {key: value for key, value in row.items()
            if key.startswith(('terminal_hold_', 'terminal_foot_', 'terminal_command_', 'maximum_command_', 'command_excess_actor_', 'command_actor_'))},
        'maximum_command_excess_joint': str(trace['joint_names'][np.argmax(excess.max(0))]),
        'maximum_command_excess_rad': float(excess.max()),
        'joint_names': list(map(str, trace['joint_names'])), 'body_names': body_names,
        'semantics': {'reference_model': 'Scene(make_env_cfg(resolved experiment)).compile(); same selected H2 placement, no standalone renderer placement',
            'fixed_world': 'Origin-free training scene: saved actual qpos/body positions already subtract environment origin; stored reference stays fixed in this frame',
            'relative': 'Existing ongoing actual-pelvis XY / yaw pose alignment; velocities remain world vectors',
            'sampling': 'Actor endpoint kinematics. Foot arrays retain preceding native solver state; initial solver time -1 excluded. Terminal hold summary retains all original 5 ms samples.',
            'phase_definition': 'Strike source [0.30,0.90]; approach [0.12,0.510]; extension (0.510,0.680]; withdrawal after0.680 until scheduled reference end; terminal hold after T+0.910',
            'statistics': 'One saved deterministic nominal trajectory, not an independent evaluation sample or acceptance gate',
            'selection': 'Diagnostic only. No automatic checkpoint selection; extension amplitude exposes standing-still solutions.'}}
    strike_range = phases['strike_window']['elapsed_range_s']
    _plot(output, series, requested, row.get('actual_ttc_s'), strike_range)
    if render:
        metrics['video'] = _render(output, model, trace['qpos'], reference_qpos, elapsed,
                                   source, target, row, metrics)
    _json(output/'metrics.json', metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    result = analyze_evaluation(args.evaluation_dir, args.output_dir, args.render)
    print(json.dumps({'output': str(args.output_dir or args.evaluation_dir/'nominal_control_analysis'),
                      'imitation_frame': result['imitation_frame'],
                      'actual_outcome': result['actual_outcome']}))


if __name__ == '__main__':
    main()
