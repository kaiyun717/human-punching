"""Focused CPU kinematic checks and rescore of the retained jab diagnosis.

This module never steps a simulator or runs a policy. The rescore uses the
recorded 96 actor transitions and reconstructs link velocities at their existing
reward boundaries from the saved 5-ms states. It does not predict a trained
policy's performance under the corrected objective.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import TorchReference
from .events import guard_weight, reference_clock, scheduled_normalizers
from .imitation import (
    COMMAND_EXCESS_SCALE_RAD, COMMAND_EXCESS_SMOOTH_L1_BETA,
    H2_TRACKED_BODY_NAMES, IMITATION_STDS, IMITATION_WEIGHTS,
    align_reference_poses, command_excess_terms, imitation_rates,
    named_joint_groups, whole_body_errors, whole_body_scores,
)
from .robot import H2_XML, JOINT_NAMES, ROOT, joint_addresses

PERTURBED_JOINTS = (
    'head_pitch_joint', 'head_yaw_joint',
    'left_knee_joint', 'right_knee_joint',
    'left_ankle_roll_joint', 'left_ankle_pitch_joint',
    'right_ankle_roll_joint', 'right_ankle_pitch_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
)


def tensor(value):
    return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float64)


def scalar_dict(values):
    return {name: float(value.item()) for name, value in values.items()}


def body_state(model, qpos, qvel, velocity_qpos=None, velocity_qvel=None):
    """CPU FK only; XBODY origins, world vectors, and wxyz quaternions.

    Optional velocity states reproduce mjlab's pre-solve cvel/subtree_com plus
    post-integration link origins, rather than silently substituting fresh
    post-integration velocities for the recorded reward boundary.
    """
    qpos, qvel = np.atleast_2d(qpos), np.atleast_2d(qvel)
    velocity_qpos = qpos if velocity_qpos is None else np.atleast_2d(velocity_qpos)
    velocity_qvel = qvel if velocity_qvel is None else np.atleast_2d(velocity_qvel)
    ids = [model.body(name).id for name in H2_TRACKED_BODY_NAMES]
    anchor = model.body('pelvis').id
    data = mujoco.MjData(model)
    result = [[] for _ in range(4)]
    for pose, vel, vpose, vvel in zip(qpos, qvel, velocity_qpos, velocity_qvel, strict=True):
        data.qpos[:] = vpose
        data.qvel[:] = vvel
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        cvel = data.cvel[ids].copy()
        center = data.subtree_com[anchor].copy()
        data.qpos[:] = pose
        data.qvel[:] = vel
        mujoco.mj_kinematics(model, data)
        position = data.xpos[ids].copy()
        linear = cvel[:, 3:] - np.cross(cvel[:, :3], center[None] - position)
        for destination, value in zip(result, (position, data.xquat[ids].copy(), linear, cvel[:, :3]), strict=True):
            destination.append(value)
    return tuple(tensor(value) for value in result)


def sensitivity_checks(model, terminal_qpos):
    """Each named perturbation must affect an appropriate restored term."""
    qa, va = joint_addresses(model)
    zero_velocity = np.zeros(model.nv)
    base = body_state(model, terminal_qpos, zero_velocity)
    identity = whole_body_scores(*base, *base)
    checks = {'reference_identity': scalar_dict(identity), 'joint_perturbations': {}}
    for joint in PERTURBED_JOINTS:
        index = JOINT_NAMES.index(joint)
        qpos = terminal_qpos.copy()
        qpos[qa[index]] += 0.1
        changed = body_state(model, qpos, zero_velocity)
        qvel = zero_velocity.copy()
        qvel[va[index]] = 1.0
        moving = body_state(model, terminal_qpos, qvel)
        aligned_pos, aligned_quat = align_reference_poses(changed[0], changed[1], base[0], base[1])
        position_error = (changed[0] - aligned_pos).norm(dim=-1)[0]
        orientation_error = quat_error_magnitude(changed[1], aligned_quat)[0]
        scores = whole_body_scores(*changed, *base)
        velocity_scores = whole_body_scores(*moving, *base)
        checks['joint_perturbations'][joint] = {
            'position_perturbation_rad': 0.1, 'velocity_perturbation_rad_s': 1.0,
            'static_scores': scalar_dict(scores), 'velocity_scores': scalar_dict(velocity_scores),
            'aligned_body_position_error_m': dict(zip(H2_TRACKED_BODY_NAMES, position_error.tolist(), strict=True)),
            'aligned_body_orientation_error_rad': dict(zip(H2_TRACKED_BODY_NAMES, orientation_error.tolist(), strict=True)),
            'orientation_responds': bool(scores['body_orientation'].item() < 1.0 - 1e-8),
            'angular_velocity_responds': bool(velocity_scores['body_angular_velocity'].item() < 1.0 - 1e-8),
        }
    feet = {}
    for side in ('left', 'right'):
        changed = [value.clone() for value in base]
        index = H2_TRACKED_BODY_NAMES.index(f'{side}_ankle_pitch_link')
        changed[0][:, index, 2] += 0.1
        scores = whole_body_scores(*changed, *base)
        feet[side] = {'isolated_body_height_error_m': 0.1, 'scores': scalar_dict(scores),
                      'position_responds': bool(scores['body_position'].item() < 1.0 - 1e-8)}
    checks['final_foot_body_perturbations'] = feet
    checks['passed'] = (all(abs(x - 1.0) < 1e-10 for x in checks['reference_identity'].values())
                        and all(row['orientation_responds'] and row['angular_velocity_responds']
                                for row in checks['joint_perturbations'].values())
                        and all(row['position_responds'] for row in feet.values()))
    return checks


def rescore(diagnosis: Path, reference_path: Path):
    actor_path, physics_path = diagnosis / 'instrumented_actor.npz', diagnosis / 'instrumented_physics.npz'
    with np.load(actor_path, allow_pickle=False) as saved:
        actor = {name: saved[name].copy() for name in saved.files}
    with np.load(physics_path, allow_pickle=False) as saved:
        physics = {name: saved[name].copy() for name in saved.files}
    mapping = json.loads((diagnosis / 'native_mapping.json').read_text())
    identity = json.loads((diagnosis / 'replay_identity.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    dt, physics_dt = identity['actor_dt'], identity['physics_dt']
    decimation = round(dt / physics_dt)
    count = len(actor['elapsed_s']) - 1
    if len(physics['elapsed_end_s']) != count * decimation:
        raise ValueError('Saved physics trace does not cover every actor transition')
    end_indices = np.arange(1, count + 1) * decimation - 1
    pre_indices = end_indices - 1
    np.testing.assert_allclose(actor['elapsed_s'][1:], physics['elapsed_end_s'][end_indices], atol=1e-6)
    np.testing.assert_array_equal(actor['qpos'][1:], physics['qpos_after_integration'][end_indices])
    reconstructed = body_state(model, actor['qpos'][1:], actor['qvel'][1:],
                               physics['qpos_after_integration'][pre_indices],
                               physics['qvel_after_integration'][pre_indices])
    actual_ids = [mapping['body_names'].index(name) for name in H2_TRACKED_BODY_NAMES]
    recorded_actual = (tensor(actor['body_pos_world'][1:, actual_ids]),
                       tensor(actor['body_quat_world_wxyz'][1:, actual_ids]))
    position_parity = float((reconstructed[0] - recorded_actual[0]).abs().max())
    quaternion_parity = float(quat_error_magnitude(reconstructed[1], recorded_actual[1]).max())
    if position_parity > 1e-5 or quaternion_parity > 1e-5:
        raise ValueError('CPU reconstruction does not match recorded native link poses')
    # Recorded poses are authoritative; reconstructed velocities follow the live
    # link-velocity accessor's phase convention, using those same link positions.
    actual = (*recorded_actual, *reconstructed[2:])
    reference = TorchReference(reference_path, 'cpu', mapping['joint_names'], family='jab')
    elapsed = torch.as_tensor(actor['elapsed_s'][1:], dtype=torch.float32)
    requested = torch.full_like(elapsed, identity['recipe']['requested_ttc'])
    source_time, rate = reference_clock(elapsed, requested, reference.strike_time,
                                       reference.end_time, reference.recovery)
    sampled = reference.sample(source_time, rate)
    ids = [reference.body_names.index(name) for name in H2_TRACKED_BODY_NAMES]
    ref = tuple(sampled[f'body_{name}_w'][:, ids].to(torch.float64)
                for name in ('pos', 'quat', 'lin_vel', 'ang_vel'))
    recorded_ref_ids = [mapping['reference_body_names'].index(name) for name in H2_TRACKED_BODY_NAMES]
    reference_position_parity = float((ref[0] - tensor(actor['reference_body_pos_world'][1:, recorded_ref_ids])).abs().max())
    reference_quaternion_parity = float(quat_error_magnitude(
        ref[1], tensor(actor['reference_body_quat_world_wxyz'][1:, recorded_ref_ids])).max())
    if reference_position_parity > 1e-6 or reference_quaternion_parity > 1e-6:
        raise ValueError('Reference resampling does not match recorded reward boundaries')
    horizon, _, guard_z = scheduled_normalizers(requested, reference.recovery, dt)
    schedule = guard_weight(elapsed, requested, reference.recovery)
    scores = whole_body_scores(*actual, *ref)
    errors = whole_body_errors(*actual, *ref)
    rates = imitation_rates(scores, horizon, schedule, guard_z)
    base_rates = imitation_rates(scores, horizon, schedule, guard_z, recovery_budget=0.0)
    recovery_rates = imitation_rates(scores, horizon, schedule, guard_z, imitation_budget=0.0)
    names = mapping['joint_names']
    groups = named_joint_groups(names)
    limits = np.asarray([model.jnt_range[model.joint(name).id] for name in names])
    np.testing.assert_allclose(limits, mapping['hard_joint_limits'], atol=2e-7)
    targets = tensor(actor['command_actual_pd_position_target'])
    np.testing.assert_array_equal(actor['command_actual_pd_position_target'], actor['joint_pos_target'][1:])
    command_groups = command_excess_terms(targets, tensor(limits), groups)
    command_rate = -sum(command_groups.values()) / horizon
    old = {str(name): np.diff(actor['component_totals'][:, i])
           for i, name in enumerate(actor['component_names'])}
    # Reuse the retained legacy implementation to verify velocity reconstruction
    # against an actually recorded reward, not merely CPU geometry agreement.
    from .mdp import style as legacy_style
    data = SimpleNamespace(body_link_pos_w=actual[0], body_link_quat_w=actual[1],
                           body_link_lin_vel_w=actual[2], root_link_lin_vel_w=actual[2][:, 0])
    command = SimpleNamespace(
        robot=SimpleNamespace(data=data, body_names=H2_TRACKED_BODY_NAMES),
        reference=SimpleNamespace(body_names=H2_TRACKED_BODY_NAMES),
        torso_index=H2_TRACKED_BODY_NAMES.index('torso_link'),
        reference_torso_index=H2_TRACKED_BODY_NAMES.index('torso_link'), root_index=0,
        ref=dict(zip(('body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'), ref, strict=True)),
        horizon=horizon, accumulate=lambda name, value: value * identity['reward_scales'][name])
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    reconstructed_style = legacy_style(env).numpy() * dt
    legacy_style_interval_max_error = float(np.abs(reconstructed_style - old['style']).max())
    if legacy_style_interval_max_error > 1e-5:
        raise ValueError('Reconstructed link velocities do not reproduce recorded legacy style')
    retained = {name: values for name, values in old.items() if name not in ('style', 'balance', 'guard')}
    new_components = {**retained, **{f'imitation_{name}': value.numpy() * dt for name, value in rates.items()},
                      'command_excess': command_rate.numpy() * dt}
    hold_start = float(requested[0]) + reference.recovery
    windows = {}
    for window, mask in (
        ('whole_episode', np.ones(count, dtype=bool)),
        ('actor_end_at_or_after_hold_start', actor['elapsed_s'][1:] >= hold_start - 1e-6),
    ):
        old_totals = {name: float(values[mask].sum()) for name, values in old.items()}
        new_totals = {name: float(values[mask].sum()) for name, values in new_components.items()}
        windows[window] = {'actor_intervals': int(mask.sum()), 'old_components': old_totals,
                           'new_components': new_totals, 'old_total': sum(old_totals.values()),
                           'new_total': sum(new_totals.values())}
    aligned_pos, aligned_quat = align_reference_poses(actual[0], actual[1], ref[0], ref[1])
    terminal_body_error = {
        name: {'position_m': float((actual[0][-1, i] - aligned_pos[-1, i]).norm()),
               'orientation_rad': float(quat_error_magnitude(actual[1][-1, i], aligned_quat[-1, i]))}
        for i, name in enumerate(H2_TRACKED_BODY_NAMES)}
    result = {
        'scope': 'Counterfactual objective rescore of existing saved states, not a new policy rollout.',
        'source_actor_sha256': hashlib.sha256(actor_path.read_bytes()).hexdigest(),
        'source_physics_sha256': hashlib.sha256(physics_path.read_bytes()).hexdigest(),
        'reference_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        'actor_transitions': count, 'new_simulator_steps': 0, 'optimizer_updates': 0,
        'actor_dt_s': dt, 'physics_dt_s': physics_dt, 'horizon_s': float(horizon[0]),
        'requested_ttc_s': float(requested[0]), 'terminal_hold_start_s': hold_start,
        'body_names': list(H2_TRACKED_BODY_NAMES), 'weights': IMITATION_WEIGHTS, 'stds': IMITATION_STDS,
        'budgets': {'whole_episode': 7.0, 'recovery': 6.0, 'command_excess_weight': 1.0},
        'parity': {'native_position_max_abs_m': position_parity,
                   'native_quaternion_max_error_rad': quaternion_parity,
                   'reference_position_max_abs_m': reference_position_parity,
                   'reference_quaternion_max_error_rad': reference_quaternion_parity,
                   'legacy_style_actor_interval_max_abs_error': legacy_style_interval_max_error,
                   'legacy_style_reconstructed_total': float(reconstructed_style.sum()),
                   'legacy_style_recorded_total': float(old['style'].sum()),
                   'recorded_commands_match_actor_successor_targets': True,
                   'named_model_hard_limits_match_recorded_native_limits': True},
        'velocity_timing': 'Last physics solve pre-integration cvel/subtree_com combined with post-integration body origins, matching existing mjlab EntityData and Task10C callback. CPU kinematics/comPos/comVel only; no mj_step or solver/contact execution.',
        'component_contributions': {
            name: {'imitation': float(base_rates[name].sum() * dt),
                   'recovery': float(recovery_rates[name].sum() * dt),
                   'total': float(rates[name].sum() * dt),
                   'terminal_score': float(scores[name][-1]),
                   'terminal_squared_error': float(errors[name][-1])}
            for name in scores},
        'command_excess': {'scale_rad': COMMAND_EXCESS_SCALE_RAD,
                           'smooth_l1_beta_dimensionless': COMMAND_EXCESS_SMOOTH_L1_BETA,
                           'aggregation': 'sum of six named group means; no division by six',
                           'group_contributions': {name: float((-values / horizon).sum() * dt)
                                                   for name, values in command_groups.items()},
                           'total': float(command_rate.sum() * dt)},
        'terminal_aligned_body_errors': terminal_body_error, 'reward_windows': windows,
        'sensitivity': sensitivity_checks(model, reference.arrays['qpos'][-1].numpy()),
    }
    if not result['sensitivity']['passed']:
        raise AssertionError('Required head/knee/ankle/torso/foot reward sensitivity failed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnosis', type=Path, default=ROOT / 'outputs/task10c/jab/diagnosis_20261022_01')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.relative_to(ROOT.resolve() / 'outputs/task10c/jab')
    report_path = destination / 'reward_checks.json'
    if report_path.exists():
        raise FileExistsError('Preserve the previous check result; choose a new output directory')
    selection = json.loads((ROOT / 'configs/task10c/references.json').read_text())
    result = rescore(args.diagnosis.resolve(), (ROOT / selection['jab']).resolve())
    destination.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(report_path), 'sensitivity_passed': result['sensitivity']['passed'],
                      'components': result['component_contributions'],
                      'command_excess': result['command_excess'],
                      'reward_windows': result['reward_windows']}, indent=2))


if __name__ == '__main__':
    main()
