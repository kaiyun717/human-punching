"""Read-only re-audit of recorded observer qualification; never integrates physics.

The original audit assumed zero actions and compared quaternion components.
This verifier preserves that receipt, checks the actual sinusoidal commands,
compares rotations, and explicitly records contact-patch boundary uncertainty.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from human_punching.wbc.cycle_trace import TraceBindings, derive_kinematics, classify_contacts
from human_punching.wbc.native_probe import audit_probe_arrays
from human_punching.wbc.provenance import PROJECT_ROOT, sha256_file, writable_path, write_json
from human_punching.wbc.runtime.bundle import load_runtime_bundle


def rotation_error_rad(a, b):
    """Sign-invariant geodesic SO(3) error, stable near equal orientations."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.shape[-1] != 4 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Matching finite wxyz quaternions required")
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    if np.any(na == 0) or np.any(nb == 0):
        raise ValueError("Zero quaternion cannot represent a rotation")
    a, b = a/na[..., None], b/nb[..., None]
    b = np.where((np.sum(a*b, axis=-1) < 0)[..., None], -b, b)
    # For unit quaternions ||a-b||/||a+b|| = tan(relative_angle/4).
    return 4*np.arctan2(np.linalg.norm(a-b, axis=-1), np.linalg.norm(a+b, axis=-1))


def command_drive_audit(native, bundle, fixture):
    """Reconstruct recorded residual, 200 Hz references, native drive and forces."""
    if fixture not in ('guard', 'punch'):
        raise ValueError("Explicit guard or punch fixture required")
    a = {name: np.asarray(value) for name, value in native.items()}
    rows, worlds, _ = a['raw_action'].shape
    steps = (rows-1)//4
    original = audit_probe_arrays(a, policy_steps=steps, physics_dt_s=.005,
                                  effort_limits=np.asarray(bundle.body['effort']))
    checks = dict(original['checks'])
    # Replace only the inapplicable zero-action premise. Keep its recorded value.
    zero_premise = checks.pop('zero_learned_residual')
    action = a['raw_action'].astype(float)
    tick = np.maximum(np.arange(rows)-1, 0)
    actor_step = tick//4
    # Match each explicitly FP32 operation; CPU/GPU sin differ by small ULPs.
    phase = np.arange(48, dtype=np.float32)[None, :]*np.float32(.17)
    phase = phase + actor_step[:, None].astype(np.float32)*np.float32(.11)
    intended = np.float32(.02)*np.sin(phase)
    intended[0] = 0
    action_error = float(np.max(abs(action-intended[:, None, :])))
    held = all(np.array_equal(a['raw_action'][k:k+4], np.broadcast_to(a['raw_action'][k], (4, worlds, 48)))
               for k in range(1, rows, 4))
    checks.update(recorded_nonzero_command=action_error <= 1e-7,
                  residual_held_for_four_integrations=held,
                  bounded_residual=bool(np.max(abs(action)) <= .02000001))
    qref, vref, ffref = (np.empty_like(action) for _ in range(3))
    ids = a['trajectory_id'][0].astype(int)
    for index, clip in enumerate(bundle.clips):
        with np.load(bundle.path(clip['master_path']), allow_pickle=False) as master:
            selected = ids == index
            if not selected.any():
                continue
            frames = np.zeros(rows, int) if fixture == 'guard' else np.minimum(tick, len(master['qpos'])-1)
            qref[:, selected] = master['qpos'].astype(np.float32)[frames, None, 7:]
            vref[:, selected] = master['qvel'].astype(np.float32)[frames, None, 6:]
            ffref[:, selected] = master['joint_feedforward'].astype(np.float32)[frames, None]
            if fixture == 'guard':
                vref[:, selected] = 0; ffref[:, selected] = 0
            else:
                vref[np.ix_(tick >= len(master['qpos'])-1, selected)] = 0
    limits = np.asarray(bundle.body['joint_limits_rad'], dtype=np.float32).astype(float)
    kp, kd, caps = (np.asarray(bundle.body[key], float) for key in ('kp', 'kd', 'effort'))
    target = np.clip(qref+.25*np.clip(action, -1, 1), limits[:, 0], limits[:, 1])
    drive = kp*a['joint_position_target']+kd*a['joint_velocity_target']+ffref
    q = np.concatenate((a['qpos'][:1, :, 7:], a['qpos'][:-1, :, 7:])).astype(float)
    v = np.concatenate((a['qvel'][:1, :, 6:], a['qvel'][:-1, :, 6:])).astype(float)
    force = np.clip(a['ctrl'].astype(float)-kp*q-kd*v, -caps, caps)
    errors = {'command': action_error,
        'target_rad': float(np.max(abs(a['joint_position_target']-target))),
        'reference_velocity_rad_s': float(np.max(abs(a['joint_velocity_target']-vref))),
        'feedforward_nm': float(np.max(abs(a['joint_effort_target']-ffref))),
        'affine_drive_nm': float(np.max(abs(a['ctrl']-drive))),
        'capped_force_nm': float(np.max(abs(a['actuator_force']-force)))}
    checks.update(reference_and_residual_target=errors['target_rad'] <= 2e-6,
        reference_velocity=errors['reference_velocity_rad_s'] <= 2e-6,
        reference_feedforward=errors['feedforward_nm'] <= 5e-6,
        unclipped_affine_drive=errors['affine_drive_nm'] <= 2e-4,
        preceding_interval_capped_force=errors['capped_force_nm'] <= 2e-3)
    return {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
        'maximum_errors': errors, 'original_zero_action_premise': zero_premise,
        'action_recipe': '.02*sin(float32(joint_index)*.17+float32(policy_step)*.11)',
        'force_semantics': 'row k uses preceding endpoint q/v; row zero is initialization forward',
        'inherited_audit': original}


def patch_boundary_record(points, cpu_anchor, gpu_anchor, *, radius=.065):
    """Bound only actual anchor error and FP32 distance/threshold arithmetic.

    The same recorded native contact positions enter both comparisons. There is
    no independent contact-position estimate or altered patch radius.
    """
    points = np.asarray(points, np.float32)[:, :2]
    ca, ga = np.asarray(cpu_anchor, float)[:2], np.asarray(gpu_anchor, np.float32)[:2]
    if len(points) == 0:
        raise ValueError("Boundary analysis requires loaded foot-floor contacts")
    delta_exact = points.astype(float)-ga.astype(float)
    delta_f32 = points-ga
    cpu_d = np.linalg.norm(points.astype(float)-ca, axis=-1)
    gpu_input_d = np.linalg.norm(delta_exact, axis=-1)
    subtraction_error = np.linalg.norm(delta_exact-delta_f32.astype(float), axis=-1)
    eps = np.finfo(np.float32).eps
    gamma3 = 3*eps/(1-3*eps)
    norm_rounding = gamma3*np.linalg.norm(delta_f32.astype(float), axis=-1)
    anchor_error = float(np.linalg.norm(ca-ga.astype(float)))
    threshold_rounding = abs(float(np.float32(radius))-radius)
    bound = anchor_error+subtraction_error+norm_rounding+threshold_rounding
    # Minimum distance decides the OR over all loaded contacts for this foot.
    margin = float(cpu_d.min()-radius)
    worst_bound = float(bound.max())
    return {'cpu_minimum_distance_m': float(cpu_d.min()),
        'cpu_distance_minus_radius_m': margin,
        'gpu_anchor_minimum_distance_float64_m': float(gpu_input_d.min()),
        'gpu_anchor_distance_minus_radius_m': float(gpu_input_d.min()-radius),
        'anchor_xy_error_m': anchor_error,
        'native_contact_position_difference_m': 0.,
        'native_contact_position_semantics': 'identical saved native positions; no independent geometry estimate',
        'maximum_subtraction_rounding_m': float(subtraction_error.max()),
        'maximum_norm_rounding_bound_m': float(norm_rounding.max()),
        'radius_float32_rounding_m': threshold_rounding,
        'boundary_uncertainty_bound_m': worst_bound,
        'boundary_uncertain': abs(margin) <= worst_bound,
        'classification': 'boundary_uncertain' if abs(margin) <= worst_bound else ('interior' if margin < 0 else 'exterior')}


def offline_audit(model, raw, states, bindings=TraceBindings()):
    maxima, rotations, norms = {}, {}, {}
    names = ('support_contact', 'support_patch_contact', 'forbidden_collision', 'human_g1_collision')
    mismatches = {name: 0 for name in names}
    boundary_rows = []; interior_exterior_mismatches = 0
    classified_counts = {'interior': 0, 'exterior': 0, 'boundary_uncertain': 0, 'no_loaded_contact': 0, 'initial_anchor_unrecorded': 0}
    feet = [model.geom(bindings.prefix+name).id for name in bindings.foot_geoms]
    floor = {model.geom(name).id for name in bindings.floor_geoms}
    offsets = raw['contact_offsets']
    for world in range(raw['qpos'].shape[1]):
        expected = derive_kinematics(model, raw['qpos'][:, world], raw['qvel'][:, world], bindings)
        for name, value in expected.items():
            if name not in states:
                raise ValueError(f'Missing observer FK field {name}')
            actual = states[name][:, world]
            if 'quat' in name:
                rotations[name] = max(rotations.get(name, 0.), float(rotation_error_rad(value[1:], actual).max()))
                norms[name] = max(norms.get(name, 0.), float(abs(np.linalg.norm(actual.astype(float), axis=-1)-1).max()))
            else:
                maxima[name] = max(maxima.get(name, 0.), float(abs(value[1:]-actual).max()))
        for k in range(1, len(raw['qpos'])):
            lo, hi = offsets[k:k+2]
            owned = raw['contact_worldid'][lo:hi] == world
            contact = {name: raw['contact_'+name][lo:hi][owned] for name in ('geom', 'pos', 'dist', 'force')}
            classified = classify_contacts(model, contact, expected['support_anchor_position_w'][k-1], bindings)
            for name in names:
                mismatches[name] += int(np.count_nonzero(classified[name] != states[name][k-1, world]))
            for side, foot in enumerate(feet):
                observed = bool(states['support_patch_contact'][k-1, world, side])
                wanted = bool(classified['support_patch_contact'][side])
                loaded = np.array([foot in pair and bool(set(pair).intersection(floor))
                    and force[0] >= bindings.minimum_normal_force_n
                    for pair, force in zip(contact['geom'], contact['force'], strict=True)], bool)
                if not loaded.any():
                    classified_counts['no_loaded_contact'] += 1
                    interior_exterior_mismatches += int(observed != wanted)
                    continue
                if k == 1:
                    # The observer trace starts after integration; do not invent
                    # its initial private anchor. Exact Boolean agreement is required.
                    classified_counts['initial_anchor_unrecorded'] += 1
                    interior_exterior_mismatches += int(observed != wanted)
                    continue
                row = patch_boundary_record(contact['pos'][loaded],
                    expected['support_anchor_position_w'][k-1, side],
                    states['support_anchor_position_w'][k-2, world, side], radius=bindings.patch_radius_m)
                classified_counts[row['classification']] += 1
                mismatch = observed != wanted
                if row['boundary_uncertain'] or mismatch:
                    boundary_rows.append({'world': world, 'physics_tick': k, 'side': side,
                        'cpu_patch_contact': wanted, 'gpu_patch_contact': observed,
                        'boolean_mismatch': mismatch, **row})
                if mismatch and not row['boundary_uncertain']:
                    interior_exterior_mismatches += 1
    checks = {'fk_position_velocity': all(v <= 2e-5 for v in maxima.values()),
        'rotation_geodesic': all(v <= 2e-5 for v in rotations.values()),
        'quaternion_norm': all(v <= 2e-5 for v in norms.values()),
        'other_contact_booleans_exact': all(mismatches[n] == 0 for n in names if n != 'support_patch_contact'),
        'interior_exterior_patch_classification_exact': interior_exterior_mismatches == 0,
        'all_patch_mismatches_explicitly_bounded': sum(r['boolean_mismatch'] and r['boundary_uncertain'] for r in boundary_rows) == mismatches['support_patch_contact']}
    return {'status': 'passed_with_boundary_precision_limit' if all(checks.values()) and mismatches['support_patch_contact'] else ('passed' if all(checks.values()) else 'failed'),
        'passed': all(checks.values()), 'checks': checks, 'maximum_absolute_fk_error': maxima,
        'maximum_rotation_error_rad': rotations, 'maximum_quaternion_norm_error': norms,
        'fk_tolerance': 2e-5, 'rotation_tolerance_rad': 2e-5, 'quaternion_norm_tolerance': 2e-5,
        'contact_boolean_mismatches': mismatches, 'patch_radius_m': bindings.patch_radius_m,
        'patch_radius_changed': False, 'patch_classifications': classified_counts,
        'interior_exterior_mismatches': interior_exterior_mismatches,
        'boundary_records': boundary_rows, 'contact_semantics': 'preceding interval contacts at previous endpoint anchors'}


def _identity(path):
    return {'path': str(path.resolve().relative_to(PROJECT_ROOT)), 'sha256': sha256_file(path)}


def reverify(packet_path, output):
    import mujoco
    packet_path = Path(packet_path).resolve(strict=True)
    packet = json.loads(packet_path.read_text())
    run = PROJECT_ROOT/packet['output']
    original = json.loads((run/'report.json').read_text())
    bundle = load_runtime_bundle(PROJECT_ROOT/packet['bundle'])
    # The physical implementation and actual reference assets must still match
    # what executed; other collaborators may add unrelated source modules.
    bound_files = {name: checksum for name, checksum in packet['files'].items()
        if name.startswith(packet['bundle']+'/') or name in (
            'src/human_punching/wbc/telemetry/kinematics.py',
            'src/human_punching/wbc/telemetry/contacts.py',
            'src/human_punching/wbc/telemetry/streams.py',
            'src/human_punching/wbc/telemetry/runtime_telemetry.py',
            'src/human_punching/wbc/runtime/action.py',
            'src/human_punching/wbc/runtime/command.py',
            'src/human_punching/wbc/runtime/body.py',
            'src/human_punching/wbc/cycle_trace.py')}
    for name, checksum in bound_files.items():
        if sha256_file(PROJECT_ROOT/name) != checksum:
            raise ValueError(f'Executed physical/reference source changed: {name}')
    destination = writable_path(output); destination.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'human_wbc_observer_posthoc_qualification_v1',
        'original_packet': _identity(packet_path), 'original_report': _identity(run/'report.json'),
        'original_execution': _identity(packet_path.parent/'execution.json'),
        'bundle_manifest': _identity(bundle.root/'manifest.json'), 'runtime': original['runtime'],
        'device': original['device'], 'training': False, 'acceptance_qualified': False,
        'new_integration_steps': 0, 'original_status': original['status'],
        'strict_rollout_status': original['strict_rollout_status'], 'fixtures': {},
        'verifier_source': _identity(Path(__file__)),
        'verified_physical_reference_files': bound_files,
        'executed_source_sha256': packet['files']}
    for fixture in ('guard', 'punch'):
        rows = []
        for label in ('a', 'a_repeat', 'b'):
            directory = run/(fixture+'_'+label)
            old = json.loads((directory/'report.json').read_text())
            inputs = {name: _identity(directory/name) for name in ('native_trace.npz', 'raw_trace.npz', 'observer_trace.npz', 'compiled_scene.mjb', 'report.json')}
            for name in ('native_trace', 'raw_trace', 'observer_trace'):
                if inputs[name+'.npz']['sha256'] != old['hashes'][name]:
                    raise ValueError(f'Original recorded hash mismatch: {fixture}/{label}/{name}')
            if inputs['compiled_scene.mjb']['sha256'] != old['compiled_scene_sha256']:
                raise ValueError('Compiled scene changed')
            with np.load(directory/'native_trace.npz', allow_pickle=False) as data:
                controller = command_drive_audit(data, bundle, fixture)
            parity = None
            if label == 'b':
                with np.load(directory/'raw_trace.npz', allow_pickle=False) as raw, np.load(directory/'observer_trace.npz', allow_pickle=False) as observed:
                    parity = offline_audit(mujoco.MjModel.from_binary_path(str(directory/'compiled_scene.mjb')),
                                           dict(raw), dict(observed))
            rows.append({'label': label, 'inputs': inputs, 'controller_audit': controller,
                'offline_parity': parity, 'nonmutation': old['nonmutation'],
                'stream_capture_roundtrip': old.get('stream_capture_roundtrip'),
                'policy_steps': old['policy_steps'], 'physics_samples': old['physics_samples']})
        passed = all(row['controller_audit']['status'] == 'passed' for row in rows)
        passed &= rows[-1]['offline_parity']['passed'] and rows[-1]['nonmutation']['passed']
        report['fixtures'][fixture] = {'instrumentation_passed': passed, 'runs': rows,
            'baseline_determinism': original['fixtures'][fixture]['baseline_determinism'],
            'observer_comparison': original['fixtures'][fixture]['observer_comparison'],
            'steady_step_wall_overhead_ratio': original['fixtures'][fixture]['steady_step_wall_overhead_ratio'],
            'overhead_semantics': original['fixtures'][fixture]['overhead_semantics']}
    report['instrumentation_status'] = 'passed' if all(v['instrumentation_passed'] for v in report['fixtures'].values()) else 'failed'
    report['status'] = 'instrumentation_passed_rollouts_nondeterministic' if report['instrumentation_status'] == 'passed' else 'failed'
    report['limitations'] = ['Strict A/A and A/B full trajectories are not bitwise reproducible; no trajectory-equivalence claim.',
        'FP32 support-patch classification is ambiguous at explicitly quantified 0.065 m boundary cases.',
        'Instrumentation only: no policy, reference-domain, physical guard, readiness/launch bridge or impact acceptance.',
        'Measured overhead includes expensive diagnostic witnesses and CPU trace copies.']
    write_json(destination/'report.json', report)
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); report = reverify(args.packet, args.output)
    print(json.dumps({'status': report['status'], 'output': args.output}))
    return 0 if report['instrumentation_status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
