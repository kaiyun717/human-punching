"""Bounded native audit of explicitly ineligible frame-zero training starts.

The inherited environment owns every reset and integration. This diagnostic has
no actor, optimizer, assistance, reference alteration or acceptance promotion.
Strict readiness and training exposure are reported as different start types.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from human_punching.wbc.native_probe import _copy, _require_allocated_cuda
from human_punching.wbc.native_cycle_probe import allocated_device_metadata, save_compiled_model, _save_npz
from human_punching.wbc.provenance import writable_path, write_json, sha256_file, runtime_record
from human_punching.wbc.runtime.bundle import load_runtime_bundle
from human_punching.wbc.runtime.imitation_rewards import configure_fresh_imitation_rewards, fresh_imitation_health
from human_punching.wbc.runtime.training_initialization import TrainingInitializationCfg
from human_punching.wbc.runtime.engineering_recipe import LEGACY_RECIPE, REVISED_RECIPE
from human_punching.wbc.runtime.task_rewards import DEFAULT_WEIGHTS
from .cycle_qualification import (ENVIRONMENTS, QualificationRecorder, build_cfg,
    native_stream_scope, audit_arrays, stack)
from .cycle_runtime import JsonReceiptSink, attach_cycle_runtime, get_cycle_runtime, cycle_condition
from .readiness_clock import ReadinessConfig, Phase
from .runtime_telemetry import get_observer
from .snapshot_providers import make_snapshot_providers
from .streams import verify_native_stream_capture


def array_identity(values):
    """Hash full immutable reference arrays, including dtype and shape."""
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = np.ascontiguousarray(_copy(value) if isinstance(value, torch.Tensor) else value)
        digest.update(name.encode()); digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def build_initialization_cfg(bundle, recorder, *, probability):
    if probability not in (0., .75):
        raise ValueError('Qualification covers only strict default and explicit revised recipe')
    recipe = REVISED_RECIPE if probability else LEGACY_RECIPE
    settings = recipe.training
    cfg = build_cfg(bundle, recorder, readiness=ReadinessConfig())
    cfg.commands['motion'].training_initialization = TrainingInitializationCfg(**settings.initialization_parameters())
    for name, weight in settings.task_reward_weights(DEFAULT_WEIGHTS).items():
        term = cfg.rewards['human_'+name]
        if term.params != {'name': name} or term.weight != DEFAULT_WEIGHTS[name]:
            raise ValueError('Qualification fixture must start from unchanged historical task objectives')
        term.weight = weight
        if settings.guard_joint_posture_metric != 'rms':
            term.params = {**term.params, 'guard_joint_posture_metric': settings.guard_joint_posture_metric}
    fresh_contract = configure_fresh_imitation_rewards(cfg)
    return cfg, recipe, fresh_contract


class InitializationRecorder(QualificationRecorder):
    """Add start provenance to the existing independently audited native rows."""

    def before_interval(self, env):
        super().before_interval(env)
        command = env.command_manager.get_term('motion')
        bridge = get_cycle_runtime(env)
        r, initializer = bridge.readiness, bridge.training_initializer
        state = r.last_state  # Includes fresh reset geometry before the first integration.
        self.pre[-1].update({name: _copy(value) for name, value in dict(
            training_selected=initializer.selected, initialization_prepared=initializer.prepared,
            training_start_count=initializer.reference_starts_count, launch_count=r.launch_count,
            quiet_intervals=r.quiet_intervals, ready_endpoint=r.previous_ready,
            support_contact=state['support_contact'], event_armed=bridge.events.armed,
            event_active=bridge.events.active, support_seen=initializer.support_seen,
            grace_total_ticks=initializer.grace_total_ticks,
            event_previous_guard=bridge.events.previous_guard,
            trajectory_id=command.trajectory_ids, target=bridge.packet['target_w'],
            actor_context=bridge.packet['actor_g1_context'],
            plan_handle=bridge.packet['plan_handle'], plan_digest=bridge.packet['plan_digest'],
            g1_launch_state=bridge.packet['g1_launch_state'],
            condition=cycle_condition(env),
        ).items()})

    def metric(self):
        inherited = super().metric()
        def capture(env):
            result = inherited(env)
            bridge = get_cycle_runtime(env); initializer = bridge.training_initializer
            self.endpoints[-1].update({name: _copy(value) for name, value in dict(
                training_selected=initializer.selected, support_seen=initializer.support_seen,
                grace_total_ticks=initializer.grace_total_ticks, grace_ticks=initializer.grace_ticks,
                first_failure_tick=bridge.events.first_failure_tick,
                first_failure_category=bridge.events.first_failure_category,
            ).items()})
            return result
        capture.reset = inherited.reset
        return capture

    def policy_boundary(self, env, observations, reward, terminated, timed_out):
        super().policy_boundary(env, observations, reward, terminated, timed_out)
        command = env.command_manager.get_term('motion')
        bridge = get_cycle_runtime(env)
        tick, active = command._cycle_clock()
        manager = env.observation_manager
        names, dims = manager.active_terms['actor'], manager.group_obs_term_dim['actor']
        sizes = [int(np.prod(dim)) for dim in dims]
        index = names.index('future_joint_reference'); offset = sum(sizes[:index])
        expected_future = torch.cat([command.sample_master('qpos', delta)[:, 7:]
            - command.robot_joint_pos for delta in (12, 24)], -1)
        actual_future = observations['actor'][:, offset:offset+sizes[index]]
        self.policies[-1].update({name: _copy(value) for name, value in dict(
            training_selected=bridge.training_initializer.selected,
            initialization_prepared=bridge.training_initializer.prepared,
            training_start_count=bridge.training_initializer.reference_starts_count,
            effective_reference_tick=tick, effective_reference_active=active,
            trajectory_id=command.trajectory_ids,
            current_reference=command.sample_master('qpos')[:, 7:],
            future12=command.sample_master('qpos', 12)[:, 7:],
            future24=command.sample_master('qpos', 24)[:, 7:],
            future_actor_error=(actual_future-expected_future).abs().amax(-1),
        ).items()})


def audit_initialization(pre, endpoint, policy, initial, *, probability, master_unchanged, receipts):
    """Pure saved-array checks; training starts never acquire strict eligibility."""
    selected = pre['training_selected'].astype(bool)
    first = pre['episode_tick'] == 0
    training_start = first & selected
    strict = first & ~selected
    first_rows = {name: np.concatenate((initial[name], policy[name]), axis=0)
        for name in initial}
    pending = first_rows['pending_reset'].astype(bool)
    pending_training = pending & first_rows['training_selected'].astype(bool)
    pending_strict = pending & ~first_rows['training_selected'].astype(bool)
    phase = np.argmax(first_rows['condition'][..., -4:], -1)
    checks = {
        'full_reference_arrays_unchanged': bool(master_unchanged),
        'strict_initialization_exercised': bool(strict.any()),
        'training_initialization_exercised': bool(training_start.any()) if probability else not selected.any(),
        'training_rows_start_at_frame_zero': bool(np.all(pre['reference_tick'][training_start] == 0)
            and np.all(pre['launch_tick'][training_start] == 0) and np.all(pre['launched'][training_start])
            and np.all(pre['phase'][training_start] == Phase.STRIKE)),
        'strict_rows_do_not_launch_at_reset': bool(np.all(~pre['launched'][strict])
            and np.all(pre['launch_tick'][strict] == -1) and np.all(pre['phase'][strict] == Phase.SETTLING)),
        'reset_has_no_fabricated_readiness_dwell': bool(np.all(pre['quiet_intervals'][first] == 0)
            and np.all(~pre['ready_endpoint'][first]) and np.all(~pre['event_previous_guard'][first])),
        'training_start_has_explicit_preparation': bool(np.all(pre['initialization_prepared'][training_start])
            and np.all(pre['event_armed'][training_start])),
        'first_observation_training_phase': bool(np.all(phase[pending_training] == Phase.STRIKE)
            and np.all(first_rows['effective_reference_active'][pending_training])),
        'first_observation_strict_phase': bool(np.all(phase[pending_strict] == Phase.SETTLING)
            and np.all(~first_rows['effective_reference_active'][pending_strict])),
        'first_observation_has_zero_elapsed': bool(np.all(first_rows['condition'][..., 8][pending] == 0)
            and np.all(first_rows['effective_reference_tick'][pending] == 0)),
        'actor_future_reference_matches_command': bool(np.all(first_rows['future_actor_error'] == 0)),
    }
    # Every pre-integration frame-zero row must match the preceding returned
    # actor observation. This catches phase/target/reference changes between
    # action selection and the first native integration after a real reset.
    actor_rows = {name: first_rows[name][:-1] for name in first_rows}
    pre_policy = {name: value[::4] for name, value in pre.items()}
    zero = pre_policy['episode_tick'] == 0
    checks['first_action_condition_unchanged_after_reset_commit'] = bool(np.array_equal(
        actor_rows['condition'][zero], pre_policy['condition'][zero]))
    checks['first_action_reference_unchanged_after_reset_commit'] = all(np.array_equal(
        actor_rows[name][zero], pre_policy[key][zero]) for name, key in
        (('current_reference', 'qref'), ('future12', 'future12'), ('future24', 'future24')))
    frozen = True
    for world in range(selected.shape[1]):
        rows = np.flatnonzero(pre['launched'][:, world])
        for request in np.unique(pre['request_id'][rows, world]):
            use = rows[pre['request_id'][rows, world] == request]
            for name in ('target', 'actor_context', 'plan_handle', 'plan_digest', 'g1_launch_state', 'trajectory_id'):
                data = pre[name][use, world]
                frozen &= bool(np.all(data == data[0]))
    checks['full_plan_identity_target_and_g1_context_frozen'] = frozen
    grace_limit = 4 if probability else 0
    support = endpoint['support_contact'].any(-1)
    tick = pre['reference_tick'] + 1
    expected_grace = (selected & pre['event_active'] & ~pre['support_seen'] & ~support
        & (tick >= 1) & (tick <= grace_limit))
    used_grace = endpoint['grace_total_ticks']-pre['grace_total_ticks']
    checks['grace_exactly_matches_initial_unsupported_endpoints'] = bool(np.array_equal(used_grace, expected_grace.astype(int)))
    checks['grace_has_four_tick_absolute_cap'] = bool(np.all(endpoint['grace_ticks'] <= grace_limit))
    checks['raw_support_flags_preserved'] = bool(np.all(~support[used_grace.astype(bool)]))
    support_records = []
    for world in range(selected.shape[1]):
        for request in np.unique(pre['request_id'][selected[:, world], world]):
            use = np.flatnonzero(selected[:, world] & (pre['request_id'][:, world] == request))
            active = use[pre['event_active'][use, world]]
            observed = use[support[use, world]]
            support_records.append({'environment_id': world, 'request_id': int(request),
                'first_actual_support_tick': int(tick[observed[0], world]) if len(observed) else None,
                'unsupported_active_endpoints': int((~support[active, world]).sum()),
                'suppressed_no_support_endpoints': int(used_grace[use, world].sum()),
                'suppressed_tick_indices': tick[use[used_grace[use, world].astype(bool)], world].astype(int).tolist()})
    metadata = True
    for receipt in receipts:
        init = receipt.get('initialization', {})
        types = np.asarray(init.get('start_type_codes', []), dtype=int)
        eligible = np.asarray(init.get('strict_launch_eligible', []), dtype=bool)
        training = np.asarray(init.get('training_reference_start', []), dtype=bool)
        strict_launch = np.asarray(init.get('strict_ready_launch', []), dtype=bool)
        metadata &= (init.get('physical_reset_only') is True and init.get('interior_rsi') is False
            and init.get('config', {}).get('reference_start_probability') == probability
            and init.get('initial_support_grace_ticks_limit') == grace_limit
            and len(types) > 0 and eligible.shape == types.shape
            and training.shape == types.shape and strict_launch.shape == types.shape
            and np.array_equal(eligible, types == 0) and not np.any(training & eligible)
            and not np.any(strict_launch & (types == 1))
            and receipt.get('acceptance_eligible') is False)
    checks['all_receipts_explicitly_ineligible_and_start_typed'] = bool(receipts) and bool(metadata)
    return {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
        'ordinary_reset_starts': int(first.sum()), 'training_frame_zero_starts': int(training_start.sum()),
        'strict_guard_reset_starts': int(strict.sum()),
        'initial_training_without_support': int((training_start & ~pre['support_contact'].any(-1)).sum()),
        'support_grace_limit_ticks': grace_limit,
        'actual_suppressed_no_support_endpoints': int(used_grace.sum()),
        'training_support_records': support_records,
        'strict_launch_qualification': 'not_granted_by_this_instrumentation_audit'}


def run_mode(bundle, output, *, probability, steps=400):
    if type(steps) is not int or steps != 400 or probability not in (0., .75):
        raise ValueError('Fixed qualification scope: 400 policy steps, p=0 or p=.75')
    _require_allocated_cuda(); runtime = runtime_record()
    if not runtime['declared_pins_match']:
        raise RuntimeError('Pinned native runtime required')
    import mujoco
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    torch.cuda.init(); torch.cuda.set_device(0); wp.init()
    destination = writable_path(output); destination.mkdir(parents=True, exist_ok=False)
    checked = load_runtime_bundle(bundle); recorder = InitializationRecorder()
    readiness = ReadinessConfig()
    cfg, recipe, fresh_contract = build_initialization_cfg(checked, recorder, probability=probability)
    cfg.sim.nan_guard.output_dir = str(destination/'nan_guard')
    report = {'schema': 'human_wbc_training_initialization_native_v1', 'runtime': runtime,
        'device': allocated_device_metadata('cuda:0'), 'bundle_manifest_sha256': sha256_file(checked.root/'manifest.json'),
        'training': False, 'learning_updates': 0, 'acceptance_qualified': False,
        'reference_domain_qualified': False, 'initialization': asdict(cfg.commands['motion'].training_initialization),
        'recipe': recipe.manifest(), 'recipe_sha256': recipe.sha256,
        'environments': ENVIRONMENTS, 'requested_policy_steps': steps, 'readiness': asdict(readiness),
        'fresh_imitation_contract': fresh_contract, 'actor': 'zero_residual_no_network',
        'later_g1_metadata_perturbation': True, 'seed': 10010}
    env = None; error = None; started = time.monotonic(); receipts = []; initial = []
    writes = resets = np.zeros(ENVIRONMENTS, int); write_events = []; master_before = master_after = None
    try:
        with native_stream_scope(), torch.no_grad():
            env = ManagerBasedRlEnv(cfg, device='cuda:0'); env._cycle_qualification = recorder
            save_compiled_model(env.sim.mj_model, destination/'compiled_scene.mjb')
            disk_sink = JsonReceiptSink(destination/'attempts')
            def sink(receipt):
                disk_sink(receipt); receipts.append(JsonReceiptSink._plain(receipt))
            providers = make_snapshot_providers(env, engineering_admitted=True, perturb_after_launch=True)
            bridge = attach_cycle_runtime(env, **providers, receipt_sink=sink, readiness=readiness)
            observations, _ = env.reset(seed=10010)
            command = env.command_manager.get_term('motion')
            immutable = {**command.master, 'glove_master': command.glove_master,
                'master_starts': command.master_starts, 'master_lengths': command.master_lengths}
            master_before = array_identity(immutable)
            zero = torch.zeros(ENVIRONMENTS, device=env.device)
            recorder.policy_boundary(env, observations, zero, zero.bool(), zero.bool())
            initial.append(recorder.policies.pop())
            report['actor_observation_dimension'] = list(env.observation_manager.group_obs_dim['actor'])
            report['initial_family_ids'] = _copy(command.family_ids[command.trajectory_ids]).tolist()
            action = torch.zeros((ENVIRONMENTS, 48), device=env.device)
            for _ in range(steps):
                obs, reward, terminated, timed_out, _ = env.step(action)
                recorder.policy_boundary(env, obs, reward, terminated, timed_out)
            report['stream_capture'] = verify_native_stream_capture('cuda:0')
            report['fresh_imitation_health'] = fresh_imitation_health(env)
            report['observer_samples'] = get_observer(env).actual_samples
            report['attempt_counts'] = JsonReceiptSink._plain(bridge.readiness.snapshot())
            master_after = array_identity({**command.master, 'glove_master': command.glove_master,
                'master_starts': command.master_starts, 'master_lengths': command.master_lengths})
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(limit=12)[-16384:]}
    finally:
        if env is not None:
            cmd = env.command_manager.get_term('motion')
            writes, resets, write_events = cmd.physical_writes.copy(), cmd.physical_resets.copy(), cmd.write_events
            env.close()
    arrays = {'native': recorder.native.arrays(), 'pre_interval': stack(recorder.pre),
        'endpoint': stack(recorder.endpoints), 'policy_boundary': stack(recorder.policies),
        'initial_observation': stack(initial)}
    report['hashes'] = {name: _save_npz(destination/(name+'.npz'), value) for name, value in arrays.items()}
    report.update(exception=error, wall_s=time.monotonic()-started, physical_writes=writes.tolist(),
        physical_resets=resets.tolist(), write_events=write_events, master_before_sha256=master_before,
        master_after_sha256=master_after, completed_policy_steps=len(recorder.policies),
        actual_physics_samples=len(recorder.native.rows), terminal_receipts=len(receipts))
    if error is None:
        base = audit_arrays(arrays['native'], arrays['pre_interval'], arrays['endpoint'], arrays['policy_boundary'],
            checked, mujoco.MjModel.from_binary_path(str(destination/'compiled_scene.mjb')),
            steps=steps, writes=writes, resets=resets)
        # The inherited ready-dwell check covers only launches appearing at a
        # completed policy boundary. Frame-zero training_start starts are checked
        # explicitly below and never claimed to have earned strict readiness.
        base['launch_qualification'] = 'instrumentation_only_training_starts_separate'
        report['inherited_native_audit'] = base
        report['initialization_audit'] = audit_initialization(arrays['pre_interval'], arrays['endpoint'], arrays['policy_boundary'],
            arrays['initial_observation'], probability=probability,
            master_unchanged=master_before == master_after, receipts=receipts)
        report['status'] = 'passed' if (base['status'] == 'passed'
            and report['initialization_audit']['status'] == 'passed'
            and report['stream_capture']['passed']
            and report['fresh_imitation_health']['all_raw_terms_finite']
            and report['fresh_imitation_health']['every_term_each_window']) else 'failed'
    else:
        report['status'] = 'failed'
    write_json(destination/'report.json', report)
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    output = writable_path(args.output); output.mkdir(parents=True, exist_ok=False)
    results = []
    for name, probability in (('strict_default', 0.), ('mixed_training', .75)):
        result = run_mode(args.bundle, output/name, probability=probability)
        results.append({'mode': name, 'status': result['status'],
            'report_sha256': sha256_file(output/name/'report.json')})
        if result['exception'] is not None:
            break
    report = {'schema': 'human_wbc_training_initialization_panel_v1', 'modes': results,
        'status': 'passed' if len(results) == 2 and all(x['status'] == 'passed' for x in results) else 'failed',
        'training': False, 'acceptance_qualified': False, 'strict_launch_qualified': False}
    write_json(output/'report.json', report); print(json.dumps(report))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
