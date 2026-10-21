"""Bounded native readiness/launch/reset diagnostic with inherited mjlab stepping.

No PPO or actor initialization. Zero residual is an explicit engineering input;
unready attempts remain unlaunched failures and never become punch evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, asdict
from contextlib import contextmanager
import json
from pathlib import Path
import time
import numpy as np
import torch

from human_punching.wbc.native_probe import NativeProbeRecorder, _copy, _configure_caches, _require_allocated_cuda
from human_punching.wbc.native_cycle_probe import allocated_device_metadata, save_compiled_model, _save_npz
from human_punching.wbc.provenance import writable_path, write_json, sha256_file, runtime_record
from human_punching.wbc.runtime.bundle import load_runtime_bundle
from human_punching.wbc.runtime.task import build_human_env_cfg
from human_punching.wbc.cycle_trace import derive_kinematics
from .posthoc_qualification import rotation_error_rad
from .cycle_runtime import (CycleMotionCommand, CycleMotionCommandCfg, CycleObserverAction,
    CycleObserverActionCfg, JsonReceiptSink, attach_cycle_runtime, configure_cycle_command,
    cycle_condition, get_cycle_runtime)
from .runtime_telemetry import configure_observer, ObserverOptions, get_observer
from .snapshot_providers import make_snapshot_providers
from .readiness_clock import ReadinessConfig
from .attempt_receipts import AggregateAttemptSink

ENVIRONMENTS = 16


@contextmanager
def native_stream_scope(device='cuda:0'):
    """Order Torch action/FK work on the actual Warp simulation stream."""
    import warp as wp
    stream = torch.cuda.ExternalStream(wp.get_stream(device).cuda_stream, device=device)
    caller = torch.cuda.current_stream(device)
    stream.wait_stream(caller)
    try:
        with torch.cuda.stream(stream):
            yield
    finally:
        caller.wait_stream(stream)


class QualificationCommand(CycleMotionCommand):
    """Count only inherited episode-initialization writes; no alternate reset path."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.physical_resets = np.zeros(env.num_envs, np.int64)
        self.physical_writes = np.zeros(env.num_envs, np.int64)
        self.write_events = []

    def reset(self, env_ids):
        ids = _copy(env_ids).astype(int)
        if np.any(self.physical_resets[ids] > 0):
            reasons = _copy(self._env.reset_terminated | self._env.reset_time_outs)
            if not reasons[ids].all():
                raise RuntimeError('Physical reset lacks an actual inherited terminal reason')
        self.physical_resets[ids] += 1
        return super().reset(env_ids)

    def _write_reference_state_to_sim(self, env_ids, *args, **kwargs):
        if not getattr(self, '_reset_in_progress', False):
            raise RuntimeError('Physical writes are allowed only in inherited episode initialization')
        ids = _copy(env_ids).astype(int)
        self.physical_writes[ids] += 1
        self.write_events.append({'policy_step': int(self._env.common_step_counter),
            'environment_ids': ids.tolist(), 'kind': 'ordinary_episode_initialization'})
        return super()._write_reference_state_to_sim(env_ids, *args, **kwargs)


@dataclass(kw_only=True)
class QualificationCommandCfg(CycleMotionCommandCfg):
    def build(self, env):
        return QualificationCommand(self, env)


class QualificationAction(CycleObserverAction):
    def apply_actions(self):
        self._env._cycle_qualification.before_interval(self._env)
        super().apply_actions()


@dataclass(kw_only=True)
class QualificationActionCfg(CycleObserverActionCfg):
    def build(self, env):
        return QualificationAction(self, env)


class QualificationRecorder:
    def __init__(self):
        self.native = NativeProbeRecorder(ENVIRONMENTS)
        self.pre, self.endpoints, self.policies = [], [], []

    def before_interval(self, env):
        cmd = env.command_manager.get_term('motion'); r = get_cycle_runtime(env).readiness
        self.pre.append({name: _copy(value) for name, value in dict(
            qpos=env.sim.data.qpos, qvel=env.sim.data.qvel, time=env.sim.data.time,
            qref=cmd.sample_master('qpos')[:, 7:], vref=cmd.sample_master('qvel')[:, 6:],
            feedforward=cmd.sample_master('joint_feedforward'),
            future12=cmd.sample_master('qpos', 12)[:, 7:], future24=cmd.sample_master('qpos', 24)[:, 7:],
            guard=cmd.master['qpos'][cmd.master_starts[cmd.trajectory_ids], 7:],
            episode_tick=cmd.physics_steps, reference_tick=r.reference_tick,
            launched=r.launched, launch_tick=r.launch_tick, request_id=r.request_id,
            phase=r.phase).items()})

    def metric(self):
        def capture(env):
            self.native.capture(env)
            state = {name: _copy(value) for name, value in get_observer(env).endpoint_state().items()}
            r = get_cycle_runtime(env).readiness
            state['ready_endpoint'] = _copy(r.previous_ready)
            state['quiet_intervals'] = _copy(r.quiet_intervals)
            self.endpoints.append(state)
            return torch.zeros(env.num_envs, device=env.device)
        capture.reset = self.native.note_reset
        return capture

    def policy_boundary(self, env, observations, reward, terminated, timed_out):
        cmd = env.command_manager.get_term('motion'); bridge = get_cycle_runtime(env); r = bridge.readiness
        manager = env.observation_manager
        names, dims = manager.active_terms['actor'], manager.group_obs_term_dim['actor']
        sizes = [int(np.prod(dim)) for dim in dims]
        index = names.index('punch_condition'); start = sum(sizes[:index])
        condition = observations['actor'][:, start:start+sizes[index]]
        recomputed = cycle_condition(env)
        if not torch.equal(condition, recomputed):
            raise RuntimeError('Returned actor condition differs from the current command boundary')
        row = dict(condition=condition, qpos=env.sim.data.qpos, qvel=env.sim.data.qvel,
            root_position=cmd.robot_anchor_pos_w, root_quaternion=cmd.robot_anchor_quat_w,
            reference_tick=r.reference_tick, launched=r.launched, launch_tick=r.launch_tick,
            request_id=r.request_id, phase=r.phase, pending_reset=bridge.pending_reset,
            target=cmd.target_w, actor_context=bridge.actor_context,
            reward=reward, terminated=terminated, timed_out=timed_out,
            episode_tick=cmd.physics_steps, time=env.sim.data.time,
            observation_finite=torch.isfinite(observations['actor']).all(-1)&torch.isfinite(observations['critic']).all(-1))
        self.policies.append({name: _copy(value) for name, value in row.items()})


def build_cfg(bundle, recorder, *, readiness):
    from mjlab.managers.metrics_manager import MetricsTermCfg
    cfg = build_human_env_cfg(bundle, num_envs=ENVIRONMENTS, evaluation=True)
    cfg.auto_reset = True
    # Exercise the bridge's measured terminal decisions; inherited geometric
    # tracking terminations would otherwise hide its own readiness outcome.
    cfg.terminations = {}
    cfg.metrics = {}
    cfg.seed = 10010
    configure_observer(cfg, ObserverOptions(enabled=True))
    configure_cycle_command(cfg, maximum_ttc_s=max(c['requested_ttc_s'] for c in bundle.clips),
        readiness=readiness, require_physical_qualification=False, task_objectives=True)
    for collection, key, cls in ((cfg.commands, 'motion', QualificationCommandCfg),
                                  (cfg.actions, 'joint_pos', QualificationActionCfg)):
        old = collection[key]
        collection[key] = cls(**{f.name: getattr(old, f.name) for f in fields(old) if f.init})
    cfg.metrics['cycle_qualification_raw'] = MetricsTermCfg(func=recorder.metric(), per_substep=True)
    return cfg


def stack(rows):
    return {name: np.stack([row[name] for row in rows]) for name in rows[0]} if rows else {}


def audit_arrays(native, pre, endpoint, policy, bundle, model, *, steps, writes, resets, quiet_ticks=40):
    """Checks retain actual resets and never presume a launched/ready cycle."""
    count = 4*steps
    checks = {'complete_native_samples': len(pre.get('qpos', [])) == count == len(native.get('qpos', [])),
        'complete_policy_samples': len(policy.get('qpos', [])) == steps,
        'native_timestep': float(model.opt.timestep) == .005,
        'one_initialization_per_reset': bool(np.array_equal(writes, resets)),
        'ordinary_reset_exercised': bool(np.any(resets > 1))}
    for label, arrays in (('native', native), ('preinterval', pre), ('endpoint', endpoint), ('policy', policy)):
        checks[label+'_finite'] = all(np.isfinite(v).all() for v in arrays.values() if np.issubdtype(v.dtype, np.number))
    if not checks['complete_native_samples'] or not checks['complete_policy_samples']:
        return {'status': 'failed', 'checks': checks}
    caps, kp, kd = (np.asarray(bundle.body[key]) for key in ('effort', 'kp', 'kd'))
    limits = np.asarray(bundle.body['joint_limits_rad'], np.float32)
    expected_target = np.clip(pre['qref'].astype(float)+.25*native['raw_action'], limits[:, 0], limits[:, 1])
    expected_drive = kp*native['joint_position_target']+kd*pre['vref']+pre['feedforward']
    expected_force = np.clip(native['ctrl'].astype(float)-kp*pre['qpos'][..., 7:]-kd*pre['qvel'][..., 6:], -caps, caps)
    errors = {'target_rad': float(abs(expected_target-native['joint_position_target']).max()),
        'velocity_rad_s': float(abs(pre['vref']-native['joint_velocity_target']).max()),
        'feedforward_nm': float(abs(pre['feedforward']-native['joint_effort_target']).max()),
        'drive_nm': float(abs(expected_drive-native['ctrl']).max()),
        'force_nm': float(abs(expected_force-native['actuator_force']).max()),
        'clock_s': float(abs(native['time'].astype(float)-pre['time'].astype(float)-.005).max())}
    checks.update(zero_residual=bool(np.all(native['raw_action'] == 0)),
        native_counter=bool(np.array_equal(native['sim_step_index'], np.arange(1, count+1))),
        policy_counter=bool(np.array_equal(native['policy_step_index'], np.arange(count)//4)),
        native_force_caps=bool(np.all(abs(native['actuator_force']) <= caps+1e-4)),
        reference_target=errors['target_rad'] <= 2e-6, reference_velocity=errors['velocity_rad_s'] <= 2e-6,
        reference_feedforward=errors['feedforward_nm'] <= 5e-6,
        native_affine_drive=errors['drive_nm'] <= 2e-4, native_capped_force=errors['force_nm'] <= 2e-3,
        physics_clock=errors['clock_s'] <= 2e-6,
        no_external_forces=bool(np.all(native['qfrc_applied'] == 0) and np.all(native['xfrc_applied'] == 0)),
        free_root=bool(np.all(abs(native['qfrc_actuator'][..., :6]) <= 1e-9)),
        no_overflow=bool(np.all(native['overflow'] == 0)),
        zero_reference_before_launch=bool(np.all(pre['reference_tick'][~pre['launched']] == 0)),
        future_held_before_launch=bool(np.array_equal(pre['qref'][~pre['launched']], pre['guard'][~pre['launched']])
            and np.array_equal(pre['future12'][~pre['launched']], pre['guard'][~pre['launched']])
            and np.array_equal(pre['future24'][~pre['launched']], pre['guard'][~pre['launched']])
            and np.all(pre['vref'][~pre['launched']] == 0)),
        no_invented_launch=bool(np.all(pre['launch_tick'][~pre['launched']] == -1)),
        launch_at_policy_boundary=bool(np.all(pre['launch_tick'][pre['launched']] % 4 == 0)),
        elapsed_from_actual_launch=bool(np.array_equal(pre['reference_tick'][pre['launched']],
            (pre['episode_tick']-pre['launch_tick'])[pre['launched']])),
        finite_task_rewards=bool(np.isfinite(policy['reward']).all()),
        finite_observations=bool(policy['observation_finite'].all()))
    fk_errors = {}; quaternion_errors = {}
    for world in range(native['qpos'].shape[1]):
        fk = derive_kinematics(model, native['qpos'][:, world], native['qvel'][:, world])
        for name, expected in fk.items():
            actual = endpoint[name][:, world]
            if 'quat' in name:
                quaternion_errors[name] = max(quaternion_errors.get(name, 0.), float(rotation_error_rad(expected, actual).max()))
            else:
                fk_errors[name] = max(fk_errors.get(name, 0.), float(abs(expected-actual).max()))
    checks['fresh_endpoint_fk'] = all(v <= 2e-5 for v in fk_errors.values()) and all(v <= 2e-5 for v in quaternion_errors.values())
    # Floating-root Hips is the command anchor. Returned observations must use
    # post-integration/post-reset root data, not the preceding metric interval.
    root_error = float(abs(policy['root_position']-policy['qpos'][..., :3]).max())
    root_rotation = float(rotation_error_rad(policy['root_quaternion'], policy['qpos'][..., 3:7]).max())
    checks['fresh_actor_anchor'] = root_error <= 2e-6 and root_rotation <= 2e-6
    frozen = True
    for world in range(ENVIRONMENTS):
        rows = np.flatnonzero(policy['launched'][:, world] & ~policy['pending_reset'][:, world])
        for request in np.unique(policy['request_id'][rows, world]):
            selected = rows[policy['request_id'][rows, world] == request]
            for key in ('target', 'actor_context', 'launch_tick'):
                value = policy[key][selected, world]
                frozen &= bool(np.all(value == value[0]))
    checks['frozen_context_despite_later_g1_perturbation'] = frozen
    launches = (policy['launched'] & ~policy['pending_reset']
                & (policy['episode_tick'] == policy['launch_tick']))
    ready_rows = endpoint['ready_endpoint'][3::4]
    dwell_rows = endpoint['quiet_intervals'][3::4]
    support_rows = endpoint['support_contact'][3::4].all(-1)
    checks['launch_from_recorded_ready_dwell'] = bool(np.all(ready_rows[launches])
        and np.all(dwell_rows[launches] >= quiet_ticks) and np.all(support_rows[launches]))
    return {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
        'errors': errors, 'fk_errors': fk_errors, 'rotation_errors_rad': quaternion_errors,
        'actor_anchor_error_m': root_error, 'actor_anchor_rotation_error_rad': root_rotation,
        'launch_observed': bool(pre['launched'].any()),
        'launch_checks_exercised': bool(pre['launched'].any()),
        'launch_qualification': 'not_tested_unready' if not pre['launched'].any() else 'diagnostic_only'}


def run(bundle, output, *, steps=400, quiet_ticks=40, support_artifact=None):
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError('Bounded diagnostic requires 1..1000 policy steps')
    _require_allocated_cuda(); runtime = runtime_record()
    if not runtime['declared_pins_match']:
        raise RuntimeError('Pinned native runtime required')
    destination = writable_path(output); destination.mkdir(parents=True, exist_ok=False)
    _configure_caches(destination)
    import mujoco
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    torch.cuda.init(); torch.cuda.set_device(0); wp.init()
    checked = load_runtime_bundle(bundle); recorder = QualificationRecorder()
    readiness = ReadinessConfig(quiet_ticks=quiet_ticks)
    cfg = build_cfg(checked, recorder, readiness=readiness)
    cfg.sim.nan_guard.output_dir = str(destination/'nan_guard')
    report = {'schema': 'human_wbc_cycle_native_qualification_v1', 'runtime': runtime,
        'device': allocated_device_metadata('cuda:0'), 'bundle_manifest_sha256': sha256_file(checked.root/'manifest.json'),
        'training': False, 'acceptance_qualified': False, 'reference_domain_qualified': False,
        'environments': ENVIRONMENTS, 'requested_policy_steps': steps, 'readiness': asdict(readiness),
        'readiness_variant': 'default' if quiet_ticks == 40 else 'explicit_engineering_dwell_variant',
        'live_snapshot_semantics': 'isolated immutable G1 snapshot; no moving G1 entity',
        'support_artifact_sha256': sha256_file(Path(support_artifact)) if support_artifact else None,
        'actor': 'zero_residual_no_network', 'later_g1_metadata_perturbation': True}
    env = None; bridge = None; aggregate = None; error = None; started = time.perf_counter()
    writes = resets = np.zeros(ENVIRONMENTS, int); write_events = []
    try:
        with native_stream_scope(), torch.no_grad():
            env = ManagerBasedRlEnv(cfg, device='cuda:0'); env._cycle_qualification = recorder
            save_compiled_model(env.sim.mj_model, destination/'compiled_scene.mjb')
            receipts = JsonReceiptSink(destination/'attempts')
            aggregate = AggregateAttemptSink(destination/'aggregate', shared_identity=report, flush_every_batches=32)
            def sink(receipt):
                receipts(receipt); aggregate(receipt)
            providers = make_snapshot_providers(env, engineering_admitted=True, perturb_after_launch=True)
            bridge = attach_cycle_runtime(env, **providers, receipt_sink=sink,
                feedforward_artifact=support_artifact, readiness=readiness)
            env.reset(seed=10010)
            report['actor_observation_dimension'] = list(env.observation_manager.group_obs_dim['actor'])
            report['initial_families'] = _copy(env.command_manager.get_term('motion').family_ids[
                env.command_manager.get_term('motion').trajectory_ids]).tolist()
            action = torch.zeros((ENVIRONMENTS, 48), device=env.device)
            for _ in range(steps):
                obs, reward, terminated, timed_out, _ = env.step(action)
                recorder.policy_boundary(env, obs, reward, terminated, timed_out)
            report['observer_samples'] = get_observer(env).actual_samples
            report['attempt_counts'] = JsonReceiptSink._plain(bridge.readiness.snapshot())
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        if aggregate is not None:
            aggregate.flush(); report['aggregate_attempts'] = aggregate.report()
        if env is not None:
            cmd = env.command_manager.get_term('motion')
            writes, resets, write_events = cmd.physical_writes.copy(), cmd.physical_resets.copy(), cmd.write_events
            env.close()
    arrays = {'native': recorder.native.arrays(), 'pre_interval': stack(recorder.pre),
        'endpoint': stack(recorder.endpoints), 'policy_boundary': stack(recorder.policies)}
    report['hashes'] = {name: _save_npz(destination/(name+'.npz'), value) for name, value in arrays.items()}
    report.update(exception=error, wall_s=time.perf_counter()-started, physical_writes=writes.tolist(),
        physical_resets=resets.tolist(), write_events=write_events,
        completed_policy_steps=len(recorder.policies), actual_physics_samples=len(recorder.native.rows))
    if error is None:
        report['audit'] = audit_arrays(arrays['native'], arrays['pre_interval'], arrays['endpoint'], arrays['policy_boundary'],
            checked, mujoco.MjModel.from_binary_path(str(destination/'compiled_scene.mjb')),
            steps=steps, writes=writes, resets=resets, quiet_ticks=quiet_ticks)
        report['status'] = report['audit']['status']
    else:
        report['status'] = 'failed'
    write_json(destination/'report.json', report)
    return report


def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--bundle', required=True); p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=400); p.add_argument('--quiet-ticks', type=int, default=40)
    p.add_argument('--support-artifact')
    a = p.parse_args(); report = run(a.bundle, a.output, steps=a.steps, quiet_ticks=a.quiet_ticks, support_artifact=a.support_artifact)
    print(json.dumps({'status': report['status'], 'output': a.output, 'physics_samples': report['actual_physics_samples']}))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
