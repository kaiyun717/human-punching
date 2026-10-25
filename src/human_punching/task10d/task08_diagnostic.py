"""Matched source-speed H2 replay evaluation of the frozen Task08 H1 policy.

No scheduler, learning, TTC control or reference alteration. Batches retain the
original world ordering and seeds so selected media can replay identical batches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .g1 import PROJECT
from .phase1 import cpu, dump
from .phase2 import load_library
from .phase3_render import create_batch, state_copy, assert_unchanged, load_rows
from .placement_phase3 import INDEX, PRIMITIVE_IDS
from .task08_policy_diagnostic import (FrozenTask08Policy, Task08ActorHistory,
                                       Task08EpisodeOutcomes)

ROOT = PROJECT / 'outputs/task10d/task08_diagnostic_20261024'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


class BatchedSourceMotion:
    """One sampled native trajectory plus constant per-world yaw/XY transforms."""

    _sample_cache = {}

    def __init__(self, primitive, placements_xy, placements_yaw, *, device, dtype):
        self.primitive = primitive
        self.end = float(primitive.reference.end_time_s)
        key = (primitive.reference.sha256, tuple(primitive.placement.xy), primitive.placement.yaw, str(device), dtype)
        if key not in self._sample_cache:
            times = np.minimum(np.arange(int(np.ceil(self.end/.005))+1)*.005, self.end)
            samples = [primitive.reference.sample(float(t), primitive.placement) for t in times]
            self._sample_cache[key] = (
                torch.as_tensor(np.stack([s.body_pos for s in samples]), device=device, dtype=dtype),
                torch.as_tensor(np.stack([s.body_quat for s in samples]), device=device, dtype=dtype))
        self.positions, self.quaternions = self._sample_cache[key]
        delta = np.asarray(placements_yaw) - primitive.placement.yaw
        c, s = np.cos(delta), np.sin(delta)
        accepted = primitive.placement.xy
        shifted = np.asarray(placements_xy) - np.stack((c*accepted[0]-s*accepted[1],
                                                        s*accepted[0]+c*accepted[1]), -1)
        self.c = torch.as_tensor(c[:, None], device=device, dtype=dtype)
        self.s = torch.as_tensor(s[:, None], device=device, dtype=dtype)
        self.ch = torch.as_tensor(np.cos(delta/2)[:, None], device=device, dtype=dtype)
        self.sh = torch.as_tensor(np.sin(delta/2)[:, None], device=device, dtype=dtype)
        self.shift = torch.as_tensor(shifted, device=device, dtype=dtype)
        self.count = len(delta)

    def pose(self, step):
        index = min(step, len(self.positions)-1)
        p, q = self.positions[index], self.quaternions[index]
        pos = torch.stack((self.c*p[:, 0]-self.s*p[:, 1]+self.shift[:, 0, None],
                           self.s*p[:, 0]+self.c*p[:, 1]+self.shift[:, 1, None],
                           p[:, 2].expand(self.count, -1)), -1)
        quat = torch.stack((self.ch*q[:, 0]-self.sh*q[:, 3],
                            self.ch*q[:, 1]-self.sh*q[:, 2],
                            self.ch*q[:, 2]+self.sh*q[:, 1],
                            self.ch*q[:, 3]+self.sh*q[:, 0]), -1)
        return pos, quat

    def write(self, binding, data, step):
        pos, quat = self.pose(step)
        ids = torch.as_tensor(binding.mocap_ids, device=pos.device)
        data.mocap_pos[:, ids] = pos
        data.mocap_quat[:, ids] = quat


def batch_seed(block):
    return 202610240 + 1009*int(block)


def run_batch(g1, binding, camera, policy, primitive, inputs, indices, geometry,
              *, block, media_rows=None, media_output=None):
    from boxing.perception.image_faults import ForegroundVisualFaultProcess
    from boxing.perception.latency import VisualDelayBuffer
    from .task08_diagnostic_media import Task08DiagnosticMedia

    count, valid = g1.num_envs, len(indices)
    padded = np.pad(indices, (0, count-valid), mode='edge')
    seed = batch_seed(block)
    g1.noise_seed = seed
    # Reset independent episode streams to the same original world ordering.
    # Other primitives use the same snapshot blocks and exogenous seed scheme.
    with g1.ordered_execution():
        camera.faults = ForegroundVisualFaultProcess(count, device=g1.device, seed=seed+1)
        camera.delay = VisualDelayBuffer(count, device=g1.device, seed=seed+2)
    rows = load_rows(geometry, inputs['geometry_row_index'][padded])
    reset = g1.reset(rows)
    before = state_copy(g1)
    motion = BatchedSourceMotion(primitive, inputs['placement_xy'][padded],
        inputs['placement_yaw_rad'][padded], device=g1.device, dtype=g1.dtype)
    history = Task08ActorHistory(count, g1.device, g1.dtype)
    outcomes = Task08EpisodeOutcomes(inputs['source_recovery_complete_s'][padded], device=g1.device)
    outcomes.active[valid:] = False
    frozen_first = torch.full((count, 2), float('nan'), device=g1.device, dtype=torch.float64)
    frozen_min = torch.full((count, 2), float('inf'), device=g1.device, dtype=torch.float64)
    moving_first = frozen_first.clone()
    selected = {int(r['world_index']): r for r in (media_rows or [])}
    recorders, media_results = {}, []
    if selected:
        for world, row in selected.items():
            assert world < valid and int(inputs['episode_id'][padded[world]]) == int(row['episode_id'])
            out = media_output / primitive.primitive_id / f"episode_{row['episode_id']:04d}"
            recorders[world] = Task08DiagnosticMedia(out, camera.renderer.host_model,
                'Frozen Task08 H1 final | source-speed H2 replay', episode=row)
    with g1.ordered_execution(), torch.no_grad():
        motion.write(binding, g1.data, 0)
        g1.refresh()
        geom = g1.geometry()
        frozen = geom['head_position'].clone()
        torch.testing.assert_close(frozen, torch.as_tensor(inputs['frozen_head_world'][padded],
            device=g1.device, dtype=g1.dtype), atol=5e-6, rtol=0.)
    health, initial, exposure, obs = None, None, None, None
    horizon = min(30., float(np.max(inputs['source_recovery_complete_s'][padded])) + 3.005)
    steps = int(np.ceil(horizon/.005))
    started = time.monotonic()
    active_hand = int(inputs['active_hand_index'][indices[0]])
    try:
        for step in range(steps+1):
            elapsed = step*.005
            with g1.ordered_execution(), torch.no_grad():
                prior_active = outcomes.active.clone()
                geom = g1.geometry()
                gloves = g1.data.geom_xpos[:, binding.glove_geom_ids]
                distance = torch.linalg.vector_norm(gloves-frozen[:, None], dim=-1)
                moving = torch.linalg.vector_norm(gloves-geom['head_position'][:, None], dim=-1)
                frozen_first = torch.where(torch.isnan(frozen_first) & (distance <= .14), elapsed, frozen_first)
                frozen_min = torch.minimum(frozen_min, distance.double())
                moving_first = torch.where(prior_active[:, None] & torch.isnan(moving_first)
                    & (moving <= .14), elapsed, moving_first)
                outcome_tick = outcomes.update(geom, gloves, elapsed)
                if bool((outcome_tick['stopped'] & (outcomes.reason == 10)).any()):
                    raise RuntimeError('Fatal nonfinite state; batch is not a completed diagnostic result')
                if step % 4 == 0:
                    exposure = camera.capture(elapsed, reset=step == 0, as_numpy=False,
                        diagnostics=step == 0, include_rgb=bool(recorders))
                    obs = history.acquire(g1, exposure, elapsed, reset=step == 0,
                        mask=None if step == 0 else prior_active)
                    if step == 0:
                        contract = assert_unchanged(g1, before)
                        assert all(contract.values()) and bool((g1.data.time == 0).all())
                        assert bool((exposure['capture_timestamps'] == 0).all())
                        assert bool(exposure['padding'].all())
                        clean = (exposure['glove_rgb_valid_source_pixels'] > 0).all(-1)
                        clean &= (exposure['glove_acquired_min_cells'] > 0).all(-1)
                        assert bool(clean[:valid].all()), 'Reviewed initial clean observations did not reproduce'
                        initial = dict(atomic_preserved=True, reset=reset,
                            actual_frozen_head_world=cpu(frozen)[:valid],
                            clean_both_represented=cpu(clean)[:valid],
                            clean_source_pixels=cpu(exposure['glove_rgb_valid_source_pixels'])[:valid],
                            clean_minimum_cells=cpu(exposure['glove_acquired_min_cells'])[:valid],
                            delay_ticks=cpu(exposure['delay_ticks'])[:valid],
                            action_observation_sha256=hashlib.sha256(cpu(obs['O_rsa']).tobytes()).hexdigest())
                        if not hasattr(policy, '_diagnostic_health'):
                            policy._diagnostic_health = policy.inference_health(obs['O_rsa'])
                        health = policy._diagnostic_health
                        assert health['passed'], health
                    for key in ('visual_frame_ids', 'visual_timestamps', 'visual_padding'):
                        if key in obs:
                            exposure[key] = obs[key]
                    for world, recorder in list(recorders.items()):
                        recorder.append(g1.data, exposure, index=world, time_s=elapsed,
                            event=None, status='first-event evaluation')
                    if bool(outcomes.active.any()):
                        targets = policy.action(obs['O_rsa'])
                        assert bool(torch.isfinite(targets).all()), 'Nonfinite policy targets'
                        g1.apply_physical_target(targets)
                stopped = cpu(outcome_tick['stopped']) if recorders else None
                if recorders and np.any(stopped):
                    records = outcomes.records()
                    for world in list(recorders):
                        if stopped[world]:
                            actual = {**selected[world], **records[world]}
                            media_results.append(recorders.pop(world).close(
                                outcome=actual, expected_outcome=selected[world]))
                if not bool(outcomes.active.any()):
                    break
                if step == steps:
                    raise RuntimeError('Unresolved episode at explicit source-return+3s horizon')
                motion.write(binding, g1.data, step+1)
                g1.step_substep()
    finally:
        for world, recorder in recorders.items():
            recorder.close(outcome={**selected[world], 'disposition':'interrupted'})
    records = outcomes.records()[:valid]
    frozen_first, frozen_min, moving_first = map(cpu, (frozen_first, frozen_min, moving_first))
    for world, (index, row) in enumerate(zip(indices, records, strict=True)):
        nullable = lambda values: [float(v) if np.isfinite(v) else None for v in values]
        row.update(episode_id=int(inputs['episode_id'][index]), primitive_id=primitive.primitive_id,
            family=primitive.family, snapshot_id=int(inputs['snapshot_id'][index]),
            snapshot_index=int(inputs['snapshot_index'][index]), batch_index=block, world_index=world,
            batch_size=count, seed=seed, visual_fault_seed=seed+1, delay_seed=seed+2,
            delay_ticks=int(initial['delay_ticks'][world]), camera_pitch_up_deg=25., source_clock_rate=1.,
            source_recovery_complete_s=float(inputs['source_recovery_complete_s'][index]),
            source_outward_entry_sample_s=float(inputs['source_outward_entry_sample_s'][index]),
            source_outward_entry_interpolated_s=float(inputs['source_outward_entry_interpolated_s'][index]),
            runtime_frozen_first_by_hand_s=nullable(frozen_first[world]),
            runtime_frozen_min_by_hand_m=nullable(frozen_min[world]),
            runtime_moving_first_by_hand_s=nullable(moving_first[world]),
            frozen_entry_measurement_censoring='Batch may end before entry; use full-source offline entry for usual entrance timing',
            active_hand=('left','right')[active_hand],
            initial_clean_both=bool(initial['clean_both_represented'][world]))
    return records, dict(completed=len(records), batch_index=block,
        wall_s=time.monotonic()-started, health=health, initial=initial,
        endpoint_time_s=elapsed, source_clock_rate=1., media=media_results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=ROOT/'inputs01')
    parser.add_argument('--primitive-indices', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--media-selection', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT.resolve()) or not 1 <= args.batch_size <= 128:
        raise ValueError('Local diagnostic outputs and bounded batch size required')
    if (len(set(args.primitive_indices)) != len(args.primitive_indices)
            or any(i not in range(8) for i in args.primitive_indices)):
        raise ValueError('Distinct accepted primitive indices required')
    output.mkdir(parents=True, exist_ok=False)
    import warp as wp
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
        raise RuntimeError('One allocated CUDA GPU required')
    wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
    manifest = json.loads((args.inputs/'manifest.json').read_text())
    assert digest(args.inputs/'episodes.npz') == manifest['episodes_sha256']
    geometry_path = PROJECT/'outputs/task10d/phase3/snapshots/geometry.npz'
    for path in (geometry_path, INDEX):
        relative = path.relative_to(PROJECT).as_posix()
        expected = [value for name, value in manifest['input_sha256'].items()
                    if name.endswith('/' + relative)]
        assert len(expected) == 1 and digest(path) == expected[0], str(path)
    inputs = dict(np.load(args.inputs/'episodes.npz', allow_pickle=False))
    geometry = dict(np.load(geometry_path, allow_pickle=False))
    assert len(inputs['episode_id']) == 8000
    assert np.array_equal(geometry['snapshot_id'][inputs['geometry_row_index']], inputs['snapshot_id'])
    assert manifest['primitive_ids'] == list(PRIMITIVE_IDS)
    assert np.array_equal(inputs['episode_id'], np.arange(8000))
    assert np.all(inputs['source_clock_rate'] == 1.)
    for j in range(8):
        assert np.array_equal(inputs['snapshot_id'][j*1000:(j+1)*1000], inputs['snapshot_id'][:1000])
    primitives = load_library(INDEX, list(PRIMITIVE_IDS))
    selected = []
    if args.media_selection:
        selection = json.loads(args.media_selection.read_text())
        selected = selection['selected']
        if not selected:
            raise ValueError('Empty media selection must not start new evaluations')
        for row in selected:
            assert row['batch_size'] == args.batch_size
            assert row['seed'] == batch_seed(row['batch_index'])
    dump(output/'invocation.json', dict(job_id=os.environ['SLURM_JOB_ID'],
        primitive_indices=args.primitive_indices, batch_size=args.batch_size,
        inputs_sha256=digest(args.inputs/'episodes.npz'), manifest_sha256=digest(args.inputs/'manifest.json'),
        geometry_sha256=digest(geometry_path), media_selection=None if not args.media_selection else str(args.media_selection.resolve()),
        media_selection_sha256=None if not args.media_selection else digest(args.media_selection),
        source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in Path(__file__).parent.glob('*.py')},
        task08_source_sha256={str(p):digest(p) for p in (PROJECT.parent/'robust_sav/src/boxing').rglob('*.py')
            if p.name in ('residual_learning.py','residual_observations.py','collector.py','glove_geometry.py','camera.py','rectified_camera.py')},
        source_speed=True, scheduler=False, learning=False, purpose='selected media replay' if selected else '8000 matched source-speed episodes'))
    env, g1, binding, camera = create_batch(dict(seed=202610240), output,
        primitives[0].reference, 25., args.batch_size)
    started, completed, all_rows = time.monotonic(), 0, []
    try:
        policy = FrozenTask08Policy(g1)
        camera_identity = {**camera.metadata(), 'actor_camera_inputs': True,
            'controller': 'Frozen Task08 H1 final residual plus frozen Fresh96',
            'observation_use': 'Ordinary delayed corrupted C in accepted 948-feature actor history'}
        dump(output/'runtime.json',dict(policy=policy.identity, runtime=g1.runtime_identity,
            model=g1.model_audit, camera=camera_identity, batch_seed_rule='202610240+1009*snapshot_block',
            safety='Unchanged Task08 labels and first_event; complete source recovery then arrival/E<=50 or+3s cutoff'))
        for primitive_index in args.primitive_indices:
            primitive = primitives[primitive_index]
            indices = np.flatnonzero(inputs['primitive_index'] == primitive_index)
            assert len(indices) == 1000
            for block,start in enumerate(range(0,1000,args.batch_size)):
                media_rows = [r for r in selected if r['primitive_id']==primitive.primitive_id and r['batch_index']==block]
                if selected and not media_rows:
                    continue
                batch_indices = indices[start:start+args.batch_size]
                rows, evidence = run_batch(g1,binding,camera,policy,primitive,inputs,batch_indices,
                    geometry,block=block,media_rows=media_rows,media_output=output/'videos')
                dump(output/f'batch_{primitive_index}_{block:02d}.json', evidence)
                with (output/'episodes.jsonl').open('a') as handle:
                    for row in rows:
                        handle.write(json.dumps(row,allow_nan=False)+'\n')
                all_rows.extend(rows);completed +=len(rows)
                counts={name:sum(r['disposition']==name for r in all_rows) for name in set(r['disposition'] for r in all_rows)}
                dump(output/'progress.json',dict(completed=completed,wall_s=time.monotonic()-started,
                    latest_primitive=primitive.primitive_id,latest_batch=block,outcomes=counts))
                print(f'{primitive.primitive_id} batch={block} completed={completed} wall={time.monotonic()-started:.1f}s outcomes={counts}',flush=True)
        unchanged=policy.assert_unchanged()
        dump(output/'summary.json',dict(completed=completed,wall_s=time.monotonic()-started,
            per_primitive={p:dict(attempts=sum(r['primitive_id']==p for r in all_rows),
                safe_arrivals=sum(r['primitive_id']==p and r['safe_arrival'] for r in all_rows))
                for p in dict.fromkeys(r['primitive_id'] for r in all_rows)},
            unchanged_policy=unchanged, source_clock_rate=1., learning=False,
            purpose='selected media replay; not new counted evaluations' if selected else 'main diagnostic evaluations'))
    finally:
        env.close()


if __name__ == '__main__':
    main()
