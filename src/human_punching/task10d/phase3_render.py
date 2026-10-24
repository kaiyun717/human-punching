"""Focused Phase 3 initial-scene measurements and source-clock diagnostic media.

All decisions are made at reset; the dynamic loop uses a committed rigid
placement. The existing renderer, foreground reduction and sensor history run
unchanged. Per-hand attribution is diagnostic only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np

from .g1 import PROJECT, RESET_KEYS, DynamicG1, make_task08_cfg
from .phase1 import cpu, dump, synchronized_geometry
from .phase2 import load_library
from .replay import CommittedReplay, Placement

PRIMITIVE_IDS = ['jab_nominal', 'cross_nominal', 'left_hook_nominal',
    'right_hook_nominal', 'jab_004_01107', 'cross_012_01441',
    'left_hook_013_03243', 'right_hook_014_02074']
LIBRARY = PROJECT / 'constants/kinematic_punches/v1_20261023/library_index.json'


def state_copy(g1):
    manager = g1.env.action_manager
    return {k: cpu(v) for k, v in dict(time=g1.data.time, qpos=g1.data.qpos,
        qvel=g1.data.qvel, action=manager.action, prev_action=manager.prev_action,
        prev_prev_action=manager.prev_prev_action, ctrl=g1.data.ctrl).items()}


def assert_unchanged(g1, before):
    after = state_copy(g1)
    for key in before:
        np.testing.assert_array_equal(after[key], before[key], err_msg=key)
    return {key: True for key in before}


class InitialIntersections:
    """Cross-character native convex distance diagnostic; never enable contact.

Only broad-phase near pairs need exact native queries. A penetration deeper than
3 cm is a declared gross-intersection search exclusion, not a safety certificate.
Native mesh distances use convex hulls; visual review retains that qualification.
"""
    def __init__(self, model, binding):
        self.model, self.binding = model, binding
        self.data = mujoco.MjData(model)
        self.g1 = np.array([i for i in range(model.ngeom)
            if (model.body(int(model.geom_bodyid[i])).name or '').startswith('robot/')
            and (model.geom_contype[i] or model.geom_conaffinity[i])])
        self.h2 = np.flatnonzero(np.isin(model.geom_bodyid, binding.body_ids))

    def measure(self, qpos, qvel, sample):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.binding.write(sample, self.data)
        mujoco.mj_kinematics(self.model, self.data)
        separation = np.linalg.norm(self.data.geom_xpos[self.g1, None]
            - self.data.geom_xpos[self.h2][None], axis=-1)
        bound = self.model.geom_rbound[self.g1, None] + self.model.geom_rbound[self.h2][None]
        pairs = np.argwhere(separation <= bound + .02)
        minimum, names = .02, None
        for a, b in pairs:
            ga, gb = int(self.g1[a]), int(self.h2[b])
            distance = mujoco.mj_geomDistance(self.model, self.data, ga, gb, .02, None)
            if distance < minimum:
                minimum, names = float(distance), [self.model.geom(ga).name, self.model.geom(gb).name]
        return dict(gross_intersection=minimum < -.03,
            minimum_native_convex_distance_capped_m=minimum, closest_pair=names,
            tested_near_pairs=len(pairs), threshold_m=-.03)


def create_batch(config, output, reference, pitch, count):
    from mjlab.envs import ManagerBasedRlEnv
    from .camera import Task08Camera, add_camera_only, mount_config
    mount = mount_config(pitch)
    cfg = make_task08_cfg(count=count, seed=config['seed'], output_dir=output,
        extra_entities={'h2': reference.entity_cfg()},
        scene_spec_fn=lambda spec: add_camera_only(spec, camera_config=mount))
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    g1 = DynamicG1(env, noise_seed=config['seed'] + 1,
        observation_noise_enabled=True)
    binding = reference.bind(env.sim.mj_model)
    camera = Task08Camera(env.sim, binding, seed=config['seed'] + 2, camera_config=mount)
    return env, g1, binding, camera


def load_rows(geometry, indices):
    return dict(snapshot_ids=geometry['snapshot_id'][indices],
        arrays={key: geometry[key][indices] for key in RESET_KEYS},
        transforms=geometry['world_from_collection'][indices])


def initial_panel(path, title, external, exposure, index=0):
    from PIL import Image, ImageDraw
    from .media import depth_panel
    panel = Image.new('RGB', (2016, 848), '#151b23')
    views = [Image.fromarray(external[0]), Image.fromarray(external[1]),
             Image.fromarray(exposure['rgb'][index])]
    for col, view in enumerate(views):
        panel.paste(view, (col * 672, 48))
    raw = exposure['source_depth'][index]
    mask = np.isfinite(raw) & (raw >= .1) & (raw <= 3.)
    source = depth_panel((raw - .1)/2.9, mask)
    panel.paste(source, (0, 472))
    panel.paste(depth_panel(exposure['clean_normalized'][index],
        exposure['clean_masks'][index]), (672, 472))
    panel.paste(depth_panel(exposure['acquired_normalized'][index],
        exposure['acquired_masks'][index]), (1344, 472))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), title, fill='white')
    for col, label in enumerate(['External oblique', 'External side', 'Actual clean rendered RGB']):
        draw.text((10+672*col, 28), label, fill='white')
    for col, label in enumerate(['Full-scene valid optical depth 672x376',
        'Clean ID-union masked-min 16x9', 'Clean RGB-HSV union masked-min 16x9']):
        draw.text((10+672*col, 440), label, fill='white')
    panel.save(path)


def dynamic_preview(g1, binding, camera, primitive, rows, placement, output, label):
    """Source timing only; admission is independently active-outward in search."""
    import torch
    from .media import DiagnosticMedia
    reset = g1.reset(rows)
    before = state_copy(g1)
    with g1.ordered_execution():
        geometry, _ = synchronized_geometry(g1, binding, primitive.reference.sample(0, placement))
        frozen = cpu(geometry['head_position'])[0]
    replay = CommittedReplay(primitive.reference, tuple(frozen), placement)
    media = DiagnosticMedia(output / label, camera.renderer.host_model, label)
    steps = 4 * int(np.ceil((replay.duration_s + .4)/.02))
    trace, camera_rows = [], []
    frozen_first, moving_first = [None, None], [None, None]
    first_hazard = None
    contract = None
    try:
        for step in range(steps+1):
            elapsed = .005*step
            sample = replay.sample(elapsed)
            with g1.ordered_execution(), torch.no_grad():
                geometry, gloves = synchronized_geometry(g1, binding, sample)
                head = cpu(geometry['head_position'])[0]
                frozen_dist = np.linalg.norm(gloves-frozen, axis=-1)
                moving_dist = np.linalg.norm(gloves-head, axis=-1)
                for hand in range(2):
                    if frozen_first[hand] is None and frozen_dist[hand] <= .14:
                        frozen_first[hand] = elapsed
                    if moving_first[hand] is None and moving_dist[hand] <= .14:
                        moving_first[hand] = elapsed
                if first_hazard is None and moving_dist.min() < .16:
                    first_hazard = elapsed
                trace.append([elapsed, *head, *gloves.ravel(), *frozen_dist, *moving_dist,
                    *cpu(g1.data.qpos)[0, :3]])
                if step % 4 == 0:
                    exposure = camera.capture(elapsed, reset=step == 0, include_source=step == 0)
                    if step == 0:
                        contract = assert_unchanged(g1, before)
                        assert float(cpu(g1.data.time)[0]) == 0
                        assert bool(exposure['padding'][0])
                        assert int(exposure['capture_frame_ids'][0]) == 0
                        assert float(exposure['capture_timestamps'][0]) == 0
                        ext = media.renderer.render(g1.data, 0)
                        initial_panel(output / label / 'initial_scene.png', label, ext, exposure)
                    camera_rows.append({key: value.tolist() for key, value in exposure.items()
                        if key in ('capture_timestamps', 'delivered_capture_timestamps',
                        'delivery_timestamps', 'padding', 'glove_valid_source_pixels',
                        'glove_clean_min_cells', 'glove_acquired_min_cells', 'delay_ticks',
                        'delivered_masks', 'delivered_normalized', 'corrupted_masks')})
                    media.append(g1.data, exposure, time_s=elapsed, source_s=sample.source_time_s,
                        hazard=first_hazard, keyframe=step == 0 or step == steps)
                    if step < steps:
                        g1.apply_physical_target(g1.control_action())
                if step < steps:
                    binding.write(replay.sample((step+1)*.005), g1.data)
                    g1.step_substep()
    finally:
        media.close()
    values = np.asarray(trace)
    np.savez_compressed(output / label / 'trace.npz', values=values,
        columns=np.array(['time','head_x','head_y','head_z','left_x','left_y','left_z',
        'right_x','right_y','right_z','frozen_left','frozen_right','moving_left',
        'moving_right','pelvis_x','pelvis_y','pelvis_z']))
    result = dict(label=label, primitive_id=primitive.primitive_id,
        snapshot_id=int(rows['snapshot_ids'][0]), reset=reset, atomic_unchanged=contract,
        frozen_target=frozen, placement=asdict(placement), clock='source',
        frozen_first_by_hand_s=frozen_first, moving_first_by_hand_s=moving_first,
        first_clearance_violation_s=first_hazard, full_cycle_end_s=steps*.005,
        g1_max_pelvis_displacement_m=float(np.linalg.norm(values[:,-3:]-values[0,-3:],axis=1).max()),
        frames=media.count, diagnostic_continuation=True, safe_arrival_evaluated=False)
    dump(output / label / 'summary.json', result)
    dump(output / label / 'camera.json', camera_rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--dynamic', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((PROJECT/'outputs/task10d/phase3').resolve()):
        raise ValueError('Output must be inside Phase 3')
    output.mkdir(parents=True, exist_ok=False)
    import torch
    import warp as wp
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
        raise RuntimeError('One allocated GPU required')
    wp.config.kernel_cache_dir = os.environ['WARP_CACHE_PATH']
    requests = json.loads(args.requests.read_text())
    requests = requests['requests'] if isinstance(requests, dict) else requests
    geometry = dict(np.load(args.geometry, allow_pickle=False))
    lookup = {int(sid): i for i, sid in enumerate(geometry['snapshot_id'])}
    primitives = {p.primitive_id: p for p in load_library(LIBRARY, PRIMITIVE_IDS)}
    config = dict(seed=102403, nominal_observation_noise=True)
    dump(output/'invocation.json', dict(requests=str(args.requests.resolve()),
        geometry=str(args.geometry.resolve()), batch_size=args.batch_size,
        job_id=os.environ['SLURM_JOB_ID'], dynamic=args.dynamic,
        requests_sha256=hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        geometry_sha256=hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob('*.py')}))
    started, results = time.monotonic(), []
    for pitch in sorted({float(r['pitch_up_deg']) for r in requests}):
        selected = [r for r in requests if float(r['pitch_up_deg']) == pitch]
        count = 1 if args.dynamic else args.batch_size
        env, g1, binding, camera = create_batch(config, output,
            primitives[PRIMITIVE_IDS[0]].reference, pitch, count)
        external_renderer = None
        try:
            dump(output/f'runtime_pitch_{pitch:g}.json', dict(runtime=g1.runtime_identity,
                model=g1.model_audit, camera=camera.metadata()))
            intersections = InitialIntersections(env.sim.mj_model, binding)
            if not args.dynamic and any(r.get('save_panel') for r in selected):
                from .media import ExternalRenderer
                external_renderer = ExternalRenderer(camera.renderer.host_model)
            for offset in range(0, len(selected), count):
                items = selected[offset:offset+count]
                if args.dynamic:
                    r = items[0]
                    placement = Placement(**r['placement'])
                    rows = load_rows(geometry, [lookup[int(r['snapshot_id'])]])
                    results.append(dynamic_preview(g1, binding, camera,
                        primitives[r['primitive_id']], rows, placement, output, r['label']))
                else:
                    padded = items + [items[-1]]*(count-len(items))
                    indices = [lookup[int(r['snapshot_id'])] for r in padded]
                    reset = g1.reset(load_rows(geometry, indices))
                    before = state_copy(g1)
                    samples = [primitives[r['primitive_id']].reference.sample(0,
                        Placement(**r['placement'])) for r in padded]
                    with g1.ordered_execution():
                        for i, sample in enumerate(samples):
                            binding.write(sample, g1.data, env_ids=[i])
                        exposure = camera.capture(0, reset=True,
                            include_source=any(r.get('save_panel') for r in items))
                        from .placement_phase3 import mount_optical_rotation
                        # Native batched geometry must match the independently
                        # prepared snapshot FK and exactly one H2 placement.
                        actual_head = cpu(g1.geometry()['head_position'])
                        np.testing.assert_allclose(actual_head,
                            geometry['head_world'][indices], atol=5e-6, rtol=0.)
                        np.testing.assert_allclose(exposure['camera_position'],
                            geometry['camera_position_world'][indices], atol=5e-6, rtol=0.)
                        expected_rotation = (geometry['torso_rotation'][indices]
                            @ mount_optical_rotation(pitch))
                        np.testing.assert_allclose(exposure['camera_rotation'],
                            expected_rotation, atol=5e-6, rtol=0.)
                        np.testing.assert_allclose(cpu(g1.data.geom_xpos[:, binding.glove_geom_ids]),
                            np.stack([sample.glove_pos for sample in samples]), atol=5e-6, rtol=0.)
                        for key in ('capture_timestamps', 'delivered_capture_timestamps',
                                    'delivery_timestamps', 'capture_frame_ids', 'delivered_frame_ids'):
                            np.testing.assert_array_equal(exposure[key], 0,
                                err_msg=f'Initial batch {key} must retain the true reset timestamp/ID')
                        np.testing.assert_array_equal(exposure['padding'], True)
                        np.testing.assert_array_equal(cpu(g1.data.time), 0)
                        contract = assert_unchanged(g1, before)
                        contract.update(prepared_head_camera_match=True,
                            placed_glove_geometry_match=True, reset_padding_truthful=True)
                    for i, r in enumerate(items):
                        if 'common_initial_pelvis_xy' in r:
                            from scipy.spatial.transform import Rotation
                            np.testing.assert_allclose(samples[i].body_pos[0, :2],
                                r['common_initial_pelvis_xy'], atol=5e-7, rtol=0.)
                            np.testing.assert_allclose(
                                cpu(g1.data.mocap_pos)[i, binding.mocap_ids[0], :2],
                                r['common_initial_pelvis_xy'], atol=5e-7, rtol=0.)
                            np.testing.assert_allclose(
                                cpu(g1.data.mocap_pos)[i, binding.mocap_ids[0], 2],
                                r['expected_initial_pelvis_z_m'], atol=5e-7, rtol=0.)
                            forward = Rotation.from_quat(
                                cpu(g1.data.mocap_quat)[i, binding.mocap_ids[0]],
                                scalar_first=True).as_matrix()[:, 0]
                            actual_yaw = np.arctan2(forward[1], forward[0])
                            delta = actual_yaw - r['common_initial_pelvis_yaw_rad']
                            assert abs(np.arctan2(np.sin(delta), np.cos(delta))) < 5e-7
                        if r.get('save_panel'):
                            from types import SimpleNamespace
                            label = r['label']
                            if Path(label).name != label or label in ('', '.', '..'):
                                raise ValueError('Panel label must be one filename component')
                            panel_path = output/'initial_scenes'/f'{label}.png'
                            panel_path.parent.mkdir(exist_ok=True)
                            assert not panel_path.exists(), panel_path
                            subset = SimpleNamespace(**{name: cpu(getattr(g1.data, name))[i:i+1]
                                for name in ('qpos', 'qvel', 'mocap_pos', 'mocap_quat')})
                            external = external_renderer.render(subset, 0.)
                            initial_panel(panel_path, label, external, exposure, index=i)
                        overlap = intersections.measure(before['qpos'][i], before['qvel'][i], samples[i])
                        metrics = {key: value[i].tolist() for key, value in exposure.items()
                            if key.startswith('glove_') or key in ('capture_timestamps',
                            'delivered_capture_timestamps', 'padding')}
                        clean = all(x > 0 for x in metrics['glove_clean_min_cells'])
                        acquired = all(x > 0 for x in metrics['glove_acquired_min_cells'])
                        results.append(dict(**r, **overlap, metrics=metrics,
                            clean_both_represented=clean, rgb_both_represented=acquired,
                            actual_initial_pelvis_world=cpu(g1.data.mocap_pos)[i,
                                binding.mocap_ids[0]].tolist(),
                            atomic_preserved=all(contract.values())))
                if offset % (count*10) == 0:
                    print(f'pitch={pitch:g} {offset+len(items)}/{len(selected)} wall={time.monotonic()-started:.1f}', flush=True)
                    dump(output/'progress.json', dict(completed=len(results), wall_s=time.monotonic()-started))
                with (output/'measurements.jsonl').open('a') as handle:
                    for row in results[-len(items):]:
                        from .phase1 import jsonable
                        handle.write(json.dumps(row, default=jsonable, allow_nan=False)+'\n')
        finally:
            if external_renderer is not None:
                external_renderer.close()
            env.close()
    dump(output/'summary.json', dict(completed=len(results), wall_s=time.monotonic()-started,
        job_id=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    main()
