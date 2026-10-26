"""Execute the bounded standing-state Phase 3C bank and source-rate diagnostics.

No snapshot bank is enumerated and no reference is copied or changed. Native
initial checks and clean rendering cover every geometry-eligible candidate,
including non-hitting headings needed by the largest-valid-component rule.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .g1 import PROJECT
from .phase1 import cpu, dump
from .placement_phase3 import INDEX, cache_library
from .standing_bank import (build_geometry, calibrate_landmarks, candidate_placement,
    choice_record, expected_sampler_exposure, finalize_bank, sample_placement_first,
    summarize_bank)
from .standing_fixture import StandingFixture

SEED = 102503


def identity(fixture_path: Path) -> dict:
    sources = list(Path(__file__).parent.glob('*.py'))
    index = json.loads(INDEX.read_text())
    references = [PROJECT / p[key] for p in index['primitives']
                  for key in ('reference_npz', 'reference_json')]
    files = sources + references + [INDEX, fixture_path / 'fixture.npz',
        fixture_path / 'fixture.json', PROJECT/'configs/task10d/phase3c_standing_20261025.json',
        PROJECT/'scripts/task10d_standing.sbatch']
    return {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def native_and_clean(paths, fixture, geometry, output: Path, batch_size: int):
    """Cheap geometry -> native cross-character screen -> actual clean C render."""
    from .phase3_render import InitialIntersections, assert_unchanged, create_batch, state_copy
    shape = geometry['outward_hit'].shape
    flags = {key: np.zeros(shape, bool) for key in
             ('native_checked', 'native_valid', 'clean_checked', 'clean_valid')}
    native_min = np.full(shape, np.nan, np.float32)
    pairs = np.full(shape, -1, np.int16)
    metrics = {key: np.full((*shape, 2), -1, np.int32) for key in
        ('glove_visible_pixels', 'glove_valid_source_pixels', 'glove_rgb_valid_source_pixels',
         'glove_clean_cells', 'glove_acquired_cells', 'glove_clean_min_cells',
         'glove_acquired_min_cells', 'glove_h2_body_occluded_pixels')}
    env, g1, binding, camera = create_batch(dict(seed=SEED), output,
        paths[0].primitive.reference, 25., batch_size)
    started = time.monotonic()
    checked = rendered = 0
    try:
        restore = fixture.restore(g1, camera_id=camera.ids.camera_id)
        before = state_copy(g1)
        dump(output/'runtime.json', dict(runtime=g1.runtime_identity, model=g1.model_audit,
            camera=camera.metadata(), fixture_restore=restore))
        checker = InitialIntersections(env.sim.mj_model, binding)
        q = fixture.arrays['sim/data/qpos'][0]
        v = fixture.arrays['sim/data/qvel'][0]
        for p, path in enumerate(paths):
            candidates = np.argwhere(geometry['geometric_initial_valid'][p])
            pending = []

            def render_batch(items):
                nonlocal rendered
                padded = items + [items[-1]]*(batch_size-len(items))
                with g1.ordered_execution():
                    for i, (_, _, sample) in enumerate(padded):
                        binding.write(sample, g1.data, env_ids=[i])
                    # acquired_* is clean RGB-HSV before faults. Random fault
                    # draws are never consulted to admit this initial scene.
                    exposure = camera.capture(0., reset=True, include_rgb=False)
                    np.testing.assert_allclose(exposure['camera_position'],
                        np.repeat(fixture.geometry['camera_position_world'][None],batch_size,axis=0),
                        atol=2e-6, rtol=0)
                    np.testing.assert_allclose(exposure['camera_rotation'],
                        np.repeat(fixture.geometry['camera_rotation'][None],batch_size,axis=0),
                        atol=2e-6, rtol=0)
                    np.testing.assert_allclose(cpu(g1.data.geom_xpos[:,binding.glove_geom_ids]),
                        np.stack([s.glove_pos for _,_,s in padded]), atol=2e-6, rtol=0)
                    np.testing.assert_array_equal(exposure['padding'], True)
                    np.testing.assert_array_equal(exposure['capture_timestamps'], 0)
                    assert_unchanged(g1, before)
                for i, (x,y,_) in enumerate(items):
                    flags['clean_checked'][p,x,y] = True
                    for key in metrics:
                        metrics[key][p,x,y] = exposure[key][i]
                    flags['clean_valid'][p,x,y] = bool(
                        (exposure['glove_rgb_valid_source_pixels'][i] > 0).all()
                        and (exposure['glove_acquired_min_cells'][i] > 0).all())
                rendered += len(items)

            for x,y in candidates:
                placement = candidate_placement(path, fixture.geometry,
                    geometry['xy_G'][x], geometry['yaw_deg'][y])
                sample = path.primitive.reference.sample(0., placement)
                np.testing.assert_allclose(sample.body_pos[0,:2],
                    geometry['actual_pelvis_world_xy'][x], atol=1e-11, rtol=0)
                np.testing.assert_allclose(sample.body_pos[:,2],
                    path.primitive.reference.sample(0.).body_pos[:,2], atol=1e-11, rtol=0)
                overlap = checker.measure(q,v,sample)
                flags['native_checked'][p,x,y] = True
                flags['native_valid'][p,x,y] = not overlap['gross_intersection']
                native_min[p,x,y] = overlap['minimum_native_convex_distance_capped_m']
                pairs[p,x,y] = overlap['tested_near_pairs']
                checked += 1
                if not overlap['gross_intersection']:
                    pending.append((int(x),int(y),sample))
                if len(pending) == batch_size:
                    render_batch(pending)
                    pending = []
                if checked % 2000 == 0:
                    progress = dict(native_checked=checked, clean_rendered=rendered,
                        primitive_id=path.primitive.primitive_id, wall_s=time.monotonic()-started)
                    print(json.dumps(progress), flush=True)
                    dump(output/'progress.json', progress)
            if pending:
                render_batch(pending)
            print(f'COMPLETE {path.primitive.primitive_id} native={len(candidates)} '
                  f'clean={flags["clean_valid"][p].sum()} wall={time.monotonic()-started:.1f}', flush=True)
            # Compact partial result makes interrupted cases explicitly unknown.
            np.savez_compressed(output/'measurements.npz', **flags, **metrics,
                native_min_convex_distance_capped_m=native_min, native_near_pairs=pairs)
        assert_unchanged(g1, before)
    finally:
        env.close()
    return dict(**flags, **metrics, native_min_convex_distance_capped_m=native_min,
                native_near_pairs=pairs)


def make_maps(bank, output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xy=bank['xy_G']
    panels=[('Raw aimed choices',bank['raw_aimed'].sum(axis=(0,2))),
            ('Retained aimed choices',bank['retained_aimed'].sum(axis=(0,2))),
            ('Retained aimed primitive count',bank['retained_aimed'].any(-1).sum(0)),
            ('Retained aimed family count',np.stack([
                bank['retained_aimed'][bank['families']==f].any(axis=(0,2))
                for f in np.unique(bank['families'])]).sum(0))]
    for domain, mask in [('primary',bank['primary_xy']),('exploratory',~bank['primary_xy'])]:
        fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
        for ax,(label,values) in zip(axes.ravel(),panels):
            ax.scatter(xy[:,0],xy[:,1],c='#e8e8e8',marker='s',s=30)
            artist=ax.scatter(xy[mask,0],xy[mask,1],c=values[mask],marker='s',s=34,
                cmap='viridis',vmin=0)
            fig.colorbar(artist,ax=ax)
            ax.set(xlabel='H2 initial pelvis X in G (m)',ylabel='Y in G (m)',
                title=label,aspect='equal',xlim=(.4,1.55),ylim=(-.6,.6))
        fig.suptitle(f'{domain.capitalize()} domain — one standing fixture; exact tested headings')
        fig.savefig(output/f'{domain}_availability.png',dpi=170)
        plt.close(fig)
    for key,label in [('raw_valid','Clean-valid yaw samples'),('retained_aimed','Retained aimed yaw samples')]:
        fig,axes=plt.subplots(2,4,figsize=(18,9),layout='constrained')
        for p,ax in enumerate(axes.ravel()):
            values=bank[key][p].sum(-1)
            artist=ax.scatter(xy[:,0],xy[:,1],c=values,marker='s',s=18,cmap='viridis',vmin=0)
            ex=~bank['primary_xy'] & (values>0)
            ax.scatter(xy[ex,0],xy[ex,1],facecolors='none',edgecolors='red',s=25,linewidths=.5)
            ax.set(title=str(bank['primitive_ids'][p]),xlabel='X (m)',ylabel='Y (m)',aspect='equal')
            fig.colorbar(artist,ax=ax)
        fig.suptitle(label+'; red outline = exploratory, not primary admission')
        fig.savefig(output/f'{key}_by_primitive.png',dpi=160)
        plt.close(fig)


def selected_trials(bank, landmarks):
    trials=[]
    for p,landmark in enumerate(landmarks['landmarks']):
        if not landmark['supported']:
            continue
        nominal=landmark['calibration_choice']
        trials.append(dict(**nominal, selection_kind='nominal_calibration',
            label=f'{p:02d}_{nominal["primitive_id"]}_calibration'))
        candidates=np.argwhere(bank['retained_aimed'][p]&bank['primary_xy'][:,None])
        different=[(int(x),int(y)) for x,y in candidates if int(x)!=nominal['xy_index']]
        if different:
            x,y=min(different,key=lambda t:(
                .14-bank['outward_min_distance_m'][p,t[0],t[1]],t[0],t[1]))
            trials.append(dict(**choice_record(bank,p,x,y), selection_kind='tight_frozen_reach_margin',
                label=f'{p:02d}_{nominal["primitive_id"]}_tight'))
    if bank['primary_sampler_xy'].any():
        for i,row in enumerate(sample_placement_first(bank,np.random.default_rng(102525),8)):
            trials.append(dict(**row,selection_kind='placement_first_random',label=f'sample_{i:02d}_{row["primitive_id"]}'))
    return trials


def dynamic_trials(paths, fixture, bank, trials, output):
    from .phase3_render import create_batch
    from .standing_replay import run_dynamic
    env,g1,binding,camera=create_batch(dict(seed=SEED),output,paths[0].primitive.reference,25.,1)
    results=[]
    try:
        for row in trials:
            path=paths[row['primitive_index']]
            selector=lambda state: candidate_placement(path,state,row['xy_G_m'],row['yaw_deg'])
            result=run_dynamic(g1,binding,camera,path.primitive,fixture,selector,output,
                row['label'],selection=row)
            results.append(result)
            dump(output/'results.json',results)
            print(f'DYNAMIC {row["label"]} '+json.dumps({key:result.get(key) for key in
                ('moving_head_clearance_violated','moving_head_analytic_encounter','unaided_analytic_miss','mechanical_failure')}),flush=True)
    finally:
        env.close()
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--bank',type=Path)
    parser.add_argument('--mode',choices=('build','sample','replay'),default='build')
    parser.add_argument('--count',type=int,default=8)
    parser.add_argument('--seed',type=int,default=102525)
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    fixture_path=args.fixture.resolve()
    if fixture_path.is_file():
        fixture_path=fixture_path.parent
    fixture=StandingFixture.load(fixture_path)
    if args.mode=='sample':
        if args.bank is None:
            parser.error('--bank required for sample')
        bank=dict(np.load(args.bank,allow_pickle=False))
        print(json.dumps(dict(samples=sample_placement_first(bank,np.random.default_rng(args.seed),args.count),
            exposure=expected_sampler_exposure(bank)),indent=2))
        return
    import torch
    import warp as wp
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1:
        raise RuntimeError('One allocated CUDA GPU is required')
    wp.config.kernel_cache_dir=os.environ['WARP_CACHE_PATH']
    if args.output is None:
        parser.error('--output required for build/replay')
    output=args.output.resolve()
    if not output.is_relative_to(PROJECT/'outputs/task10d/phase3'):
        raise ValueError('Use a fresh Phase 3 output directory')
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    source_identity=identity(fixture_path)
    dump(output/'invocation.json',dict(mode=args.mode,fixture=str(fixture_path),
        batch_size=args.batch_size,seed=SEED,sampler_seed=args.seed,job_id=os.environ['SLURM_JOB_ID'],
        source_identity=source_identity,noise=True,residual=0,source_rate=1,
        pdf_sha256='5cb252daf7b303fc47af176098cab085707a5c5c3d4d179e45855fcc05362cbc'))
    paths=cache_library()
    if args.mode=='build':
        geometry=build_geometry(paths,fixture.geometry)
        print(f'GEOMETRY candidates={geometry["outward_hit"].size} '
              f'initial_screen={geometry["geometric_initial_valid"].sum()} '
              f'outward_hits={geometry["outward_hit"].sum()}',flush=True)
        measurements=native_and_clean(paths,fixture,geometry,output,args.batch_size)
        flags={key:measurements.pop(key) for key in ('native_checked','native_valid','clean_checked','clean_valid')}
        bank=finalize_bank(geometry,**flags)
        bank.update(measurements)
        bank['fixture_id']=np.array(fixture.metadata['fixture_id'])
        np.savez_compressed(output/'bank.npz',**bank)
        summary=summarize_bank(bank)
        summary['expected_sampler_exposure']=expected_sampler_exposure(bank)
        dump(output/'bank_summary.json',summary)
        landmarks=calibrate_landmarks(paths,bank,fixture_id=fixture.metadata['fixture_id'])
        dump(output/'fixed_landmarks.json',landmarks)
        make_maps(bank,output)
        trials=selected_trials(bank,landmarks)
    else:
        if args.bank is None:
            parser.error('--bank required for replay')
        bank=dict(np.load(args.bank,allow_pickle=False))
        if str(bank['fixture_id'])!=fixture.metadata['fixture_id']:
            raise ValueError('Bank fixture identity mismatch')
        trials=[dict(**row,selection_kind='placement_first_random',label=f'sample_{i:02d}_{row["primitive_id"]}')
                for i,row in enumerate(sample_placement_first(bank,np.random.default_rng(args.seed),args.count))]
    dump(output/'selected_trials.json',dict(selection_before_dynamic_outcomes=True,trials=trials,
        family_counts=dict(Counter(row['family'] for row in trials)),
        variant_counts=dict(Counter(row['primitive_id'] for row in trials))))
    results=dynamic_trials(paths,fixture,bank,trials,output/'dynamic') if trials else []
    unchanged=identity(fixture_path)==source_identity
    dump(output/'completion.json',dict(job_id=os.environ['SLURM_JOB_ID'],wall_s=time.monotonic()-started,
        trials=len(results),source_and_input_identity_unchanged=unchanged,
        fixture_id=fixture.metadata['fixture_id'],source_rate=1,phase4_run=False,
        phase3c_complete_for_review=True))
    if not unchanged:
        raise AssertionError('Source/input identity changed during execution')


if __name__=='__main__':
    main()
