#!/usr/bin/env python3
"""Render predeclared sample and all flagged Phase2 saved trials; no stepping."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from human_punching.wbc.provenance import writable_path,write_json,sha256_file


def sample_bank_id(candidate_id: str, count: int = 8) -> int:
    if count < 1:raise ValueError('Positive declared bank size required')
    return int(hashlib.sha256(('phase2-video-21021:'+candidate_id).encode()).hexdigest()[:16],16)%count


def selection(panel, bank_count=8):
    """Keep the predeclared bank identity even if a ledger is out of order."""
    grouped={}
    for row in panel['trials']:
        grouped.setdefault(row['candidate_id'],[]).append(row)
    chosen=[]
    for candidate,rows in sorted(grouped.items()):
        if Path(candidate).name!=candidate or candidate in ('.','..'):
            raise ValueError('Candidate identity must be one path component')
        random_id=sample_bank_id(candidate,bank_count)
        worlds={}
        for row in rows:
            match=re.fullmatch(re.escape(candidate)+r'_bank_(\d+)',row['trial_id'])
            if match is None:raise ValueError('D0 video trial IDs must preserve explicit candidate/bank identity')
            index=int(match.group(1))
            if not 0<=index<bank_count or index in worlds:raise ValueError('Duplicate or out-of-range bank identity')
            worlds[index]=row
        if set(worlds)!=set(range(bank_count)):
            raise ValueError('Incomplete planned bank ledger; do not silently resample video selection')
        for index,row in sorted(worlds.items()):
            # Rejected/unattempted rows cannot supply nonexistent video evidence.
            reason=[]
            if index==random_id:reason.append('predeclared_hash_sample')
            if not row.get('numeric_complete_cycle',False):reason.append('flagged_failure')
            if reason:chosen.append(dict(candidate_id=candidate,trial_id=row['trial_id'],world=index,
                disposition=row.get('disposition'),reasons=reason))
    return chosen


def finite_render_prefix(qpos,qvel, *, quaternion_tolerance=1.e-5):
    """Keep a visibly labeled finite prefix when a terminal fault is unrenderable."""
    q,v=np.asarray(qpos),np.asarray(qvel)
    if q.ndim!=2 or q.shape[1]!=55 or v.shape!=(len(q),54):
        raise ValueError('Expected complete scalar-body trace coordinates')
    good=np.isfinite(q).all(axis=1)&np.isfinite(v).all(axis=1)
    good&=np.abs(np.linalg.norm(q[:,3:7],axis=1)-1.)<=quaternion_tolerance
    invalid=np.flatnonzero(~good)
    count=int(invalid[0]) if len(invalid) else len(q)
    return count,dict(first_invalid_sample=None if not len(invalid) else int(invalid[0]),
        invalid_sample_indices=invalid.tolist(),raw_samples=len(q),rendered_prefix_samples=count,
        truncated_at_native_invalid_state=bool(len(invalid)),physical_trace_modified=False)


def render(run,panel,output, *, wall_seconds=840., shutdown_reserve_seconds=45.):
    if not np.isfinite(wall_seconds) or not 0<shutdown_reserve_seconds<wall_seconds<=4*3600:
        raise ValueError('Bounded wall limit <=four hours with a positive shutdown reserve required')
    # Reject a symlink escape before graphics configuration or any writes.
    out=writable_path(output)
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Counted GPU allocation required')
    started=time.monotonic()
    from human_punching.wbc.egl_binding import configure_allocated_egl
    binding=configure_allocated_egl()
    from human_punching.jab_evaluation import render_rollout
    import imageio.v2 as imageio
    run=Path(run).resolve(strict=True);run.relative_to(ROOT/'outputs/task10b')
    panel_path=Path(panel).resolve(strict=True);panel_path.relative_to(run)
    p=json.loads(panel_path.read_text());launch=json.loads((run/'launch.json').read_text())
    catalog=json.loads(Path(launch['catalog']).read_text())
    if sha256_file(launch['catalog'])!=launch['catalog_sha256']:raise ValueError('Bound catalog changed')
    cells={r['candidate_id']:r for r in catalog['cells']}
    out.mkdir(parents=True,exist_ok=False)
    planned=selection(p)
    write_json(out/'plan.json',dict(schema='task10b_phase2_media_selection_v1',selection=planned,
        panel=str(panel_path),panel_sha256=sha256_file(panel_path),egl_binding=binding,
        integration=False,inference=False,seed=21021,fps=50,playback_speeds=[1.],
        model_path=str(run/'runtime.xml'),model_sha256=sha256_file(run/'runtime.xml'),
        catalog_sha256=launch['catalog_sha256'],source_sha256=sha256_file(__file__),
        wall_seconds=wall_seconds,shutdown_reserve_seconds=shutdown_reserve_seconds,
        slurm_job_id=os.environ['SLURM_JOB_ID']))
    results=[];stop_reason=None;longest_render=0.
    for chosen in planned:
        candidate=chosen['candidate_id'];source=panel_path.parent/candidate/'raw_trace.npz'
        row=dict(chosen,source=str(source))
        # One video's measured prior cost is reserved in addition to shutdown.
        # An external counted launcher still owns the absolute allocation cutoff.
        if time.monotonic()-started+longest_render>=wall_seconds-shutdown_reserve_seconds:
            stop_reason='media_wall_budget'
        if stop_reason:
            row.update(status='unattempted',reason=stop_reason)
            results.append(row);continue
        if not source.exists():
            row.update(status='unavailable',reason='No executed physical trace for rejected/unattempted request')
        else:
            item_started=time.monotonic()
            try:
                folder=writable_path(out/chosen['trial_id']);folder.relative_to(out)
                folder.mkdir(exist_ok=False)
                with np.load(source,allow_pickle=False) as archive:
                    world=chosen['world'];raw_samples=len(archive['qpos'])
                    q=archive['qpos'][:,world].copy();v=archive['qvel'][:,world].copy()
                    resets=archive['reset_count'][:,world].copy()
                    native_time=archive['time'][:,world].copy()
                n,prefix=finite_render_prefix(q,v)
                row.update(source_sha256=sha256_file(source),prefix=prefix)
                if n<2:
                    row.update(status='unrenderable',reason='Fewer than two finite valid quaternion states in retained prefix',samples=raw_samples)
                else:
                    arrays=dict(time_s=np.arange(n)*.005,qpos=q[:n],qvel=v[:n],physics_step_count=n-1,
                        # The runtime counter is cumulative across separately reset
                        # comparison cells; its difference measures this replay.
                        state_reset_count=1+int(resets[n-1]-resets[0]),simulation_time_s=native_time[:n],
                        physics_timestep_s=.005,policy_timestep_s=.020)
                    partial=(prefix['truncated_at_native_invalid_state'] or chosen['disposition']!='executed')
                    label='Task10B Phase2 '+chosen['trial_id']+' | '+str(chosen['disposition'])
                    if partial:label+=' | PARTIAL FAILED TRACE'
                    result=render_rollout(run/'runtime.xml',arrays,folder,head_center=cells[candidate]['target_w'],
                        label=label,fps=50,playback_speeds=(1.,),artifact_prefix=launch['family'])
                    with imageio.get_reader(result['video']) as video:decoded=sum(1 for _ in video)
                    if decoded!=result['video_frames']:raise ValueError('Full video decode failed')
                    row.update(status='rendered',samples=n,decoded_frames=decoded,media=result,
                        partial_failed_trace=partial,cumulative_reset_counters=[int(resets[0]),int(resets[n-1])],
                        artifact_sha256={Path(result[key]).name:sha256_file(result[key]) for key in ('video','closest_still','final_still')})
            except Exception as error:
                row.update(status='failed',exception=dict(type=type(error).__name__,message=str(error)))
                # A broken renderer/context is not a reason to spend the rest of
                # this allocation repeating the same graphics failure.
                stop_reason='renderer_or_input_failure'
            finally:
                row['wall_time_s']=time.monotonic()-item_started
                longest_render=max(longest_render,row['wall_time_s'])
        results.append(row)
        write_json(out/f'progress_{len(results):04d}.json',dict(results=results,unattempted=planned[len(results):]))
    result=dict(schema='task10b_phase2_media_v1',results=results,planned=len(planned),
        rendered=sum(r['status']=='rendered' for r in results),simulation_steps=0,actor_calls=0,
        unavailable=sum(r['status']=='unavailable' for r in results),unrenderable=sum(r['status']=='unrenderable' for r in results),
        failed=sum(r['status']=='failed' for r in results),unattempted=sum(r['status']=='unattempted' for r in results),
        stop_reason=stop_reason,wall_time_s=time.monotonic()-started,
        visual_recognition_accepted=False,review_required=True)
    write_json(out/'summary.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','panel','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--wall-seconds',type=float,default=840.)
    args=parser.parse_args();print(json.dumps(render(args.run,args.panel,args.output,wall_seconds=args.wall_seconds),indent=2))
