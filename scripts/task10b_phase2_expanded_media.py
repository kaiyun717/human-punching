#!/usr/bin/env python3
"""Render declared expanded/shared traces and every flagged attempt; no replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from human_punching.phase2_expanded_evaluation import load_declaration
from human_punching.phase2_phases import PhaseSchedule
from human_punching.phase2_deadline import ExecutionDeadline
from human_punching.wbc.provenance import writable_path,write_json,sha256_file
from task10b_phase2_media import finite_render_prefix


def owned(path,*,parent=None):
    value=Path(path).resolve(strict=True)
    if not value.is_relative_to(ROOT.resolve()) or not value.is_file():
        raise ValueError('Media inputs must resolve to regular human-punching files')
    if parent is not None and not value.is_relative_to(Path(parent).resolve()):
        raise ValueError('Media trace/plan escapes its declared run')
    return value


def selection(declaration,outcomes):
    """Keep fixed video IDs and all observed failures without outcome resampling."""
    records={row['trial_id']:row for row in outcomes}
    if len(records)!=len(outcomes) or set(records)-{b.trial_id for b in declaration.bindings}:
        raise ValueError('Duplicate or undeclared outcome trial ID')
    chosen=[]
    for binding in declaration.bindings:
        result=records.get(binding.trial_id);reasons=[]
        completed=(result or {}).get('disposition')=='completed_native_attempt'
        if binding.trial_id in declaration.video_trial_ids:reasons.append('predeclared_video_trial_id')
        if result is None:reasons.append('missing_outcome')
        elif (result.get('numeric_complete_cycle') is not True or result.get('execution_complete',completed) is not True
                or result.get('status','executed' if completed else None) not in ('executed','executed_pending_audit_and_video')
                or any(result.get(key) is True for key in ('fallen','nonfinite','root_assistance','effort_cap_violation','movement_exploit'))
                or any(result.get(key) is False for key in ('trace_audit_pass','immutable_launch_pass','continuity_history_pass',
                    'functional_recovery_pass','single_outward_stroke_pass')) or result.get('visual_review')=='rejected'):
            reasons.append('flagged_failure_rejection_or_partial')
        if reasons:chosen.append(dict(trial_id=binding.trial_id,command_id=binding.command_id,ttc_s=binding.ttc_s,
            start_index=binding.start_index,reasons=reasons,request_identity=binding.identity,
            disposition=(result or {}).get('disposition','missing_outcome')))
    return chosen


def source_runs(run):
    """Follow only hash-bound continuation parents; never search other packets."""
    result=[];seen=set();current=Path(run).resolve(strict=True)
    while True:
        if current in seen or len(result)>=64:raise ValueError('Cyclic or unbounded media continuation ancestry')
        seen.add(current);result.append(current)
        path=current/'continuation.json'
        if not path.exists():return result
        continuation=json.loads(owned(path,parent=current).read_text())
        for key in ('parent_execution','parent_outcomes'):
            evidence=owned(continuation[key])
            if sha256_file(evidence)!=continuation[key+'_sha256']:raise ValueError('Media continuation parent evidence changed')
        identity=json.loads(owned(current/'execution_identity.json',parent=current).read_text())
        parent_execution=json.loads(owned(continuation['parent_execution']).read_text())
        if (identity['identity']!=continuation['same_frozen_execution_identity']
                or parent_execution['execution_identity']!=identity['identity']):
            raise ValueError('Continuation media changes frozen execution identity')
        parent=Path(continuation['parent_run']).resolve(strict=True)
        if not parent.is_relative_to(ROOT/'outputs/task10b') or not parent.is_dir():raise ValueError('Continuation parent outside Task10B outputs')
        current=parent


def index_batches(run,declaration,cells):
    bindings={b.trial_id:b for b in declaration.bindings};result={}
    for path in sorted(Path(run).glob('batch_*/plan.json')):
        path=owned(path,parent=run);plan=json.loads(path.read_text())
        trials=plan.get('trials',[]);candidates=plan.get('candidates',[])
        commitment_path=path.parent/'launch_commitments.json';commitments={}
        if commitment_path.exists():
            receipt=json.loads(owned(commitment_path,parent=run).read_text())
            records=receipt.get('commitments',[]);commitments={row['trial_id']:row for row in records}
            if len(commitments)!=len(records) or set(commitments)-set(trials):raise ValueError('Duplicate/unknown launch commitment in media batch')
        if (len(trials)!=len(candidates) or len(set(trials))!=len(trials)
                or type(plan.get('planned_physics_steps')) is not int or not 0<plan['planned_physics_steps']<=6204):
            raise ValueError('Bounded complete native batch plan with explicit worlds required')
        for world,(trial,candidate) in enumerate(zip(trials,candidates,strict=True)):
            if trial not in bindings or trial in result or candidate not in cells:
                raise ValueError('Unknown or duplicate physical trial/candidate in native plans')
            binding=bindings[trial];row=cells[candidate]
            if (row['family']!=declaration.family or row['reference_sha256']!=binding.reference_sha256
                    or row['requested_ttc_s']!=binding.ttc_s or row['command_id']!=binding.command_id
                    or tuple(row['target_w'])!=binding.target_w
                    or (declaration.start_mode=='shared_jab_saved_state' and row.get('start_index')!=binding.start_index)):
                raise ValueError('Native media world differs from exact frozen trial binding')
            trace=path.parent/'raw_trace.npz'
            if trace.exists():trace=owned(trace,parent=run)
            result[trial]=dict(world=world,candidate_id=candidate,plan_path=str(path),plan_sha256=sha256_file(path),
                raw_trace_path=str(trace),planned_samples=plan['planned_physics_steps']+1,
                planned_worlds=len(trials),row=row,
                runtime_model_path=str(Path(run)/'runtime.xml'),
                actual_launch_s=commitments.get(trial,{}).get('actual_launch_time_s'),
                launch_disposition=commitments.get(trial,{}).get('disposition','no_retained_commitment'),
                launch_commitment_sha256=sha256_file(commitment_path) if commitment_path.exists() else None,
                actual_actor_tensor_digest=plan.get('actor_sha256'))
    return result


def extract_world(source,world,*,planned_samples,planned_worlds):
    with np.load(source,allow_pickle=False) as archive:
        q=archive['qpos'];v=archive['qvel']
        if (q.ndim!=3 or q.shape[1:]!=(planned_worlds,55) or v.shape!=(len(q),planned_worlds,54)
                or not 1<=len(q)<=planned_samples or not 0<=world<planned_worlds):
            raise ValueError('Trace shape, world or predeclared horizon differs')
        poses=q[:,world].copy();velocities=v[:,world].copy()
        resets=archive['reset_count'][:,world].copy();native_time=archive['time'][:,world].copy()
    n,prefix=finite_render_prefix(poses,velocities)
    if resets.shape!=(len(poses),) or native_time.shape!=(len(poses),):raise ValueError('Trace clocks/reset counters differ')
    prefix.update(planned_samples=planned_samples,retained_samples=len(poses),
        incomplete_native_horizon=len(poses)<planned_samples)
    arrays=dict(time_s=np.arange(n)*.005,qpos=poses[:n],qvel=velocities[:n],physics_step_count=max(0,n-1),
        state_reset_count=1+int(resets[n-1]-resets[0]) if n else 0,simulation_time_s=native_time[:n],
        physics_timestep_s=.005,policy_timestep_s=.020)
    return arrays,prefix,dict(initial_reset_counter=int(resets[0]),final_reset_counter=int(resets[n-1]) if n else None)


def event_overlay(row,episode_time_s,*,partial,actual_launch_s=None):
    planned_launch=row['launch_tick']*.005;outward=row['outward_end_tick']*.005
    prep=row.get('preparation_complete_tick',0)*.005
    if episode_time_s<prep:phase='PREPARATION'
    elif actual_launch_s is None:phase='READY HOLD' if episode_time_s<planned_launch else 'NO COMMITTED LAUNCH'
    elif episode_time_s<actual_launch_s:phase='READY HOLD'
    elif episode_time_s<outward:phase='OUTWARD'
    elif episode_time_s<row.get('guard_hold_tick',round(row.get('guard_time_s',row['duration_s'])/.005))*.005:phase='RECOVERY'
    else:phase='GUARD HOLD'
    target=row['target_w'];clock=episode_time_s-(actual_launch_s if actual_launch_s is not None else planned_launch)
    clock_name='launch dt' if actual_launch_s is not None else 'planned launch dt'
    return (f'{phase} | request TTC={row["requested_ttc_s"]:.3f}s | t={episode_time_s:.3f}s '
        f'{clock_name}={clock:+.3f}s | target=({target[0]:.3f},{target[1]:.3f},{target[2]:.3f})m'
        +(' | PARTIAL TRACE' if partial else ''))


def annotate_video(base,output,arrays,row,*,partial,deadline,product_check,actual_launch_s=None):
    """Add phase/launch-relative clocks to saved rendered frames; no physics."""
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    from human_punching.render import font
    from human_punching.target_render import _playback_indices
    indices=_playback_indices(np.asarray(arrays['time_s']),1.,50);frames=0;small=font(16)
    with imageio.get_reader(base) as reader,imageio.get_writer(output,fps=50,codec='libx264',quality=8,
            ffmpeg_params=['-movflags','+faststart'],macro_block_size=16) as writer:
        for index,frame in enumerate(reader):
            deadline.checkpoint('expanded_video_annotation_frame',progress=dict(frame=index))
            if index>=len(indices):raise ValueError('Renderer emitted more frames than exact1x50fps schedule')
            canvas=Image.fromarray(frame);draw=ImageDraw.Draw(canvas);draw.rectangle((0,35,canvas.width,63),fill=(15,22,32))
            draw.text((12,40),event_overlay(row,float(arrays['time_s'][indices[index]]),partial=partial,actual_launch_s=actual_launch_s),font=small,
                fill=(255,180,130) if partial else (210,230,240))
            bottom=max(0,canvas.height-22);draw.rectangle((0,bottom,canvas.width,canvas.height),fill=(65,12,12) if partial else (15,22,32))
            badge='PARTIAL RECORDED TRACE — REMAINDER UNAVAILABLE' if partial else 'FULL RETAINED TRACE — OUTCOME REVIEW REQUIRED'
            if actual_launch_s is None:badge+=' | NO COMMITTED LAUNCH'
            draw.text((12,bottom+2),badge,font=small,fill=(255,210,160))
            writer.append_data(np.asarray(canvas));frames+=1
            if index%50==0:product_check()
    if frames!=len(indices):raise ValueError('Rendered video truncated before full retained trace schedule')
    with imageio.get_reader(output) as reader:decoded=sum(1 for _ in reader)
    if decoded!=frames:raise ValueError('Annotated video full decode differs from written count')
    return dict(video=str(output),video_frames=frames,decoded_frames=decoded,fps=50,playback_speeds=[1.],
        source_sample_indices=indices.tolist(),integration=False,inference=False)


def render(run,output,*,wall_seconds=840.,product_byte_cap=2*1024**3):
    if type(product_byte_cap) is not int or not 0<product_byte_cap<=2*1024**3:raise ValueError('Finite<=2GiB media product cap required')
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Expanded/shared media requires a counted allocated GPU')
    started=time.monotonic();deadline=ExecutionDeadline(wall_seconds,serialization_reserve_s=min(60.,wall_seconds/4),session_started_monotonic=started)
    run=Path(run).resolve(strict=True);run.relative_to(ROOT/'outputs/task10b')
    declaration_path=run/'declaration.json'
    if not declaration_path.exists():declaration_path=run/'physical_declaration.json'
    declaration=load_declaration(owned(declaration_path,parent=run))
    outcomes_path=run/'outcomes.json';outcome_source='final_outcomes'
    if outcomes_path.exists():outcomes=json.loads(owned(outcomes_path,parent=run).read_text())
    else:
        ledgers=list(run.glob('ledger*.json'))
        if ledgers:
            outcomes_path=max(ledgers,key=lambda path:(path.stat().st_mtime_ns,path.name))
            outcomes=json.loads(owned(outcomes_path,parent=run).read_text());outcome_source='preserved_partial_ledger'
        elif (run/'summary.json').exists():
            outcomes_path=run/'summary.json';outcomes=json.loads(owned(outcomes_path,parent=run).read_text()).get('trials',[])
            outcome_source='physical_validation_summary'
        else:outcomes_path=None;outcomes=[];outcome_source='no_outcome_ledger_all_missing'
    launch_path=owned(run/'launch.json',parent=run);launch=json.loads(launch_path.read_text())
    if launch.get('declaration_identity',launch.get('physical_request_template_identity'))!=declaration.identity:
        raise ValueError('Media declaration differs from bound native launch')
    catalog_path=owned(launch['catalog']);catalog=json.loads(catalog_path.read_text())
    if sha256_file(catalog_path)!=launch['catalog_sha256']:raise ValueError('Native catalog changed')
    cells={row['candidate_id']:row for row in catalog['cells']}
    if len(cells)!=len(catalog['cells']):raise ValueError('Duplicate media catalog candidate')
    runs=source_runs(run);index={}
    for source_run in runs:
        for trial,binding in index_batches(source_run,declaration,cells).items():
            if trial in index:raise ValueError('An attempted trial appears in multiple continuation runs')
            index[trial]=binding
    planned=selection(declaration,outcomes)
    model=owned(run/'runtime.xml',parent=run) if (run/'runtime.xml').exists() else None
    out=writable_path(output);out.mkdir(parents=True,exist_ok=False)
    from human_punching.wbc.egl_binding import configure_allocated_egl
    egl=configure_allocated_egl()
    from human_punching.jab_evaluation import render_rollout
    write_json(out/'plan.json',dict(schema='task10b_phase2_expanded_media_plan_v1',selection=planned,
        declaration=asdict(declaration),declaration_identity=declaration.identity,predeclared_video_ids=list(declaration.video_trial_ids),
        run=str(run),source_inputs={str(path):sha256_file(path) for path in (declaration_path,outcomes_path,launch_path,catalog_path,model) if path is not None},
        outcome_source=outcome_source,runtime_model_available=model is not None,
        continuation_source_runs=[str(path) for path in runs],
        checkpoint_sha256=launch.get('evaluated_checkpoint',{}).get('sha256',declaration.checkpoint_sha256),
        declaration_checkpoint_template_is_unbound=launch.get('checkpoint_template_is_unbound',False),predicate_sha256=declaration.predicate_sha256,
        source_sha256=sha256_file(__file__),fps=50,playback_speeds=[1.],integration=False,inference=False,
        g1_scene_or_actor=False,egl_binding=egl,slurm_job_id=os.environ['SLURM_JOB_ID'],wall_seconds=wall_seconds,
        product_byte_cap=product_byte_cap,video_selection_frozen_before_outcomes=True))
    def product_check():
        total=sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
        if total>product_byte_cap:raise RuntimeError('Expanded media product byte cap exceeded; preserve partial products')
        return total
    reports=[];stop=None;largest_render=0.;largest_products=128*1024**2
    with deadline.signal_handlers():
        for chosen in planned:
            record=dict(chosen);item_started=time.monotonic()
            if stop or deadline.work_deadline-time.monotonic()<=largest_render:
                stop=stop or 'media_wall_reservation';record.update(status='unattempted',reason=stop);reports.append(record);continue
            binding=index.get(chosen['trial_id'])
            item_model=Path(binding['runtime_model_path']) if binding is not None else None
            if binding is None or not Path(binding['raw_trace_path']).exists() or not item_model.exists():
                record.update(status='unavailable',reason='No retained executed physical trace/runtime model for this exact declared trial')
                reports.append(record);continue
            try:
                before_bytes=product_check()
                if before_bytes+largest_products>product_byte_cap:
                    stop='media_product_reservation';record.update(status='unattempted',reason=stop);reports.append(record);continue
                deadline.checkpoint('before_expanded_media_trial',progress=dict(trial_id=chosen['trial_id']))
                folder=writable_path(out/f'trial_{len(reports):04d}');folder.mkdir()
                write_json(folder/'attempt_started.json',dict(selection=chosen,batch=binding))
                arrays,prefix,counters=extract_world(binding['raw_trace_path'],binding['world'],
                    planned_samples=binding['planned_samples'],planned_worlds=binding['planned_worlds'])
                record.update(raw_trace_path=binding['raw_trace_path'],raw_trace_sha256=sha256_file(binding['raw_trace_path']),
                    native_plan_sha256=binding['plan_sha256'],world=binding['world'],prefix=prefix,reset_counters=counters)
                item_model=owned(item_model,parent=Path(binding['plan_path']).parent.parent)
                record.update(runtime_model_path=str(item_model),runtime_model_sha256=sha256_file(item_model))
                if len(arrays['qpos'])<2:
                    record.update(status='unrenderable',reason='Fewer than2 finite valid quaternion states in retained prefix')
                else:
                    partial=prefix['truncated_at_native_invalid_state'] or prefix['incomplete_native_horizon']
                    label=f'{declaration.family} | {chosen["trial_id"]} | {chosen["disposition"]}'
                    if partial:label+=' | PARTIAL TRACE'
                    base=render_rollout(item_model,arrays,folder/'base',head_center=binding['row']['target_w'],label=label,
                        fps=50,playback_speeds=(1.,),artifact_prefix=declaration.family)
                    product_check();deadline.checkpoint('after_expanded_base_render')
                    media=annotate_video(base['video'],folder/'phase-overlay.mp4',arrays,binding['row'],
                        partial=partial,deadline=deadline,product_check=product_check,actual_launch_s=binding['actual_launch_s'])
                    record.update(status='rendered',media=media,partial_failed_trace=partial,
                        full_trace_completed=not partial,base_media=base,artifacts={str(path.relative_to(out)):sha256_file(path)
                            for path in folder.rglob('*') if path.is_file()},visual_review='pending',
                        actual_launch_s=binding['actual_launch_s'],launch_disposition=binding['launch_disposition'],
                        actual_actor_tensor_digest=binding['actual_actor_tensor_digest'],
                        launch_commitment_sha256=binding['launch_commitment_sha256'],
                        full_preparation_outward_recovery_shown=not partial and binding['actual_launch_s'] is not None)
                    largest_products=max(largest_products,sum(path.stat().st_size for path in folder.rglob('*') if path.is_file()))
            except Exception as error:
                record.update(status='failed',exception=dict(type=type(error).__name__,message=str(error)))
                stop='renderer_input_or_resource_failure'
            record['wall_s']=time.monotonic()-item_started;largest_render=max(largest_render,record['wall_s'])
            reports.append(record);write_json(out/f'progress_{len(reports):04d}.json',dict(results=reports,unattempted=planned[len(reports):]))
    final_bytes=sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
    result=dict(schema='task10b_phase2_expanded_media_v1',results=reports,planned=len(planned),
        **{key:sum(row['status']==key for row in reports) for key in ('rendered','failed','unavailable','unrenderable','unattempted')},
        stop_reason=stop,wall_s=time.monotonic()-started,product_bytes=final_bytes,product_byte_cap_exceeded=final_bytes>product_byte_cap,
        simulation_steps=0,actor_calls=0,
        visual_recognition_accepted=False,independent_review_required=True,phase2_complete=False)
    write_json(out/'summary.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=float,default=840.);parser.add_argument('--product-byte-cap',type=int,default=2*1024**3)
    args=parser.parse_args();result=render(args.run,args.output,wall_seconds=args.wall_seconds,product_byte_cap=args.product_byte_cap)
    print(json.dumps(result,allow_nan=False,sort_keys=True));return 0 if not result['failed'] and not result['unattempted'] else 1


if __name__=='__main__':raise SystemExit(main())
