"""Saved grouped-render receipt and full video-decode audit; no scene or actor."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from .phase2_nonuniform_control import Inputs
from .phase2_grouped_media_inputs_v1 import selected,renderer_module
from .wbc.provenance import PROJECT_ROOT,sha256_file,writable_path

VERSION='task10b_phase2_grouped_saved_media_audit_v1'


def selection_checks(expected,plan,summary):
    wanted=expected['selection'];ids=[r['trial_id'] for r in wanted]
    results=summary.get('results',[]);actual=[r.get('trial_id') for r in results]
    statuses=('rendered','failed','unavailable','unrenderable','unattempted')
    checks=dict(exact_plan_selection=plan.get('selection')==wanted,
        exact_presampled_ids=plan.get('predeclared_video_ids')==expected['predeclared_video_ids'],
        every_selected_trial_present_once=actual==ids and len(actual)==len(set(actual)),
        exact_planned_count=summary.get('planned')==len(ids),
        declared_no_physics=summary.get('simulation_steps')==0 and summary.get('actor_calls')==0,
        renderer_no_physics=plan.get('integration') is False and plan.get('inference') is False,
        checkpoint_exact=plan.get('checkpoint_sha256')==expected['checkpoint']['sha256'],
        template_correctly_unbound=plan.get('declaration_checkpoint_template_is_unbound') is True,
        exact_status_counters=all(summary.get(s)==sum(r.get('status')==s for r in results) for s in statuses),
        all_statuses_known=all(r.get('status') in statuses for r in results))
    for wanted_row,result in zip(wanted,results):
        if any(result.get(k)!=v for k,v in wanted_row.items()):checks['selected_request_fields_unchanged']=False
    checks.setdefault('selected_request_fields_unchanged',True)
    return checks


def decode_video(path):
    import imageio.v2 as imageio
    count=0;shape=None
    with imageio.get_reader(path) as reader:
        fps=float(reader.get_meta_data()['fps'])
        for frame in reader:
            if frame.ndim!=3 or frame.shape[2] not in (3,4):raise ValueError('Decoded RGB video frames required')
            if shape is None:shape=list(frame.shape)
            if list(frame.shape)!=shape:raise ValueError('Video dimensions changed during decode')
            count+=1
    if count==0 or not math.isfinite(fps):raise ValueError('Nonempty finite frame schedule required')
    return dict(decoded_frames=count,fps=fps,shape=shape)


def audit(packet_path,output):
    path=Path(packet_path).resolve(strict=True);packet=json.loads(path.read_text())
    proof=packet['runtime_namespace_proof'];ctx=Inputs(proof['path']);ctx.add(path)
    if (packet.get('schema')!='task10b_phase2_grouped_media_packet_v1' or packet.get('mode')!='media'
            or set(packet.get('inputs',{}))!={'run'} or packet.get('training') is not False
            or packet.get('actor_calls')!=0 or packet.get('physics_steps')!=0
            or packet.get('allocation_wall_seconds')!=300 or packet.get('application_wall_seconds')!=240):
        raise ValueError('Exact grouped saved-media300/240 packet required')
    for name,digest in packet['files'].items():
        ctx.add(name,digest);ctx.add(path.parent/'source_snapshot'/name,digest)
    execution=ctx.read(path.parent/'execution.json');terminal=ctx.read(path.parent/'terminal_accounting.json')
    if (execution['packet_sha256']!=sha256_file(path) or terminal['packet_sha256']!=sha256_file(path)
            or execution['job_id']!=terminal['job_id'] or execution.get('native_processes_started')!=1
            or terminal.get('state') not in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','PREEMPTED')
            or not 0<terminal.get('allocated_gpu_seconds',0)<=300):
        raise ValueError('Exact counted terminal media job required')
    run=ctx.owned(packet['inputs']['run']);_,expected=selected(run,proof['path'])
    out=ctx.owned(packet['output']);plan=ctx.read(out/'plan.json');summary=ctx.read(out/'summary.json')
    if (plan.get('schema')!='task10b_phase2_expanded_media_plan_v1' or summary.get('schema')!='task10b_phase2_expanded_media_v1'
            or plan.get('slurm_job_id')!=execution['job_id'] or plan.get('source_sha256')!=packet['files']['scripts/task10b_phase2_expanded_media.py']):
        raise ValueError('Original expanded renderer and exact counted job required')
    if ctx.owned(plan['run'])!=run:raise ValueError('Rendered parent run differs')
    for name,digest in plan['source_inputs'].items():ctx.add(name,digest)
    checks=selection_checks(expected,plan,summary);videos=[]
    renderer=renderer_module();declaration=renderer.load_declaration(run/'physical_declaration.json')
    catalog=ctx.read(run/'catalog.json');index=renderer.index_batches(run,declaration,{r['candidate_id']:r for r in catalog['cells']})
    for row in summary.get('results',[]):
        if row.get('status')!='rendered':continue
        source=index[row['trial_id']];media=row['media'];video=ctx.owned(media['video'])
        if not video.is_relative_to(out):raise ValueError('Rendered video escaped its output packet')
        for name,digest in row['artifacts'].items():
            p=ctx.owned(out/name)
            if not p.is_relative_to(out):raise ValueError('Rendered artifact escaped output')
            ctx.add(p,digest)
        if str(video.relative_to(out)) not in row['artifacts']:raise ValueError('Primary video absent from sealed artifacts')
        raw=ctx.add(source['raw_trace_path'],row['raw_trace_sha256'])
        model=ctx.add(row['runtime_model_path'],row['runtime_model_sha256'])
        arrays,prefix,counters=renderer.extract_world(raw,source['world'],planned_samples=source['planned_samples'],planned_worlds=source['planned_worlds'])
        from .target_render import _playback_indices
        indices=_playback_indices(arrays['time_s'],1.,50).tolist();decoded=decode_video(video)
        partial=prefix['truncated_at_native_invalid_state'] or prefix['incomplete_native_horizon']
        local_checks=dict(source_plan=row['native_plan_sha256']==source['plan_sha256'],world=row['world']==source['world'],
            runtime_model=model==ctx.owned(source['runtime_model_path']),
            prefix=row['prefix']==prefix,reset_counters=row['reset_counters']==counters,
            actual_launch=row.get('actual_launch_s')==source['actual_launch_s'],
            actor_digest=row.get('actual_actor_tensor_digest')==source['actual_actor_tensor_digest'],
            commitment=row.get('launch_commitment_sha256')==source['launch_commitment_sha256'],
            exact_frame_schedule=media.get('source_sample_indices')==indices,
            actual_decode=decoded['decoded_frames']==media.get('video_frames')==media.get('decoded_frames')==len(indices),
            fps=decoded['fps']==media.get('fps')==50,partial_flag=row['partial_failed_trace']==partial,
            full_trace_flag=row['full_trace_completed']==(not partial),
            full_cycle_label=row['full_preparation_outward_recovery_shown']==(not partial and source['actual_launch_s'] is not None),
            no_integration_or_inference=media.get('integration') is False and media.get('inference') is False)
        videos.append(dict(trial_id=row['trial_id'],video=ctx.binding(video),decoded=decoded,checks=local_checks,passed=all(local_checks.values())))
    passed=all(checks.values()) and all(r['passed'] for r in videos)
    complete=passed and len(videos)==expected['planned'] and summary.get('stop_reason') is None
    result=dict(schema=VERSION,passed_saved_media_integrity=passed,complete_selected_media=complete,
        parent_packet=ctx.binding(path),source_update=expected['update'],planned=expected['planned'],rendered=len(videos),checks=checks,
        videos=videos,all_dispositions=summary['results'],original_summary=ctx.binding(out/'summary.json'),
        original_plan=ctx.binding(out/'plan.json'),terminal_accounting=ctx.binding(path.parent/'terminal_accounting.json'),
        source=dict(path=str(Path(__file__).resolve()),sha256=sha256_file(__file__)),
        manual_visual_review_performed=False,visual_recognition_accepted=False,physical_qualification=False,
        simulation_steps=0,actor_calls=0,original_artifacts_changed=False)
    destination=writable_path(output);destination.mkdir(parents=True,exist_ok=False)
    with (destination/'audit.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv);r=audit(a.packet,a.output);print(json.dumps({k:r[k] for k in ('passed_saved_media_integrity','complete_selected_media','planned','rendered')}))
    return 0 if r['passed_saved_media_integrity'] else 1

if __name__=='__main__':raise SystemExit(main())
