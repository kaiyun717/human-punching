"""Saved-only rendering of the exact four-cell target control and all failures."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

from . import phase2_target_control_v1 as control
from .wbc.provenance import PROJECT_ROOT as ROOT, sha256_file, writable_path, write_json

VERSION='task10b_phase2_target_control_saved_media_v1'


def validate_parent(packet,outer,terminal,launch,packet_sha):
    job=launch.get('slurm_job_id')
    if (packet.get('schema')!='task10b_phase2_four_snapshot_frozen_control_packet_v1'
            or packet.get('mode')!='target-control' or packet.get('training') is not False
            or packet.get('gpus')!=1 or packet.get('fresh_transition_cap')!=0
            or terminal.get('state')!='COMPLETED' or terminal.get('exit_code')!='0:0'
            or not 0<terminal.get('allocated_gpu_seconds',0)<=360
            or terminal.get('packet_sha256')!=packet_sha or terminal.get('job_id')!=job
            or outer.get('status')!='executed' or outer.get('returncode')!=0
            or outer.get('native_processes_started')!=1 or outer.get('budget_stop') is not None
            or outer.get('packet_sha256')!=packet_sha or outer.get('job_id')!=job):
        raise ValueError('Exact successfully executed frozen-control parent allocation required')


def selected(run):
    run=Path(run).resolve(strict=True)
    if not run.is_relative_to(ROOT/'outputs/task10b'):raise ValueError('Owned saved target control required')
    launch=json.loads((run/'launch.json').read_text())
    if (launch.get('schema')!='task10b_phase2_original_route_launch_v1'
            or launch.get('target_control_version')!=control.VERSION
            or launch.get('mode')!='frozen' or launch.get('training') is not False
            or launch.get('planned_cells')!=control.IDS or launch.get('total_planned_trials')!=32):
        raise ValueError('Exact frozen four-cell target-control launch required')
    # Inputs resolves only the already-proved logical/compute aliases, with
    # inode and byte identity; never rewrite saved paths by prefix alone.
    ctx=control.Inputs('outputs/task10b/p2hostalias01/probe.json')
    planpath=ctx.bound(launch['control_plan']);plan=ctx.read(planpath)
    control.validate_plan(plan)
    if ctx.read(run/'control_plan.json')!=plan:raise ValueError('Saved plan differs')
    terminal=ctx.read(run.parent/'terminal_accounting.json')
    packet=ctx.add(run.parent/'packet.json')
    validate_parent(ctx.read(packet),ctx.read(run.parent/'execution.json'),terminal,launch,sha256_file(packet))
    execution=ctx.read(run/'execution.json')
    if (execution.get('completed_trials')!=32 or execution.get('fresh_transitions')!=0
            or execution.get('frozen_actor_unchanged') is not True):
        raise ValueError('All32 saved frozen control trials required')
    panels=[]
    for index,candidate in enumerate(control.IDS):
        path=run/f'validation_{index:05d}/summary.json';panel=ctx.read(path)
        expected=[f'{candidate}_bank_{i:02d}' for i in range(8)]
        if (panel.get('planned_trials')!=8 or panel.get('executed_trials')!=8
                or [r['trial_id'] for r in panel.get('trials',[])]!=expected
                or any(r.get('candidate_id')!=candidate for r in panel['trials'])):
            raise ValueError('Exact ordered candidate/bank ledger required')
        panels.append(path)
    return ctx,planpath,plan,panels


def files(run):
    run=Path(run).resolve(strict=True);ctx,planpath,plan,panels=selected(run)
    for name,digest in control.files(planpath).items():ctx.add(name,digest)
    packet=ctx.read(run.parent/'packet.json')
    for name,digest in packet['files'].items():
        ctx.add(name,digest);ctx.add(run.parent/'source_snapshot'/name,digest)
    for name in ('packet.json','submission.json','terminal_accounting.json','execution.json'):
        ctx.add(run.parent/name)
    total=0
    for path in sorted(run.rglob('*')):
        if path.is_file():
            if not path.resolve(strict=True).is_relative_to(run):raise ValueError('Saved evidence escaped run')
            total+=path.stat().st_size
            if total>3*1024**3:raise ValueError('Saved target control exceeds3GiB bound')
            ctx.add(path)
    return dict(sorted(ctx.files.items()))


def render(run,output,wall_seconds):
    if not os.environ.get('SLURM_JOB_ID') or not 120<float(wall_seconds)<=210:
        raise ValueError('Counted saved-render allocation and finite121–210s child required')
    started=time.monotonic();run=Path(run).resolve(strict=True)
    ctx,planpath,plan,panels=selected(run)
    out=writable_path(output);out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('target_control_saved_renderer',ROOT/'scripts/task10b_phase2_media.py')
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    selections=[renderer.selection(ctx.read(path)) for path in panels]
    expected=sorted(row['trial_id'] for panel in selections for row in panel)
    if expected!=ctx.read(run/'media_selection.json')['planned_required_trial_ids']:
        raise ValueError('Saved predeclared/flagged media union differs')
    write_json(out/'plan.json',dict(schema=VERSION,source_run=str(run),source_plan_sha256=sha256_file(planpath),
        selected_trial_ids=expected,planned=len(expected),panels=[str(p) for p in panels],
        wall_seconds=wall_seconds,physics_steps=0,actor_calls=0))
    results=[];error=None
    try:
        for index,path in enumerate(panels):
            remaining=wall_seconds-(time.monotonic()-started)
            if remaining<=60:
                results.extend(dict(panel=i,status='unattempted',reason='media_wall_budget',planned=len(selections[i])) for i in range(index,4))
                break
            value=renderer.render(run,path,out/f'panel_{index:02d}',wall_seconds=remaining,shutdown_reserve_seconds=30.)
            results.append(dict(panel=index,status='rendered' if value['rendered']==value['planned'] else 'incomplete',
                planned=value['planned'],rendered=value['rendered'],summary_sha256=sha256_file(out/f'panel_{index:02d}/summary.json')))
            write_json(out/f'progress_{index:02d}.json',dict(results=results))
            if results[-1]['status']!='rendered':break
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
        if index not in {r['panel'] for r in results}:
            results.append(dict(panel=index,status='failed_extent_unknown',planned=len(selections[index]),
                reason='renderer_exception',exception=error,retained_output=str(out/f'panel_{index:02d}')))
    attempted={r['panel'] for r in results}
    results.extend(dict(panel=i,status='unattempted',reason='prior_panel_failure',planned=len(selections[i])) for i in range(4) if i not in attempted)
    result=dict(schema=VERSION,status='rendered' if not error and all(r['status']=='rendered' for r in results) else 'incomplete',
        planned=len(expected),rendered=sum(r.get('rendered',0) for r in results),panels=results,exception=error,
        physics_steps=0,actor_calls=0,wall_seconds=time.monotonic()-started,manual_review_complete=False,physical_qualification=False)
    write_json(out/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--wall-seconds',type=float,required=True);a=p.parse_args()
    result=render(a.run,a.output,a.wall_seconds);print(json.dumps(result));return 0 if result['status']=='rendered' else 1


if __name__=='__main__':raise SystemExit(main())
