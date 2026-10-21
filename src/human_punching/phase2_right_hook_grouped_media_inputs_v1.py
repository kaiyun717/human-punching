"""Exact saved grouped-panel media closure; original renderer remains unchanged."""
from __future__ import annotations
import importlib.util,json,math
from pathlib import Path
from .phase2_right_hook_grouped_saved_v1 import ROOT,UPDATES,context,sha,read

VERSION='task10b_phase2_right_hook_grouped_saved_media_inputs_v1'


def media_limits(planned):
    if type(planned) is not int or not 1 <= planned <= 96:
        raise ValueError('One to 96 exact selected or flagged requests required')
    return (300,240) if planned <= 32 else (600,540)


def renderer_module():
    spec=importlib.util.spec_from_file_location('task10b_grouped_unchanged_renderer',ROOT/'scripts/task10b_phase2_expanded_media.py')
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer);return renderer


def validate_parent(packet,execution,terminal,launch,packet_sha):
    seconds=terminal.get('allocated_gpu_seconds')
    if (packet.get('schema')!='task10b_phase2_right_hook_grouped_packet_v1' or packet.get('mode')!='grouped-pilot'
            or packet.get('training') is not True or packet.get('gpus')!=1
            or packet.get('fresh_transition_cap')!=65536
            or packet.get('family')!='right_hook' or packet.get('allocation_wall_seconds')!=960
            or packet.get('application_wall_seconds')!=900 or packet.get('planned_validation_updates')!=[0,8]
            or execution.get('packet_sha256')!=packet_sha or execution.get('native_processes_started')!=1
            or terminal.get('packet_sha256')!=packet_sha or terminal.get('state') not in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','PREEMPTED')
            or not isinstance(seconds,(int,float)) or isinstance(seconds,bool) or not math.isfinite(seconds) or not 0<seconds<=960
            or execution.get('job_id')!=terminal.get('job_id') or execution.get('job_id')!=launch.get('slurm_job_id')):
        raise ValueError('Exact terminal counted grouped native parent required; failures stay renderable')


def outcome_source(run,ctx):
    """Mirror the unchanged renderer, including interrupted/no-ledger panels."""
    run=Path(run)
    if (run/'outcomes.json').exists():path=run/'outcomes.json'
    else:
        ledgers=list(run.glob('ledger*.json'))
        path=(max(ledgers,key=lambda p:(p.stat().st_mtime_ns,p.name)) if ledgers else
            run/'summary.json' if (run/'summary.json').exists() else None)
    if path is None:return None,[]
    data=ctx.read(path)
    return path,data if isinstance(data,list) else data['trials']


def selected(run,namespace_proof):
    run=Path(run).resolve(strict=True)
    if not run.is_relative_to(ROOT/'outputs/task10b') or run.name not in {f'validation_{u:05d}' for u in UPDATES}:
        raise ValueError('One exact grouped validation cadence required')
    update=int(run.name.removeprefix('validation_'))
    if run.parent.name!='held_out_validation' or run.parent.parent.name!='run':
        raise ValueError('Exact grouped saved hierarchy required')
    ctx,packet_path,packet,configuration,declaration,_=context(run.parents[2]/'packet.json',namespace_proof)
    ctx.add(packet_path)
    if run.parent.parent!=ROOT/packet['output']:raise ValueError('Grouped media run differs from packet')
    launch=ctx.read(run/'launch.json')
    if (launch.get('schema')!='task10b_phase2_grouped_validation_launch_v1'
            or launch.get('checkpoint_template_is_unbound') is not True
            or launch.get('actual_training_update')!=update
            or launch.get('actual_completed_fresh_transitions')!=update*128*64
            or launch.get('family','right_hook')!='right_hook'):
        raise ValueError('Exact actual grouped callback/checkpoint clock required')
    validate_parent(packet,ctx.read(packet_path.parent/'execution.json'),ctx.read(packet_path.parent/'terminal_accounting.json'),launch,sha(packet_path))
    vconfig=ctx.read(ctx.bound(configuration['validation_configuration']))
    source=ctx.bound(vconfig['declaration_template']);physical=ctx.add(run/'physical_declaration.json')
    if ctx.read(source)!=ctx.read(physical):raise ValueError('Saved physical requests changed from predeclaration')
    if launch['configuration']['sha256']!=configuration['validation_configuration']['sha256']:
        raise ValueError('Saved callback configuration differs')
    ctx.bound(launch['configuration']);ctx.bound(launch['physical_declaration']);ctx.bound(launch['evaluated_checkpoint'])
    catalog_path=ctx.bound(dict(path=launch['catalog'],sha256=launch['catalog_sha256']));catalog=ctx.read(catalog_path)
    manifest=ctx.read(ctx.bound(vconfig['manifest']));expected=ctx.read(ctx.bound(dict(path=manifest['catalog_path'],sha256=manifest['catalog_sha256'])))
    if catalog!=expected:raise ValueError('Saved copied runtime reference catalog changed')
    # The unchanged renderer gives preserved ledgers precedence over summary.
    outcome_path,outcomes=outcome_source(run,ctx)
    renderer=renderer_module();decl=renderer.load_declaration(physical)
    chosen=renderer.selection(decl,outcomes)
    index=renderer.index_batches(run,decl,{r['candidate_id']:r for r in catalog['cells']})
    # Missing/partial requested videos remain selected and are reported by the
    # original renderer as unavailable; never silently reduce the union.
    return ctx,dict(update=update,family=decl.family,planned=len(chosen),selection=chosen,
        predeclared_video_ids=list(decl.video_trial_ids),outcome_source=None if outcome_path is None else str(outcome_path),
        indexed_trial_ids=sorted(index),checkpoint=launch['evaluated_checkpoint'],source_packet_sha256=sha(packet_path),
        physical_qualification=False,actor_calls=0,physics_steps=0)


def files(run,namespace_proof):
    run=Path(run).resolve(strict=True);ctx,selection=selected(run,namespace_proof)
    total=0
    for path in sorted(run.rglob('*')):
        if path.is_file():
            if not path.resolve(strict=True).is_relative_to(run):raise ValueError('Saved grouped evidence escaped selected panel')
            total+=path.stat().st_size
            if total>1024**3:raise ValueError('One saved grouped panel exceeds1GiB input bound')
            ctx.add(path)
    # Only actual renderer dependencies and exact parent protocol are closed.
    # Historical domain/body metadata stays opaque in the preserved packet.
    return dict(sorted(ctx.files.items())),selection
