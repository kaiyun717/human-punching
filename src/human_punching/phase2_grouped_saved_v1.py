"""Independent complete cadence ledgers for the exact first grouped pilot.

No simulator/model/actor imports. Physical audit dispatch is separate. A failed
constructor or watchdog can leave no raw trace; the active batch then has unknown
execution extent, even if the original callback ledger called it unattempted.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from .phase2_nonuniform_control import Inputs

ROOT=Path(__file__).resolve().parents[2]
VERSION='task10b_phase2_grouped_saved_ledger_v1'
UPDATES=(0,8,16,24,32)

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def binding(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def expected_batches(declaration,manifest):
    entries={r['trial_id']:r for r in manifest['trials']};groups={}
    if (len(entries)!=96 or len(declaration['bindings'])!=96
            or {b['trial_id'] for b in declaration['bindings']}!=set(entries)):
        raise ValueError('Exact96 unique requests required')
    for row in declaration['bindings']:
        item=entries[row['trial_id']]
        groups.setdefault(item['candidate_id'],[]).append(row)
    if len(groups)!=24 or any(len(rows)!=4 for rows in groups.values()):
        raise ValueError('Exact24 reference batches with four starts required')
    return [(candidate,rows) for candidate,rows in groups.items()]


def panel_ledger(panel,batches,*,pending=False):
    panel=Path(panel);exists=panel.exists();summary=read(panel/'summary.json') if (panel/'summary.json').exists() else None
    ledger_indices=[int(p.stem.removeprefix('ledger_')) for p in panel.glob('ledger_[0-9][0-9][0-9][0-9].json')]
    existing_batches=[int(p.name.removeprefix('batch_')) for p in panel.glob('batch_[0-9][0-9][0-9][0-9]') if p.is_dir()]
    # Completed callback writes ledger_N even when constructing batch N throws.
    # A watchdog after ledger_N can interrupt construction of N+1 before its
    # output directory exists. Never infer zero execution from absent raw data.
    active=(max(ledger_indices,default=-1) if summary is not None else
            max(max(ledger_indices,default=-1)+1,max(existing_batches,default=0))) if exists else None
    if summary is not None and summary.get('status')=='executed':active=None
    rows=[];panels=[]
    for index,(candidate,bindings) in enumerate(batches):
        folder=panel/f'batch_{index:04d}';rawpath=folder/'raw_trace.npz';planpath=folder/'plan.json';resultpath=folder/'summary.json'
        plan=read(planpath) if planpath.exists() else None;result=read(resultpath) if resultpath.exists() else None
        reports={r['trial_id']:r for r in (result or {}).get('trials',[])}
        raw=None
        if rawpath.exists():
            with np.load(rawpath,allow_pickle=False) as a:
                intervals=len(a['pre_qpos']) if 'pre_qpos' in a.files else None
                states=len(a['qpos']) if 'qpos' in a.files else None
                actors=len(a['policy_episode_tick']) if 'policy_episode_tick' in a.files else None
            raw=dict(path=str(rawpath),sha256=sha(rawpath),physics_steps=intervals,state_samples=states,actor_updates=actors)
        full=bool(raw is not None and plan is not None and raw['physics_steps']==plan['planned_physics_steps']
            and raw['state_samples']==raw['physics_steps']+1 and raw['actor_updates']*4==raw['physics_steps'])
        panels.append(dict(batch=index,candidate_id=candidate,raw=raw,complete_saved_counts=full,
            recorded_status=None if result is None else result.get('status')))
        for b in bindings:
            trial=b['trial_id'];reported=reports.get(trial)
            status=('pending_job' if pending else
                'completed_native_attempt_pending_independent_audit' if full and reported is not None else
                'retained_raw_execution_extent_known' if raw is not None else
                'execution_extent_unknown' if exists and (index==active or folder.exists() or (active is not None and index<active)) else
                'unattempted_callback_not_entered' if not exists else 'unattempted_after_callback_stop')
            rows.append(dict(**b,batch=index,candidate_id=candidate,status=status,
                recorded_numeric_complete_cycle=None if reported is None else reported.get('numeric_complete_cycle'),
                raw_trace_sha256=None if raw is None else raw['sha256'],
                execution_extent_unknown=status=='execution_extent_unknown',
                independent_physical_qualification=False))
    return dict(trials=rows,panels=panels,recorded_callback_status=None if summary is None else summary.get('status'),
        active_batch_inferred_from_progress=active)


def context(packet_path,namespace_proof):
    ctx=Inputs(namespace_proof,root=ROOT)
    packet_path=ctx.owned(packet_path);packet=read(packet_path)
    if (packet.get('schema')!='task10b_phase2_grouped_clock_packet_v1' or packet.get('mode')!='grouped-pilot'
            or packet.get('allocation_wall_seconds')!=1800 or packet.get('application_wall_seconds')!=1680
            or packet.get('successful_route_freeze') is not None):
        raise ValueError('Exact initial-budget clock-adapted grouped packet required')
    for name,digest in packet['files'].items():
        ctx.bound(dict(path=name,sha256=digest));ctx.bound(dict(path=packet_path.parent/'source_snapshot'/name,sha256=digest))
    for name,item in packet.get('provenance_files',{}).items():ctx.bound(dict(path=name,sha256=item['sha256']))
    configpath=ctx.owned(packet['inputs']['configuration']);relative=str(configpath.relative_to(ROOT))
    ctx.bound(dict(path=configpath,sha256=packet['files'][relative]));configuration=read(configpath)
    budget=configuration['budget']
    if (budget['environments']!=128 or budget['rollout_steps']!=64 or budget['maximum_updates']!=32
            or configuration['stopping_rules']['validation_updates']!=list(UPDATES)):
        raise ValueError('Exact predeclared128x64x32 and five validation cadences required')
    vconfig=read(ctx.bound(configuration['validation_configuration']))
    declaration=read(ctx.bound(vconfig['declaration_template']));manifest=read(ctx.bound(vconfig['manifest']))
    if declaration['split']!='validation' or declaration['family']!='jab' or declaration['seed']!=21021:
        raise ValueError('Declared jab validation seed/split required')
    batches=expected_batches(declaration,manifest)
    return ctx,packet_path,packet,configuration,declaration,batches


def ledger(packet_path,namespace_proof,*,pending=False):
    ctx,path,packet,configuration,declaration,batches=context(packet_path,namespace_proof)
    run=ROOT/packet['output'];folder=path.parent
    if pending and run.exists():raise ValueError('A started run cannot be labeled a not-yet-executed pending job')
    terminal=read(folder/'terminal_accounting.json') if (folder/'terminal_accounting.json').exists() else None
    execution=read(folder/'execution.json') if (folder/'execution.json').exists() else None
    if not pending and (terminal is None or terminal.get('packet_sha256')!=sha(path)):
        raise ValueError('Terminal accounted packet required; pending is an explicit separate mode')
    if execution is not None and execution.get('packet_sha256')!=sha(path):raise ValueError('Execution packet changed')
    if execution is not None and terminal is not None and execution.get('job_id')!=terminal.get('job_id'):
        raise ValueError('Native/terminal job identities differ')
    panels=[];trials=[]
    for update in UPDATES:
        p=panel_ledger(run/'held_out_validation'/f'validation_{update:05d}',batches,pending=pending)
        panels.append(dict(update=update,**{k:v for k,v in p.items() if k!='trials'}))
        trials.extend(dict(update=update,declared_training_transitions=update*128*64,**row) for row in p['trials'])
    counts={status:sum(r['status']==status for r in trials) for status in sorted({r['status'] for r in trials})}
    return dict(schema=VERSION,packet=binding(path),configuration=binding(ctx.owned(packet['inputs']['configuration'])),
        run=str(run),planned_updates=list(UPDATES),planned_trials=480,counts=counts,trials=trials,panels=panels,
        source=binding(__file__),terminal=terminal,execution=execution,pending=pending,
        fresh_transition_cap=262144,physical_qualification=False,route_credit=False,physics_steps=0,actor_calls=0)
