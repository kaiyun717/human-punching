"""Saved-only ledger for the exact right-hook eight-update grouped pilot.

Imports no simulator or actor. The retained generic per-batch ledger preserves
constructor/watchdog ambiguity and numerical failures. This separate schema
never relabels the historical five-panel jab experiment.
"""
from __future__ import annotations
from pathlib import Path
from .phase2_nonuniform_control import Inputs
from .phase2_grouped_saved_v1 import read, sha, binding, expected_batches, panel_ledger

ROOT=Path(__file__).resolve().parents[2]
VERSION='task10b_phase2_right_hook_grouped_saved_ledger_v1'
UPDATES=(0,8)

def context(packet_path,namespace_proof):
    ctx=Inputs(namespace_proof,root=ROOT)
    packet_path=ctx.owned(packet_path);packet=read(packet_path)
    if (packet.get('schema')!='task10b_phase2_right_hook_grouped_packet_v1' or packet.get('mode')!='grouped-pilot'
            or packet.get('allocation_wall_seconds')!=960 or packet.get('application_wall_seconds')!=900
            or packet.get('successful_route_freeze') is not None
            or packet.get('family')!='right_hook' or packet.get('seed')!=21021
            or packet.get('fresh_transition_cap')!=65536
            or packet.get('grouped_child_wall_seconds')!=720
            or packet.get('planned_validation_updates')!=list(UPDATES)
            or packet.get('planned_validation_attempts')!=192
            or packet.get('grouped_integrity_gate_version')!='task10b_phase2_grouped_entry_integrity_v3'):
        raise ValueError('Exact initial-budget clock-adapted grouped packet required')
    for name,digest in packet['files'].items():
        ctx.bound(dict(path=name,sha256=digest));ctx.bound(dict(path=packet_path.parent/'source_snapshot'/name,sha256=digest))
    for name,item in packet.get('provenance_files',{}).items():ctx.bound(dict(path=name,sha256=item['sha256']))
    configpath=ctx.owned(packet['inputs']['configuration']);relative=str(configpath.relative_to(ROOT))
    ctx.bound(dict(path=configpath,sha256=packet['files'][relative]));configuration=read(configpath)
    budget=configuration['budget']
    if (budget['environments']!=128 or budget['rollout_steps']!=64 or budget['maximum_updates']!=8
            or budget.get('maximum_wall_s')!=720 or budget.get('shutdown_reserve_s')!=90
            or configuration['stopping_rules']['fresh_transition_cap']!=65536
            or configuration['stopping_rules']['validation_updates']!=list(UPDATES)):
        raise ValueError('Exact predeclared128x64x8 and two validation cadences required')
    vconfig=read(ctx.bound(configuration['validation_configuration']))
    declaration=read(ctx.bound(vconfig['declaration_template']));manifest=read(ctx.bound(vconfig['manifest']))
    if declaration.get('trained_timings_s')!=[1.0,1.2] or declaration.get('interpolation_timing_s')!=1.1:
        raise ValueError('Exact selected RH trained/interior timings required')
    if declaration['split']!='validation' or declaration['family']!='right_hook' or declaration['seed']!=21021:
        raise ValueError('Declared right-hook validation seed/split required')
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
        run=str(run),planned_updates=list(UPDATES),planned_trials=192,counts=counts,trials=trials,panels=panels,
        source=binding(__file__),terminal=terminal,execution=execution,pending=pending,
        fresh_transition_cap=65536,physical_qualification=False,route_credit=False,physics_steps=0,actor_calls=0)


def consumer_receipts(run,configuration_binding):
    """Check exact saved consumers without converting failures into successes.

    The physical audit has its own result. Missing receipts after an interrupted
    job remain missing; reported parity never grants scientific route credit.
    """
    from .phase2_grouped_entry_v3 import checkpoint_integrity
    run=Path(run);paths={name:run/file for name,file in {
        'training':'grouped_training_receipt.json','integrity':'grouped_integrity_v3.json',
        'clocks':'grouped_event_clocks_v2.json'}.items()}
    values={name:read(path) if path.exists() else None for name,path in paths.items()}
    training=values['training'] or {};parity=values['integrity'] or {};clocks=values['clocks'] or {}
    checks={
        'training_schema':training.get('grouped_schema')=='task10b_phase2_grouped_training_receipt_v1',
        'integrity_v3_schema':parity.get('schema')=='task10b_phase2_grouped_entry_integrity_v3',
        'clocks_v2_schema':clocks.get('schema')=='task10b_phase2_grouped_clock_entry_v2',
        'exact_reported_reload_parity':checkpoint_integrity(training)['passed'],
        'integrity_consumer_passed':parity.get('passed') is True,
        'clock_consumer_passed':clocks.get('passed') is True,
        'integrity_configuration_sha':parity.get('configuration',{}).get('sha256')==configuration_binding['sha256'],
        'clock_configuration_sha':clocks.get('configuration',{}).get('sha256')==configuration_binding['sha256'],
        'fresh_optimizer':training.get('ancestry',{}).get('fresh_optimizer') is True,
        'fresh_simulation':training.get('ancestry',{}).get('fresh_simulation') is True,
        'seed_21021':training.get('ancestry',{}).get('seed')==21021,
        'finite_counters_within_cap':(type(training.get('completed_updates')) is int
            and 0<=training['completed_updates']<=8
            and training.get('fresh_transitions')==training['completed_updates']*8192
            and type(training.get('fresh_transitions')) is int),
        'initial_final_digest_weight_change_consistent':(
            isinstance(training.get('initial_policy_digest'),str)
            and isinstance(training.get('final_policy_digest'),str)
            and training.get('weight_changed') is
                (training.get('initial_policy_digest')!=training.get('final_policy_digest'))),
    }
    checks['original_training_receipt_bound']=(paths['training'].exists()
        and (parity.get('original_receipt') or {}).get('sha256')==sha(paths['training']))
    checks['clock_inherited_integrity_bound']=(paths['integrity'].exists()
        and (clocks.get('inherited_integrity') or {}).get('sha256')==sha(paths['integrity']))
    errors=[]
    final=run/'training/checkpoint_final.pt'
    checks['final_checkpoint_bytes_bound']=False
    checks['accepted_initial_parameters_digest']=False
    checks['final_parameters_digest']=False
    try:
        from .phase2_checkpoint_selection_v1 import matching_evaluated_checkpoint
        accepted=ROOT/'outputs/four-punch-milestone/right_hook/checkpoint.pt'
        ancestry=training.get('ancestry') or {}
        initial_binding={'path':str(accepted),'sha256':ancestry.get('sha256')}
        matching_evaluated_checkpoint([accepted],training.get('initial_policy_digest'),initial_binding)
        checks['accepted_initial_parameters_digest']=True
        final_binding=training.get('checkpoint') or {}
        if Path(final_binding.get('path','')).resolve()!=final.resolve():
            raise ValueError('Training final checkpoint is not this run final file')
        matching_evaluated_checkpoint([final],training.get('final_policy_digest'),final_binding)
        checks['final_parameters_digest']=True
        checks['final_checkpoint_bytes_bound']=((parity.get('checkpoint') or {}).get('sha256')==sha(final)
            and final_binding.get('sha256')==sha(final))
    except Exception as error:errors.append(dict(type=type(error).__name__,message=str(error)))
    return dict(passed=all(checks.values()),checks=checks,
        receipts={name:binding(path) if path.exists() else None for name,path in paths.items()},
        recorded_status=training.get('status'),completed_updates=training.get('completed_updates'),
        fresh_transitions=training.get('fresh_transitions'),hard_faults=training.get('hard_faults'),
        recorded_reload_parity=training.get('reload_parity'),weight_changed=training.get('weight_changed'),
        missing_receipts=[name for name,value in values.items() if value is None],errors=errors,
        route_credit=False,physical_qualification=False)
