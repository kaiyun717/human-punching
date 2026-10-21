"""Frozen Task09 actor on four exact training snapshot targets,32 bank cycles.

This development control reuses the original native controller, closed-prefix
launch guard, and clock-v2 evaluator. No learning or held-out qualification runs.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from . import phase2_d0_clock_v2 as route
from .phase2_deadline import ExecutionDeadline
from .phase2_domain import fingerprint
from .phase2_expanded_evaluation import (ClosedProtocolNativeSpecialistEnv,
    FrozenSnapshotInput, initial_state_identity, write_launch_boundaries, audit_initialization)
from .phase2_nonuniform_control import Inputs
from .wbc.provenance import PROJECT_ROOT, sha256_file, writable_path, write_json, runtime_record

VERSION='task10b_phase2_four_snapshot_frozen_control_v1'
SCHEMA=VERSION+'_plan'
IDS=[f'jab_train_c{g}_{t}_sequential_v3_probe' for g in (4184,4251) for t in ('0p600','0p750')]
SOURCES=('src/human_punching/phase2_target_control_v1.py',
    'src/human_punching/phase2_expanded_evaluation.py', 'src/human_punching/phase2_launch.py',
    'src/human_punching/phase2_d0_clock_v2.py', 'src/human_punching/phase2_learning.py')
LIMITS=(360,300,240)


def horizons(rows):
    return {r['candidate_id']:dict(episode_duration_s=r['episode_duration_s'],
        phase_denominator_s=r['phase_denominator_s'],launch_tick=r['launch_tick'],
        outward_end_tick=r['outward_end_tick'],
        planned_physics_steps=int(math.ceil(r['episode_duration_s']/.020-1e-10))*4) for r in rows}


def requests_for(row,target,checkpoint_sha,qpos,qvel):
    if np.asarray(qpos).shape!=(8,55) or np.asarray(qvel).shape!=(8,54):
        raise ValueError('Exactly eight original family-bank states required')
    result=[]
    for bank in range(8):
        trial=f"{row['candidate_id']}_bank_{bank:02d}"
        request=dict(trial_id=trial,family='jab',requested_ttc_s=row['requested_ttc_s'],
            style=row['candidate_id'],candidate_id=row['candidate_id'],reference_sha256=row['reference_sha256'],
            target_candidate_identity=row['target_candidate_identity'],snapshot_payload_sha256=target['payload_sha256'],
            checkpoint_sha256=checkpoint_sha,snapshot_slot_id=target['snapshot_slot_id'],
            placement=target['placement'],head_point_velocity_w=target['head_point_velocity_w'],target_w=list(row['target_w']))
        request['request_identity']=fingerprint(dict(version=VERSION,request=request,bank_id=bank,
            initial_state_identity=initial_state_identity(qpos[bank],qvel[bank])))
        result.append(request)
    return result


def selected_inputs(catalog_path,audit_path,proof_path):
    from .phase2_sequential_projection_v1 import verify_projection, files as projection_files
    ctx=Inputs(proof_path)
    catalog_path=ctx.add(catalog_path);catalog,projection=verify_projection(catalog_path)
    for name,digest in projection_files(catalog_path).items():ctx.add(name,digest)
    audit_path=ctx.add(audit_path);audit=ctx.read(audit_path)
    if (audit.get('passed') is not True or audit.get('physics_steps')!=0
            or audit.get('schema')!='task10b_phase2_saved_target_reference_independent_audit_v1'
            or catalog.get('unpacked_catalog_sha256')!=audit.get('manifest_sha256')
            or catalog.get('path_solver_version')!='task10b_sequential_active_arm_target_path_v3'
            or catalog.get('planned_cells')!=4 or catalog.get('executed_cells')!=4
            or catalog.get('generated_cells')!=4 or catalog.get('admitted_candidate_cells')!=4
            or [r.get('candidate_id') for r in catalog.get('cells',[])]!=IDS
            or {r.get('candidate_id') for r in audit.get('cells',[])}!=set(IDS)
            or any(r.get('all_checks_passed') is not True for r in audit['cells'])):
        raise ValueError('Exactly four independently audited generated training probe cells required')
    native=ctx.add(catalog['unpacked_catalog_path'],catalog['unpacked_catalog_sha256'])
    source_path=ctx.add(native.parent/'plan.json',catalog['plan_sha256']);source=ctx.read(source_path)
    for key in ('domain','target_geometry','body_metadata','recipe'):
        ctx.bound(source[key])
    domain=ctx.read(ctx.bound(source['domain']));geometry=ctx.read(ctx.bound(source['target_geometry']))
    targets={t['candidate_identity']:t for t in geometry['geometry_candidates']}
    groups={g['snapshot']['command_id']:g for g in domain['selected_command_groups']}
    rows=catalog['cells']
    for row,(group,ttc) in zip(rows,[(g,t) for g in (4184,4251) for t in (.6,.75)],strict=True):
        target=targets[row['target_candidate_identity']]
        if (row.get('family')!='jab' or row.get('split')!='train' or row.get('command_id')!=group
                or row.get('requested_ttc_s')!=ttc or row.get('kinematic_admission') is not True
                or row.get('start_mode')!='family_guard' or row.get('preparation_complete_tick')!=0
                or target['split']!='train' or target['command_id']!=group or groups[group]['split']!='train'
                or target['head_target_w']!=row['target_w'] or target['original_replica_id']!=0
                or target['placement']!=dict(forward_m=0.,left_m=0.,yaw_rad=0.)
                or fingerprint({k:v for k,v in target.items() if k!='candidate_identity'})!=row['target_candidate_identity']):
            raise ValueError('Exact training group, target, family, TTC and isolated start required')
        for prefix in ('reference','legacy_runtime','source_model'):
            ctx.add(row[prefix+'_path'],row[prefix+'_sha256'])
    release=ctx.add('outputs/four-punch-milestone/manifest.json')
    args,accepted=route.family_arguments(release.parent,'jab')
    for path in accepted.values():ctx.add(path)
    if any(row['source_model_sha256']!=sha256_file(accepted['model']) for row in rows):
        raise ValueError('Retarget source model differs from the accepted native body')
    predicate=ctx.add('outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json')
    for name in SOURCES:ctx.add(name)
    for item in (ctx.namespace['proof'],ctx.namespace['nonce'],*ctx.namespace['alias_bindings']):ctx.bound(item)
    return ctx,catalog_path,audit_path,rows,targets,source,args,accepted,predicate


def bank_states(accepted):
    """Native reset stores float32; preserve that declared initialization exactly."""
    with np.load(accepted['bank'],allow_pickle=False) as archive:
        q,v=archive['qpos'][:8].astype(np.float32),archive['qvel'][:8].astype(np.float32)
    if q.shape!=(8,55) or v.shape!=(8,54):raise ValueError('Exactly eight preserved bank rows required')
    for a,b in zip(q,v,strict=True):initial_state_identity(a,b)
    return q,v


def _plan(ctx,catalog,audit,rows,targets,source,accepted,predicate):
    q,v=bank_states(accepted)
    checkpoint_sha=sha256_file(accepted['checkpoint'])
    requests=[r for row in rows for r in requests_for(row,targets[row['target_candidate_identity']],checkpoint_sha,q,v)]
    return dict(schema=SCHEMA,version=VERSION,family='jab',seed=21021,validation_seed=22021,
        catalog=ctx.binding(catalog),reference_audit=ctx.binding(audit),namespace_proof=ctx.namespace['proof'],
        accepted_checkpoint=ctx.binding(accepted['checkpoint']),predicate=ctx.binding(predicate),
        selected_candidate_ids=IDS,bank_ids=list(range(8)),environments=8,planned_trials=32,
        allocation_wall_seconds=360,application_wall_seconds=300,child_wall_seconds=240,
        serialization_reserve_seconds=30,training=False,maximum_updates=0,fresh_transition_cap=0,
        no_test_access=True,physical_qualification=False,policy_promotion=False,
        actual_launch_protocol='closed_prefix_exact_frozen_snapshot_target_20ms',requests=requests,
        horizons=horizons(rows),initial_state_identities=[initial_state_identity(a,b) for a,b in zip(q,v,strict=True)],
        video_selection=dict(rule='every flagged failure plus one deterministic hash sample per candidate',
            presampled_trial_ids=[f"{r['candidate_id']}_bank_{int(hashlib.sha256(('phase2-video-21021:'+r['candidate_id']).encode()).hexdigest()[:16],16)%8:02d}" for r in rows]),
        scope='Four training snapshot targets at.60/.75; eight original jab bank states each. Frozen actor development control, not96validation or192test qualification.',
        input_inventory=dict(sorted(ctx.files.items())))


def create_plan(catalog,audit,*,proof='outputs/task10b/p2hostalias01/probe.json'):
    ctx,catalog,audit,rows,targets,source,args,accepted,predicate=selected_inputs(catalog,audit,proof)
    return _plan(ctx,catalog,audit,rows,targets,source,accepted,predicate)


def validate_plan(plan):
    if plan.get('schema')!=SCHEMA or plan.get('version')!=VERSION:raise ValueError('Exact four-target control plan required')
    data=selected_inputs(plan['catalog']['path'],plan['reference_audit']['path'],plan['namespace_proof']['path'])
    ctx,catalog,audit,rows,targets,source,args,accepted,predicate=data
    if plan!=_plan(ctx,catalog,audit,rows,targets,source,accepted,predicate):
        raise ValueError('Frozen control plan, requests, ancestry, horizons or finite limits changed')
    return data


def files(plan_path):
    path=Path(plan_path).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT.resolve()):raise ValueError('Owned control plan required')
    plan=json.loads(path.read_text());validate_plan(plan)
    return {**plan['input_inventory'],str(path.relative_to(PROJECT_ROOT)):sha256_file(path)}


def trial_ledger(plan):
    return [dict(trial_id=r['trial_id'],candidate_id=r['candidate_id'],status='unattempted',
        execution_extent='known_zero_before_attempt',execution_complete=False,numeric_complete_cycle=False) for r in plan['requests']]


def media_selection(plan,trials):
    wanted=set(plan['video_selection']['presampled_trial_ids'])
    wanted.update(r['trial_id'] for r in trials if r.get('numeric_complete_cycle') is not True)
    return dict(planned_required_trial_ids=sorted(wanted),all_flagged_preserved=True,
        rendering_executed=False,review_completed=False,physical_qualification=False)


def record_panel(ledger,panel,decisions,*,initialization_passed,actual_steps,planned_steps):
    """Retain every actual result; failed audits cannot erase executed intervals."""
    for outcome in panel['trials']:
        if outcome['trial_id'] not in ledger:raise ValueError('Unexpected target control trial')
        matching=[d for d in decisions if d['trial_id']==outcome['trial_id']]
        launched=len(matching)==1 and matching[0]['accepted'] is True
        complete=bool(actual_steps==planned_steps)
        ledger[outcome['trial_id']].update(status=outcome['disposition'],native_outcome=outcome,
            execution_extent='recorded',execution_complete=complete,
            actual_physics_steps=actual_steps,planned_physics_steps=planned_steps,
            closed_protocol_launch_accepted=launched,
            numeric_complete_cycle=bool(complete and panel['status']=='executed'
                and outcome.get('numeric_complete_cycle') and launched and initialization_passed))


def execute(plan_path,output,*,wall_seconds=240.):
    started=time.monotonic()
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Counted Slurm CUDA allocation required')
    plan_path=Path(plan_path).resolve(strict=True);plan=json.loads(plan_path.read_text())
    ctx,catalog,audit,rows,targets,source,args,accepted,predicate=validate_plan(plan)
    if type(wall_seconds) not in (int,float) or wall_seconds!=240.:raise ValueError('Exact240s child required')
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    deadline=ExecutionDeadline(240.,serialization_reserve_s=30.,session_started_monotonic=started)
    ledger=trial_ledger(plan);by_id={r['trial_id']:r for r in ledger};panels=[];error=None
    write_json(folder/'control_plan.json',plan);write_json(folder/'declared_ledger.json',ledger)
    runtime=runtime_record()
    if not runtime['declared_pins_match']:raise RuntimeError('Preserved runtime pins changed')
    write_json(folder/'launch.json',dict(schema='task10b_phase2_original_route_launch_v1',mode='frozen',
        target_control_version=VERSION,family='jab',seed=21021,slurm_job_id=os.environ['SLURM_JOB_ID'],runtime=runtime,
        catalog=str(catalog),catalog_sha256=sha256_file(catalog),predicate_sha256=sha256_file(predicate),
        control_plan=dict(path=str(plan_path),sha256=sha256_file(plan_path)),
        release_inputs={k:dict(path=str(p),sha256=sha256_file(p)) for k,p in accepted.items()},
        sources={str(PROJECT_ROOT/n):sha256_file(PROJECT_ROOT/n) for n in SOURCES},
        planned_cells=IDS,diagnostic_bank_ids=list(range(8)),planned_trials_per_panel=8,total_planned_trials=32,
        budget=None,fresh_transition_cap=0,training=False,postprocessing_clock_adapter=route.evaluator_receipt(),
        closed_protocol_launch_supported=True,scope=plan['scope'],physical_qualification=False))
    torch.set_num_threads(1);torch.manual_seed(21021);np.random.seed(21021)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    q,v=bank_states(accepted)
    import warp as wp
    from .wbc.telemetry.cycle_qualification import native_stream_scope
    wp.init();policy=None;initial_digest=None
    try:
        with deadline.signal_handlers():
            for index,row in enumerate(rows):
                deadline.checkpoint('before_target_control_cell')
                requests=[r for r in plan['requests'] if r['candidate_id']==row['candidate_id']]
                cell=folder/f'validation_{index:05d}';raw_folder=cell/row['candidate_id'];env=None
                for r in requests:by_id[r['trial_id']].update(status='execution_extent_unknown',execution_extent='constructor_or_native_attempt_started')
                write_json(folder/f'attempt_{index:04d}.json',dict(candidate_id=row['candidate_id'],trial_ids=[r['trial_id'] for r in requests],constructor_started=True))
                write_json(folder/f'ledger_started_{index:04d}.json',ledger)
                try:
                    with native_stream_scope(),torch.no_grad():
                        env=ClosedProtocolNativeSpecialistEnv(args,[row],num_envs=8,deadline=deadline,
                            expected_qpos=q,expected_qvel=v,requests=requests,target_input=FrozenSnapshotInput(requests))
                        if env.library.bytes>256*1024**2:raise ValueError('Resident reference cap exceeded')
                        deadline.checkpoint('after_target_control_constructor')
                        if policy is None:
                            policy=route.fresh_policy(accepted,env.device);initial_digest=route.policy_digest(policy)
                            route.export_runtime_model(args.model,folder/'runtime.xml',env.model)
                            parity=route.restored_policy_parity(accepted['checkpoint'],policy,torch.zeros((1,369),device=env.device))
                            write_json(folder/'restored_checkpoint_parity.json',parity)
                            if not all(parity[k] is True for k in ('parameters_equal','actions_equal')) or parity['maximum_action_error']!=0:
                                raise ValueError('Accepted frozen checkpoint parity failed')
                        env.bind_policy(policy)
                        panel=route.run_cell_panel(env,policy,[row],cell,initial_qpos=q,initial_qvel=v,seed=22021,predicate=ctx.read(predicate))
                        init=audit_initialization(raw_folder/'raw_trace.npz',q,v)
                        write_json(raw_folder/'initialization_audit.json',init)
                        record_panel(by_id,panel,env.closed_launch_records,initialization_passed=init['passed'],
                            actual_steps=env.physics_steps_executed,
                            planned_steps=plan['horizons'][row['candidate_id']]['planned_physics_steps'])
                        panels.append(dict(candidate_id=row['candidate_id'],path=str(cell),status=panel['status'],
                            actual_physics_steps=env.physics_steps_executed,actual_policy_updates=env.actor_updates_executed,
                            measured_resident_reference_bytes=env.library.bytes))
                        if not init['passed']:raise ValueError('Original bank initialization differs')
                        if panel['status']!='executed':raise RuntimeError('Native physical interruption; preserve panel and remainder')
                        if route.policy_digest(policy)!=initial_digest:raise ValueError('Frozen actor weights changed')
                finally:
                    if env is not None:
                        if raw_folder.exists():
                            snapshots=write_launch_boundaries(raw_folder,env.closed_boundary_snapshots)
                            write_json(raw_folder/'closed_protocol_launch.json',dict(schema='task10b_phase2_closed_protocol_native_receipt_v1',
                                requests=requests,decisions=env.closed_launch_records,boundary_snapshots=snapshots,
                                arbitrary_live_state_admission=False,physical_state_or_history_replaced=False,
                                real_time_operation_qualified=False,independent_expanded_gate_audit_required=True,
                                wall_scope='Measured target callback/cache lookup/finalization while simulation pauses; actor excluded'))
                            write_json(raw_folder/'target_control_scope.json',dict(version=VERSION,scope=plan['scope'],
                                inherited_D0_label_is_historical=True,closed_protocol_requests=requests,
                                qualification=False,training=False))
                        del env
                    write_json(folder/f'ledger_finished_{index:04d}.json',ledger)
                    gc.collect();torch.cuda.empty_cache()
    except Exception as caught:error=dict(type=type(caught).__name__,message=str(caught))
    write_json(folder/'outcomes.json',ledger);write_json(folder/'media_selection.json',media_selection(plan,ledger))
    result=dict(schema=VERSION,status='failed' if error else 'executed_pending_audit_and_video',exception=error,
        planned_trials=32,completed_trials=sum(r['execution_complete'] for r in ledger),
        numeric_complete_cycles=sum(r['numeric_complete_cycle'] for r in ledger),panels=panels,
        frozen_actor_unchanged=bool(policy is not None and route.policy_digest(policy)==initial_digest),
        fresh_transitions=0,wall_seconds=time.monotonic()-started,physical_qualification=False,
        final_test_executed=False,phase2_complete=False,automatic_followup=False)
    write_json(folder/'execution.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--wall-seconds',type=float,required=True)
    args=parser.parse_args(argv);result=execute(args.plan,args.output,wall_seconds=args.wall_seconds)
    print(json.dumps(result,allow_nan=False));return 1 if result['status']=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
