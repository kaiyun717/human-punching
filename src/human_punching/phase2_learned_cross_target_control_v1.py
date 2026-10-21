"""One learned D0 cross checkpoint comparison; no learning or qualification.

Eight exact original-reference regressions must pass before the existing .205
target cells execute. All32 target requests, including16 unavailable references,
remain in the40-request development ledger. No new reference is generated.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from . import phase2_d0_clock_v2 as route
from . import phase2_family_target_probe_v1 as generation
from .phase2_deadline import ExecutionDeadline
from .phase2_domain import fingerprint
from .phase2_expanded_clocks_v1 import EventMetadata, VERSION as CLOCK_VERSION
from .phase2_expanded_evaluation import (ClosedProtocolNativeSpecialistEnv,
    FrozenSnapshotInput, initial_state_identity, write_launch_boundaries, audit_initialization)
from .phase2_nonuniform_control import Inputs
from .phase2_target_control_v1 import bank_states, horizons, record_panel, media_selection
from .wbc.provenance import PROJECT_ROOT, sha256_file, writable_path, write_json, runtime_record

VERSION='task10b_phase2_learned_d0_cross_target_diagnostic_v1'
SCHEMA=VERSION+'_plan'
LIMITS=(360,300,210)
SOURCES=('src/human_punching/phase2_learned_cross_target_control_v1.py',
    'src/human_punching/phase2_family_target_control_v1.py',
    'src/human_punching/phase2_family_projection_v1.py',
    'src/human_punching/phase2_target_control_v1.py',
    'src/human_punching/phase2_expanded_clocks_v1.py',
    'src/human_punching/phase2_expanded_evaluation.py',
    'src/human_punching/phase2_launch.py',
    'src/human_punching/phase2_d0_clock_v2.py','src/human_punching/phase2_learning.py')


def requests_for(row,target,checkpoint_sha,qpos,qvel):
    if np.asarray(qpos).shape!=(8,55) or np.asarray(qvel).shape!=(8,54):
        raise ValueError('Exactly eight original family-bank states required')
    requests=[]
    for bank in range(8):
        request=dict(trial_id=f"{row['candidate_id']}_bank_{bank:02d}",family=row['family'],
            requested_ttc_s=row['requested_ttc_s'],style=row['candidate_id'],
            candidate_id=row['candidate_id'],reference_sha256=row['reference_sha256'],
            reference_identity_kind=row['reference_identity_kind'],
            target_candidate_identity=row['target_candidate_identity'],snapshot_payload_sha256=target['payload_sha256'],
            checkpoint_sha256=checkpoint_sha,snapshot_slot_id=target['snapshot_slot_id'],
            placement=target['placement'],head_point_velocity_w=target['head_point_velocity_w'],target_w=list(row['target_w']))
        request['request_identity']=fingerprint(dict(version=VERSION,request=request,bank_id=bank,
            initial_state_identity=initial_state_identity(qpos[bank],qvel[bank])))
        requests.append(request)
    return requests


def declared_rows(catalog,source,cells):
    """Complete intended ledger without pretending a missing reference exists."""
    actual=catalog.get('cells',[]);unattempted=catalog.get('unattempted_cells',[])
    expected=[r['candidate_id'] for r in source['cells']]
    if (len(expected)!=4 or len(set(expected))!=4
            or len({r['candidate_id'] for r in actual})!=len(actual)
            or {r['candidate_id'] for r in actual}&{r['candidate_id'] for r in unattempted}
            or {r['candidate_id'] for r in actual+unattempted}!=set(expected)
            or catalog.get('planned_cells')!=4 or catalog.get('executed_cells')!=len(actual)):
        raise ValueError('All four reference dispositions must remain explicit and unique')
    by_id={r['candidate_id']:r for r in actual};result=[]
    for entry,target,parent,_,_ in cells:
        row=by_id.get(entry['candidate_id'])
        if row is None:
            saved=next(r for r in unattempted if r['candidate_id']==entry['candidate_id'])
            if saved!=entry:raise ValueError('Unattempted entry differs from the original planned request')
            digest=fingerprint(dict(manifest_sha256=catalog['unpacked_catalog_sha256'],entry=entry,disposition='unattempted'))
            row=dict(**entry,family=source['family'],split='train',command_id=target['command_id'],
                requested_ttc_s=parent['requested_ttc_s'],target_w=target['head_target_w'],
                generation_disposition='unattempted',kinematic_admission=False,
                rejection_reasons=['reference_generation_unattempted'],reference_sha256=digest,
                reference_identity_kind='unattempted_declaration',start_mode='family_guard',preparation_complete_tick=0,
                **{k:parent[k] for k in ('episode_duration_s','phase_denominator_s','launch_tick','outward_end_tick')})
        if (row.get('candidate_id')!=entry['candidate_id'] or row.get('family')!=source['family']
                or row.get('split')!='train' or row.get('command_id')!=target['command_id']
                or row.get('target_candidate_identity')!=target['candidate_identity']
                or row.get('target_w')!=target['head_target_w'] or row.get('requested_ttc_s')!=parent['requested_ttc_s']
                or row.get('start_mode')!='family_guard' or row.get('preparation_complete_tick')!=0
                or row.get('generation_disposition') not in ('generated','failed','timeout','unattempted')):
            raise ValueError('Reference disposition changed the exact family/group/target/TTC request')
        admitted=row['generation_disposition']=='generated' and row.get('kinematic_admission') is True
        if not admitted and not row.get('rejection_reasons'):
            raise ValueError('A negative cell requires its preserved explicit reason')
        if admitted and row.get('reference_identity_kind')!='reference_content':
            raise ValueError('Only content-bound admitted references may execute')
        result.append(dict(row,native_execution_admitted=admitted))
    return result


def selected_inputs(catalog_path,audit_path,proof_path):
    from .phase2_family_projection_v1 import verify_projection,files as projection_files
    ctx=Inputs(proof_path);catalog_path=ctx.add(catalog_path)
    catalog,projection=verify_projection(catalog_path)
    for name,digest in projection_files(catalog_path).items():ctx.add(name,digest)
    audit_path=ctx.add(audit_path);audit=ctx.read(audit_path)
    native=ctx.add(catalog['unpacked_catalog_path'],catalog['unpacked_catalog_sha256'])
    source_path=ctx.add(native.parent/'plan.json',catalog['plan_sha256']);source=ctx.read(source_path)
    _,cells,paths=generation.resolve_cells(source)
    for name,path in paths.items():ctx.add(path,source[name]['sha256'])
    if (audit.get('schema')!='task10b_phase2_family_saved_target_reference_independent_audit_v1'
            or audit.get('passed') is not True or audit.get('physics_steps')!=0
            or audit.get('manifest_sha256')!=catalog['unpacked_catalog_sha256']
            or audit.get('family')!=source.get('family')
            or audit.get('active_hand_index')!=(0 if source.get('family')=='left_hook' else 1)
            or audit.get('packet_sha256')!=projection['generation_packet']['sha256']
            or source.get('family') not in generation.FAMILIES):
        raise ValueError('Independent saved integrity audit for this exact family catalog required')
    audited={r['candidate_id']:r for r in audit.get('cells',[])}
    actual={r['candidate_id']:r for r in catalog['cells']}
    if set(audited)!=set(actual) or any(r.get('all_checks_passed') is not True for r in audited.values()):
        raise ValueError('All attempted reference outcomes, including negatives, require integrity audit')
    rows=declared_rows(catalog,source,cells)
    targets={t['candidate_identity']:t for _,t,_,_,_ in cells}
    for row in rows:
        if row['generation_disposition']=='generated':
            for prefix in ('reference','legacy_runtime','source_model'):
                ctx.add(row[prefix+'_path'],row[prefix+'_sha256'])
        elif row['generation_disposition']!='unattempted':
            ctx.add(row['generation_failure_path'],row['generation_failure_sha256'])
    release=ctx.add('outputs/four-punch-milestone/manifest.json')
    args,accepted=route.family_arguments(release.parent,source['family'])
    for path in accepted.values():ctx.add(path)
    if any(row['source_model_sha256']!=sha256_file(accepted['model']) for row in rows if row['native_execution_admitted']):
        raise ValueError('Retargeted source model differs from accepted native family body')
    if source['family']!='cross' or sha256_file(catalog_path)!='dc52182c9227e3ca8542ceca9635dee8eef6f5104b40f72657afde030b044c1b':
        raise ValueError('Only the exact prior.205 cross target catalog may execute')
    checkpoint=ctx.add('outputs/task10b/p2cp02/run/training/checkpoint_final.pt',
        'e3d720096c46de135c5e512b6db5a952032d8d32ad677ec22b00c23e07a6a2ce')
    summary=ctx.read('outputs/task10b/p2cp02/run/training/summary.json')
    terminal=ctx.read('outputs/task10b/p2cp02/terminal_accounting.json')
    ctx.add('outputs/task10b/p2cp02/packet.json');ctx.add('outputs/task10b/p2cp02/run/launch.json')
    ctx.add('outputs/task10b/p2crosslhnextreview01/recommendation.json')
    if (summary['completed_updates']!=32 or summary['fresh_transitions']!=262144
            or summary['checkpoint']['sha256']!=sha256_file(checkpoint)
            or summary['ancestry']['sha256']!=sha256_file(accepted['checkpoint'])
            or summary['weight_changed'] is not True or summary['hard_faults']
            or summary['reload_parity']!=dict(parameters_equal=True,actions_equal=True,maximum_action_error=0.)
            or terminal['job_id']!='26491514' or terminal['state']!='COMPLETED'):
        raise ValueError('Exact terminal learned cross ancestry and saved parity required')
    accepted=dict(accepted,accepted_task09_checkpoint=accepted['checkpoint'],checkpoint=checkpoint)
    predicate=ctx.add('outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json')
    for name in SOURCES:ctx.add(name)
    for item in (ctx.namespace['proof'],ctx.namespace['nonce'],*ctx.namespace['alias_bindings']):ctx.bound(item)
    return ctx,catalog_path,audit_path,rows,targets,source,args,accepted,predicate



def _original_inputs(ctx):
    catalog=ctx.add('outputs/task10b/p2nr01/run/manifest.json','fedddbcf5ab1ebb8a7d61be7a2cd7e913cfc9d7d544985841acce035e1ba9a6b')
    row=next(r for r in ctx.read(catalog)['cells'] if r['candidate_id']=='cross_original')
    if (row['kind']!='original_control' or row['reference_sha256']!='18658a584de27ecea67fd2670b481a9c8d1b9ee85dce1c37feee9a7b94168633'
            or row['legacy_runtime_sha256']!='57b0a730141b644ca855bc85d40806d128d5ad6d9eb5cd094c8d2e089a10de9b'
            or row['requested_ttc_s'] is not None or row['launch_tick']!=96 or row['outward_end_tick']!=440):
        raise ValueError('Exact accepted original cross control required')
    for path_key,sha_key in (('reference_path','reference_sha256'),('legacy_runtime_path','legacy_runtime_sha256'),('source_model_path','source_model_sha256')):ctx.add(row[path_key],row[sha_key])
    return dict(catalog=ctx.binding(catalog),row=row,planned_trials=8,bank_ids=list(range(8)),
        stop_before_targets_on_any_numeric_or_dynamics_failure=True,
        planned_actor_padded_physics_steps=960,own_horizon_physics_steps=960,
        requested_TTC_qualification=False)


def _original_ledger():
    return [dict(trial_id=f'cross_original_bank_{i:02d}',candidate_id='cross_original',status='unattempted',
        execution_extent='known_zero_before_attempt',actual_physics_steps=0,execution_complete=False,
        numeric_complete_cycle=False,native_execution_admitted=True) for i in range(8)]


def regression_passed(panel,initialization,actual_steps):
    expected=[f'cross_original_bank_{i:02d}' for i in range(8)]
    return bool(panel.get('status')=='executed' and initialization.get('passed') is True and actual_steps==960
        and not panel.get('hard_faults') and [r.get('trial_id') for r in panel.get('trials',[])]==expected
        and all(r.get('numeric_complete_cycle') is True and r.get('disposition')=='executed' for r in panel['trials']))


def _run_original_regression(folder,args,accepted,row,q,v,predicate,ctx,deadline):
    import torch
    from .wbc.telemetry.cycle_qualification import native_stream_scope
    folder.mkdir();ledger=_original_ledger();env=None;passed=False;error=None;panel=None;init=None
    write_json(folder/'declared_ledger.json',ledger)
    for trial in ledger:trial.update(status='execution_extent_unknown',execution_extent='constructor_or_native_attempt_started')
    write_json(folder/'attempt_intent.json',dict(trial_ids=[r['trial_id'] for r in ledger],constructor_started=True))
    runtime=runtime_record()
    write_json(folder/'launch.json',dict(schema='task10b_phase2_original_route_launch_v1',mode='frozen',family='cross',seed=21021,
        runtime=runtime,slurm_job_id=os.environ['SLURM_JOB_ID'],catalog=str(ctx.owned('outputs/task10b/p2nr01/run/manifest.json')),
        catalog_sha256='fedddbcf5ab1ebb8a7d61be7a2cd7e913cfc9d7d544985841acce035e1ba9a6b',predicate=str(predicate),predicate_sha256=sha256_file(predicate),
        release_inputs={k:dict(path=str(p),sha256=sha256_file(p)) for k,p in accepted.items()},
        planned_cells=['cross_original'],diagnostic_bank_ids=list(range(8)),training=False,fresh_transition_cap=0,
        postprocessing_clock_adapter=route.evaluator_receipt(),scope='Original exact-reference learned-checkpoint regression only; not targetTTC qualification'))
    try:
        with native_stream_scope(),torch.no_grad():
            deadline.checkpoint('before_original_cross_regression')
            env=route.NativeSpecialistEnv(args,[row],num_envs=8,deadline=deadline)
            policy=route.fresh_policy(accepted,env.device);digest=route.policy_digest(policy)
            route.export_runtime_model(args.model,folder/'runtime.xml',env.model)
            parity=route.restored_policy_parity(accepted['checkpoint'],policy,torch.zeros((1,369),device=env.device))
            write_json(folder/'restored_checkpoint_parity.json',parity)
            if parity!=dict(parameters_equal=True,actions_equal=True,maximum_action_error=0.):raise ValueError('Learned checkpoint restore parity failed')
            panel=route.run_cell_panel(env,policy,[row],folder/'validation_00000',initial_qpos=q,initial_qvel=v,seed=22021,predicate=ctx.read(predicate))
            raw=folder/'validation_00000/cross_original/raw_trace.npz';init=audit_initialization(raw,q,v)
            write_json(raw.parent/'initialization_audit.json',init)
            for outcome in panel['trials']:
                trial=next(r for r in ledger if r['trial_id']==outcome['trial_id'])
                trial.update(status=outcome['disposition'],execution_extent='recorded',native_outcome=outcome,
                    actual_physics_steps=env.physics_steps_executed,execution_complete=env.physics_steps_executed==960,
                    numeric_complete_cycle=bool(outcome.get('numeric_complete_cycle') and init['passed'] and panel['status']=='executed'))
            passed=regression_passed(panel,init,env.physics_steps_executed) and route.policy_digest(policy)==digest
    except Exception as caught:error=dict(type=type(caught).__name__,message=str(caught))
    finally:
        write_json(folder/'outcomes.json',ledger)
        if env is not None:del env
        gc.collect();torch.cuda.empty_cache()
    result=dict(passed=passed,exception=error,planned_trials=8,completed_trials=sum(r['execution_complete'] for r in ledger),
        numeric_complete_cycles=sum(r['numeric_complete_cycle'] for r in ledger),new_learning=False,physical_qualification=False)
    write_json(folder/'regression.json',result);return result


def _plan(ctx,catalog,audit,rows,targets,source,accepted,predicate):
    q,v=bank_states(accepted);checkpoint_sha=sha256_file(accepted['checkpoint'])
    requests=[r for row in rows for r in requests_for(row,targets[row['target_candidate_identity']],checkpoint_sha,q,v)]
    admitted=[r['candidate_id'] for r in rows if r['native_execution_admitted']]
    original=_original_inputs(ctx)
    if len(admitted)!=2 or any(r['command_id']!=4251 for r in rows if r['native_execution_admitted']):raise ValueError('Exactly the prior16 executable target requests required')
    return dict(schema=SCHEMA,version=VERSION,family=source['family'],seed=21021,validation_seed=22021,
        catalog=ctx.binding(catalog),reference_audit=ctx.binding(audit),namespace_proof=ctx.namespace['proof'],
        evaluation_checkpoint=ctx.binding(accepted['checkpoint']),accepted_task09_ancestor=ctx.binding(accepted['accepted_task09_checkpoint']),predicate=ctx.binding(predicate),original_regression=original,
        selected_candidate_ids=[r['candidate_id'] for r in rows],admitted_candidate_ids=admitted,
        reference_dispositions=[dict(candidate_id=r['candidate_id'],generation_disposition=r['generation_disposition'],
            native_execution_admitted=r['native_execution_admitted'],rejection_reasons=r.get('rejection_reasons',[])) for r in rows],
        bank_ids=list(range(8)),environments=8,planned_trials=40,target_planned_trials=32,native_attempt_capacity=8+len(admitted)*8,target_native_attempt_capacity=len(admitted)*8,
        allocation_wall_seconds=360,application_wall_seconds=300,child_wall_seconds=210,
        serialization_reserve_seconds=30,training=False,maximum_updates=0,fresh_transition_cap=0,
        no_test_access=True,physical_qualification=False,policy_promotion=False,
        actual_launch_protocol='closed_prefix_exact_frozen_snapshot_target_20ms',requests=requests,
        horizons=horizons(rows),event_clock_version=CLOCK_VERSION,
        initial_state_identities=[initial_state_identity(a,b) for a,b in zip(q,v,strict=True)],
        video_selection=dict(rule='every flagged executed failure plus one deterministic hash sample per candidate; impossible negative videos remain unavailable',
            presampled_trial_ids=[f"{r['candidate_id']}_bank_{int(hashlib.sha256(('phase2-video-21021:'+r['candidate_id']).encode()).hexdigest()[:16],16)%8:02d}" for r in rows]+['cross_original_bank_03']),
        scope='One fixed learnedD0cross checkpoint; eight original-reference regression trials first, then32 declared training target requests with16 unavailable references. No learning,96validation,192test,independent seed,routefreeze,sharedstart or physical qualification.',
        input_inventory=dict(sorted(ctx.files.items())))


def create_plan(catalog,audit,*,proof='outputs/task10b/p2hostalias01/probe.json'):
    ctx,catalog,audit,rows,targets,source,args,accepted,predicate=selected_inputs(catalog,audit,proof)
    return _plan(ctx,catalog,audit,rows,targets,source,accepted,predicate)


def validate_plan(plan):
    if plan.get('schema')!=SCHEMA or plan.get('version')!=VERSION:raise ValueError('Explicit family-control plan required')
    data=selected_inputs(plan['catalog']['path'],plan['reference_audit']['path'],plan['namespace_proof']['path'])
    ctx,catalog,audit,rows,targets,source,args,accepted,predicate=data
    if plan!=_plan(ctx,catalog,audit,rows,targets,source,accepted,predicate):raise ValueError('Exact control plan changed')
    return data


def files(plan_path):
    path=Path(plan_path).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT.resolve()):raise ValueError('Owned control plan required')
    plan=json.loads(path.read_text());validate_plan(plan)
    return {**plan['input_inventory'],str(path.relative_to(PROJECT_ROOT)):sha256_file(path)}


def trial_ledger(plan):
    dispositions={r['candidate_id']:r for r in plan['reference_dispositions']}
    result=[]
    for request in plan['requests']:
        row=dispositions[request['candidate_id']];admitted=row['native_execution_admitted']
        result.append(dict(trial_id=request['trial_id'],candidate_id=request['candidate_id'],
            status='unattempted' if admitted else 'reference_unavailable',
            reference_generation_disposition=row['generation_disposition'],rejection_reasons=row['rejection_reasons'],
            execution_extent='known_zero_before_attempt',actual_physics_steps=0,
            execution_complete=False,numeric_complete_cycle=False,native_execution_admitted=admitted))
    return result


@contextmanager
def event_clock_scope():
    """Intercept first writes from the clock-v2 evaluator, preserving its math."""
    evaluator=route.evaluate_policy
    def evaluate(env,policy,output,**kwargs):
        metadata=EventMetadata(env,output,kwargs['reference_ids'],kwargs['trial_ids'])
        metadata.ancestry['clock_v2_evaluator']=dict(path=str(Path(route.__file__).resolve()),sha256=sha256_file(route.__file__))
        writer=route.write_json
        try:
            route.write_json=lambda path,value:writer(path,metadata.before_write(path,value))
            return evaluator(env,policy,output,**kwargs)
        finally:route.write_json=writer
    try:
        route.evaluate_policy=evaluate
        yield
    finally:route.evaluate_policy=evaluator


def execute(plan_path,output,*,wall_seconds=210.):
    started=time.monotonic()
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Counted Slurm CUDA allocation required')
    plan_path=Path(plan_path).resolve(strict=True);plan=json.loads(plan_path.read_text())
    ctx,catalog,audit,rows,targets,source,args,accepted,predicate=validate_plan(plan)
    if plan['native_attempt_capacity']==0:raise ValueError('No admitted references; use CPU negative ledger without allocation')
    if type(wall_seconds) not in (int,float) or wall_seconds!=210.:raise ValueError('Exact210s child required')
    comparison=writable_path(output);comparison.mkdir(parents=True,exist_ok=False)
    write_json(comparison/'comparison_plan.json',plan)
    folder=comparison/'target_control';folder.mkdir()
    deadline=ExecutionDeadline(210.,serialization_reserve_s=30.,session_started_monotonic=started)
    ledger=trial_ledger(plan);by_id={r['trial_id']:r for r in ledger};panels=[];error=None
    write_json(folder/'control_plan.json',plan);write_json(folder/'declared_ledger.json',ledger)
    write_json(comparison/'declared_ledger.json',_original_ledger()+ledger)
    runtime=runtime_record()
    if not runtime['declared_pins_match']:raise RuntimeError('Preserved runtime pins changed')
    write_json(folder/'launch.json',dict(schema='task10b_phase2_family_target_control_launch_v1',mode='learned_checkpoint_frozen_diagnostic',
        target_control_version=VERSION,family=plan['family'],seed=21021,slurm_job_id=os.environ['SLURM_JOB_ID'],runtime=runtime,
        catalog=str(catalog),catalog_sha256=sha256_file(catalog),predicate_sha256=sha256_file(predicate),
        control_plan=dict(path=str(plan_path),sha256=sha256_file(plan_path)),
        release_inputs={k:dict(path=str(p),sha256=sha256_file(p)) for k,p in accepted.items()},
        sources={str(PROJECT_ROOT/n):sha256_file(PROJECT_ROOT/n) for n in SOURCES},
        planned_cells=plan['selected_candidate_ids'],admitted_cells=plan['admitted_candidate_ids'],
        diagnostic_bank_ids=list(range(8)),planned_trials_per_panel=8,total_planned_trials=32,
        budget=None,fresh_transition_cap=0,training=False,postprocessing_clock_adapter=route.evaluator_receipt(),
        event_clock_version=CLOCK_VERSION,closed_protocol_launch_supported=True,scope=plan['scope'],physical_qualification=False))
    torch.set_num_threads(1);torch.manual_seed(21021);np.random.seed(21021)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    q,v=bank_states(accepted)
    import warp as wp
    from .wbc.telemetry.cycle_qualification import native_stream_scope
    wp.init();policy=None;initial_digest=None
    try:
        with deadline.signal_handlers():
            regression=_run_original_regression(comparison/'original_regression',args,accepted,plan['original_regression']['row'],q,v,predicate,ctx,deadline)
            if not regression['passed']:raise ValueError('Original learned-checkpoint regression failed; all target executions remain unattempted')
        with deadline.signal_handlers(),event_clock_scope():
            for index,row in enumerate(rows):
                if not row['native_execution_admitted']:continue
                deadline.checkpoint('before_family_target_control_cell')
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
                        deadline.checkpoint('after_family_target_control_constructor')
                        if policy is None:
                            policy=route.fresh_policy(accepted,env.device);initial_digest=route.policy_digest(policy)
                            route.export_runtime_model(args.model,folder/'runtime.xml',env.model)
                            parity=route.restored_policy_parity(accepted['checkpoint'],policy,torch.zeros((1,369),device=env.device))
                            write_json(folder/'restored_checkpoint_parity.json',parity)
                            if not all(parity[k] is True for k in ('parameters_equal','actions_equal')) or parity['maximum_action_error']!=0:
                                raise ValueError('Selected learned checkpoint restore parity failed')
                        env.bind_policy(policy)
                        panel=route.run_cell_panel(env,policy,[row],cell,initial_qpos=q,initial_qvel=v,seed=22021,predicate=ctx.read(predicate))
                        init=audit_initialization(raw_folder/'raw_trace.npz',q,v);write_json(raw_folder/'initialization_audit.json',init)
                        record_panel(by_id,panel,env.closed_launch_records,initialization_passed=init['passed'],
                            actual_steps=env.physics_steps_executed,planned_steps=plan['horizons'][row['candidate_id']]['planned_physics_steps'])
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
                                event_clock_version=CLOCK_VERSION,closed_protocol_requests=requests,qualification=False,training=False))
                        del env
                    write_json(folder/f'ledger_finished_{index:04d}.json',ledger)
                    gc.collect();torch.cuda.empty_cache()
    except Exception as caught:error=dict(type=type(caught).__name__,message=str(caught))
    write_json(folder/'outcomes.json',ledger)
    selection=media_selection(plan,ledger)
    selection.update(unavailable_trial_ids=[r['trial_id'] for r in ledger if not r['native_execution_admitted']],
        unavailable_reason='Reference generation failure/rejection remains an unexecuted requested cycle')
    write_json(folder/'media_selection.json',selection)
    result=dict(schema=VERSION,status='failed' if error else 'executed_pending_audit_and_video',exception=error,
        planned_trials=32,completed_trials=sum(r['execution_complete'] for r in ledger),
        reference_unavailable_trials=sum(not r['native_execution_admitted'] for r in ledger),
        numeric_complete_cycles=sum(r['numeric_complete_cycle'] for r in ledger),panels=panels,
        frozen_actor_unchanged=None if policy is None else route.policy_digest(policy)==initial_digest,
        fresh_transitions=0,wall_seconds=time.monotonic()-started,physical_qualification=False,
        final_test_executed=False,phase2_complete=False,automatic_followup=False)
    write_json(folder/'execution.json',result)
    original_path=comparison/'original_regression/outcomes.json'
    original=json.loads(original_path.read_text()) if original_path.exists() else _original_ledger()
    all_outcomes=original+ledger
    if len(all_outcomes)!=40 or len({r['trial_id'] for r in all_outcomes})!=40:raise ValueError('Exact40 development requests required')
    write_json(comparison/'outcomes.json',all_outcomes)
    selection=media_selection(plan,all_outcomes)
    selection['planned_required_trial_ids']=sorted(set(selection['planned_required_trial_ids'])|{'cross_original_bank_03'})
    selection.update(unavailable_trial_ids=[r['trial_id'] for r in all_outcomes if r.get('native_execution_admitted') is False],
        rule='One prespecified original bank03 plus existing candidate hash samples and every flagged trial; unavailable or unattempted clips remain unavailable',
        no_numeric_failure_relabeling=True)
    write_json(comparison/'media_selection.json',selection)
    result.update(planned_trials=40,target_planned_trials=32,original_regression_trials=8,
        completed_trials=sum(r['execution_complete'] for r in all_outcomes),
        numeric_complete_cycles=sum(r['numeric_complete_cycle'] for r in all_outcomes),
        evaluation_checkpoint=plan['evaluation_checkpoint'],accepted_task09_ancestor=plan['accepted_task09_ancestor'],
        no_new_learning=True,no_new_references=True,learned_route_qualification=False)
    write_json(comparison/'execution.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--wall-seconds',type=float,required=True)
    args=parser.parse_args(argv);result=execute(args.plan,args.output,wall_seconds=args.wall_seconds)
    print(json.dumps(result,allow_nan=False));return 1 if result['status']=='failed' else 0


if __name__=='__main__':raise SystemExit(main())
