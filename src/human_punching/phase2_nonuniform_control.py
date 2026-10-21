"""Frozen controls for only the six independently reviewed nonuniform requests.

Reference construction is never invoked. Original16 rows remain ancestry in the
unchanged22-row manifest; only two selected family rows enter run_cell_panel.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from . import phase2_nonuniform as generation
from . import phase2_d0_clock_v2 as route
from .phase2_runtime_namespace import evidence as namespace_evidence
from .phase2_deadline import ExecutionDeadline
from .wbc.provenance import sha256_file,write_json,writable_path,runtime_record

ROOT=Path(__file__).resolve().parents[2]
VERSION='task10b_phase2_six_nonuniform_frozen_control_v1'
SCHEMA='task10b_phase2_nonuniform_control_plan_v1'
FAMILIES=('cross','left_hook','right_hook')


class Inputs:
    def __init__(self,proof_path,root=None):
        self.root=Path(ROOT if root is None else root).resolve(strict=True);self.namespace=namespace_evidence(self.root,proof_path)
        self.files={}
    def owned(self,value):
        raw=Path(value)
        if '..' in raw.parts:raise ValueError('Nonuniform control path traversal rejected')
        resolved=(raw if raw.is_absolute() else self.root/raw).resolve(strict=True)
        if resolved.is_relative_to(self.root):return resolved
        for key in ('logical_root','compute_root'):
            prefix=Path(self.namespace[key])
            if raw.is_absolute() and raw.is_relative_to(prefix):
                mapped=(self.root/raw.relative_to(prefix)).resolve(strict=True)
                if (mapped.is_relative_to(self.root) and mapped.stat().st_ino==resolved.stat().st_ino
                        and (not mapped.is_file() or sha256_file(mapped)==sha256_file(resolved))):return mapped
        raise ValueError('Nonuniform control input escaped its exact proved namespace')
    def add(self,value,expected=None):
        path=self.owned(value)
        if not path.is_file():raise ValueError('Existing nonuniform control input file required')
        actual=sha256_file(path)
        if expected is not None and actual!=expected:raise ValueError('Nonuniform control input SHA changed')
        self.files[str(path.relative_to(self.root))]=actual;return path
    def bound(self,item):
        if not isinstance(item,dict) or set(item)!={'path','sha256'}:raise ValueError('Exact nonuniform input binding required')
        return self.add(item['path'],item['sha256'])
    def binding(self,path):
        p=self.add(path);return dict(path=str(p.relative_to(self.root)),sha256=self.files[str(p.relative_to(self.root))])
    def read(self,path):return json.loads(self.add(path).read_text())


def verified_selection(review_path,family,proof_path,root=None):
    if family not in FAMILIES:raise ValueError('Only cross/left_hook/right_hook nonuniform controls are declared')
    ctx=Inputs(proof_path,root);review_path=ctx.add(review_path);review=ctx.read(review_path)
    if (review.get('schema')!='task10b_phase2_nonuniform_reference_independent_review_v1'
            or review.get('passed_saved_data_consistency') is not True or review.get('physical_qualification') is not False
            or review.get('original16_rows_unchanged') is not True or review.get('complete_catalog_rows')!=22
            or review.get('all331_live_and_archived_inputs_match') is not True):
        raise ValueError('Exact independent saved-reference review required; no native qualification inferred')
    audit_path=ctx.bound(review['audit']);audit=ctx.read(audit_path)
    ctx.bound(review['auditor']);ctx.bound(review['preserved_uniform_audit']);ctx.bound(review['terminal_accounting'])
    manifest_path=ctx.bound(review['manifest']);manifest=ctx.read(manifest_path)
    execution_path=ctx.bound(review['execution']);execution=ctx.read(execution_path)
    packet_path=ctx.add(execution_path.parent/'packet.json',execution['packet_sha256']);packet=ctx.read(packet_path)
    if (execution.get('schema')!='task10b_phase2_execution_v1' or execution.get('status')!='executed'
            or execution.get('returncode')!=0 or execution.get('native_processes_started')!=1 or not execution.get('job_id')
            or packet.get('mode')!='nonuniform-references' or packet.get('training') is not False
            or packet.get('allocation_wall_seconds')!=300 or packet.get('application_wall_seconds')!=270
            or ctx.owned(packet['output'])!=manifest_path.parent):
        raise ValueError('Successful counted nonuniform generation receipt differs')
    plan_path=ctx.add(manifest_path.parent/'plan.json',manifest['plan_sha256']);source_plan=ctx.read(plan_path)
    parent_path=ctx.bound(source_plan['parent_catalog']);parent=ctx.read(parent_path)
    expected=generation.cell_plan();all_ids=[r['candidate_id'] for r in expected]
    if (manifest.get('schema')!=generation.CATALOG or manifest.get('generator_version')!=generation.VERSION
            or manifest.get('complete') is not True or manifest.get('planned_cells')!=22 or len(manifest.get('cells',[]))!=22
            or manifest.get('inherited_cells')!=16 or manifest.get('planned_new_cells')!=6
            or manifest.get('executed_new_cells')!=6 or manifest.get('exception') is not None
            or manifest['cells'][:16]!=parent.get('cells') or len(parent.get('cells',[]))!=16
            or source_plan.get('schema')!=generation.SCHEMA or source_plan.get('generator_version')!=generation.VERSION
            or source_plan.get('cells')!=expected or source_plan.get('application_wall_seconds')!=210
            or source_plan.get('comparison',{}).get('planned_cycles')!=48 or source_plan['comparison'].get('training') is not False):
        raise ValueError('Original16 ancestry or exact six-new-row generation contract changed')
    for key in ('parent_catalog','domain','assessment','release','bundle','source_ancestry','generator_sources'):
        if manifest.get(key)!=source_plan.get(key):raise ValueError('Generation manifest/plan ancestry differs')
    if (audit.get('schema')!='task10b_phase2_nonuniform_saved_reference_audit_v1'
            or audit.get('all_saved_array_checks_passed') is not True or audit.get('original16_rows_preserved') is not True
            or audit.get('packet_sha256')!=execution['packet_sha256'] or audit.get('plan_sha256')!=manifest['plan_sha256']
            or audit.get('inspected_new_cells')!=6 or len(audit.get('rows',[]))!=6
            or not audit.get('source_closure',{}).get('archived_all_match')):
        raise ValueError('Independent nonuniform array/force audit is incomplete')
    # Old source versions are archival ancestry, not a demand to revert the
    # terminal generator's launcher after a new bounded mode is implemented.
    for relative,digest in packet['files'].items():
        archive=ctx.owned(packet_path.parent/'source_snapshot'/relative)
        if not archive.is_relative_to(packet_path.parent) or sha256_file(archive)!=digest:raise ValueError('Archived generation source changed')
    for key in ('domain','predicate','release','bundle','assessment'):ctx.bound(source_plan[key])
    assessment=ctx.read(ctx.bound(source_plan['assessment']))
    if (assessment.get('schema')!='task10b_phase2_uniform_nonuniform_route_assessment_v1'
            or assessment['measured'].get('failed_cell_ids')!=[r['parent_uniform_candidate_id'] for r in expected]):
        raise ValueError('Prespecified failed-uniform comparison changed')
    for prefix in ('cells','error_localization'):
        ctx.bound(dict(path=assessment['measured'][prefix+'_path'],sha256=assessment['measured'][prefix+'_sha256']))
    for name,item in source_plan['generator_sources'].items():
        if ctx.bound(item)!=ctx.owned(name) or packet['files'].get(name)!=item['sha256']:raise ValueError('Generation implementation ancestry changed')
    release=ctx.bound(source_plan['release']).parent
    _,accepted=route.family_arguments(release,family)
    for path in accepted.values():ctx.add(path)
    for item in source_plan['source_ancestry'][family].values():ctx.bound(item)
    rows=manifest['cells'][16:];audited={r['candidate_id']:r for r in audit['rows']}
    if [r.get('candidate_id') for r in rows]!=all_ids or set(audited)!=set(all_ids):raise ValueError('Six nonuniform rows were dropped, reordered or substituted')
    selected=[r for r in rows if r['family']==family];selected_ids={r['candidate_id'] for r in selected}
    if len(selected)!=2:raise ValueError('Exactly two declared new rows per family required')
    for row,entry in zip(rows,expected,strict=True):
        if any(row.get(k)!=v for k,v in entry.items()) or row.get('generation_disposition')!='generated':
            raise ValueError('Unsupported or incomplete nonuniform row retained: '+str(row.get('candidate_id')))
        checked=audited[row['candidate_id']]
        if (checked.get('all_checks_passed') is not True or checked.get('family')!=row['family']
                or checked.get('requested_ttc_s')!=row['requested_ttc_s'] or checked.get('kinematic_admission')!=row['kinematic_admission']
                or row.get('nonuniform_plan_sha256')!=manifest['plan_sha256']
                or row.get('parent_catalog_sha256')!=source_plan['parent_catalog']['sha256']
                or row.get('generation_source_sha256')!=source_plan['generator_sources'][generation.SOURCES[0]]['sha256']
                or row.get('source_reference_sha256')!=source_plan['source_ancestry'][row['family']]['reference.npz']['sha256']):
            raise ValueError('Nonuniform row source/audit identity differs')
        if not row['kinematic_admission'] and not row.get('rejection_reasons'):raise ValueError('Unsupported row requires explicit reasons')
        if row['candidate_id'] in selected_ids:
            ref=ctx.add(row['reference_path'],row['reference_sha256'])
            if ref!=ctx.owned(manifest_path.parent/row['candidate_id']/'reference.npz') or ctx.add(row['legacy_runtime_path'],row['legacy_runtime_sha256'])!=ref:
                raise ValueError('Selected reference/legacy file differs')
    # Check every preserved generated artifact against the independent audit;
    # archive only shared metadata and the selected family's runtime products.
    actual={str(p.relative_to(ctx.root)) for p in manifest_path.parent.rglob('*') if p.is_file()}
    if actual!=set(audit['artifact_hashes']):raise ValueError('Generated artifact inventory changed')
    for relative,digest in audit['artifact_hashes'].items():
        path=ctx.owned(relative)
        if sha256_file(path)!=digest:raise ValueError('Independently audited reference artifact changed')
        if path.parent==manifest_path.parent or any(path.is_relative_to(manifest_path.parent/c) for c in selected_ids):ctx.add(path,digest)
    for item in [ctx.namespace['proof'],ctx.namespace['nonce'],*ctx.namespace['alias_bindings']]:ctx.bound(item)
    for name in ('src/human_punching/phase2_nonuniform_control.py','src/human_punching/phase2_d0_clock_v2.py','src/human_punching/phase2_learning.py'):
        ctx.add(name)
    return ctx,manifest_path,manifest,source_plan,selected,all_ids


def _plan_data(ctx,path,source,rows,all_ids,reference_review,family):
    return dict(schema=SCHEMA,version=VERSION,family=family,seed=21021,validation_seed=22021,
        bank_ids=list(range(8)),environments=8,planned_trials=16,selected_candidate_ids=[r['candidate_id'] for r in rows],
        all_six_candidate_ids=all_ids,original16_execution=False,training=False,max_updates=0,fresh_transition_cap=0,
        allocation_wall_seconds=480,application_wall_seconds=420,child_wall_seconds=360,serialization_reserve_seconds=30,
        physics_dt_s=.005,actor_dt_s=.020,horizon_rule='unchanged_clock_v2_full_reference_plus1s_actor_boundary_padding',
        reference_review=ctx.binding(reference_review),namespace_proof=ctx.namespace['proof'],
        manifest=ctx.binding(path),release=source['release'],predicate=source['predicate'],input_inventory=dict(sorted(ctx.files.items())),
        failed_or_rejected_rows=[dict(candidate_id=r['candidate_id'],reasons=r['rejection_reasons']) for r in rows if not r['kinematic_admission']],
        no_policy_promotion=True,no_test_access=True,physical_qualification=False)


def create_plan(reference_review,family,*,alias_proof='outputs/task10b/p2hostalias01/probe.json',root=None):
    ctx,path,manifest,source,rows,all_ids=verified_selection(reference_review,family,alias_proof,root)
    return _plan_data(ctx,path,source,rows,all_ids,reference_review,family)


def validate_plan(plan,*,root=None):
    if plan.get('schema')!=SCHEMA or plan.get('version')!=VERSION:raise ValueError('Exact nonuniform control plan required')
    ctx,path,manifest,source,rows,all_ids=verified_selection(plan['reference_review']['path'],plan['family'],plan['namespace_proof']['path'],root)
    expected=_plan_data(ctx,path,source,rows,all_ids,plan['reference_review']['path'],plan['family'])
    if plan!=expected:raise ValueError('Frozen nonuniform control plan or input inventory changed')
    return ctx,path,manifest,source,rows


def files(plan_path,*,root=None):
    root=Path(ROOT if root is None else root).resolve(strict=True);plan_path=Path(plan_path)
    plan_path=(plan_path if plan_path.is_absolute() else root/plan_path).resolve(strict=True)
    if not plan_path.is_relative_to(root):raise ValueError('Owned control plan required')
    plan=json.loads(plan_path.read_text());validate_plan(plan,root=root)
    return {**plan['input_inventory'],str(plan_path.relative_to(root)):sha256_file(plan_path)}


def media_files(root,inputs):
    if set(inputs)!={'run','panel'}:raise ValueError('Exact nonuniform saved media inputs required')
    root=Path(root).resolve(strict=True);run=Path(inputs['run']).resolve(strict=True)
    if not run.is_relative_to(root):raise ValueError('Owned nonuniform saved run required')
    launch=json.loads((run/'launch.json').read_text())
    if launch.get('nonuniform_control_version')!=VERSION or launch.get('mode')!='frozen':raise ValueError('Honest frozen nonuniform media launch required')
    descriptor=launch['control_plan'];plan_path=Path(descriptor['path']).resolve(strict=True)
    if sha256_file(plan_path)!=descriptor['sha256']:raise ValueError('Nonuniform media control plan changed')
    plan=json.loads(plan_path.read_text());ctx,catalog,manifest,source,rows=validate_plan(plan,root=root)
    if (ctx.owned(launch['catalog'])!=catalog or launch['catalog_sha256']!=sha256_file(catalog)
            or launch.get('planned_cells')!=plan['selected_candidate_ids'] or launch.get('family')!=plan['family']):
        raise ValueError('Nonuniform media selected cells/catalog differ')
    selected=set(plan['selected_candidate_ids']);panel=ctx.add(inputs['panel'])
    if panel.parent.parent!=run or panel.name!='summary.json':raise ValueError('Exact aggregate nonuniform panel required')
    summary=ctx.read(panel)
    if (summary.get('schema')!='task10b_phase2_d0_panel_v1' or summary.get('planned_trials')!=16
            or {r['trial_id'] for r in summary['trials']}!={f'{c}_bank_{i:02d}' for c in selected for i in range(8)}):
        raise ValueError('Nonuniform media must preserve all16 selected trial identities')
    _,accepted=route.family_arguments(ctx.bound(source['release']).parent,plan['family'])
    if set(launch['release_inputs'])!=set(accepted):raise ValueError('Nonuniform media accepted source roles differ')
    for key,path in accepted.items():
        if ctx.bound(launch['release_inputs'][key])!=path:raise ValueError('Nonuniform media accepted source differs')
    for path,digest in launch['sources'].items():ctx.add(path,digest)
    ctx.add(plan_path,descriptor['sha256']);ctx.add(run/'launch.json')
    for path in run.rglob('*'):
        if path.is_file():
            if path.suffix not in ('.json','.npz','.xml'):raise ValueError('Unexpected nonuniform saved evidence file')
            ctx.add(path)
    return dict(sorted(ctx.files.items()))


class BoundedControlEnv(route.NativeSpecialistEnv):
    def launch_gate(self):
        self.control_deadline.checkpoint('nonuniform_control_actor_boundary')
        return super().launch_gate()


def execute(plan_path,output,*,wall_seconds=360.):
    started=time.monotonic()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Counted Slurm CUDA allocation required')
    plan_path=Path(plan_path).resolve(strict=True)
    if not plan_path.is_relative_to(ROOT.resolve()):raise ValueError('Owned nonuniform control plan required')
    plan=json.loads(plan_path.read_text())
    ctx,catalog,manifest,source,rows=validate_plan(plan)
    if wall_seconds!=360 or wall_seconds!=plan['child_wall_seconds']:raise ValueError('Exact360s frozen child required')
    folder=writable_path(output);folder.mkdir(parents=True,exist_ok=False)
    deadline=ExecutionDeadline(360.,serialization_reserve_s=30.,session_started_monotonic=started)
    args,paths=route.family_arguments(ctx.bound(source['release']).parent,plan['family'])
    runtime=runtime_record()
    if not runtime['declared_pins_match']:raise RuntimeError('Preserved runtime pins changed')
    torch.set_num_threads(1);torch.manual_seed(21021);np.random.seed(21021)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write_json(folder/'control_plan.json',plan)
    write_json(folder/'launch.json',dict(schema='task10b_phase2_original_route_launch_v1',mode='frozen',family=plan['family'],seed=21021,
        slurm_job_id=os.environ['SLURM_JOB_ID'],runtime=runtime,catalog=str(catalog),catalog_sha256=sha256_file(catalog),
        predicate_sha256=source['predicate']['sha256'],release_inputs={k:dict(path=str(p),sha256=sha256_file(p)) for k,p in paths.items()},
        sources={str(Path(__file__).resolve()):sha256_file(__file__),str(Path(route.__file__).resolve()):sha256_file(route.__file__)},
        planned_cells=plan['selected_candidate_ids'],diagnostic_bank_ids=list(range(8)),planned_trials_per_panel=16,budget=None,
        fresh_transition_cap=0,no_test_split_access=True,no_g1_actor=True,no_policy_promotion=True,
        nonuniform_control_version=VERSION,control_plan=dict(path=str(plan_path),sha256=sha256_file(plan_path)),
        global_selected_six=plan['all_six_candidate_ids'],original16_execution=False,postprocessing_clock_adapter=route.evaluator_receipt()))
    predicate=ctx.read(ctx.bound(source['predicate']));admitted=[r for r in rows if r['kinematic_admission']]
    if not admitted:
        target=folder/'validation_00000';target.mkdir()
        trials=[dict(trial_id=f"{r['candidate_id']}_bank_{i:02d}",candidate_id=r['candidate_id'],disposition='offline_rejection',
            reasons=r['rejection_reasons'],numeric_complete_cycle=False) for r in rows for i in range(8)]
        result=dict(schema='task10b_phase2_d0_panel_v1',status='executed',planned_trials=16,executed_trials=0,trials=trials,
            panels=[],numeric_complete_cycles=0,offline_rejections=16,unattempted_trials=0,incomplete_trials=0,hard_faults=[],exception=None,
            independent_domain_qualification=False,no_native_attempts=True)
        write_json(target/'summary.json',result)
    else:
        import warp as wp
        from .wbc.telemetry.cycle_qualification import native_stream_scope
        wp.init()
        with deadline.signal_handlers(),native_stream_scope():
            env=BoundedControlEnv(args,admitted,num_envs=8);env.control_deadline=deadline
            policy=route.fresh_policy(paths,env.device);route.export_runtime_model(args.model,folder/'runtime.xml',env.model)
            with np.load(paths['bank'],allow_pickle=False) as bank:q,v=bank['qpos'][:8].copy(),bank['qvel'][:8].copy()
            result=route.run_cell_panel(env,policy,rows,folder/'validation_00000',initial_qpos=q,initial_qvel=v,
                seed=22021,predicate=predicate)
    write_json(folder/'execution.json',dict(mode='frozen',status=result['status'],summary_path=str(folder/'validation_00000/summary.json'),
        total_wall_s=time.monotonic()-started,slurm_job_id=os.environ['SLURM_JOB_ID'],phase2_complete=False,nonuniform_control_version=VERSION))
    return 0 if result['status']=='executed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--wall-seconds',type=float,required=True)
    args=parser.parse_args();existed=args.output.exists()
    try:return execute(args.plan,args.output,wall_seconds=args.wall_seconds)
    except Exception as error:
        if not existed and args.output.exists():write_json(args.output/'execution_error.json',dict(schema='task10b_phase2_nonuniform_control_failure_v1',
            status='failed',exception=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()),no_automatic_retry=True))
        raise


if __name__=='__main__':raise SystemExit(main())
