"""One exact ten-cell RH continuation after a counted cooperative deadline.

The first54 completed references and all prior negative dispositions remain
immutable. The unchanged solver regenerates the interrupted cell from its
original input plus the nine original unattempted cells, never a new domain.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
from . import phase2_right_hook_full_catalog_v1 as full
from .wbc.provenance import PROJECT_ROOT, sha256_file, writable_path, write_json

SCHEMA='task10b_phase2_right_hook_suffix_catalog_plan_v2'
VERSION='task10b_phase2_right_hook_suffix_catalog_v2'
ANCESTRY=('prior_plan','prior_manifest','prior_packet','prior_execution','prior_accounting','prior_audit')
SOURCE='src/human_punching/phase2_right_hook_suffix_catalog_v2.py'
AUDITOR='outputs/task10b/p2rhfullauditprep01/audit_saved_right_hook_full.py'
own=full.own
binding=full.binding
read=full.read


def prior_evidence(plan):
    data={key:read(plan[key]) for key in ANCESTRY}
    prior=data['prior_plan'];manifest=data['prior_manifest'];audit=data['prior_audit']
    packet=data['prior_packet'];execution=data['prior_execution'];accounting=data['prior_accounting']
    if (prior.get('schema')!=full.SCHEMA or prior.get('split')!='train'
            or manifest.get('complete') is not False or manifest.get('plan_sha256')!=plan['prior_plan']['sha256']
            or [manifest.get(k) for k in ('planned_cells','executed_cells','generated_cells','admitted_candidate_cells')]!=[64,55,54,54]
            or len(manifest.get('cells',[]))!=55 or manifest.get('unattempted_cells')!=prior['cells'][55:]
            or manifest.get('exception',{}).get('type')!='BoundedExecutionInterruption'
            or manifest.get('exception',{}).get('message')!='serialization_reserve at target_screen_chunk'
            or manifest.get('recipe_sha256')!=prior['recipe']['sha256']
            or packet.get('schema')!='task10b_phase2_right_hook_full_packet_v1'
            or packet.get('family')!='right_hook' or packet.get('sequential_split')!='train'
            or execution.get('status')!='failed' or execution.get('returncode')!=1
            or execution.get('native_processes_started')!=1
            or accounting.get('state')!='FAILED' or accounting.get('exit_code')!='2:0'
            or not 0<accounting.get('allocated_gpu_seconds',0)<=420
            or any(x.get('packet_sha256')!=plan['prior_packet']['sha256'] for x in (execution,accounting))
            or not isinstance(execution.get('job_id'),str) or not execution['job_id'].isdigit()
            or execution['job_id']!=accounting.get('job_id')):
        raise ValueError('Only the measured54+timeout+9 counted partial RH training run is supported')
    rows=manifest['cells']
    if ([r['candidate_id'] for r in rows]!=[r['candidate_id'] for r in prior['cells'][:55]]
            or any(r.get('generation_disposition')!='generated' or r.get('kinematic_admission') is not True for r in rows[:54])
            or any(any(r.get(k)!=v for k,v in request.items()) for r,request in zip(rows,prior['cells'][:55],strict=True))):
        raise ValueError('Exact original prefix identities and54 admitted references required')
    interrupted=rows[54]
    if (interrupted.get('generation_disposition')!='timeout' or interrupted.get('kinematic_admission') is not False
            or interrupted.get('rejection_reasons')!=['bounded_timeout']
            or interrupted.get('details',{}).get('physical_failure') is not False
            or interrupted['details'].get('boundary')!='target_screen_chunk'
            or interrupted['details'].get('reason')!='serialization_reserve'):
        raise ValueError('A geometry failure cannot be relabeled as a timeout continuation')
    if (audit.get('schema')!='task10b_phase2_rh_full_saved_target_reference_audit_v1'
            or audit.get('passed') is not True or audit.get('family')!='right_hook' or audit.get('split')!='train'
            or audit.get('source_sha256')!=sha256_file(PROJECT_ROOT/AUDITOR)
            or audit.get('physics_steps')!=0 or audit.get('reference_generation') is not False
            or len(audit.get('cells',[]))!=64 or any(r.get('all_checks_passed') is not True for r in audit['cells'])
            or [r.get('candidate_id') for r in audit['cells']]!=[r['candidate_id'] for r in prior['cells']]):
        raise ValueError('Exact independently audited64-disposition prior ledger required')
    for key in ('plan','manifest','packet','execution','accounting'):
        if own(plan['prior_'+key])!=own(dict(path=audit[key+'_path'],sha256=audit[key+'_sha256'])):
            raise ValueError('Independent audit names a different prior input')
    return prior,manifest,audit


def resolve_cells(plan):
    if (plan.get('schema')!=SCHEMA or plan.get('entry_version')!=VERSION
            or plan.get('family')!='right_hook' or plan.get('split')!='train'
            or plan.get('maximum_cells')!=10 or plan.get('maximum_child_seconds')!=90
            or plan.get('output_byte_cap')!=64*1024**2 or plan.get('physical_qualification') is not False
            or plan.get('automatic_further_retry') is not False
            or plan.get('native_frozen_control_required_before_learning') is not True
            or set(plan.get('continuation_sources',{}))!={SOURCE}):
        raise ValueError('Exact one-off10-cell90s unchanged-solver continuation required')
    if own(plan['continuation_sources'][SOURCE])!=(PROJECT_ROOT/SOURCE).resolve(strict=True):
        raise ValueError('Continuation source must resolve to this exact owned module with its declared SHA')
    prior,manifest,_=prior_evidence(plan)
    changed={'schema','entry_version','maximum_cells','maximum_child_seconds','output_byte_cap','cells'}
    if any(plan.get(k)!=v for k,v in prior.items() if k not in changed):
        raise ValueError('Continuation cannot change the original recipe, target domain, settings or source ancestry')
    if plan.get('cells')!=prior['cells'][54:]:
        raise ValueError('Exactly the original interrupted cell and nine-cell suffix, in original order, are required')
    settings,cells,paths=full.resolve_cells(prior)
    paths.update({key:own(plan[key]) for key in ANCESTRY})
    return settings,cells[54:],paths


def files(plan_path):
    plan_path=own(binding(plan_path));plan=json.loads(plan_path.read_text());resolve_cells(plan)
    closure=full.files(own(plan['prior_plan']))
    for path in [plan_path,PROJECT_ROOT/AUDITOR,*[own(plan[k]) for k in ANCESTRY],
                 *[own(item) for item in plan['continuation_sources'].values()]]:
        closure[str(path.relative_to(PROJECT_ROOT))]=sha256_file(path)
    # These are documentary prior packets, not recursively selected new inputs.
    # Their exact recorded hashes and the independent prior audit remain bound.
    return dict(sorted(closure.items()))


def generate(plan_path,output,*,wall_seconds):
    plan_path=own(binding(plan_path));plan=json.loads(plan_path.read_text());resolve_cells(plan)
    if type(wall_seconds) not in (int,float) or wall_seconds!=90:
        raise ValueError('Fixed90s child required; no late shortened execution')
    engine=full._engine();engine.resolve_cells=resolve_cells
    result=engine.generate(plan_path,output,wall_seconds=wall_seconds)
    prior=read(plan['prior_plan'])
    write_json(writable_path(output)/'right_hook_suffix_entry_v2.json',dict(schema=VERSION,
        plan_sha256=sha256_file(plan_path),wrapper_source_sha256=sha256_file(__file__),
        frozen_generator_source_sha256=sha256_file(full.engine_source.__file__),
        manifest_sha256=sha256_file(writable_path(output)/'manifest.json'),
        ancestry={key:plan[key] for key in ANCESTRY},recipe_sha256=plan['recipe']['sha256'],
        retained_prefix_ids=[r['candidate_id'] for r in prior['cells'][:54]],
        regenerated_original_ids=[r['candidate_id'] for r in prior['cells'][54:]],
        family='right_hook',split='train',child_wall_seconds=90,solver_changed=False,
        engineering_cap_rad=.205,prior_outcomes_rewritten=False,automatic_further_retry=False,
        physical_qualification=False,native_frozen_control_required_before_learning=True))
    return result


def create_plan(prior_audit_path,output):
    audit=read(binding(prior_audit_path));ancestry={'prior_audit':binding(prior_audit_path)}
    for key in ('plan','manifest','packet','execution','accounting'):
        item=binding(audit[key+'_path'])
        if item['sha256']!=audit[key+'_sha256']:raise ValueError('Prior audit input SHA changed')
        ancestry['prior_'+key]=item
    prior=read(ancestry['prior_plan']);plan=deepcopy(prior)
    plan.update(schema=SCHEMA,entry_version=VERSION,maximum_cells=10,maximum_child_seconds=90,
        output_byte_cap=64*1024**2,cells=deepcopy(prior['cells'][54:]),**ancestry,
        continuation_sources={SOURCE:binding(PROJECT_ROOT/SOURCE)},automatic_further_retry=False)
    resolve_cells(plan)
    out=writable_path(output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'plan.json',plan);write_json(out/'inputs.json',dict(plan=str(out/'plan.json')))
    write_json(out/'preflight.json',dict(schema=VERSION,files=files(out/'plan.json'),
        regenerated_original_ids=[r['candidate_id'] for r in plan['cells']],
        physics_steps=0,reference_generation=False,prior_outcomes_rewritten=False))
    return binding(out/'plan.json')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wall-seconds',type=float,required=True);a=p.parse_args(argv)
    result=generate(a.plan,a.output,wall_seconds=a.wall_seconds)
    print(json.dumps({k:result[k] for k in ('complete','planned_cells','executed_cells','generated_cells','exception')}))
    return 0 if result['complete'] and result['exception'] is None else 1


if __name__=='__main__':raise SystemExit(main())
