#!/usr/bin/env python3
"""Source-bound Phase2 expansion packets; no automatic submissions or follow-ups.

The original Phase2 launcher remains frozen. Submission/accounting schemas are
shared with it so every allocation is charged by the same durable ledger.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
_spec=importlib.util.spec_from_file_location('task10b_phase2_base',ROOT/'scripts/task10b_phase2.py')
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
from human_punching.phase2_budget import admit_reservation,budget_status
from human_punching.wbc.provenance import sha256_file,writable_path,write_json,runtime_record

SCHEMA='task10b_phase2_expansion_packet_v1'
MODES=('nonuniform-control','timing-pilot','nonuniform-references','interior-references','timing-references','frozen-control','timing-control','target-references','runtime-pack','grouped-pilot','grouped-finetune','expanded-evaluation','media','native-state-parity','shared-state-capture')
CONDITIONAL_MODES=('grouped-finetune','expanded-evaluation','media','shared-state-capture')
EVAL_KEYS=('declaration','manifest','route','checkpoint','training','predicate','mask')
CAP=3*1024**3


def read(path):return json.loads(base.local(path).read_text())


def bound(item):
    if not isinstance(item,dict) or set(item)!={'path','sha256'}:raise ValueError('Exact path/SHA256 binding required')
    path=base.local(item['path'])
    if not path.is_file() or sha256_file(path)!=item['sha256']:raise ValueError('Bound input changed')
    return path


def rounded_wall(seconds):
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0:
        raise ValueError('Finite positive allocation wall required')
    result=int(math.ceil(seconds/60))*60
    if result>14400:raise ValueError('Every rounded Slurm session is at most four hours')
    return result


def shallow_output(out):
    if len(os.fsencode(Path(out)/'run/cache/tmp'))>120:raise ValueError('Use shallow packet paths')


def snapshot_object(source,target,expected):
    """Copy immutable bytes once, then link archives to the copied pool object.

    Live inputs are never hardlinked. Failed copies remain as distinct objects;
    an index is published only after both source and copied bytes verify. Pool
    objects/indexes are append-only and owned by this project.
    """
    source=base.local(source);target=writable_path(target)
    if not source.is_file() or sha256_file(source)!=expected:raise ValueError('Live input changed before source snapshot')
    raw_pool=ROOT/'.cache/t10b2/source_objects';pool=writable_path(raw_pool)
    if pool!=raw_pool or raw_pool.is_symlink():raise ValueError('Source-object pool must use the resolved owned cache path')
    pool.mkdir(parents=True,exist_ok=True)
    raw_lock=ROOT/'.cache/t10b2/source_objects.lock';lock_path=writable_path(raw_lock)
    if raw_lock.is_symlink() or lock_path!=raw_lock:raise ValueError('Source-object lock cannot be redirected')
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        index=pool/(expected+'.json');created=False
        if index.exists():
            if index.is_symlink():raise ValueError('Source-object index may not be a symlink')
            entry=json.loads(index.read_text());obj=pool/entry['object']
            if (entry.get('schema')!='task10b_phase2_source_object_v1' or entry.get('sha256')!=expected
                    or not obj.resolve(strict=True).is_relative_to(pool) or obj.is_symlink()
                    or obj.stat().st_mode&0o222 or obj.stat().st_size!=entry.get('bytes') or sha256_file(obj)!=expected):
                raise ValueError('Immutable source-object pool entry changed')
        else:
            obj=pool/(expected+'_'+uuid.uuid4().hex)
            try:
                with source.open('rb') as src,obj.open('xb') as dst:shutil.copyfileobj(src,dst)
                if sha256_file(obj)!=expected or sha256_file(source)!=expected:raise ValueError('Input changed while copying source object')
                obj.chmod(0o444)
                entry=dict(schema='task10b_phase2_source_object_v1',object=obj.name,sha256=expected,bytes=obj.stat().st_size,
                    first_source=str(source),created_utc=base.now(),live_original_hardlinked=False)
                write_json(index,entry);index.chmod(0o444);created=True
            except Exception as error:
                if obj.exists():
                    obj.chmod(0o444)
                    write_json(pool/(obj.name+'.failed.json'),dict(schema='task10b_phase2_source_object_failure_v1',
                        expected_sha256=expected,source=str(source),retained_object=obj.name,exception=dict(type=type(error).__name__,message=str(error))))
                raise
        if sha256_file(source)!=expected:raise ValueError('Live input changed before archive link')
        target.parent.mkdir(parents=True,exist_ok=True)
        os.link(obj,target)
        return dict(object=str(obj),sha256=expected,new_physical_object=created,live_original_hardlinked=False)


def parity_implementation(path):
    receipt=read(path)
    if (receipt.get('schema')!='task10b_phase2_native_state_implementation_v1'
            or receipt.get('native_execution_performed') is not False or receipt.get('native_parity_qualified') is not False
            or receipt.get('compiled_state_spec')!=16383 or receipt.get('compiled_state_size')!=410):
        raise ValueError('Exact reviewed native-state implementation receipt required')
    source=bound(receipt['source'])
    if source!=ROOT/'src/human_punching/phase2_native_state.py':raise ValueError('Native-state implementation source differs')
    tests=receipt['focused_tests'];test_path=bound({k:tests[k] for k in ('path','sha256')})
    suites=ET.parse(test_path).getroot();suites=([suites] if suites.tag=='testsuite' else list(suites.iter('testsuite')))
    actual={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if actual['tests']<18 or any(actual[k] for k in ('failures','errors','skipped')) or any(tests.get(k)!=v for k,v in actual.items()):
        raise ValueError('Native-state CPU tests did not pass in full')
    bound(receipt['accepted_cpu_model_inventory'])
    for item in receipt.get('source_api_receipts',[]):
        source=Path(item['path']).resolve(strict=True)
        if (not any(source.is_relative_to(root.resolve()) for root in (ROOT/'.venv-wbc',ROOT.parent/'mjlab'))
                or not source.is_file() or sha256_file(source)!=item['sha256']):
            raise ValueError('Read-only native API provenance changed or escaped permitted sources')
    return receipt


def pre_capture_gate(mode,inputs):
    if mode=='shared-state-capture' and inputs.get('split')=='test':
        from human_punching.phase2_native_state import validate_test_bank_freeze
        return validate_test_bank_freeze(base.local(inputs['pre_test_freeze']))
    return None


def successful_route_freeze(proof_path):
    """Verify measured four-family validation before conditional authorization.

    Reads validation evidence only. No caller-provided successful boolean can
    unlock the additional ceiling; no final-test outcome is opened here.
    """
    from human_punching.phase2_qualification import PanelDeclaration,TrialBinding,qualify_panel,validate_resolved_audit
    from human_punching.phase2_domain import fingerprint
    from human_punching.wbc.splits import assign_trajectory_groups
    proof=read(proof_path)
    if (proof.get('schema')!='task10b_phase2_successful_route_freeze_v1'
            or set(proof.get('families',{}))!=set(base.FAMILIES)
            or proof.get('selection_splits')!=['train','validation'] or proof.get('test_used_for_selection') is not False):
        raise ValueError('Four-family measured train/validation freeze proof required')
    domain_path=bound(proof['domain']);domain=read(domain_path)
    groups=domain['selected_command_groups']
    splits=[dict(command_id=g['snapshot']['command_id'],replica_id=g['snapshot']['replica_id'],split=g['split']) for g in groups]
    if (domain.get('schema')!='Task10BPhase2Domain/v1' or len(splits)!=56
            or len({g['command_id'] for g in splits})!=56
            or {s:sum(g['split']==s for g in splits) for s in ('train','validation','test')}!={'train':32,'validation':8,'test':16}
            or any(g['command_id']==3307 or assign_trajectory_groups([g['command_id']])[g['command_id']]!=g['split'] for g in splits)
            or fingerprint(splits)!=proof.get('split_identity')):
        raise ValueError('Frozen domain and deterministic32/8/16 split identity differ')
    selected={}
    for family,item in proof['families'].items():
        route_path=bound(item);route=read(route_path)
        if (route.get('schema')!='task10b_phase2_evaluation_route_v1' or route.get('family')!=family or route.get('state')!='frozen'
                or route.get('selection_splits')!=['train','validation'] or route.get('test_used_for_selection') is not False
                or route.get('domain_sha256')!=proof['domain']['sha256'] or route.get('interpolation_used_for_training') is not False):
            raise ValueError('Recipe/domain route is not frozen from training and validation')
        evidence={k:read(bound(route['successful_validation'][k])) for k in ('qualification','trace_audit','launch','training')}
        report,audit,launch,training=(evidence[k] for k in ('qualification','trace_audit','launch','training'))
        declaration=PanelDeclaration(**{**report['declaration'],'bindings':tuple(TrialBinding(**b) for b in report['declaration']['bindings'])})
        if (declaration.split!='validation' or declaration.family!=family or report.get('passed') is not True
                or report.get('report_identity')!=fingerprint({k:v for k,v in report.items() if k!='report_identity'})
                or report.get('expected_attempts')!=96 or report.get('report_identity')!=qualify_panel(declaration,[r['source_outcome'] for r in report['trials']])['report_identity']
                or declaration.identity!=launch.get('declaration_identity') or not launch.get('slurm_job_id')
                or launch.get('native_execution_requested') is not True or launch.get('schema')!='task10b_phase2_expanded_native_launch_v1'
                or audit.get('schema')!='task10b_phase2_independent_trace_audit_v1' or audit.get('passed') is not True
                or audit.get('closed_protocol_launch_audited') is not True
                or (declaration.start_mode=='shared_jab_saved_state' and audit.get('shared_saved_state_initialization_audited') is not True)
                or validate_resolved_audit(audit,declaration.bindings).get('passed') is not True
                or training.get('weight_changed') is not True or training.get('fresh_transitions')!=declaration.fresh_transitions
                or training.get('checkpoint',{}).get('sha256')!=declaration.checkpoint_sha256
                or training.get('ancestry',{}).get('sha256')!=declaration.parent_checkpoint_sha256
                or training.get('ancestry',{}).get('fresh_optimizer') is not True
                or training.get('ancestry',{}).get('fresh_simulation') is not True
                or declaration.training_receipt_sha256!=route['successful_validation']['training']['sha256']
                or any(launch.get(k)!=route.get(k) or training.get(k)!=route.get(k) for k in ('recipe_sha256','domain_sha256','target_geometry_sha256'))
                or declaration.predicate_sha256!=route.get('predicate_sha256')
                or declaration.declared_mask_sha256!=route.get('declared_mask_sha256')):
            raise ValueError('Measured96-cycle learned validation/audit/ancestry evidence is incomplete')
        run=base.local(route['successful_validation']['launch']['path']).parent
        if Path(audit.get('run','')).resolve()!=run:raise ValueError('Validation receipts refer to different native runs')
        allowed_runs=[run]
        if (run/'continuation.json').exists():
            from human_punching.phase2_continuation import verified_run_ancestry
            allowed_runs=[Path(p) for p in verified_run_ancestry(run)['run_directories']]
        for panel in audit['panels']:
            trace=base.local(Path(panel['path'])/'raw_trace.npz')
            if not any(trace.is_relative_to(parent) for parent in allowed_runs) or sha256_file(trace)!=panel['raw_sha256']:
                raise ValueError('Audited validation trace changed or escaped verified continuation ancestry')
        if any(row.get('offline_rejection_kind')=='screen_or_mask_rejection'
                and row.get('declared_mask_sha256')!=declaration.declared_mask_sha256 for row in audit['trial_outcome_resolutions']):
            raise ValueError('Independent rejection used a different frozen mask')
        checkpoint=bound(training['checkpoint'])
        if sha256_file(checkpoint)!=declaration.checkpoint_sha256:raise ValueError('Learned checkpoint bytes changed')
        parent=bound(launch['release_inputs']['checkpoint'])
        if parent!=(ROOT/'outputs/four-punch-milestone'/family/'checkpoint.pt').resolve(strict=True) or sha256_file(parent)!=declaration.parent_checkpoint_sha256:
            raise ValueError('Learned ancestry does not match accepted Task09 weights')
        from human_punching.phase2_expanded_evaluation import validate_training_identity
        import torch
        validate_training_identity(declaration,training,torch.load(checkpoint,map_location='cpu',weights_only=True),
            declaration.parent_checkpoint_sha256,route=route)
        selected[family]=dict(route_sha256=sha256_file(route_path),recipe_sha256=route['recipe_sha256'],
            domain_sha256=route['domain_sha256'],target_geometry_sha256=route['target_geometry_sha256'],
            predicate_sha256=route['predicate_sha256'],declared_mask_sha256=route['declared_mask_sha256'],
            parent_checkpoint_sha256=declaration.parent_checkpoint_sha256,
            start_mode=declaration.start_mode,
            trained_timings_s=list(declaration.trained_timings_s),interpolation_timing_s=declaration.interpolation_timing_s)
    return dict(proof_sha256=sha256_file(base.local(proof_path)),families=selected,split_identity=proof['split_identity'])


def semantic_files(seeds,*,provenance_files=None):
    """Close existing input links, checking adjacent SHA bindings recursively.

    Only input path fields are traversed. Directory expansion is explicit for the
    accepted release and saved media run; environments/caches are never walked.
    """
    found=set();pending=[base.local(p) for p in seeds]
    ignored={'output','output_path','output_directory','run_directory','video','closest_still','final_still'}
    provenance_roles={'unpacked_reference_path','unpacked_catalog_path','source_reference_path'}
    named_inputs={'catalog','model','reference','checkpoint','predicate','domain','target_geometry','route','panel','run','source'}
    def visit(value):
        if isinstance(value,list):
            for v in value:visit(v)
        elif isinstance(value,dict):
            if set(value)=={'path','sha256'}:pending.append(bound(value));return
            if value.get('schema')=='task10b_phase2_evaluation_continuation_v1':
                parent=base.local(value['parent_run'])
                if not parent.is_dir() or not parent.is_relative_to(ROOT/'outputs'):raise ValueError('Owned continuation parent required')
                pending.extend(parent.glob('native_attempt_intent_*.json'))
            for key,item in value.items():
                if key in ignored:continue
                if key=='source_api_receipts':
                    # These identify reviewed dependency APIs, not native data
                    # or an alternate import root. Native parity separately
                    # rechecks their read-only hashes; packages remain in the
                    # owned pinned runtime instead of being copied as inputs.
                    continue
                if key in provenance_roles and isinstance(item,str):
                    p=base.local(item);expected=value.get(key[:-5]+'_sha256')
                    if (not p.is_file() or not isinstance(expected,str) or len(expected)!=64
                            or sha256_file(p)!=expected):raise ValueError('Preserved ancestor provenance changed: '+key)
                    if provenance_files is not None:
                        rel=str(p.relative_to(ROOT));entry=provenance_files.setdefault(rel,dict(sha256=expected,bytes=p.stat().st_size,roles=[]))
                        if entry['sha256']!=expected:raise ValueError('Conflicting ancestor evidence identities')
                        entry['roles']=sorted(set(entry['roles'])|{key})
                    continue
                path_named=(key in named_inputs and isinstance(item,str) and
                    (Path(item).is_absolute() or Path(item).suffix in ('.json','.npz','.xml','.pt')))
                if isinstance(item,str) and (key=='path' or key.endswith('_path') or path_named):
                    p=base.local(item)
                    if p.is_dir():
                        if not p.is_relative_to(ROOT/'outputs'):raise ValueError('Input directory cannot walk dependencies/environments')
                        pending.extend(f for f in p.rglob('*') if f.is_file() and f.suffix in ('.json','.npz','.pt','.xml'))
                        continue
                    if not p.is_file():raise ValueError('Semantic input path is not a file: '+key)
                    expected=value.get(key[:-5]+'_sha256') if key.endswith('_path') else value.get('sha256')
                    if expected is not None and sha256_file(p)!=expected:raise ValueError('Semantic input SHA differs: '+key)
                    pending.append(p)
                else:visit(item)
    while pending:
        path=pending.pop()
        if path in found:continue
        if not path.is_file():raise ValueError('Seed input must be a file')
        found.add(path)
        if path.suffix=='.json':visit(json.loads(path.read_text()))
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(found)}


def local_import_closure(paths):
    """Include static local imports, including lazy protocol/runtime helpers."""
    found=set(paths);pending=[ROOT/p for p in paths if p.endswith('.py')]
    while pending:
        path=pending.pop();tree=ast.parse(path.read_text(),filename=str(path))
        try:package=list(path.relative_to(ROOT/'src').parts[:-1])
        except ValueError:package=[]
        modules=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):modules.extend(alias.name.split('.') for alias in node.names if alias.name.startswith('human_punching.'))
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    if not package or node.level>len(package):continue
                    prefix=package[:len(package)-node.level+1]
                    module=prefix+(node.module.split('.') if node.module else [])
                elif node.module and node.module.startswith('human_punching'):module=node.module.split('.')
                else:continue
                modules.append(module)
                modules.extend(module+[alias.name] for alias in node.names if alias.name!='*')
        for parts in modules:
            candidate=ROOT/'src'/Path(*parts)
            for source in (candidate.with_suffix('.py'),candidate/'__init__.py'):
                if not source.is_file():continue
                source=base.local(source);rel=str(source.relative_to(ROOT))
                if rel not in found:found.add(rel);pending.append(source)
    return found


def original_control_catalog(inputs,*,catalog_override=None):
    """Admit only unchanged D0 controls; native execution still qualifies them."""
    import numpy as np
    family=inputs['family'];catalog=read(inputs['catalog']) if catalog_override is None else catalog_override;cells=catalog.get('cells',[])
    predicate=read('outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json')
    bundle=read('outputs/task10/release_baseline_02/manifest.json')
    clips={row['family']:row for row in bundle['clips']}
    if (catalog.get('schema')!='task10b_phase2_reference_catalog_v1' or catalog.get('complete') is not True
            or catalog.get('predicate_sha256')!=sha256_file(base.local('outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json'))
            or not 1<=len(cells)<=4 or len({row.get('family') for row in cells})!=len(cells)
            or sum(row.get('family')==family for row in cells)!=1):
        raise ValueError('Complete original-only D0 catalog with one requested-family control required')
    for row in cells:
        current=row.get('family')
        if current not in base.FAMILIES:raise ValueError('Unknown original-control family')
        clip=clips[current];schedule=predicate['schedule'][current]
        reference=base.local(ROOT/'outputs/four-punch-milestone'/current/'reference.npz')
        legacy=base.local(ROOT/'outputs/task10/release_baseline_02'/clip['legacy_runtime_path'])
        expected=dict(candidate_id=current+'_original',kind='original_control',requested_ttc_s=None,
            kinematic_admission=True,preserved_original_control=True,events_identity='unchanged_phase1_reviewed_schedule',
            duration_s=clip['source_duration_s'],episode_duration_s=clip['source_duration_s']+1.,
            phase_denominator_s=clip['source_duration_s']+1.,launch_tick=schedule['launch_actor_tick'],
            launch_time_s=schedule['launch_time_s'],outward_end_tick=schedule['outward_completion_tick'],
            outward_completion_time_s=schedule['outward_completion_time_s'])
        if any(row.get(key)!=value for key,value in expected.items()):raise ValueError('Original control clocks or identity changed')
        for key,path in (('reference',reference),('legacy_runtime',legacy)):
            if base.local(row[key+'_path'])!=path or row[key+'_sha256']!=sha256_file(path):
                raise ValueError('Original control reference bytes or selected file changed')
        with np.load(reference,allow_pickle=False) as arrays:
            if (row.get('target_w')!=arrays['head_center'].tolist() or row.get('head_center')!=arrays['head_center'].tolist()
                    or row.get('active_hand_index')!=int(arrays['active_hand_index'])
                    or row.get('guard_time_s')!=float(arrays['guard_hold_start'])):
                raise ValueError('Original control target, hand or guard timing changed')
    return catalog


def validate_spec(mode,inputs,wall_s,app_s,*,proof=None):
    if mode not in MODES:raise ValueError('Unknown expansion mode')
    wall=rounded_wall(wall_s)
    if wall!=wall_s or isinstance(app_s,bool) or not isinstance(app_s,(int,float)) or not math.isfinite(app_s) or not 120<app_s<=wall-30:
        raise ValueError('Application wall must be finite, >120s and at least30s below rounded allocation')
    if proof is not None and mode not in CONDITIONAL_MODES:raise ValueError('Conditional hours cover frozen fine-tuning/qualification only')
    family=None;seed=21021;caps=0
    if mode=='timing-references':
        if inputs or wall!=900 or app_s!=840:raise ValueError('Timing references use fixed baseline inputs and exactly900s/840s allocation/application caps')
    elif mode=='frozen-control':
        if set(inputs)!={'catalog','family'} or wall!=480 or app_s!=390:
            raise ValueError('Frozen controls require catalog/family and exactly480s/390s allocation/application caps')
        family=inputs['family'];original_control_catalog(inputs)
    elif mode=='timing-control':
        if set(inputs)!={'catalog','family'} or wall!=480 or app_s!=390:
            raise ValueError('Timing controls require catalog/family and exactly480s/390s allocation/application caps')
        from human_punching.phase2_timing_control import validate as validate_timing
        catalog,evidence=validate_timing(base.local(inputs['catalog']));family=inputs['family']
        original_control_catalog(inputs,catalog_override={**catalog,'cells':[r for r in catalog['cells'] if r['kind']=='original_control']})
    elif mode=='interior-references':
        if set(inputs)!={'plan'} or wall!=420 or app_s!=360:raise ValueError('Interior references require exact plan and420s/360s caps')
        from human_punching.phase2_interiors import validate_plan
        plan=read(inputs['plan']);validate_plan(plan)
        if plan['application_wall_seconds']!=300:raise ValueError('Interior child wall is exactly300s')
    elif mode=='nonuniform-references':
        if set(inputs)!={'plan'} or wall!=300 or app_s!=270:raise ValueError('Nonuniform references require exact plan and300s/270s caps')
        from human_punching.phase2_nonuniform import validate_plan
        plan=read(inputs['plan']);validate_plan(plan)
        if plan['application_wall_seconds']!=210:raise ValueError('Nonuniform child wall is exactly210s')
    elif mode=='nonuniform-control':
        if set(inputs)!={'plan'} or wall!=480 or app_s!=420:raise ValueError('Nonuniform controls require exact plan and480s/420s caps')
        from human_punching.phase2_nonuniform_control import validate_plan
        plan=read(inputs['plan']);validate_plan(plan,root=ROOT);family=plan['family']
    elif mode=='timing-pilot':
        if set(inputs)!={'plan'} or wall!=1200 or app_s!=1140:raise ValueError('Timing pilots require exact plan and1200s/1140s caps')
        from human_punching.phase2_timing_pilot import validate_plan
        plan=read(inputs['plan']);validate_plan(plan);family=plan['family'];caps=131072
        if plan['application_wall_seconds']!=1080:raise ValueError('Timing pilot child wall is exactly1080s')
    elif mode=='target-references':
        if set(inputs)!={'plan'}:raise ValueError('Exact target reference --plan required')
        plan=read(inputs['plan'])
        if plan.get('schema')=='task10b_phase2_shared_catalog_plan_v1':
            from human_punching.phase2_shared_catalog import validate_plan
        else:
            from human_punching.phase2_target_catalog import validate_plan
        validate_plan(plan);family=plan['family']
    elif mode=='runtime-pack':
        if set(inputs)!={'plan'}:raise ValueError('Exact runtime-pack --plan required')
        from human_punching.phase2_runtime_pack import VERSION,CAP as POOL_CAP
        plan=read(inputs['plan'])
        if (plan.get('schema')!='task10b_phase2_runtime_pack_plan_v1' or plan.get('storage_version')!=VERSION
                or type(plan.get('byte_cap')) is not int or not 0<plan['byte_cap']<=POOL_CAP
                or not 1<=len(plan.get('catalogs',[]))<=64 or not 1<=len(plan.get('recipes',[]))<=16):
            raise ValueError('Bounded exact runtime packing plan required')
        catalogs=[read(bound(binding)) for binding in plan['catalogs']]
        if (sum(len(cat['cells']) for cat in catalogs)>4096
                or any(cat.get('schema')!='task10b_phase2_reference_catalog_v1' for cat in catalogs)
                or any(row.get('family') not in base.FAMILIES or row.get('kind')=='original_control' for cat in catalogs for row in cat['cells'])):
            raise ValueError('Finite candidate runtime catalogs required; accepted controls remain byte-identical')
        for item in plan['recipes']:bound(item)
        family=None  # Offline construction only; no specialist actor is run.
    elif mode in ('grouped-pilot','grouped-finetune'):
        if set(inputs)!={'configuration'}:raise ValueError('Exact immutable training configuration required')
        configuration=read(inputs['configuration'])
        if configuration.get('schema')=='task10b_phase2_shared_training_configuration_v1':
            from human_punching.phase2_shared_learning import load_configuration
        else:
            from human_punching.phase2_grouped_learning import load_configuration
        path=base.local(inputs['configuration']);decl,_,budget,_,release=load_configuration(path,sha256_file(path))
        if release.resolve()!=ROOT/'outputs/four-punch-milestone':raise ValueError('Transferred specialist must use accepted Task09 release')
        family,seed,caps=decl.family,decl.seed,budget.transition_cap
        if budget.maximum_wall_s>app_s-30 or budget.shutdown_reserve_s<30:raise ValueError('Training deadline must leave watchdog flush margin')
        if mode=='grouped-pilot' and (seed!=21021 or wall>1800 or app_s>1680):
            raise ValueError('Initial pilot recipe is seed21021, <=1800s allocation and <=1680s application')
        if mode=='grouped-finetune' and proof is None:raise ValueError('Conditional fine-tuning requires measured route-freeze proof')
        if proof is not None:
            selected=proof['families'][family]
            if configuration.get('start_mode','family_guard')!=selected['start_mode']:raise ValueError('Conditional training start protocol changed')
            for key in ('recipe_sha256','domain_sha256','target_geometry_sha256','predicate_sha256','declared_mask_sha256','parent_checkpoint_sha256','interpolation_timing_s'):
                if getattr(decl,key)!=selected[key]:raise ValueError('Conditional fine-tuning recipe/domain/ancestry changed')
            if list(decl.trained_timings_s)!=selected['trained_timings_s']:raise ValueError('Conditional timing recipe changed')
    elif mode=='expanded-evaluation':
        if set(inputs) not in (set(EVAL_KEYS),set(EVAL_KEYS)|{'continuation'}):raise ValueError('Exact expanded evaluator inputs required')
        from human_punching.phase2_expanded_evaluation import load_declaration,validate_route,INITIALIZATION_VERSION
        declaration=load_declaration(inputs['declaration']);family,seed=declaration.family,declaration.seed
        route=base.local(inputs['route']);validate_route(declaration,read(route),route_sha256=sha256_file(route),supported_start_mode=declaration.start_mode)
        if proof is not None and proof['families'][family]['recipe_sha256']!=read(route)['recipe_sha256']:
            raise ValueError('Conditional evaluation recipe differs from frozen route')
        if proof is None and seed!=21021:raise ValueError('Three-seed qualification requires frozen route authorization')
        if proof is not None:
            selected=proof['families'][family];route_data=read(route)
            if (any(route_data.get(k)!=selected[k] for k in ('recipe_sha256','domain_sha256','target_geometry_sha256','predicate_sha256','declared_mask_sha256'))
                    or declaration.parent_checkpoint_sha256!=selected['parent_checkpoint_sha256']
                    or declaration.start_mode!=selected['start_mode']
                    or list(declaration.trained_timings_s)!=selected['trained_timings_s']
                    or declaration.interpolation_timing_s!=selected['interpolation_timing_s']):
                raise ValueError('Conditional evaluation changed the selected domain or ancestry')
        if inputs.get('continuation'):
            from human_punching.phase2_continuation import execution_identity,load_continuation
            version=INITIALIZATION_VERSION
            if declaration.start_mode=='shared_jab_saved_state':
                from human_punching.phase2_shared_evaluation import VERSION
                version=VERSION
            identity=execution_identity(declaration,{k:base.local(inputs[k]) for k in EVAL_KEYS},
                release=ROOT/'outputs/four-punch-milestone',protocol_version=version,maximum_worlds=16)
            load_continuation(base.local(inputs['continuation']),identity,declaration)
    elif mode=='media':
        if set(inputs) not in ({'run','panel'},{'run'}):raise ValueError('Exact saved run or legacy run/panel required')
        run=base.local(inputs['run']);launch=read(run/'launch.json')
        expanded=launch.get('schema') in ('task10b_phase2_expanded_native_launch_v1','task10b_phase2_grouped_validation_launch_v1')
        if expanded!=(set(inputs)=={'run'}):raise ValueError('Expanded media uses its immutable declaration and all ledgers; legacy media requires its panel')
        declaration={}
        if 'family' not in launch:
            file=run/('declaration.json' if (run/'declaration.json').exists() else 'physical_declaration.json')
            declaration=read(file)
        family=launch.get('family',declaration.get('family'));seed=launch.get('seed',declaration.get('seed',21021))
        if not expanded:
            panel=base.local(inputs['panel'])
            if not panel.is_relative_to(run):raise ValueError('Media panel must belong to its saved run')
        if proof is not None and launch.get('recipe_sha256')!=proof['families'][family]['recipe_sha256']:
            raise ValueError('Conditional media must belong to selected frozen recipe')
    elif mode=='native-state-parity':
        if set(inputs)!={'catalog','family','implementation_receipt'}:raise ValueError('Qualified native-state-parity implementation receipt required')
        family=inputs['family'];parity_implementation(inputs['implementation_receipt'])
        catalog=read(inputs['catalog'])
        if len(catalog.get('cells',[]))!=1 or catalog['cells'][0].get('family')!=family or catalog['cells'][0].get('kind')!='original_control':
            raise ValueError('Native parity requires exactly one original-control reference for the requested family')
        if app_s>900:raise ValueError('Native-state parity is limited to900s')
    else:
        core={'catalog','domain','split','geometry','implementation_receipt'}
        split=inputs.get('split')
        if split not in ('train','validation','test') or set(inputs)!=(core|({'pre_test_freeze'} if split=='test' else set())):
            raise ValueError('Exact four-world capture inputs and pre-test freeze for test split required')
        freeze=None
        if split=='test':
            # This mechanical validation-only gate precedes opening/hash-reading
            # the test domain, geometry bank or original native reference.
            from human_punching.phase2_native_state import validate_test_bank_freeze
            freeze=validate_test_bank_freeze(base.local(inputs['pre_test_freeze']))
        if proof is not None and (split!='test' or not any(
                r['start_mode']=='shared_jab_saved_state' and r['recipe_sha256']==freeze['recipe_sha256']
                and r['domain_sha256']==freeze['domain_sha256'] for r in proof['families'].values())):
            raise ValueError('Conditional capture is only qualification of the same frozen shared recipe')
        parity_implementation(inputs['implementation_receipt']);family='jab'
        catalog=read(inputs['catalog'])
        if len(catalog.get('cells',[]))!=1 or catalog['cells'][0].get('family')!='jab' or catalog['cells'][0].get('kind')!='original_control':
            raise ValueError('Shared state capture requires one accepted original jab reference')
        if freeze and sha256_file(base.local(inputs['domain']))!=freeze['domain_sha256']:raise ValueError('Final bank capture domain changed')
        if app_s>900:raise ValueError('Four-world native state capture is limited to900s')
    if (family not in base.FAMILIES and not (mode in ('runtime-pack','timing-references','interior-references','nonuniform-references') and family is None)) or seed not in (21021,21022,21023):
        raise ValueError('One known specialist and declared seed required; runtime packing executes no actor')
    scope=dict(family=family,seed=seed,fresh_transition_cap=caps)
    if mode=='timing-control':scope['timing_control_evidence']=evidence
    if mode in ('timing-control','nonuniform-control','timing-pilot'):scope['d0_clock_versions']=dict(evaluator='task10b_phase2_d0_postprocessing_clock_v2',independent_audit='task10b_phase2_independent_clock_v2')
    if mode=='timing-control':
        from human_punching.phase2_timing_control import bound as timing_bound,binding as timing_binding
        meta=catalog['timing_control_projection'];probe=timing_bound(meta['alias_probe'])
        scope['timing_control_proof_files']=[timing_binding(probe),timing_binding(probe.parent/'nonce.json'),
            timing_binding(timing_bound(meta['generation_execution']))]
    return scope


def required_inputs(record,*,provenance_files=None):
    baseline=dict(mode='frozen' if record['mode'] in ('frozen-control','timing-control') else 'references',
        bundle=record['bundle'],release=record['release'],predicate=record['predicate'])
    if record['mode'] in ('frozen-control','timing-control'):baseline['catalog']=str(base.local(record['inputs']['catalog']).relative_to(ROOT))
    paths=base.required_inputs(baseline)
    # Transitive local implementation closure is cheap and deliberately explicit:
    # future unrelated modules do not invalidate an already frozen packet.
    modules=('phase2_learning','phase2_domain','phase2_cache','phase2_deadline','phase2_launch','phase2_phases',
        'phase2_targets','phase2_target_reference','phase2_target_catalog','phase2_qualification','phase2_expanded_evaluation',
        'phase2_grouped_learning','phase2_validation_callback','phase2_native_state','phase2_shared_reference','phase2_shared_target_reference')
    modules+=('phase2_sampling','phase2_shared_catalog','phase2_shared_evaluation','phase2_shared_learning','phase2_shared_validation_callback','phase2_runtime_pack','phase2_continuation')
    paths.update('src/human_punching/'+name+'.py' for name in modules if (ROOT/'src/human_punching'/f'{name}.py').exists())
    if record['mode'] in ('timing-control','nonuniform-control','timing-pilot'):paths.update(('src/human_punching/phase2_d0_clock_v2.py','scripts/task10b_phase2_audit_clock_v2.py'))
    paths.update(('scripts/task10b_phase2_expand.py','scripts/task10b_phase2_expand.sbatch',
        'scripts/task10b_phase2_target_references.py','scripts/task10b_phase2_media.py','scripts/task10b_phase2_audit.py'))
    if (ROOT/'scripts/task10b_phase2_shared_references.py').exists():paths.add('scripts/task10b_phase2_shared_references.py')
    for script in ('task10b_phase2_runtime_pack.py','task10b_phase2_expanded_media.py'):
        if (ROOT/'scripts'/script).exists():paths.add('scripts/'+script)
    media_launch=read(base.local(record['inputs']['run'])/'launch.json') if record['mode']=='media' and set(record['inputs'])=={'run','panel'} else {}
    d0_media=media_launch.get('schema')=='task10b_phase2_original_route_launch_v1'
    pilot_media=media_launch.get('schema')=='task10b_phase2_selected_d0_timing_pilot_v1'
    if d0_media or pilot_media:paths.add('scripts/task10b_phase2_audit_clock_v2.py')
    target_exact=record['mode']=='target-references' and read(record['inputs']['plan']).get('schema')=='task10b_phase2_target_catalog_plan_v1'
    seeds=[]
    for key,value in ({} if d0_media or pilot_media or target_exact or record['mode'] in ('interior-references','nonuniform-references','nonuniform-control','timing-pilot') else record['inputs']).items():
        if key in ('family','split'):continue
        p=base.local(value)
        if p.is_dir():
            if key!='run':raise ValueError('Only explicit saved media run directory may expand')
            seeds.extend(f for f in p.rglob('*') if f.is_file() and f.suffix in ('.json','.npz','.pt','.xml'))
        else:seeds.append(p)
    if record.get('successful_route_freeze'):seeds.append(base.local(record['successful_route_freeze']))
    if d0_media:
        if media_launch.get('nonuniform_control_version')=='task10b_phase2_six_nonuniform_frozen_control_v1':
            from human_punching.phase2_nonuniform_control import media_files
            closure=media_files(ROOT,record['inputs'])
        else:
            from human_punching.phase2_d0_media_inputs import files as d0_media_files
            closure=d0_media_files(ROOT,record['inputs'],record['release'],record['bundle'],record['predicate'],original_control_catalog)
    elif pilot_media:
        from human_punching.phase2_timing_pilot_media_inputs import files as pilot_media_files
        proof=bound(record['runtime_namespace_proof']) if 'runtime_namespace_proof' in record else None
        closure=pilot_media_files(ROOT,record['inputs'],record['release'],record['bundle'],record['predicate'],namespace_proof=proof)
    elif target_exact:
        from human_punching.phase2_target_inputs import files as target_files
        closure=target_files(base.local(record['inputs']['plan']))
    elif record['mode']=='nonuniform-control':
        from human_punching.phase2_nonuniform_control import files as control_files
        closure=control_files(base.local(record['inputs']['plan']),root=ROOT)
    elif record['mode']=='timing-pilot':
        from human_punching.phase2_timing_pilot import validate_plan
        plan_path=base.local(record['inputs']['plan']);plan=read(plan_path);_,rows,_=validate_plan(plan)
        closure={str(plan_path.relative_to(ROOT)):sha256_file(plan_path)}
        def exact_pilot(value):
            if isinstance(value,dict):
                if set(value)=={'path','sha256'}:
                    path=bound(value);closure[str(path.relative_to(ROOT))]=value['sha256']
                else:
                    for child in value.values():exact_pilot(child)
            elif isinstance(value,list):
                for child in value:exact_pilot(child)
        exact_pilot(plan)
        for name in ('route_receipt','video_receipt'):exact_pilot(read(bound(plan[name])))
        for row in rows:
            for name in ('reference','legacy_runtime'):exact_pilot(dict(path=row[name+'_path'],sha256=row[name+'_sha256']))
            if row.get('diagnostic_screen_path'):
                path=base.local(row['diagnostic_screen_path']);closure[str(path.relative_to(ROOT))]=sha256_file(path)
    elif record['mode'] in ('interior-references','nonuniform-references'):
        if record['mode']=='interior-references':
            from human_punching.phase2_interiors import validate_plan
        else:
            from human_punching.phase2_nonuniform import validate_plan
        from human_punching.phase2_timing_control import validate as validate_timing,bound as timing_bound
        plan=read(record['inputs']['plan']);validate_plan(plan)
        closure=semantic_files([plan['parent_catalog']['path']],provenance_files=provenance_files)
        def exact_inputs(value):
            if isinstance(value,dict):
                if set(value)=={'path','sha256'}:
                    path=bound(value);closure[str(path.relative_to(ROOT))]=value['sha256']
                else:
                    for child in value.values():exact_inputs(child)
        for key in ('parent_catalog','domain','predicate','release','bundle','generator_sources','source_ancestry'):exact_inputs(plan[key])
        if record['mode']=='nonuniform-references':
            exact_inputs(plan['assessment']);assessment=read(bound(plan['assessment']))['measured']
            for prefix in ('cells','error_localization'):exact_inputs(dict(path=assessment[prefix+'_path'],sha256=assessment[prefix+'_sha256']))
        path=base.local(record['inputs']['plan']);closure[str(path.relative_to(ROOT))]=sha256_file(path)
        catalog,evidence=validate_timing(bound(plan['parent_catalog']))
        for cell in evidence:
            for item in cell['files'].values():exact_inputs(item)
        for item in catalog['timing_control_projection'].values():
            if isinstance(item,dict) and set(item)=={'relative_name','sha256'}:
                path=timing_bound(item);closure[str(path.relative_to(ROOT))]=item['sha256']
    else:closure=semantic_files(seeds,provenance_files=provenance_files)
    if record['mode']=='timing-control':
        for cell in record['timing_control_evidence']:
            for item in cell['files'].values():
                path=bound(item);paths.add(str(path.relative_to(ROOT)))
        for item in record['timing_control_proof_files']:
            path=bound(item);paths.add(str(path.relative_to(ROOT)))
    if record['mode']=='interior-references':paths.add('scripts/task10b_phase2_interiors.py')
    if record['mode']=='nonuniform-references':paths.add('scripts/task10b_phase2_nonuniform.py')
    if 'runtime_namespace_proof' in record:
        from human_punching.phase2_runtime_namespace import evidence
        meta=evidence(ROOT,bound(record['runtime_namespace_proof']))
        if meta!=record.get('runtime_namespace_evidence'):raise ValueError('Runtime namespace evidence changed')
        paths.update(item['path'] for item in [meta['proof'],meta['nonce'],meta['source_verifier_ancestry'],*meta['alias_bindings']])
        paths.add('src/human_punching/phase2_runtime_namespace.py')
    paths=local_import_closure(paths|set(closure))
    closure.update({p:sha256_file(base.local(p)) for p in paths})
    return closure


def submission_command(r,out):
    command=base.submission_command(r,out)
    command[command.index(str(ROOT/'scripts/task10b_phase2.sbatch'))]=str(ROOT/'scripts/task10b_phase2_expand.sbatch')
    return command


def submission_comparison(actual,expected):
    """Canonicalize exactly five owned path slots, preserving every other byte.

    Login and allocated nodes can expose the same NFS tree under different
    symlink spellings. Only the existing working root, exact batch script and
    packet, plus existing log parents may use such aliases. Log basenames and
    every resource/partition/job/order argument remain immutable.
    """
    result=dict(schema='task10b_phase2_submission_path_comparison_v1',passed=False,
        actual_root=str(ROOT),resolved_root=str(ROOT.resolve(strict=True)),expected_command=expected,actual_command=actual,
        canonical_expected=None,canonical_actual=None,path_slots=[],differences=[])
    try:
        if (not isinstance(actual,list) or not isinstance(expected,list) or len(actual)!=len(expected)
                or not all(isinstance(item,str) for item in actual+expected)):
            raise ValueError('Submission argument count/type changed')
        root=ROOT.resolve(strict=True);slots={}
        for prefix,kind in (('--chdir=','directory'),('--output=','stdout'),('--error=','stderr')):
            found=[index for index,arg in enumerate(expected) if arg.startswith(prefix)]
            if len(found)!=1:raise ValueError('Expected submission path inventory differs')
            slots[found[0]]=(prefix,kind)
        slots[len(expected)-2]=('', 'script');slots[len(expected)-1]=('', 'packet')
        if len(slots)!=5:raise ValueError('Expected exactly five fixed submission paths')
        canon_actual=list(actual);canon_expected=list(expected)
        for index,(prefix,kind) in sorted(slots.items()):
            item=dict(index=index,kind=kind,expected_raw=expected[index],actual_raw=actual[index])
            result['path_slots'].append(item)
            for label,command,canonical in (('expected',expected,canon_expected),('actual',actual,canon_actual)):
                arg=command[index]
                if prefix and not arg.startswith(prefix):raise ValueError('Submission path flag/order changed at '+str(index))
                value=arg[len(prefix):];path=Path(value)
                if not path.is_absolute():raise ValueError('Absolute owned submission paths required')
                if kind in ('stdout','stderr'):
                    basename='%j.out' if kind=='stdout' else '%j.err'
                    if path.name!=basename:raise ValueError('Submission log basename changed')
                    parent=path.parent.resolve(strict=True)
                    if not parent.is_dir() or not parent.is_relative_to(root):raise ValueError('Submission log parent escaped owned root')
                    resolved=parent/basename
                else:
                    resolved=path.resolve(strict=True)
                    if not resolved.is_relative_to(root):raise ValueError('Submission path escaped owned root')
                    if kind=='directory' and(resolved!=root or not resolved.is_dir()):raise ValueError('Submission working root changed')
                    if kind in ('script','packet') and not resolved.is_file():raise ValueError('Submission script/packet is not an existing file')
                canonical[index]=prefix+str(resolved);item[label+'_canonical']=str(resolved)
        result.update(canonical_expected=canon_expected,canonical_actual=canon_actual)
        for index,(left,right) in enumerate(zip(canon_actual,canon_expected)):
            if left!=right:result['differences'].append(dict(index=index,kind=slots.get(index,('', 'immutable_argument'))[1],
                expected=expected[index],actual=actual[index],expected_canonical=right,actual_canonical=left))
        result['passed']=not result['differences']
    except (OSError,ValueError,RuntimeError) as error:
        result['exception']=dict(type=type(error).__name__,message=str(error))
        result['differences'].append(dict(kind='invalid_path_or_argument_inventory',reason=str(error)))
    return result


def native_command(r,remaining_wall_s=None):
    limit=r['application_wall_seconds']-30
    if remaining_wall_s is not None:
        if not math.isfinite(remaining_wall_s) or not 120<remaining_wall_s<=limit:raise ValueError('Remaining child wall is exhausted or expanded')
        limit=remaining_wall_s
    p=[str(base.PYTHON),'-B'];i=r['inputs'];out=str(ROOT/r['output']);wall=str(limit)
    if r['mode']=='timing-references':return p+[str(ROOT/'scripts/task10b_phase2_references.py'),
        '--bundle',str(ROOT/r['bundle']),'--release',str(ROOT/r['release']),'--output',out,'--predicate',str(ROOT/r['predicate'])]
    if r['mode'] in ('frozen-control','timing-control'):
        module='human_punching.phase2_d0_clock_v2' if r['mode']=='timing-control' else 'human_punching.phase2_learning'
        return p+['-m',module,'--mode','frozen','--family',r['family'],
            '--catalog',str(base.local(i['catalog'])),'--output',out,'--seed','21021','--max-updates','0',
            '--wall-seconds',str(min(limit,360.)),'--release',str(ROOT/r['release']),'--predicate',str(ROOT/r['predicate'])]
    if r['mode']=='interior-references':
        if limit<300:raise RuntimeError('Setup consumed the fixed interior child deadline margin')
        return p+[str(ROOT/'scripts/task10b_phase2_interiors.py'),'--plan',str(base.local(i['plan'])),'--output',out,'--wall-seconds','300.0']
    if r['mode']=='nonuniform-references':
        if limit<210:raise RuntimeError('Setup consumed the fixed nonuniform child deadline margin')
        return p+[str(ROOT/'scripts/task10b_phase2_nonuniform.py'),'--plan',str(base.local(i['plan'])),'--output',out,'--wall-seconds','210.0']
    if r['mode']=='nonuniform-control':
        if limit<360:raise RuntimeError('Setup consumed the fixed nonuniform control child margin')
        return p+['-m','human_punching.phase2_nonuniform_control','--plan',str(base.local(i['plan'])),'--output',out,'--wall-seconds','360.0']
    if r['mode']=='timing-pilot':
        if limit<1080:raise RuntimeError('Setup consumed the fixed timing pilot child margin')
        return p+[str(ROOT/'scripts/task10b_phase2_timing_pilot.py'),'--plan',str(base.local(i['plan'])),'--output',out]
    if r['mode']=='target-references':
        script='task10b_phase2_shared_references.py' if read(i['plan']).get('schema')=='task10b_phase2_shared_catalog_plan_v1' else 'task10b_phase2_target_references.py'
        return p+[str(ROOT/'scripts'/script),'--plan',str(base.local(i['plan'])),'--output',out,'--wall-seconds',wall]
    if r['mode']=='runtime-pack':return p+[str(ROOT/'scripts/task10b_phase2_runtime_pack.py'),'--plan',str(base.local(i['plan'])),'--output',out,'--wall-seconds',wall]
    if r['mode'] in ('grouped-pilot','grouped-finetune'):
        config=base.local(i['configuration'])
        module='human_punching.phase2_shared_learning' if read(config).get('schema')=='task10b_phase2_shared_training_configuration_v1' else 'human_punching.phase2_grouped_learning'
        return p+['-m',module,'--configuration',str(config),'--configuration-sha256',r['files'][str(config.relative_to(ROOT))],'--output',out]
    if r['mode']=='expanded-evaluation':
        start_mode=read(i['declaration']).get('start_mode','family_guard')
        if start_mode not in ('family_guard','shared_jab_saved_state'):raise ValueError('Unknown expanded start protocol')
        module='human_punching.phase2_shared_evaluation' if start_mode=='shared_jab_saved_state' else 'human_punching.phase2_expanded_evaluation'
        command=p+['-m',module,*[part for k in EVAL_KEYS for part in ('--'+k,str(base.local(i[k])))],
            '--release',str(ROOT/r['release']),'--output',out,'--wall-seconds',wall,'--maximum-worlds','16']
        if i.get('continuation'):command+=['--continuation',str(base.local(i['continuation']))]
        return command
    if r['mode']=='media':
        if set(i)=={'run'}:
            available=CAP-sum(base.local(path).stat().st_size for path in r['files'])-64*1024**2
            if available<=0:raise ValueError('No packet product budget remains for media')
            return p+[str(ROOT/'scripts/task10b_phase2_expanded_media.py'),'--run',str(base.local(i['run'])),'--output',out,
                '--wall-seconds',wall,'--product-byte-cap',str(min(2*1024**3,available))]
        return p+[str(ROOT/'scripts/task10b_phase2_media.py'),'--run',str(base.local(i['run'])),'--panel',str(base.local(i['panel'])),'--output',out,'--wall-seconds',wall]
    if r['mode']=='native-state-parity':return p+['-m','human_punching.phase2_native_state','--catalog',str(base.local(i['catalog'])),'--family',r['family'],'--release',str(ROOT/r['release']),'--output',out,'--wall-seconds',wall]
    if r['mode']=='shared-state-capture':
        command=p+['-m','human_punching.phase2_native_state','--catalog',str(base.local(i['catalog'])),'--family','jab',
            '--release',str(ROOT/r['release']),'--output',out,'--wall-seconds',wall,
            '--capture-bank-domain',str(base.local(i['domain'])),'--capture-bank-split',i['split'],'--capture-bank-geometry',str(base.local(i['geometry']))]
        if i['split']=='test':command+=['--pre-test-freeze',str(base.local(i['pre_test_freeze']))]
        return command
    raise ValueError('No authorized command for mode')


def verify(packet):
    path=base.local(packet);out=base.packet_directory(path.parent);r=read(path)
    if r.get('schema')!=SCHEMA or path.name!='packet.json' or writable_path(ROOT/r['output'])!=out/'run':raise ValueError('Expansion packet identity differs')
    pre_capture_gate(r['mode'],r['inputs'])
    proof=successful_route_freeze(r['successful_route_freeze']) if r.get('successful_route_freeze') else None
    scope=validate_spec(r['mode'],r['inputs'],r['allocation_wall_seconds'],r['application_wall_seconds'],proof=proof)
    expected=dict(phase=2,gpus=1,products_bytes_cap=CAP,automatic_followup=False,promotion=False,g1_actor_execution=False,
        combined_scene=False,training=r['mode'] in ('grouped-pilot','grouped-finetune','timing-pilot'),num_envs=128,rollout_steps=64,
        bundle='outputs/task10/release_baseline_02',release='outputs/four-punch-milestone',
        predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json',**scope)
    if any(r.get(k)!=v for k,v in expected.items()) or r.get('verified_route_freeze')!=proof:raise ValueError('Scope/resource/route binding differs')
    if r.get('partition') not in ('mit_normal_gpu','mit_preemptable') or r.get('excluded_nodes')!=['node3006']:raise ValueError('Partition/node contract differs')
    if r.get('runtime')!=runtime_record() or not r['runtime']['declared_pins_match'] or r.get('authority_sha256')!=sha256_file(base.PREP/'authorization.json'):raise ValueError('Runtime/authority changed')
    if r.get('interpreter_sha256')!=sha256_file(base.PYTHON):raise ValueError('Owned interpreter changed')
    if 'runtime_namespace_proof' not in r and not base.PYTHON.resolve(strict=True).is_relative_to(ROOT):raise ValueError('Owned interpreter changed')
    provenance={};expected_files=required_inputs(r,provenance_files=provenance)
    if r.get('files')!=expected_files or r.get('provenance_files')!=provenance:raise ValueError('Input closure/provenance differs')
    comparison=submission_comparison(r.get('submission_command'),submission_command(r,out))
    if not comparison['passed']:raise ValueError('Submission command differs from bounded canonical packet: '+json.dumps(comparison['differences'],sort_keys=True))
    for rel,digest in r['files'].items():
        archive=(out/'source_snapshot'/rel).resolve(strict=True)
        if not archive.is_relative_to(out) or sha256_file(archive)!=digest:raise ValueError('Archived source/input changed')
    verify_packet_runtime(r)
    return r


def verify_packet_runtime(record):
    if 'runtime_namespace_proof' not in record:
        if 'runtime_namespace_evidence' in record:raise ValueError('Runtime namespace evidence requires explicit proof')
        return base.verify_runtime()
    from human_punching.phase2_runtime_namespace import verify_runtime
    result=verify_runtime(ROOT,base.PREP,bound(record['runtime_namespace_proof']),python_path=base.PYTHON)
    if result['proof']!=record.get('runtime_namespace_evidence'):raise ValueError('Runtime namespace proof identity changed')
    return result


def prepare(output,mode,inputs,*,allocation_wall_seconds,application_wall_seconds,partition='mit_normal_gpu',successful_route_freeze_path=None,runtime_namespace_proof=None):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='wbc':raise ValueError('Task10B requires wbc branch')
    out=base.packet_directory(writable_path(output))
    shallow_output(out)
    inputs={k:(v if k in ('family','split') else str(base.local(v))) for k,v in inputs.items()}
    pre_capture_gate(mode,inputs)
    proof_path=str(base.local(successful_route_freeze_path)) if successful_route_freeze_path else None
    proof=successful_route_freeze(proof_path) if proof_path else None
    wall=rounded_wall(allocation_wall_seconds)
    scope=validate_spec(mode,inputs,wall,application_wall_seconds,proof=proof)
    r=dict(schema=SCHEMA,phase=2,mode=mode,inputs=inputs,prepared_utc=base.now(),
        allocation_wall_seconds=wall,application_wall_seconds=application_wall_seconds,gpus=1,products_bytes_cap=CAP,
        training=mode in ('grouped-pilot','grouped-finetune','timing-pilot'),num_envs=128,rollout_steps=64,
        automatic_followup=False,promotion=False,g1_actor_execution=False,combined_scene=False,
        partition=partition,excluded_nodes=['node3006'],runtime=runtime_record(),interpreter_sha256=sha256_file(base.PYTHON),
        authority_sha256=sha256_file(base.PREP/'authorization.json'),output=str((out/'run').relative_to(ROOT)),
        bundle='outputs/task10/release_baseline_02',release='outputs/four-punch-milestone',
        predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json',
        successful_route_freeze=proof_path,verified_route_freeze=proof,**scope)
    if runtime_namespace_proof is not None:
        from human_punching.phase2_runtime_namespace import evidence
        path=base.local(runtime_namespace_proof);r['runtime_namespace_proof']=dict(path=str(path.relative_to(ROOT)),sha256=sha256_file(path))
        r['runtime_namespace_evidence']=evidence(ROOT,path)
    if partition not in ('mit_normal_gpu','mit_preemptable'):raise ValueError('Unauthorized partition')
    provenance={};r['files']=required_inputs(r,provenance_files=provenance);r['provenance_files']=provenance
    if sum(base.local(p).stat().st_size for p in r['files'])>=CAP:raise ValueError('Snapshot closure already exceeds packet products cap')
    out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir()
    pool_receipts={}
    for rel,digest in r['files'].items():
        target=out/'source_snapshot'/rel;pool_receipts[rel]=snapshot_object(base.local(rel),target,digest)
        if sha256_file(target)!=digest:raise ValueError('Input changed while copying')
    write_json(out/'source_objects.json',dict(schema='task10b_phase2_packet_source_objects_v1',objects=pool_receipts,
        logical_snapshot_bytes=sum(base.local(p).stat().st_size for p in r['files']),
        preserved_source_evidence_logical_bytes=sum(item['bytes'] for item in provenance.values()),
        preserved_source_evidence_paths=list(provenance),provenance_in_runtime_input_copies=False,
        new_pool_payload_bytes=sum(base.local(r['object']).stat().st_size for r in pool_receipts.values() if r['new_physical_object']),
        live_originals_hardlinked=False,logical_packet_products_cap=CAP))
    command=['du','-B1','--summarize',str(ROOT/'.cache/t10b2/source_objects'),str(out)]
    try:
        measured=subprocess.run(command,capture_output=True,text=True,timeout=30,check=False)
        storage=dict(command=command,returncode=measured.returncode,stdout=measured.stdout,stderr=measured.stderr,
            scope='One du invocation counts source-pool/packet hardlinks once; logical packet bytes still control3GiB watchdog')
    except (OSError,subprocess.TimeoutExpired) as error:
        storage=dict(command=command,physical_storage_measurement_available=False,exception=dict(type=type(error).__name__,message=str(error)))
    write_json(out/'source_storage.json',storage)
    r['submission_command']=submission_command(r,out);write_json(out/'packet.json',r);verify(out/'packet.json')
    return dict(packet=str(out/'packet.json'),sha256=sha256_file(out/'packet.json'),inputs=len(r['files']),reserved_seconds_if_submitted=wall)


def submit(packet):
    path=base.local(packet);r=verify(path)
    with (base.PREP/'allocation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (path.parent/'submission_intent.json').exists():raise ValueError('Packet already has durable submission intent')
        outstanding=0
        for intent in (ROOT/'outputs/task10b').glob('p2*/submission_intent.json'):
            if intent.with_name('terminal_accounting.json').exists() or intent.with_name('submission_rejected.json').exists():continue
            outstanding+=('--partition='+r['partition']) in read(intent)['command']
        if outstanding>=(2 if r['partition']=='mit_normal_gpu' else 4):raise ValueError('Concurrent partition GPU limit reached')
        conditional=r['verified_route_freeze'] is not None
        budget=admit_reservation(base.ledger(),r['allocation_wall_seconds'],evaluation_reserve_seconds=4500 if r['training'] else 0,successful_route_frozen=conditional)
        write_json(path.parent/'submission_intent.json',dict(utc=base.now(),packet_sha256=sha256_file(path),reserved_gpu_seconds=r['allocation_wall_seconds'],command=r['submission_command'],budget_before=budget))
        response=subprocess.run(r['submission_command'],cwd=ROOT,capture_output=True,text=True,check=False,timeout=30)
        receipt=dict(schema='task10b_phase2_submission_v1',utc=base.now(),packet_sha256=sha256_file(path),command=r['submission_command'],
            returncode=response.returncode,stdout=response.stdout,stderr=response.stderr,budget_before=budget,reserved_gpu_seconds=r['allocation_wall_seconds'])
        job=response.stdout.strip().split(';')[0]
        if response.returncode or not job.isdigit():
            name='submission_uncertain.json' if any(c.isdigit() for c in response.stdout) or not response.returncode else 'submission_rejected.json'
            write_json(path.parent/name,receipt);raise RuntimeError('Submission failed or uncertain; preserve intent until scheduler reconciliation')
        receipt['job_id']=job;write_json(path.parent/'submission.json',receipt)
    return receipt


def execute(packet):
    path=base.local(packet)
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise RuntimeError('Explicit Slurm GPU allocation required')
    started=time.monotonic();process=None;code=2
    receipt=dict(schema='task10b_phase2_execution_v1',started_utc=base.now(),job_id=os.environ['SLURM_JOB_ID'],
        packet_sha256=sha256_file(path),status='failed',native_processes_started=0,budget_stop=None,
        runtime_path_context=dict(root=str(ROOT),resolved_root=str(ROOT.resolve(strict=True)),
            base_root=str(base.ROOT),module_file=str(Path(__file__).absolute()),
            interpreter=str(base.PYTHON),resolved_interpreter=str(base.PYTHON.resolve(strict=True)),
            packet=str(path),working_directory=str(Path.cwd()),shell_pwd=os.environ.get('PWD')))
    try:
        preview=read(path)
        receipt['runtime_namespace_proof']=preview.get('runtime_namespace_proof')
        receipt['submission_path_comparison']=submission_comparison(preview.get('submission_command'),submission_command(preview,path.parent))
    except Exception as error:
        receipt['submission_path_comparison']=dict(passed=False,exception=dict(type=type(error).__name__,message=str(error)))
    write_json(path.parent/'execution_started.json',receipt)
    try:
        r=verify(path);submission=read(path.parent/'submission.json')
        if submission['job_id']!=os.environ['SLURM_JOB_ID'] or submission['packet_sha256']!=sha256_file(path):raise ValueError('Native allocation does not match submitted packet')
        health=base.old_helper('task10b_phase0').cuda_health_receipt(path.parent/'cuda_health.json')
        if health['status']!='passed':raise RuntimeError('Allocated CUDA health failed')
        work=path.parent/'work';work.mkdir(exist_ok=False)
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled')
        cache=path.parent/'c';cache.mkdir(exist_ok=False)
        cache_fields=dict(XDG_CACHE_HOME='xdg',XDG_CONFIG_HOME='config',TMPDIR='tmp',TORCH_HOME='torch',
            TORCH_EXTENSIONS_DIR='te',CUDA_CACHE_PATH='cuda',WARP_CACHE_PATH='warp',MPLCONFIGDIR='mpl',
            PYTHONPYCACHEPREFIX='pycache',WANDB_DIR='wandb',WANDB_CONFIG_DIR='wandbconfig',HF_HOME='hf',NUMBA_CACHE_DIR='numba')
        for key,name in cache_fields.items():
            directory=writable_path(cache/name);directory.mkdir();env[key]=str(directory)
        receipt['cache_environment']={key:env[key] for key in cache_fields}
        remaining=r['application_wall_seconds']-(time.monotonic()-started)-30
        if r['training'] and read(r['inputs'].get('configuration',r['inputs'].get('plan')))['budget']['maximum_wall_s']>remaining:
            raise RuntimeError('Setup consumed the frozen training deadline margin; do not launch a truncated session')
        command=native_command(r,remaining);receipt['native_command']=command
        process=subprocess.Popen(command,cwd=work,env=env,start_new_session=True);receipt['native_processes_started']=1;peak=0
        while process.poll() is None:
            size=base.old_helper('task10b_phase0').tree_bytes(path.parent);peak=max(peak,size)
            if size>CAP:receipt['budget_stop']='products_budget'
            if time.monotonic()-started>=r['application_wall_seconds']:receipt['budget_stop']='application_wall'
            if receipt['budget_stop']:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                break
            time.sleep(2)
        code=process.wait();receipt['peak_sampled_packet_bytes']=peak
        receipt['status']='executed' if code==0 and not receipt['budget_stop'] else 'failed'
        receipt['status_is_physical_qualification']=False
    except Exception as error:receipt['exception']=dict(type=type(error).__name__,message=str(error))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        receipt.update(returncode=code,ended_utc=base.now(),wall_seconds=time.monotonic()-started)
        if not receipt['native_processes_started']:receipt['actual_physics_steps']=0
        write_json(path.parent/'execution.json',receipt)
    return 0 if receipt['status']=='executed' else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True);p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--inputs',required=True,help='Human-owned JSON object of exact mode arguments')
    p.add_argument('--allocation-wall-seconds',type=float,required=True);p.add_argument('--application-wall-seconds',type=float,required=True)
    p.add_argument('--partition',default='mit_normal_gpu');p.add_argument('--successful-route-freeze')
    p.add_argument('--runtime-namespace-proof',help='Exact reviewed owned mount proof; never a general path override')
    for name in ('verify','submit','execute','account'):sub.add_parser(name).add_argument('--packet',required=True)
    p=sub.add_parser('budget');p.add_argument('--successful-route-freeze')
    p=sub.add_parser('project-timing-catalog');p.add_argument('--catalog',required=True);p.add_argument('--alias-probe',required=True)
    p.add_argument('--generation-execution',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.action=='execute':return execute(args.packet)
    if args.action=='project-timing-catalog':
        from human_punching.phase2_timing_control import project
        result=project(args.catalog,args.alias_probe,args.generation_execution,args.output)
    elif args.action=='prepare':result=prepare(args.output,args.mode,read(args.inputs),allocation_wall_seconds=args.allocation_wall_seconds,
        application_wall_seconds=args.application_wall_seconds,partition=args.partition,successful_route_freeze_path=args.successful_route_freeze,runtime_namespace_proof=args.runtime_namespace_proof)
    elif args.action=='account':result=base.account(args.packet)
    elif args.action=='budget':result=budget_status(base.ledger(),successful_route_frozen=successful_route_freeze(args.successful_route_freeze) is not None if args.successful_route_freeze else False)
    else:result=globals()[args.action](args.packet)
    print(json.dumps(result,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
