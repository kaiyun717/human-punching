#!/usr/bin/env python3
"""Separate saved-only media launcher for the five-panel cross curve v2."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('task10b_curve_media_frozen_driver',ROOT/'scripts/task10b_phase2_expand.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
MODE='media';SCHEMA='task10b_phase2_cross_curve_media_packet_v2'
ALLOCATION_SECONDS,APPLICATION_SECONDS=300,240
_required_inputs=legacy.required_inputs
_submission_command=legacy.submission_command
_native_command=legacy.native_command


def validate_spec(mode,inputs,wall_s,app_s,*,proof=None):
    if (mode!=MODE or set(inputs)!={'run','panel'} or proof is not None
            or type(wall_s) not in (int,float) or type(app_s) not in (int,float)
            or wall_s!=ALLOCATION_SECONDS or app_s!=APPLICATION_SECONDS):
        raise ValueError('Only saved cross curve media with300/240second limits is supported')
    from human_punching.phase2_curve_saved_v2 import pilot
    run=legacy.base.local(inputs['run']);panel=legacy.base.local(inputs['panel'])
    launch=legacy.read(run/'launch.json')
    if (launch.get('schema')!=pilot.VERSION or launch.get('mode')!='timing_learning_curve'
            or launch.get('family')!='cross' or launch.get('seed')!=21021
            or panel.parent.parent!=run or panel.name!='summary.json'
            or panel.parent.name not in {f'validation_{u:05d}' for u in pilot.UPDATES}):
        raise ValueError('Exact saved curve launch and one declared validation panel required')
    return dict(family='cross',seed=21021,fresh_transition_cap=0,actor_calls=0,
        saved_curve_update=int(panel.parent.name.split('_')[1]),planned_source_validation_trials=120)


def required_inputs(record,*,provenance_files=None):
    if (record['mode']!=MODE or set(record['inputs'])!={'run','panel'}
            or record.get('successful_route_freeze') or not record.get('runtime_namespace_proof')):
        raise ValueError('Exact saved curve media and reviewed namespace proof required')
    from human_punching.phase2_curve_saved_v2 import files
    # Retain the original runtime/source inventory, with no generic ancestry
    # traversal. Only the explicit versioned saved-panel closure opens inputs.
    inventory=dict(record,mode='timing-references',inputs={})
    original=legacy.semantic_files
    def empty(seeds,*,provenance_files=None):
        if list(seeds):raise ValueError('Unexpected media inventory seed')
        return {}
    try:
        legacy.semantic_files=empty
        closure=_required_inputs(inventory,provenance_files=provenance_files)
    finally:legacy.semantic_files=original
    closure.update(files(ROOT,record['inputs'],record['release'],record['bundle'],record['predicate'],
        namespace_proof=legacy.bound(record['runtime_namespace_proof'])))
    extra={'scripts/task10b_phase2_curve_media_v2.py','scripts/task10b_phase2_curve_media_v2.sbatch',
        'scripts/task10b_phase2_curve_audit_v2.py','scripts/task10b_phase2_audit_clock_v2.py',
        'src/human_punching/phase2_curve_saved_v2.py'}
    for relative in legacy.local_import_closure(extra):
        closure[relative]=legacy.sha256_file(legacy.base.local(relative))
    return dict(sorted(closure.items()))


def native_command(record,remaining_wall_s=None):
    if (record['mode']!=MODE or record['allocation_wall_seconds']!=ALLOCATION_SECONDS
            or record['application_wall_seconds']!=APPLICATION_SECONDS or record.get('training') is not False
            or record.get('actor_calls')!=0 or record.get('planned_source_validation_trials')!=120):
        raise ValueError('Exact nontraining saved-media packet required')
    available=APPLICATION_SECONDS-30 if remaining_wall_s is None else remaining_wall_s
    if (type(available) not in (int,float) or not math.isfinite(available)
            or not 120<available<=APPLICATION_SECONDS-30):
        raise RuntimeError('Insufficient bounded rendering/flush margin after setup')
    return _native_command(record,available)


def submission_command(record,output):
    command=_submission_command(record,output)
    command[command.index(str(ROOT/'scripts/task10b_phase2_expand.sbatch'))]=str(ROOT/'scripts/task10b_phase2_curve_media_v2.sbatch')
    return command


legacy.SCHEMA=SCHEMA;legacy.MODES=(MODE,);legacy.CONDITIONAL_MODES=()
legacy.validate_spec=validate_spec;legacy.required_inputs=required_inputs
legacy.native_command=native_command;legacy.submission_command=submission_command


def prepare(output,mode,inputs,**kwargs):
    if kwargs.get('runtime_namespace_proof') is None:raise ValueError('Reviewed namespace proof required')
    return legacy.prepare(output,mode,inputs,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True);p.add_argument('--inputs',required=True)
    p.add_argument('--mode',choices=(MODE,),required=True)
    p.add_argument('--allocation-wall-seconds',type=float,required=True);p.add_argument('--application-wall-seconds',type=float,required=True)
    p.add_argument('--partition',default='mit_normal_gpu');p.add_argument('--runtime-namespace-proof',required=True)
    for name in ('verify','submit','execute','account'):sub.add_parser(name).add_argument('--packet',required=True)
    sub.add_parser('budget');args=parser.parse_args()
    if args.action=='execute':return legacy.execute(args.packet)
    if args.action=='prepare':
        result=prepare(args.output,args.mode,legacy.read(args.inputs),allocation_wall_seconds=args.allocation_wall_seconds,
            application_wall_seconds=args.application_wall_seconds,partition=args.partition,runtime_namespace_proof=args.runtime_namespace_proof)
    elif args.action=='account':result=legacy.base.account(args.packet)
    elif args.action=='budget':result=legacy.budget_status(legacy.base.ledger(),successful_route_frozen=False)
    else:result=getattr(legacy,args.action)(args.packet)
    print(json.dumps(result,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
