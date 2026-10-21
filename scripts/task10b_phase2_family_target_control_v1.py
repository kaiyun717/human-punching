#!/usr/bin/env python3
"""Source-bound 32-cycle frozen target control, with no learning or test access."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('task10b_target_control_frozen_driver',ROOT/'scripts/task10b_phase2_expand.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
from human_punching.phase2_family_target_control_v1 import validate_plan,files,VERSION
MODE='family-target-control'
SCHEMA='task10b_phase2_family_target_control_packet_v1'
_required_inputs=legacy.required_inputs
_submission_command=legacy.submission_command


def validate_spec(mode,inputs,wall_s,app_s,*,proof=None):
    if (mode!=MODE or set(inputs)!={'plan'} or proof is not None
            or type(wall_s) not in (int,float) or type(app_s) not in (int,float)
            or wall_s!=360 or app_s!=300):
        raise ValueError('Exact frozen target plan and360/300 limits required')
    plan=legacy.read(legacy.base.local(inputs['plan']))
    validate_plan(plan)
    if plan['native_attempt_capacity']==0:
        raise ValueError('Zero admitted references require CPU negative ledger, not a GPU packet')
    return dict(family=plan['family'],seed=21021,fresh_transition_cap=0,target_control_environments=8,
        target_control_version=VERSION,target_control_trials=32,target_control_cells=4,
        target_control_child_wall_seconds=210,target_control_native_attempt_capacity=plan['native_attempt_capacity'],physical_qualification=False)


def required_inputs(record,*,provenance_files=None):
    if (record['mode']!=MODE or set(record['inputs'])!={'plan'}
            or record.get('successful_route_freeze') or not record.get('runtime_namespace_proof')):
        raise ValueError('Exact frozen target inputs and namespace proof required')
    expected=legacy.base.local(record['inputs']['plan'])
    def exact(seeds,*,provenance_files=None):
        if list(map(legacy.base.local,seeds))!=[expected]:
            raise ValueError('Unexpected target-control semantic seed')
        return files(expected)
    original=legacy.semantic_files
    try:
        legacy.semantic_files=exact
        closure=_required_inputs(record,provenance_files=provenance_files)
    finally:legacy.semantic_files=original
    for name in legacy.local_import_closure({'scripts/task10b_phase2_family_target_control_v1.py',
            'scripts/task10b_phase2_family_target_control_v1.sbatch',
            'src/human_punching/phase2_family_target_control_v1.py',
            'src/human_punching/phase2_family_projection_v1.py'}):
        closure[name]=legacy.sha256_file(legacy.base.local(name))
    return dict(sorted(closure.items()))


def submission_command(record,output):
    command=_submission_command(record,output)
    command[command.index(str(ROOT/'scripts/task10b_phase2_expand.sbatch'))]=str(ROOT/'scripts/task10b_phase2_family_target_control_v1.sbatch')
    return command


def native_command(record,remaining_wall_s=None):
    if (record.get('mode')!=MODE or set(record.get('inputs',{}))!={'plan'}
            or record.get('application_wall_seconds')!=300 or record.get('allocation_wall_seconds')!=360
            or record.get('target_control_version')!=VERSION or record.get('target_control_trials')!=32
            or record.get('target_control_cells')!=4 or record.get('target_control_child_wall_seconds')!=210
            or record.get('target_control_environments')!=8 or record.get('fresh_transition_cap')!=0
            or type(record.get('target_control_native_attempt_capacity')) is not int
            or record['target_control_native_attempt_capacity'] not in (8,16,24,32)
            or record.get('physical_qualification') is not False):
        raise ValueError('Exact frozen32-trial packet required')
    available=270 if remaining_wall_s is None else remaining_wall_s
    if (type(available) not in (int,float) or not math.isfinite(available) or not 210<=available<=270):
        raise RuntimeError('Setup consumed fixed210s child margin; refuse truncated control')
    return [str(legacy.base.PYTHON),'-B','-m','human_punching.phase2_family_target_control_v1',
        '--plan',str(legacy.base.local(record['inputs']['plan'])),
        '--output',str(ROOT/record['output']),'--wall-seconds','210.0']


legacy.SCHEMA=SCHEMA
legacy.MODES=(MODE,)
legacy.CONDITIONAL_MODES=()
legacy.validate_spec=validate_spec
legacy.required_inputs=required_inputs
legacy.submission_command=submission_command
legacy.native_command=native_command


def prepare(output,mode,inputs,**kwargs):
    if kwargs.get('runtime_namespace_proof') is None:raise ValueError('Reviewed namespace proof required')
    return legacy.prepare(output,mode,inputs,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True);p.add_argument('--mode',choices=(MODE,),required=True)
    p.add_argument('--inputs',required=True);p.add_argument('--allocation-wall-seconds',type=float,required=True)
    p.add_argument('--application-wall-seconds',type=float,required=True);p.add_argument('--partition',default='mit_normal_gpu')
    p.add_argument('--runtime-namespace-proof',required=True)
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
