#!/usr/bin/env python3
"""One bounded saved-video job for all four exact target-control panels."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('target_saved_media_base',ROOT/'scripts/task10b_phase2_expand.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
from human_punching.phase2_family_control_media_v1 import selected,files,VERSION
MODE='media';SCHEMA='task10b_phase2_family_control_saved_media_packet_v1'
_required=legacy.required_inputs;_submission=legacy.submission_command


def validate_spec(mode,inputs,wall_s,app_s,*,proof=None):
    if (mode!=MODE or set(inputs)!={'run'} or proof is not None
            or type(wall_s) not in (int,float) or type(app_s) not in (int,float)
            or wall_s!=300 or app_s!=240):raise ValueError('Exact300/240 saved target-media scope required')
    _,_,plan,_=selected(legacy.base.local(inputs['run']))
    return dict(family=plan['family'],seed=21021,fresh_transition_cap=0,actor_calls=0,
        target_media_version=VERSION,source_control_trials=32)


def required_inputs(record,*,provenance_files=None):
    if (record['mode']!=MODE or set(record['inputs'])!={'run'}
            or record.get('successful_route_freeze') or not record.get('runtime_namespace_proof')):
        raise ValueError('Exact saved target-media input and namespace proof required')
    inventory=dict(record,mode='timing-references',inputs={});original=legacy.semantic_files
    def empty(seeds,*,provenance_files=None):
        if list(seeds):raise ValueError('Unexpected base inventory seed')
        return {}
    try:
        legacy.semantic_files=empty;closure=_required(inventory,provenance_files=provenance_files)
    finally:legacy.semantic_files=original
    closure.update(files(legacy.base.local(record['inputs']['run'])))
    for name in legacy.local_import_closure({'scripts/task10b_phase2_family_media_v1.py',
            'scripts/task10b_phase2_family_media_v1.sbatch','src/human_punching/phase2_family_control_media_v1.py'}):
        closure[name]=legacy.sha256_file(legacy.base.local(name))
    return dict(sorted(closure.items()))


def native_command(record,remaining_wall_s=None):
    if (record.get('mode')!=MODE or set(record.get('inputs',{}))!={'run'}
            or record.get('allocation_wall_seconds')!=300 or record.get('application_wall_seconds')!=240
            or record.get('target_media_version')!=VERSION or record.get('source_control_trials')!=32
            or record.get('training') is not False or record.get('actor_calls')!=0):
        raise ValueError('Exact frozen target-media packet required')
    remaining=210 if remaining_wall_s is None else remaining_wall_s
    if type(remaining) not in (int,float) or not math.isfinite(remaining) or not 120<remaining<=210:
        raise RuntimeError('Insufficient saved-render/flush margin after setup')
    return [str(legacy.base.PYTHON),'-B','-m','human_punching.phase2_family_control_media_v1',
        '--run',str(legacy.base.local(record['inputs']['run'])),'--output',str(ROOT/record['output']),
        '--wall-seconds',str(remaining)]


def submission_command(record,output):
    command=_submission(record,output)
    command[command.index(str(ROOT/'scripts/task10b_phase2_expand.sbatch'))]=str(ROOT/'scripts/task10b_phase2_family_media_v1.sbatch')
    return command


legacy.SCHEMA=SCHEMA;legacy.MODES=(MODE,);legacy.CONDITIONAL_MODES=()
legacy.validate_spec=validate_spec;legacy.required_inputs=required_inputs
legacy.native_command=native_command;legacy.submission_command=submission_command


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    q=s.add_parser('prepare');q.add_argument('--output',required=True);q.add_argument('--inputs',required=True)
    q.add_argument('--partition',default='mit_normal_gpu');q.add_argument('--runtime-namespace-proof',required=True)
    for name in ('verify','submit','execute','account'):s.add_parser(name).add_argument('--packet',required=True)
    a=p.parse_args()
    if a.action=='execute':return legacy.execute(a.packet)
    if a.action=='prepare':result=legacy.prepare(a.output,MODE,legacy.read(a.inputs),allocation_wall_seconds=300,
        application_wall_seconds=240,partition=a.partition,runtime_namespace_proof=a.runtime_namespace_proof)
    elif a.action=='account':result=legacy.base.account(a.packet)
    else:result=getattr(legacy,a.action)(a.packet)
    print(json.dumps(result,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
