#!/usr/bin/env python3
"""Isolated full sequential packing/grouped route with unchanged scientific math."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('task10b_sequential_pipeline_parent',ROOT/'scripts/task10b_phase2_expand_v2.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
from human_punching.phase2_sequential_pipeline_inputs_v1 import files,VERSION

SCHEMA='task10b_phase2_sequential_pipeline_packet_v1'
_inputs=parent.required_inputs
_submission=parent.submission_command
_native=parent.native_command
_validate=parent.validate_spec
parent.files=files
parent.VERSION=VERSION
parent.legacy.SCHEMA=SCHEMA


def validate_spec(mode,inputs,wall_s,app_s,*,proof=None):
    if mode=='runtime-pack' and (type(wall_s) not in (int,float) or type(app_s) not in (int,float)
            or wall_s!=300 or app_s!=240):
        raise ValueError('This complete88-reference pack uses exact300/240 allocation/application limits')
    return _validate(mode,inputs,wall_s,app_s,proof=proof)


def required_inputs(record,*,provenance_files=None):
    closure=_inputs(record,provenance_files=provenance_files)
    extra={'src/human_punching/phase2_sequential_pipeline_inputs_v1.py',
        'src/human_punching/phase2_runtime_pack_entry_v3.py',
        'scripts/task10b_phase2_sequential_pipeline_v1.py','scripts/task10b_phase2_sequential_pipeline_v1.sbatch'}
    for name in parent.legacy.local_import_closure(extra):
        closure[name]=parent.legacy.sha256_file(parent.legacy.base.local(name))
    return dict(sorted(closure.items()))


def submission_command(record,output):
    command=_submission(record,output)
    command[command.index(str(ROOT/'scripts/task10b_phase2_expand_v2.sbatch'))]=str(ROOT/'scripts/task10b_phase2_sequential_pipeline_v1.sbatch')
    return command


def native_command(record,remaining_wall_s=None):
    if record['mode']=='runtime-pack' and (record.get('allocation_wall_seconds')!=300
            or record.get('application_wall_seconds')!=240):
        raise ValueError('Exact300/240 packing packet required')
    command=_native(record,remaining_wall_s)
    if record['mode']=='runtime-pack':
        command[command.index('human_punching.phase2_runtime_pack_entry_v2')]='human_punching.phase2_runtime_pack_entry_v3'
    return command


parent.legacy.required_inputs=required_inputs
parent.legacy.submission_command=submission_command
parent.legacy.native_command=native_command
parent.legacy.validate_spec=validate_spec
prepare=parent.prepare
verify=parent.legacy.verify


if __name__=='__main__':raise SystemExit(parent.main())
