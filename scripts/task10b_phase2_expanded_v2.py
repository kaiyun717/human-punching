#!/usr/bin/env python3
"""Source-bound isolated direct evaluation with corrected first-write clocks."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('task10b_direct_expanded_parent',ROOT/'scripts/task10b_phase2_expand.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
from human_punching.phase2_expanded_entry_v2 import (VERSION, LAUNCH_SCHEMA, identity_scope, validate_scope,
    validate_selected_validation_clocks)
from human_punching.phase2_expanded_inputs_v2 import files

SCHEMA='task10b_phase2_direct_expanded_packet_v2'
MODE='expanded-evaluation'
_validate=parent.validate_spec;_inputs=parent.required_inputs
_native=parent.native_command;_submission=parent.submission_command
_freeze=parent.successful_route_freeze


def successful_route_freeze(path):
    """New native schemas require clock evidence before the unchanged freeze math."""
    original=parent.read
    def typed_read(value):
        result=original(value)
        if isinstance(result,dict) and result.get('schema')==LAUNCH_SCHEMA:
            from human_punching.phase2_expanded_review_v2 import verify_event_evidence
            verify_event_evidence(parent.base.local(value).parent)
            return {**result,'schema':'task10b_phase2_expanded_native_launch_v1'}
        return result
    try:
        parent.read=typed_read
        return _freeze(path)
    finally:parent.read=original


def validate_spec(mode,inputs,wall_s,app_s,*,proof=None):
    if mode!=MODE:raise ValueError('Direct isolated evaluation only')
    declaration=validate_scope(inputs['declaration'],inputs['manifest'])
    validate_selected_validation_clocks(declaration,inputs['route'])
    with identity_scope():
        result=_validate(mode,inputs,wall_s,app_s,proof=proof)
    return {**result,'direct_consumer_version':VERSION,
        'event_clock_version':'task10b_phase2_expanded_event_clocks_v1',
        'event_clock_review_receipt_required':'event_clock_entry.json'}


def required_inputs(record,*,provenance_files=None):
    if record['mode']!=MODE:raise ValueError('Direct isolated evaluation only')
    expected=[parent.base.local(v) for v in record['inputs'].values()]
    if record.get('successful_route_freeze'):expected.append(parent.base.local(record['successful_route_freeze']))
    def exact(seeds,*,provenance_files=None):
        if sorted(map(parent.base.local,seeds))!=sorted(expected):raise ValueError('Unexpected direct runtime input edge')
        return files(record['inputs'],root=ROOT,provenance_files=provenance_files,
            successful_route_freeze=record.get('successful_route_freeze'))
    old=parent.semantic_files
    try:
        parent.semantic_files=exact
        result=_inputs(record,provenance_files=provenance_files)
    finally:parent.semantic_files=old
    names={'scripts/task10b_phase2_expanded_v2.py','scripts/task10b_phase2_expanded_v2.sbatch',
        'scripts/task10b_phase2_expanded_audit_v2.py',
        'src/human_punching/phase2_expanded_entry_v2.py','src/human_punching/phase2_expanded_inputs_v2.py'}
    for name in parent.local_import_closure(names):result[name]=parent.sha256_file(parent.base.local(name))
    return dict(sorted(result.items()))


def native_command(record,remaining_wall_s=None):
    if record['mode']!=MODE:raise ValueError('Direct isolated evaluation only')
    validate_scope(record['inputs']['declaration'],record['inputs']['manifest'])
    command=_native(record,remaining_wall_s)
    command[command.index('human_punching.phase2_expanded_evaluation')]='human_punching.phase2_expanded_entry_v2'
    return command


def submission_command(record,output):
    command=_submission(record,output)
    command[command.index(str(ROOT/'scripts/task10b_phase2_expand.sbatch'))]=str(ROOT/'scripts/task10b_phase2_expanded_v2.sbatch')
    return command


parent.SCHEMA=SCHEMA;parent.MODES=(MODE,);parent.CONDITIONAL_MODES=(MODE,)
parent.validate_spec=validate_spec;parent.required_inputs=required_inputs
parent.native_command=native_command;parent.submission_command=submission_command
parent.successful_route_freeze=successful_route_freeze
prepare=parent.prepare;verify=parent.verify
if __name__=='__main__':raise SystemExit(parent.main())
