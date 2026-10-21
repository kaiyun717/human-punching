#!/usr/bin/env python3
"""Isolated grouped transport adding truthful expanded event metadata only."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('task10b_grouped_clock_parent',ROOT/'scripts/task10b_phase2_sequential_pipeline_v1.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
engine=parent.parent.legacy
SCHEMA='task10b_phase2_grouped_clock_packet_v2'
_inputs=parent.required_inputs;_native=parent.native_command;_submission=parent.submission_command;_validate=parent.validate_spec

def validate_spec(mode,inputs,wall_s,app_s,*,proof=None):
    if mode!='grouped-pilot' or wall_s!=1800 or app_s!=1680:
        raise ValueError('Clock-adapted grouped pilot requires exact1800/1680 bounded envelope')
    return {**_validate(mode,inputs,wall_s,app_s,proof=proof),'expanded_event_clock_version':'task10b_phase2_expanded_event_clocks_v1',
        'grouped_integrity_gate_version':'task10b_phase2_grouped_entry_integrity_v3'}

def required_inputs(record,*,provenance_files=None):
    if record['mode']!='grouped-pilot':raise ValueError('Grouped-only clock adapter')
    result=_inputs(record,provenance_files=provenance_files)
    extra={'src/human_punching/phase2_expanded_clocks_v1.py','src/human_punching/phase2_expanded_clock_audit_v1.py',
        'src/human_punching/phase2_grouped_clock_entry_v2.py','scripts/task10b_phase2_grouped_clocks_v2.py',
        'scripts/task10b_phase2_grouped_clocks_v2.sbatch'}
    for name in engine.local_import_closure(extra):result[name]=engine.sha256_file(engine.base.local(name))
    return dict(sorted(result.items()))

def native_command(record,remaining_wall_s=None):
    if record['mode']!='grouped-pilot':raise ValueError('Grouped-only clock adapter')
    command=_native(record,remaining_wall_s)
    command[command.index('human_punching.phase2_grouped_entry_v2')]='human_punching.phase2_grouped_clock_entry_v2'
    return command

def submission_command(record,output):
    command=_submission(record,output)
    command[command.index(str(ROOT/'scripts/task10b_phase2_sequential_pipeline_v1.sbatch'))]=str(ROOT/'scripts/task10b_phase2_grouped_clocks_v2.sbatch')
    return command

engine.SCHEMA=SCHEMA;engine.MODES=('grouped-pilot',);parent.parent.MODES=('grouped-pilot',)
engine.required_inputs=required_inputs;engine.native_command=native_command
engine.submission_command=submission_command;engine.validate_spec=validate_spec
prepare=parent.prepare;verify=engine.verify
if __name__=='__main__':raise SystemExit(parent.parent.main())
