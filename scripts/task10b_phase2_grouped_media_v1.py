#!/usr/bin/env python3
"""Narrow saved grouped media entry; unchanged expanded renderer, no replay."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('task10b_grouped_media_original_driver',ROOT/'scripts/task10b_phase2_expand.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
from human_punching.phase2_grouped_media_inputs_v1 import files,selected,VERSION
SCHEMA='task10b_phase2_grouped_media_packet_v1'
_inputs=legacy.required_inputs;_submission=legacy.submission_command;_validate=legacy.validate_spec


def validate_spec(mode,inputs,wall_s,app_s,*,proof=None):
    if mode!='media' or set(inputs)!={'run'} or wall_s!=300 or app_s!=240 or proof is not None:
        raise ValueError('Exact300/240 isolated initial-budget grouped saved media required')
    result=_validate(mode,inputs,wall_s,app_s,proof=None)
    _,selection=selected(legacy.base.local(inputs['run']),'outputs/task10b/p2hostalias01/probe.json')
    return {**result,'grouped_media_input_version':VERSION,'source_training_update':selection['update'],
        'planned_required_clips':selection['planned'],'physics_steps':0,'actor_calls':0}


def required_inputs(record,*,provenance_files=None):
    if record['mode']!='media' or set(record['inputs'])!={'run'} or not record.get('runtime_namespace_proof'):
        raise ValueError('Exact grouped saved panel and namespace proof required')
    run=legacy.base.local(record['inputs']['run']);proof=legacy.bound(record['runtime_namespace_proof'])
    expected=sorted(p.resolve() for p in run.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.pt','.xml'))
    def exact(seeds,*,provenance_files=None):
        if sorted(Path(p).resolve() for p in seeds)!=expected:raise ValueError('Unreviewed grouped media semantic seeds')
        closure,selection=files(run,proof)
        if selection['planned']!=record['planned_required_clips']:raise ValueError('Prespecified/flagged media union changed')
        return closure
    original=legacy.semantic_files
    try:
        legacy.semantic_files=exact;result=_inputs(record,provenance_files=provenance_files)
    finally:legacy.semantic_files=original
    extra={'src/human_punching/phase2_grouped_media_inputs_v1.py','src/human_punching/phase2_grouped_saved_v1.py',
        'scripts/task10b_phase2_grouped_media_v1.py','scripts/task10b_phase2_grouped_media_v1.sbatch'}
    for name in legacy.local_import_closure(extra):result[name]=legacy.sha256_file(legacy.base.local(name))
    return dict(sorted(result.items()))


def submission_command(record,output):
    command=_submission(record,output)
    command[command.index(str(ROOT/'scripts/task10b_phase2_expand.sbatch'))]=str(ROOT/'scripts/task10b_phase2_grouped_media_v1.sbatch')
    return command


legacy.SCHEMA=SCHEMA;legacy.MODES=('media',);legacy.CONDITIONAL_MODES=()
legacy.validate_spec=validate_spec;legacy.required_inputs=required_inputs;legacy.submission_command=submission_command
prepare=legacy.prepare;verify=legacy.verify
if __name__=='__main__':raise SystemExit(legacy.main())
