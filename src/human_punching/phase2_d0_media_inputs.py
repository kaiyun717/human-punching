"""Exact D0 saved-media inputs; accepted arguments remain opaque ancestry.

No paths or scientific values are rewritten. The renderer consumes saved states,
its runtime XML and the target centers from the verified cached catalog.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

from .phase2_learning import family_arguments
from .wbc.provenance import sha256_file

VERSION='task10b_phase2_d0_saved_media_inputs_v1'


def files(root,inputs,release,bundle,predicate,original_control_catalog):
    root=Path(root).resolve(strict=True);found={}
    def owned(value):
        p=Path(value);p=(p if p.is_absolute() else root/p).resolve(strict=True)
        if not p.is_relative_to(root):raise ValueError('D0 media input escaped owned root')
        return p
    def add(value,expected=None):
        p=owned(value)
        if not p.is_file():raise ValueError('D0 media input must be an existing file')
        actual=sha256_file(p)
        if expected is not None and actual!=expected:raise ValueError('D0 media bound input changed')
        found[str(p.relative_to(root))]=actual;return p
    run=owned(inputs['run']);panel=add(inputs['panel'])
    if not run.is_dir() or panel.parent.parent!=run or panel.name!='summary.json':
        raise ValueError('D0 media requires its exact saved aggregate summary')
    launch=json.loads(add(run/'launch.json').read_text());summary=json.loads(panel.read_text())
    if (launch.get('schema')!='task10b_phase2_original_route_launch_v1' or launch.get('mode')!='frozen'
            or not launch.get('slurm_job_id') or summary.get('schema')!='task10b_phase2_d0_panel_v1'):
        raise ValueError('Exact counted frozen D0 launch and panel required')
    family=launch['family'];_,accepted=family_arguments(owned(release),family)
    if set(launch.get('release_inputs',{}))!=set(accepted):raise ValueError('D0 media accepted input roles differ')
    for key,source in accepted.items():
        item=launch['release_inputs'][key]
        if owned(item['path'])!=source:raise ValueError('D0 media accepted source path differs')
        add(source,item['sha256'])
    for source,expected in launch['sources'].items():add(source,expected)
    catalog=add(launch['catalog'],launch['catalog_sha256']);data=json.loads(catalog.read_text())
    if 'timing_control_projection' in data:
        # Load only this fixed, source-bound helper under the same root spelling;
        # its existing validator owns the exact logical-path projection proof.
        spec=importlib.util.spec_from_file_location('human_punching._d0_media_timing_control',root/'src/human_punching/phase2_timing_control.py')
        timing=importlib.util.module_from_spec(spec);spec.loader.exec_module(timing)
        data,evidence=timing.validate(catalog)
        meta=data['timing_control_projection'];logical=Path(json.loads(timing.bound(meta['alias_probe']).read_text())['local_root'])
        originals=[dict(row) for row in data['cells'] if row['kind']=='original_control']
        for row in originals:
            # Inspection-only spelling projection of these three proved files;
            # the bound catalog, scientific values and arrays stay untouched.
            for key in ('reference_path','legacy_runtime_path','diagnostic_screen_path'):
                row[key]=str(owned(Path(row[key]).relative_to(logical)))
        original_control_catalog(dict(catalog=str(catalog),family=family),catalog_override={**data,'cells':originals})
        for row in evidence:
            for item in row['files'].values():add(item['path'],item['sha256'])
        for item in meta['mapped_files']:add(item['path'],item['sha256'])
        for item in (meta['alias_probe'],meta['generation_execution']):add(item['relative_name'],item['sha256'])
        add(meta['nonce']['path'],meta['nonce']['sha256'])
        add(data['unpacked_catalog_path'],data['unpacked_catalog_sha256'])
    else:
        original_control_catalog(dict(catalog=str(catalog),family=family))
        for row in data['cells']:
            for key in ('reference','legacy_runtime'):
                add(row[key+'_path'],row[key+'_sha256'])
            if row.get('diagnostic_screen_path'):add(row['diagnostic_screen_path'])
    candidates={row['candidate_id'] for row in data['cells'] if row['family']==family}
    if set(launch.get('planned_cells',[]))!=candidates:raise ValueError('D0 media requested cells differ from frozen catalog')
    trials=summary.get('trials',[])
    if (len(trials)!=8*len(candidates) or {r['trial_id'] for r in trials}!={f'{c}_bank_{i:02d}' for c in candidates for i in range(8)}):
        raise ValueError('D0 media preserves every declared candidate/bank identity')
    for item in summary['panels']:
        directory=owned(item['path'])
        if directory.parent!=panel.parent or directory.name!=item['candidate_id'] or directory.name not in candidates:
            raise ValueError('D0 media panel path/candidate differs')
    add(run/'runtime.xml')
    for p in run.rglob('*'):
        if not p.is_file():continue
        relative=p.relative_to(run);name=p.name
        if relative.parts[0] not in ('launch.json','execution.json','runtime.xml',panel.parent.name):
            raise ValueError('Unexpected D0 media run input')
        if p.suffix=='.json':
            if name in ('launch.json','execution.json'):pass
            elif name=='summary.json':
                if json.loads(p.read_text()).get('schema') not in ('task10b_phase2_d0_panel_v1','task10b_phase2_native_panel_v1'):
                    raise ValueError('Unknown D0 media summary schema')
            elif name=='plan.json':
                if json.loads(p.read_text()).get('schema')!='task10b_phase2_execution_plan_v1':raise ValueError('Unknown D0 execution plan')
            elif name=='launch_commitments.json':
                if json.loads(p.read_text()).get('schema')!='task10b_phase2_launch_commitments_v1':raise ValueError('Unknown D0 launch commitments')
            elif not re.fullmatch(r'trial_\d{4}\.json',name):raise ValueError('Unknown D0 media JSON evidence')
            add(p)
        elif p.suffix=='.npz':
            if name!='raw_trace.npz' and not re.fullmatch(r'trial_\d{4}\.npz',name):raise ValueError('Unknown D0 media array evidence')
            add(p)
        elif name!='runtime.xml':raise ValueError('Unknown D0 media input file type')
    return found
