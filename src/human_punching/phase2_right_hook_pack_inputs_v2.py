"""Exact RH partial+suffix training composite and full validation packing."""
from __future__ import annotations
import json
from pathlib import Path
from . import phase2_pipeline_inputs_v2 as parent
from .phase2_right_hook_full_projection_v1 import verify_projection, files as projection_files, FULL_SCHEMA
from .phase2_right_hook_composite_projection_v1 import (verify_projection as verify_composite,
    files as composite_files, SUFFIX_SCHEMA)
from .wbc.provenance import PROJECT_ROOT, sha256_file, writable_path, write_json

VERSION = 'task10b_phase2_right_hook_pack_inputs_v2'
MODE = 'right-hook-runtime-pack-v2'
AUDIT = 'task10b_phase2_rh_full_saved_target_reference_audit_v1'
SUFFIX_AUDIT = 'task10b_phase2_rh_suffix_saved_target_reference_audit_v2'


class Inputs(parent.Inputs):
    def projection(self, path, catalog):
        if any(k in catalog for k in ('target_catalog_projection', 'timing_control_projection',
                                     'sequential_catalog_projection', 'family_catalog_projection')):
            raise ValueError('Exact distinct RH projection identity required')
        if 'right_hook_composite_catalog_projection' in catalog:
            if 'right_hook_full_catalog_projection' in catalog:
                raise ValueError('A composite may not masquerade as a single full generation')
            view,receipt=verify_composite(path)
            if (view!=catalog or receipt['components']['suffix']['packet_schema']!=SUFFIX_SCHEMA
                    or catalog.get('construction_kind')!='audited_partial_plus_exact_suffix'
                    or catalog.get('split')!='train'):
                raise ValueError('Exact original54 plus audited suffix10 training composition required')
            closure=composite_files(path)
        else:
            if 'right_hook_full_catalog_projection' not in catalog:
                raise ValueError('Exact full validation projection required')
            view,receipt=verify_projection(path)
            if view!=catalog or receipt['packet_schema']!=FULL_SCHEMA or receipt['protocol']['split']!='validation':
                raise ValueError('Only the full24 validation allocation is supported here')
            closure=projection_files(path)
        for name, digest in closure.items():
            role = 'verified_right_hook_generation_archive' if 'source_snapshot' in Path(name).parts else None
            self.add(name, digest, provenance_role=role)


def files(mode, inputs, *, root=PROJECT_ROOT, provenance_files=None):
    if mode != MODE or set(inputs) != {'plan'}:
        raise ValueError('Only the exact RH runtime-pack plan is supported')
    collector = Inputs(root, provenance_files)
    plan = collector.read(collector.add(inputs['plan']))
    if (plan.get('packing_inputs_version') != VERSION or len(plan.get('catalogs', [])) != 2
            or len(plan.get('right_hook_reference_audits', [])) != 2 or len(plan.get('recipes', [])) != 1
            or plan.get('previous_packed_catalogs') != [] or plan.get('family') != 'right_hook'
            or plan.get('physical_qualification') is not False or plan.get('training_authorized_by_packing') is not False):
        raise ValueError('One RH recipe, full64/24 catalogs, two audits and a distinct bounded active pool required')
    # Existing scientific and runtime array admission is retained; no new packer.
    parent.runtime_pack_files(collector, inputs['plan'])
    recipe_sha = plan['recipes'][0]['sha256']
    for index, (binding, audit_binding) in enumerate(zip(plan['catalogs'], plan['right_hook_reference_audits'], strict=True)):
        view = collector.read(collector.bound(binding)); audit_path = collector.bound(audit_binding)
        audit = collector.read(audit_path)
        split, count, timings = ('train',64,(1.,1.2)) if index == 0 else ('validation',24,(1.,1.1,1.2))
        rows = view['cells']; audit_rows = audit.get('cells', [])
        if index==0:
            if 'right_hook_composite_catalog_projection' not in view:
                raise ValueError('First catalog must be the exact training composite')
            component=view['right_hook_composite_catalog_projection']['components']['suffix']
            expected_schema=SUFFIX_AUDIT;expected_rows=rows[54:];expected_manifest=component['native_catalog']['sha256']
        else:
            if 'right_hook_full_catalog_projection' not in view:
                raise ValueError('Second catalog must be the exact full validation view')
            component=view['right_hook_full_catalog_projection']
            expected_schema=AUDIT;expected_rows=rows;expected_manifest=view['unpacked_catalog_sha256']
        projected_audit=component['saved_audit']
        if (len(rows) != count or any(r.get('family') != 'right_hook' or r.get('split') != split
                    or r.get('requested_ttc_s') not in timings or r.get('recipe_sha256') != recipe_sha for r in rows)
                or view.get('recipe_sha256') != recipe_sha
                or projected_audit != dict(relative_name=str(audit_path.relative_to(collector.root)), sha256=audit_binding['sha256'])
                or audit.get('schema') != expected_schema or audit.get('passed') is not True
                or audit.get('family') != 'right_hook' or audit.get('split') != split or audit.get('physics_steps') != 0
                or audit.get('manifest_sha256') != expected_manifest
                or len(audit_rows) != len(expected_rows) or [r.get('candidate_id') for r in audit_rows] != [r['candidate_id'] for r in expected_rows]
                or any(r.get('all_checks_passed') is not True for r in audit_rows)):
            raise ValueError('Exact unchanged RH split, common recipe and saved audit required')
    return collector.finish()


def create_plan(training_catalog, validation_catalog, recipe, training_audit, validation_audit, output):
    """Metadata only; full failed rows remain declared and are never resampled."""
    from .phase2_runtime_pack import VERSION as STORAGE, CAP
    def binding(value):
        path = Inputs(PROJECT_ROOT).path(value)
        return dict(path=str(path), sha256=sha256_file(path))
    destination = writable_path(output)
    if destination.exists(): raise FileExistsError(destination)
    plan = dict(schema='task10b_phase2_runtime_pack_plan_v1', storage_version=STORAGE, byte_cap=CAP,
        packing_inputs_version=VERSION, family='right_hook', catalogs=list(map(binding, (training_catalog,validation_catalog))),
        recipes=[binding(recipe)], previous_packed_catalogs=[],
        right_hook_reference_audits=list(map(binding,(training_audit,validation_audit))),
        physical_qualification=False, training_authorized_by_packing=False)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination/'plan.json', plan)
    closure = files(MODE, {'plan':str(destination/'plan.json')})
    write_json(destination/'inputs.json', {'plan':str(destination/'plan.json')})
    write_json(destination/'preflight.json', dict(schema=VERSION, planned_rows=88, files=closure,
        allocation_wall_seconds=240, application_wall_seconds=210, child_wall_seconds=60,
        physics_steps=0, reference_generation=False, runtime_packing_executed=False))
    return binding(destination/'plan.json')
