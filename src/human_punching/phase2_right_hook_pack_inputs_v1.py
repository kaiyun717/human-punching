"""Exact audited RH64/24 packing inputs; no execution or learning credit."""
from __future__ import annotations
import json
from pathlib import Path
from . import phase2_pipeline_inputs_v2 as parent
from .phase2_right_hook_full_projection_v1 import verify_projection, files as projection_files, FULL_SCHEMA
from .wbc.provenance import PROJECT_ROOT, sha256_file, writable_path, write_json

VERSION = 'task10b_phase2_right_hook_pack_inputs_v1'
MODE = 'right-hook-runtime-pack-v1'
AUDIT = 'task10b_phase2_rh_full_saved_target_reference_audit_v1'


class Inputs(parent.Inputs):
    def projection(self, path, catalog):
        if ('right_hook_full_catalog_projection' not in catalog
                or any(k in catalog for k in ('target_catalog_projection', 'timing_control_projection',
                                              'sequential_catalog_projection', 'family_catalog_projection'))):
            raise ValueError('Exact distinct RH full catalog projection required')
        view, receipt = verify_projection(path)
        if view != catalog or receipt['packet_schema'] != FULL_SCHEMA:
            raise ValueError('Only the audited RH full64/24 projection is supported')
        for name, digest in projection_files(path).items():
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
        projected_audit = view['right_hook_full_catalog_projection']['saved_audit']
        if (len(rows) != count or any(r.get('family') != 'right_hook' or r.get('split') != split
                    or r.get('requested_ttc_s') not in timings or r.get('recipe_sha256') != recipe_sha for r in rows)
                or view.get('recipe_sha256') != recipe_sha
                or projected_audit != dict(relative_name=str(audit_path.relative_to(collector.root)), sha256=audit_binding['sha256'])
                or audit.get('schema') != AUDIT or audit.get('passed') is not True
                or audit.get('family') != 'right_hook' or audit.get('split') != split or audit.get('physics_steps') != 0
                or audit.get('manifest_sha256') != view['unpacked_catalog_sha256']
                or len(audit_rows) != count or [r.get('candidate_id') for r in audit_rows] != [r['candidate_id'] for r in rows]
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
