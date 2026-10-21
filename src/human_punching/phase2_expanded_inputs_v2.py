"""Exact runtime file edges for isolated direct expanded evaluation.

Opaque historical provenance is bound as bytes, never interpreted as runtime
paths. Domain-local ledger paths remain domain-relative. Native scientific
validators still enforce declaration, split, training and mask semantics.
"""
from __future__ import annotations
from pathlib import Path
from .phase2_pipeline_inputs_v2 import Inputs
from .phase2_expanded_entry_v2 import validate_scope
from .wbc.provenance import PROJECT_ROOT, sha256_file

KEYS = ('declaration','manifest','route','checkpoint','training','predicate','mask')
VERSION = 'task10b_phase2_direct_expanded_inputs_v2'


def _ancestry(collector, run):
    from .phase2_continuation import verified_run_ancestry
    ancestry = verified_run_ancestry(run)
    for name in ancestry['run_directories']:
        folder = Path(name)
        identity = collector.read(collector.add(folder/'execution_identity.json'))
        for key, path in identity['input_paths'].items():
            collector.add(path, identity['frozen']['input_sha256'][key])
        for name in ('continuation.json','event_clock_entry.json','execution.json','outcomes.json',
                     'launch.json','declaration.json'):
            path = folder/name
            if path.exists():
                record = collector.read(collector.add(path))
                if name == 'continuation.json':
                    collector.add(record['configuration_path'], record['configuration_sha256'])
                    for key in ('parent_execution','parent_outcomes'):
                        collector.add(record[key], record[key+'_sha256'])
        for path in sorted(folder.glob('native_attempt_intent_*.json')):
            collector.add(path)
        # The new selected-validation clock consumer rereads these exact batch
        # artifacts. Do not close arbitrary logs, videos or ancestry metadata.
        if (folder/'event_clock_entry.json').exists():
            for batch in sorted(folder.glob('batch_*')):
                if not batch.is_dir():continue
                for name in ('plan.json','summary.json','closed_protocol_launch.json',
                             'launch_commitments.json','raw_trace.npz'):
                    collector.add(batch/name)
                for path in sorted(batch.glob('trial_*.json')):collector.add(path)


def _route(collector, path):
    route = collector.read(collector.add(path))
    if route.get('schema') != 'task10b_phase2_evaluation_route_v1':
        raise ValueError('Exact evaluation route required')
    evidence = route.get('successful_validation')
    if evidence:
        records = {key: collector.read(collector.bound(evidence[key])) for key in
                   ('qualification','trace_audit','launch','training')}
        collector.bound(records['training']['checkpoint'])
        collector.bound(records['launch']['release_inputs']['checkpoint'])
        audit = records['trace_audit']
        for panel in audit['panels']:
            collector.add(Path(panel['path'])/'raw_trace.npz', panel['raw_sha256'])
        run = collector.path(evidence['launch']['path']).parent
        if (run/'execution_identity.json').exists():
            _ancestry(collector, run)
    return route


def files(inputs, *, root=PROJECT_ROOT, provenance_files=None, successful_route_freeze=None):
    if set(inputs) not in (set(KEYS),set(KEYS)|{'continuation'}):
        raise ValueError('Exact direct expanded input names required')
    collector = Inputs(root, provenance_files)
    paths = {key: collector.add(inputs[key]) for key in KEYS}
    validate_scope(paths['declaration'], paths['manifest'])
    manifest = collector.read(paths['manifest'])
    collector.domain(manifest['domain_path'], manifest['domain_sha256'])
    collector.add(manifest['target_geometry_path'], manifest['target_geometry_sha256'])
    catalog_path = collector.add(manifest['catalog_path'], manifest['catalog_sha256'])
    catalog = collector.read(catalog_path)
    rows = {row['candidate_id']:row for row in catalog['cells']}
    for entry in manifest['trials']:
        collector.add(entry['start_bank_path'], entry['start_bank_sha256'])
        row = rows[entry['candidate_id']]
        if row.get('reference_path'):
            collector.model(row)
            ref = collector.add(row['reference_path'], row['reference_sha256'])
            legacy = collector.add(row['legacy_runtime_path'], row['legacy_runtime_sha256'])
            if ref != legacy or row['reference_sha256'] != row['legacy_runtime_sha256']:
                raise ValueError('One exact state/feedforward archive required')
        else:
            from .phase2_expanded_evaluation import validate_negative_reference_identity
            if not validate_negative_reference_identity(row):
                raise ValueError('Absent reference requires explicit negative identity')
            collector.add(row['generation_failure_path'], row['generation_failure_sha256'])
    _route(collector, paths['route'])
    if inputs.get('continuation'):
        continuation = collector.read(collector.add(inputs['continuation']))
        for key in ('parent_execution_identity','parent_execution','parent_outcomes'):
            collector.bound(continuation[key])
        _ancestry(collector, collector.path(continuation['parent_execution_identity']['path']).parent)
    if successful_route_freeze:
        proof = collector.read(collector.add(successful_route_freeze))
        domain = collector.bound(proof['domain'])
        collector.domain(domain, proof['domain']['sha256'])
        for item in proof['families'].values():
            _route(collector, collector.bound(item))
    return collector.finish()
