"""Direct isolated evaluation with the reviewed first-write event clocks.

The original native evaluator, admission and qualification equations are reused.
Only event metadata and its mandatory consumer identity differ. No old outputs
are edited. This version deliberately refuses moving/shared preparation.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from . import phase2_expanded_evaluation as parent
from . import phase2_continuation as continuation_module
from . import phase2_expanded_clocks_v1 as clocks
from . import phase2_direct_clock_audit_v2 as clock_audit
from .phase2_domain import fingerprint
from .wbc.provenance import writable_path, sha256_file, write_json

VERSION = 'task10b_phase2_direct_expanded_clock_entry_v2'
LAUNCH_SCHEMA = 'task10b_phase2_direct_expanded_native_launch_v2'
SIDECAR = 'event_clock_entry.json'
_identity = parent.execution_identity


def source_bindings():
    return {key: clocks.binding(module.__file__) for key, module in
            (('evaluator', parent), ('clocks', clocks), ('clock_audit', clock_audit))} | {
                'entry': clocks.binding(__file__)}


def clock_execution_identity(*args, **kwargs):
    result = _identity(*args, **kwargs)
    result['frozen']['event_metadata'] = dict(version=VERSION,
        event_clock_version=clocks.VERSION,
        source_sha256={key: item['sha256'] for key, item in source_bindings().items()})
    result['identity'] = fingerprint(result['frozen'])
    return result


@contextmanager
def identity_scope():
    """Use the same immutable metadata identity in prepare and native execution."""
    old_parent, old_continuation = parent.execution_identity, continuation_module.execution_identity
    try:
        parent.execution_identity = continuation_module.execution_identity = clock_execution_identity
        yield
    finally:
        parent.execution_identity, continuation_module.execution_identity = old_parent, old_continuation


def validate_scope(declaration_path, manifest_path):
    declaration = parent.load_declaration(declaration_path)
    manifest = parent.read_json(manifest_path)
    if declaration.start_mode != 'family_guard' or manifest.get('schema') != 'task10b_phase2_expanded_manifest_v1':
        raise ValueError('Direct clock v2 supports isolated family-guard panels only')
    catalog = parent.read_json(manifest['catalog_path'], manifest['catalog_sha256'])
    rows = {row['candidate_id']: row for row in catalog['cells']}
    if len(rows) != len(catalog['cells']):
        raise ValueError('Duplicate candidate identities')
    requested = {binding.trial_id for binding in declaration.bindings}
    entries = manifest['trials']
    if len(entries) != len(requested) or {entry['trial_id'] for entry in entries} != requested:
        raise ValueError('Exact complete declared panel manifest required')
    for entry in entries:
        row = rows[entry['candidate_id']]
        if row.get('start_mode') != 'family_guard' or row.get('preparation_complete_tick') != 0:
            raise ValueError('Explicit no-connector prep0 rows required; shared/moving preparation is separate')
    return declaration


def validate_selected_validation_clocks(declaration, route_path):
    """Final access cannot consume an old positive report lacking new clocks."""
    if getattr(declaration, 'split', None) != 'test':
        return None
    route = parent.read_json(route_path)
    launch = route['successful_validation']['launch']
    run = parent.owned_file(launch['path'], launch['sha256']).parent
    from .phase2_expanded_review_v2 import verify_event_evidence
    return verify_event_evidence(run)


def execute(declaration_path, manifest_path, route_path, checkpoint_path, training_path,
            predicate_path, mask_path, output, *, release, limits, protocol=None, continuation=None):
    if protocol is not None:
        raise ValueError('A shared or replacement native protocol cannot enter isolated clock v2')
    declaration = validate_scope(declaration_path, manifest_path)
    validate_selected_validation_clocks(declaration, route_path)
    folder = writable_path(output)
    if folder.exists():
        raise FileExistsError('Fresh direct expanded output required')
    old_evaluator, old_writer = parent.evaluate_policy, parent.write_json
    def first_write(path, value):
        path = Path(path).resolve()
        if path.exists():
            raise ValueError('Direct event metadata must be first-written inside its fresh output')
        if not path.is_relative_to(folder):
            # The unchanged final-test access latch writes its terminal record
            # outside the run. Only the exact already-claimed session may do so.
            claim_root = (parent.PROJECT_ROOT/'outputs/task10b/p2_final_test_claims').resolve()
            claim_file = path.parent/'claim.json'
            if (path.name != 'terminal.json' or not path.is_relative_to(claim_root)
                    or not claim_file.is_file() or value.get('application_run') != str(folder)
                    or json.loads(claim_file.read_text()).get('run') != str(folder)):
                raise ValueError('Unexpected external evaluator output')
            return old_writer(path, value)
        if path.name == 'launch.json':
            value.update(schema=LAUNCH_SCHEMA,direct_consumer_version=VERSION, event_clock_version=clocks.VERSION,
                         event_metadata_sources=source_bindings(),
                         event_clock_review_receipt_required=SIDECAR)
        elif path.name == 'expanded_scope.json':
            value.pop('historical_inner_planning_label', None)
            value.update(direct_consumer_version=VERSION, event_clock_version=clocks.VERSION,
                         first_write_event_metadata=True, planning_scope=clocks.SCOPE)
        elif path.name == 'execution.json':
            value.update(direct_consumer_version=VERSION, event_clock_version=clocks.VERSION,
                         event_clock_review_receipt_required=SIDECAR)
        return old_writer(path, value)
    try:
        parent.evaluate_policy, parent.write_json = clocks.evaluate_policy, first_write
        with identity_scope():
            result = parent.execute(declaration_path, manifest_path, route_path, checkpoint_path,
                training_path, predicate_path, mask_path, folder, release=release, limits=limits,
                continuation=continuation)
    finally:
        parent.evaluate_policy, parent.write_json = old_evaluator, old_writer
    audits = []
    for batch in sorted(folder.glob('batch_*')):
        if not batch.is_dir():
            continue
        try:
            audit = clock_audit.audit_batch(batch)
        except Exception as error:
            audit = dict(passed=False, exception=dict(type=type(error).__name__, message=str(error)))
        audits.append(dict(batch=str(batch.relative_to(folder)), **audit))
    # Offline rejections need no invented native clock. A started native attempt
    # without its batch is unresolved, including a constructor interruption.
    started = result.get('native_attempts_started')
    passed = (type(started) is int and started >= 0 and all(a['passed'] for a in audits)
              and sum(a.get('trials', 0) for a in audits) == started)
    receipt = dict(schema=VERSION, event_clock_version=clocks.VERSION, passed=passed,
        sources=source_bindings(), execution=clocks.binding(folder/'execution.json'),
        declaration=clocks.binding(declaration_path), manifest=clocks.binding(manifest_path),
        batch_audits=audits, native_attempts_started=started,
        all_native_batches_required=True, source_evaluator_changed=False,
        physical_actor_reference_or_qualification_changes=False, qualification=False,
        formal_acceptance=False, original_result_status=result['status'])
    write_json(folder/SIDECAR, receipt)
    return {**result, 'status': result['status'] if passed else 'failed',
            'event_clocks_pass': passed, 'event_clock_receipt': clocks.binding(folder/SIDECAR)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('declaration','manifest','route','checkpoint','training','predicate','mask','output','release'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--wall-seconds', type=float, required=True)
    parser.add_argument('--maximum-worlds', type=int, default=16)
    parser.add_argument('--continuation', type=Path)
    args = parser.parse_args(argv)
    result = execute(args.declaration,args.manifest,args.route,args.checkpoint,args.training,
        args.predicate,args.mask,args.output,release=args.release,
        limits=parent.EvaluationLimits(args.wall_seconds,maximum_worlds=args.maximum_worlds),
        continuation=args.continuation)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0 if result['status']=='executed_pending_audit_and_video' and result['event_clocks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
