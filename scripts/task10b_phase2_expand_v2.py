#!/usr/bin/env python3
"""Isolated target runtime-packing/grouped-pilot entry; old packets stay frozen.

Reuses the reviewed launcher's resource ledger, immutable snapshots, runtime
pins, reservation lock, source verification and watchdog. Only the two named
input closures, batch entry and grouped final-integrity consumer differ.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
spec = importlib.util.spec_from_file_location('task10b_phase2_frozen_expansion_v1',
                                             ROOT/'scripts/task10b_phase2_expand.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
from human_punching.phase2_pipeline_inputs_v2 import files, MODES, VERSION

SCHEMA = 'task10b_phase2_expansion_packet_v2'
_required_inputs = legacy.required_inputs
_validate_spec = legacy.validate_spec
_submission_command = legacy.submission_command
_native_command = legacy.native_command


def validate_spec(mode, inputs, wall_s, app_s, *, proof=None):
    if mode not in MODES or proof is not None:
        raise ValueError('V2 currently supports bounded runtime-pack/grouped-pilot only')
    scope = _validate_spec(mode, inputs, wall_s, app_s, proof=None)
    return dict(**scope, pipeline_inputs_version=VERSION,
                final_checkpoint_integrity_required=mode == 'grouped-pilot')


def required_inputs(record, *, provenance_files=None):
    if (record['mode'] not in MODES or record.get('successful_route_freeze')
            or not record.get('runtime_namespace_proof')):
        raise ValueError('Exact isolated initial-budget pipeline modes required')
    expected = [legacy.base.local(value) for value in record['inputs'].values()]
    def exact_seeds(seeds, *, provenance_files=None):
        if sorted(map(legacy.base.local, seeds)) != sorted(expected):
            raise ValueError('Unexpected semantic seed outside exact pipeline inputs')
        return files(record['mode'], record['inputs'], root=ROOT,
                     provenance_files=provenance_files)
    # Process-local dependency injection into the unchanged launcher; no source
    # edits and no concurrency. All other snapshot/runtime verification remains.
    original = legacy.semantic_files
    try:
        legacy.semantic_files = exact_seeds
        closure = _required_inputs(record, provenance_files=provenance_files)
    finally:
        legacy.semantic_files = original
    extra = {'scripts/task10b_phase2_expand_v2.py', 'scripts/task10b_phase2_expand_v2.sbatch',
             'src/human_punching/phase2_pipeline_inputs_v2.py',
             'src/human_punching/phase2_runtime_pack_entry_v2.py',
             'src/human_punching/phase2_grouped_entry_v2.py'}
    for name in legacy.local_import_closure(extra):
        closure[name] = legacy.sha256_file(legacy.base.local(name))
    return dict(sorted(closure.items()))


def submission_command(record, output):
    command = _submission_command(record, output)
    index = command.index(str(ROOT/'scripts/task10b_phase2_expand.sbatch'))
    command[index] = str(ROOT/'scripts/task10b_phase2_expand_v2.sbatch')
    return command


def native_command(record, remaining_wall_s=None):
    if record['mode'] not in MODES:
        raise ValueError('No V2 native command for another mode')
    command = _native_command(record, remaining_wall_s)
    if record['mode'] == 'grouped-pilot':
        index = command.index('human_punching.phase2_grouped_learning')
        command[index] = 'human_punching.phase2_grouped_entry_v2'
        return command
    proof = legacy.bound(record['runtime_namespace_proof'])
    meta = legacy.read(proof)
    packet = ROOT/record['output']
    packet = packet.parent/'packet.json'
    index = command.index(str(ROOT/'scripts/task10b_phase2_runtime_pack.py'))
    command[index:index+1] = ['-m', 'human_punching.phase2_runtime_pack_entry_v2']
    return command + ['--logical-project-root', meta['local_root'],
        '--runtime-namespace-proof', str(proof), '--packet', str(packet),
        '--packet-sha256', legacy.sha256_file(packet)]


# All calls below share the durable V1 GPU budget ledger and source-object pool.
# The distinct schema/batch script prevents an old entry from accepting V2.
legacy.SCHEMA = SCHEMA
legacy.MODES = MODES
legacy.CONDITIONAL_MODES = ()
legacy.validate_spec = validate_spec
legacy.required_inputs = required_inputs
legacy.submission_command = submission_command
legacy.native_command = native_command


def prepare(output, mode, inputs, **kwargs):
    if kwargs.get('runtime_namespace_proof') is None:
        raise ValueError('V2 requires the exact reviewed runtime namespace proof')
    return legacy.prepare(output, mode, inputs, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--inputs', required=True)
    p.add_argument('--allocation-wall-seconds', type=float, required=True)
    p.add_argument('--application-wall-seconds', type=float, required=True)
    p.add_argument('--partition', default='mit_normal_gpu')
    p.add_argument('--runtime-namespace-proof', required=True)
    for name in ('verify', 'submit', 'execute', 'account'):
        sub.add_parser(name).add_argument('--packet', required=True)
    sub.add_parser('budget')
    args = parser.parse_args()
    if args.action == 'execute':
        return legacy.execute(args.packet)
    if args.action == 'prepare':
        result = prepare(args.output, args.mode, legacy.read(args.inputs),
            allocation_wall_seconds=args.allocation_wall_seconds,
            application_wall_seconds=args.application_wall_seconds,
            partition=args.partition, runtime_namespace_proof=args.runtime_namespace_proof)
    elif args.action == 'account':
        result = legacy.base.account(args.packet)
    elif args.action == 'budget':
        result = legacy.budget_status(legacy.base.ledger(), successful_route_frozen=False)
    else:
        result = getattr(legacy, args.action)(args.packet)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
