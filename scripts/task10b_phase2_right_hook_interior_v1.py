#!/usr/bin/env python3
"""One explicitly declared RH 1.10s interior; no actor or integration.

The frozen expansion transport supplies owned caches, source snapshots, runtime
verification, aggregate accounting and watchdogs. Only this imported instance
has the new scope. All old launchers and packets remain unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location(
    'task10b_rh_interior_private_transport', ROOT / 'scripts/task10b_phase2_expand.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

MODE = 'right-hook-interior-v1'
SCHEMA = 'task10b_phase2_right_hook_interior_packet_v1'
ALLOCATION_SECONDS, APPLICATION_SECONDS, CHILD_SECONDS = 180, 150, 90
EVALUATION_RESERVE_SECONDS = 4500
PRODUCT_BYTES = 1024 ** 3
_required_inputs = legacy.required_inputs
_submission_command = legacy.submission_command
_admit_reservation = legacy.admit_reservation


def validate_spec(mode, inputs, wall_s, app_s, *, proof=None):
    if (mode != MODE or set(inputs) != {'plan'} or proof is not None
            or isinstance(wall_s, bool) or isinstance(app_s, bool)
            or wall_s != ALLOCATION_SECONDS or app_s != APPLICATION_SECONDS):
        raise ValueError('Exactly one RH interior with 180/150/90 second bounds required')
    from human_punching.phase2_right_hook_interior_v1 import validate_plan
    validate_plan(legacy.read(legacy.base.local(inputs['plan'])))
    return dict(family='right_hook', seed=21021, fresh_transition_cap=0,
                interior_cells=1, interior_ttc_s=1.10, child_wall_seconds=CHILD_SECONDS,
                actor_calls=0, explicit_integration_steps=0,
                evaluation_reserve_seconds=EVALUATION_RESERVE_SECONDS,
                physical_qualification=False)


def required_inputs(record, *, provenance_files=None):
    if (record['mode'] != MODE or record.get('successful_route_freeze')
            or not record.get('runtime_namespace_proof') or set(record['inputs']) != {'plan'}):
        raise ValueError('Only the exact interior plan and reviewed runtime proof are accepted')
    expected = legacy.base.local(record['inputs']['plan'])

    def exact(seeds, *, provenance_files=None):
        if list(map(legacy.base.local, seeds)) != [expected]:
            raise ValueError('Unexpected interior input seed')
        from human_punching.phase2_right_hook_interior_v1 import files
        return files(expected)

    original = legacy.semantic_files
    try:
        legacy.semantic_files = exact
        closure = _required_inputs(record, provenance_files=provenance_files)
    finally:
        legacy.semantic_files = original
    extra = {'scripts/task10b_phase2_right_hook_interior_v1.py',
             'scripts/task10b_phase2_right_hook_interior_v1.sbatch',
             'src/human_punching/phase2_right_hook_interior_v1.py'}
    for name in legacy.local_import_closure(extra):
        closure[name] = legacy.sha256_file(legacy.base.local(name))
    return dict(sorted(closure.items()))


def submission_command(record, output):
    command = _submission_command(record, output)
    index = command.index(str(ROOT / 'scripts/task10b_phase2_expand.sbatch'))
    command[index] = str(ROOT / 'scripts/task10b_phase2_right_hook_interior_v1.sbatch')
    return command


def protected_reservation(allocations, seconds, *, evaluation_reserve_seconds=0,
                          successful_route_frozen=False):
    if successful_route_frozen or seconds != ALLOCATION_SECONDS:
        raise ValueError('Only the initial-budget 180-second generation reservation is supported')
    return _admit_reservation(allocations, seconds,
        evaluation_reserve_seconds=max(EVALUATION_RESERVE_SECONDS, evaluation_reserve_seconds),
        successful_route_frozen=False)


def native_command(record, remaining_wall_s=None):
    if (record['mode'] != MODE or record.get('child_wall_seconds') != CHILD_SECONDS
            or record['application_wall_seconds'] != APPLICATION_SECONDS
            or record['allocation_wall_seconds'] != ALLOCATION_SECONDS
            or record.get('interior_cells') != 1 or record.get('interior_ttc_s') != 1.10
            or record.get('actor_calls') != 0 or record.get('explicit_integration_steps') != 0
            or record.get('training') is not False or set(record['inputs']) != {'plan'}):
        raise ValueError('Native dispatch requires the fixed generation-only packet')
    available = APPLICATION_SECONDS - 30 if remaining_wall_s is None else remaining_wall_s
    if (isinstance(available, bool) or not isinstance(available, (int, float))
            or not math.isfinite(available) or not CHILD_SECONDS <= available <= APPLICATION_SECONDS - 30):
        raise RuntimeError('Setup consumed the fixed child margin; no truncated generation')
    return [str(legacy.base.PYTHON), '-B', '-m', 'human_punching.phase2_right_hook_interior_v1',
            '--plan', str(legacy.base.local(record['inputs']['plan'])),
            '--output', str(ROOT / record['output']), '--wall-seconds', str(float(CHILD_SECONDS))]


legacy.SCHEMA = SCHEMA
legacy.MODES = (MODE,)
legacy.CONDITIONAL_MODES = ()
legacy.CAP = PRODUCT_BYTES
legacy.validate_spec = validate_spec
legacy.required_inputs = required_inputs
legacy.submission_command = submission_command
legacy.native_command = native_command
legacy.admit_reservation = protected_reservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--partition', default='mit_normal_gpu')
    p.add_argument('--runtime-namespace-proof', required=True)
    for action in ('verify', 'submit', 'execute', 'account'):
        sub.add_parser(action).add_argument('--packet', required=True)
    sub.add_parser('budget')
    args = parser.parse_args()
    if args.action == 'execute':
        return legacy.execute(args.packet)
    if args.action == 'prepare':
        result = legacy.prepare(args.output, MODE, {'plan': args.plan},
            allocation_wall_seconds=ALLOCATION_SECONDS,
            application_wall_seconds=APPLICATION_SECONDS,
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
