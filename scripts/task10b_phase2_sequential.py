#!/usr/bin/env python3
"""Isolated four-cell sequential IK launcher; immutable original entries remain.

This entry reuses the original Phase2 runtime, budget lock, source snapshots,
watchdog and accounting. It supports one metadata-bound construction probe only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
spec = importlib.util.spec_from_file_location('task10b_phase2_sequential_frozen_driver',
    ROOT/'scripts/task10b_phase2_expand.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

MODE = 'sequential-target-references'
SCHEMA = 'task10b_phase2_sequential_reference_packet_v2'
ALLOCATION_SECONDS, APPLICATION_SECONDS, CHILD_SECONDS = 360, 300, 240
_required_inputs = legacy.required_inputs
_submission_command = legacy.submission_command


def files(plan_path):
    """Exact reviewed target edges with a different explicitly bound solver."""
    from human_punching import phase2_target_inputs as target_inputs
    from human_punching.phase2_sequential_target_catalog import resolve_cells
    original = target_inputs.resolve_cells
    try:
        # This single-threaded process-local dependency restores on every exit;
        # the original ownership, opaque ancestry and model closure stay intact.
        target_inputs.resolve_cells = resolve_cells
        closure = target_inputs.files(plan_path)
    finally:
        target_inputs.resolve_cells = original
    plan = legacy.read(legacy.base.local(plan_path))
    for item in plan['sources'].values():
        path = legacy.bound(item)
        closure[str(path.relative_to(ROOT))] = item['sha256']
    for relative, digest in closure.items():
        if legacy.sha256_file(legacy.base.local(relative)) != digest:
            raise ValueError('Sequential input changed during closure')
    return dict(sorted(closure.items()))


def validate_spec(mode, inputs, wall_s, app_s, *, proof=None):
    if (mode != MODE or set(inputs) != {'plan'} or proof is not None
            or isinstance(wall_s, bool) or isinstance(app_s, bool)
            or wall_s != ALLOCATION_SECONDS or app_s != APPLICATION_SECONDS
            or legacy.rounded_wall(wall_s) != wall_s):
        raise ValueError('Only exact four-cell sequential probe with360/300/240s limits is supported')
    from human_punching.phase2_sequential_target_catalog import resolve_cells, MAX_OUTPUT_BYTES, MAX_CHILD_SECONDS
    plan = legacy.read(legacy.base.local(inputs['plan']))
    _, cells, _ = resolve_cells(plan)
    if len(cells) != 4 or MAX_CHILD_SECONDS != CHILD_SECONDS:
        raise ValueError('Four cells and immutable240second child limit required')
    return dict(family='jab', seed=21021, fresh_transition_cap=0, sequential_probe_cells=4,
        sequential_generator_output_byte_cap=MAX_OUTPUT_BYTES, sequential_child_wall_seconds=CHILD_SECONDS,
        sequential_path_solver_version=plan['path_solver_version'])


def required_inputs(record, *, provenance_files=None):
    if (record['mode'] != MODE or record.get('successful_route_freeze')
            or not record.get('runtime_namespace_proof') or set(record['inputs']) != {'plan'}):
        raise ValueError('Exact sequential probe inputs and reviewed runtime proof required')
    expected = legacy.base.local(record['inputs']['plan'])
    def exact(seeds, *, provenance_files=None):
        if list(map(legacy.base.local, seeds)) != [expected]:
            raise ValueError('Unexpected sequential semantic input seed')
        return files(expected)
    original = legacy.semantic_files
    try:
        legacy.semantic_files = exact
        closure = _required_inputs(record, provenance_files=provenance_files)
    finally:
        legacy.semantic_files = original
    extra = {'scripts/task10b_phase2_sequential.py', 'scripts/task10b_phase2_sequential.sbatch',
        'src/human_punching/phase2_sequential_target_catalog.py',
        'src/human_punching/phase2_target_path_reference_v2.py',
        'src/human_punching/phase2_target_inputs.py'}
    for name in legacy.local_import_closure(extra):
        closure[name] = legacy.sha256_file(legacy.base.local(name))
    return dict(sorted(closure.items()))


def submission_command(record, output):
    command = _submission_command(record, output)
    index = command.index(str(ROOT/'scripts/task10b_phase2_expand.sbatch'))
    command[index] = str(ROOT/'scripts/task10b_phase2_sequential.sbatch')
    return command


def native_command(record, remaining_wall_s=None):
    if (record['mode'] != MODE or record.get('sequential_child_wall_seconds') != CHILD_SECONDS
            or record['application_wall_seconds'] != APPLICATION_SECONDS
            or record['allocation_wall_seconds'] != ALLOCATION_SECONDS
            or set(record['inputs']) != {'plan'}):
        raise ValueError('Sequential native dispatch requires the exact immutable probe packet')
    available = APPLICATION_SECONDS-30 if remaining_wall_s is None else remaining_wall_s
    if (isinstance(available, bool) or not isinstance(available, (int, float))
            or not math.isfinite(available) or not CHILD_SECONDS <= available <= APPLICATION_SECONDS-30):
        raise RuntimeError('Setup consumed the fixed sequential child margin; no truncated probe is launched')
    return [str(legacy.base.PYTHON), '-B', '-m', 'human_punching.phase2_sequential_target_catalog',
        '--plan', str(legacy.base.local(record['inputs']['plan'])), '--output', str(ROOT/record['output']),
        '--wall-seconds', str(float(CHILD_SECONDS))]


legacy.SCHEMA = SCHEMA
legacy.MODES = (MODE,)
legacy.CONDITIONAL_MODES = ()
legacy.validate_spec = validate_spec
legacy.required_inputs = required_inputs
legacy.submission_command = submission_command
legacy.native_command = native_command


def prepare(output, mode, inputs, **kwargs):
    if kwargs.get('runtime_namespace_proof') is None:
        raise ValueError('Exact reviewed namespace proof required')
    return legacy.prepare(output, mode, inputs, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=(MODE,), required=True)
    p.add_argument('--inputs', required=True)
    p.add_argument('--allocation-wall-seconds', type=float, required=True)
    p.add_argument('--application-wall-seconds', type=float, required=True)
    p.add_argument('--partition', default='mit_normal_gpu')
    p.add_argument('--runtime-namespace-proof', required=True)
    for name in ('verify', 'submit', 'execute', 'account'):
        sub.add_parser(name).add_argument('--packet', required=True)
    sub.add_parser('budget')
    args = parser.parse_args()
    if args.action == 'execute': return legacy.execute(args.packet)
    if args.action == 'prepare':
        result = prepare(args.output, args.mode, legacy.read(args.inputs),
            allocation_wall_seconds=args.allocation_wall_seconds,
            application_wall_seconds=args.application_wall_seconds,
            partition=args.partition, runtime_namespace_proof=args.runtime_namespace_proof)
    elif args.action == 'account': result = legacy.base.account(args.packet)
    elif args.action == 'budget': result = legacy.budget_status(legacy.base.ledger(), successful_route_frozen=False)
    else: result = getattr(legacy, args.action)(args.packet)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
