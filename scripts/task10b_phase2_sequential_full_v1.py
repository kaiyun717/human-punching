#!/usr/bin/env python3
"""Exact64/24 .205rad sequential reference construction; old probes remain immutable.

This entry reuses the original Phase2 runtime, budget lock, source snapshots,
watchdog and accounting. It supports only the complete predefined training and validation catalogs.
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
spec = importlib.util.spec_from_file_location('task10b_phase2_sequential_full_v1_frozen_driver',
    ROOT/'scripts/task10b_phase2_expand.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

MODE = 'sequential-target-references'
SCHEMA = 'task10b_phase2_sequential_full_packet_v1'
from human_punching.phase2_sequential_full_catalog_v1 import LIMITS
_required_inputs = legacy.required_inputs
_submission_command = legacy.submission_command


def files(plan_path):
    """Exact reviewed target edges with a different explicitly bound solver."""
    from human_punching import phase2_target_inputs as target_inputs
    from human_punching.phase2_sequential_full_catalog_v1 import resolve_cells
    original = target_inputs.resolve_cells
    try:
        # This single-threaded process-local dependency restores on every exit;
        # the original ownership, opaque ancestry and model closure stay intact.
        target_inputs.resolve_cells = resolve_cells
        closure = target_inputs.files(plan_path)
    finally:
        target_inputs.resolve_cells = original
    plan = legacy.read(legacy.base.local(plan_path))
    probe = legacy.read(legacy.bound(plan['probe_plan']))
    for item in (*plan['sources'].values(), *plan['original_plans'].values(),
                 probe['engineering_bound_review'], probe['selection_audit']):
        path = legacy.bound(item)
        closure[str(path.relative_to(ROOT))] = item['sha256']
    for relative, digest in closure.items():
        if legacy.sha256_file(legacy.base.local(relative)) != digest:
            raise ValueError('Sequential input changed during closure')
    return dict(sorted(closure.items()))


def validate_spec(mode, inputs, wall_s, app_s, *, proof=None):
    if mode != MODE or set(inputs) != {'plan'} or proof is not None:
        raise ValueError('Exact full sequential plan required')
    from human_punching.phase2_sequential_full_catalog_v1 import resolve_cells
    plan = legacy.read(legacy.base.local(inputs['plan']))
    settings, cells, _ = resolve_cells(plan)
    count, allocation, application, child = LIMITS[plan['split']]
    if (type(wall_s) not in (int,float) or type(app_s) not in (int,float)
            or wall_s != allocation or app_s != application
            or len(cells) != count or settings.base.maximum_joint_delta_rad != .205):
        raise ValueError('Exact split limits and unchanged .205 correction required')
    return dict(family='jab', seed=21021, fresh_transition_cap=0, sequential_cells=count,
        sequential_split=plan['split'], sequential_generator_output_byte_cap=plan['output_byte_cap'],
        sequential_child_wall_seconds=child, sequential_path_solver_version=plan['path_solver_version'],
        sequential_engineering_correction_cap_rad=.205, physical_joint_limits_relaxed=False,
        native_frozen_control_required_before_learning=True)


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
    extra = {'scripts/task10b_phase2_sequential_full_v1.py', 'scripts/task10b_phase2_sequential_full_v1.sbatch',
        'src/human_punching/phase2_sequential_full_catalog_v1.py',
        'src/human_punching/phase2_target_path_reference_v3.py',
        'src/human_punching/phase2_sequential_target_catalog_v3.py',
        'src/human_punching/phase2_target_inputs.py'}
    for name in legacy.local_import_closure(extra):
        closure[name] = legacy.sha256_file(legacy.base.local(name))
    return dict(sorted(closure.items()))


def submission_command(record, output):
    command = _submission_command(record, output)
    index = command.index(str(ROOT/'scripts/task10b_phase2_expand.sbatch'))
    command[index] = str(ROOT/'scripts/task10b_phase2_sequential_full_v1.sbatch')
    return command


def native_command(record, remaining_wall_s=None):
    split = record.get('sequential_split')
    if split not in LIMITS: raise ValueError('Exact full sequential split required')
    count, allocation, application, child = LIMITS[split]
    if (record['mode'] != MODE or record.get('sequential_cells') != count
            or record.get('sequential_child_wall_seconds') != child
            or record['application_wall_seconds'] != application
            or record['allocation_wall_seconds'] != allocation
            or record.get('sequential_engineering_correction_cap_rad') != .205
            or record.get('physical_joint_limits_relaxed') is not False
            or record.get('native_frozen_control_required_before_learning') is not True
            or set(record['inputs']) != {'plan'}):
        raise ValueError('Exact finite full sequential packet required')
    available = application-30 if remaining_wall_s is None else remaining_wall_s
    if (type(available) not in (int,float) or not math.isfinite(available)
            or not child <= available <= application-30):
        raise RuntimeError('Setup consumed the fixed child margin; no truncated catalog is launched')
    return [str(legacy.base.PYTHON), '-B', '-m', 'human_punching.phase2_sequential_full_catalog_v1',
        '--plan', str(legacy.base.local(record['inputs']['plan'])), '--output', str(ROOT/record['output']),
        '--wall-seconds', str(float(child))]


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
