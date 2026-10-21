#!/usr/bin/env python3
"""Source-bound transport for the single reviewed cross learning-curve probe.

Reuse the immutable-source verifier, durable GPU ledger/reservation lock, normal
partition limits and watchdog. This entry has no conditional-budget mode.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
spec = importlib.util.spec_from_file_location('task10b_curve_frozen_transport',
                                             ROOT/'scripts/task10b_phase2_expand.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
from human_punching import phase2_timing_pilot_curve_v2 as curve

SCHEMA = 'task10b_phase2_cross_learning_curve_packet_v2'
_required_inputs = legacy.required_inputs
_submission_command = legacy.submission_command


def validate_spec(mode, inputs, wall_s, app_s, *, proof=None):
    if (mode != 'timing-pilot' or set(inputs) != {'plan'} or proof is not None
            or type(wall_s) not in (int, float) or wall_s != 480
            or type(app_s) not in (int, float) or app_s != 420):
        raise ValueError('Only exact480/420s initial-budget cross learning curve is supported')
    curve.validate_plan(legacy.read(inputs['plan']))
    return dict(family='cross', seed=21021, fresh_transition_cap=curve.CAP,
        learning_curve_version=curve.VERSION,
        d0_clock_versions=dict(evaluator='task10b_phase2_d0_postprocessing_clock_v2',
                              independent_audit='task10b_phase2_independent_clock_v2'))


def curve_files(plan_path):
    plan_path = legacy.base.local(plan_path)
    plan = legacy.read(plan_path)
    _, rows, _ = curve.validate_plan(plan)
    closure = {str(plan_path.relative_to(ROOT)): legacy.sha256_file(plan_path)}
    def exact(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256'}:
                path = legacy.bound(value)
                closure[str(path.relative_to(ROOT))] = value['sha256']
            else:
                for child in value.values(): exact(child)
        elif isinstance(value, list):
            for child in value: exact(child)
    exact(plan)
    parent = legacy.read(legacy.bound(plan['parent_plan']))
    exact(parent)
    for name in ('route_receipt', 'video_receipt'):
        exact(legacy.read(legacy.bound(parent[name])))
    for name in ('numerical_review', 'video_review_v2'):
        exact(legacy.read(legacy.bound(plan[name])))
    for row in rows:
        for name in ('reference', 'legacy_runtime'):
            exact(dict(path=row[name+'_path'], sha256=row[name+'_sha256']))
        if row.get('diagnostic_screen_path'):
            path = legacy.base.local(row['diagnostic_screen_path'])
            closure[str(path.relative_to(ROOT))] = legacy.sha256_file(path)
    return closure


def required_inputs(record, *, provenance_files=None):
    if (record['mode'] != 'timing-pilot' or record.get('successful_route_freeze')
            or not record.get('runtime_namespace_proof')):
        raise ValueError('Exact initial-budget learning curve with namespace proof required')
    closure = curve_files(record['inputs']['plan'])
    # The baseline reference source inventory is reused only as an inventory.
    # Actual packet mode/training flag stay timing-pilot; no old native command
    # or old 16-update plan validator is used by this separate transport.
    inventory = deepcopy(record)
    inventory.update(mode='timing-references', inputs={})
    previous = legacy.semantic_files
    def exact_seeds(seeds, *, provenance_files=None):
        if seeds: raise ValueError('Unexpected baseline inventory seeds')
        return dict(closure)
    try:
        legacy.semantic_files = exact_seeds
        result = _required_inputs(inventory, provenance_files=provenance_files)
    finally:
        legacy.semantic_files = previous
    extra = set(curve.SOURCES) | {'scripts/task10b_phase2_audit_clock_v2.py'}
    for name in legacy.local_import_closure(extra):
        result[name] = legacy.sha256_file(legacy.base.local(name))
    return dict(sorted(result.items()))


def submission_command(record, output):
    command = _submission_command(record, output)
    index = command.index(str(ROOT/'scripts/task10b_phase2_expand.sbatch'))
    command[index] = str(ROOT/'scripts/task10b_phase2_curve_v2.sbatch')
    return command


def native_command(record, remaining_wall_s=None):
    if record['mode'] != 'timing-pilot':raise ValueError('Unknown curve command')
    limit = record['application_wall_seconds']-30
    if remaining_wall_s is not None:
        if (type(remaining_wall_s) not in (int, float) or not math.isfinite(remaining_wall_s)
                or not 360 <= remaining_wall_s <= limit):
            raise ValueError('Fixed360s child no longer fits the remaining deadline')
    elif limit < 360:
        raise ValueError('Fixed360s child exceeds wrapper deadline')
    return [str(legacy.base.PYTHON), '-B', '-m', 'human_punching.phase2_timing_pilot_curve_v2',
        '--plan', str(legacy.base.local(record['inputs']['plan'])),
        '--output', str(ROOT/record['output'])]


legacy.SCHEMA = SCHEMA
legacy.MODES = ('timing-pilot',)
legacy.CONDITIONAL_MODES = ()
legacy.validate_spec = validate_spec
legacy.required_inputs = required_inputs
legacy.submission_command = submission_command
legacy.native_command = native_command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--partition', default='mit_normal_gpu')
    p.add_argument('--runtime-namespace-proof', required=True)
    for name in ('verify', 'submit', 'execute', 'account'):
        sub.add_parser(name).add_argument('--packet', required=True)
    sub.add_parser('budget')
    args = parser.parse_args(argv)
    if args.action == 'execute':return legacy.execute(args.packet)
    if args.action == 'prepare':
        result = legacy.prepare(args.output, 'timing-pilot', {'plan': args.plan},
            allocation_wall_seconds=480, application_wall_seconds=420,
            partition=args.partition, runtime_namespace_proof=args.runtime_namespace_proof)
    elif args.action == 'account':result = legacy.base.account(args.packet)
    elif args.action == 'budget':result = legacy.budget_status(legacy.base.ledger(), successful_route_frozen=False)
    else:result = getattr(legacy, args.action)(args.packet)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':raise SystemExit(main())
