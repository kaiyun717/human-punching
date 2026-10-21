#!/usr/bin/env python3
"""Bounded RH eight-update grouped original-PPO pilot with reviewed expanded clocks.

This entry reuses the original Phase2 runtime, budget lock, source snapshots,
watchdog and accounting. Only the exact transferred RH specialist, initial-budget recipe and two96-trial validation panels are supported.
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
spec = importlib.util.spec_from_file_location('task10b_phase2_right_hook_grouped_v1_frozen_driver',
    ROOT/'scripts/task10b_phase2_expand.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

MODE = 'grouped-pilot'
SCHEMA = 'task10b_phase2_right_hook_grouped_packet_v1'
from human_punching.phase2_right_hook_grouped_inputs_v1 import VERSION, KEYS
_required_inputs = legacy.required_inputs
_validate_spec = legacy.validate_spec
_submission_command = legacy.submission_command
_admit_reservation = legacy.admit_reservation
EVALUATION_RESERVE_SECONDS = 4500


def files(inputs, *, provenance_files=None):
    from human_punching.phase2_right_hook_grouped_inputs_v1 import files as exact_files
    return exact_files(inputs, root=ROOT, provenance_files=provenance_files)


def validate_spec(mode, inputs, wall_s, app_s, *, proof=None):
    if (mode != MODE or set(inputs) != KEYS or proof is not None
            or type(wall_s) not in (int,float) or type(app_s) not in (int,float)
            or wall_s != 960 or app_s != 900):
        raise ValueError('Exact initial-budget RH8 960/900 allocation/application required')
    scope=_validate_spec(mode, {'configuration':inputs['configuration']}, wall_s, app_s, proof=None)
    config=legacy.read(inputs['configuration']);budget=config['budget']
    if (scope['family']!='right_hook' or scope['seed']!=21021 or scope['fresh_transition_cap']!=65536
            or budget.get('environments')!=128 or budget.get('rollout_steps')!=64
            or budget.get('maximum_updates')!=8 or budget.get('maximum_wall_s')!=720
            or budget.get('shutdown_reserve_s')!=90 or budget.get('validation_every')!=8):
        raise ValueError('Exact128x64 eight-update fresh-transfer and720s/90s budget required')
    return dict(**scope, grouped_child_wall_seconds=720, evaluation_reserve_seconds=4500,
        grouped_inputs_version=VERSION, expanded_event_clock_version='task10b_phase2_expanded_event_clocks_v1',
        grouped_integrity_gate_version='task10b_phase2_grouped_entry_integrity_v3',
        planned_validation_updates=[0,8], planned_validation_attempts=192)


def required_inputs(record, *, provenance_files=None):
    if (record['mode']!=MODE or record.get('successful_route_freeze')
            or not record.get('runtime_namespace_proof') or set(record['inputs'])!=KEYS):
        raise ValueError('Exact initial-budget grouped inputs and runtime proof required')
    expected=sorted(legacy.base.local(p) for p in record['inputs'].values())
    def exact(seeds, *, provenance_files=None):
        if sorted(legacy.base.local(p) for p in seeds)!=expected:
            raise ValueError('Unexpected RH grouped semantic seed')
        return files(record['inputs'],provenance_files=provenance_files)
    original=legacy.semantic_files
    try:
        legacy.semantic_files=exact;closure=_required_inputs(record,provenance_files=provenance_files)
    finally:legacy.semantic_files=original
    extra={'scripts/task10b_phase2_right_hook_grouped_v1.py','scripts/task10b_phase2_right_hook_grouped_v1.sbatch',
        'src/human_punching/phase2_right_hook_grouped_inputs_v1.py',
        'src/human_punching/phase2_grouped_clock_entry_v2.py'}
    for name in legacy.local_import_closure(extra):closure[name]=legacy.sha256_file(legacy.base.local(name))
    return dict(sorted(closure.items()))


def submission_command(record, output):
    command = _submission_command(record, output)
    index = command.index(str(ROOT/'scripts/task10b_phase2_expand.sbatch'))
    command[index] = str(ROOT/'scripts/task10b_phase2_right_hook_grouped_v1.sbatch')
    return command


def native_command(record, remaining_wall_s=None):
    if (record.get('mode')!=MODE or record.get('family')!='right_hook' or record.get('training') is not True
            or record.get('fresh_transition_cap')!=65536 or record.get('grouped_child_wall_seconds')!=720
            or record.get('evaluation_reserve_seconds')!=4500 or record.get('allocation_wall_seconds')!=960
            or record.get('application_wall_seconds')!=900 or record.get('planned_validation_updates')!=[0,8]
            or record.get('planned_validation_attempts')!=192 or set(record['inputs'])!=KEYS):
        raise ValueError('Exact RH8 grouped packet required')
    available=870 if remaining_wall_s is None else remaining_wall_s
    if type(available) not in (int,float) or not math.isfinite(available) or not 720<=available<=870:
        raise RuntimeError('Setup consumed the fixed720s child margin; no shortened pilot is launched')
    config=legacy.base.local(record['inputs']['configuration'])
    return [str(legacy.base.PYTHON),'-B','-m','human_punching.phase2_grouped_clock_entry_v2',
        '--configuration',str(config),'--configuration-sha256',record['files'][str(config.relative_to(ROOT))],
        '--output',str(ROOT/record['output'])]


def protected_reservation(allocations, seconds, *, evaluation_reserve_seconds=0,
                          successful_route_frozen=False):
    if successful_route_frozen or seconds != 960:
        raise ValueError('Only the bounded initial-budget RH8 pilot reservation is supported')
    return _admit_reservation(allocations,seconds,
        evaluation_reserve_seconds=max(EVALUATION_RESERVE_SECONDS,evaluation_reserve_seconds),
        successful_route_frozen=False)


legacy.SCHEMA = SCHEMA
legacy.CAP = 3*1024**3
legacy.admit_reservation = protected_reservation
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
