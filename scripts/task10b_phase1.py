#!/usr/bin/env python3
"""Prepare, verify and execute bounded frozen-policy Phase 1 packets; never submit."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from human_punching.wbc.provenance import runtime_record, sha256_file, writable_path, write_json

SCHEMA = 'human_task10b_phase1_packet_v1'
PYTHON = ROOT / '.venv-task10b/bin/python'
RUNTIME = ROOT / 'outputs/task10b/p1prep_20261020_01/runtime.json'
RUNTIME_ENV = RUNTIME.with_name('runtime.env')
LIMITS = {'pilot': (1800, 1680, 4 * 1024**3, 16), 'qualification': (3600, 3420, 6 * 1024**3, 64)}


def now():
    return datetime.now(timezone.utc).isoformat()


def helper(name):
    spec = importlib.util.spec_from_file_location('task10b_phase1_' + name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_input(path):
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    resolved = value.resolve(strict=True)
    resolved.relative_to(ROOT.resolve())
    return resolved


def required_inputs(record):
    paths = helper('task10_release_baseline').required_inputs(record)
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / 'src/human_punching').rglob('*.py'))
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / 'tests').glob('*phase1*.py'))
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / 'tests').glob('*functional_recovery*.py'))
    paths.update(('scripts/task10b_phase1.py', 'scripts/task10b_phase1.sbatch', 'scripts/task10b_phase0.py',
                  'scripts/task10b_phase1_audit.py', 'scripts/task10b_phase0_audit.py',
                  'scripts/task10b_phase1_summary.py',
                  'scripts/task10b_phase1_no_contact_fixture.py',
                  'src/human_punching/phase1_probe.py', 'tests/test_task10b_phase1_packet.py',
                  str(RUNTIME.relative_to(ROOT)), str(RUNTIME_ENV.relative_to(ROOT)),
                  str(RUNTIME.with_name('runtime_interpreter_manifest.json').relative_to(ROOT)),
                  '.venv-task10b/pyvenv.cfg',
                  '.venv-task10b/lib/python3.12/site-packages/task10b_readonly_packages.pth',
                  record['predicate']))
    for key in ('pilot_receipt', 'pilot_packet', 'pilot_audit', 'review'):
        if key in record:
            paths.add(record[key])
    return paths


def validate_scope(record):
    mode = record.get('mode')
    if mode not in LIMITS:
        raise ValueError('Only Phase 1 pilot and qualification are supported')
    allocation, application, products, worlds = LIMITS[mode]
    expected = dict(schema=SCHEMA, task='10B', program_phase=1, training=False,
        learning_updates=0, automatic_followup=False, solver='accepted_g1', feedforward='original',
        allocation_wall_seconds=allocation, application_wall_seconds=application,
        products_and_caches_bytes=products, conditions=4, environments_per_condition=worlds,
        requested_full_cycle_attempts=4 * worlds, bank_rows=list(range(worlds)))
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Phase 1 scope or bounded resources changed')
    if record.get('partition') not in ('mit_normal_gpu', 'mit_preemptable'):
        raise ValueError('Partition is outside Phase 1 authorization')
    nodes = record.get('excluded_nodes')
    if (not isinstance(nodes, list) or 'node3006' not in nodes or nodes != sorted(set(nodes))
            or any(not isinstance(n, str) or re.fullmatch(r'node[0-9]+', n) is None for n in nodes)):
        raise ValueError('Excluded nodes must include node3006 and use canonical names')


def verify_runtime_identity(record):
    runtime = json.loads(RUNTIME.read_text())
    local = local_input(runtime['local_python_root'])
    manifest = RUNTIME.with_name('runtime_interpreter_manifest.json')
    if sha256_file(manifest) != runtime['interpreter_manifest_sha256']:
        raise ValueError('Interpreter manifest changed')
    for item in json.loads(manifest.read_text()):
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid interpreter manifest path')
        member = local / relative
        if not member.resolve(strict=True).is_relative_to(local):
            raise ValueError('Interpreter member escaped local root')
        if 'symlink' in item:
            if not member.is_symlink() or os.readlink(member) != item['symlink']:
                raise ValueError('Interpreter symlink changed')
        elif member.is_symlink() or member.stat().st_size != item['bytes'] or sha256_file(member) != item['sha256']:
            raise ValueError('Interpreter payload changed')
    for key, expected in runtime['distribution_metadata_hashes'].items():
        package, filename = key.rsplit('/', 1)
        dist = importlib.metadata.distribution(package)
        # Distribution metadata path avoids the expensive files/RECORD stat expansion.
        metadata = local_input(Path(dist._path) / filename)
        if metadata.stat().st_size != expected['bytes'] or sha256_file(metadata) != expected['sha256']:
            raise ValueError('Installed distribution metadata changed: ' + key)
    current = runtime_record()
    if not current['declared_pins_match'] or current != record['runtime']:
        raise ValueError('Runtime pins or identity changed')


def scientific_files(record):
    evidence = {record[k] for k in ('predicate', 'pilot_receipt', 'pilot_packet', 'pilot_audit', 'review') if k in record}
    return {p: h for p, h in record['files'].items() if p not in evidence}


def verify_pilot_review(record):
    bound = {}
    for key in ('pilot_receipt', 'pilot_packet', 'pilot_audit', 'review'):
        path = local_input(record[key])
        if sha256_file(path) != record[key + '_sha256']:
            raise ValueError('Pilot review binding changed: ' + key)
        bound[key] = json.loads(path.read_text())
    previous, execution, audit, review = (bound[k] for k in ('pilot_packet', 'pilot_receipt', 'pilot_audit', 'review'))
    if previous.get('mode') != 'pilot' or execution.get('status') != 'passed':
        raise ValueError('A completed passed pilot is required')
    if execution.get('packet_sha256') != record['pilot_packet_sha256']:
        raise ValueError('Pilot execution does not bind its packet')
    if audit.get('status') != 'passed' or audit.get('packet_sha256') != record['pilot_packet_sha256']:
        raise ValueError('Independent pilot audit is missing or does not bind the pilot')
    expected_review = dict(status='reviewed', predicate_sha256=record['predicate_sha256'],
        pilot_packet_sha256=record['pilot_packet_sha256'], pilot_execution_sha256=record['pilot_receipt_sha256'],
        pilot_audit_sha256=record['pilot_audit_sha256'], thresholds_changed=False)
    if any(review.get(k) != v for k, v in expected_review.items()):
        raise ValueError('Predicate review does not bind unchanged thresholds and passed pilot evidence')
    for key in ('bundle', 'release', 'runtime', 'interpreter_path', 'interpreter_sha256'):
        if record[key] != previous[key]:
            raise ValueError('Qualification inputs differ from reviewed pilot: ' + key)
    if scientific_files(record) != scientific_files(previous):
        raise ValueError('Qualification scientific source inputs differ from reviewed pilot')
    # Verify the prior immutable source snapshot without recursively evaluating its review gate.
    verify(local_input(record['pilot_packet']), check_review=False)


def verify(packet, *, check_review=True):
    path = local_input(packet)
    record = json.loads(path.read_text())
    validate_scope(record)
    if not required_inputs(record).issubset(record.get('files', {})):
        raise ValueError('Incomplete frozen input closure')
    for relative, digest in record['files'].items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Invalid frozen input path')
        source = local_input(relative)
        archive = (path.parent / 'source_snapshot' / relative).resolve(strict=True)
        if sha256_file(source) != digest:
            raise ValueError('Frozen input changed: ' + relative)
        if not archive.is_relative_to(path.parent) or sha256_file(archive) != digest:
            raise ValueError('Frozen archive changed: ' + relative)
    if sha256_file(local_input(record['predicate'])) != record['predicate_sha256']:
        raise ValueError('Operational predicate changed')
    interpreter = PYTHON.resolve(strict=True)
    if not interpreter.is_relative_to(ROOT.resolve()) or sha256_file(interpreter) != record['interpreter_sha256']:
        raise ValueError('Human-owned interpreter identity changed')
    verify_runtime_identity(record)
    if record['mode'] == 'qualification' and check_review:
        verify_pilot_review(record)
    return record


def prepare(bundle, release, output, mode, *, predicate, partition='mit_normal_gpu',
            pilot=None, pilot_audit=None, review=None, excluded_nodes=('node3006',)):
    from human_punching.wbc.release_baseline_probe import load_release_inputs
    from human_punching.wbc.runtime.bundle import load_runtime_bundle
    if mode not in LIMITS:
        raise ValueError('Unsupported Phase 1 mode')
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() != 'wbc':
        raise ValueError('Task 10B requires human-punching branch wbc')
    checked = load_runtime_bundle(local_input(bundle))
    release, predicate = local_input(release), local_input(predicate)
    allocation, application, products, worlds = LIMITS[mode]
    load_release_inputs(checked, release, num_envs=worlds)
    destination = writable_path(output)
    destination.relative_to(ROOT / 'outputs/task10b')
    if len(os.fsencode(destination / 'run/cache/tmp')) > 120:
        raise ValueError('Use a shallow native compiler temporary path')
    record = dict(schema=SCHEMA, task='10B', program_phase=1, mode=mode, prepared_utc=now(),
        status='prepared_not_submitted', partition=partition, excluded_nodes=sorted(set(excluded_nodes)),
        bundle=str(checked.root.relative_to(ROOT)), release=str(release.relative_to(ROOT)),
        predicate=str(predicate.relative_to(ROOT)), predicate_sha256=sha256_file(predicate),
        output=str((destination / 'run').relative_to(ROOT)), solver='accepted_g1', feedforward='original',
        training=False, learning_updates=0, automatic_followup=False, production_qualified=False,
        allocation_wall_seconds=allocation, application_wall_seconds=application, products_and_caches_bytes=products,
        interpreter_path=str(PYTHON.resolve(strict=True)), interpreter_sha256=sha256_file(PYTHON),
        runtime=runtime_record(), conditions=4, environments_per_condition=worlds, bank_rows=list(range(worlds)),
        requested_full_cycle_attempts=4 * worlds,
        node3006_exclusion_reason='Repeated Phase0 and Task10 CUDA startup failures; no requalification performed.',
        authorization='2026-10-20 explicit Task10B Phase1 only; frozen policies first; no later phases or automatic training')
    if mode == 'qualification':
        if any(v is None for v in (pilot, pilot_audit, review)):
            raise ValueError('Qualification requires pilot execution, independent audit and predicate review')
        paths = dict(pilot_receipt=local_input(pilot), pilot_packet=local_input(pilot).parent/'packet.json',
                     pilot_audit=local_input(pilot_audit), review=local_input(review))
        for key, path in paths.items():
            record[key], record[key + '_sha256'] = str(path.relative_to(ROOT)), sha256_file(path)
    validate_scope(record)
    record['files'] = {name: sha256_file(local_input(name)) for name in sorted(required_inputs(record))}
    if mode == 'qualification':
        verify_pilot_review(record)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'logs').mkdir()
    for relative, digest in record['files'].items():
        archived = destination / 'source_snapshot' / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_input(relative), archived)
        if sha256_file(archived) != digest or sha256_file(local_input(relative)) != digest:
            raise ValueError('Input changed during archival: ' + relative)
    record['submission_command'] = ['sbatch', '--parsable', '--account=mit_general', '--partition=' + partition,
        '--nodes=1', '--ntasks=1', '--gres=gpu:l40s:1', '--cpus-per-task=4', '--mem=32G',
        '--time=' + ('00:30:00' if mode == 'pilot' else '01:00:00'), '--no-requeue',
        '--exclude=' + ','.join(record['excluded_nodes']), '--job-name=t10b-p1-' + mode, '--chdir=' + str(ROOT),
        '--output=' + str(destination/'logs/%j.out'), '--error=' + str(destination/'logs/%j.err'),
        str(ROOT/'scripts/task10b_phase1.sbatch'), str(destination/'packet.json')]
    write_json(destination/'packet.json', record)
    verify(destination/'packet.json')
    return dict(packet=str(destination/'packet.json'), files=len(record['files']),
                packet_sha256=sha256_file(destination/'packet.json'))


def cuda_health_receipt(destination):
    # Pure reuse of the qualified health check: same human-owned interpreter, no global mutation.
    return helper('task10b_phase0').cuda_health_receipt(destination)


def tree_bytes(root):
    return helper('task10b_phase0').tree_bytes(root)


def execute(packet):
    path = local_input(packet)
    if path.name != 'packet.json' or not path.is_relative_to(ROOT / 'outputs/task10b'):
        raise ValueError('Execution packet must be in the Task10B output root')
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Allocated Slurm CUDA execution is required')
    start = time.monotonic()
    receipt = dict(schema='human_task10b_phase1_execution_v1', status='running', started_utc=now(),
        packet_sha256=sha256_file(path), slurm_job_id=os.environ['SLURM_JOB_ID'], training=False,
        learning_updates=0, budget_stop=None, peak_sampled_products_bytes=0, native_processes_started=0)
    write_json(path.parent/'execution_started.json', receipt)
    code = 2
    try:
        record = verify(path)
        health_path = path.parent/'cuda_health.json'
        health = cuda_health_receipt(health_path)
        receipt.update(cuda_health_status=health['status'], cuda_health_sha256=sha256_file(health_path))
        if health['status'] != 'passed':
            raise RuntimeError('Allocated CUDA health check failed before native execution')
        output = writable_path(ROOT/record['output'])
        command = [str(PYTHON), '-B', '-m', 'human_punching.phase1_probe', '--mode', record['mode'],
            '--bundle', str(ROOT/record['bundle']), '--release', str(ROOT/record['release']),
            '--predicate', str(ROOT/record['predicate']), '--output', str(output)]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', WANDB_MODE='disabled')
        # Native probe exclusively creates run/. The already-created packet directory
        # safely confines startup logs without colliding with that exclusive creation.
        process = subprocess.Popen(command, cwd=path.parent, env=env, start_new_session=True)
        receipt.update(native_processes_started=1, native_command=command)
        try:
            while process.poll() is None:
                size = tree_bytes(path.parent)
                receipt['peak_sampled_products_bytes'] = max(size, receipt['peak_sampled_products_bytes'])
                if size > record['products_and_caches_bytes']:
                    receipt['budget_stop'] = 'products_and_caches_budget'
                elif time.monotonic()-start >= record['application_wall_seconds']:
                    receipt['budget_stop'] = 'application_wall_budget'
                if receipt['budget_stop']:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(2)
            code = process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
        audit_path = output/'audit.json'
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        receipt.update(mode=record['mode'], audit_status=audit.get('status'),
            audit_sha256=sha256_file(audit_path) if audit else None,
            actual_world_physics_transitions=audit.get('actual_world_physics_transitions'),
            unattempted_conditions=audit.get('unattempted_conditions'))
        receipt['status'] = 'passed' if code == 0 and not receipt['budget_stop'] and audit.get('status') == 'passed' else 'failed'
    except Exception as error:
        receipt.update(status='failed', exception=dict(type=type(error).__name__, message=str(error)))
    finally:
        if receipt['native_processes_started'] == 0:
            receipt.update(all_native_modes_unattempted=True, actual_world_physics_transitions=0, actual_actor_updates=0)
        receipt.update(wall_s=time.monotonic()-start, ended_utc=now(), returncode=code)
        write_json(path.parent/'execution.json', receipt)
    return 0 if receipt['status'] == 'passed' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    for name in ('bundle', 'release', 'output', 'predicate'):
        prep.add_argument('--'+name, type=Path, required=True)
    prep.add_argument('--mode', choices=tuple(LIMITS), required=True)
    prep.add_argument('--partition', choices=('mit_normal_gpu','mit_preemptable'), default='mit_normal_gpu')
    for name in ('pilot', 'pilot-audit', 'review'):
        prep.add_argument('--'+name, type=Path)
    prep.add_argument('--exclude-node', action='append', default=['node3006'])
    for name in ('verify', 'execute'):
        sub.add_parser(name).add_argument('--packet', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.bundle,args.release,args.output,args.mode,predicate=args.predicate,
            partition=args.partition,pilot=args.pilot,pilot_audit=args.pilot_audit,review=args.review,excluded_nodes=args.exclude_node)))
        return 0
    if args.action == 'verify':
        record = verify(args.packet)
        print(json.dumps(dict(status='passed',files=len(record['files']),mode=record['mode'])))
        return 0
    return execute(args.packet)


if __name__ == '__main__':
    raise SystemExit(main())
