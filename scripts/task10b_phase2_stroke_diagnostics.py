#!/usr/bin/env python3
"""Bounded CPU plots/arrays for every audited expanded/shared native attempt."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import numpy as np
from human_punching.phase2_cache import _preflight
from human_punching.phase2_deadline import ExecutionDeadline
from human_punching.phase2_expanded_evaluation import load_declaration
from human_punching.phase2_stroke_diagnostics import (VERSION, MAX_WORKSPACE_BYTES,
    diagnostic_arrays, write_review_plots)
from human_punching.wbc.provenance import writable_path, write_json, sha256_file
from task10b_phase2_expanded_media import owned, source_runs, index_batches
from task10b_phase2_media import finite_render_prefix

SCHEMA = "task10b_phase2_stroke_diagnostic_ledger_v1"
FIELDS = ("qpos", "qvel", "pre_qref", "pre_vref", "reference_tick", "pre_reference_tick")


def extract_exact_world(path, digest, binding):
    """Header-bound subset decode; no contact arrays or other worlds retained."""
    header = _preflight(Path(path), digest, 2*1024**3)
    schemas = {row.name: row for row in header.arrays}
    if not set(FIELDS) <= set(schemas):
        raise ValueError("Every exact recorded drive/native state field is required")
    selected_bytes = sum(schemas[key].nbytes for key in FIELDS)
    if selected_bytes > MAX_WORKSPACE_BYTES:
        raise ValueError("Selected native batch payload exceeds256MiB decode allowance")
    world, worlds = binding['world'], binding['planned_worlds']
    n = schemas['qpos'].shape[0]
    if (not 1 <= worlds <= 16 or not 0 <= world < worlds
            or not 1 <= n <= binding['planned_samples'] <= 6205
            or schemas['qpos'].shape != (n, worlds, 55)
            or schemas['qvel'].shape != (n, worlds, 54)
            or schemas['reference_tick'].shape != (n, worlds)):
        raise ValueError("Exact bounded native world/horizon shape required")
    p = schemas['pre_qref'].shape[0]
    if (not 0 <= p <= n or schemas['pre_qref'].shape != (p, worlds, 55)
            or schemas['pre_vref'].shape != (p, worlds, 54)
            or schemas['pre_reference_tick'].shape != (p, worlds)):
        raise ValueError("Recorded pre-integration arrays have inconsistent interval counts")
    data = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in FIELDS:
            batch = archive[key]
            data[key] = batch[:, world].copy()
            del batch
    valid, prefix = finite_render_prefix(data['qpos'], data['qvel'])
    if valid == 0:
        return None, dict(prefix, reason="No finite valid-quaternion native prefix", selected_batch_payload_bytes=selected_bytes)
    m = min(max(0, valid-1), p)
    drive_good = np.isfinite(data['pre_qref'][:m]).all(axis=1) & np.isfinite(data['pre_vref'][:m]).all(axis=1)
    drive_good &= abs(np.linalg.norm(data['pre_qref'][:m, 3:7], axis=1)-1.) <= 1.e-5
    invalid = np.flatnonzero(~drive_good)
    paired = int(invalid[0]) if len(invalid) else m
    result = dict(qpos=data['qpos'][:valid], qvel=data['qvel'][:valid],
        reference_qpos=data['pre_qref'][:paired], reference_qvel=data['pre_vref'][:paired],
        reference_ticks=data['reference_tick'][:valid], drive_reference_ticks=data['pre_reference_tick'][:paired])
    prefix.update(planned_samples=binding['planned_samples'], retained_samples=n,
        incomplete_native_horizon=n < binding['planned_samples'],
        finite_physical_samples=valid, paired_preintegration_samples=paired,
        invalid_drive_sample=None if not len(invalid) else int(invalid[0]),
        unmatched_preintegration_samples=p-paired,
        full_retained_trace_has_all_interval_drives=paired == n-1 and valid == n,
        selected_batch_payload_bytes=selected_bytes,
        input_array_decode_allowance_bytes=MAX_WORKSPACE_BYTES,
        bound_scope="Selected batch fields, one field decoded at a time;256MiB diagnostic array allowance is separate from interpreter/model/plot overhead")
    return result, prefix


def audit_bindings(audit, run, roots):
    if (audit.get('schema') != 'task10b_phase2_independent_trace_audit_v1'
            or Path(audit['run']).resolve(strict=True) != run):
        raise ValueError("Exact run independent native audit required")
    runtime = audit.get('runtime_model')
    runtime_sha = None
    if runtime is not None and audit.get('native_runtime_model_present') is True:
        model_path = owned(runtime['path'])
        if sha256_file(model_path) != runtime['sha256']:
            raise ValueError("Independently audited runtime model changed")
        runtime_sha = runtime['sha256']
    result = {}
    for panel in audit.get('panels', []):
        folder = Path(panel['path']).resolve(strict=True)
        if folder.parent not in roots:
            raise ValueError("Diagnostic audit trace escapes verified execution lineage")
        raw = owned(folder/'raw_trace.npz', parent=folder.parent)
        if sha256_file(raw) != panel['raw_sha256']:
            raise ValueError("Independently audited native trace changed")
        for row in panel.get('trials', []):
            key = row['trial_id']
            if key in result:raise ValueError("Native diagnostic trial audited twice")
            result[key] = dict(raw_trace_path=str(raw), raw_trace_sha256=panel['raw_sha256'],
                audit_trial_pass=row.get('passed') is True, audit_panel_pass=panel.get('passed') is True,
                execution_complete=row.get('execution_complete'), audit_trial=row,
                audited_runtime_model_sha256=runtime_sha)
    return result


def diagnose(run, audit_path, audit_sha256, output, *, wall_seconds=1800., product_byte_cap=2*1024**3):
    """Keep all declared IDs; missing, invalid and interrupted evidence stays so."""
    if type(product_byte_cap) is not int or not 4*1024**2 <= product_byte_cap <= 2*1024**3:
        raise ValueError("Finite4MiB..2GiB product allowance required, including final ledger reserve")
    started = time.monotonic()
    deadline = ExecutionDeadline(wall_seconds, serialization_reserve_s=min(60., wall_seconds/4),
        session_started_monotonic=started)
    run = Path(run).resolve(strict=True)
    run.relative_to(ROOT/'outputs/task10b')
    audit_path = owned(audit_path)
    if sha256_file(audit_path) != audit_sha256:raise ValueError("Immutable audit identity required")
    audit = json.loads(audit_path.read_text())
    declaration_path = run/'declaration.json'
    if not declaration_path.exists():declaration_path = run/'physical_declaration.json'
    declaration_path = owned(declaration_path, parent=run)
    declaration = load_declaration(declaration_path)
    if len(declaration.bindings) not in (96, 192):raise ValueError("Declared96/192 physical panel required")
    launch_path = owned(run/'launch.json', parent=run)
    launch = json.loads(launch_path.read_text())
    if launch.get('declaration_identity', launch.get('physical_request_template_identity')) != declaration.identity:
        raise ValueError("Diagnostic declaration differs from exact native launch")
    catalog_path = owned(launch['catalog'])
    if sha256_file(catalog_path) != launch['catalog_sha256']:raise ValueError("Frozen catalog changed")
    catalog = json.loads(catalog_path.read_text())
    cells = {row['candidate_id']: row for row in catalog['cells']}
    if len(cells) != len(catalog['cells']):raise ValueError("Duplicate candidate catalog entry")
    roots = source_runs(run)
    audited = audit_bindings(audit, run, roots)
    index = {}
    for root in roots:
        for key, value in index_batches(root, declaration, cells).items():
            if key in index:raise ValueError("Attempted native trial occurs in multiple runs")
            index[key] = value
    expected = {b.trial_id for b in declaration.bindings}
    if set(audited)-expected:raise ValueError("Independent audit includes an undeclared diagnostic trial")
    out = writable_path(output)
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(schema=SCHEMA, diagnostic_schema=VERSION, run=str(run),
        declaration=asdict(declaration), declaration_identity=declaration.identity,
        input_bindings={str(p): sha256_file(p) for p in (declaration_path, launch_path, catalog_path, audit_path)},
        planned_trials=len(expected), selection="Every declared trial; every retained audited native attempt, without outcome filtering",
        source_sha256=sha256_file(__file__), module_sha256=sha256_file(ROOT/'src/human_punching/phase2_stroke_diagnostics.py'),
        wall_seconds=wall_seconds, product_byte_cap=product_byte_cap, final_ledger_reserve_bytes=2*1024**2,
        declared_array_allowance_bytes=MAX_WORKSPACE_BYTES,
        model_file_byte_cap=16*1024**2, total_process_memory_bound_claimed=False,
        new_physics_steps=0, actor_calls=0, gpu_used=False, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        automatic_stroke_or_exploit_truth=False, formal_acceptance=False)
    write_json(out/'plan.json', plan)
    def product_check():
        value = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        if value > product_byte_cap:raise RuntimeError("Diagnostic product cap exceeded; preserve partial files")
        return value
    records = []
    stop = None
    with deadline.signal_handlers():
        for number, request in enumerate(declaration.bindings):
            key = request.trial_id
            record = dict(trial_id=key, request_identity=request.identity,
                single_outward_stroke_pass=None, movement_exploit=None, human_trace_review_required=True)
            binding, evidence = index.get(key), audited.get(key)
            if binding is None:
                record.update(status='unavailable', reason='No retained native attempt for declared request')
            elif evidence is None:
                record.update(status='unavailable', reason='Native attempt has no retained independent per-trial audit')
            elif evidence['audited_runtime_model_sha256'] is None:
                record.update(status='unavailable', reason='Independent audit has no immutable runtime-model binding')
            elif stop:
                record.update(status='unattempted', reason=stop)
            else:
                try:
                    deadline.checkpoint('before_stroke_diagnostic_trial', progress={'trial_id': key})
                    # Reserve at most one bounded array product plus four small plots.
                    if product_check()+34*1024**2 > product_byte_cap:
                        stop = 'diagnostic_product_reservation'
                        record.update(status='unattempted', reason=stop)
                        records.append(record)
                        write_json(out/f'trial_disposition_{number:04d}.json', record)
                        continue
                    folder = writable_path(out/f'trial_{number:04d}')
                    folder.mkdir()
                    write_json(folder/'attempt_started.json', dict(trial_id=key, batch=binding, audit=evidence))
                    raw = owned(binding['raw_trace_path'])
                    if raw != Path(evidence['raw_trace_path']):raise ValueError("Audit and native world index traces differ")
                    record.update({k: evidence[k] for k in ('raw_trace_path', 'raw_trace_sha256',
                        'audit_trial_pass', 'audit_panel_pass', 'execution_complete')})
                    states, prefix = extract_exact_world(raw, evidence['raw_trace_sha256'], binding)
                    record.update(prefix=prefix, world=binding['world'], native_plan_sha256=binding['plan_sha256'])
                    if states is None:
                        record.update(status='unrenderable', reason=prefix['reason'])
                    else:
                        model_path = owned(binding['runtime_model_path'], parent=Path(binding['plan_path']).parent.parent)
                        if model_path.stat().st_size > 16*1024**2:raise ValueError("Runtime model exceeds16MiB XML cap")
                        if sha256_file(model_path) != evidence['audited_runtime_model_sha256']:
                            raise ValueError("Diagnostic runtime model differs from independently audited model")
                        model = mujoco.MjModel.from_xml_path(str(model_path))
                        row = binding['row']
                        arrays, metadata = diagnostic_arrays(model, **states, target_w=request.target_w,
                            active_hand=row['active_hand_index'], checkpoint=deadline.checkpoint)
                        data_path = folder/'stored-stroke-evidence.npz'
                        with data_path.open('xb') as stream:np.savez_compressed(stream, **arrays)
                        events = dict(preparation_complete=row.get('preparation_complete_tick', 0)*.005,
                            actual_launch=binding['actual_launch_s'], outward_reference_complete=row['outward_end_tick']*.005,
                            guard_reference_hold=row.get('guard_hold_tick', round(row.get('guard_time_s', row['duration_s'])/.005))*.005,
                            measured_recovery=evidence['audit_trial'].get('recovery_time_s'))
                        plots = write_review_plots(arrays, folder, title=f'{declaration.family} | {key} | requested TTC={request.ttc_s:.3f}s',
                            events=events, checkpoint=deadline.checkpoint)
                        metadata.update(raw_trace_sha256=evidence['raw_trace_sha256'],
                            runtime_model_path=str(model_path), runtime_model_sha256=sha256_file(model_path),
                            reference_sha256=request.reference_sha256, predicate_sha256=declaration.predicate_sha256,
                            checkpoint_sha256=launch.get('evaluated_checkpoint', {}).get('sha256', declaration.checkpoint_sha256),
                            declaration_checkpoint_template_is_unbound=launch.get('checkpoint_template_is_unbound', False),
                            actual_actor_tensor_digest=binding['actual_actor_tensor_digest'],
                            events=events, prefix=prefix, source_audit=dict(path=str(audit_path), sha256=audit_sha256))
                        write_json(folder/'diagnostics.json', metadata)
                        partial = (prefix['truncated_at_native_invalid_state'] or prefix['incomplete_native_horizon']
                            or not prefix['full_retained_trace_has_all_interval_drives']
                            or evidence['execution_complete'] is not True)
                        record.update(status='diagnosed_prefix' if partial else 'diagnosed_full_retained_trace',
                            evidence_scope='partial_retained_native_trace' if partial else 'full_retained_native_trace',
                            diagnostic_evidence=[dict(path=str(path), sha256=sha256_file(path))
                                for path in (data_path, folder/'diagnostics.json', *plots)],
                            automatic_judgment=False)
                        del arrays, model, states
                    product_check()
                except Exception as error:
                    record.update(status='failed', reason=dict(type=type(error).__name__, message=str(error)[:2000]))
                    if getattr(error, 'receipt', None):
                        record['deadline_receipt'] = error.receipt
                        stop = 'bounded_wall_or_signal_stop'
                    elif 'product cap' in str(error):stop = 'diagnostic_product_cap'
            records.append(record)
            write_json(out/f'trial_disposition_{number:04d}.json', record)
    result = dict(plan, results=records, counts={status: sum(r['status'] == status for r in records)
        for status in sorted({r['status'] for r in records})}, stop_reason=stop,
        measured_products_before_final_receipt_bytes=product_check(), elapsed_wall_s=time.monotonic()-started,
        manual_review_records_created=0)
    write_json(out/'diagnostics.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--trace-audit', type=Path, required=True)
    parser.add_argument('--trace-audit-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=float, default=1800.)
    parser.add_argument('--product-byte-cap', type=int, default=2*1024**3)
    args = parser.parse_args()
    result = diagnose(args.run, args.trace_audit, args.trace_audit_sha256, args.output,
        wall_seconds=args.wall_seconds, product_byte_cap=args.product_byte_cap)
    print(json.dumps(dict(planned_trials=result['planned_trials'], counts=result['counts'],
        manual_review_required=True, automatic_judgment=False)))


if __name__ == '__main__':main()
