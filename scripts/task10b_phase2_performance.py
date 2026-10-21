#!/usr/bin/env python3
"""Offline Phase 2 timing/storage receipts; unknown clocks remain unknown.

Reads retained native records only. It never imports a simulator, restores an
actor, launches a job, or interprets unpaced throughput as real-time execution.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from human_punching.wbc.provenance import writable_path,write_json,sha256_file


def array_shapes(path):
    """Read NumPy headers without materializing a large preserved trace."""
    result={}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):continue
            with archive.open(name) as member:
                version=np.lib.format.read_magic(member)
                if version==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(member)
                elif version in ((2,0),(3,0)):shape,_,dtype=np.lib.format.read_array_header_2_0(member)
                else:raise ValueError('Unknown NumPy array header')
            result[name[:-4]]=dict(shape=list(shape),dtype=str(dtype),
                array_payload_bytes=int(np.prod(shape,dtype=np.int64))*dtype.itemsize)
    return result


def scalar_statistics(values):
    a=np.asarray(values,dtype=float).reshape(-1)
    finite=a[np.isfinite(a)]
    return dict(samples=len(a),nonfinite=int(len(a)-len(finite)),
        sum_s=float(finite.sum()) if len(finite) else None,
        mean_s=float(finite.mean()) if len(finite) else None,
        median_s=float(np.median(finite)) if len(finite) else None,
        p95_s=float(np.quantile(finite,.95)) if len(finite) else None,
        maximum_s=float(finite.max()) if len(finite) else None,
        over_20ms=int(np.count_nonzero(finite>.020)))


def update_throughput(records,validation_every=8):
    """Exclude startup and intervals crossing a declared validation boundary."""
    intervals=[];excluded=[]
    for before,after in zip(records[:-1],records[1:]):
        dt=float(after['elapsed_s'])-float(before['elapsed_s'])
        steps=int(after['fresh_transitions'])-int(before['fresh_transitions'])
        valid=(after['update']==before['update']+1 and before['update']%validation_every!=0
               and np.isfinite(dt) and dt>0 and steps>0)
        row=dict(from_update=before['update'],to_update=after['update'],wall_s=dt,fresh_transitions=steps)
        if valid:intervals.append(row)
        else:excluded.append(row)
    wall=sum(r['wall_s'] for r in intervals);steps=sum(r['fresh_transitions'] for r in intervals)
    return dict(included_intervals=intervals,excluded_intervals=excluded,
        included_wall_s=wall,included_fresh_transitions=steps,
        fresh_transitions_per_s=steps/wall if wall else None,
        denominator='Consecutive completed PPO-update log intervals after the first update; validation-boundary intervals excluded. Includes rollout, PPO, synchronization and log/digest work.',
        pure_physics_transitions_per_s=None,real_time_operation_qualified=False)


def compile_log_observations(paths):
    observations=[]
    pattern=re.compile(r'^Module .*?load on device .*?took ([\d.]+) ms.*$',re.MULTILINE)
    for path in paths:
        text=path.read_text(errors='replace')
        for match in pattern.finditer(text):
            observations.append(dict(path=str(path),reported_wall_s=float(match.group(1))/1000.,line=match.group(0)))
    return dict(observations=observations,reported_load_sum_s=sum(r['reported_wall_s'] for r in observations) if observations else None,
        complete_compilation_wall_s=None,
        limitation='Warp module-load log observations only; graph capture, other compilation and initialization can be unlogged, overlapping or cached.')


def summarize(packet):
    packet=Path(packet).resolve(strict=True)
    packet.relative_to((ROOT/'outputs/task10b').resolve())
    p=json.loads(packet.read_text());folder=packet.parent;run=folder/'run';inputs={str(packet):sha256_file(packet)}
    def read(path):
        if not path.exists():return None
        inputs[str(path)]=sha256_file(path)
        return json.loads(path.read_text())
    outer=read(folder/'execution.json');native=read(run/'execution.json')
    training=read(run/'training/summary.json');terminal=read(folder/'terminal_accounting.json')
    launch=read(run/'launch.json');panel_reports=[];traces=[]
    for path in sorted(run.glob('validation_*/*/summary.json')):
        report=read(path);panel_reports.append(dict(path=str(path),**report))
    for path in sorted(run.glob('validation_*/*/raw_trace.npz')):
        shapes=array_shapes(path);inputs[str(path)]=sha256_file(path)
        q_shape=shapes.get('qpos',{}).get('shape',[])
        samples=q_shape[0] if q_shape else None;worlds=q_shape[1] if len(q_shape)>1 else None
        trace=dict(path=str(path),bytes=path.stat().st_size,samples=samples,worlds=worlds,
            saved_physics_intervals=max(0,samples-1) if samples is not None else None,
            per_world_simulated_s=(samples-1)*.005 if samples else None,
            uncompressed_array_payload_bytes=sum(row['array_payload_bytes'] for row in shapes.values()))
        with np.load(path,allow_pickle=False) as archive:
            if 'time' in archive:
                times=np.asarray(archive['time'],float)
                durations=times[-1]-times[0]
                trace['native_simulated_duration_per_world_s']=durations.tolist()
                trace['native_clock_finite']=bool(np.isfinite(times).all())
            name='policy_admission_observation_actor_wall_s'
            if name in archive:trace['boundary_work_wall']=scalar_statistics(archive[name])
            trace['boundary_work_scope']='Admission, observation, actor and evidence copies; not isolated cached final planning. Physics integration/forward and most recording/postprocessing are outside this field.'
        traces.append(trace)
    jsonl=run/'training/training.jsonl';updates=[];jsonl_errors=[]
    if jsonl.exists():
        inputs[str(jsonl)]=sha256_file(jsonl)
        for number,line in enumerate(jsonl.read_text().splitlines(),1):
            try:updates.append(json.loads(line))
            except json.JSONDecodeError as error:jsonl_errors.append(dict(line=number,error=str(error)))
    evaluation_wall=sum(float(r['wall_time_s']) for r in panel_reports if r.get('wall_time_s') is not None)
    total=native.get('total_wall_s') if native else None
    train_wall=training.get('wall_time_s') if training else None
    accounted=train_wall if train_wall is not None else evaluation_wall
    remainder=None if total is None else total-accounted
    if remainder is not None and remainder<-.001:raise ValueError('Nested clocks contradict native total clock')
    catalog_path=ROOT/p['catalog'] if 'catalog' in p else None
    references=[]
    if catalog_path is not None:
        catalog=read(catalog_path)
        for row in catalog['cells']:
            if row['family']!=p.get('family'):continue
            record=dict(candidate_id=row['candidate_id'],kind=row['kind'],kinematic_admission=row['kinematic_admission'])
            for field in ('reference_path','legacy_runtime_path'):
                source=Path(row[field]).resolve(strict=True);source.relative_to(ROOT.resolve())
                record[field]=str(source);record[field+'_bytes']=source.stat().st_size
            references.append(record)
    reference_paths={r[field] for r in references for field in ('reference_path','legacy_runtime_path')}
    return dict(schema='task10b_phase2_offline_performance_v1',packet=str(packet),mode=p['mode'],family=p.get('family'),
        execution_status=outer.get('status') if outer else 'not_yet_recorded',
        time=dict(allocated_gpu_seconds=terminal.get('allocated_gpu_seconds') if terminal else None,
            allocation_ceiling_reserved_gpu_seconds=p['allocation_wall_seconds'],
            wrapper_wall_s=outer.get('wall_seconds') if outer else None,native_total_wall_s=total,
            recorded_cell_evaluation_wall_s=evaluation_wall,training_session_wall_s=train_wall,
            initialization_and_unclassified_remainder_s=remainder,
            remainder_scope='Native total minus training-session wall (pilot), or completed cell walls (frozen). Includes constructors/export/bookkeeping and unclassified gaps; not a pure compilation measurement.',
            pure_compilation_wall_s=None,pure_initialization_wall_s=None,pure_training_rollout_wall_s=None,
            pure_final_planning_latency_s=None,simulation_pause_wall_s=None,
            missing_clocks='Current receipts lack per-physics wall stamps and separately clocked compilation, initialization, final planning and simulation pauses. Whole-session clocks include elapsed pauses and postprocessing.'),
        compile_log=compile_log_observations(sorted((folder/'logs').glob('*.out'))),
        training=dict(final_fresh_transitions=training.get('fresh_transitions') if training else None,
            logged_updates=len(updates),jsonl_errors=jsonl_errors,post_startup=update_throughput(updates)),
        references=dict(rows=references,unique_archive_bytes=sum(Path(path).stat().st_size for path in reference_paths),
            measured_reference_resident_bytes=None,measured_peak_gpu_bytes=None,
            limitation='Archive bytes are measured on disk; current runtime does not publish ReferenceLibrary.bytes or CUDA peak resident memory. Do not equate compressed archives with resident tensors.'),
        traces=traces,total_trace_bytes=sum(r['bytes'] for r in traces),
        cell_receipt_count=len(panel_reports),complete_cell_count=sum(r.get('status')=='executed' for r in panel_reports),real_time_operation_qualified=False,
        simulated_to_wall_ratio_is_real_time_test=False,inputs=inputs,
        source_sha256=sha256_file(__file__),native_execution_performed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=summarize(args.packet)
    write_json(writable_path(args.output),result)
    print(json.dumps(result,indent=2))
