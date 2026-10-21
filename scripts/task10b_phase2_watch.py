#!/usr/bin/env python3
"""Read-only scheduler watch; preserve status and terminal allocation receipts.

This never submits, cancels, requeues or launches an application.
"""
from datetime import datetime, timezone
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=3600)
    args=parser.parse_args()
    if not 1<=args.seconds<=14400:raise ValueError('Finite watch up to four hours required')
    output=args.output.resolve();output.relative_to(ROOT/'outputs/task10b')
    output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('phase2_launcher',ROOT/'scripts/task10b_phase2.py')
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    packets=[launcher.local(p) for p in args.packet]
    jobs={json.loads(p.with_name('submission.json').read_text())['job_id']:p for p in packets}
    started=time.monotonic();last=None
    with (output/'status.jsonl').open('x') as stream:
        while time.monotonic()-started<args.seconds:
            command=['squeue','-h','-j',','.join(jobs),'-o','%i|%T|%M|%S|%R']
            try:
                response=subprocess.run(command,capture_output=True,text=True,timeout=30)
            except subprocess.TimeoutExpired:
                response=subprocess.CompletedProcess(command,124,'','Scheduler query timed out; reservations remain charged')
            record=dict(utc=datetime.now(timezone.utc).isoformat(),returncode=response.returncode,
                stdout=response.stdout,stderr=response.stderr,terminal=[])
            for job,path in jobs.items():
                if path.with_name('terminal_accounting.json').exists():
                    record['terminal'].append(job)
                elif response.returncode==0 and job not in {line.split('|')[0] for line in response.stdout.splitlines()}:
                    try:
                        receipt=launcher.account(path)
                        if 'allocated_gpu_seconds' in receipt:record['terminal'].append(job)
                    except Exception as exc:
                        record.setdefault('accounting_errors',{})[job]=repr(exc)
            stream.write(json.dumps(record)+'\n');stream.flush()
            signature=(response.stdout,tuple(record['terminal']),response.returncode)
            if signature!=last:
                print(json.dumps(record),flush=True);last=signature
            if len(record['terminal'])==len(jobs):break
            time.sleep(min(30,max(0,args.seconds-(time.monotonic()-started))))
    summary=dict(status='all_terminal' if len(record['terminal'])==len(jobs) else 'watch_limit',
        terminal_jobs=record['terminal'],jobs=list(jobs),elapsed_wall_s=time.monotonic()-started,
        application_launches=0,submissions=0,cancellations=0)
    with (output/'summary.json').open('x') as stream:json.dump(summary,stream,indent=2)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
