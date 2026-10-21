#!/usr/bin/env python3
"""Prepare or execute one bounded observer qualification; never auto-submit."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from human_punching.wbc.provenance import sha256_file,writable_path,write_json


def prepare(bundle,output):
    from human_punching.wbc.runtime.bundle import load_runtime_bundle
    checked=load_runtime_bundle(bundle);destination=writable_path(output)
    if len(os.fsencode(destination/'run/cache/tmp'))>120:raise ValueError('Use a shallow output path')
    destination.mkdir(parents=True,exist_ok=False);(destination/'logs').mkdir()
    paths=list((ROOT/'src/human_punching/wbc').rglob('*.py'))+[Path(__file__),ROOT/'scripts/task10_telemetry.sbatch']
    paths.extend(p for p in checked.root.rglob('*') if p.is_file())
    files={str(p.resolve().relative_to(ROOT.resolve())):sha256_file(p) for p in sorted(set(paths))}
    for relative,expected in files.items():
        source=ROOT/relative;target=destination/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha256_file(target)!=expected or sha256_file(source)!=expected:raise ValueError('Source changed during preparation')
    command=['sbatch','--parsable','--account=mit_general','--partition=mit_normal_gpu',
        '--nodes=1','--ntasks=1','--gres=gpu:l40s:1',
        '--cpus-per-task=4','--mem=32G','--time=00:30:00','--no-requeue','--exclude=node3006',
        '--job-name=human-wbc-telemetry',f'--chdir={ROOT}',f'--output={destination}/logs/%j.out',f'--error={destination}/logs/%j.err',
        str(ROOT/'scripts/task10_telemetry.sbatch'),str(destination/'packet.json')]
    record=dict(schema='human_wbc_telemetry_packet_v1',prepared_utc=datetime.now(timezone.utc).isoformat(),
        files=files,bundle=str(checked.root.relative_to(ROOT.resolve())),output=str((destination/'run').relative_to(ROOT.resolve())),
        application_wall_seconds=1600,products_and_caches_bytes=6*1024**3,
        environments=16,seconds=6,fixtures=['guard','punch'],repeats=['off','off','on'],
        training=False,acceptance_qualified=False,
        authorization='User standing authorization for GPU tests and sessions up to one hour',submission_command=command)
    write_json(destination/'packet.json',record)
    print(json.dumps({'packet':str(destination/'packet.json'),'files':len(files),'packet_sha256':sha256_file(destination/'packet.json')}))
    return record


def tree_bytes(root):
    total = 0
    for path in root.rglob('*'):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # Native compiler scratch may disappear during this sample.
    return total


def execute(packet):
    path=Path(packet).resolve(strict=True);record=json.loads(path.read_text());base=path.parent
    if record.get('schema')!='human_wbc_telemetry_packet_v1' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Explicit prepared packet inside allocated Slurm job required')
    for relative,expected in record['files'].items():
        p=(ROOT/relative).resolve(strict=True)
        if not p.is_relative_to(ROOT.resolve()) or sha256_file(p)!=expected:raise ValueError('Frozen input changed: '+relative)
    device={'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'SLURM_JOB_GPUS':os.environ.get('SLURM_JOB_GPUS'),
            'SLURM_JOB_ID':os.environ['SLURM_JOB_ID'],'hostname':os.uname().nodename}
    probe=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True)
    device.update(nvidia_smi_returncode=probe.returncode,nvidia_smi_stdout=probe.stdout,nvidia_smi_stderr=probe.stderr)
    write_json(base/'device_preflight.json',device)
    output=writable_path(ROOT/record['output'])
    if output.exists():raise ValueError('Never overwrite a prior native run')
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[str(ROOT/'.venv-wbc/bin/python'),'-B','-m','human_punching.wbc.telemetry.native_qualification',
             '--bundle',str(ROOT/record['bundle']),'--output',str(output)]
    begin=time.monotonic();reason=None
    process=subprocess.Popen(command,cwd=ROOT,env=env,start_new_session=True)
    try:
        while process.poll() is None:
            if time.monotonic()-begin>record['application_wall_seconds']:reason='application_wall_budget'
            if tree_bytes(base)>record['products_and_caches_bytes']:reason='products_and_caches_budget'
            if reason:
                import signal
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                break
            time.sleep(2)
        code=process.wait()
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=15)
    receipt={'schema':'human_wbc_telemetry_execution_v1','returncode':code,'budget_stop':reason,
             'wall_s':time.monotonic()-begin,'measured_products_and_caches_bytes':tree_bytes(base),
             'slurm_job_id':os.environ['SLURM_JOB_ID'],'packet_sha256':sha256_file(path),'training':False}
    write_json(base/'execution.json',receipt)
    return code if code else (2 if reason else 0)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare');a.add_argument('--bundle',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('execute');a.add_argument('--packet',required=True)
    args=p.parse_args()
    if args.mode=='prepare':prepare(args.bundle,args.output);return 0
    return execute(args.packet)


if __name__=='__main__':raise SystemExit(main())
