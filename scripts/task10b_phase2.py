#!/usr/bin/env python3
"""Immutable Phase2 packets, bounded execution and explicit resource receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from human_punching.wbc.provenance import sha256_file, writable_path, write_json, runtime_record
from human_punching.phase2_budget import Allocation, admit_reservation, budget_status

PREP = ROOT/'outputs/task10b/p2prep_20261021_01'
PYTHON = ROOT/'.venv-task10b/bin/python'
LIMITS = {'references': (900, 840), 'frozen': (450, 390), 'pilot': (1800, 1680)}
FAMILIES = ('jab', 'cross', 'left_hook', 'right_hook')
SCHEMA = 'task10b_phase2_execution_packet_v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def local(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT/p
    p = p.resolve(strict=True)
    p.relative_to(ROOT.resolve())
    return p


def packet_directory(path):
    p=Path(path).resolve()
    if p.parent != ROOT/'outputs/task10b' or not p.name.startswith('p2'):
        raise ValueError('Phase2 packets must be direct p2-prefixed children of outputs/task10b')
    return p


def old_helper(name):
    spec = importlib.util.spec_from_file_location('p2_helper_'+name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(record):
    if record.get('schema') != SCHEMA or record.get('mode') not in LIMITS:
        raise ValueError('Invalid Phase2 packet')
    mode = record['mode']
    wall, app = LIMITS[mode]
    expected = dict(phase=2, allocation_wall_seconds=wall, application_wall_seconds=app,
                    gpus=1, automatic_followup=False, promotion=False, g1_actor_execution=False,
                    combined_scene=False, training=mode == 'pilot', seed=21021,
                    num_envs=128, rollout_steps=64, max_updates=32 if mode == 'pilot' else 0,
                    fresh_transition_cap=262144 if mode == 'pilot' else 0,
                    products_bytes_cap=3*1024**3)
    if any(record.get(k) != v for k,v in expected.items()):
        raise ValueError('Scope, route or resource limits changed')
    if mode != 'references' and record.get('family') not in FAMILIES:
        raise ValueError('One family per native specialist packet required')
    if record.get('partition') not in ('mit_normal_gpu','mit_preemptable'):
        raise ValueError('Unauthorized partition')
    if 'node3006' not in record.get('excluded_nodes', []):
        raise ValueError('node3006 exclusion remains required')
    if record.get('runtime') != runtime_record() or not record['runtime']['declared_pins_match']:
        raise ValueError('Pinned runtime changed')
    for key,value in dict(bundle='outputs/task10/release_baseline_02',release='outputs/four-punch-milestone',
                          predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json').items():
        if record.get(key)!=value:raise ValueError('Accepted baseline path changed: '+key)
    if record.get('authority_sha256') != sha256_file(PREP/'authorization.json'):
        raise ValueError('Current Phase2 authority changed')


def required_inputs(r):
    paths=old_helper('task10_release_baseline').required_inputs(r)
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'src/human_punching').rglob('*.py')
                 if not p.name.startswith('phase2'))
    # Explicit Phase2 additions: absent unrelated future modules do not invalidate
    # an already-frozen reference-generation packet.
    core=['scalar_reference.py','contact.py','jab_learning.py','jab_evaluation.py',
          'functional_recovery.py','phase2_reference.py','phase2_budget.py']
    if r['mode']!='references':core+=['phase2_learning.py','phase2_domain.py']
    paths.update('src/human_punching/'+name for name in core)
    paths.update(('scripts/task10b_phase2.py','scripts/task10b_phase2.sbatch',
                  'scripts/task10b_phase2_references.py','scripts/task10b_phase0.py'))
    paths.update(str(p.relative_to(ROOT)) for p in (PREP/'runtime.json',PREP/'runtime.env',PREP/'authorization.json',
        PREP/'runtime_interpreter_manifest.json',ROOT/'.venv-task10b/pyvenv.cfg',
        ROOT/'.venv-task10b/lib/python3.12/site-packages/task10b_readonly_packages.pth'))
    paths.add(r['predicate'])
    if r['mode']!='references':
        catalog=local(r['catalog']);catalog.relative_to(ROOT/'outputs/task10b')
        paths.add(str(catalog.relative_to(ROOT)))
        cat=json.loads(catalog.read_text())
        for cell in cat['cells']:
            for key in ('reference_path','legacy_runtime_path','diagnostic_screen_path'):
                if cell.get(key):paths.add(str(local(cell[key]).relative_to(ROOT)))
    return paths


def verify_runtime():
    runtime=json.loads((PREP/'runtime.json').read_text())
    python_root=local(runtime['local_python_root'])
    manifest=PREP/'runtime_interpreter_manifest.json'
    if sha256_file(manifest)!=runtime['interpreter_manifest_sha256']:
        raise ValueError('Interpreter manifest changed')
    for item in json.loads(manifest.read_text()):
        member=python_root/item['path']
        if not member.resolve(strict=True).is_relative_to(python_root):raise ValueError('Interpreter path escaped')
        if 'symlink' in item:
            if not member.is_symlink() or os.readlink(member)!=item['symlink']:raise ValueError('Interpreter symlink changed')
        elif member.is_symlink() or member.stat().st_size!=item['bytes'] or sha256_file(member)!=item['sha256']:
            raise ValueError('Interpreter payload changed')
    for key,expected in runtime['distribution_metadata_hashes'].items():
        package,filename=key.rsplit('/',1)
        member=local(Path(importlib.metadata.distribution(package)._path)/filename)
        if member.stat().st_size!=expected['bytes'] or sha256_file(member)!=expected['sha256']:
            raise ValueError('Installed dependency metadata changed')


def submission_command(r,out):
    return ['sbatch','--parsable','--account=mit_general','--partition='+r['partition'],
        '--nodes=1','--ntasks=1','--gres=gpu:l40s:1','--cpus-per-task=4','--mem=32G',
        '--time='+str(r['allocation_wall_seconds']//60),'--no-requeue','--exclude=node3006','--job-name=t10b-p2-'+r['mode'],
        '--chdir='+str(ROOT),'--output='+str(out/'logs/%j.out'),'--error='+str(out/'logs/%j.err'),
        str(ROOT/'scripts/task10b_phase2.sbatch'),str(out/'packet.json')]


def verify(packet):
    path = local(packet)
    packet_directory(path.parent)
    r = json.loads(path.read_text())
    validate(r)
    path.relative_to(ROOT/'outputs/task10b')
    if path.name!='packet.json' or writable_path(ROOT/r['output'])!=path.parent/'run':
        raise ValueError('Packet/output identity changed')
    if r['submission_command']!=submission_command(r,path.parent):
        raise ValueError('Submission command differs from bounded packet')
    if set(r['files'])!=required_inputs(r):raise ValueError('Required source/input closure changed')
    interpreter = PYTHON.resolve(strict=True)
    if not interpreter.is_relative_to(ROOT) or sha256_file(interpreter) != r['interpreter_sha256']:
        raise ValueError('Human-owned interpreter changed')
    for rel,digest in r['files'].items():
        source=local(rel)
        archive=(path.parent/'source_snapshot'/rel).resolve(strict=True)
        if not archive.is_relative_to(path.parent) or sha256_file(source)!=digest or sha256_file(archive)!=digest:
            raise ValueError('Source/input binding changed: '+rel)
    verify_runtime()
    return r


def prepare(output, mode, *, family=None, catalog=None, partition='mit_normal_gpu'):
    if mode not in LIMITS:
        raise ValueError('Unknown bounded mode')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='wbc':
        raise ValueError('Task10B requires wbc branch')
    out=writable_path(output)
    packet_directory(out)
    if len(os.fsencode(out/'run/cache/tmp'))>120:
        raise ValueError('Use shallow compiler/output paths')
    wall,app=LIMITS[mode]
    r=dict(schema=SCHEMA, phase=2, mode=mode, family=family, prepared_utc=now(),
        authority_sha256=sha256_file(PREP/'authorization.json'), allocation_wall_seconds=wall,
        application_wall_seconds=app, products_bytes_cap=3*1024**3, gpus=1,
        automatic_followup=False,promotion=False,g1_actor_execution=False,combined_scene=False,
        training=mode=='pilot',num_envs=128,rollout_steps=64,max_updates=32 if mode=='pilot' else 0,
        fresh_transition_cap=262144 if mode=='pilot' else 0, seed=21021,
        partition=partition,excluded_nodes=['node3006'],runtime=runtime_record(),
        interpreter_sha256=sha256_file(PYTHON), output=str((out/'run').relative_to(ROOT)),
        bundle='outputs/task10/release_baseline_02',release='outputs/four-punch-milestone',
        predicate='outputs/task10b/p1prep_20261020_01/functional_predicate_v1.json')
    if mode!='references':
        r['catalog']=str(local(catalog).relative_to(ROOT))
    validate(r)
    paths=required_inputs(r)
    r['files']={p:sha256_file(local(p)) for p in sorted(paths)}
    out.mkdir(parents=True,exist_ok=False)
    (out/'logs').mkdir()
    for rel,digest in r['files'].items():
        dest=out/'source_snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(local(rel),dest)
        if sha256_file(dest)!=digest:raise ValueError('Source changed during snapshot')
    r['submission_command']=submission_command(r,out)
    write_json(out/'packet.json',r)
    verify(out/'packet.json')
    return {'packet':str(out/'packet.json'),'sha256':sha256_file(out/'packet.json'),'inputs':len(paths)}


def ledger():
    rows=[]
    for intent in sorted((ROOT/'outputs/task10b').glob('p2*/submission_intent.json')):
        path=intent.with_name('submission.json')
        if not path.exists():
            if intent.with_name('submission_rejected.json').exists():continue
            i=json.loads(intent.read_text())
            rows.append(Allocation('unresolved:'+str(intent.parent),i['reserved_gpu_seconds']))
            continue
        r=json.loads(path.read_text())
        i=json.loads(intent.read_text())
        packet=intent.with_name('packet.json')
        if (r.get('schema')!='task10b_phase2_submission_v1'
            or r.get('packet_sha256')!=sha256_file(packet)
            or i.get('packet_sha256')!=r['packet_sha256']
            or i.get('reserved_gpu_seconds')!=r.get('reserved_gpu_seconds')):
            raise ValueError('Submission/intent binding invalid; keep reservation unresolved')
        terminal=path.with_name('terminal_accounting.json')
        if terminal.exists():
            t=json.loads(terminal.read_text())
            if (t.get('schema')!='task10b_phase2_terminal_accounting_v1'
                or t['job_id']!=r['job_id'] or t['packet_sha256']!=r['packet_sha256']):
                raise ValueError('Accounting job/packet binding mismatch')
            rows.append(Allocation(r['job_id'],r['reserved_gpu_seconds'],elapsed_seconds=t['allocated_gpu_seconds'],terminal=True))
        else:rows.append(Allocation(r['job_id'],r['reserved_gpu_seconds']))
    return rows


def submit(packet):
    path=local(packet);r=verify(path)
    with (PREP/'allocation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (path.parent/'submission_intent.json').exists():raise ValueError('Packet already has submission intent; resolve before retry')
        outstanding=0
        for intent in (ROOT/'outputs/task10b').glob('p2*/submission_intent.json'):
            if intent.with_name('terminal_accounting.json').exists() or intent.with_name('submission_rejected.json').exists():continue
            item=json.loads(intent.read_text())
            outstanding+=('--partition='+r['partition']) in item['command']
        if outstanding >= (2 if r['partition']=='mit_normal_gpu' else 4):
            raise ValueError('Authorized concurrent GPU partition limit reached')
        # Pilot jobs cannot consume the 75 minute physical-validation reserve.
        budget=admit_reservation(ledger(),r['allocation_wall_seconds'],evaluation_reserve_seconds=4500 if r['mode']=='pilot' else 0)
        # Durable reservation before contacting Slurm. A timeout or lost reply
        # remains charged and cannot be retried as though no allocation exists.
        write_json(path.parent/'submission_intent.json',dict(utc=now(),packet_sha256=sha256_file(path),
            reserved_gpu_seconds=r['allocation_wall_seconds'],command=r['submission_command'],budget_before=budget))
        response=subprocess.run(r['submission_command'],cwd=ROOT,capture_output=True,text=True,check=False,timeout=30)
        receipt=dict(schema='task10b_phase2_submission_v1',utc=now(),packet_sha256=sha256_file(path),
                     command=r['submission_command'],returncode=response.returncode,stdout=response.stdout,
                     stderr=response.stderr,budget_before=budget,reserved_gpu_seconds=r['allocation_wall_seconds'])
        if response.returncode:
            # Unexpected numeric stdout can indicate a lost/ambiguous success;
            # keep its intent charged until scheduler reconciliation.
            name='submission_uncertain.json' if any(char.isdigit() for char in response.stdout) else 'submission_rejected.json'
            write_json(path.parent/name,receipt)
            raise RuntimeError(response.stderr)
        job=response.stdout.strip().split(';')[0]
        if not job.isdigit():
            write_json(path.parent/'submission_uncertain.json',receipt)
            raise RuntimeError('Submission identity uncertain; inspect scheduler before any further reservation')
        receipt['job_id']=job
        write_json(path.parent/'submission.json',receipt)
    return receipt


def account(packet):
    path=local(packet);s=json.loads((path.parent/'submission.json').read_text());job=s['job_id']
    cmd=['sacct','-j',job,'-n','-P','--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,NodeList,Start,End']
    response=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=30)
    lines=[line.split('|') for line in response.stdout.splitlines() if line.split('|')[0]==job]
    if len(lines)!=1:raise RuntimeError('No unique Slurm allocation row')
    row=lines[0]
    if row[1].split()[0] not in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','PREEMPTED','NODE_FAIL','OUT_OF_MEMORY','BOOT_FAIL','DEADLINE','REVOKED'):
        return {'job_id':job,'state':row[1],'reservation_remains':True}
    tres=dict(x.split('=',1) for x in row[4].split(',') if '=' in x)
    if int(tres.get('gres/gpu',0))!=1:raise ValueError('Actual allocation is not one GPU')
    result=dict(schema='task10b_phase2_terminal_accounting_v1',utc=now(),job_id=job,
        state=row[1],exit_code=row[2],allocated_gpu_seconds=int(row[3]),allocated_tres=tres,
        node=row[5],start=row[6],end=row[7],command=cmd,stdout=response.stdout,stderr=response.stderr,
        packet_sha256=sha256_file(path))
    write_json(path.parent/'terminal_accounting.json',result)
    return result


def native_command(r):
    command=[str(PYTHON),'-B']
    if r['mode']=='references':
        return command+[str(ROOT/'scripts/task10b_phase2_references.py'),'--bundle',str(ROOT/r['bundle']),
            '--release',str(ROOT/r['release']),'--output',str(ROOT/r['output']),
            '--predicate',str(ROOT/r['predicate'])]
    return command+['-m','human_punching.phase2_learning','--mode',r['mode'],'--family',r['family'],
        '--catalog',str(ROOT/r['catalog']),'--output',str(ROOT/r['output']),'--seed',str(r['seed']),
        '--max-updates',str(r['max_updates']),'--wall-seconds',str(r['application_wall_seconds']-30)]


def execute(packet):
    path=local(packet)
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Explicit Slurm CUDA allocation required')
    started=time.monotonic();process=None;code=2
    receipt=dict(schema='task10b_phase2_execution_v1',started_utc=now(),job_id=os.environ['SLURM_JOB_ID'],
        packet_sha256=sha256_file(path),status='failed',native_processes_started=0,budget_stop=None)
    write_json(path.parent/'execution_started.json',receipt)
    try:
        r=verify(path)
        health=old_helper('task10b_phase0').cuda_health_receipt(path.parent/'cuda_health.json')
        if health['status']!='passed':raise RuntimeError('Allocated CUDA health failed')
        cmd=native_command(r);receipt['native_command']=cmd
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
                 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled')
        process=subprocess.Popen(cmd,cwd=path.parent,env=env,start_new_session=True)
        receipt['native_processes_started']=1
        peak=0
        while process.poll() is None:
            size=old_helper('task10b_phase0').tree_bytes(path.parent)
            peak=max(peak,size)
            if size>r['products_bytes_cap']:receipt['budget_stop']='products_budget'
            if time.monotonic()-started>=r['application_wall_seconds']:receipt['budget_stop']='application_wall'
            if receipt['budget_stop']:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                break
            time.sleep(2)
        code=process.wait();receipt['peak_sampled_packet_bytes']=peak
        receipt['status']='executed' if code==0 and not receipt['budget_stop'] else 'failed'
        receipt['status_is_physical_qualification']=False
    except Exception as exc:
        receipt['exception']={'type':type(exc).__name__,'message':str(exc)}
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=15)
        receipt.update(returncode=code,ended_utc=now(),wall_seconds=time.monotonic()-started)
        if receipt['native_processes_started']==0:receipt['actual_physics_steps']=0
        write_json(path.parent/'execution.json',receipt)
    return 0 if receipt['status']=='executed' else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('prepare');s.add_argument('--output',required=True);s.add_argument('--mode',choices=LIMITS,required=True)
    s.add_argument('--family',choices=FAMILIES);s.add_argument('--catalog');s.add_argument('--partition',default='mit_normal_gpu')
    for name in ('verify','submit','account','execute'):sub.add_parser(name).add_argument('--packet',required=True)
    sub.add_parser('budget')
    a=p.parse_args()
    if a.action=='execute':return execute(a.packet)
    if a.action=='prepare':result=prepare(a.output,a.mode,family=a.family,catalog=a.catalog,partition=a.partition)
    elif a.action=='budget':result=budget_status(ledger())
    else:result=globals()[a.action](a.packet)
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
