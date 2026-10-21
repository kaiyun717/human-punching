#!/usr/bin/env python3
"""Prepare/replay one frozen scratch guard checkpoint; never train or submit."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from human_punching.wbc.provenance import configure_local_caches,sha256_file,writable_path,write_json



def project_input(path, expected_sha256):
    """Read a hash-identical local artifact for a recorded compute-mount path.

    NFS aliases can have different device/inode identities on the login node.
    Both readable copies must match the recorded content digest; writes target
    only the resolved local project root. This is content, not inode, identity.
    """
    source=Path(path).resolve(strict=True)
    if sha256_file(source)!=expected_sha256:raise ValueError("Recorded input digest changed")
    if source.is_relative_to(ROOT):return source
    matches=[]
    for index,name in enumerate(source.parts):
        if name==ROOT.name:
            candidate=ROOT.joinpath(*source.parts[index+1:]).resolve()
            if candidate.is_relative_to(ROOT) and candidate.is_file() and sha256_file(candidate)==expected_sha256:matches.append(candidate)
    if len(set(matches))!=1:raise ValueError('Input is not a verified project file: '+str(source))
    return matches[0]


def checkpoint_inputs(bundle,phase_path):
    """Validate stored learned/export identity before requesting native execution."""
    configure_local_caches()
    import torch
    from human_punching.wbc.runtime.bundle import load_runtime_bundle
    from human_punching.wbc.guard_training import validate_guard_checkpoint
    from human_punching.wbc.learned_guard_evaluation import ORIGIN,verify_guard_actor_identity
    from human_punching.wbc.trainedactor_evaluation import actor_identity
    from human_punching.wbc.validation import digest
    checked=load_runtime_bundle(bundle);phase_path=Path(phase_path).resolve(strict=True)
    phase=json.loads(phase_path.read_text())
    if (phase.get('phase')!='smoke' or phase.get('status')!='passed' or not phase.get('scratch_initialized')
            or not phase.get('finite_learning') or not phase.get('normalization_finite')
            or not phase.get('checkpoint_export_verified') or phase.get('completed_iterations',0)<1):
        raise ValueError('A completed scratch smoke with verified export is required')
    checkpoint=project_input(phase['checkpoint_path'],phase['checkpoint_sha256']);jit=project_input(phase['jit_path'],phase['jit_sha256'])
    for path,key in ((checkpoint,'checkpoint_sha256'),(jit,'jit_sha256')):
        if sha256_file(path)!=phase[key]:raise ValueError('Learned artifact hash differs: '+str(path))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    admission=saved['infos']['guard_training']['admission']
    validate_guard_checkpoint(saved,expected_admission=admission,expected_observation_schema=phase['observation_schema'])
    actor=torch.jit.load(str(jit),map_location='cpu').eval()
    for parameter in actor.parameters():parameter.requires_grad_(False)
    plan=saved['infos']['guard_training']['plan']
    identity={**actor_identity(actor),'reference_manifest_sha256':sha256_file(checked.root/'manifest.json'),
        'model_sha256':sha256_file(checked.model_path),'controller_sha256':checked.controller_sha256(),
        'conditioning':plan['conditioning'],'observation_schema':phase['observation_schema'],
        'guard_admission_sha256':admission['sha256'],'joint_feedforward_sha256':digest(admission['joint_feedforward_nm']),
        'training_checkpoint_sha256':phase['checkpoint_sha256'],'guard_training_origin':ORIGIN}
    verify_guard_actor_identity(actor,checked,identity,admission['joint_feedforward_nm'],plan['conditioning'])
    if (identity['reference_manifest_sha256']!=saved['infos']['guard_training']['admission']['reference_manifest_sha256']
            or identity['model_sha256']!=admission['model_sha256'] or identity['controller_sha256']!=admission['controller_sha256']):
        raise ValueError('Training body/reference/controller differs from replay bundle')
    return checked,phase,checkpoint,jit,admission,plan,identity


def prepare(bundle,phase,output):
    checked,learned,checkpoint,jit,admission,plan,identity=checkpoint_inputs(bundle,phase)
    destination=writable_path(output)
    if len(os.fsencode(destination/'run/cache/tmp'))>120:raise ValueError('Use a shallow output path')
    destination.mkdir(parents=True,exist_ok=False);(destination/'logs').mkdir()
    paths=list((ROOT/'src/human_punching/wbc').rglob('*.py'))+[Path(__file__),ROOT/'scripts/task10_guard_evaluation.sbatch',Path(phase).resolve(),checkpoint,jit]
    paths += [p for p in checked.root.rglob('*') if p.is_file()]
    files={str(p):sha256_file(p) for p in sorted(set(paths))}
    for name,expected in files.items():
        source=Path(name);target=destination/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha256_file(target)!=expected:raise ValueError('Input changed during preparation')
    command=['sbatch','--parsable','--account=mit_general','--partition=mit_normal_gpu','--gpus=l40s:1',
        '--cpus-per-task=4','--mem=32G','--time=00:20:00','--no-requeue','--exclude=node3006',
        '--job-name=human-wbc-guard-eval',f'--chdir={ROOT}',f'--output={destination}/logs/%j.out',f'--error={destination}/logs/%j.err',
        str(ROOT/'scripts/task10_guard_evaluation.sbatch'),str(destination/'packet.json')]
    record=dict(schema='human_wbc_frozen_guard_replay_packet_v1',prepared_utc=datetime.now(timezone.utc).isoformat(),
        files=files,bundle=str(checked.root),phase=str(Path(phase).resolve()),jit=str(jit),checkpoint=str(checkpoint),
        actor_identity=identity,feedforward=admission['joint_feedforward_nm'],conditioning=plan['conditioning'],seed=plan['seed'],
        output=str(destination/'run'),training=False,policy_normalization_frozen=True,physics_dt=.005,policy_dt=.020,
        environments=16,seconds=6,application_wall_seconds=1000,products_and_caches_bytes=3*1024**3,
        authorization='User standing authorization for GPU tests and sessions up to one hour',submission_command=command)
    write_json(destination/'packet.json',record);print(json.dumps({'packet':str(destination/'packet.json'),'files':len(files),'trained_iterations':learned['completed_iterations']}))


def worker(packet):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Allocated GPU required')
    record=json.loads(packet.read_text());output=Path(record['output']);output.mkdir(exist_ok=False)
    for key,suffix in [('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('MPLCONFIGDIR','mpl'),('WARP_CACHE_PATH','warp'),('CUDA_CACHE_PATH','cuda'),('TORCH_HOME','torch')]:
        path=output/'cache'/suffix;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    import torch
    from human_punching.wbc.learned_guard_evaluation import run_learned_guard_evaluation
    # The reusable evaluator preserves an existing CUDA caller's RNG. This
    # standalone replay owns initialization before entering that boundary.
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Frozen replay requires exactly one allocated visible GPU')
    torch.cuda.set_device(0);torch.cuda.init()
    import warp as wp
    wp.init();wp.set_device('cuda:0')
    actor=torch.jit.load(record['jit'],map_location='cpu').eval()
    for parameter in actor.parameters():parameter.requires_grad_(False)
    report=run_learned_guard_evaluation(record['bundle'],actor,output/'evaluation',
        joint_feedforward_nm=record['feedforward'],conditioning=record['conditioning'],seed=record['seed'],actor_contract_identity=record['actor_identity'])
    outcome=json.loads(report.read_text())
    print(json.dumps({'evaluation':str(report),'status':outcome['status'],'guard_success_count':outcome['guard_success_count'],'recorded_samples':outcome['recorded_samples']}))
    return int(outcome['status']!='recorded')


def execute(packet):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise RuntimeError('Allocated GPU required')
    record=json.loads(packet.read_text())
    if record.get('schema')!='human_wbc_frozen_guard_replay_packet_v1' or record['training']:raise ValueError('Invalid frozen replay packet')
    for name,expected in record['files'].items():
        if sha256_file(name)!=expected:raise ValueError('Prepared source/input changed: '+name)
    if Path(record['output']).exists():raise FileExistsError(record['output'])
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'worker','--packet',str(packet)]
    started=time.monotonic();peak=0;reason=None
    child=subprocess.Popen(command,cwd=packet.parent,env=env,start_new_session=True)
    try:
        while child.poll() is None:
            size=0
            for path in packet.parent.rglob('*'):
                try:
                    if path.is_file():size+=path.stat().st_size
                except FileNotFoundError:pass
            peak=max(peak,size)
            if peak>record['products_and_caches_bytes']:reason='storage_limit'
            elif time.monotonic()-started>=record['application_wall_seconds']:reason='wall_limit'
            if reason:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                break
            time.sleep(2)
        code=child.wait()
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
    write_json(packet.parent/'execution.json',dict(job_id=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,
        command=command,returncode=code,stop_reason=reason,wall_seconds=time.monotonic()-started,maximum_sampled_packet_bytes=peak,
        packet_sha256=sha256_file(packet),training=False,normalization_updates_requested=False))
    return code if code else int(reason is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare')
    for name in ('bundle','phase','output'):prep.add_argument('--'+name,type=Path,required=True)
    for name in ('execute','worker'):sub.add_parser(name).add_argument('--packet',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.bundle,a.phase,a.output)
    else:raise SystemExit(globals()[a.mode](writable_path(a.packet).resolve(strict=True)))
