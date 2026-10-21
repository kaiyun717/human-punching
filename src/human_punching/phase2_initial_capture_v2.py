"""Counted four-start v2 capture with exact job/source/input allocation proof."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from . import phase2_initial_state_v2 as initial
from . import phase2_native_bank_geometry_v2 as geometry_capture
from .phase2_shared_catalog_v2 import START_QUANTIZATION
from . import phase2_shared_catalog_v2
from .wbc.provenance import PROJECT_ROOT,sha256_file,write_json

PACKET='task10b_phase2_initial_state_capture_packet_v2'
ALLOCATION='task10b_phase2_initial_state_capture_allocation_v2'


def check_packet(path,*,require_terminal):
    path=initial.owned(path);packet=json.loads(path.read_text());folder=path.parent
    if (packet.get('schema')!=PACKET or packet.get('mode')!='initial-state-capture-v2'
            or packet.get('gpus')!=1 or packet.get('training') is not False
            or packet.get('num_envs')!=4 or packet.get('rollout_steps')!=0
            or packet.get('allocation_wall_seconds')!=360 or packet.get('application_wall_seconds')!=300
            or packet.get('child_wall_upper_bound_seconds')!=270
            or packet.get('planned_explicit_integration_world_steps')!=0 or packet.get('captured_worlds')!=4
            or packet.get('family')!='jab' or packet.get('capture_split') not in ('train','validation','test')
            or packet.get('automatic_followup') is not False or packet.get('g1_actor_execution') is not False
            or packet.get('combined_scene') is not False or packet.get('promotion') is not False):
        raise ValueError('Fixed counted four-start initial-capture scope required')
    # This gate precedes opening or hashing any test geometry/domain payload.
    if packet['capture_split']=='test':
        frozen=packet['inputs'].get('pre_test_freeze')
        if frozen is None:raise ValueError('Test capture requires prior measured route freeze')
        frozen_path=initial.owned(frozen['path'])
        if sha256_file(frozen_path)!=frozen['sha256']:raise ValueError('Pre-test freeze changed')
        proof=initial.native.validate_test_bank_freeze(frozen_path)
        if proof['domain_sha256']!=packet['inputs']['domain']['sha256']:
            raise ValueError('Test capture domain differs from measured prior freeze')
    expected={'src/human_punching/phase2_initial_state_v2.py':sha256_file(initial.__file__),
              'src/human_punching/phase2_initial_capture_v2.py':sha256_file(__file__),
              'src/human_punching/phase2_native_bank_geometry_v2.py':sha256_file(geometry_capture.__file__),
              'src/human_punching/phase2_shared_catalog_v2.py':sha256_file(phase2_shared_catalog_v2.__file__)}
    if any(packet['files'].get(name)!=digest for name,digest in expected.items()):
        raise ValueError('Capture packet does not bind current exact v2 sources')
    for name,digest in packet['files'].items():
        if sha256_file(initial.owned(name))!=digest or sha256_file(initial.owned(folder/'source_snapshot'/name))!=digest:
            raise ValueError('Counted capture live/archive source binding differs')
    submission=json.loads((folder/'submission.json').read_text())
    if submission.get('packet_sha256')!=sha256_file(path) or not str(submission.get('job_id','')).isdigit():
        raise ValueError('Explicit counted capture submission identity required')
    if require_terminal:
        terminal=json.loads((folder/'terminal_accounting.json').read_text())
        if (terminal.get('packet_sha256')!=sha256_file(path) or terminal.get('job_id')!=submission['job_id']
                or terminal.get('state')!='COMPLETED' or not 0<terminal.get('allocated_gpu_seconds',0)<=360):
            raise ValueError('Exact successful terminal counted capture receipt required')
    elif os.environ.get('SLURM_JOB_ID')!=submission['job_id']:
        raise ValueError('Current capture process differs from bound counted job')
    return packet,submission


def validate_capture_allocation(item,*,capture):
    path,receipt=initial.read_bound(item)
    if receipt.get('schema')!=ALLOCATION or receipt.get('source_sha256')!=sha256_file(__file__):
        raise ValueError('Versioned actual capture allocation proof required')
    packet_path,_=initial.read_bound(receipt['packet']);packet,submission=check_packet(packet_path,require_terminal=True)
    capture_path,captured=initial.read_bound(receipt['capture'])
    if (initial.owned(capture['path'])!=capture_path or capture['sha256']!=receipt['capture']['sha256']
            or receipt.get('slurm_job_id')!=submission['job_id'] or captured.get('slurm_job_id')!=submission['job_id']
            or captured.get('schema')!=initial.CAPTURE or captured.get('complete_capture') is not True
            or captured.get('physics_steps_executed')!=0 or captured.get('actor_updates_executed')!=0
            or captured.get('captured_native_states')!=4 or receipt.get('bound_source_files')!=len(packet['files'])
            or initial.owned(PROJECT_ROOT/packet['output'])!=capture_path.parent):
        raise ValueError('Capture allocation proof differs from actual four-state result')
    for key in ('catalog','domain','geometry','witness'):
        record=packet['inputs'][key];p=initial.owned(record['path'])
        if sha256_file(p)!=record['sha256']:raise ValueError('Counted capture input changed')
    if (captured['domain_sha256']!=packet['inputs']['domain']['sha256']
            or captured['geometry_start_bank']['sha256']!=packet['inputs']['geometry']['sha256']
            or captured['initial_identity_witness']['sha256']!=packet['inputs']['witness']['sha256']):
        raise ValueError('Actual captured domain/start/witness differs from fixed packet')
    return dict(path=str(path),sha256=sha256_file(path),packet_sha256=sha256_file(packet_path),
        slurm_job_id=submission['job_id'],bound_source_files=len(packet['files']),complete_four_start_capture=True)


def execute(packet_path,wall_seconds):
    packet,submission=check_packet(packet_path,require_terminal=False)
    if not 120<wall_seconds<=270:raise ValueError('Finite <=270second capture child required')
    inputs=packet['inputs']
    for item in inputs.values():
        p=initial.owned(item['path'])
        if sha256_file(p)!=item['sha256']:raise ValueError('Fixed capture input changed')
    output=PROJECT_ROOT/packet['output']
    witness=initial.validate_witness(inputs['witness'])
    base=geometry_capture.capture_shared_start_bank(initial.owned(inputs['catalog']['path']),initial.owned(inputs['domain']['path']),
        packet['capture_split'],initial.owned(inputs['geometry']['path']),PROJECT_ROOT/'outputs/four-punch-milestone',
        output,wall_seconds,
        pre_test_freeze=(initial.owned(inputs['pre_test_freeze']['path']) if packet['capture_split']=='test' else None))
    result={**base,'schema':initial.CAPTURE,'source_sha256':sha256_file(__file__),
        'base_capture':initial.binding(output/'bank_capture.json'),
        'base_capture_source_sha256':sha256_file(geometry_capture.__file__),
        'parent_native_adapter_sha256':initial.PARENT_SHA,'initial_identity_consumer_sha256':sha256_file(initial.__file__),
        'initial_identity_version':initial.VERSION,'initial_identity_witness':witness,
        'start_quantization_version':START_QUANTIZATION,'old_future_bitwise_result_preserved':True,
        'shared_start_feasibility_qualified':False}
    write_json(output/'bank_capture_initial_identity_v2.json',result)
    capture_path=output/'bank_capture_initial_identity_v2.json'
    proof=dict(schema=ALLOCATION,source_sha256=sha256_file(__file__),packet=initial.binding(packet_path),
        capture=initial.binding(capture_path),slurm_job_id=submission['job_id'],bound_source_files=len(packet['files']),
        new_explicit_physics_steps=result.get('physics_steps_executed'),actor_updates=result.get('actor_updates_executed'),
        complete_capture=result.get('complete_capture'),shared_start_feasibility_qualified=False)
    write_json(output/'bank_capture_allocation_v2.json',proof)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--wall-seconds',type=float,required=True);a=p.parse_args()
    result=execute(a.packet,a.wall_seconds);return 0 if result['complete_capture'] else 1


if __name__=='__main__':raise SystemExit(main())
