"""Bounded pathname projection and immutable D0 uniform-control admission.

No reference arrays, scientific row metadata, policy or simulation are changed.
The view names the same SHA-bound files through a proved local mount spelling.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import mujoco

from .phase2_reference import CANDIDATE_TTCS,retime_uniform
from .wbc.provenance import sha256_file,write_json

ROOT=Path(__file__).resolve().parents[2]
VERSION='task10b_phase2_timing_catalog_path_projection_v1'
PATH_KEYS=frozenset(('predicate_path','reference_path','legacy_runtime_path','source_model_path','diagnostic_screen_path'))


def owned(path):
    p=Path(path);p=(p if p.is_absolute() else ROOT/p).resolve(strict=True)
    if not p.is_relative_to(ROOT.resolve(strict=True)) or not p.is_file():raise ValueError('Owned timing-control input required')
    return p


def read(path):return json.loads(owned(path).read_text())


def binding(path):
    p=owned(path);return dict(path=str(p.relative_to(ROOT.resolve())),sha256=sha256_file(p))


def bound(item):
    p=owned(item.get('path',item.get('relative_name')))
    if sha256_file(p)!=item['sha256']:raise ValueError('Timing-control evidence changed')
    return p


def projection(source_path,probe_path,execution_path):
    source_path=owned(source_path);probe_path=owned(probe_path);execution_path=owned(execution_path)
    source=read(source_path);probe=read(probe_path);execution=read(execution_path)
    local=ROOT.resolve(strict=True);foreign=Path(probe['compute_root']);logical=Path(probe['local_root'])
    if (probe.get('schema')!='task10b_phase2_host_mount_alias_probe_v1' or probe.get('passed') is not True
            or probe.get('fresh_owned_nonce_visible_identically_through_both_mounts') is not True
            or str(local) not in (str(logical),str(foreign))
            or execution.get('schema')!='task10b_phase2_execution_v1' or execution.get('status')!='executed'
            or execution.get('returncode')!=0 or execution.get('native_processes_started')!=1 or not execution.get('job_id')
            or execution.get('runtime_path_context',{}).get('root')!=str(foreign)):
        raise ValueError('Successful counted generation and exact fresh mount-alias proof required')
    nonce=probe_path.parent/'nonce.json';remote_nonce=foreign/nonce.relative_to(local)
    if (sha256_file(nonce)!=probe['nonce_sha256'] or sha256_file(remote_nonce)!=probe['nonce_sha256']
            or nonce.stat().st_ino!=remote_nonce.stat().st_ino):raise ValueError('Fresh mount-alias nonce differs')
    if source_path.parent.parent!=execution_path.parent or source.get('generation_source_sha256')!=sha256_file(local/'src/human_punching/phase2_reference.py'):
        raise ValueError('Catalog does not belong to the successful unchanged generation source')
    packet=execution_path.parent/'packet.json'
    if sha256_file(packet)!=execution.get('packet_sha256') or read(packet).get('mode')!='timing-references':
        raise ValueError('Generation packet identity differs')
    records={}
    def mapped(value):
        path=Path(value)
        if not path.is_absolute() or not path.is_relative_to(foreign):raise ValueError('Only the recorded generation-root prefix may be projected')
        relative=path.relative_to(foreign);destination=owned(local/relative)
        if not destination.is_relative_to(local) or not path.is_file():raise ValueError('Projected file escaped its proved project')
        actual=sha256_file(destination)
        if actual!=sha256_file(path) or destination.stat().st_ino!=path.stat().st_ino:
            raise ValueError('Projected mount inode/content differs')
        records[str(relative)]=dict(path=str(relative),sha256=actual,foreign_spelling=str(path),
            local_inode=destination.stat().st_ino,compute_inode=path.stat().st_ino)
        return str(logical/relative)
    expected=json.loads(json.dumps(source))
    expected['predicate_path']=mapped(source['predicate_path'])
    for row in expected['cells']:
        for key,value in list(row.items()):
            if key.endswith('_path') and value is not None:
                if key not in PATH_KEYS:raise ValueError('Unreviewed scientific-row pathname field')
                row[key]=mapped(value)
    # Proof JSON contains historical foreign path spellings. It is hashed and
    # interpreted here only, rather than recursively becoming runtime inputs.
    proofs=lambda p:dict(relative_name=str(p.relative_to(local)),sha256=sha256_file(p))
    return expected,dict(version=VERSION,alias_probe=proofs(probe_path),generation_execution=proofs(execution_path),nonce=binding(nonce),
        helper_sha256=sha256_file(__file__),mapped_files=[records[k] for k in sorted(records)],
        scientific_metadata_or_arrays_changed=False)


def project(source_path,probe_path,execution_path,output):
    expected,receipt=projection(source_path,probe_path,execution_path)
    folder=Path(output).resolve()
    if not folder.is_relative_to(ROOT.resolve()) or folder.exists():raise ValueError('Fresh owned projection output required')
    folder.mkdir(parents=True,exist_ok=False)
    expected.update(unpacked_catalog_path=str(owned(source_path).relative_to(ROOT.resolve())),unpacked_catalog_sha256=sha256_file(owned(source_path)),
        timing_control_projection=receipt)
    path=folder/'manifest.json';write_json(path,expected)
    validate(path)
    return binding(path)


def validate(path):
    path=owned(path);catalog=read(path);meta=catalog['timing_control_projection']
    original=owned(catalog['unpacked_catalog_path'])
    if sha256_file(original)!=catalog['unpacked_catalog_sha256']:raise ValueError('Original generated catalog changed')
    expected,receipt=projection(original,bound(meta['alias_probe']),bound(meta['generation_execution']))
    if (meta!=receipt or {k:v for k,v in catalog.items() if k not in ('unpacked_catalog_path','unpacked_catalog_sha256','timing_control_projection')}!=expected):
        raise ValueError('Projected catalog changed scientific metadata, identity or path evidence')
    if (catalog.get('schema')!='task10b_phase2_reference_catalog_v1' or catalog.get('complete') is not True
            or catalog.get('planned_cells')!=16 or catalog.get('executed_cells')!=16 or catalog.get('unattempted_cells')!=[]
            or len(catalog['cells'])!=16 or catalog.get('physics_executed') is not False):
        raise ValueError('Complete exact sixteen-cell offline D0 timing catalog required')
    expected_ids={f+'_original' for f in CANDIDATE_TTCS}
    expected_ids.update(f+'_uniform_'+f'{ttc:.3f}'.replace('.','p') for f,ttcs in CANDIDATE_TTCS.items() for ttc in ttcs)
    if {r['candidate_id'] for r in catalog['cells']}!=expected_ids:raise ValueError('Declared original/timing cells changed')
    logical=Path(read(bound(meta['alias_probe']))['local_root'])
    def row_file(value):
        path=Path(value)
        if not path.is_absolute() or not path.is_relative_to(logical):raise ValueError('Projected row left its fixed logical project spelling')
        return owned(path.relative_to(logical))
    evidence=[];models={};sources={}
    for row in catalog['cells']:
        family=row['family']
        if family not in CANDIDATE_TTCS:raise ValueError('Unknown D0 family')
        if row['kind']=='original_control':
            folder=row_file(row['diagnostic_screen_path']).parent
            evidence.append(dict(candidate_id=row['candidate_id'],files={name:binding(folder/name) for name in ('screen.json','screen_arrays.npz')},
                reference=binding(row_file(row['reference_path'])),unchanged_original_control=True))
            continue
        if (row['kind']!='uniform' or row.get('preserved_original_control') is not False
                or row['requested_ttc_s'] not in CANDIDATE_TTCS[family] or row['candidate_id']!=family+'_uniform_'+f"{row['requested_ttc_s']:.3f}".replace('.','p')):
            raise ValueError('Uniform family/timing identity changed')
        if row.get('generation_disposition')!='generated':
            if (row.get('generation_disposition')!='failed' or row.get('reference_path') is not None
                    or row.get('kinematic_admission') is not False or row.get('rejection_reasons')!=['reference_generation_failure']):
                raise ValueError('Missing timing payload needs its exact explicit offline failure')
            failure=owned(original.parent/row['candidate_id']/'failure.json')
            source_row=next(item for item in read(original)['cells'] if item['candidate_id']==row['candidate_id'])
            if read(failure)!=source_row:raise ValueError('Generation failure changed from its preserved catalog cell')
            evidence.append(dict(candidate_id=row['candidate_id'],files={'failure.json':binding(failure)},
                offline_rejection=True,native_attempt=False))
            continue
        folder=row_file(row['diagnostic_screen_path']).parent
        files={name:binding(folder/name) for name in ('screen.json','screen_arrays.npz','events.json','search.json')}
        reference=row_file(row['reference_path'])
        if (reference.parent!=folder or row_file(row['legacy_runtime_path'])!=reference
                or sha256_file(reference)!=row['reference_sha256'] or row['legacy_runtime_sha256']!=row['reference_sha256']):
            raise ValueError('Uniform runtime/reference identity differs')
        source=owned(ROOT/'outputs/four-punch-milestone'/family/'reference.npz')
        if row.get('source_reference_sha256')!=sha256_file(source):raise ValueError('Accepted source reference changed')
        if family not in sources:
            with np.load(source,allow_pickle=False) as archive:sources[family]={key:archive[key].copy() for key in archive.files}
            model=row_file(row['source_model_path'])
            if row.get('source_model_sha256')!=sha256_file(model):raise ValueError('Accepted source model changed')
            models[family]=mujoco.MjModel.from_xml_path(str(model))
        with np.load(reference,allow_pickle=False) as archive:data={key:archive[key].copy() for key in archive.files}
        scale=float(data['duration_scale']);rebuilt=retime_uniform(models[family],sources[family],scale)
        if (not np.isfinite(scale) or row.get('duration_scale')!=scale
                or str(data['source_reference_sha256'])!=sha256_file(source)
                or str(data['timing_kind'])!='uniform_complete_reference_clock' or not bool(data['source_timing_modified'])
                or float(data['requested_ttc_s'])!=row['requested_ttc_s']):raise ValueError('Retiming/source parameters differ')
        for key in ('t','time_s','qpos','qvel','qacc','source_reference_time_s','unrounded_duration_s','terminal_padding_s',
                    'launch_time','return_start','guard_hold_start','head_center','active_hand_index','attack_window_reference_s'):
            if not np.array_equal(data[key],rebuilt[key]):raise ValueError('Uniform derivative/path/clock reconstruction differs: '+key)
        events=read(files['events.json']['path']);search=read(files['search.json']['path']);screen=read(files['screen.json']['path'])
        if events!=rebuilt['events'] or any(row.get(k)!=v for k,v in events.items()):raise ValueError('Reviewed retimed event semantics differ')
        if (row.get('duration_s')!=float(data['t'][-1]) or row.get('episode_duration_s')!=row['duration_s']+1.
                or row.get('phase_denominator_s')!=row['episode_duration_s'] or row.get('guard_time_s')!=float(data['guard_hold_start'])
                or row.get('target_w')!=data['head_center'].tolist() or row.get('head_center')!=data['head_center'].tolist()
                or row.get('active_hand_index')!=int(data['active_hand_index'])):raise ValueError('Full horizon/target/hand changed')
        if (search.get('qualified_physical_execution') is not False or search.get('selected_contact')!=row.get('kinematic_contact')
                or not any(trial.get('duration_scale')==scale and trial.get('events')==events for trial in search['trials'])
                or screen.get('schema')!='task10b_phase2_fixed_path_screen_v1' or screen.get('physics_executed') is not False
                or screen.get('frames')!=len(data['t'])):raise ValueError('Search or diagnostic screen binding differs')
        with np.load(owned(files['screen_arrays.npz']['path']),allow_pickle=False) as archive:
            if not np.array_equal(archive['time_s'],data['t']) or not np.array_equal(archive['qacc'],data['qacc']):
                raise ValueError('Screen force/derivative time grid differs')
        contact=row['kinematic_contact'];reasons=[]
        if not contact['timing_valid']:reasons.append('uniform_reference_first_contact_timing_outside_contract')
        if contact['inactive_hand_first_or_simultaneous']:reasons.append('wrong_hand_first_or_simultaneous')
        if contact['any_contact_before_launch']:reasons.append('contact_before_launch')
        if row.get('rejection_reasons')!=reasons or row.get('kinematic_admission') is not (not reasons):
            raise ValueError('Offline admission/rejection was changed')
        evidence.append(dict(candidate_id=row['candidate_id'],files=files,reference=binding(reference),derivative_reconstruction_passed=True))
    return catalog,evidence
