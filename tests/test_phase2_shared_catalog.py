"""Synthetic offline shared catalog checks; no project references or rollouts."""
from dataclasses import asdict
import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from human_punching import phase2_shared_catalog as module
from human_punching.phase2_shared_catalog import (validate_plan,validate_start_bank,construct_shared_starts,resolve_cells,generate,archive_payload)
from human_punching.phase2_shared_reference import RECIPE_VERSION,SharedPreparationBounds,array_digest
from human_punching.phase2_target_reference import RetargetSettings,RetargetRejection
from human_punching.phase2_deadline import BoundedExecutionInterruption
from human_punching.phase2_domain import fingerprint
from human_punching.wbc.provenance import sha256_file
from human_punching.wbc.splits import assign_trajectory_groups
from test_phase2_shared_reference import fixture as shared_fixture


@pytest.fixture
def planned(tmp_path,monkeypatch):
    f=shared_fixture.__wrapped__();model=f['model'];model.opt.timestep=.005
    xml=tmp_path/'model.xml';mujoco.mj_saveLastXML(str(xml),model)
    tree=ET.parse(xml);root=tree.getroot();root.find('worldbody/body').set('name','Chest')
    root.find(".//joint[@name='upper_0']").set('name','LeftShoulder_z')
    root.find(".//joint[@name='upper_1']").set('name','RightArm_z')
    actuators=ET.SubElement(root,'actuator')
    for joint in root.findall('.//joint'):
        if joint.get('name'):ET.SubElement(actuators,'motor',joint=joint.get('name'),forcerange='-100 100')
    tree.write(xml);model=mujoco.MjModel.from_xml_path(str(xml))
    nominal=f['accepted_jab_qpos'].astype(np.float32)
    source=f['source'];source['qpos'][:,:7]=nominal[:7]
    source['qpos'][:,model.jnt_qposadr[-14:]]=0
    source['t']=source['time_s'];source['qvel']=np.zeros((201,54));source['qacc']=np.zeros((201,54))
    source_path=tmp_path/'source.npz';np.savez(source_path,**source)
    jab=tmp_path/'jab.npz';np.savez(jab,qpos=nominal[None,:])
    qa=model.jnt_qposadr[1:];limits=model.jnt_range[1:]
    metadata=dict(qpos_addresses=qa.tolist(),joint_limits_rad=limits.tolist())
    def save(name,value):
        path=tmp_path/(name+'.json');path.write_text(json.dumps(value));return dict(path=str(path),sha256=sha256_file(path))
    body=save('body',metadata)
    paths=dict(reference=jab,body_metadata=__import__('pathlib').Path(body['path']),model=xml)
    monkeypatch.setattr(module,'family_arguments',lambda *args:(None,paths))
    group=next(g for g,s in assign_trajectory_groups(range(100)).items() if s=='train')
    dq=np.zeros((4,48));dv=np.zeros((4,54))
    for i in range(1,4):dq[i]=.003*i;dv[i,6:]=.01*i
    domain=dict(schema='Task10BPhase2Domain/v1',selected_command_groups=[dict(split='train',snapshot=dict(
        command_id=group,replica_id=0,slot_id=19,payload_sha256='a'*64))],
        starts=dict(train=dict(joint_position_delta_rad=dq.tolist(),generalized_velocity_delta=dv.tolist())))
    domain_record=save('domain',domain)
    target=dict(command_id=group,split='train',original_replica_id=0,snapshot_slot_id=19,payload_sha256='a'*64,
        source_domain_sha256=domain_record['sha256'],head_target_w=source['head_center'].tolist(),domain_stage='D1')
    target['candidate_identity']=fingerprint(target)
    geometry=save('geometry',dict(schema='Task10BPhase2RegisteredTargets/v1',geometry_candidates=[target]))
    parent_rows=[dict(candidate_id=f'parent{timing}',family='jab',requested_ttc_s=timing,kinematic_admission=True,
        launch_tick=40,outward_end_tick=160,guard_time_s=.9,duration_s=1.,episode_duration_s=2.,phase_denominator_s=2.,
        reference_path=str(source_path),reference_sha256=sha256_file(source_path),source_model_path=str(xml),source_model_sha256=sha256_file(xml)) for timing in (.6,.75)]
    parents=save('parents',dict(schema=module.CATALOG_SCHEMA,cells=parent_rows))
    q=np.tile(nominal,(4,1));v=dv.astype(np.float32)
    for i in range(4):q[i,qa]=np.clip(nominal[qa].astype(float)+dq[i],limits[:,0],limits[:,1]).astype(np.float32)
    bank=tmp_path/'starts.npz';np.savez(bank,qpos=q,qvel=v)
    settings=asdict(RetargetSettings('Left',('LeftShoulder_z',)))
    # JSON tuples become lists; plans/recipes use their exact serialized form.
    settings=json.loads(json.dumps(settings))
    source_schedule=asdict(f['source_schedule'])
    plan=dict(schema=module.SCHEMA,family='jab',split='train',start_mode='shared_jab_saved_state',
        coordinate_precision='native_float32_canonical',domain_scope='full_split',maximum_cells=8,
        reference_byte_cap=2*1024**3,resident_byte_cap=256*1024**2,preparation_s=.4,
        preparation_bounds=asdict(SharedPreparationBounds()),retarget_settings=settings,shared_target_settings={},
        command_groups=[group],timings_s=[.6,.75],domain=domain_record,target_geometry=geometry,parent_catalog=parents,
        start_bank=dict(path=str(bank),sha256=sha256_file(bank)),release=str(tmp_path),
        start_bank_sha256_by_split={split:sha256_file(bank) for split in ('train','validation','test')},
        source_schedules={row['candidate_id']:source_schedule for row in parent_rows},
        cells=[dict(candidate_id=f'cell_{j}_{i}',parent_candidate_id=row['candidate_id'],
            target_candidate_identity=target['candidate_identity'],start_index=i) for j,row in enumerate(parent_rows) for i in range(4)])
    recipe=dict(schema=module.RECIPE_SCHEMA,family='jab',shared_recipe_version=RECIPE_VERSION,
        start_quantization_version=module.START_QUANTIZATION,
        physics_dt_s=.005,actor_dt_s=.020,**{k:plan[k] for k in ('start_mode','coordinate_precision','preparation_s',
            'preparation_bounds','retarget_settings','shared_target_settings','source_schedules')},
        start_bank_sha256_by_split=plan['start_bank_sha256_by_split'],
        **{key+'_sha256':plan[key]['sha256'] for key in ('domain','target_geometry','parent_catalog')},
        accepted_jab_reference_sha256=sha256_file(jab),accepted_jab_body_metadata_sha256=body['sha256'])
    plan['recipe']=save('recipe',recipe)
    plan_record=save('plan',plan)
    return SimpleNamespace(plan=plan,path=plan_record['path'],q=q,v=v,nominal=nominal.astype(float),metadata=metadata,domain=domain)


def test_resolve_exact_four_starts_and_fail_closed_group_target_bindings(planned):
    cells,_,context=resolve_cells(planned.plan)
    assert len(cells)==8 and context['native_saved_states'] is None
    assert context['start_ids']==[array_digest(planned.q[i],planned.v[i]) for i in range(4)]
    changed=json.loads(json.dumps(planned.plan));changed['cells']=changed['cells'][:-1]
    with pytest.raises(ValueError,match='four starts'):resolve_cells(changed)
    changed=json.loads(json.dumps(planned.plan));changed['source_schedules']['parent0.6']['launch_tick']=44
    # Recipe binding catches event drift before any project payload load.
    with pytest.raises(ValueError,match='recipe'):resolve_cells(changed)


def test_bank_has_no_family_pose_swap_root_noise_or_double_d3(planned):
    assert len(validate_start_bank(planned.q,planned.v,planned.nominal,planned.metadata,planned.domain,'train'))==4
    q=planned.q.copy();q[1,7]+=.003
    with pytest.raises(ValueError,match='frozen D3'):validate_start_bank(q,planned.v,planned.nominal,planned.metadata,planned.domain,'train')
    v=planned.v.copy();v[0,0]=1e-8
    with pytest.raises(ValueError):validate_start_bank(planned.q,v,planned.nominal,planned.metadata,planned.domain,'train')


def test_inward_native_quantization_keeps_exact_radius_and_records_applied_deltas(planned):
    nominal=planned.nominal.copy();nominal[7]=float(np.float32(.1))
    domain=json.loads(json.dumps(planned.domain))
    domain['starts']['train']['joint_position_delta_rad'][1][0]=.015
    q,v,receipt=construct_shared_starts(nominal,planned.metadata,domain,'train')
    naive=np.float32(nominal[7]+.015)
    assert float(naive)-nominal[7]>.015
    assert float(q[1,7])-nominal[7]<=.015 and q[1,7]==np.nextafter(naive,np.float32(nominal[7]))
    assert np.max(abs(v.astype(float)))<=.06
    assert receipt['quantization_version']==module.START_QUANTIZATION
    assert receipt['actual_joint_position_delta_rad'][1][0]==float(q[1,7])-nominal[7]
    validate_start_bank(q,v,nominal,planned.metadata,domain,'train')


def test_plan_requires_finite_explicit_recipe_and_per_start_identity(planned):
    assert validate_plan(planned.plan)
    for change in ({'preparation_s':.405},{'resident_byte_cap':2*1024**3},{'coordinate_precision':'silent_rounding'},
                   {'timings_s':[.6,.675,.75]}):
        with pytest.raises(ValueError):validate_plan({**planned.plan,**change})
    duplicate={**planned.plan,'cells':planned.plan['cells']+[planned.plan['cells'][0]],'maximum_cells':9}
    with pytest.raises(ValueError):validate_plan(duplicate)


def test_generation_preserves_exact_start_events_guard_targets_and_no_fake_native_history(planned,tmp_path,monkeypatch):
    from human_punching.phase2_reference import glove_positions
    def finalizer(model,candidate,metadata):return {**candidate,'torque':np.zeros((len(candidate['t']),48)),
        'glove_pos':glove_positions(model,candidate['qpos'])},metadata
    monkeypatch.setattr(module,'finalize_shared_target_feedforward',finalizer)
    def screen(model,candidate,caps,folder,deadline):
        (folder/'screen.json').write_text(json.dumps(dict(synthetic_only=True)))
    monkeypatch.setattr(module,'screen_chunks',screen)
    result=generate(planned.path,tmp_path/'generated',wall_seconds=120.)
    assert result['complete'] and result['executed_cells']==8 and not result['exception']
    assert not result['native_saved_state_available'] and not result['physics_executed']
    hashes=[]
    for row in result['cells']:
        assert row['start_state_sha256'] is None and row['requires_native_controller_capture']
        assert row['preparation_complete_tick']==80 and row['launch_tick']==120
        assert row['outward_end_tick']==240 and row['duration_s']==pytest.approx(1.4) and row['phase_denominator_s']==pytest.approx(2.4)
        assert row['own_family_guard_targets_serialized'] and not row['physical_qualification']
        with np.load(row['reference_path'],allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['qpos'][0],planned.q[row['start_index']])
            np.testing.assert_array_equal(archive['qvel'][0],planned.v[row['start_index']])
            np.testing.assert_array_equal(archive['family_guard_qpos'],archive['qpos'][80])
            assert archive['family_guard_glove_chest_pos'].shape==(2,3)
            assert not archive['phase_schedule_json'].dtype.hasobject
        hashes.append(row['reference_sha256'])
    assert len(set(hashes[:4]))==4


@pytest.mark.parametrize('kind',['geometric_rejection','timeout'])
def test_negative_and_timeout_evidence_stays_in_ledger(planned,tmp_path,monkeypatch,kind):
    def fail(*args,**kwargs):
        if kind=='timeout':raise BoundedExecutionInterruption(dict(reason='wall_budget',boundary='synthetic_test'))
        raise RetargetRejection('synthetic_unreachable')
    monkeypatch.setattr(module,'compose_shared_target',fail)
    result=generate(planned.path,tmp_path/'negative',wall_seconds=120.)
    assert not any(row['kinematic_admission'] for row in result['cells'])
    assert all(row['reference_identity_kind']=='generation_failure_receipt' for row in result['cells'])
    if kind=='timeout':assert result['executed_cells']==1 and len(result['unattempted_cells'])==7 and not result['complete']
    else:assert result['executed_cells']==8 and result['complete']
    for row in result['cells']:assert sha256_file(row['generation_failure_path'])==row['reference_sha256']


def test_archive_payload_refuses_objects_and_keeps_explicit_schedule():
    from human_punching.phase2_phases import PhaseSchedule
    value=archive_payload(dict(schedule=PhaseSchedule(4,8,12,16,20),qpos=np.zeros((2,55))))
    assert json.loads(str(value['phase_schedule_json']))['launch_tick']==8
    with pytest.raises(ValueError):archive_payload(dict(schedule=PhaseSchedule(4,8,12,16,20),bad=dict(hidden=True)))
