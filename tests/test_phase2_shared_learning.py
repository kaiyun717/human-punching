"""Shared training data/reset/reward checks; no native execution."""
from dataclasses import asdict,replace
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from human_punching.phase2_shared_learning import (SharedTrainingDeclaration,SharedSpecialistEnv,
    RESET_PROTOCOL,select_training_rows,preflight_references,load_configuration,execute_configuration,CONFIGURATION_SCHEMA)
from human_punching.phase2_grouped_learning import grouped_stopping_rules
from human_punching.phase2_learning import NativeSpecialistEnv,PilotBudget
from human_punching.phase2_phases import PhaseSchedule
from human_punching.phase2_sampling import IntegerReferenceSampler
from human_punching.wbc.provenance import sha256_file
from test_phase2_grouped_learning import fixture as grouped_fixture


def shared_fixture():
    base,catalog,domain,targets=grouped_fixture()
    declaration=SharedTrainingDeclaration(**asdict(base),start_bank_sha256='1'*64)
    schedule=PhaseSchedule(80,120,240,260,280);rows=[]
    for old in catalog['cells']:
        for index in range(4):
            rows.append(dict(**old|dict(candidate_id=old['candidate_id']+f'_start{index}',start_index=index,
                start_mode='shared_jab_saved_state',start_bank_path='synthetic_not_opened.npz',start_bank_sha256='1'*64,
                preparation_complete_tick=80,launch_tick=120,outward_end_tick=240,guard_hold_tick=260,reference_end_tick=280,
                phase_semantics_version=schedule.semantics_version,duration_s=280*.005,
                phase_denominator_s=schedule.phase_denominator_s,episode_duration_s=schedule.phase_denominator_s,
                own_family_guard_targets_serialized=True)))
    return declaration,dict(schema=catalog['schema'],cells=rows),domain,targets


def test_shared_selection_is_exact_train_endpoints_and_four_references_per_request():
    declaration,catalog,domain,targets=shared_fixture()
    rows,receipt=select_training_rows(declaration,catalog,domain,targets)
    assert len(rows)==256 and {r['start_index'] for r in rows}==set(range(4))
    assert {r['requested_ttc_s'] for r in rows}=={.6,.75} and {r['split'] for r in rows}=={'train'}
    assert receipt['reset_protocol']==RESET_PROTOCOL and not receipt['interpolation_used_for_training']
    changed=rows[0]['target_w'][0];rows[0]['target_w'][0]=999
    assert catalog['cells'][0]['target_w'][0]==changed


@pytest.mark.parametrize('change',[{'start_index':4},{'start_mode':'family_guard'},
    {'start_bank_sha256':'0'*64},{'own_family_guard_targets_serialized':False},
    {'kinematic_admission':False},{'preparation_complete_tick':0},{'phase_denominator_s':2.5}])
def test_shared_reference_mismatch_is_rejected_without_substitution(change):
    declaration,catalog,domain,targets=shared_fixture();catalog['cells'][0].update(change)
    with pytest.raises(ValueError):select_training_rows(declaration,catalog,domain,targets)


def test_missing_start_or_wrong_reset_sampling_version_is_rejected():
    declaration,catalog,domain,targets=shared_fixture()
    with pytest.raises(ValueError,match='Exactly256'):select_training_rows(declaration,{**catalog,'cells':catalog['cells'][1:]},domain,targets)
    for change in ({'reset_protocol':'original_extra_noise'},{'sampling_version':'linear'},
                   {'start_quantization_version':'silent_round'},{'start_bank_sha256':''}):
        with pytest.raises(ValueError):replace(declaration,**change)


def fake_environment():
    env=SharedSpecialistEnv.__new__(SharedSpecialistEnv);env.n=4;env.device='cpu';env.dt=.005;env.qadr=np.arange(7,55);env.vadr=np.arange(6,54)
    t=torch.arange(201)*.005;q=torch.zeros((201,55));q[:,3]=1.;q[:,7]=.01
    v=torch.zeros((201,54));v[:,6]=.03
    ref=dict(t=t,qpos=q,qvel=v,guard_hold_start=torch.tensor(.9),guard_glove_chest_pos=torch.ones((2,3)))
    rows=[dict(preparation_complete_tick=40,reference_end_tick=200)]*4
    env.library=SimpleNamespace(rows=rows,arrays=[ref]*4)
    env.integer_reference_sampler=IntegerReferenceSampler(env.library.arrays)
    env.ids_ref=torch.zeros(4,dtype=torch.long);env.ticks=torch.zeros(4,dtype=torch.long)
    env.rejected=torch.zeros(4,dtype=torch.bool);env.launched=env.rejected.clone();env.last_action=torch.ones((4,48))
    env.reset_counts=torch.zeros(4,dtype=torch.long);env.hips=0;env.initial_up=torch.zeros((4,3))
    physical_q=torch.zeros((4,55));physical_v=torch.zeros((4,54));resets=[]
    matrices=torch.eye(3).reshape(1,1,9).repeat(4,1,1)
    env.sim=SimpleNamespace(data=SimpleNamespace(qpos=physical_q,qvel=physical_v,xmat=matrices),
        reset=lambda ids:resets.append(ids.clone()),forward=lambda:None)
    env.support_contact=lambda:torch.ones((4,2),dtype=torch.bool);env.observe=lambda:torch.cat((physical_q,physical_v),1)
    return env,resets


def test_exact_connector_reset_has_no_second_d3_noise_and_only_one_native_reset():
    env,resets=fake_environment();env.reset(random_phase=False,perturb=True,reference_ids=[0,1,2,3])
    assert len(resets)==1 and torch.equal(env.sim.data.qpos[:,7],torch.full((4,),.01))
    assert torch.equal(env.sim.data.qvel[:,6],torch.full((4,),.03))
    assert not env.ticks.any() and not env.last_action.any() and torch.equal(env.reset_counts,torch.ones(4,dtype=torch.long))
    assert not env.sim.data.qvel[:,:6].any()


def test_source_rehearsal_is_explicitly_after_prep_and_never_adds_noise(monkeypatch):
    env,resets=fake_environment()
    draws=iter([torch.tensor([.1,.8,.2,.9]),torch.tensor([.2,.4,.6,.8])])
    monkeypatch.setattr(torch,'rand',lambda *args,**kwargs:next(draws))
    env.reset(random_phase=True,perturb=True,reference_ids=[0,1,2,3])
    assert len(resets)==1 and env.ticks[0]==env.ticks[2]==0
    assert env.ticks[1]>=40 and env.ticks[3]>=40 and env.ticks.max()<40+.8*(200-40)
    assert torch.equal(env.sim.data.qpos[:,7],torch.full((4,),.01))


def test_own_family_joint_reward_algebra_replaces_only_original_initial_target(monkeypatch):
    env,_=fake_environment();env.args=SimpleNamespace(guard_joint_weight=.5)
    env.ticks[:]=200;env.sim.data.qpos[:,7]=.2
    env.family_guard_qpos=torch.zeros((4,55));env.family_guard_qpos[:,3]=1.;env.family_guard_qpos[:,7]=.2
    env.library.sample=lambda key,ids,clock:torch.zeros((4,55))
    monkeypatch.setattr(NativeSpecialistEnv,'reward',lambda self,action,peak:torch.zeros(4))
    actual=env.reward(torch.zeros((4,48)),torch.zeros(4))
    torch.testing.assert_close(actual,torch.full((4,),.5*(1-np.exp(-4))))
    env.ticks[:]=0
    assert not env.reward(torch.zeros((4,48)),torch.zeros(4)).any()


def test_shared_configuration_hashes_caps_and_no_uncounted_execution(tmp_path,monkeypatch):
    declaration,_,_,_=shared_fixture();items={}
    for name in ('catalog','domain','target_geometry','predicate','mask','recipe','validation_configuration'):
        path=tmp_path/(name+'.json');path.write_text('{}');items[name]=dict(path=str(path),sha256=sha256_file(path))
    declaration=replace(declaration,**{key:items[name]['sha256'] for name,key in (('domain','domain_sha256'),
        ('target_geometry','target_geometry_sha256'),('predicate','predicate_sha256'),('mask','declared_mask_sha256'),('recipe','recipe_sha256'))})
    path=tmp_path/'declaration.json';path.write_text(json.dumps(asdict(declaration)));items['declaration']=dict(path=str(path),sha256=sha256_file(path))
    budget=PilotBudget();config=dict(schema=CONFIGURATION_SCHEMA,start_mode='shared_jab_saved_state',**items,
        budget=asdict(budget),stopping_rules=grouped_stopping_rules(budget),release=str(tmp_path))
    path=tmp_path/'configuration.json';path.write_text(json.dumps(config))
    actual,_,limits,_,_=load_configuration(path,sha256_file(path))
    assert actual.identity==declaration.identity and limits.transition_cap==262144
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='counted Slurm'):execute_configuration(path,sha256_file(path),tmp_path/'no_output')
    assert not (tmp_path/'no_output').exists()


def test_shared_preflight_bounds_tensor_inventory_and_binds_exact_guard_start(tmp_path):
    declaration,catalog,_,_=shared_fixture();row=catalog['cells'][0]
    nominal=np.zeros(55,np.float32);nominal[3]=1.;q=np.tile(nominal,(4,1));v=np.zeros((4,54),np.float32)
    bank=tmp_path/'bank.npz';np.savez(bank,qpos=q,qvel=v)
    jab=tmp_path/'jab.npz';np.savez(jab,qpos=nominal[None,:])
    body=tmp_path/'body.json';body.write_text(json.dumps(dict(qpos_addresses=list(range(7,55)),joint_limits_rad=[[-2,2]]*48)))
    domain=dict(starts=dict(train=dict(joint_position_delta_rad=np.zeros((4,48)).tolist(),generalized_velocity_delta=v.tolist())))
    path=tmp_path/'reference.npz';n=81;t=np.arange(n)*.005
    np.savez(path,t=t,qpos=np.tile(nominal,(n,1)),qvel=np.zeros((n,54)),qacc=np.zeros((n,54)),torque=np.zeros((n,48)),
        head_center=row['target_w'],active_hand_index=0,family_guard_qpos=nominal,family_guard_glove_chest_pos=np.ones((2,3)))
    row={**row,'start_bank_path':str(bank),'start_bank_sha256':sha256_file(bank),'reference_path':str(path),
        'reference_sha256':sha256_file(path),'legacy_runtime_path':str(path),'legacy_runtime_sha256':sha256_file(path),
        'duration_s':.4,'preparation_complete_tick':4}
    declaration=replace(declaration,start_bank_sha256=sha256_file(bank))
    receipt=preflight_references([row],declaration,domain,dict(reference=jab,body_metadata=body))
    assert receipt['unique_reference_archives']==1 and receipt['host_working_chunk_reservation_bytes']>0
    assert receipt['persistent_reference_tensor_reservation_bytes']<declaration.resident_reference_bytes
    with pytest.raises(ValueError,match='reservation'):
        preflight_references([row],replace(declaration,resident_reference_bytes=100),domain,dict(reference=jab,body_metadata=body))
    with pytest.raises(ValueError,match='identity'):
        preflight_references([{**row,'target_w':[9.,9.,9.]}],declaration,domain,dict(reference=jab,body_metadata=body))
    # Packed runtime omits qacc, keeps full-precision guard metadata, and binds
    # source derivative evidence independently from its active runtime bytes.
    from human_punching.phase2_runtime_pack import runtime_arrays,VERSION
    with np.load(path,allow_pickle=False) as archive:source={key:archive[key].copy() for key in archive.files}
    source['guard_hold_start']=np.asarray(.3)
    source['qpos']=source['qpos'].astype(float)
    source['qpos'][4:,7]=.123456789;source['family_guard_qpos']=source['qpos'][4].copy()
    evidence=tmp_path/'unpacked.npz';np.savez(evidence,**source)
    arrays,_=runtime_arrays(source);packed=tmp_path/'packed.npz';np.savez(packed,**arrays)
    packed_row={**row,'reference_path':str(packed),'reference_sha256':sha256_file(packed),
        'legacy_runtime_path':str(packed),'legacy_runtime_sha256':sha256_file(packed),
        'runtime_storage_version':VERSION,'unpacked_reference_path':str(evidence),'unpacked_reference_sha256':sha256_file(evidence)}
    packed_receipt=preflight_references([packed_row],declaration,domain,dict(reference=jab,body_metadata=body))
    assert packed_receipt['packed_offline_derivative_evidence_validated']
    assert 'qacc' not in arrays and arrays['family_guard_qpos'][7]!=arrays['qpos'][4,7]
    wrong={**arrays,'torque':np.ones_like(arrays['torque'])};tampered=tmp_path/'wrong_pack.npz';np.savez(tampered,**wrong)
    with pytest.raises(ValueError,match='native conversion'):
        preflight_references([{**packed_row,'reference_path':str(tampered),'reference_sha256':sha256_file(tampered),
            'legacy_runtime_path':str(tampered),'legacy_runtime_sha256':sha256_file(tampered)}],declaration,domain,dict(reference=jab,body_metadata=body))
