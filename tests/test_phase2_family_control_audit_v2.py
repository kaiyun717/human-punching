"""Synthetic saved-evidence tamper checks; no native execution or models."""
from pathlib import Path
from copy import deepcopy
import importlib.util
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load('fixture_source',ROOT/'tests/test_phase2_audit.py')
from human_punching import phase2_family_control_audit_v2 as audit
def fixture(late=False):
 raw,receipt,plan,cells,launch,declaration,domain,targets,arrays=old.closed_fixture(late=late)
 row=next(iter(cells.values()));row.update(split='train',original_replica_id=1,family='cross',reference_identity_kind='reference_content')
 target=targets['geometry_candidates'][0];target['split']='train';target['candidate_identity']=audit.json_identity({k:v for k,v in target.items() if k!='candidate_identity'})
 row['target_candidate_identity']=target['candidate_identity'];domain['selected_command_groups'][0]['split']='train';domain['bound_sha256']='c'*64
 ident=row['candidate_id']+'_bank_00';plan['trials']=[ident]
 q=np.repeat(raw['qpos'][0],8,axis=0);v=np.repeat(raw['qvel'][0],8,axis=0)
 initial=dict(version='task10b_independent_family_guard_zero_history_v1',qpos=q[0].tolist(),qvel=v[0].tolist(),previous_raw_action=[0.]*48,
  held_action_initialization='original independent sim.reset, before first actor call',controller_initialization='fresh native Simulation and original sim.reset; no saved-state restoration')
 control=dict(version='task10b_phase2_family_training_target_frozen_control_v1',accepted_checkpoint={'sha256':'d'*64},initial_state_identities=[audit.json_identity(initial)]*8)
 req=dict(receipt['requests'][0]);req.update(trial_id=ident,target_candidate_identity=target['candidate_identity'],family='cross',reference_identity_kind='reference_content');req.pop('request_identity')
 req['request_identity']=audit.json_identity(dict(version=control['version'],request=req.copy(),bank_id=0,initial_state_identity=audit.json_identity(initial)))
 control['requests']=[req]+[dict(req,trial_id=f'unused_synthetic_{i}') for i in range(31)]
 receipt['requests']=[req];item=receipt['decisions'][0];item.update(trial_id=ident,request=req,observed_current_request=deepcopy(req))
 item['request']=dict(item['request']);item['request'].pop('reference_identity_kind')
 item['target_input_receipt'].update({k:req[k] for k in ('trial_id','request_identity','target_candidate_identity')})
 return raw,receipt,plan,cells,launch,control,domain,targets,arrays,q,v

def test_valid_and_late_rejection():
 for late in (False,True):
  result=audit.closed_launch_audit(*fixture(late));assert result['passed'],result
  assert result['accepted_by_world']=={0:not late}

@pytest.mark.parametrize('tamper',['initialq','initialv','acceptedcheckpoint','reference','target','slot','payload','targettime','latency','reset','history','warmstart','phase','clock','boundaryarray','requestidentity','missingdecision','duplicatedecision'])
def test_tamper(tamper):
 args=list(fixture());raw,receipt,plan,cells,launch,control,domain,targets,arrays,q,v=args
 item=receipt['decisions'][0];fresh=item['target_input_receipt']
 if tamper=='initialq':q[0,7]+=.1
 if tamper=='initialv':v[0,7]+=.1
 if tamper=='acceptedcheckpoint':control['accepted_checkpoint']['sha256']='f'*64
 if tamper=='reference':next(iter(cells.values()))['reference_sha256']='e'*64
 if tamper=='target':fresh['target_w']=[1.01,0.,1.2]
 if tamper=='slot':fresh['snapshot_slot_id']+=1
 if tamper=='payload':fresh['snapshot_payload_sha256']='f'*64
 if tamper=='targettime':fresh['capture_episode_tick']=0
 if tamper=='latency':item['planning_wall_s']=item['simulation_pause_wall_s']=.021
 if tamper=='reset':raw['reset_count'][5,0]=2
 if tamper=='history':raw['policy_previous_raw_action'][1,0,0]=.01
 if tamper=='warmstart':raw['policy_qacc_warmstart'][1,0,0]=.01
 if tamper=='phase':raw['policy_phase_denominator_s'][1,0]=3.
 if tamper=='clock':raw['reference_tick'][6,0]=5
 if tamper=='boundaryarray':arrays['before_qpos'][0,7]+=.01
 if tamper=='requestidentity':control['requests'][0]['request_identity']='e'*64
 if tamper=='missingdecision':receipt['decisions'].clear()
 if tamper=='duplicatedecision':receipt['decisions'].append(deepcopy(item))
 assert not audit.closed_launch_audit(*args)['passed']


@pytest.mark.parametrize('tamper',[None,'success','steps','extent','drop','admission'])
def test_negative_cycle_ledger_cannot_disappear_or_receive_success(tamper):
    control=dict(requests=[dict(trial_id=f't{i}',candidate_id='bad' if i<16 else 'good') for i in range(32)],
        reference_dispositions=[dict(candidate_id='bad',native_execution_admitted=False,generation_disposition='failed',rejection_reasons=['geometry_failed']),dict(candidate_id='good',native_execution_admitted=True,generation_disposition='generated',rejection_reasons=[])])
    native=[dict(trial_id=f't{i}',native_execution_admitted=i>=16,reference_generation_disposition='failed' if i<16 else 'generated',
        rejection_reasons=['geometry_failed'] if i<16 else [],status='reference_unavailable' if i<16 else 'unattempted',
        execution_extent='known_zero_before_attempt',actual_physics_steps=0,execution_complete=False,numeric_complete_cycle=False) for i in range(32)]
    if tamper=='success':native[0]['numeric_complete_cycle']=True
    if tamper=='steps':native[0]['actual_physics_steps']=1
    if tamper=='extent':native[0]['execution_extent']='unknown'
    if tamper=='drop':native.pop()
    if tamper=='admission':native[0]['native_execution_admitted']=True
    result=audit.ledger_checks(control,native)
    assert result['passed'] is (tamper is None)
    assert len(result['negative_trial_ids'])==16


@pytest.mark.parametrize('tamper',['extra_diagnostic','missing_family','changed_reference','changed_kind'])
def test_actual_typed_boundary_request_omits_only_diagnostic_kind(tamper):
 args=list(fixture());receipt=args[1];cells=args[3]
 if tamper=='extra_diagnostic':receipt['decisions'][0]['request']['reference_identity_kind']='reference_content'
 elif tamper=='missing_family':receipt['decisions'][0]['request'].pop('family')
 elif tamper=='changed_reference':receipt['decisions'][0]['request']['reference_sha256']='f'*64
 else:next(iter(cells.values()))['reference_identity_kind']='generation_failure_receipt'
 assert not audit.closed_launch_audit(*args)['passed']


def test_all48_retained_native_typed_boundaries_with_exact_namespace_proof():
 import json,hashlib
 from human_punching.phase2_nonuniform_control import Inputs
 ctx=Inputs(ROOT/'outputs/task10b/p2hostalias01/probe.json',root=ROOT)
 def read(p):return json.loads(ctx.owned(p).read_text())
 def arrays(p):
  with np.load(ctx.owned(p),allow_pickle=False)as a:return {k:a[k].copy() for k in a.files}
 domain=read(ROOT/'outputs/task10b/p2domain01/domain.json')
 domain['bound_sha256']=hashlib.sha256((ROOT/'outputs/task10b/p2domain01/domain.json').read_bytes()).hexdigest()
 targets=read(ROOT/'outputs/task10b/p2targets01/target_geometry.json');count=0
 for short in ('c','r'):
  run=ROOT/f'outputs/task10b/p2fc01{short}/run';control=read(run/'control_plan.json');launch=read(run/'launch.json')
  catalog=read(control['catalog']['path']);cells={r['candidate_id']:r for r in catalog['cells']}
  bank=arrays(launch['release_inputs']['bank']['path']);q=bank['qpos'][:8].astype(np.float32);v=bank['qvel'][:8].astype(np.float32)
  for index,candidate in enumerate(control['selected_candidate_ids']):
   if candidate not in control['admitted_candidate_ids']:continue
   folder=run/f'validation_{index:05d}'/candidate
   raw=arrays(folder/'raw_trace.npz');receipt=read(folder/'closed_protocol_launch.json');plan=read(folder/'plan.json')
   archive=ctx.owned(receipt['boundary_snapshots']['path'])
   assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['boundary_snapshots']['sha256']
   for full,item in zip(receipt['requests'],receipt['decisions'],strict=True):
    assert set(full)-set(item['request'])=={'reference_identity_kind'}
    assert full['reference_identity_kind']=='reference_content'
   result=audit.closed_launch_audit(raw,receipt,plan,cells,launch,control,domain,targets,arrays(archive),q,v)
   assert result['passed'],result
   count+=len(result['accepted_by_world'])
 assert count==48
