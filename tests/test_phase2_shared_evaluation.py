"""Shared initialization/phase tests; all tensors and receipts are synthetic."""
from dataclasses import replace
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from human_punching import phase2_shared_evaluation as shared
from human_punching.phase2_domain import fingerprint
from human_punching.phase2_expanded_evaluation import FrozenSnapshotInput,panel_domain_identity,validate_route
from human_punching.phase2_native_state import (
    NativeComparisonSnapshot,CONTROLLER,model_identity,validate_test_bank_freeze,capture_shared_start_bank,
)
from human_punching.phase2_phases import SavedExecutionState
from human_punching.phase2_qualification import PanelDeclaration,TrialBinding,qualify_panel
from human_punching.phase2_sampling import IntegerReferenceSampler
from human_punching.wbc.provenance import PROJECT_ROOT,sha256_file
from human_punching.wbc.splits import assign_trajectory_groups


def declaration():
    groups=tuple(key for key,value in assign_trajectory_groups(range(1000)).items() if value=='validation')[:8]
    bindings=tuple(TrialBinding(f'synthetic_{group}_{timing}_{start}',group,1,timing,start,'a'*64,
        fingerprint((group,timing,start)),(.8,.2,1.23)) for group in groups for timing in (.6,.675,.75) for start in range(4))
    return PanelDeclaration('jab','validation',21021,groups,(.6,.75),.675,bindings,(bindings[0].trial_id,),
        'b'*64,'c'*64,'d'*64,'e'*64,'f'*64,8192,'1'*64,start_mode='shared_jab_saved_state')


def row_for(binding):
    return dict(family='jab',reference_sha256=binding.reference_sha256,requested_ttc_s=binding.ttc_s,
        target_w=list(binding.target_w),command_id=binding.command_id,split='validation',original_replica_id=1,
        kind='shared_target_candidate',candidate_id='synthetic_shared_reference',start_mode='shared_jab_saved_state',
        start_index=binding.start_index,preparation_complete_tick=80,launch_tick=176,outward_end_tick=300,
        guard_hold_tick=360,reference_end_tick=400,duration_s=2.,episode_duration_s=3.,phase_denominator_s=3.,
        phase_semantics_version='task10b_explicit_connector_phases_v1')


@pytest.mark.parametrize('change',[{'preparation_complete_tick':0},{'preparation_complete_tick':201},
    {'launch_tick':175},{'start_index':3},{'phase_semantics_version':'implicit_first_motion'},
    {'guard_hold_tick':299},{'reference_end_tick':401}])
def test_shared_phases_require_explicit_connector_start_and_complete_denominator(change):
    panel=declaration();binding=panel.bindings[0];row=row_for(binding)
    assert shared.validate_shared_row(binding,panel,row).launch_tick==176
    with pytest.raises(ValueError):shared.validate_shared_row(binding,panel,row|change)


def test_shared_route_is_enabled_only_by_explicit_fixed_protocol():
    panel=declaration()
    route=dict(schema='task10b_phase2_evaluation_route_v1',family='jab',selection_splits=['train','validation'],
        test_used_for_selection=False,predicate_sha256=panel.predicate_sha256,declared_mask_sha256=panel.declared_mask_sha256,
        panel_domain_identities=[panel_domain_identity(panel)],trained_timings_s=[.6,.75],interpolation_timing_s=.675,
        interpolation_used_for_training=False,recipe_sha256='2'*64,state='validation_candidate')
    with pytest.raises(ValueError,match='Shared starts'):validate_route(panel,route,route_sha256=panel.route_sha256)
    assert validate_route(panel,route,route_sha256=panel.route_sha256,
        supported_start_mode='shared_jab_saved_state')['mechanical_gate_passed']


def native_bank_fixture():
    model=mujoco.MjModel.from_xml_path(str(PROJECT_ROOT/'outputs/four-punch-milestone/jab/model/human.xml'))
    data=mujoco.MjData(model);vector=np.empty(410);mujoco.mj_getState(model,data,vector,16383)
    states=[]
    for _ in range(4):
        states.append(SavedExecutionState(model_identity(model),CONTROLLER,16383,vector,410,np.zeros(48),np.zeros(48),
            '1'*64,0.,'preparation',0,0,np.ones(2,bool),0.,True,dict(ticks=np.array(0),reset_counts=np.array(1),
                launched=np.array(False),rejected=np.array(False),completed=np.array(0),fallen=np.array(0),
                physics_steps_executed=np.array(0),actor_updates_executed=np.array(0),hard_faults_json=np.frombuffer(b'[]',np.uint8))))
    inventory=json.dumps(dict(scope='independent_comparison_initialization_only',static_controller_sha256='2'*64))
    return model,NativeComparisonSnapshot(tuple(states),inventory)


def test_shared_bank_requires_all_four_complete_genuine_initial_states():
    model,bank=native_bank_fixture();q,v=shared.validate_native_bank(bank,model,'1'*64)
    assert q.shape==(4,55) and v.shape==(4,54)
    for change in ({'reference_phase':'ready_hold'},{'reference_clock_s':.02},{'previous_raw_action':np.ones(48)}):
        changed=replace(bank.states[0],**change)
        bad=NativeComparisonSnapshot((changed,*bank.states[1:]),bank.inventory_json)
        with pytest.raises(ValueError,match='genuine fresh jab'):shared.validate_native_bank(bad,model,'1'*64)
    with pytest.raises(ValueError,match='four shared starts'):
        shared.validate_native_bank(NativeComparisonSnapshot(bank.states[:1],bank.inventory_json),model,'1'*64)


class FakeBase:
    def __init__(self,args,rows,num_envs):
        self.n=num_envs;self.device='cpu';self.model=None;self.physics_steps_executed=self.actor_updates_executed=0
        self.ids_ref=torch.zeros(num_envs,dtype=torch.long);self.ticks=torch.zeros_like(self.ids_ref)
        self.last_action=torch.zeros(num_envs,48);self.last_support=torch.ones(num_envs,2,dtype=torch.bool)
        self.reset_counts=torch.zeros_like(self.ids_ref);self.rejected=torch.zeros(num_envs,dtype=torch.bool)
        self.launched=torch.zeros_like(self.rejected)
        q=torch.zeros(num_envs,55);q[:,3]=1.
        self.sim=SimpleNamespace(data=SimpleNamespace(qpos=q,qvel=torch.zeros(num_envs,54),ctrl=torch.zeros(num_envs,48),
            qacc_warmstart=torch.zeros(num_envs,54),qfrc_applied=torch.zeros(num_envs,54),xfrc_applied=torch.zeros(num_envs,24,6),
            act=torch.empty(num_envs,0),time=torch.zeros(num_envs)))
        self.library=SimpleNamespace(rows=rows,arrays=[dict(t=torch.arange(10)*.005,qpos=q[:1].repeat(10,1))],
            launch_ticks=torch.tensor([4]),outward_ticks=torch.tensor([8]),durations=torch.tensor([1.045]))
    @property
    def duration(self):return self.library.durations[self.ids_ref]
    def reset(self,**kwargs):
        self.sim.data.qpos[:]=torch.as_tensor(kwargs['initial_qpos']);self.sim.data.qvel[:]=torch.as_tensor(kwargs['initial_qvel'])
        self.reset_counts+=1;return self.observe()
    def observe(self):return torch.zeros(self.n,369)
    def readiness(self):return torch.ones(self.n,dtype=torch.bool)
    def install_integer_reference_playback(self):self.integer_reference_sampler=IntegerReferenceSampler(self.library.arrays)
    def advance_substep(self,action,**kwargs):
        self.sim.data.time+=.005;self.physics_steps_executed+=1;self.ticks+=1


def test_composed_wrapper_restores_once_before_prefix_and_preserves_counter_ownership(monkeypatch):
    model,bank=native_bank_fixture();state=bank.states[0]
    q=np.zeros((1,55),np.float32);q[:,3]=1.;v=np.zeros((1,54),np.float32)
    events=[]
    def restore(base,saved,**kwargs):
        events.append('full_restore');assert not hasattr(base,'integer_reference_sampler')
        base.sim.data.qacc_warmstart.fill_(.125)
        return {'complete_native_integration_equal':True}
    monkeypatch.setattr(shared,'SharedNativeBase',FakeBase);monkeypatch.setattr(shared,'restore_comparison',restore)
    monkeypatch.setattr(shared,'copy_native_integration',lambda *a:state.native_integration_state[None].copy())
    row=dict(family='jab',candidate_id='synthetic',reference_sha256='1'*64,requested_ttc_s=.6,
        target_candidate_identity='2'*64,target_w=[.8,.2,1.23])
    request=dict(trial_id='synthetic',request_identity='3'*64,family='jab',requested_ttc_s=.6,style='synthetic',
        candidate_id='synthetic',reference_sha256='1'*64,checkpoint_sha256='4'*64,target_candidate_identity='2'*64,
        snapshot_payload_sha256='5'*64,snapshot_slot_id=1,placement=dict(forward_m=0.,left_m=0.,yaw_rad=0.),
        head_point_velocity_w=[0.,0.,0.],target_w=row['target_w'])
    env=shared.SharedProtocolEnv(None,row,states=NativeComparisonSnapshot((state,),bank.inventory_json),
        trial_ids=['synthetic'],deadline=SimpleNamespace(checkpoint=lambda *a,**k:None),qpos=q,qvel=v,
        requests=[request],target_input=FrozenSnapshotInput([request]))
    env.bind_policy(torch.nn.Linear(369,48))
    env.reset(random_phase=False,perturb=False,reference_ids=np.array([0]),initial_qpos=q,initial_qvel=v)
    assert events==['full_restore'] and hasattr(env._base,'integer_reference_sampler')
    assert torch.all(env.sim.data.qacc_warmstart==.125)
    env.launch_gate();action=torch.zeros(1,48)
    for _ in range(4):env.advance_substep(action)
    env.last_action[:]=action;env.actor_updates_executed+=1
    assert env._base.actor_updates_executed==1 and env._base.physics_steps_executed==4
    assert env.launch_gate()[1].all() and env.closed_launch_records[0]['accepted']
    with pytest.raises(ValueError,match='subsequent reset'):
        env.reset(random_phase=False,perturb=False,reference_ids=np.array([0]),initial_qpos=q,initial_qvel=v)


def test_test_capture_freeze_failure_precedes_any_test_physical_input_read(tmp_path,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','SYNTHETIC_UNIT_TEST_ONLY');monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    def reject(_):raise ValueError('synthetic freeze rejected before physical input')
    monkeypatch.setattr('human_punching.phase2_native_state.validate_test_bank_freeze',reject)
    with pytest.raises(ValueError,match='before physical input'):
        capture_shared_start_bank('missing_catalog','missing_domain','test','missing_geometry','missing_release',
            tmp_path/'must_not_exist',300.,pre_test_freeze='synthetic')
    assert not (tmp_path/'must_not_exist').exists()


def synthetic_successful_freeze(tmp_path):
    selected=declaration();training=dict(checkpoint={'sha256':selected.checkpoint_sha256},weight_changed=True,
        exception=None,hard_faults=[],recipe_sha256='2'*64,domain_sha256='3'*64,target_geometry_sha256='4'*64)
    training_path=tmp_path/'training.json';training_path.write_text(json.dumps(training))
    selected=replace(selected,training_receipt_sha256=sha256_file(training_path));outcomes=[]
    for b in selected.bindings:
        outcomes.append(dict(trial_id=b.trial_id,status='executed',request_identity=b.identity,
            checkpoint_sha256=selected.checkpoint_sha256,predicate_sha256=selected.predicate_sha256,
            trace_audit_pass=True,fallen=False,nonfinite=False,root_assistance=False,effort_cap_violation=False,
            movement_exploit=False,readiness_pass=True,immutable_launch_pass=True,continuity_history_pass=True,
            functional_recovery_pass=True,correct_hand=True,single_outward_stroke_pass=True,
            joint_limit_excursion_rad=0.,full_active_glove_rms_m=.001,outward_active_glove_rms_m=.001,
            request_s=0.,preparation_complete_s=.4,readiness_s=.8,final_planning_s=.88,launch_s=.88,
            outward_complete_s=1.8,recovery_s=2.5,first_active_contact_s=.88+b.ttc_s,
            first_any_contact_s=.88+b.ttc_s,first_inactive_contact_s=None,
            visual_review='reviewed_acceptable',video_sha256='9'*64))
    qualification=qualify_panel(selected,outcomes);assert qualification['passed']
    launch=dict(declaration_identity=selected.identity,slurm_job_id='SYNTHETIC_UNIT_TEST_ONLY',native_execution_requested=True,
        recipe_sha256='2'*64,domain_sha256='3'*64,target_geometry_sha256='4'*64)
    raw=tmp_path/'batch/raw_trace.npz';raw.parent.mkdir();raw.write_bytes(b'SYNTHETIC not actual native data')
    audit=dict(schema='task10b_phase2_independent_trace_audit_v1',passed=True,run=str(tmp_path),
        closed_protocol_launch_audited=True,shared_saved_state_initialization_audited=True,outcome_resolution_passed=True,
        trial_outcome_resolutions=[dict(trial_id=b.trial_id,request_identity=b.identity,resolved=True,
            attempted=True,outcome_resolution='audited_complete',execution_complete=True,prefix_integrity_pass=True,
            exact_counter_pass=True,safe_actor_boundary_pass=True,hard_faults_absent=True,raw_trace_sha256=sha256_file(raw)) for b in selected.bindings],
        panels=[dict(path=str(raw.parent),
            raw_sha256=sha256_file(raw),planned_trials=96,passed=True,execution_complete=True,trials=[{'trial_id':b.trial_id} for b in selected.bindings])])
    evidence={}
    for name,value in [('qualification',qualification),('trace_audit',audit),('launch',launch),('training',training)]:
        path=tmp_path/(name+'.json')
        if name!='training':path.write_text(json.dumps(value))
        evidence[name]=dict(path=str(path),sha256=sha256_file(path))
    groups=[g for g,s in assign_trajectory_groups(range(1000)).items() if s=='test'][:16]
    route=dict(schema='task10b_phase2_pre_test_freeze_v1',state='frozen',selection_splits=['train','validation'],
        test_used_for_selection=False,start_mode='shared_jab_saved_state',
        test_bank_policy='capture_all_four_without_admission_filtering_or_start_changes',successful_validation=evidence,
        family='jab',trained_timings_s=[.6,.75],interpolation_timing_s=.675,selected_checkpoint_sha256=selected.checkpoint_sha256,
        reserved_test_command_groups=groups,recipe_sha256='2'*64,domain_sha256='3'*64,target_geometry_sha256='4'*64)
    path=tmp_path/'pre_test_freeze.json';path.write_text(json.dumps(route));return path,route


def test_pre_test_freeze_requires_actual_same_measured_shared_validation_recipe(tmp_path):
    path,route=synthetic_successful_freeze(tmp_path)
    assert validate_test_bank_freeze(path)['domain_sha256']=='3'*64
    route['recipe_sha256']='8'*64;path.write_text(json.dumps(route))
    with pytest.raises(ValueError,match='same successful measured'):validate_test_bank_freeze(path)


def test_pre_test_freeze_verifies_validation_trace_hash_and_reserved_groups(tmp_path):
    path,route=synthetic_successful_freeze(tmp_path)
    (tmp_path/'batch/raw_trace.npz').write_bytes(b'changed synthetic trace')
    with pytest.raises(ValueError,match='validation trace changed'):validate_test_bank_freeze(path)


def test_deadline_before_first_actor_still_retains_full_initial_native_state(tmp_path):
    _,bank=native_bank_fixture();panel=declaration();bindings=panel.bindings[:1]
    protocol=shared.SharedEvaluationProtocol();protocol.bank=bank
    protocol.metadata={'native_start_state_bank':{'path':'synthetic','sha256':'1'*64,'identity':bank.identity}}
    q=np.zeros((1,55),np.float32);v=np.zeros((1,54),np.float32)
    np.savez_compressed(tmp_path/'raw_trace.npz',qpos=q[None],qvel=v[None],
        reset_count=np.ones((1,1),int),reference_tick=np.zeros((1,1),int),ctrl=np.zeros((1,1,48),np.float32))
    env=SimpleNamespace(initial_native_state=bank.states[0].native_integration_state[None],restore_receipt={'synthetic':True})
    report=protocol.audit_initialization(tmp_path,env,bindings,q,v)
    assert not report['passed'] and not report['checks']['previous_raw_action']
    assert (tmp_path/'restored_native_initialization.npz').exists()
    assert report['retained_native_state']['sha256']==sha256_file(tmp_path/'restored_native_initialization.npz')
