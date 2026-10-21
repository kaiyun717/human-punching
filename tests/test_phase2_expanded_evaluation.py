"""Synthetic declaration/adapter tests only: no GPU or actual domain trial."""
from dataclasses import asdict,replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from human_punching import phase2_learning as base
from human_punching.phase2_deadline import ExecutionDeadline
from human_punching.phase2_domain import fingerprint
from human_punching.phase2_qualification import PanelDeclaration,TrialBinding,qualify_panel
from human_punching.wbc.splits import assign_trajectory_groups
from human_punching.wbc.provenance import sha256_file
from human_punching.phase2_expanded_evaluation import (
    EvaluationLimits,INITIALIZATION_VERSION,DeadlineNativeSpecialistEnv,
    initial_state_identity,panel_domain_identity,validate_route,validate_trial_row,
    batch_requests,outcome_rows,preflight,declared_mask_decisions,
    validate_geometry_bindings,validate_declared_starts,validate_training_identity,execute,
    final_test_access_identity,launch_requests,FrozenSnapshotInput,
    ClosedProtocolNativeSpecialistEnv,learned_parameter_changes,write_launch_boundaries,
    validate_negative_reference_identity)


def fixture(split='validation'):
    groups=tuple(k for k,v in assign_trajectory_groups(range(1000)).items() if v==split)[:8 if split=='validation' else 16]
    q=np.zeros((4,55),np.float32);q[:,3]=1.;v=np.zeros((4,54),np.float32)
    q[:,7]=np.arange(4)*.001
    bindings=tuple(TrialBinding(f'fixture_{group}_{j}_{start}',group,1 if split=='validation' else 2,
        timing,start,initial_state_identity(q[start],v[start]),fingerprint({'group':group,'ttc':timing}),
        (0.,0.,1.)) for group in groups for j,timing in enumerate((.6,.675,.75)) for start in range(4))
    declaration=PanelDeclaration('jab',split,21021,groups,(.6,.75),.675,bindings,(bindings[0].trial_id,),
        'a'*64,'b'*64,'c'*64,'d'*64,'e'*64,8192,'f'*64)
    rows={};entries={}
    for binding in bindings:
        candidate=f'fixture_{binding.command_id}_{binding.ttc_s}'
        rows[candidate]=dict(candidate_id=candidate,kind='expanded_isolated',family='jab',
            command_id=binding.command_id,split=split,original_replica_id=binding.original_replica_id,
            reference_sha256=binding.reference_sha256,requested_ttc_s=binding.ttc_s,target_w=list(binding.target_w),
            launch_tick=4,outward_end_tick=180,duration_s=1.,episode_duration_s=2.,phase_denominator_s=2.,
            kinematic_admission=False,rejection_reasons=['synthetic_fixture_not_a_reference'])
        entries[binding.trial_id]=dict(trial_id=binding.trial_id,candidate_id=candidate,start_row=binding.start_index)
    route=dict(schema='task10b_phase2_evaluation_route_v1',family='jab',selection_splits=['train','validation'],
        test_used_for_selection=False,predicate_sha256=declaration.predicate_sha256,
        declared_mask_sha256=declaration.declared_mask_sha256,panel_domain_identities=[panel_domain_identity(declaration)],
        trained_timings_s=[.6,.75],interpolation_timing_s=.675,interpolation_used_for_training=False,
        recipe_sha256='1'*64,state='validation_candidate')
    return declaration,rows,entries,route,q,v


def test_full_panel_batches_preserve_exact_groups_timings_and_horizons():
    for split,size in (('validation',96),('test',192)):
        declaration,rows,entries,_,_,_=fixture(split)
        batches=batch_requests(declaration,entries,rows)
        assert sum(len(bindings) for _,bindings in batches)==size
        assert all(len(bindings)==4 for _,bindings in batches)
        assert all(len({b.reference_sha256 for b in bindings})==1 for _,bindings in batches)
        assert {b.trial_id for _,bindings in batches for b in bindings}=={b.trial_id for b in declaration.bindings}
    with pytest.raises(ValueError):batch_requests(declaration,entries,rows,17)


def test_first_validation_does_not_require_successful_validation_evidence():
    declaration,_,_,route,_,_=fixture()
    assert validate_route(declaration,route,route_sha256=declaration.route_sha256)['mechanical_gate_passed']
    assert not validate_route(declaration,route,route_sha256=declaration.route_sha256)['final_test']
    with pytest.raises(ValueError,match='Shared starts'):
        shared=replace(declaration,start_mode='shared_jab_saved_state')
        validate_route(shared,route|{'panel_domain_identities':[panel_domain_identity(shared)]},route_sha256=shared.route_sha256)


def test_final_test_cannot_open_evidence_before_route_freeze(monkeypatch):
    declaration,_,_,route,_,_=fixture('test')
    def prohibited(*args,**kwargs):raise AssertionError('Test/evidence payload should remain unopened')
    monkeypatch.setattr('human_punching.phase2_expanded_evaluation.read_json',prohibited)
    with pytest.raises(ValueError,match='frozen route'):
        validate_route(declaration,route,route_sha256=declaration.route_sha256)
    with pytest.raises(ValueError,match='lacks successful'):
        validate_route(declaration,route|{'state':'frozen'},route_sha256=declaration.route_sha256)


@pytest.mark.parametrize('change',[{'test_used_for_selection':True},{'interpolation_used_for_training':True},
    {'declared_mask_sha256':'0'*64},{'trained_timings_s':[.6,.9]},{'panel_domain_identities':[]}])
def test_route_mask_timing_and_test_isolation_fail_closed(change):
    declaration,_,_,route,_,_=fixture()
    with pytest.raises(ValueError):validate_route(declaration,route|change,route_sha256=declaration.route_sha256)


def test_native_start_digest_covers_velocity_and_zero_history_protocol():
    _,_,_,_,q,v=fixture()
    expected=initial_state_identity(q[0],v[0]);changed=v[0].copy();changed[0]=.001
    assert initial_state_identity(q[0],changed)!=expected
    assert initial_state_identity(q[0].astype(float),v[0].astype(float))==expected
    with pytest.raises(ValueError):initial_state_identity(q[0][:-1],v[0])


@pytest.mark.parametrize('change',[{'command_id':3307},{'split':'train'},{'original_replica_id':0},
    {'requested_ttc_s':.65},{'target_w':[0.,0.,1.1]},{'reference_sha256':'0'*64},
    {'kind':'original_control'},{'launch_tick':5},{'preparation_complete_tick':80},
    {'phase_denominator_s':1.5}])
def test_exact_per_trial_request_and_family_guard_binding(change):
    declaration,rows,entries,_,_,_=fixture();binding=declaration.bindings[0]
    row=rows[entries[binding.trial_id]['candidate_id']]
    validate_trial_row(binding,declaration,row)
    with pytest.raises(ValueError):validate_trial_row(binding,declaration,row|change)


def test_mask_has_every_request_and_no_nearest_or_seed_dependent_fallback():
    declaration,_,_,_,_,_=fixture()
    mask=dict(schema='task10b_phase2_declared_evaluation_mask_v1',requests=[dict(family='jab',
        **{k:v for k,v in asdict(b).items() if k!='trial_id'},admitted=i%2==0,reason='fixture')
        for i,b in enumerate(declaration.bindings)])
    decisions=declared_mask_decisions(declaration,mask)
    assert sum(row['admitted'] for row in decisions.values())==48
    with pytest.raises(KeyError):declared_mask_decisions(declaration,mask|{'requests':mask['requests'][:-1]})
    with pytest.raises(ValueError):declared_mask_decisions(declaration,mask|{'requests':mask['requests']+[mask['requests'][0]]})


def test_rejected_reference_starts_are_still_bound_without_loading_references(tmp_path):
    declaration,rows,entries,_,q,v=fixture();bank=tmp_path/'synthetic_starts.npz'
    np.savez_compressed(bank,qpos=q,qvel=v)
    for entry in entries.values():entry.update(start_bank_path=str(bank),start_bank_sha256=sha256_file(bank))
    manifest=dict(schema='task10b_phase2_expanded_manifest_v1',declaration_identity=declaration.identity,
        initialization_version=INITIALIZATION_VERSION,trials=list(entries.values()))
    decisions={b.trial_id:{'admitted':False} for b in declaration.bindings}
    _,_,starts,scope=preflight(declaration,manifest,{'cells':list(rows.values())},
        limits=EvaluationLimits(600.),mask_decisions=decisions)
    assert len(starts)==96 and scope['unique_reference_uncompressed_bytes']==0
    entries[declaration.bindings[0].trial_id]['start_row']=3
    with pytest.raises(ValueError,match='identity differs'):
        preflight(declaration,manifest,{'cells':list(rows.values())},limits=EvaluationLimits(600.),mask_decisions=decisions)


def test_partial_physical_failure_is_executed_incomplete_not_unattempted():
    declaration,_,_,_,_,_=fixture();bindings=declaration.bindings[:4]
    panel=dict(trials=[],exception={'type':'RuntimeError','message':'physical stop'},audit={'passed':False},
        hard_faults=[{'faults':{'fall':[1]}}],actual_physics_steps=20,actual_policy_updates=5)
    rows=outcome_rows(declaration,bindings,panel,PathFixture())
    assert len(rows)==4 and all(r['status']=='executed' and not r['execution_complete'] for r in rows)
    assert rows[1]['fallen'] and not rows[0]['fallen']
    assert all(r['trace_audit_pass'] is None and r['single_outward_stroke_pass'] is None for r in rows)
    panel['exception']['type']='BoundedExecutionInterruption'
    assert all(r['status']=='timeout' for r in outcome_rows(declaration,bindings,panel,PathFixture()))


def PathFixture():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]/'outputs/task10b/synthetic_nonexistent_batch'


def test_existing_evaluator_serializes_initial_partial_trace_on_safe_deadline(tmp_path,monkeypatch):
    class Recorder:
        def __init__(self):self.pre=[];self.actor=[];self.rows=[]
        def after(self,env):self.rows.append(1)
        def arrays(self):return {'qpos':np.zeros((len(self.rows),1,55),np.float32)}
    monkeypatch.setattr(base,'TraceRecorder',Recorder)
    deadline=ExecutionDeadline(120.,serialization_reserve_s=30.,session_started_monotonic=0.,clock=lambda:100.)
    row=dict(candidate_id='synthetic',family='jab',episode_duration_s=1.,phase_denominator_s=1.)
    env=SimpleNamespace(n=1,library=SimpleNamespace(rows=[row],commitment='fixture',content_digest=lambda:'fixture'),
        sim=SimpleNamespace(forward=lambda:None),reset=lambda **kwargs:None,hard_faults=[],
        execution_deadline=deadline,physics_steps_executed=0,actor_updates_executed=0)
    env.launch_gate=lambda:DeadlineNativeSpecialistEnv.launch_gate(env)
    panel=base.evaluate_policy(env,torch.nn.Linear(1,1),tmp_path/'synthetic_deadline',reference_ids=np.array([0]),
        initial_qpos=np.zeros((1,55)),initial_qvel=np.zeros((1,54)),seed=1,trial_ids=['synthetic'])
    assert panel['exception']['type']=='BoundedExecutionInterruption'
    assert panel['actual_physics_steps']==0 and panel['incomplete_trials']==1
    assert (tmp_path/'synthetic_deadline/raw_trace.npz').exists()
    assert env.deadline_interruption_receipt['reason']=='serialization_reserve'


def test_no_native_evaluation_without_counted_allocation(tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='counted Slurm'):
        execute(*(['unused']*7),tmp_path/'not_created',release='unused',limits=EvaluationLimits(600.))
    assert not (tmp_path/'not_created').exists()


def test_limits_forbid_oversized_or_unbounded_sessions():
    for kwargs in ({'wall_s':float('inf')},{'wall_s':14401.},{'wall_s':600.,'maximum_worlds':17},
                   {'wall_s':600.,'resident_reference_bytes':2*1024**3},{'wall_s':100.}):
        with pytest.raises(ValueError):EvaluationLimits(**kwargs)


def geometry_fixture():
    declaration,rows,entries,_,_,_=fixture()
    domain=dict(schema='Task10BPhase2Domain/v1',selected_command_groups=[dict(split='validation',
        snapshot=dict(command_id=g,replica_id=1,slot_id=g+1000,payload_sha256=fingerprint(g)))
        for g in declaration.command_groups])
    candidates=[]
    for group in domain['selected_command_groups']:
        s=group['snapshot']
        candidate=dict(source_domain_sha256='2'*64,command_id=s['command_id'],split='validation',
            original_replica_id=1,snapshot_slot_id=s['slot_id'],payload_sha256=s['payload_sha256'],
            head_target_w=[0.,0.,1.],placement=dict(longitudinal_m=0.,lateral_m=0.,yaw_rad=0.))
        candidate['candidate_identity']=fingerprint(candidate);candidates.append(candidate)
        for b in declaration.bindings:
            if b.command_id==s['command_id']:
                entry=entries[b.trial_id];entry['target_candidate_identity']=candidate['candidate_identity']
                rows[entry['candidate_id']]['target_candidate_identity']=candidate['candidate_identity']
    return declaration,rows,entries,domain,dict(schema='Task10BPhase2RegisteredTargets/v1',geometry_candidates=candidates)


@pytest.mark.parametrize('field,value',[('snapshot_slot_id',-1),('payload_sha256','3'*64),
    ('head_target_w',[0.,0.,1.1]),('original_replica_id',2)])
def test_target_pool_exact_snapshot_payload_slot_and_coordinates(field,value):
    declaration,rows,entries,domain,targets=geometry_fixture();manifest={'domain_sha256':'2'*64}
    assert validate_geometry_bindings(declaration,manifest,domain,targets,entries,rows)['one_placement_per_group']
    target=targets['geometry_candidates'][0];old=target['candidate_identity'];target[field]=value
    target['candidate_identity']=fingerprint({k:v for k,v in target.items() if k!='candidate_identity'})
    for entry in entries.values():
        if entry['target_candidate_identity']==old:
            entry['target_candidate_identity']=target['candidate_identity']
            rows[entry['candidate_id']]['target_candidate_identity']=target['candidate_identity']
    with pytest.raises(ValueError,match='canonical snapshot'):
        validate_geometry_bindings(declaration,manifest,domain,targets,entries,rows)


def test_target_pool_rejects_distinct_placement_even_when_numeric_target_matches():
    declaration,rows,entries,domain,targets=geometry_fixture();original=targets['geometry_candidates'][0]
    other={**original,'placement':dict(longitudinal_m=.05,lateral_m=0.,yaw_rad=0.)}
    other['candidate_identity']=fingerprint({k:v for k,v in other.items() if k!='candidate_identity'})
    targets['geometry_candidates'].append(other)
    # Replace one timing's four starts, so its common exact reference still agrees.
    candidate=entries[declaration.bindings[4].trial_id]['candidate_id']
    for entry in entries.values():
        if entry['candidate_id']==candidate:entry['target_candidate_identity']=other['candidate_identity']
    rows[candidate]['target_candidate_identity']=other['candidate_identity']
    with pytest.raises(ValueError,match='one placement'):
        validate_geometry_bindings(declaration,{'domain_sha256':'2'*64},domain,targets,entries,rows)


def test_declared_starts_require_exact_frozen_joint_perturbations_and_no_new_root_velocity(tmp_path):
    declaration,_,_,_,q,v=fixture();reference=tmp_path/'synthetic_source.npz'
    np.savez_compressed(reference,qpos=q[:1],qvel=v[:1])
    metadata=tmp_path/'synthetic_body.json'
    metadata.write_text(json.dumps(dict(qpos_addresses=list(range(7,55)),joint_limits_rad=[[-1.,1.]]*48)))
    paths=dict(reference=reference,body_metadata=metadata)
    dq=np.zeros((4,48));dq[:,0]=np.arange(4)*.001;dv=np.zeros((4,54))
    domain={'starts':{'validation':dict(joint_position_delta_rad=dq.tolist(),generalized_velocity_delta=dv.tolist())}}
    starts={b.trial_id:(q[b.start_index].copy(),v[b.start_index].copy()) for b in declaration.bindings}
    validate_declared_starts(declaration,domain,starts,paths)
    starts[declaration.bindings[0].trial_id][1][7]=.001
    with pytest.raises(ValueError,match='accepted family guard'):validate_declared_starts(declaration,domain,starts,paths)
    dv[0,0]=.001;domain['starts']['validation']['generalized_velocity_delta']=dv.tolist()
    with pytest.raises(ValueError,match='joint-only D3'):validate_declared_starts(declaration,domain,starts,paths)


def synthetic_validation_evidence(tmp_path):
    """Test-only receipts: never native evidence or authority to open real tests."""
    selected,_,_,candidate,_,_=fixture()
    outcomes=[]
    for b in selected.bindings:
        outcomes.append(dict(trial_id=b.trial_id,status='executed',request_identity=b.identity,
            checkpoint_sha256=selected.checkpoint_sha256,predicate_sha256=selected.predicate_sha256,
            trace_audit_pass=True,fallen=False,nonfinite=False,root_assistance=False,effort_cap_violation=False,
            movement_exploit=False,readiness_pass=True,immutable_launch_pass=True,continuity_history_pass=True,
            functional_recovery_pass=True,correct_hand=True,single_outward_stroke_pass=True,
            joint_limit_excursion_rad=0.,full_active_glove_rms_m=.001,outward_active_glove_rms_m=.001,
            request_s=0.,preparation_complete_s=0.,readiness_s=.1,final_planning_s=.2,launch_s=.2,
            outward_complete_s=1.,recovery_s=1.5,first_active_contact_s=.2+b.ttc_s,
            first_any_contact_s=.2+b.ttc_s,first_inactive_contact_s=None,
            visual_review='reviewed_acceptable',video_sha256='9'*64))
    # The declaration training digest must bind the selected receipt exactly.
    training=dict(checkpoint={'sha256':selected.checkpoint_sha256},fresh_transitions=selected.fresh_transitions,
        weight_changed=True,recipe_sha256='1'*64,domain_sha256='2'*64,target_geometry_sha256='3'*64)
    training_path=tmp_path/'training.json';training_path.write_text(json.dumps(training))
    selected=replace(selected,training_receipt_sha256=sha256_file(training_path))
    report=qualify_panel(selected,outcomes);assert report['passed']
    launch=dict(declaration_identity=selected.identity,slurm_job_id='SYNTHETIC_UNIT_TEST_ONLY',native_execution_requested=True,
        recipe_sha256='1'*64,domain_sha256='2'*64,target_geometry_sha256='3'*64)
    raw=tmp_path/'batch/raw_trace.npz';raw.parent.mkdir();raw.write_bytes(b'synthetic not a real trace')
    audit=dict(schema='task10b_phase2_independent_trace_audit_v1',passed=True,run=str(tmp_path),
        closed_protocol_launch_audited=True,outcome_resolution_passed=True,
        trial_outcome_resolutions=[dict(trial_id=b.trial_id,request_identity=b.identity,resolved=True,
            attempted=True,outcome_resolution='audited_complete',execution_complete=True,prefix_integrity_pass=True,
            exact_counter_pass=True,safe_actor_boundary_pass=True,hard_faults_absent=True,raw_trace_sha256=sha256_file(raw)) for b in selected.bindings],
        panels=[dict(closed_protocol_launch={'passed':True},
        path=str(raw.parent),raw_sha256=sha256_file(raw),planned_trials=96,execution_complete=True,passed=True,
        trials=[{'trial_id':b.trial_id} for b in selected.bindings])])
    evidence={}
    for name,value in [('qualification',report),('trace_audit',audit),('launch',launch),('training',training)]:
        path=tmp_path/(name+'.json')
        if name!='training':path.write_text(json.dumps(value))
        evidence[name]=dict(path=str(path),sha256=sha256_file(path))
    return evidence


def test_frozen_route_recomputes_successful_validation_and_binds_recipe_domain_and_target(tmp_path):
    declaration,_,_,route,_,_=fixture('test')
    route.update(state='frozen',domain_sha256='2'*64,target_geometry_sha256='3'*64,
        successful_validation=synthetic_validation_evidence(tmp_path))
    assert validate_route(declaration,route,route_sha256=declaration.route_sha256)['final_test']
    for key in ('recipe_sha256','domain_sha256','target_geometry_sha256'):
        with pytest.raises(ValueError,match='same selected recipe'):
            validate_route(declaration,route|{key:'8'*64},route_sha256=declaration.route_sha256)


def test_frozen_route_accepts_resolved_failed_prefixes_without_credit_or_reruns(tmp_path):
    declaration,_,_,route,_,_=fixture('test')
    evidence=synthetic_validation_evidence(tmp_path)
    qualification=json.loads((tmp_path/'qualification.json').read_text())
    selected=PanelDeclaration(**{**qualification['declaration'],
        'bindings':tuple(TrialBinding(**b) for b in qualification['declaration']['bindings'])})
    audit=json.loads((tmp_path/'trace_audit.json').read_text())
    outcomes=[row['source_outcome'] for row in qualification['trials']]
    for index in (0,4,8,12):
        resolution=audit['trial_outcome_resolutions'][index]
        resolution.update(outcome_resolution='audited_failed_prefix',execution_complete=False)
        outcomes[index].update(status='timeout',execution_complete=False,audit_resolution=resolution,
            functional_recovery_pass=False,visual_review='reviewed_failure',
            observed_active_glove_rms_m=.001,observed_outward_active_glove_rms_m=.001)
    audit['panels'][0]['execution_complete']=False
    replacement=qualify_panel(selected,outcomes)
    assert replacement['passed'] and replacement['successes']==92
    for name,value in [('qualification',replacement),('trace_audit',audit)]:
        path=tmp_path/(name+'.json');path.write_text(json.dumps(value));evidence[name]['sha256']=sha256_file(path)
    route.update(state='frozen',domain_sha256='2'*64,target_geometry_sha256='3'*64,successful_validation=evidence)
    assert validate_route(declaration,route,route_sha256=declaration.route_sha256)['final_test']


def test_learned_training_receipt_must_bind_selected_expanded_recipe():
    declaration,_,_,route,_,_=fixture();route.update(domain_sha256='2'*64,target_geometry_sha256='3'*64)
    ancestry=dict(sha256=declaration.parent_checkpoint_sha256,seed=declaration.seed,
        fresh_optimizer=True,fresh_simulation=True,independent_scratch_replication=False)
    training=dict(ancestry=ancestry,fresh_transitions=8192,status='bounded_stop',weight_changed=True,
        final_policy_digest='7'*64,checkpoint={'sha256':declaration.checkpoint_sha256},
        reload_parity={'parameters_equal':True,'actions_equal':True},trained_timings_s=[.6,.75],
        interpolation_used_for_training=False,**{k:route[k] for k in ('recipe_sha256','domain_sha256','target_geometry_sha256')})
    checkpoint=dict(schema='task10b_phase2_original_ppo_v1',observations=369,actions=48,
        fresh_transitions=8192,actor_parameters_sha256='7'*64,ancestry=ancestry)
    validate_training_identity(declaration,training,checkpoint,declaration.parent_checkpoint_sha256,route=route)
    with pytest.raises(ValueError,match='Learned original-PPO'):
        validate_training_identity(declaration,training|{'recipe_sha256':'6'*64},checkpoint,
            declaration.parent_checkpoint_sha256,route=route)


def test_final_test_claim_survives_route_mask_and_output_version_changes():
    declaration,_,_,_,_,_=fixture('test')
    initial=final_test_access_identity(declaration)
    assert final_test_access_identity(replace(declaration,route_sha256='9'*64,declared_mask_sha256='8'*64))==initial
    assert final_test_access_identity(replace(declaration,seed=21022))!=initial
    with pytest.raises(ValueError):final_test_access_identity(fixture()[0])


def test_frozen_snapshot_input_rechecks_exact_geometry_and_records_fresh_event_clock():
    declaration,rows,entries,domain,targets=geometry_fixture()
    binding=declaration.bindings[0];row=rows[entries[binding.trial_id]['candidate_id']]
    target=targets['geometry_candidates'][0]
    target['head_point_velocity_w']=[.01,.02,.03]
    requests=launch_requests(declaration,(binding,),row,{target['candidate_identity']:target})
    provider=FrozenSnapshotInput(requests)
    requests[0]['target_w'][0]=100.
    receipt=provider(24)[0]
    assert receipt['target_w']==list(binding.target_w)
    assert receipt['head_point_velocity_w']==[.01,.02,.03]
    assert receipt['checkpoint_sha256']==declaration.checkpoint_sha256
    assert receipt['snapshot_slot_id']==domain['selected_command_groups'][0]['snapshot']['slot_id']
    assert receipt['capture_episode_tick']==24 and receipt['input_episode_time_s']==.12
    receipt['head_point_velocity_w'][0]=999.
    assert provider(28)[0]['head_point_velocity_w']==[.01,.02,.03]
    for invalid in (-4,5,24.,True):
        with pytest.raises(ValueError):provider(invalid)


def test_checkpoint_versions_fail_closed_after_outward_launch_without_new_request():
    env=object.__new__(ClosedProtocolNativeSpecialistEnv)
    row=dict(family='jab',requested_ttc_s=.6,candidate_id='fixture',reference_sha256='1'*64,
        target_candidate_identity='2'*64,target_w=[0.,0.,1.])
    env.library=SimpleNamespace(rows=[row],arrays=[{'qpos':torch.zeros(2,55)}],
        durations=torch.tensor([2.]),launch_ticks=torch.tensor([4]),outward_ticks=torch.tensor([180]))
    env._requests=[{}];env.ids_ref=torch.zeros(1,dtype=torch.long);env.n=1;env.device='cpu'
    env._row_identities=(fingerprint(row),);env._cache_versions=env.cache_versions()
    env._bound_policy=None;env.physics_steps_executed=8;env.actor_updates_executed=2
    env.execution_deadline=SimpleNamespace(checkpoint=lambda *a,**kw:None)
    env.protocol_snapshot=lambda:{}
    env.protocol=SimpleNamespace(actor_boundary=lambda _:np.array([False]),assert_requests=lambda _:None)
    env.readiness=lambda:torch.ones(1,dtype=torch.bool)
    policy=torch.nn.Linear(3,2)
    env.physics_steps_executed=env.actor_updates_executed=0;env.bind_policy(policy)
    env.physics_steps_executed=8;env.actor_updates_executed=2
    assert not env.launch_gate()[0].any()
    with torch.no_grad():policy.weight.add_(.01)
    with pytest.raises(ValueError,match='checkpoint tensors changed'):env.launch_gate()


def test_learned_parameter_comparison_checks_actual_restored_bytes(tmp_path):
    policy=torch.nn.Linear(3,2);parent=tmp_path/'synthetic_parent.pt'
    torch.save({'policy':policy.state_dict()},parent)
    with pytest.raises(ValueError,match='unchanged accepted'):learned_parameter_changes(policy,parent)
    with torch.no_grad():policy.weight[0,0]+=.1
    report=learned_parameter_changes(policy,parent)
    assert report['changed_tensor_names']==['weight'] and report['independent_byte_comparison']


def test_lossless_boundary_sidecar_preserves_full_snapshot_digest_and_owned_dtypes(tmp_path):
    from human_punching.phase2_launch import snapshot_identity
    before=dict(qpos=np.arange(55,dtype=np.float32).reshape(1,55),support=np.array([[True,False]]),
        act=np.empty((1,0),np.float32),physics_steps=4,actor_updates=1)
    after={key:value.copy() if isinstance(value,np.ndarray) else value for key,value in before.items()}
    receipt=write_launch_boundaries(tmp_path,[dict(episode_tick=4,before=before,after=after)])
    with np.load(receipt['path'],allow_pickle=False) as arrays:
        for side in ('before','after'):
            stored=receipt['snapshots'][0][side]
            restored=stored['scalar_fields']|{key:arrays[name].copy() for key,name in stored['array_fields'].items()}
            assert snapshot_identity(restored)==snapshot_identity(before)==stored['snapshot_sha256']
            assert restored['qpos'].dtype==np.float32 and restored['support'].dtype==bool
    with pytest.raises(FileExistsError):write_launch_boundaries(tmp_path,[])


def test_failed_generation_receipt_is_negative_evidence_and_never_a_missing_npz_hash(tmp_path):
    assert not validate_negative_reference_identity({'reference_identity_kind':'npz_sha256'})
    path=tmp_path/'synthetic_generation_failure.json';path.write_text('{"reason":"synthetic IK failure"}')
    row=dict(reference_identity_kind='generation_failure_receipt',kinematic_admission=False,
        generation_failure_path=str(path),generation_failure_sha256=sha256_file(path),
        reference_sha256=sha256_file(path),rejection_reasons=['synthetic IK failure'])
    assert validate_negative_reference_identity(row)
    for change in ({'kinematic_admission':True},{'reference_path':'invented.npz'},
                   {'reference_sha256':'0'*64},{'rejection_reasons':[]}):
        with pytest.raises(ValueError):validate_negative_reference_identity(row|change)
    path.write_text('{"reason":"changed"}')
    with pytest.raises(ValueError,match='hash differs'):validate_negative_reference_identity(row)
