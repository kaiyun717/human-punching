"""Independent auditor fixtures catch contact/history/support bookkeeping faults."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location("phase2_independent", Path(__file__).resolve().parents[1]/"scripts/task10b_phase2_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_crossing_between_two_outside_samples_is_latched():
    times = np.array([0., .6, .61, 1.])
    centers = np.array([[1., 0., 0.], [.2, 0., 0.], [-.2, 0., 0.], [.1, 0., 0.]])
    assert audit.sphere_crossing(times, centers, np.zeros(3)) == pytest.approx(.6015)


def test_early_first_contact_cannot_be_replaced_by_late_hit():
    t = np.array([0., .48, .49, .6, .61])
    points = np.column_stack(([.5, .1, .5, .1, .1], np.zeros(5), np.zeros(5)))
    assert audit.sphere_crossing(t, points, np.zeros(3)) < .48
    with pytest.raises(ValueError):
        audit.sphere_crossing(t[::-1], points, np.zeros(3))


def test_postoutward_interval_dwell_never_borrows_preoutward_quiet():
    times = np.arange(101)*.005
    pose, support = np.ones(101, bool), np.ones(101, bool)
    dwell = audit.interval_dwell(times, pose, support, 40)
    assert dwell[40] == 0
    assert dwell[80] == pytest.approx(.2)
    support[65] = False
    interrupted = audit.interval_dwell(times, pose, support, 40)
    assert interrupted[65] == 0 and interrupted[80] == pytest.approx(.075)


def test_support_uses_each_interval_owner_and_strict_normal_force():
    raw = dict(time=np.zeros((3,2)), contact_offsets=np.array([0,0,2,4]),
        contact_geom=np.array([[0,1],[0,2],[0,1],[0,2]]),
        contact_force=np.array([[1.,0,0,0,0,0],[1.,0,0,0,0,0],[1e-6,0,0,0,0,0],[1.,0,0,0,0,0]]),
        contact_worldid=np.array([0,0,1,1]))
    support=audit.independent_support(raw,0,[1,2])
    assert not support[0].any()
    assert support[1,0].all() and not support[1,1].any()
    assert not support[2,1,0] and support[2,1,1]
    raw['contact_worldid'][0]=2
    with pytest.raises(ValueError):
        audit.independent_support(raw,0,[1,2])


def test_independent_float32_sampling_normalizes_root_and_holds_ends():
    t=np.array([0.,.5,1.])
    q=np.zeros((3,8));q[:,3]=1.;q[:,7]=[0.,.5,1.]
    result=audit.interpolate(t,q,np.array([-.1,.25,2.]),quaternion=True)
    np.testing.assert_allclose(result[:,7],[0,.25,1.])
    np.testing.assert_allclose(np.linalg.norm(result[:,3:7],axis=1),1.)


def test_pyramidal_support_normal_sums_owned_constraint_edges():
    raw=dict(contact_offsets=np.array([0,1,3]),contact_worldid=np.array([0,1,0]),
        contact_dimension=np.array([3,3,1]),contact_efc_address=np.array([[0],[1],[5]]),
        efc_force=np.zeros((2,2,8)))
    raw['efc_force'][0,0,:4]=[1,2,3,4]
    raw['efc_force'][1,1,1:5]=[10,20,30,40]
    raw['efc_force'][1,0,5]=7
    np.testing.assert_array_equal(audit.decode_normal_forces(raw),[10,100,7])
    np.testing.assert_array_equal(audit.decode_normal_forces(raw,pyramidal=False),[1,10,7])
    raw['contact_efc_address'][1,0]=6
    with pytest.raises(ValueError):
        audit.decode_normal_forces(raw)


def test_controller_audit_detects_root_force_history_and_force_pairing():
    n,w=8,2
    schema={"qpos_addresses":list(range(7,55)),"dof_addresses":list(range(6,54)),
        "kp":[2.]*48,"kd":[3.]*48,"torque_limits":[[-10.,10.]]*48}
    model=SimpleNamespace(jnt_range=np.tile([-1.,1.],(49,1)))
    q=np.zeros((n+1,w,55));q[:,:,3]=1.
    v=np.zeros((n+1,w,54));action=np.zeros((2,w,48))
    raw=dict(qpos=q,qvel=v,pre_qpos=q[:-1].copy(),pre_qvel=v[:-1].copy(),
        pre_qref=q[:-1].copy(),pre_vref=v[:-1].copy(),pre_raw_action=np.repeat(action,4,axis=0),
        pre_target=np.zeros((n,w,48)),pre_drive=np.zeros((n,w,48)),pre_feedforward=np.zeros((n,w,48)),
        actuator_force=np.zeros((n+1,w,48)),ctrl=np.zeros((n+1,w,48)),
        qfrc_actuator=np.zeros((n+1,w,54)),qfrc_applied=np.zeros((n+1,w,54)),xfrc_applied=np.zeros((n+1,w,3,6)),
        time=np.broadcast_to((np.arange(n+1)*.005)[:,None],(n+1,w)),reset_count=np.ones((n+1,w)),
        policy_episode_tick=np.array([0,4]),policy_action=action,policy_previous_raw_action=action.copy(),overflow=np.zeros((n+1,w)))
    assert audit.controller_audit(raw,model,schema)["passed"]
    raw["qfrc_actuator"][2,0,0]=1.
    assert not audit.controller_audit(raw,model,schema)["checks"]["no_root_actuation"]
    raw["qfrc_actuator"][2,0,0]=0.
    raw["policy_previous_raw_action"][1,0,0]=.25
    assert not audit.controller_audit(raw,model,schema)["checks"]["raw_previous_action"]
    raw["policy_previous_raw_action"][1,0,0]=0.
    raw["actuator_force"][1,0,0]=.01
    assert not audit.controller_audit(raw,model,schema)["checks"]["force_law_interval_pairing"]


def test_observation_preserves_raw_history_and_rejected_future_clock():
    t=np.array([0.,.5,1.],np.float32)
    qref=np.zeros((3,55),np.float32);qref[:,3]=1.;qref[:,7]=[0.,.5,1.]
    ref={"t":t,"qpos":qref,"qvel":np.zeros((3,54),np.float32),"glove_pos":np.zeros((3,2,3),np.float32)}
    q=np.zeros((2,55),np.float32);q[:,3]=1.
    previous=np.full((2,48),1.7,np.float32)
    observation=audit.observation(q,np.zeros((2,54),np.float32),np.zeros((2,2,3),np.float32),previous,
        np.array([.2,.2],np.float32),ref,2.,np.array([False,True]))
    assert observation.shape==(2,369)
    np.testing.assert_array_equal(observation[:,213:261],previous)
    assert observation[0,264]>observation[1,264]  # First future hinge sees .1s unless rejected.
    np.testing.assert_array_equal(previous,np.full((2,48),1.7,np.float32))


def test_launch_commitment_binds_current_state_request_and_history():
    q=np.zeros((9,1,55));v=np.zeros((9,1,54));ctrl=np.zeros((9,1,48))
    raw={"policy_action":np.zeros((2,1,48)),"policy_previous_raw_action":np.zeros((2,1,48)),
        "policy_qacc_warmstart":np.zeros((2,1,54)),"policy_readiness":np.ones((2,1),bool),
        "qpos":q,"qvel":v,"ctrl":ctrl}
    cell={"candidate_id":"jab_test","launch_tick":4,"family":"jab","requested_ttc_s":.6,
        "reference_sha256":"q","legacy_runtime_sha256":"l","target_w":[1.,0.,1.]}
    entry={"world":0,"requested_episode_tick":4,"measured_human_state":{"qpos":q[4,0].tolist(),
        "qvel":v[4,0].tolist(),"ctrl":ctrl[4,0].tolist(),"qacc_warmstart":[0.]*54},
        "previous_raw_action":[0.]*48,"outgoing_raw_action":[0.]*48,"requested_family":"jab",
        "requested_ttc_s":.6,"requested_style":"jab_test","reference_sha256":"q","legacy_runtime_sha256":"l",
        "target_w":[1.,0.,1.],"actual_launch_time_s":.02,"disposition":"committed","state_and_history_replaced":False}
    result=audit.commitment_audit(raw,[entry],{"candidates":["jab_test"]},{"jab_test":cell})
    assert result["passed"]
    rejected=deepcopy(entry);rejected["actual_launch_time_s"]=None;rejected["disposition"]="current_readiness_rejection"
    # A late finalizer can reject a physically ready state. D0 retains its
    # historical readiness-only launch, while expanded evidence uses the
    # independently checked closed-protocol disposition.
    assert not audit.commitment_audit(raw,[rejected],{"candidates":["jab_test"]},{"jab_test":cell})["passed"]
    assert audit.commitment_audit(raw,[rejected],{"candidates":["jab_test"]},{"jab_test":cell},accepted_by_world={0:False})["passed"]
    entry["target_w"]=[2.,0.,1.]
    assert not audit.commitment_audit(raw,[entry],{"candidates":["jab_test"]},{"jab_test":cell})["passed"]


def closed_fixture(*,late=False):
    """A synthetic one-world receipt; does not execute or qualify a policy."""
    q=np.zeros((9,1,55),np.float32);q[:,:,3]=1.;q[:,0,7]=np.arange(9)*.001
    v=np.zeros((9,1,54),np.float32);v[:,:,9]=.7
    a=np.zeros((2,1,48),np.float32);ctrl=np.zeros((9,1,48),np.float32)
    t=(np.arange(9,dtype=np.float32)*np.float32(.005))[:,None]
    ticks=np.arange(9,dtype=np.int64)[:,None]
    if late:ticks=np.minimum(ticks,4)
    raw=dict(qpos=q,qvel=v,ctrl=ctrl,time=t,reference_tick=ticks,
        qfrc_applied=np.zeros((9,1,54),np.float32),xfrc_applied=np.zeros((9,1,3,6),np.float32),
        reset_count=np.ones((9,1),np.int64),support_contact=np.ones((9,1,2),bool),
        policy_qpos=q[[0,4]].copy(),policy_qvel=v[[0,4]].copy(),policy_ctrl=ctrl[[0,4]].copy(),
        policy_qacc_warmstart=np.zeros((2,1,54),np.float32),policy_action=a,
        policy_previous_raw_action=a.copy(),policy_readiness=np.ones((2,1),bool),
        policy_reference_tick=ticks[[0,4]].copy(),policy_reference_id=np.zeros((2,1),np.int64),
        policy_phase_denominator_s=np.full((2,1),2.,np.float32),
        policy_launched=np.array([[False],[not late]]),policy_rejected=np.array([[False],[late]]))
    raw["launched"]=(np.arange(9)[:,None]>4)&(not late)
    raw["rejected"]=(np.arange(9)[:,None]>4)&late
    initial=dict(version="task10b_independent_family_guard_zero_history_v1",qpos=q[0,0].tolist(),qvel=v[0,0].tolist(),
        previous_raw_action=[0.]*48,held_action_initialization="original independent sim.reset, before first actor call",
        controller_initialization="fresh native Simulation and original sim.reset; no saved-state restoration")
    binding=dict(trial_id="synthetic_launch",command_id=31,original_replica_id=1,ttc_s=.6,start_index=0,
        start_state_sha256=audit.json_identity(initial),reference_sha256="a"*64,target_w=[1.,0.,1.2])
    domain=dict(selected_command_groups=[dict(split="validation",snapshot=dict(command_id=31,replica_id=1,
        slot_id=123,payload_sha256="b"*64))])
    target=dict(source_domain_sha256="c"*64,command_id=31,split="validation",original_replica_id=1,
        snapshot_slot_id=123,payload_sha256="b"*64,placement=dict(forward_m=0.,left_m=0.,yaw_rad=0.),
        head_target_w=binding["target_w"],head_point_velocity_w=[.04,-.01,.02])
    target["candidate_identity"]=audit.json_identity(target)
    row=dict(candidate_id="synthetic_jab",family="jab",command_id=31,split="validation",launch_tick=4,
        requested_ttc_s=.6,reference_sha256="a"*64,target_w=binding["target_w"],
        target_candidate_identity=target["candidate_identity"])
    declaration=dict(bindings=[binding],checkpoint_sha256="d"*64,split="validation")
    launch=dict(domain_sha256="c"*64,evaluated_checkpoint=dict(sha256="d"*64))
    plan=dict(candidates=[row["candidate_id"]],trials=[binding["trial_id"]],reference_ids=[0],phase_denominators_s=[2.])
    request=dict(trial_id=binding["trial_id"],request_identity=audit.json_identity(binding),family="jab",
        requested_ttc_s=.6,style=row["candidate_id"],candidate_id=row["candidate_id"],reference_sha256="a"*64,
        checkpoint_sha256="d"*64,target_candidate_identity=target["candidate_identity"],snapshot_payload_sha256="b"*64,
        snapshot_slot_id=123,placement=target["placement"],head_point_velocity_w=target["head_point_velocity_w"],target_w=binding["target_w"])
    before=dict(qpos=q[4].copy(),qvel=v[4].copy(),ctrl=ctrl[4].copy(),qacc_warmstart=raw["policy_qacc_warmstart"][1].copy(),
        qfrc_applied=raw["qfrc_applied"][4].copy(),xfrc_applied=raw["xfrc_applied"][4].copy(),act=np.zeros((1,0),np.float32),
        previous_raw_action=a[1].copy(),ticks=ticks[4].copy(),reference_ids=np.array([0],np.int64),
        reset_counts=np.array([1],np.int64),support=raw["support_contact"][4].copy(),time=t[4].copy(),
        phase_denominators_s=np.array([2.],np.float32),physics_steps=4,actor_updates=1,
        rejected=np.array([False]),launched=np.array([False]))
    arrays={};record=dict(episode_tick=4)
    for side in ("before","after"):
        fields={k:side+"_"+k for k,v in before.items() if isinstance(v,np.ndarray)}
        arrays.update({name:before[k].copy() for k,name in fields.items()})
        record[side]=dict(array_fields=fields,scalar_fields={k:v for k,v in before.items() if k not in fields},
            snapshot_sha256=audit.launch_snapshot_digest(before))
    fresh={key:request[key] for key in ("trial_id","request_identity","target_candidate_identity","snapshot_payload_sha256",
        "snapshot_slot_id","placement","head_point_velocity_w","target_w")}
    fresh.update(capture_episode_tick=4,input_episode_time_s=.020,source_kind="fixed_canonical_snapshot",no_prediction=True,
        head_geometry="head_collision",head_velocity_method="jacobian_qvel")
    reason="final_planning_late" if late else None
    item=dict(world=0,trial_id=binding["trial_id"],boundary_snapshot_index=0,reference_tick=4,request=request,
        measured_boundary_state_sha256=audit.launch_snapshot_digest(before),after_planning_state_sha256=audit.launch_snapshot_digest(before),
        native_fields_captured=sorted(before),planning_wall_s=.021 if late else .001,target_callback_wall_s=.0001,
        cache_lookup_wall_s=.0001,simulation_pause_wall_s=.021 if late else .001,simulated_time_advanced_s=0.,
        target_input_receipt=fresh,observed_current_request=request.copy(),current_positional_support_ready=True,
        accepted=not late,reason=reason,reasons=[reason] if reason else [],actual_launch_tick=None if late else 4,
        same_continuous_cached_reference=True,exact_joint_return_required=False,quiet_hook_gate_added=False,
        live_g1_state_recaptured=False,full_saved_state_restore=False,hardware_real_time_qualified=False)
    receipt=dict(schema="task10b_phase2_closed_protocol_native_receipt_v1",initialization_version="task10b_independent_family_guard_zero_history_v1",
        physical_state_or_history_replaced=False,arbitrary_live_state_admission=False,real_time_operation_qualified=False,
        requests=[request],decisions=[item],boundary_snapshots=dict(schema="task10b_phase2_lossless_launch_boundaries_v1",snapshots=[record]))
    return raw,receipt,plan,{row["candidate_id"]:row},launch,declaration,domain,dict(geometry_candidates=[target]),arrays


def test_closed_protocol_independent_audit_accepts_nonquiet_tracking_error_and_preserved_late_rejection():
    for late in (False,True):
        evidence=closed_fixture(late=late)
        result=audit.closed_launch_audit(*evidence)
        assert result["passed"],result
        assert result["accepted_by_world"]=={0:not late}
        assert not result["complete_saved_state_restore_audited"]


@pytest.mark.parametrize("tamper",["payload","slot","placement","head_velocity","target","timestamp","checkpoint",
    "warmstart","previous_action","qpos","reference_clock","phase","reset","latency","snapshot_digest",
    "forged_consistent_snapshot","duplicate","missing","disposition"])
def test_closed_protocol_independent_audit_detects_nontrivial_tampering(tamper):
    evidence=list(closed_fixture());raw,receipt,plan,cells,launch,declaration,domain,targets,arrays=evidence
    item=receipt["decisions"][0];fresh=item["target_input_receipt"]
    if tamper=="payload":fresh["snapshot_payload_sha256"]="f"*64
    if tamper=="slot":fresh["snapshot_slot_id"]+=1
    if tamper=="placement":fresh["placement"]=dict(forward_m=.05,left_m=0.,yaw_rad=0.)
    if tamper=="head_velocity":fresh["head_point_velocity_w"]=[0.,0.,0.]
    if tamper=="target":fresh["target_w"]=[1.001,0.,1.2]
    if tamper=="timestamp":fresh["capture_episode_tick"]=0
    if tamper=="checkpoint":launch["evaluated_checkpoint"]["sha256"]="f"*64
    if tamper=="warmstart":raw["policy_qacc_warmstart"][1,0,0]=.01
    if tamper=="previous_action":raw["policy_previous_raw_action"][1,0,0]=.01
    if tamper=="qpos":raw["policy_qpos"][1,0,7]+=.01
    if tamper=="reference_clock":raw["reference_tick"][6,0]=5
    if tamper=="phase":raw["policy_phase_denominator_s"][1,0]=3.
    if tamper=="reset":raw["reset_count"][5,0]=2
    if tamper=="latency":item["planning_wall_s"]=.021;item["simulation_pause_wall_s"]=.021
    if tamper=="snapshot_digest":arrays["before_qpos"][0,7]+=.01
    if tamper=="forged_consistent_snapshot":
        record=receipt["boundary_snapshots"]["snapshots"][0]
        for side in ("before","after"):
            arrays[side+"_qpos"][0,7]+=.01
            state={**record[side]["scalar_fields"],**{k:arrays[name] for k,name in record[side]["array_fields"].items()}}
            record[side]["snapshot_sha256"]=audit.launch_snapshot_digest(state)
        item["measured_boundary_state_sha256"]=record["before"]["snapshot_sha256"]
        item["after_planning_state_sha256"]=record["after"]["snapshot_sha256"]
    if tamper=="duplicate":receipt["decisions"].append(deepcopy(item))
    if tamper=="missing":receipt["decisions"].clear()
    if tamper=="disposition":raw["launched"][7,0]=False
    assert not audit.closed_launch_audit(*evidence)["passed"]


def test_integer_reference_sampling_uses_exact_knots_and_rejected_future_offsets():
    ticks=np.array([4001,4001],np.int64);clock=ticks.astype(np.float32)*np.float32(.005)
    qref=np.zeros((6001,55),np.float32);qref[:,3]=1.;qref[:,7]=np.arange(6001)%3
    reference=dict(t=np.arange(6001,dtype=np.float64)*.005,qpos=qref,qvel=np.zeros((6001,54),np.float32),
        glove_pos=np.zeros((6001,2,3),np.float32))
    q=np.zeros((2,55),np.float32);q[:,3]=1.
    observed=audit.observation(q,np.zeros((2,54),np.float32),np.zeros((2,2,3),np.float32),np.zeros((2,48),np.float32),
        clock,reference,35.,np.array([False,True]),reference_ticks=ticks)
    np.testing.assert_array_equal(audit.integer_sample(qref,ticks),qref[ticks])
    # .25s is exactly50 master ticks; no second float clock or interpolation.
    assert observed[0,315]==qref[4051,7] and observed[1,315]==qref[4001,7]
    with pytest.raises(ValueError):audit.integer_sample(qref,np.array([2.5]))


def test_packed_reference_fk_uses_native_float32_qpos(tmp_path):
    import mujoco
    path=audit.ROOT/'outputs/four-punch-milestone/jab/model/human.xml'
    model=mujoco.MjModel.from_xml_path(str(path));q=np.stack([model.qpos0.copy()]*3);q[:,7]+=.123456789
    reference=tmp_path/'packed.npz';np.savez(reference,t=np.arange(3)*.005,qpos=q.astype(np.float32),qvel=np.zeros((3,54)),torque=np.zeros((3,48)))
    arrays=audit.reference_arrays(reference,model);data=mujoco.MjData(model);data.qpos[:]=q[0].astype(np.float32)
    mujoco.mj_kinematics(model,data);ids=[model.site(side+'Hand_center').id for side in ('Left','Right')]
    np.testing.assert_array_equal(arrays['glove_pos'][0],data.site_xpos[ids].astype(np.float32))


def test_shared_launch_requires_independently_verified_full_initialization():
    evidence=list(closed_fixture());raw,receipt,plan,cells,launch,declaration,*_=evidence
    declaration['start_mode']='shared_jab_saved_state';receipt['initialization_version']='task10b_shared_jab_complete_native_initialization_v1'
    trial=declaration['bindings'][0];trial['start_state_sha256']='d'*64
    request=receipt['requests'][0];request['request_identity']=audit.json_identity(trial)
    receipt['decisions'][0]['request']['request_identity']=request['request_identity']
    receipt['decisions'][0]['observed_current_request']['request_identity']=request['request_identity']
    receipt['decisions'][0]['target_input_receipt']['request_identity']=request['request_identity']
    assert not audit.closed_launch_audit(*evidence)['passed']
    initialization=dict(passed=True,trial_state_bindings={trial['trial_id']:trial['start_state_sha256']})
    assert audit.closed_launch_audit(*evidence,shared_initialization=initialization)['passed']


def shared_initialization_fixture(tmp_path):
    """CPU-storage copy fixture; fake counted receipts are explicitly synthetic."""
    import mujoco
    from human_punching.phase2_native_state import capture_comparison,save_snapshot,NativeComparisonSnapshot
    spec=importlib.util.spec_from_file_location('native_fixture',Path(__file__).with_name('test_phase2_native_state.py'))
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    model=mujoco.MjModel.from_xml_path(str(audit.ROOT/'outputs/four-punch-milestone/jab/model/human.xml'))
    env=fixtures.fake_env(model,n=4,source=False);env.sim.data.time.zero_();env.sim.data.ctrl.zero_()
    env.sim.data.qfrc_applied.zero_();env.sim.data.xfrc_applied.zero_();env.sim.data.qvel.zero_()
    snapshot=capture_comparison(env,held_raw_action=np.zeros((4,48),np.float32),reference_phase='preparation',support_interval_end_s=np.zeros(4))
    path=save_snapshot(snapshot,tmp_path/'bank')
    bank=dict(path=str(path),sha256=audit.digest(path),identity=snapshot.identity)
    trial_ids=['synthetic_shared_2','synthetic_shared_0'];indices=[2,0]
    bindings=[dict(trial_id=t,start_index=i,start_state_sha256=snapshot.states[i].digest(),reference_sha256='f'*64) for t,i in zip(trial_ids,indices)]
    expected=np.stack([snapshot.states[i].native_integration_state for i in indices]);native=tmp_path/'initial.npz'
    np.savez(native,native_integration_state=expected)
    subset=NativeComparisonSnapshot(tuple(snapshot.states[i] for i in indices),snapshot.inventory_json)
    restore=dict(complete_native_integration_equal=True,physical_state_history_preserved=True,selected_reference_is_explicit=True,
        allowed_reference_metadata_changes=['reference_id','ids_ref','phase_denominator_s'],selected_reference_ids=[0,0],
        selected_reference_sha256=['f'*64]*2,source_snapshot_identity=subset.identity)
    receipt=dict(schema='task10b_shared_native_initialization_audit_v1',bank=bank,trial_ids=trial_ids,start_indices=indices,
        start_state_sha256=[b['start_state_sha256'] for b in bindings],retained_native_state=dict(path=str(native),sha256=audit.digest(native)),restore_receipt=restore)
    raw=dict(qpos=env.sim.data.qpos.numpy()[None,indices],qvel=env.sim.data.qvel.numpy()[None,indices],
        ctrl=env.sim.data.ctrl.numpy()[None,indices],qfrc_applied=env.sim.data.qfrc_applied.numpy()[None,indices],
        xfrc_applied=env.sim.data.xfrc_applied.numpy()[None,indices],time=np.zeros((1,2)),support_contact=np.ones((1,2,2),bool),
        policy_previous_raw_action=np.zeros((1,2,48)),reset_count=np.ones((1,2)),reference_tick=np.zeros((1,2),np.int64),
        policy_reference_tick=np.zeros((1,2),np.int64),policy_reference_id=np.zeros((1,2),np.int64),policy_phase_denominator_s=np.full((1,2),2.,np.float32),
        policy_qpos=env.sim.data.qpos.numpy()[None,indices],policy_qvel=env.sim.data.qvel.numpy()[None,indices],
        policy_ctrl=env.sim.data.ctrl.numpy()[None,indices],policy_qacc_warmstart=np.full((1,2,54),999.))
    def save(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return dict(path=str(p),sha256=audit.digest(p))
    capture=save('synthetic_capture.json',dict(schema='task10b_phase2_native_shared_bank_capture_v1',complete_capture=True,
        slurm_job_id='synthetic-no-job',exception=None,captured_native_states=4,physics_steps_executed=0,actor_updates_executed=0,
        no_actor_execution=True,split='validation',native_start_state_bank=bank))
    parity=save('synthetic_parity.json',dict(schema='task10b_phase2_native_state_parity_v1',passed=True,
        slurm_job_id='synthetic-no-job',family='jab',exception=None,receipt=dict(observations_equal=True,actor_outputs_equal=True,
        next_four_integration_states_equal=[True]*4)))
    return raw,receipt,dict(trials=trial_ids,reference_ids=[0,0],phase_denominators_s=[2.,2.]),dict(bindings=bindings,split='validation'),dict(
        native_start_state_bank=bank,native_start_state_capture=capture,native_state_parity=parity),model


def test_shared_initialization_independently_checks_all410_values_and_forward_warmstart(tmp_path):
    evidence=shared_initialization_fixture(tmp_path)
    result=audit.shared_initialization_audit(*evidence)
    assert result['passed'],result
    assert result['full410_initial_state_audited'] and result['no_integration_or_restore_performed']


@pytest.mark.parametrize('tamper',['native_hidden','bank_hash','bank_index','trial_digest','history','support','reset','clock','ctrl','policy_qpos','parity','capture'])
def test_shared_initialization_detects_hidden_state_and_history_tampering(tmp_path,tamper):
    evidence=list(shared_initialization_fixture(tmp_path));raw,receipt,plan,declaration,metadata,_=evidence
    if tamper=='native_hidden':
        path=Path(receipt['retained_native_state']['path']);values=audit.read_npz(path);values['native_integration_state'][0,-1]=.25
        np.savez(path,**values);receipt['retained_native_state']['sha256']=audit.digest(path)
    elif tamper=='bank_hash':metadata['native_start_state_bank']['sha256']='a'*64
    elif tamper=='bank_index':declaration['bindings'][0]['start_index']=1
    elif tamper=='trial_digest':declaration['bindings'][0]['start_state_sha256']='a'*64
    elif tamper=='history':raw['policy_previous_raw_action'][0,0,0]=.01
    elif tamper=='support':raw['support_contact'][0,0,0]=False
    elif tamper=='reset':raw['reset_count'][0,0]=2
    elif tamper=='clock':raw['policy_reference_tick'][0,0]=1
    elif tamper=='ctrl':raw['ctrl'][0,0,0]=.01
    elif tamper=='policy_qpos':raw['policy_qpos']=raw['policy_qpos'].copy();raw['policy_qpos'][0,0,8]+=.01
    elif tamper in ('parity','capture'):
        record=metadata['native_state_'+('parity' if tamper=='parity' else 'capture') if tamper=='parity' else 'native_start_state_capture']
        p=Path(record['path']);value=json.loads(p.read_text());value['slurm_job_id']=None;p.write_text(json.dumps(value));record['sha256']=audit.digest(p)
    assert not audit.shared_initialization_audit(*evidence)['passed']


def offline_fixture(tmp_path):
    target=[1.,0.,1.2];ref=tmp_path/'ref.npz'
    np.savez(ref,t=np.array([0.,.5,1.]),glove_pos=np.full((3,2,3),4.))
    screen=tmp_path/'screen.json';screen.write_text(json.dumps(dict(frames=3,completed_chunks=1)))
    binding=dict(trial_id='synthetic_rejected',command_id=31,original_replica_id=1,ttc_s=.6,start_index=0,
        start_state_sha256='a'*64,reference_sha256=audit.digest(ref),target_w=target)
    row=dict(candidate_id='synthetic_rejected_ref',family='jab',requested_ttc_s=.6,command_id=31,original_replica_id=1,
        reference_sha256=audit.digest(ref),reference_path=str(ref),legacy_runtime_path=str(ref),legacy_runtime_sha256=audit.digest(ref),
        target_w=target,kinematic_admission=True,rejection_reasons=[],diagnostic_screen_path=str(screen),recipe_sha256='b'*64,
        launch_tick=0,outward_end_tick=160,active_hand_index=0)
    mask=dict(schema='task10b_phase2_declared_evaluation_mask_v1',requests=[dict(
        family='jab',**{k:v for k,v in binding.items() if k!='trial_id'},admitted=False,reason='outside frozen target coverage')])
    return binding,row,mask,'c'*64,'d'*64,None


def test_offline_rejection_is_exact_mask_and_real_hashed_screen_evidence(tmp_path):
    evidence=offline_fixture(tmp_path);receipt=audit.offline_rejection_audit(*evidence)
    assert receipt['passed'] and receipt['kind']=='screen_or_mask_rejection'
    assert receipt['no_physical_attempt'] and receipt['cycle_success_credit']==0
    assert receipt['evidence']['diagnostic_screen']['sha256']==audit.digest(evidence[1]['diagnostic_screen_path'])


@pytest.mark.parametrize('tamper',['different_target','mask_binding','mask_admitted','reference_hash','missing_screen','unknown_kinematic_reason'])
def test_offline_rejection_detects_request_mask_reference_and_screen_mismatches(tmp_path,tamper):
    evidence=list(offline_fixture(tmp_path));binding,row,mask,*_=evidence
    if tamper=='different_target':binding['target_w']=[1.001,0.,1.2]
    if tamper=='mask_binding':mask['requests'][0]['start_state_sha256']='e'*64
    if tamper=='mask_admitted':mask['requests'][0]['admitted']=True
    if tamper=='reference_hash':Path(row['reference_path']).write_bytes(b'changed reference')
    if tamper=='missing_screen':row['diagnostic_screen_path']=str(tmp_path/'absent.json')
    if tamper=='unknown_kinematic_reason':row['kinematic_admission']=False;row['rejection_reasons']=['unsupported invented rule']
    assert not audit.offline_rejection_audit(*evidence)['passed']


def test_generation_failure_remains_distinct_from_a_generated_rejected_reference(tmp_path):
    evidence=list(offline_fixture(tmp_path));binding,row,*_=evidence
    for key in ('reference_path','legacy_runtime_path','legacy_runtime_sha256'):row.pop(key)
    row.update(reference_identity_kind='generation_failure_receipt',kinematic_admission=False,rejection_reasons=['bounded_timeout'])
    receipt=tmp_path/'generation_failure.json';receipt.write_text(json.dumps(row));sha=audit.digest(receipt)
    row.update(generation_failure_path=str(receipt),generation_failure_sha256=sha,reference_sha256=sha);binding['reference_sha256']=sha
    result=audit.offline_rejection_audit(*evidence)
    assert result['passed'] and result['kind']=='generation_failure'
    row['reference_path']=str(tmp_path/'invented_reference.npz')
    assert not audit.offline_rejection_audit(*evidence)['passed']


def packed_failure_fixture(tmp_path):
    """Pack a full generator-shaped negative receipt, without native execution."""
    from human_punching import phase2_runtime_pack as packing
    def save(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return dict(path=str(path),sha256=audit.digest(path))
    recipe=save('source_recipe.json',dict(schema='task10b_phase2_shared_recipe_v2',family='jab',diagnostic_only=True))
    failure=dict(candidate_id='synthetic_failed_generation',family='jab',split='validation',command_id=31,
        original_replica_id=1,requested_ttc_s=.6,target_w=[1.,0.,1.2],head_center=[1.,0.,1.2],domain_stage='D1',
        kind='bounded_active_arm_target_generation_failure',start_mode='family_guard',preparation_complete_tick=0,
        launch_tick=4,outward_end_tick=160,launch_time_s=.02,outward_completion_time_s=.8,
        duration_s=1.,episode_duration_s=2.,phase_denominator_s=2.,
        negative_clock_semantics='Planned clocks declare an unexecuted negative cell; no contact claim',
        recipe_sha256=recipe['sha256'],kinematic_admission=False,rejection_reasons=['bounded_timeout'],
        generation_disposition='timeout',physical_qualification=False,retained_partial_reference_path=None,
        error=dict(type='BoundedExecutionInterruption',message='synthetic finite screening timeout'),
        details=dict(checkpoint='target_screen_chunk',completed_frames=16,simulated_time_advanced_s=0.))
    source=save('original_generation_failure.json',failure)
    row=dict(**failure,reference_identity_kind='generation_failure_receipt',generation_failure_path=source['path'],
        generation_failure_sha256=source['sha256'],reference_sha256=source['sha256'])
    catalog=save('source_catalog.json',dict(schema='task10b_phase2_reference_catalog_v1',complete=True,
        recipe_sha256=recipe['sha256'],cells=[row]))
    plan=save('plan.json',dict(schema='task10b_phase2_runtime_pack_plan_v1',storage_version=packing.VERSION,
        byte_cap=2**20,catalogs=[catalog],recipes=[recipe],previous_packed_catalogs=[]))
    result=packing.pack(plan['path'],tmp_path/'packed',wall_seconds=30.)
    assert result['status']=='packed',result
    packed_catalog=json.loads(Path(result['catalogs'][0]['path']).read_text());packed=packed_catalog['cells'][0]
    binding=dict(trial_id='synthetic_failed_request',command_id=31,original_replica_id=1,ttc_s=.6,start_index=0,
        start_state_sha256='a'*64,reference_sha256=packed['reference_sha256'],target_w=failure['target_w'])
    return [binding,packed,None,'c'*64,result['catalogs'][0]['sha256'],None],source,recipe


def test_packed_full_generation_failure_resolves_through_exact_original_ancestry(tmp_path):
    evidence,source,recipe=packed_failure_fixture(tmp_path)
    before=Path(source['path']).read_bytes();old_recipe=Path(recipe['path']).read_bytes()
    result=audit.offline_rejection_audit(*evidence)
    assert result['passed'] and result['kind']=='generation_failure',result
    assert all(result['checks'][name] for name in ('packed_failure_owned_original','packed_failure_exact_inventory',
        'packed_failure_original_fields','packed_failure_recipe_ancestry'))
    assert result['evidence']['unpacked_generation_failure']==source
    assert evidence[1]['recipe_sha256']!=recipe['sha256']
    assert Path(source['path']).read_bytes()==before and Path(recipe['path']).read_bytes()==old_recipe


@pytest.mark.parametrize('tamper',['wrong_old_recipe','changed_wrapper_body','changed_original_body','missing_wrapper_version',
    'unsupported_wrapper_version','extra_wrapper_field','changed_row_body','wrong_original_hash','recursive_wrapper','native_file_claim'])
def test_packed_generation_failure_rejects_consistent_hashes_with_invalid_ancestry(tmp_path,tamper):
    evidence,source,_=packed_failure_fixture(tmp_path);binding,row,*_=evidence
    path=Path(row['generation_failure_path']);wrapper=json.loads(path.read_text());original=Path(source['path'])
    if tamper=='wrong_old_recipe':wrapper['unpacked_recipe_sha256']='e'*64;row['unpacked_recipe_sha256']='e'*64
    elif tamper=='changed_wrapper_body':wrapper['error']['message']='different failure';row['error']=wrapper['error']
    elif tamper=='changed_original_body':
        value=json.loads(original.read_text());value['details']['completed_frames']=32;original.write_text(json.dumps(value))
        sha=audit.digest(original);wrapper['unpacked_generation_failure_sha256']=sha;row['unpacked_generation_failure_sha256']=sha
    elif tamper=='missing_wrapper_version':wrapper.pop('runtime_failure_wrapper_version');row.pop('runtime_failure_wrapper_version')
    elif tamper=='unsupported_wrapper_version':wrapper['runtime_failure_wrapper_version']='unreviewed';row['runtime_failure_wrapper_version']='unreviewed'
    elif tamper=='extra_wrapper_field':wrapper['invented_metadata']='unreviewed'
    elif tamper=='changed_row_body':row['details']={'completed_frames':999}
    elif tamper=='wrong_original_hash':wrapper['unpacked_generation_failure_sha256']='f'*64;row['unpacked_generation_failure_sha256']='f'*64
    elif tamper=='recursive_wrapper':
        value=json.loads(original.read_text());value['runtime_failure_wrapper_version']='task10b_phase2_packed_generation_failure_v1'
        original.write_text(json.dumps(value));sha=audit.digest(original)
        wrapper['unpacked_generation_failure_sha256']=sha;row['unpacked_generation_failure_sha256']=sha
    elif tamper=='native_file_claim':
        value=json.loads(original.read_text());value['legacy_runtime_sha256']='d'*64
        original.write_text(json.dumps(value));sha=audit.digest(original)
        wrapper.update(unpacked_generation_failure_sha256=sha,legacy_runtime_sha256='d'*64)
        row.update(unpacked_generation_failure_sha256=sha,legacy_runtime_sha256='d'*64)
    path.write_text(json.dumps(wrapper));sha=audit.digest(path)
    row.update(generation_failure_sha256=sha,reference_sha256=sha);binding['reference_sha256']=sha
    assert not audit.offline_rejection_audit(*evidence)['passed']


def offline_run_fixture(tmp_path):
    """Complete one-request synthetic orchestration, never a96-panel claim."""
    from human_punching.phase2_learning import family_arguments
    folder=tmp_path/'run';folder.mkdir()
    binding,row,mask,*_=offline_fixture(folder)
    def save(name,value):
        path=folder/name;path.write_text(json.dumps(value));return dict(path=str(path),sha256=audit.digest(path))
    mask_rec=save('mask.json',mask);predicate=save('predicate.json',dict(schema='syntheticpredicate'))
    declaration=dict(bindings=[binding],split='validation',start_mode='family_guard',checkpoint_sha256='d'*64,
        declared_mask_sha256=mask_rec['sha256'],predicate_sha256=predicate['sha256'])
    decl=save('declaration.json',declaration);domain=save('domain.json',{});targets=save('targets.json',{})
    catalog=save('catalog.json',dict(cells=[row]));manifest=save('manifest.json',dict(trials=[dict(trial_id=binding['trial_id'],candidate_id=row['candidate_id'])]))
    frozen=dict(input_sha256=dict(mask=mask_rec['sha256'],manifest=manifest['sha256'],predicate=predicate['sha256']))
    identity=dict(schema='task10b_phase2_frozen_execution_identity_v1',identity=audit.json_identity(frozen),frozen=frozen,
        input_paths=dict(mask=mask_rec['path'],manifest=manifest['path'],predicate=predicate['path']))
    save('execution_identity.json',identity)
    _,release=family_arguments(audit.ROOT/'outputs/four-punch-milestone','jab')
    fake_checkpoint=folder/'synthetic_not_a_policy.pt';fake_checkpoint.write_bytes(b'noactorortrainingexecution')
    declaration['checkpoint_sha256']=audit.digest(fake_checkpoint);decl=save('declaration.json',declaration)
    save('launch.json',dict(schema='task10b_phase2_expanded_native_launch_v1',catalog=catalog['path'],catalog_sha256=catalog['sha256'],
        declaration_path=decl['path'],declaration_sha256=decl['sha256'],declaration_identity=audit.json_identity(declaration),
        domain_path=domain['path'],domain_sha256=domain['sha256'],target_geometry_path=targets['path'],target_geometry_sha256=targets['sha256'],
        evaluated_checkpoint=dict(path=str(fake_checkpoint),sha256=audit.digest(fake_checkpoint)),execution_identity=identity['identity'],
        release_inputs={key:dict(path=str(path),sha256=audit.digest(path)) for key,path in release.items()}))
    save('outcomes.json',[dict(trial_id=binding['trial_id'],status='rejected',physical_execution_attempted=False)])
    return folder,binding,identity


def test_wholly_offline_run_gets_resolved_rejection_without_claiming_native_launch(tmp_path):
    folder,binding,_=offline_run_fixture(tmp_path)
    result=audit.audit_run(folder,tmp_path/'audit')
    assert result['passed'] and result['outcome_resolution_passed']
    assert not result['native_runtime_model_present'] and not result['closed_protocol_launch_audited']
    assert result['panels']==[] and result['cpu_actor_predictions']==0
    row=result['trial_outcome_resolutions'][0]
    assert row['request_identity']==audit.json_identity(binding)
    assert row['outcome_resolution']=='verified_offline_rejection' and not row['attempted']
    assert row['raw_trace_sha256'] is None and row['offline_rejection_kind']=='screen_or_mask_rejection'


def test_packed_failure_wrapper_receives_only_verified_offline_resolution(tmp_path):
    packed=tmp_path/'packing';packed.mkdir();evidence,_,_=packed_failure_fixture(packed)
    binding,row,*_=evidence
    panel=tmp_path/'panel';panel.mkdir();folder,_,_=offline_run_fixture(panel)
    def save(name,value):
        path=folder/name;path.write_text(json.dumps(value));return audit.digest(path)
    declaration=json.loads((folder/'declaration.json').read_text());declaration['bindings']=[binding]
    mask=dict(schema='task10b_phase2_declared_evaluation_mask_v1',requests=[dict(family=row['family'],
        **{k:v for k,v in binding.items() if k!='trial_id'},admitted=False,reason='preserved generation failure')])
    declaration['declared_mask_sha256']=save('mask.json',mask)
    decl_sha=save('declaration.json',declaration)
    manifest_sha=save('manifest.json',dict(trials=[dict(trial_id=binding['trial_id'],candidate_id=row['candidate_id'])]))
    catalog_sha=save('catalog.json',dict(cells=[row]))
    identity=json.loads((folder/'execution_identity.json').read_text())
    identity['frozen']['input_sha256'].update(mask=declaration['declared_mask_sha256'],manifest=manifest_sha)
    identity['identity']=audit.json_identity(identity['frozen']);save('execution_identity.json',identity)
    launch=json.loads((folder/'launch.json').read_text());launch.update(catalog_sha256=catalog_sha,
        declaration_sha256=decl_sha,declaration_identity=audit.json_identity(declaration),execution_identity=identity['identity'])
    save('launch.json',launch)
    save('outcomes.json',[dict(trial_id=binding['trial_id'],status='generation_failure',physical_execution_attempted=False)])
    result=audit.audit_run(folder,panel/'audit')
    assert result['passed'] and result['outcome_resolution_passed']
    resolution=result['trial_outcome_resolutions'][0]
    assert resolution['outcome_resolution']=='verified_offline_rejection' and resolution['generation_failure_verified']
    assert resolution['generation_failure_sha256']==binding['reference_sha256']==row['generation_failure_sha256']
    assert not resolution['attempted'] and not resolution['execution_complete'] and resolution['raw_trace_sha256'] is None
    assert not result['native_runtime_model_present'] and not result['closed_protocol_launch_audited']


def test_durable_attempt_without_trace_cannot_be_reclassified_as_offline_rejection(tmp_path):
    folder,binding,identity=offline_run_fixture(tmp_path)
    (folder/'native_attempt_intent_0000.json').write_text(json.dumps(dict(schema='task10b_phase2_native_attempt_intent_v1',
        execution_identity=identity['identity'],trial_ids=[binding['trial_id']],request_identities=[audit.json_identity(binding)])))
    result=audit.audit_run(folder,tmp_path/'audit')
    assert not result['passed'] and not result['outcome_resolution_passed']
    resolution=result['trial_outcome_resolutions'][0]
    assert resolution['attempted'] and resolution['outcome_resolution']=='unknown_or_missing'
    assert not result['offline_rejection_receipts']


def test_duplicate_durable_attempt_intents_are_rejected_before_resolution(tmp_path):
    folder,binding,identity=offline_run_fixture(tmp_path)
    intent=dict(schema='task10b_phase2_native_attempt_intent_v1',execution_identity=identity['identity'],
        trial_ids=[binding['trial_id']],request_identities=[audit.json_identity(binding)])
    for index in range(2):(folder/f'native_attempt_intent_{index:04d}.json').write_text(json.dumps(intent))
    with pytest.raises(ValueError,match='durable native attempt'):audit.audit_run(folder,tmp_path/'audit')


@pytest.mark.parametrize('complete',[False,True])
def test_native_resolution_keeps_failed_prefix_separate_from_complete_cycle(complete):
    trial=dict(trial_id='synthetic',execution_complete=complete,hard_flags_absent=True)
    binding=dict(trial_id='synthetic',start_state_sha256='a'*64)
    resolution=audit.native_resolution(binding,trial,'b'*64,integrity=True,exact_counters=True,safe_boundary=True)
    assert resolution['resolved'] and resolution['attempted']
    assert resolution['outcome_resolution']==('audited_complete' if complete else 'audited_failed_prefix')
    assert resolution['execution_complete'] is complete


@pytest.mark.parametrize('fault',['integrity','counter','boundary','hard_fault'])
def test_native_resolution_never_resolves_unknown_extent_or_observed_hard_fault(fault):
    trial=dict(trial_id='synthetic',execution_complete=False,hard_flags_absent=fault!='hard_fault')
    resolution=audit.native_resolution(dict(trial_id='synthetic'),trial,'b'*64,
        integrity=fault!='integrity',exact_counters=fault!='counter',safe_boundary=fault!='boundary')
    assert not resolution['resolved'] and resolution['outcome_resolution']=='unknown_or_missing'
    assert resolution['attempted'] and not resolution['execution_complete']
