"""CPU bridge tests: tensor endpoints and real local config APIs; no mj_step."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from task10_telemetry_helpers import state
from conftest import task10_common_bundle
from human_punching.wbc.telemetry.readiness_clock import Phase, Failure, ReadinessConfig, PlanStatus

from human_punching.wbc.telemetry import cycle_runtime as m


def setup(n=2,context_width=7):
    s = state(n, dtype=torch.float32)
    limits, caps = torch.tensor([[-1., 1.]]*3), torch.full((3,), 10.)
    packet = {"target_w": torch.tensor([[0., 0., 1.]]*n),
        "g1_launch_state": torch.arange(n*context_width, dtype=torch.float32).reshape(n, context_width),
        "requested_ttc_s": torch.full((n,), .1), "outward_end_s": torch.full((n,), .2),
        "family_id": torch.arange(n)%4, "plan_handle": torch.arange(n),
        "plan_digest": torch.ones((n, 32), dtype=torch.uint8),
        "status": torch.full((n,), PlanStatus.ADMITTED, dtype=torch.int64),
        "plan_matches_guard_context": torch.ones(n, dtype=torch.bool),
        "actor_g1_context": torch.arange(n*77, dtype=torch.float32).reshape(n, 77)}
    receipts = []
    b = m.CycleRuntime(n, limits, caps, context_width=context_width, packet_provider=lambda: packet,
        nominal_provider=lambda: s, request_ids=lambda: torch.arange(n),
        reference_gloves=lambda: s["glove_position_w"].clone()+2., receipt_sink=receipts.append,
        config=ReadinessConfig(quiet_ticks=2, maximum_settling_ticks=12))
    b.command_boundary()  # initial construction cannot create a launch
    b.after_pending_reset(torch.arange(n), s, torch.zeros(n, dtype=torch.int64), torch.full((n,), 100., dtype=torch.float64))
    return b, s, packet, receipts


def window(b, s, *, boundary=True):
    b.begin_window()
    for _ in range(4):
        b.after_substep(b.readiness.tick+1, s)
    if boundary:
        b.command_boundary()


def test_independent_launch_clock_precedes_first_strike_action():
    b,s,p,receipts = setup()
    bad = {k:v.clone() for k,v in s.items()}; bad["joint_velocity"][1,0] = 1.
    window(b,bad)
    assert b.readiness.launch_tick.tolist() == [4,-1]
    assert b.readiness.reference_tick.tolist() == [0,0]
    assert b.events.armed.tolist() == [True,False]
    window(b,s)
    assert b.readiness.launch_tick.tolist() == [4,8]
    assert b.readiness.reference_tick.tolist() == [4,0]
    assert b.readiness.launch_count.tolist() == [1,1]
    assert not receipts


def test_exact_first_contact_time_survives_20ms_endpoint_and_repeated_reward_reads():
    b,s,p,_ = setup(); window(b,s)
    b.begin_window()
    for k in range(4):
        hit = {key:value.clone() for key,value in s.items()}
        if k == 0:
            hit["glove_position_w"][0,0] = torch.tensor([.13,0.,1.])
            hit["glove_position_w"][1,1] = torch.tensor([.13,0.,1.])
        b.after_substep(b.readiness.tick+1,hit)
    a,c = b.reward_context(),b.reward_context()
    assert a["new_contact"].all() and c["new_contact"].all()
    assert torch.allclose(a["contact_error_s"],torch.full((2,),-.095))
    assert torch.allclose(a["elapsed_s"],torch.full((2,),.020))
    assert torch.equal(a["reference_glove_w"],s["glove_position_w"]+2)
    b.command_boundary(); b.begin_window()
    assert not b.events.window_new_contact.any()


def test_target_snapshot_plan_and_actor_context_cannot_chase_postlaunch_updates():
    b,s,p,_=setup(); window(b,s)
    frozen={k:v.clone() for k,v in b.readiness.committed.items()}
    actor=b.actor_context.clone()
    p["target_w"].fill_(50.);p["g1_launch_state"].fill_(60.);p["actor_g1_context"].fill_(70.)
    p["plan_handle"] += 100;p["plan_digest"].fill_(8)
    window(b,s)
    assert all(torch.equal(v,b.readiness.committed[k]) for k,v in frozen.items())
    assert torch.equal(actor,b.actor_context)
    assert torch.equal(b.reward_context()["target_w"],frozen["target_w"])


def test_reference_lookahead_and_derivatives_remain_static_before_launch():
    tick=torch.tensor([100,3]);launched=torch.tensor([False,True]);clip=torch.tensor([0,1])
    starts=torch.tensor([0,20]);lengths=torch.tensor([20,10])
    index,stationary=m.reference_indices(tick,launched,clip,starts,lengths,12)
    assert index.tolist()==[0,29] and stationary.tolist()==[True,True]
    index,_=m.reference_indices(tick,launched,clip,starts,lengths)
    assert index.tolist()==[0,23]


def test_nominal_posture_and_actual_ready_footprint_have_distinct_sources():
    b,s,p,_=setup()
    ready={k:v.clone() for k,v in s.items()};ready["root_position_w"][:,0]+=.01
    ready["support_anchor_position_w"][:,:,0]+=.02
    window(b,ready)
    ctx=b.reward_context()
    assert torch.equal(ctx["goal"]["root_position_w"],s["root_position_w"])
    assert torch.equal(ctx["goal"]["support_anchor_position_w"],ready["support_anchor_position_w"])
    assert torch.equal(b.events.goal["root_position_w"],s["root_position_w"])


def test_no_support_initial_gap_does_not_launch_or_disappear():
    b,s,p,receipts=setup(); unsupported={k:v.clone() for k,v in s.items()}
    unsupported["support_contact"][:]=False
    unsupported["support_patch_contact"][:]=False
    unsupported["support_anchor_position_w"][:,:,2]=.0004
    for _ in range(3):window(b,unsupported)
    assert b.readiness.failure.tolist()==[Failure.GUARD_TIMEOUT]*2
    assert not b.events.armed.any() and b.events.first_contact.tolist()==[-1,-1]
    assert len(receipts)==1 and receipts[0]["environment_ids"].tolist()==[0,1]
    assert not receipts[0]["events"]["measurement_valid"].any()
    assert receipts[0]["events"]["diagnostic_measurement_valid"].all()
    assert torch.isnan(receipts[0]["readiness"]["launch_time_s"]).all()


def test_early_native_termination_retains_unlaunched_attempt_then_partial_reset():
    b,s,p,receipts=setup(); window(b,s,boundary=False)
    b.record_reset(torch.tensor([1]),torch.tensor([False,True]),torch.zeros(2,dtype=torch.bool))
    assert len(receipts)==1
    assert receipts[0]["readiness"]["launch_tick"].tolist()==[-1]
    b.command_boundary()
    assert b.events.armed.tolist()==[True,False]
    b.after_pending_reset(torch.tensor([1]),s,torch.tensor([4,0]),torch.tensor([100.02,101.],dtype=torch.float64))
    window(b,s)
    assert b.readiness.launch_tick.tolist()==[4,4]
    assert b.readiness.attempt_count.tolist()==[1,2]
    assert b.readiness.launch_count.tolist()==[1,1]
    assert b.readiness.reference_tick.tolist()==[4,0]


def test_terminal_rows_stop_accumulating_before_remaining_policy_substeps():
    b,s,p,_=setup();window(b,s);b.begin_window()
    fault={k:v.clone() for k,v in s.items()};fault["joint_position"][0,0]=1.03
    b.after_substep(b.readiness.tick+1,fault)
    assert b.readiness.phase[0]==Phase.FAILED
    for _ in range(3):
        hit={k:v.clone() for k,v in s.items()};hit["glove_position_w"][0,0]=torch.tensor([.13,0.,1.])
        b.after_substep(b.readiness.tick+1,hit)
    assert b.events.first_contact[0]==-1 and b.events.tick[0]==1
    assert b.events.first_failure_tick[0]==1
    assert b.events.tick[1]==4


@pytest.mark.parametrize("field",["root_actuation","clock_failure"])
def test_extra_native_faults_cannot_bypass_launch_reducer(field):
    b,s,p,_=setup();fault={k:v.clone() for k,v in s.items()};fault[field]=torch.ones(2,dtype=torch.bool)
    window(b,fault)
    assert (b.readiness.phase==Phase.FAILED).all() and not b.events.armed.any()


@pytest.mark.parametrize("field",["root_actuation","clock_failure"])
@pytest.mark.parametrize("fault",[False,True])
def test_optional_endpoint_flags_do_not_change_fresh_partial_reset_schema(field,fault):
    b,s,p,receipts=setup()
    endpoint={key:value.clone() for key,value in s.items()}
    endpoint[field]=torch.tensor([fault,False])
    window(b,endpoint,boundary=False)
    assert field in endpoint and field not in b.readiness.last_state
    assert not s["external_support"].any() and not s["numerical_failure"].any()
    b.record_reset(torch.tensor([0]),torch.tensor([True,False]),torch.zeros(2,dtype=torch.bool))
    b.command_boundary()
    unreset={key:value[1].clone() for key,value in b.readiness.last_state.items()}
    b.after_pending_reset(torch.tensor([0]),s,torch.tensor([0,4]),torch.tensor([101.,100.02],dtype=torch.float64))
    assert b.readiness.phase.tolist()==[Phase.SETTLING,Phase.STRIKE]
    assert b.readiness.launch_tick.tolist()==[-1,4]
    assert b.readiness.attempt_count.tolist()==[2,1]
    assert len(receipts)==1 and receipts[0]["environment_ids"].tolist()==[0]
    expected=Failure.EXTERNAL_SUPPORT if field=="root_actuation" else Failure.INVALID_TELEMETRY
    assert receipts[0]["readiness"]["failure_category"].item()==(expected if fault else Failure.ENVIRONMENT_TERMINATION)
    assert all(torch.equal(value,b.readiness.last_state[key][1]) for key,value in unreset.items())
    window(b,s)
    assert b.readiness.launch_tick.tolist()==[4,4]
    assert b.readiness.launch_count.tolist()==[1,1]


def test_receipt_sink_failure_prevents_loss_of_attempt():
    b,s,p,_=setup();window(b,s,boundary=False)
    def fail(_):raise OSError("receipt disk fault")
    b.receipt_sink=fail
    with pytest.raises(OSError,match="receipt disk"):
        b.record_reset(torch.tensor([0]),torch.tensor([True,False]),torch.zeros(2,dtype=torch.bool))
    assert not b.persisted.any()
    with pytest.raises(RuntimeError,match="may not disappear"):
        b.after_pending_reset(torch.tensor([0]),s,torch.zeros(2,dtype=torch.int64),torch.zeros(2,dtype=torch.float64))


def test_midwindow_boundary_and_unexplained_reset_rejected():
    b,s,p,_=setup();b.begin_window();b.after_substep(b.readiness.tick+1,s)
    with pytest.raises(RuntimeError,match="all four"):
        b.command_boundary()
    for _ in range(3):b.after_substep(b.readiness.tick+1,s)
    with pytest.raises(RuntimeError,match="explicit terminal"):
        b.record_reset(torch.tensor([0]),torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))


def test_native_command_samples_reference_start_before_integration_and_fresh_endpoint_after():
    b,s,p,_=setup(); cmd=m.CycleMotionCommand.__new__(m.CycleMotionCommand)
    cmd._env=SimpleNamespace(**{m.ATTRIBUTE:b},scene=SimpleNamespace(env_origins=torch.zeros(2,3)))
    cmd.physics_steps=torch.tensor([99,99]);cmd.trajectory_ids=torch.tensor([0,1])
    cmd.master_starts=torch.tensor([0,30]);cmd.master_lengths=torch.tensor([30,30])
    cmd.master={"qpos":torch.arange(60).float()[:,None].repeat(1,10),"qvel":torch.ones(60,9),
                "joint_feedforward":torch.ones(60,3)}
    cmd.settling_feedforward=torch.tensor([.1,.2,.3])
    assert cmd.sample_master("qpos",24)[:,0].tolist()==[0.,30.]
    assert not cmd.sample_master("qvel").any()
    assert torch.equal(cmd.sample_master("joint_feedforward"),cmd.settling_feedforward.expand(2,3))
    window(b,s)
    assert cmd.sample_master("qpos")[:,0].tolist()==[0.,30.]
    b.begin_window();b.after_substep(b.readiness.tick+1,s)
    assert cmd.sample_master("qpos")[:,0].tolist()==[1.,31.]
    assert cmd.elapsed.tolist()==pytest.approx([.005,.005])


def test_phase_condition_distinguishes_settling_from_zero_elapsed_launch():
    b,s,p,_=setup();cmd=SimpleNamespace(robot_anchor_quat_w=s["root_quat_wxyz"],robot_anchor_pos_w=s["root_position_w"])
    env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:cmd),**{m.ATTRIBUTE:b})
    before=m.cycle_condition(env);window(b,s);after=m.cycle_condition(env)
    assert before.shape==after.shape==(2,91)
    assert torch.equal(before[:,:87],after[:,:87])
    assert before[:,-4:].tolist()==[[1.,0.,0.,0.]]*2
    assert after[:,-4:].tolist()==[[0.,1.,0.,0.]]*2


def test_json_receipt_preserves_missing_launch_as_null_and_refuses_overwrite():
    import tempfile
    root=Path.cwd()/".cache/task10/substep_draft"
    with tempfile.TemporaryDirectory(dir=root) as directory:
        sink=m.JsonReceiptSink(directory)
        sink({"sequence":0,"time":torch.tensor([float("nan"),1.])})
        assert '"time": [null, 1.0]' in (Path(directory)/"attempt_00000000.json").read_text()
        with pytest.raises(FileExistsError):sink({"sequence":0})


def test_config_extends_deadline_and_changes_observation_schema_explicitly():
    from human_punching.wbc.runtime.command import PunchMotionCommandCfg
    from human_punching.wbc.runtime import mdp
    from mjlab.managers.observation_manager import ObservationTermCfg
    cfg=SimpleNamespace(commands={"motion":PunchMotionCommandCfg(bundle_path="fixture",motion_file="fixture",entity_name="robot",anchor_body_name="Hips",body_names=("Hips",),resampling_time_range=(1e9,1e9))},
        actions={"joint_pos":m.ObserverActionCfg(entity_name="robot",actuator_names=("joint",),command_name="motion")},
        observations={key:SimpleNamespace(terms={"punch_condition":ObservationTermCfg(func=mdp.punch_condition)}) for key in ("actor","critic")},
        terminations={},rewards={"timed_target":object()},episode_length_s=10.)
    result=m.configure_cycle_command(cfg,maximum_ttc_s=1.35)
    assert cfg.episode_length_s>=11.37
    assert cfg.commands["motion"].require_physical_qualification
    assert cfg.observations["actor"].terms["punch_condition"].func is m.cycle_condition
    assert cfg.terminations["cycle_complete"].func is m.cycle_terminal
    assert not cfg.rewards and result["policy_command_dimension"]==91


def test_observation_manager_bootstrap_has_explicit_settling_phase_without_bridge():
    s=state(dtype=torch.float32)
    cmd=SimpleNamespace(trajectory_ids=torch.tensor([0,1]),family_ids=torch.tensor([0,1]),
        ttcs=torch.tensor([.7,.7]),g1_context=torch.ones(2,77),target_w=torch.ones(2,3),
        robot_anchor_quat_w=s["root_quat_wxyz"],robot_anchor_pos_w=s["root_position_w"])
    env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _:cmd))
    assert m.cycle_condition(env)[:,-4:].tolist()==[[1.,0.,0.,0.]]*2
    env.observation_manager=object()
    with pytest.raises(RuntimeError,match="before reset"):
        m.cycle_condition(env)


def test_missing_bridge_is_rejected_before_any_action_hook():
    action=m.CycleObserverAction.__new__(m.CycleObserverAction)
    action._env=SimpleNamespace()
    with pytest.raises(RuntimeError,match="not explicitly installed"):
        action.process_actions(torch.zeros(2,3))


def fake_bound_command():
    snapshot={"g1_qpos":[0.]*36,"g1_qvel":[0.]*35,"head_position_w":[1.,2.,3.],
        "head_velocity_w":[.1,.2,.3],"human_origin_w":[4.,5.,6.],"human_quat_wxyz":[1.,0.,0.,0.]}
    return SimpleNamespace(num_envs=2,device="cpu",clips=[{"master_sha256":"01"*32,
        "outward_end_s":.9,"launch_snapshot":snapshot}],
        trajectory_ids=torch.zeros(2,dtype=torch.int64),
        targets=torch.tensor([[1.,2.,3.]]),ttcs=torch.tensor([.7]),family_ids=torch.tensor([0]),
        g1_context=torch.ones(1,77),_env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.tensor([[0.,0.,0.],[10.,20.,30.]]))))


def test_bound_plan_translates_complete_snapshot_points_but_not_velocities():
    command=fake_bound_command()
    packet={}
    provider=m.bind_packet_to_command(command,lambda:packet)
    packet.update(provider.expected())
    result=provider();full=result["g1_launch_state"]
    assert full.shape==(2,84)
    assert full[1,:3].tolist()==[10.,20.,30.]
    assert full[1,71:74].tolist()==[11.,22.,33.]
    assert full[1,77:80].tolist()==[14.,25.,36.]
    assert torch.equal(full[0,74:77],full[1,74:77])
    assert result["target_w"][1].tolist()==[11.,22.,33.]


@pytest.mark.parametrize("field",["target_w","g1_launch_state","requested_ttc_s","outward_end_s",
    "family_id","plan_handle","plan_digest","actor_g1_context"])
def test_changed_plan_contract_rejected_before_commitment(field):
    command=fake_bound_command();packet={}
    provider=m.bind_packet_to_command(command,lambda:packet);packet.update(provider.expected())
    packet[field]=packet[field].clone();packet[field].flatten()[0]+=1
    with pytest.raises(ValueError,match="immutable reference"):
        provider()


def test_success_event_survives_terminal_phase_and_receipt_until_next_policy_interval():
    b,s,p,receipts=setup();window(b,s)
    b.begin_window()
    for k in range(4):
        hit={key:value.clone() for key,value in s.items()}
        if k==0:hit["glove_position_w"][0,0]=torch.tensor([.13,0.,1.]);hit["glove_position_w"][1,1]=torch.tensor([.13,0.,1.])
        b.after_substep(b.readiness.tick+1,hit)
    b.command_boundary()
    for _ in range(60):
        window(b,s)
        if (b.readiness.phase==Phase.COMPLETE).all():break
    assert (b.readiness.phase==Phase.COMPLETE).all()
    context=b.reward_context()
    assert context["new_recovery"].all() and not context["guard_mask"].any()
    assert len(receipts)==1 and receipts[0]["events"]["recovery_tick"].min()>0
    assert torch.equal(context["new_recovery"],b.reward_context()["new_recovery"])


def test_real_wbc_loader_command_compute_and_observation_lifecycle(task10_common_bundle,monkeypatch):
    """Real pinned managers/loader; injected endpoint tensors, no dynamics claim."""
    import mujoco
    from mjlab.managers.observation_manager import ObservationManager, ObservationGroupCfg, ObservationTermCfg
    from human_punching.wbc.runtime.task import build_human_env_cfg
    from human_punching.wbc.telemetry.runtime_telemetry import configure_observer, ObserverOptions, ATTRIBUTE as OBSERVER_ATTRIBUTE
    monkeypatch.setattr(mujoco,"mj_step",lambda *a:pytest.fail("No CPU physics"))
    bundle=task10_common_bundle;n=2;writes=[]
    cfg=build_human_env_cfg(bundle,num_envs=n,evaluation=True)
    configure_observer(cfg,ObserverOptions(enabled=True))
    m.configure_cycle_command(cfg,maximum_ttc_s=1.1,require_physical_qualification=False,task_objectives=True)
    assert "human_guard_quiet" in cfg.rewards and cfg.rewards["foot_slip"].weight == 0.
    data=SimpleNamespace(soft_joint_pos_limits=torch.tensor(bundle.body["joint_limits_rad"]).unsqueeze(0).repeat(n,1,1))
    def joint_write(q,v,**kw):
        writes.append(("joint",kw["env_ids"].clone()))
        data.joint_pos=q.clone();data.joint_vel=v.clone()
    def root_write(*args,**kw):writes.append(("root",kw["env_ids"].clone()))
    robot=SimpleNamespace(joint_names=bundle.joint_names,body_names=bundle.body_names,
        find_joints=lambda names,**kw:([bundle.joint_names.index(name) for name in names],names),
        find_bodies=lambda names,**kw:([bundle.body_names.index(name) for name in names],names),
        data=data,write_joint_state_to_sim=joint_write,write_root_state_to_sim=root_write,reset=lambda **kw:None)
    class Scene(dict):env_origins=torch.zeros(n,3)
    env=SimpleNamespace(device="cpu",num_envs=n,step_dt=.020,scene=Scene(robot=robot))
    command=m.CycleMotionCommand(cfg.commands["motion"],env)
    env.command_manager=SimpleNamespace(get_term=lambda _:command)
    data.joint_pos=command.joint_pos.clone();data.joint_vel=command.joint_vel.clone()
    for name,value in (("pos",command.body_pos_w),("quat",command.body_quat_w),
        ("lin_vel",command.body_lin_vel_w),("ang_vel",command.body_ang_vel_w)):
        setattr(data,"body_link_"+name+"_w",value.clone())
    observation_cfg={"actor":ObservationGroupCfg(terms={"condition":ObservationTermCfg(func=m.cycle_condition)},enable_corruption=False)}
    env.observation_manager=ObservationManager(observation_cfg,env)
    assert env.observation_manager.group_obs_dim["actor"]==(91,)
    observer=SimpleNamespace(actual_samples=0,window=SimpleNamespace(values={"clock_failure":torch.zeros(n,dtype=torch.bool)}),
        after_pending_reset=None,begin_sink=None,endpoint_sink=None,reset_sink=None)
    setattr(env,OBSERVER_ATTRIBUTE,observer)
    sampler=m.bind_packet_to_command(command,lambda:None)
    def packet():
        return {**sampler.expected(),"status":torch.full((n,),PlanStatus.ADMITTED,dtype=torch.int64),
            "plan_matches_guard_context":torch.ones(n,dtype=torch.bool)}
    nominal={}
    b=m.attach_cycle_runtime(env,nominal_provider=lambda:nominal,packet_provider=packet,
        request_ids=lambda:torch.arange(n),receipt_sink=lambda _:None,readiness=ReadinessConfig(2,12))
    command.reset(torch.arange(n))
    for name,value in (("pos",command.body_pos_w),("quat",command.body_quat_w),
        ("lin_vel",command.body_lin_vel_w),("ang_vel",command.body_ang_vel_w)):
        setattr(data,"body_link_"+name+"_w",value.clone())
    command.compute(dt=0.)  # Actual inherited WBC+CommandTerm path before obs
    initial_obs=env.observation_manager.compute(update_history=True)["actor"]
    assert initial_obs[:,-4:].tolist()==[[1.,0.,0.,0.]]*n
    s=state(n,48,dtype=torch.float32)
    s["joint_position"]=command.joint_pos.clone()
    s["root_position_w"]=command.anchor_pos_w.clone();s["root_quat_wxyz"]=command.anchor_quat_w.clone()
    s["glove_position_w"]=command.reference_gloves().clone()
    nominal.update({key:value.clone() for key,value in s.items()})
    observer.after_pending_reset(torch.arange(n),s,command.physics_steps,torch.zeros(n,dtype=torch.float64))
    observer.begin_sink()
    for _ in range(4):
        command.physics_steps.add_(1);observer.endpoint_sink(command.physics_steps,s)
    assert not b.readiness.launched.any()  # launch still cannot occur in the 5 ms callback
    command.compute(dt=.020)
    strike_obs=env.observation_manager.compute(update_history=True)["actor"]
    assert strike_obs[:,-4:].tolist()==[[0.,1.,0.,0.]]*n
    assert torch.all(strike_obs[:,8]==0.) and b.readiness.launch_tick.tolist()==[4,4]
    assert len(writes)==2 and not b.acceptance_eligible
    # Ordinary auto-reset order: retained terminal receipt, physical command
    # initialization, compute(dt=0), next observations, then fresh endpoint.
    observer.reset_sink(torch.arange(n),torch.ones(n,dtype=torch.bool),torch.zeros(n,dtype=torch.bool))
    command.reset(torch.arange(n));command.compute(dt=0.)
    reset_obs=env.observation_manager.compute(update_history=True)["actor"]
    assert reset_obs[:,-4:].tolist()==[[1.,0.,0.,0.]]*n
    assert torch.equal(reset_obs[:,:4],torch.nn.functional.one_hot(command.family_ids[command.trajectory_ids],4).float())
    assert torch.all(reset_obs[:,8]==0.) and len(writes)==4
    assert not command.elapsed.any()
    # The real observer injects clock_failure only after integration. Complete
    # the next fresh-reset callback, rather than stopping at reset observations.
    assert "clock_failure" not in s and "clock_failure" not in b.readiness.last_state
    observer.after_pending_reset(torch.arange(n),s,command.physics_steps,torch.ones(n,dtype=torch.float64))
    assert b.readiness.attempt_count.tolist()==[2,2]
    assert b.readiness.launch_tick.tolist()==[-1,-1]
    observer.begin_sink()
    for _ in range(4):
        command.physics_steps.add_(1);observer.endpoint_sink(command.physics_steps,s)
    command.compute(dt=.020)
    assert b.readiness.launch_tick.tolist()==[4,4]


def test_terminal_receipt_contains_only_selected_rows_and_preserves_global_counts():
    b,s,p,receipts=setup();window(b,s,boundary=False)
    b.record_reset(torch.tensor([1]),torch.tensor([False,True]),torch.zeros(2,dtype=torch.bool))
    receipt=receipts[0];r=receipt["readiness"]
    assert receipt["environment_ids"].tolist()==[1]
    assert r["request_id"].tolist()==[1]
    assert r["commitment"]["g1_launch_state"].shape==(1,7)
    assert r["nominal_guard"]["joint_position"].shape==(1,3)
    assert r["global_cumulative_counts"]["attempts"].item()==2
    assert r["global_cumulative_counts"]["failures"].item()==1
    assert b.readiness.logged.tolist()==[False,True]


def test_aggregate_sink_bounds_storage_and_keeps_all_terminal_counts():
    import json,tempfile
    from human_punching.wbc.telemetry.attempt_receipts import AggregateAttemptSink
    root=Path.cwd()/".cache/task10/substep_draft"
    with tempfile.TemporaryDirectory(dir=root) as directory:
        sink=AggregateAttemptSink(directory,shared_identity={"reference_sha256":"fixture"},flush_every_batches=2)
        b,s,p,_=setup();b.receipt_sink=sink
        window(b,s,boundary=False)
        b.record_reset(torch.tensor([1]),torch.tensor([False,True]),torch.zeros(2,dtype=torch.bool))
        assert not (Path(directory)/"aggregate.json").exists()
        b.record_reset(torch.tensor([0]),torch.tensor([True,False]),torch.zeros(2,dtype=torch.bool))
        data=json.loads((Path(directory)/"aggregate.json").read_text())
        assert data["terminal_attempts"]==2 and data["unlaunched_attempts"]==2
        assert data["terminal_attempts_by_family"]==[1,1,0,0]
        assert data["terminal_attempts_by_failure"]["environment_termination"]==2
        assert data["acceptance_eligible"] is False and not data["individual_traces_retained"]
        assert len(list(Path(directory).iterdir()))==2
        assert (Path(directory)/"aggregate.json").stat().st_size<4096
        assert "g1_launch_state" not in (Path(directory)/"aggregate.json").read_text()
        sink.flush()
        assert sink.report()["terminal_attempts"]==2


def test_bound_provider_ignores_changed_live_g1_buffers_after_commitment():
    b,s,p,_=setup(context_width=84)
    command=fake_bound_command();setattr(command._env,m.ATTRIBUTE,b)
    provider=m.bind_packet_to_command(command,lambda:p)
    p.update(provider.expected());b.packet_provider=provider
    window(b,s)
    frozen={key:value.clone() for key,value in b.readiness.committed.items()}
    actor=b.actor_context.clone()
    for key in ("target_w","g1_launch_state","actor_g1_context"):
        p[key].fill_(float("nan"))
    p["family_id"].fill_(100);p["plan_handle"].fill_(200)
    p["plan_digest"].fill_(9);p["requested_ttc_s"].fill_(float("nan"))
    window(b,s)
    assert all(torch.equal(value,b.readiness.committed[key]) for key,value in frozen.items())
    assert torch.equal(actor,b.actor_context)
    assert torch.equal(b.reward_context()["target_w"],frozen["target_w"])
    # A newly reset row must pass the live packet/bank binding again.
    b.pending_reset[1]=True
    with pytest.raises(ValueError,match="immutable reference"):
        provider()
