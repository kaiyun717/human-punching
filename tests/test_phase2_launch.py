"""Pure synthetic prefix state/history evidence; no native simulation."""
from copy import deepcopy
import time

import numpy as np
import pytest

from human_punching.phase2_launch import ClosedProtocolLaunchGuard, ClosedProtocolViolation


def request():
    return dict(trial_id="synthetic_0", request_identity="a"*64, family="left_hook",
        requested_ttc_s=1.1, style="left_hook_uniform", candidate_id="left_hook_110",
        reference_sha256="b"*64, target_candidate_identity="c"*64,
        checkpoint_sha256="e"*64, snapshot_payload_sha256="d"*64,
        snapshot_slot_id=134, placement=dict(forward_m=.05,left_m=0.,yaw_rad=0.),
        head_point_velocity_w=[.04,-.02,.01], target_w=[1.,0.,1.2])


def fixture(launch_tick=4,extra=None):
    q=np.zeros((1,55),np.float32);q[:,3]=1.
    v=np.zeros((1,54),np.float32);v[:,9]=.7  # Quiet-hook velocity is not required.
    state=dict(qpos=q.copy(),qvel=v.copy(),ctrl=np.zeros((1,48),np.float32),
        qacc_warmstart=np.zeros((1,54),np.float32),qfrc_applied=np.zeros((1,54),np.float32),
        xfrc_applied=np.zeros((1,24,6),np.float32),act=np.zeros((1,0),np.float32),
        previous_raw_action=np.zeros((1,48),np.float32),ticks=np.zeros(1,np.int64),
        reference_ids=np.zeros(1,np.int64),reset_counts=np.ones(1,np.int64),
        support=np.ones((1,2),bool),time=np.array([0.],np.float64),
        phase_denominators_s=np.array([2.],np.float32),physics_steps=0,actor_updates=0)
    if extra:state.update(extra)
    guard=ClosedProtocolLaunchGuard(q,v,[request()],[launch_tick],[2.])
    guard.initialize(state)
    return guard,state


def actor_interval(guard,state,action=None):
    action=np.full((1,48),.2,np.float32) if action is None else action
    assert not guard.actor_boundary(state).any()
    for substep in range(4):
        guard.before_step(state,action)
        state["physics_steps"]+=1;state["ticks"]+=1;state["time"]+=.005
        state["qpos"][:,7]+=.003;state["ctrl"][:]=action
        state["qacc_warmstart"]+=.1
        guard.after_step(state,action)
    state["actor_updates"]+=1
    state["previous_raw_action"][:]=action


def target_receipt(tick=4):
    expected=request()
    return {key:expected[key] for key in ("trial_id","request_identity","target_candidate_identity",
        "snapshot_payload_sha256","snapshot_slot_id","placement","head_point_velocity_w","target_w")}|dict(
        capture_episode_tick=tick,input_episode_time_s=tick*.005,
        head_geometry="head_collision",head_velocity_method="jacobian_qvel",
        source_kind="fixed_canonical_snapshot",no_prediction=True)


def finalize(guard,before,after=None,*,target=None,ready=True,current=None,elapsed=.001):
    return guard.finalize(before,deepcopy(before) if after is None else after,
        [target_receipt() if target is None else target],np.array([ready]),
        current_requests=[request() if current is None else current],
        planning_started_perf_counter=time.perf_counter()-elapsed,callback_wall_s=0.,cache_lookup_wall_s=0.)


def test_nonquiet_continuous_prefix_launches_without_exact_joint_return():
    guard,state=fixture();actor_interval(guard,state)
    assert state["qvel"][0,9]>.5 and state["qpos"][0,7]>.01
    # Native forward may update warmstart before the next actor observation.
    state["qacc_warmstart"]+=1.
    assert guard.actor_boundary(state).tolist()==[True]
    result=finalize(guard,state)[0]
    assert result["accepted"] and result["reasons"]==[] and result["actual_launch_tick"]==4
    assert not result["exact_joint_return_required"] and not result["quiet_hook_gate_added"]
    assert result["simulated_time_advanced_s"]==0 and not result["hardware_real_time_qualified"]


@pytest.mark.parametrize("field",["qpos","qvel","ctrl","qfrc_applied","xfrc_applied","time","support"])
def test_prefix_state_swap_is_detected_before_launch(field):
    guard,state=fixture();actor_interval(guard,state)
    if field=="support":state[field][0,0]=False
    else:state[field].flat[0]+=.001
    with pytest.raises(ClosedProtocolViolation,match="outside integration"):
        guard.actor_boundary(state)


@pytest.mark.parametrize("change",["history","reset","reference","phase","counter"])
def test_controller_reference_and_history_cannot_be_silently_replaced(change):
    guard,state=fixture();actor_interval(guard,state)
    if change=="history":state["previous_raw_action"][0,0]+=.01
    if change=="reset":state["reset_counts"][:]=2
    if change=="reference":state["reference_ids"][:]=1
    if change=="phase":state["phase_denominators_s"][:]=3.
    if change=="counter":state["actor_updates"]+=1
    with pytest.raises(ClosedProtocolViolation):guard.actor_boundary(state)


def test_raw_residual_must_remain_held_for_four_physics_intervals():
    guard,state=fixture();guard.actor_boundary(state)
    action=np.ones((1,48),np.float32)
    guard.before_step(state,action)
    state["physics_steps"]=1;state["ticks"]+=1;state["time"]+=.005
    guard.after_step(state,action)
    with pytest.raises(ClosedProtocolViolation,match="four-step"):
        guard.before_step(state,action*.5)


def test_outward_cannot_begin_before_actor_boundary_and_commitment():
    guard,state=fixture(launch_tick=0)
    with pytest.raises(ClosedProtocolViolation,match="boundary has not"):
        guard.before_step(state,np.zeros((1,48)))
    guard.actor_boundary(state)
    with pytest.raises(ClosedProtocolViolation,match="final commitment"):
        guard.before_step(state,np.zeros((1,48)))


@pytest.mark.parametrize("kind,reason",[("target","launch_target_input"),("stale","launch_target_input"),
    ("prediction","launch_target_input"),("request","command_or_reference"),
    ("readiness","positional_support"),("late","final_planning_late"),("native","state_changed")])
def test_launch_mismatch_late_or_state_change_is_a_preserved_rejection(kind,reason):
    guard,state=fixture();actor_interval(guard,state);guard.actor_boundary(state)
    target=target_receipt();current=request();after=deepcopy(state);ready=True;elapsed=.001
    if kind=="target":target["target_w"]=[1.001,0.,1.2]
    if kind=="stale":target["capture_episode_tick"]=0
    if kind=="prediction":target["no_prediction"]=False
    if kind=="request":current["requested_ttc_s"]=1.2
    if kind=="readiness":ready=False
    if kind=="late":elapsed=.021
    if kind=="native":after["qacc_warmstart"][0,0]+=.01
    result=finalize(guard,state,after,target=target,ready=ready,current=current,elapsed=elapsed)[0]
    assert not result["accepted"] and reason in result["reason"]
    assert result["actual_launch_tick"] is None
    # No retry can relabel a subsequent target match as the same launch.
    with pytest.raises(ClosedProtocolViolation,match="No pending"):
        finalize(guard,state)


def test_rejected_reference_tick_can_freeze_while_recorded_physics_continues():
    guard,state=fixture();actor_interval(guard,state);guard.actor_boundary(state)
    assert not finalize(guard,state,ready=False)[0]["accepted"]
    action=np.full((1,48),.3,np.float32)
    for _ in range(4):
        guard.before_step(state,action)
        state["physics_steps"]+=1;state["time"]+=.005  # Rejected reference clock remains4.
        state["qpos"][:,7]+=.002
        guard.after_step(state,action)
    state["actor_updates"]+=1;state["previous_raw_action"][:]=action
    assert not guard.actor_boundary(state).any()
    assert state["time"][0]>.039 and state["ticks"][0]==4


def test_initialization_requires_declared_physical_state_and_zero_history():
    guard,state=fixture()
    with pytest.raises(ClosedProtocolViolation,match="only once"):guard.initialize(state)
    bad=ClosedProtocolLaunchGuard(state["qpos"],state["qvel"],[request()],[4],[2.])
    state["previous_raw_action"][0,0]=1.
    with pytest.raises(ClosedProtocolViolation,match="initial state/history"):
        bad.initialize(state)


def test_planning_snapshot_digest_covers_supplied_opaque_native_state():
    guard,state=fixture(extra=dict(native_extra_history=np.array([[123.]],np.float32)))
    actor_interval(guard,state)
    guard.actor_boundary(state)
    changed=deepcopy(state);changed["native_extra_history"][:]=124.
    result=finalize(guard,state,changed)[0]
    assert result["reason"]=="physical_or_controller_state_changed_during_planning"


@pytest.mark.parametrize("field,value",[("snapshot_slot_id",135),("placement",dict(forward_m=0.,left_m=0.,yaw_rad=0.)),
    ("head_point_velocity_w",[.03,-.02,.01]),("input_episode_time_s",.019),
    ("head_geometry","pelvis"),("head_velocity_method","pelvis_velocity")])
def test_fresh_target_input_binds_snapshot_placement_measured_head_velocity_and_clock(field,value):
    guard,state=fixture();actor_interval(guard,state);guard.actor_boundary(state)
    receipt=target_receipt();receipt[field]=value
    assert finalize(guard,state,target=receipt)[0]["reason"]=="launch_target_input_mismatch_or_stale"


def test_opaque_nested_state_is_copied_and_checked_through_prefix():
    guard,state=fixture(extra=dict(plugin=dict(memory=[1.,2.])))
    actor_interval(guard,state)
    state["plugin"]["memory"][0]=3.
    with pytest.raises(ClosedProtocolViolation,match="outside integration: plugin"):
        guard.actor_boundary(state)


def test_native_state_inventory_cannot_silently_grow_or_shrink():
    guard,state=fixture();state["undisclosed_state"]=np.array([1.])
    with pytest.raises(ClosedProtocolViolation,match="field inventory"):
        guard.actor_boundary(state)


def test_forward_warmstart_refresh_is_frozen_after_boundary_capture():
    guard,state=fixture();actor_interval(guard,state)
    state["qacc_warmstart"]+=1.
    guard.actor_boundary(state);finalize(guard,state)
    state["qacc_warmstart"]+=.01
    with pytest.raises(ClosedProtocolViolation,match="outside integration: qacc_warmstart"):
        guard.before_step(state,np.zeros((1,48),np.float32))


@pytest.mark.parametrize("field,value",[("reference_sha256","f"*64),("checkpoint_sha256","f"*64),
    ("family","jab"),("requested_ttc_s",1.2),("target_w",[1.001,0.,1.2]),("style","alternate")])
def test_committed_request_identity_is_enforced_through_recovery(field,value):
    guard,state=fixture();actor_interval(guard,state);guard.actor_boundary(state);finalize(guard,state)
    assert guard.assert_requests([request()])
    changed=request();changed[field]=value
    with pytest.raises(ClosedProtocolViolation,match="Committed command"):
        guard.assert_requests([changed])
