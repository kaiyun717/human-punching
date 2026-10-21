"""Explicit connector contracts; fixtures do not establish shared-start execution."""
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from human_punching.phase2_phases import (PhaseSchedule,RequestedCommand,SavedExecutionState,
    compare_saved_states,LaunchLatch,validate_phase_join,validate_reference_derivatives)


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
    <geom size=".1" mass="1"/><body><joint name="a"/><geom size=".1" mass="1"/>
    </body></body></worldbody><actuator><motor joint="a"/></actuator></mujoco>''')


def snapshot(model,*,clock=.4):
    data=mujoco.MjData(model)
    data.time=clock
    return SavedExecutionState.capture_cpu_native(model,data,model_sha256="a"*64,controller_version="test_controller",
        previous_raw_action=np.zeros(48),held_raw_action=np.zeros(48),reference_id="test_shared_start",
        reference_clock_s=clock,reference_phase="ready_hold",actor_update_index=20,decimation_phase=0,
        support_contact=[True,True],support_interval_end_s=clock,adapter_inventory_complete=True,
        adapter_state={"reset_count":np.array([1])})


def test_connector_motion_does_not_define_outward_launch():
    schedule=PhaseSchedule(60,80,180,200,240)
    assert schedule.phase_at(0)=="preparation"
    assert schedule.phase_at(60)=="ready_hold"
    assert schedule.phase_at(79)=="ready_hold"
    assert schedule.phase_at(80)=="outward"
    assert schedule.phase_at(180)=="recovery"
    assert schedule.phase_at(200)=="guard_hold"
    assert schedule.phase_denominator_s==pytest.approx(2.2)
    assert schedule.evaluation_end_tick==440
    with pytest.raises(ValueError,match="actor boundary"):
        PhaseSchedule(60,81,180,200,240)


def test_snapshot_requires_all_native_and_controller_history(model):
    state=snapshot(model)
    state.validate_model_size(model)
    assert compare_saved_states(state,snapshot(model))["identical"]
    altered=replace(state,previous_raw_action=np.ones(48))
    assert compare_saved_states(state,altered)["changed_fields"]==["previous_raw_action"]
    altered=replace(state,adapter_state={"reset_count":np.array([2])})
    assert "adapter_state" in compare_saved_states(state,altered)["changed_fields"]
    with pytest.raises(ValueError,match="qpos/qvel"):
        replace(state,native_state_spec=int(mujoco.mjtState.mjSTATE_QPOS)|int(mujoco.mjtState.mjSTATE_QVEL))
    with pytest.raises(ValueError,match="Inventory"):
        replace(state,adapter_inventory_complete=False)
    with pytest.raises(ValueError):
        state.previous_raw_action.setflags(write=True)


def test_native_warmstart_and_ctrl_are_compared_even_if_qpos_qvel_equal(model):
    data=mujoco.MjData(model)
    data.time=.4
    first=snapshot(model)
    data.qacc_warmstart[0]=.1
    spec=int(mujoco.mjtState.mjSTATE_INTEGRATION)
    vector=np.empty(mujoco.mj_stateSize(model,spec));mujoco.mj_getState(model,data,vector,spec)
    changed=replace(first,native_integration_state=vector)
    assert compare_saved_states(first,changed)["changed_fields"]==["native_integration_state"]


def test_launch_latches_current_readiness_and_freezes_request_target_reference(model):
    schedule=PhaseSchedule(60,80,180,200,240)
    command=RequestedCommand("cross",.75,"cross_shared_v1","b"*64)
    latch=LaunchLatch(schedule,command);target=np.array([1.,0.,1.])
    result=latch.attempt(current_tick=80,prepared_tick=60,current_ready=True,target_w=target,
        reference_sha256="c"*64,state=snapshot(model),final_planning_wall_s=.019,model=model)
    assert result["accepted"] and result["actual_launch_time_s"]==.4
    target[0]=2.
    assert result["commitment"].target_w==(1.,0.,1.)
    with pytest.raises(ValueError,match="changed"):
        result["commitment"].assert_unchanged(command,target,"c"*64)
    with pytest.raises(ValueError,match="disposition"):
        latch.attempt(current_tick=84,prepared_tick=60,current_ready=True,target_w=[1.,0.,1.],
            reference_sha256="c"*64,state=snapshot(model),final_planning_wall_s=.001,model=model)


@pytest.mark.parametrize("changes,reason",[({"current_ready":False},"current_readiness_failed"),
    ({"prepared_tick":61},"preparation_missing_or_late"),({"prepared_tick":None},"preparation_missing_or_late"),
    ({"current_tick":84},"wrong_launch_boundary"),({"current_tick":81},"not_actor_boundary"),
    ({"final_planning_wall_s":.02001},"final_planning_late")])
def test_late_unready_and_wrong_boundary_fail_closed(model,changes,reason):
    latch=LaunchLatch(PhaseSchedule(60,80,180,200,240),RequestedCommand("jab",.6,"test","b"*64))
    kwargs=dict(current_tick=80,prepared_tick=60,current_ready=True,target_w=[1.,0.,1.],
        reference_sha256="c"*64,state=snapshot(model),final_planning_wall_s=.01,model=model)
    result=latch.attempt(**(kwargs|changes))
    assert result=={"accepted":False,"reason":reason,"actual_launch_time_s":None}
    assert latch.commitment is None


def test_launch_rejects_stale_support_and_cannot_mutate_public_request(model):
    command=RequestedCommand("jab",.6,"test","b"*64)
    latch=LaunchLatch(PhaseSchedule(60,80,180,200,240),command)
    with pytest.raises(AttributeError):
        latch.command=RequestedCommand("cross",.6,"other","b"*64)
    result=latch.attempt(current_tick=80,prepared_tick=60,current_ready=True,target_w=[1.,0.,1.],
        reference_sha256="c"*64,state=replace(snapshot(model),support_interval_end_s=.39),
        final_planning_wall_s=.01,model=model)
    assert result["reason"]=="support_not_current"


def test_c2_join_catches_hidden_velocity_and_root_pose_swap(model):
    left={"qpos":model.qpos0.copy(),"qvel":np.zeros(model.nv),"qacc":np.zeros(model.nv)}
    right={key:value.copy() for key,value in left.items()}
    assert validate_phase_join(model,left,right)["passed"]
    right["qvel"][0]=.01
    assert not validate_phase_join(model,left,right)["passed"]
    right["qvel"][0]=0.;right["qpos"][0]=.01
    assert not validate_phase_join(model,left,right)["passed"]


def test_derivatives_follow_full_grid_while_preparation_moves(model):
    schedule=PhaseSchedule(12,16,30,36,40)
    times=np.arange(41)*.005
    q=np.tile(model.qpos0,(41,1));q[:,0]=.1*times
    v=np.zeros((41,model.nv));v[:,0]=.1
    a=np.zeros_like(v)
    checked=validate_reference_derivatives(model,times,q,v,a,schedule)
    assert checked["passed"] and checked["launch_tick"]==16
    assert not checked["launch_inferred_from_motion"]
    v[12,0]+=.01
    assert not validate_reference_derivatives(model,times,q,v,a,schedule)["passed"]
