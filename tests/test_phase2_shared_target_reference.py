from dataclasses import replace

import mujoco
import numpy as np
import pytest

from test_phase2_target_reference import synthetic
from human_punching.phase2_phases import PhaseSchedule
from human_punching.phase2_shared_target_reference import (
    SharedTargetRecipe,compose_shared_target,desired_shared_active_path,
    finalize_shared_target_feedforward,
)
from human_punching.phase2_target_reference import RetargetRejection
from human_punching.wbc.references import tangent_velocities
from human_punching.phase2_shared_reference import quintic_coefficients,evaluate_quintic


def active_fk(model,q):
    data=mujoco.MjData(model);site=model.site("LeftHand_center").id;rows=[]
    for pose in q:
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
        rows.append(data.site_xpos[site].copy())
    return np.array(rows)


@pytest.fixture
def inputs(synthetic):
    model,original,settings=synthetic
    prep=40;source_schedule=PhaseSchedule(0,16,80,90,100)
    schedule=PhaseSchedule(prep,prep+16,prep+80,prep+90,prep+100)
    source_q=original["qpos"].copy()
    # Distinct hypothetical lower-body/whole-body support geometry is an INPUT.
    # The composition helper must never translate it or replace its saved root.
    source_q[:,1]+=.005
    prefix=np.tile(source_q[0],(prep,1));qa=model.jnt_qposadr[model.joint("LeftShoulder_z").id]
    s=np.arange(prep)/prep;prefix[:,qa]+=.002*(s-1)**3
    q=np.vstack((prefix,source_q));t=np.arange(len(q))*.005
    v=tangent_velocities(model,t,q);a=np.gradient(v,t,axis=0,edge_order=2)
    s=np.arange(prep+1)/prep;v[:prep+1]=0.;a[:prep+1]=0.
    va=model.jnt_dofadr[model.joint("LeftShoulder_z").id]
    v[:prep+1,va]=.006*(s-1)**2/(prep*.005)
    a[:prep+1,va]=.012*(s-1)/(prep*.005)**2
    for left,right in ((prep,schedule.launch_tick),(schedule.guard_hold_tick,schedule.reference_end_tick)):
        v[left:right+1]=0.;a[left:right+1]=0.
    shared={"time_s":t,"t":t.copy(),"qpos":q,"qvel":v,"qacc":a,
        "head_center":original["head_center"].copy(),"active_hand_index":0,"schedule":schedule,
        "derivative_version":"synthetic_analytic_connector_v1"}
    return model,shared,original,active_fk(model,original["qpos"]),source_schedule,SharedTargetRecipe(settings)


def compose(inputs,**overrides):
    model,shared,original,path,source_schedule,recipe=inputs
    return compose_shared_target(model,overrides.get("shared",shared),overrides.get("original",original),
        overrides.get("path",path),overrides.get("target",original["head_center"]),
        source_schedule=overrides.get("schedule",source_schedule),recipe=overrides.get("recipe",recipe))


def test_explicit_original_path_blend_differs_from_constant_target_shift():
    shared=np.array([[0.,0,0],[0,0,0],[.6,.3,1.],[.7,.3,1.],[.6,.3,1.]])
    original=np.array([[.4,.2,1.],[.5,.2,1.],[.4,.2,1.]])
    e=np.array([0.,0.,0.,1.,0.]);delta=np.array([.02,-.01,0.])
    desired=desired_shared_active_path(shared,original,target_delta=delta,preparation_ticks=2,envelope=e)
    np.testing.assert_array_equal(desired[e==0],shared[e==0])
    np.testing.assert_allclose(desired[3],original[1]+delta)
    assert not np.allclose(desired[3],shared[3]+delta)


def test_shared_conversion_is_compensated_even_when_requested_target_is_unchanged(inputs):
    model,shared,original,path,_,_=inputs;out,meta=compose(inputs)
    actual=active_fk(model,out["qpos"]);plateau=80
    assert np.linalg.norm(active_fk(model,shared["qpos"])[plateau]-path[plateau-40]) > .0049
    assert np.linalg.norm(actual[plateau]-path[plateau-40]) <= .0005
    assert not np.array_equal(out["qpos"],shared["qpos"])
    np.testing.assert_array_equal(out["head_center"],original["head_center"])
    assert meta["recipe"]["version"] == "task10b_shared_original_glove_corridor_v1"


def test_shared_guard_prefix_root_and_connector_derivatives_are_exactly_preserved(inputs):
    _,shared,_,_,_,_=inputs;out,meta=compose(inputs)
    np.testing.assert_array_equal(out["qpos"][:57],shared["qpos"][:57])
    np.testing.assert_array_equal(out["qpos"][120:],shared["qpos"][120:])
    np.testing.assert_array_equal(out["qpos"][:,:7],shared["qpos"][:,:7])
    np.testing.assert_array_equal(out["qvel"][:41],shared["qvel"][:41])
    np.testing.assert_array_equal(out["qacc"][:41],shared["qacc"][:41])
    assert np.linalg.norm(out["qvel"][0]) > 0
    assert meta["connector_derivatives_preserved_through_tick"] == 40
    assert out["schedule"] == shared["schedule"] and meta["phase_denominator_s"] == pytest.approx(1.7)
    assert out["derivative_version"] != shared["derivative_version"]
    np.testing.assert_array_equal(out["source_time_s"],inputs[2]["time_s"])
    assert out["source_start_tick"] == 40


def test_relabeling_original_head_center_or_glove_path_is_rejected(inputs):
    _,shared,_,path,_,_=inputs
    with pytest.raises(ValueError,match="head_center relabeling"):
        compose(inputs,shared={**shared,"head_center":shared["head_center"]+[.01,0,0]})
    with pytest.raises(ValueError,match="source reference FK"):
        compose(inputs,path=path+[.002,0,0])


def test_mismatched_phase_mapping_or_clock_is_rejected(inputs):
    _,shared,original,_,schedule,_=inputs
    with pytest.raises(ValueError,match="source offset"):
        compose(inputs,schedule=replace(schedule,launch_tick=20))
    wrong={**original,"time_s":original["time_s"]*.9}
    with pytest.raises(ValueError):compose(inputs,original=wrong)
    with pytest.raises(ValueError,match="another connector"):
        compose(inputs,schedule=replace(schedule,preparation_complete_tick=4))


def test_missing_derivative_version_and_modified_source_upper_knots_rejected(inputs):
    _,shared,_,_,_,_=inputs
    absent={k:v for k,v in shared.items() if k!="derivative_version"}
    with pytest.raises(ValueError,match="derivative version"):compose(inputs,shared=absent)
    q=shared["qpos"].copy();q[80,9]+=.001
    with pytest.raises(ValueError,match="copied source upper-joint"):
        compose(inputs,shared={**shared,"qpos":q})


def test_excessive_shared_geometry_change_refuses_instead_of_root_translation(inputs):
    _,shared,_,_,_,_=inputs;q=shared["qpos"].copy();q[:,1]+=.3
    with pytest.raises(RetargetRejection,match="shared_geometry_compensation_outside_recipe"):
        compose(inputs,shared={**shared,"qpos":q})
    np.testing.assert_array_equal(shared["qpos"][:,1],.005)


def test_feedforward_retains_explicit_launch_after_moving_preparation(inputs):
    model,shared,_,_,_,_=inputs;candidate,meta=compose(inputs)
    out,info=finalize_shared_target_feedforward(model,candidate,meta)
    assert out["schedule"].launch_tick == 56
    assert np.linalg.norm(out["qvel"][0]) > 0
    np.testing.assert_array_equal(out["attack_window_reference_s"],np.array([56,120])*.005)
    np.testing.assert_array_equal(out["qvel"][:41],shared["qvel"][:41])
    assert info["feedforward_recomputed"] and info["first_contact_TTC_not_yet_screened"]
    assert not info["physical_qualification"] and not info["physics_integrated"]


def test_connector_envelope_must_be_zero_without_invented_original_prefix():
    with pytest.raises(ValueError,match="zero connector"):
        desired_shared_active_path(np.zeros((5,3)),np.zeros((3,3)),target_delta=np.zeros(3),
            preparation_ticks=2,envelope=np.array([0,1,0,1,0]))


def test_zero_source_holds_and_no_stale_polynomial_claim(inputs):
    _,shared,_,_,_,_=inputs
    supplied={**shared,"connector_coefficients":np.arange(12).reshape(2,6),
              "source_segment_coefficients":np.ones((100,6,3))}
    out,metadata=compose(inputs,shared=supplied)
    assert "source_segment_coefficients" not in out and metadata["source_segment_coefficients_omitted"]
    np.testing.assert_array_equal(out["connector_coefficients"],supplied["connector_coefficients"])
    for left,right in metadata["constant_source_hold_windows_ticks"]:
        assert not np.any(out["qvel"][left:right+1]) and not np.any(out["qacc"][left:right+1])
    velocity=shared["qvel"].copy();velocity[56,8]=.001
    with pytest.raises(ValueError,match="source holds"):
        compose(inputs,shared={**shared,"qvel":velocity})


def test_source_acceleration_uses_returned_hold_boundary_velocity(inputs):
    model,shared,_,_,_,_=inputs
    out,metadata=compose(inputs)
    expected=np.gradient(out["qvel"],out["time_s"],axis=0,edge_order=2)
    moving=np.arange(len(out["time_s"]))>out["source_start_tick"]
    for first,last in metadata["constant_source_hold_windows_ticks"]:
        moving[first:last+1]=False
    np.testing.assert_array_equal(out["qacc"][moving],expected[moving])
    first=out["schedule"].launch_tick+1
    raw_velocity=tangent_velocities(model,out["time_s"],out["qpos"])
    before_boundary_fix=np.gradient(raw_velocity,out["time_s"],axis=0,edge_order=2)
    assert np.max(abs(before_boundary_fix[first]-expected[first]))>1.e-6
    # This fix must never replace the analytic connector's initial derivatives.
    np.testing.assert_array_equal(out["qacc"][:41],shared["qacc"][:41])
    assert out["derivative_version"].endswith("_v2")


def test_explicit_lower_preparation_and_all_joint_coefficient_addresses_survive_composition(inputs):
    model,shared,_,_,_,_=inputs;prep=shared['schedule'].preparation_complete_tick
    q=shared['qpos'].copy();v=shared['qvel'].copy();a=shared['qacc'].copy()
    joint=model.joint('LeftLeg_z').id;qa=model.jnt_qposadr[joint];va=model.jnt_dofadr[joint]
    coefficients=quintic_coefficients([.015],[.06],[0.],[q[prep,qa]],[0.],[0.],prep*.005)
    values=evaluate_quintic(coefficients,np.arange(prep+1)*.005,prep*.005)
    for field,series,address in zip((q,v,a),values,(qa,va,va)):field[:prep+1,address]=series[:,0]
    # Remove cancellation roundoff at the endpoint before the exact constant hold.
    q[prep,qa]=shared['qpos'][prep,qa];v[prep,va]=a[prep,va]=0.
    supplied={**shared,'qpos':q,'qvel':v,'qacc':a,'connector_coefficients':coefficients,
        'connector_qpos_addresses':np.array([qa]),'connector_dof_addresses':np.array([va])}
    out,metadata=compose(inputs,shared=supplied)
    np.testing.assert_array_equal(out['qpos'][:,qa],q[:,qa])
    np.testing.assert_array_equal(out['qvel'][:prep+1],v[:prep+1])
    np.testing.assert_array_equal(out['qacc'][:prep+1],a[:prep+1])
    np.testing.assert_array_equal(out['connector_qpos_addresses'],[qa])
    np.testing.assert_array_equal(out['connector_dof_addresses'],[va])
    np.testing.assert_array_equal(out['connector_coefficients'],coefficients)
    assert metadata['connector_coefficient_addresses_preserved']
    assert out['qvel'][0,va]==.06 and out['qpos'][0,qa]==.015
    with pytest.raises(ValueError,match='addresses'):
        compose(inputs,shared={**supplied,'connector_dof_addresses':np.array([0])})


def test_shared_target_checks_session_deadline_during_fk_and_ik(inputs):
    calls=[]
    def check():calls.append(True)
    model,shared,original,path,schedule,recipe=inputs
    compose_shared_target(model,shared,original,path,original['head_center'],source_schedule=schedule,recipe=recipe,checkpoint=check)
    assert len(calls)>len(shared['time_s'])
    def stop():raise RuntimeError('synthetic_cooperative_stop')
    with pytest.raises(RuntimeError,match='cooperative_stop'):
        compose_shared_target(model,shared,original,path,original['head_center'],source_schedule=schedule,recipe=recipe,checkpoint=stop)
