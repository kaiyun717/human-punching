from dataclasses import replace

import mujoco
import numpy as np
import pytest

from human_punching.phase2_target_reference import (
    RetargetRejection,RetargetSettings,correction_envelope,finalize_original_feedforward,
    resolve_joint_mask,retarget_upper_body,
)
from human_punching.wbc.references import tangent_velocities


@pytest.fixture
def synthetic():
    xml='''<mujoco><compiler angle="radian"/><option timestep=".005"/>
      <default><joint limited="true" range="-2 2"/><geom size=".035" mass=".3"/></default>
      <worldbody><body name="Hips" pos="0 0 1"><freejoint name="root"/><geom/>
       <site name="LeftFoot_sole" pos="0 .1 -1"/><site name="RightFoot_sole" pos="0 -.1 -1"/>
       <body name="Chest"><joint name="Chest_z" axis="0 0 1"/><geom/>
        <body name="RightArm" pos="0 -.2 0"><joint name="RightArm_z" axis="0 0 1"/><geom/>
         <site name="RightHand_center" pos=".4 0 0"/>
        </body>
        <body name="LeftShoulder" pos="0 .2 0"><joint name="LeftShoulder_z" axis="0 0 1"/><geom/>
         <body name="LeftArm" pos=".2 0 0"><joint name="LeftArm_z" axis="0 0 1"/><geom/>
          <body name="LeftForeArm" pos=".2 0 0"><joint name="LeftForeArm_z" axis="0 0 1"/><geom/>
           <site name="LeftHand_center" pos=".2 0 0"/>
          </body>
         </body>
        </body>
       </body>
       <body name="LeftLeg" pos="0 .1 -.2"><joint name="LeftLeg_z" axis="0 0 1"/><geom/>
        <site name="additional_sole" pos="0 0 -.8"/>
       </body>
      </body></worldbody>
      <actuator>
       <motor joint="Chest_z" forcerange="-50 50"/>
       <motor joint="RightArm_z" forcerange="-50 50"/>
       <motor joint="LeftShoulder_z" forcerange="-50 50"/>
       <motor joint="LeftArm_z" forcerange="-50 50"/>
       <motor joint="LeftForeArm_z" forcerange="-50 50"/>
       <motor joint="LeftLeg_z" forcerange="-50 50"/>
      </actuator></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml)
    t=np.arange(101)*.005;q=np.tile(model.qpos0,(len(t),1))
    q[:,model.jnt_qposadr[model.joint("LeftShoulder_z").id]]=.1+.08*correction_envelope(t,.08,.4,.06)
    q[:,model.jnt_qposadr[model.joint("LeftArm_z").id]]=.5
    q[:,model.jnt_qposadr[model.joint("LeftForeArm_z").id]]=-.8
    source={"time_s":t,"qpos":q,"head_center":np.array([.75,.2,1.]),"guard_hold_start":.45,"active_hand_index":0}
    settings=RetargetSettings("Left",("LeftShoulder_z","LeftArm_z","LeftForeArm_z"),envelope_blend_s=.06)
    return model,source,settings


def retarget(synthetic,delta=(0,.007,0),settings=None):
    model,source,default=synthetic
    return retarget_upper_body(model,source,source["head_center"]+delta,
        source_outward_start_s=.08,source_outward_end_s=.4,settings=settings or default)


def test_mask_uses_actual_named_joint_addresses(synthetic):
    model,_,settings=synthetic;mask=resolve_joint_mask(model,settings)
    assert mask["qpos_addresses"].tolist() == [9,10,11]
    assert mask["dof_addresses"].tolist() == [8,9,10]
    # Reordering names changes matching addresses; no assumption of arm ordering.
    reversed_mask=resolve_joint_mask(model,replace(settings,optimized_joint_names=settings.optimized_joint_names[::-1]))
    assert reversed_mask["qpos_addresses"].tolist() == [11,10,9]


@pytest.mark.parametrize("name",["root","LeftLeg_z","RightArm_z","Chest_z","missing"])
def test_root_legs_opposite_arm_and_implicit_trunk_are_rejected(synthetic,name):
    model,_,settings=synthetic
    with pytest.raises(ValueError):resolve_joint_mask(model,replace(settings,optimized_joint_names=(name,)))


def test_trunk_requires_explicit_frozen_recipe_and_cannot_include_legs(synthetic):
    model,_,settings=synthetic
    with pytest.raises(ValueError):replace(settings,trunk_joint_names=("Chest_z",))
    explicit=replace(settings,trunk_joint_names=("Chest_z",),recipe_version="task10b_explicit_trunk_target_fixture_v1")
    assert len(resolve_joint_mask(model,explicit)["joint_ids"]) == 4
    with pytest.raises(ValueError):resolve_joint_mask(model,replace(explicit,trunk_joint_names=("LeftLeg_z",)))


def test_c2_envelope_zero_at_endpoints_and_has_cubic_boundary_scaling():
    eps=np.array([1e-5,2e-5])
    np.testing.assert_array_equal(correction_envelope([0,.1,.9,1],.1,.9,.1),0)
    assert correction_envelope([.4,.5,.6],.1,.9,.1).tolist() == [1,1,1]
    left=correction_envelope(.1+eps,.1,.9,.1)
    right=correction_envelope(.9-eps,.1,.9,.1)
    assert 7.99 < left[1]/left[0] < 8.01
    np.testing.assert_allclose(left,right,atol=1e-20)
    with pytest.raises(ValueError):correction_envelope([.5],.1,.9,.5)


def test_target_correction_preserves_entire_clock_root_legs_holds_and_inactive_corridor(synthetic):
    model,source,settings=synthetic;before=source["qpos"].copy()
    candidate,metadata=retarget(synthetic)
    selected=resolve_joint_mask(model,settings)["qpos_addresses"]
    fixed=np.ones(model.nq,bool);fixed[selected]=False
    np.testing.assert_array_equal(candidate["qpos"][:,fixed],before[:,fixed])
    outside=(source["time_s"]<=.08)|(source["time_s"]>=.4)
    np.testing.assert_array_equal(candidate["qpos"][outside],before[outside])
    np.testing.assert_array_equal(candidate["time_s"],source["time_s"])
    np.testing.assert_array_equal(source["qpos"],before)
    assert max(r["active_error_m"] for r in metadata["frame_geometry"]) <= .0005
    assert max(r["inactive_error_m"] for r in metadata["frame_geometry"]) == 0
    assert metadata["maximum_joint_correction_rad"] <= settings.maximum_joint_delta_rad
    assert metadata["maximum_joint_correction_rad"] > 0


def test_source_target_is_exact_path_fixed_point(synthetic):
    candidate,metadata=retarget(synthetic,delta=(0,0,0))
    np.testing.assert_array_equal(candidate["qpos"],synthetic[1]["qpos"])
    assert metadata["maximum_joint_correction_rad"] == 0


def test_unreachable_geometry_is_explicit_rejection_not_closest_target(synthetic):
    with pytest.raises(RetargetRejection) as caught:retarget(synthetic,delta=(0,0,.01))
    result=caught.value.disposition
    assert result["reason"] == "bounded_upper_body_geometry_unsolved"
    assert result["active_error_m"] > .0005 and result["frames_unattempted"] > 0
    assert result["physical_qualification"] is False


def test_target_and_iteration_bounds_are_enforced(synthetic):
    with pytest.raises(RetargetRejection,match="target_offset_outside_recipe"):
        retarget(synthetic,delta=(.13,0,0))
    with pytest.raises(ValueError):replace(synthetic[2],maximum_iterations=101)
    with pytest.raises(ValueError):replace(synthetic[2],maximum_iterations=True)
    with pytest.raises(ValueError):replace(synthetic[2],maximum_joint_delta_rad=.4)
    with pytest.raises(ValueError):replace(synthetic[2],inactive_position_corridor_m=.1)


def test_tangent_derivatives_recomputed_and_finite(synthetic):
    model,_,_=synthetic;candidate,metadata=retarget(synthetic)
    expected=tangent_velocities(model,candidate["time_s"],candidate["qpos"])
    np.testing.assert_array_equal(candidate["qvel"],expected)
    np.testing.assert_array_equal(candidate["qacc"],np.gradient(expected,candidate["time_s"],axis=0,edge_order=2))
    assert metadata["feedforward_recomputed"] is False
    assert metadata["kinematic_only"] and not metadata["physical_qualification"]


def test_original_feedforward_and_fresh_events_when_finalize_is_invoked(synthetic):
    model,_,_=synthetic;candidate,metadata=retarget(synthetic)
    full,info=finalize_original_feedforward(model,candidate,metadata)
    assert full["torque"].shape == (101,6) and np.isfinite(full["torque"]).all()
    assert info["feedforward_recomputed"] and info["phase_denominator_s"] == 1.5
    assert info["events"]["launch_tick"] % 4 == 0
    assert info["events"]["outward_end_tick"] == 80
    assert info["physical_qualification"] is False and info["kinematic_only"]
    assert "torque" not in candidate and metadata["feedforward_recomputed"] is False


def test_invalid_clock_and_shortened_guard_rejected(synthetic):
    model,source,settings=synthetic
    source={**source,"time_s":source["time_s"]*.5}
    with pytest.raises(ValueError):retarget((model,source,settings))
    source={**synthetic[1],"guard_hold_start":.3}
    with pytest.raises(ValueError):retarget((model,source,settings))


def test_wrong_or_missing_source_hand_cannot_change_the_request(synthetic):
    model,source,settings=synthetic
    with pytest.raises(ValueError,match="source active hand"):
        retarget((model,{**source,"active_hand_index":1},settings))
    absent={k:v for k,v in source.items() if k != "active_hand_index"}
    with pytest.raises(ValueError,match="source active hand"):
        retarget((model,absent,settings))


def test_standalone_recipe_rejects_moving_preparation_and_explicit_connector(synthetic):
    from human_punching.phase2_phases import PhaseSchedule
    model,source,settings=synthetic
    q=source["qpos"].copy();q[3,9]+=.001
    with pytest.raises(RetargetRejection,match="moving_preparation"):
        retarget((model,{**source,"qpos":q},settings))
    marked={**source,"schedule":PhaseSchedule(4,16,80,90,100)}
    with pytest.raises(RetargetRejection,match="preparation_requires"):
        retarget((model,marked,settings))


def test_standalone_finalization_cannot_bypass_moving_prefix_rejection(synthetic):
    model,_,_=synthetic;candidate,metadata=retarget(synthetic)
    candidate["qpos"][3,9]+=.001
    with pytest.raises(RetargetRejection,match="moving_preparation"):
        finalize_original_feedforward(model,candidate,metadata)


def test_finalized_scalar_clock_aliases_match_rederived_actor_events(synthetic):
    model,source,settings=synthetic
    candidate,metadata=retarget_upper_body(model,source,source["head_center"],
        source_outward_start_s=.081,source_outward_end_s=.4,settings=settings)
    assert candidate["launch_time"]==.081
    out,info=finalize_original_feedforward(model,candidate,metadata)
    assert out["launch_time"]==info["events"]["launch_time_s"]==.08
    assert out["return_start"]==info["events"]["outward_completion_time_s"]
    np.testing.assert_array_equal(out["attack_window_reference_s"],[out["launch_time"],out["return_start"]])
