from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from human_punching.phase2_targets import (
    MAX_YAW_RAD, Placement, calibrate_registration, pelvis_yaw,
    placement_candidates, register_snapshot, yaw_matrix,
)


def state(yaw=.4):
    q=np.r_[[12.,-8.,.75],math.cos(yaw/2),0.,0.,math.sin(yaw/2),np.arange(29)/100]
    v=np.arange(35)/100
    head=q[:3]+yaw_matrix(yaw)@np.array([.03,-.01,.48])
    return q,v,head,np.array([.7,.4,-.2])


def registration():
    q,v,head,velocity=state()
    return calibrate_registration(calibration_qpos=q,calibration_head_w=head,
        calibration_payload_sha256="a"*64,original_target_w=(.85,.2,1.24))


def transform(reg=None,placement=Placement(),values=None):
    q,v,head,velocity=state() if values is None else values
    return register_snapshot(reg or registration(),qpos=q,qvel=v,head_position_w=head,
        head_velocity_w=velocity,placement=placement)


def test_calibration_fits_horizontal_position_only():
    result=transform()
    np.testing.assert_allclose(result["head_target_w"][:2],[.85,.2],atol=1e-14)
    assert result["head_target_w"][2] == 1.23
    assert result["head_target_w"][2] != registration().original_target_w[2]
    assert result["root_translation_z_m"] == 0.


def test_full_root_and_head_receive_same_fixed_rigid_placement():
    q,v,head,velocity=state()
    result=transform(placement=Placement(.05,-.05,MAX_YAW_RAD))
    rotation=np.array(result["rotation_world_to_registered"])
    translation=np.array(result["translation_world_to_registered"])
    np.testing.assert_allclose(result["head_target_w"],rotation@head+translation)
    np.testing.assert_allclose(result["qpos_registered"][:3],rotation@q[:3]+translation)
    np.testing.assert_allclose(np.array(result["head_target_w"])-result["qpos_registered"][:3],rotation@(head-q[:3]))
    np.testing.assert_allclose(np.array(result["qpos_registered"][:2])-registration().root_xy_anchor,[.05,-.05])


def test_true_head_velocity_rotates_without_root_velocity_subtraction():
    q,v,head,velocity=state()
    result=transform()
    rotation=np.array(result["rotation_world_to_registered"])
    np.testing.assert_allclose(result["head_point_velocity_w"],rotation@velocity)
    assert not np.allclose(result["head_point_velocity_w"],rotation@(velocity-v[:3]))
    assert result["head_point_velocity_w"][2] == velocity[2]


def test_native_body_angular_and_joint_coordinates_remain_unchanged():
    q,v,_,_=state()
    result=transform()
    np.testing.assert_array_equal(result["qpos_registered"][7:],q[7:])
    np.testing.assert_array_equal(result["qvel_registered"][3:],v[3:])
    np.testing.assert_allclose(result["qvel_registered"][:3],yaw_matrix(math.pi-.4)@v[:3])
    assert abs(abs(pelvis_yaw(result["qpos_registered"][3:7]))-math.pi) < 1e-14


def test_registration_is_equivariant_to_arbitrary_bank_origin_and_heading():
    a=state(.4); b=state(-1.2)
    q,v,head,velocity=b
    shift=np.array([-4.,6.,0.]);q[:3]+=shift;head+=shift
    # Describe the same physical velocities after the arbitrary world yaw.
    v[:3]=yaw_matrix(-1.6)@a[1][:3]
    velocity=yaw_matrix(-1.6)@a[3]
    first,second=transform(values=a),transform(values=(q,v,head,velocity))
    for key in ("head_target_w","head_point_velocity_w","qvel_registered"):
        np.testing.assert_allclose(first[key],second[key],atol=1e-14)
    np.testing.assert_allclose(first["qpos_registered"][:3],second["qpos_registered"][:3],atol=1e-14)
    np.testing.assert_allclose(abs(np.array(first["qpos_registered"][3:7])),abs(np.array(second["qpos_registered"][3:7])),atol=1e-14)


def test_snapshot_head_height_and_articulation_variations_are_retained():
    q,v,head,velocity=state(); head += [.05,0.,.02]
    result=transform(values=(q,v,head,velocity))
    nominal=np.array(transform()["head_target_w"])
    np.testing.assert_allclose(np.array(result["head_target_w"])-nominal,yaw_matrix(math.pi-.4)@np.array([.05,0.,.02]))
    assert result["registered_head_height_m"] == 1.25


def test_candidate_grid_is_complete_unique_and_nominal_first():
    rows=placement_candidates()
    assert len(rows) == len(set(rows)) == 27 and rows[0] == Placement()
    assert {p.forward_m for p in rows} == {-.05,0.,.05}
    assert {p.left_m for p in rows} == {-.05,0.,.05}
    assert {p.yaw_rad for p in rows} == {-MAX_YAW_RAD,0.,MAX_YAW_RAD}


@pytest.mark.parametrize("args",[(.050001,0,0),(0,-.050001,0),(0,0,MAX_YAW_RAD+1e-9),(float("nan"),0,0),(0,0,float("inf"))])
def test_unsupported_placements_are_rejected_without_clipping(args):
    with pytest.raises(ValueError):Placement(*args)


def test_immutable_registration_and_inputs_are_copied():
    q,v,head,velocity=state()
    reg=calibrate_registration(calibration_qpos=q,calibration_head_w=head,
        calibration_payload_sha256="a"*64,original_target_w=(.85,.2,1.24))
    original=reg.identity; q[0]=123.;head[0]=456.
    assert reg.identity == original and reg.calibration_root_qpos[0] == 12.
    with pytest.raises(FrozenInstanceError):reg.opponent_heading_rad=0
    with pytest.raises(FrozenInstanceError):Placement().forward_m=.1


def test_calibration_identity_and_derived_anchor_cannot_be_replaced():
    reg=registration()
    for kwargs in ({"calibration_slot_id":52913},{"calibration_command_id":2},
                   {"calibration_replica_id":1},{"opponent_heading_rad":0.},
                   {"calibration_payload_sha256":"z"*64},{"root_xy_anchor":(0.,0.)}):
        with pytest.raises(ValueError):replace(reg,**kwargs)


def test_invalid_saved_states_are_rejected_without_normalizing_bad_root():
    q,v,head,velocity=state();q[3:7] *= 2
    with pytest.raises(ValueError):transform(values=(q,v,head,velocity))
    q,v,head,velocity=state();velocity[1]=float("nan")
    with pytest.raises(ValueError):transform(values=(q,v,head,velocity))
    with pytest.raises(ValueError):transform(values=(q[:-1],v,head,np.zeros(3)))


def test_geometry_cannot_claim_execution_or_qualification():
    result=transform()
    assert result["physics_executed"] is False
    assert result["physical_qualification"] is False
    assert result["live_g1_actor"] is False
