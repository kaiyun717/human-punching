"""Final-reference diagnostics must observe actual FK and separate their units."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.physical_audit import _support_margin, audit_reference

PHYSICAL = Path(__file__).resolve().parents[1] / "outputs/physics-pilot/model/human_physical.xml"


def test_support_halfspace_margin_has_correct_sign_and_scale():
    corners = np.array([[-1., -1.], [-1., 1.], [1., 1.], [1., -1.]])
    assert _support_margin(corners, np.array([0., 0.])) == pytest.approx(1.)
    assert _support_margin(corners, np.array([1., .4])) == pytest.approx(0.)
    assert _support_margin(corners, np.array([1.2, 0.])) == pytest.approx(-.2)
    assert _support_margin(np.zeros((4, 2)), np.zeros(2)) is None


@pytest.fixture
def physical_model():
    if not PHYSICAL.is_file():
        pytest.skip("Generated physical-human model is required")
    return mujoco.MjModel.from_xml_path(str(PHYSICAL))


def test_final_audit_uses_glove_center_and_closed_contact_contract(physical_model):
    model = physical_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    center = data.geom_xpos[model.geom("physical/LeftHand").id].copy()
    assert np.linalg.norm(center - data.body("LeftHand").xpos) > .03
    qpos = np.repeat(data.qpos[None], 2, axis=0)
    inside = audit_reference(model, qpos, [0., .1], center + [.139, 0, 0])
    assert inside["glove_head_distance_min_m"][0] == pytest.approx(.139)
    assert inside["any_sampled_proxy_contact"]
    assert inside["any_sampled_unsafe_clearance"]
    assert inside["first_sampled_contact_time_s"] == 0.
    assert inside["first_linear_center_contact_time_s"] == 0.
    assert inside["contact_center_distance_m"] == .14
    assert inside["safe_center_distance_m"] == .16
    outside = audit_reference(model, qpos, [0., .1], center + [.160001, 0, 0])
    assert not outside["any_sampled_proxy_contact"]
    assert not outside["any_sampled_unsafe_clearance"]
    assert outside["first_sampled_contact_time_s"] is None
    assert outside["first_linear_center_contact_time_s"] is None


def test_audit_finds_cone_violation_and_separates_linear_angular_speed(physical_model):
    model = physical_model
    qpos = np.repeat(model.qpos0[None], 2, axis=0)
    joint = model.joint("LeftHand_joint")
    delta = np.zeros(model.nv)
    delta[int(joint.dofadr[0])] = 2.
    delta[0] = 1.
    mujoco.mj_integratePos(model, qpos[1], delta, 1.)
    before = qpos.copy()
    report = audit_reference(model, qpos, [0., .2], [5., 0, 1.2])
    np.testing.assert_array_equal(qpos, before)
    assert report["cone_worst_joint"] == "LeftHand_joint"
    assert report["cone_violation_max_rad"] == pytest.approx(2. - joint.range[1])
    assert report["cone_limits"]["LeftHand_joint"]["violating_frames"] == 1
    assert report["cone_limits"]["LeftHand_joint"]["maximum_violation_frame"] == 1
    assert report["angular_speed_max_joint"] == "LeftHand_joint"
    assert report["angular_speed_max_rad_s"] == pytest.approx(10.)
    assert report["root_linear_speed_max_m_s"] == pytest.approx(5.)
    np.testing.assert_allclose(report["sole_horizontal_drift_max_m"], [1., 1.], atol=1e-12)


def test_audit_rejects_invalid_times_instead_of_reporting_finite_speeds(physical_model):
    qpos = np.repeat(physical_model.qpos0[None], 2, axis=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        audit_reference(physical_model, qpos, [0., 0.], [0, 0, 1.2])
    qpos[1, 7:11] = 0
    with pytest.raises(ValueError, match="normalized"):
        audit_reference(physical_model, qpos, [0., .1], [0, 0, 1.2])
