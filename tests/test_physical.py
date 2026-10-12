"""Dynamics and control checks for the source-compatible physical-human pilot."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.human import build_human
from human_punching.motion import JOINT_NAMES, load_crop
from human_punching.physical import (
    bounded_pd,
    build_physical_human,
    controller_arrays,
    plant_feet,
    sole_targets,
)
from human_punching.retarget import retarget_clip

DATASET = Path(__file__).resolve().parents[2] / "DuoBox"


def _test_offsets():
    """Independent synthetic proportions; these tests do not require mocap."""
    offsets = np.zeros((21, 3))
    offsets[1:5] = [[0, .082, 0], [0, .183, 0], [0, .203, 0], [0, .155, .02]]
    offsets[5:9] = [[.04, .166, -.002], [.112, 0, 0], [.304, 0, 0], [.251, 0, 0]]
    offsets[9:13] = offsets[5:9] * [-1, 1, 1]
    offsets[13:17] = [[.1, 0, 0], [0, -.5, 0], [0, -.422, 0], [0, -.065, .15]]
    offsets[17:21] = offsets[13:17] * [-1, 1, 1]
    return offsets


@pytest.fixture(scope="module")
def physical_model(tmp_path_factory):
    output = tmp_path_factory.mktemp("physical-human")
    visual = build_human(output / "visual", _test_offsets())
    metadata = build_physical_human(Path(visual["xml_path"]), output / "physical")
    return mujoco.MjModel.from_xml_path(metadata["xml_path"])


def test_physical_model_has_consistent_mass_inertia_and_unactuated_root(physical_model):
    model = physical_model
    assert model.body_mass.sum() == pytest.approx(75.)
    assert np.all(model.body_mass[1:] > 0)
    inertia = model.body_inertia[1:]
    assert np.all(np.isfinite(inertia)) and np.all(inertia > 0)
    # Every principal inertia must satisfy the rigid-body triangle inequality.
    assert np.all(2 * inertia.max(axis=1) <= inertia.sum(axis=1) + 1e-12)
    assert (model.nq, model.nv, model.nu) == (87, 66, 60)
    root_joint = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    assert root_joint.tolist() == [0]
    assert not np.any(model.actuator_trnid[:, 0] == root_joint[0])
    assert model.neq == 0
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -9.81])
    # Build the actual generalized actuator map, not just an XML-name check.
    data = mujoco.MjData(model)
    data.qpos[2] += 3.
    data.ctrl[:] = .3
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator[:6], 0, atol=1e-12)
    assert np.linalg.norm(data.qfrc_actuator[6:]) > 0


def test_unpowered_rig_falls_under_gravity_without_pose_overwrite(physical_model):
    model = physical_model
    data = mujoco.MjData(model)
    data.qpos[2] += 3.
    initial = data.qpos.copy()
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    np.testing.assert_allclose(data.qacc[:3], [0, 0, -9.81], atol=1e-9)
    for _ in range(50):
        mujoco.mj_step(model, data)
    assert data.ncon == 0
    elapsed = float(data.time)
    assert data.qpos[2] < initial[2] - .04
    assert data.qpos[2] == pytest.approx(initial[2] - .5 * 9.81 * elapsed**2, abs=.002)
    np.testing.assert_allclose(data.qfrc_applied, 0)
    np.testing.assert_allclose(data.xfrc_applied, 0)
    np.testing.assert_allclose(data.qpos[3:], initial[3:], atol=1e-9)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_quaternion_pd_torque_reduces_a_small_wrist_error(physical_model, axis):
    model = physical_model
    data = mujoco.MjData(model)
    data.qpos[2] += 3.
    mujoco.mj_forward(model, data)
    joint = model.joint("LeftHand_joint")
    dof = int(joint.dofadr[0]) + axis
    perturbation = np.zeros(model.nv)
    perturbation[dof] = .03
    target = data.qpos.copy()
    mujoco.mj_integratePos(model, target, perturbation, 1.)
    bounded_pd(model, data, target, np.zeros(model.nv), stable=False)
    mujoco.mj_forward(model, data)
    assert data.qfrc_actuator[dof] > 0
    assert data.qacc[dof] > 0
    error_before = np.zeros(model.nv)
    error_after = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, error_before, 1., data.qpos, target)
    mujoco.mj_step(model, data)
    mujoco.mj_differentiatePos(model, error_after, 1., data.qpos, target)
    assert abs(error_after[dof]) < abs(error_before[dof])
    np.testing.assert_allclose(data.qfrc_actuator[:6], 0, atol=1e-12)


def test_root_reference_cannot_create_actuator_support(physical_model):
    model = physical_model
    data = mujoco.MjData(model)
    data.qpos[2] += 3.
    mujoco.mj_forward(model, data)
    target = data.qpos.copy()
    displacement = np.zeros(model.nv)
    displacement[:6] = [2, -1, 4, .1, -.1, .1]
    mujoco.mj_integratePos(model, target, displacement, 1.)
    bounded_pd(model, data, target, np.zeros(model.nv), stable=False)
    np.testing.assert_allclose(data.ctrl, 0, atol=1e-12)
    np.testing.assert_allclose(data.qfrc_applied, 0)
    np.testing.assert_allclose(data.xfrc_applied, 0)


@pytest.mark.parametrize("balance", [False, True])
def test_pd_enforces_joint_torque_vector_bounds(physical_model, balance):
    model = physical_model
    data = mujoco.MjData(model)
    data.qpos[2] += 3.
    data.qvel[6:] = np.linspace(-40, 40, model.nv - 6)
    mujoco.mj_forward(model, data)
    target = model.qpos0.copy()
    target[2] += 3.
    saturated, demand_ratio = bounded_pd(
        model, data, target, np.zeros(model.nv), balance=balance
    )
    _, _, limits = controller_arrays(model)
    assert saturated > 0 and demand_ratio > 1
    assert np.isfinite(data.ctrl).all()
    assert np.all(np.linalg.norm(data.ctrl.reshape(-1, 3), axis=1) <= limits + 1e-10)
    mujoco.mj_forward(model, data)
    assert np.all(
        np.linalg.norm(data.qfrc_actuator[6:].reshape(-1, 3), axis=1) <= limits + 1e-10
    )
    np.testing.assert_allclose(data.qfrc_actuator[:6], 0, atol=1e-12)


@pytest.mark.skipif(not DATASET.is_dir(), reason="DuoBox data not installed")
def test_real_guard_plants_soles_and_survives_brief_free_root_hold(tmp_path):
    clip = load_crop(DATASET, "jab", padding_s=.25)
    visual_info = build_human(tmp_path / "visual", clip.rest_offsets)
    visual = mujoco.MjModel.from_xml_path(visual_info["xml_path"])
    metadata = build_physical_human(Path(visual_info["xml_path"]), tmp_path / "physical")
    model = mujoco.MjModel.from_xml_path(metadata["xml_path"])
    source = retarget_clip(clip, visual, visual_info["scale"])["trajectory"]["qpos"][:3]
    saved_source = source.copy()
    targets = sole_targets(model, source[0])
    planted, metrics = plant_feet(model, source)
    np.testing.assert_array_equal(source, saved_source)
    assert metrics["foot_position_max_error_m"] < .002
    assert metrics["com_horizontal_max_error_m"] < .005
    # Upper-body local rotations are preserved while root and legs solve support.
    for name in JOINT_NAMES[1:13]:
        address = int(model.joint(f"{name}_joint").qposadr[0])
        np.testing.assert_array_equal(planted[:, address:address + 4], source[:, address:address + 4])
    data = mujoco.MjData(model)
    for pose in planted:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for site, body, pos, rotation in targets:
            np.testing.assert_allclose(data.site_xpos[site], pos, atol=.002)
            np.testing.assert_allclose(data.xmat[body].reshape(3, 3)[:, 1], rotation[:, 1], atol=.01)
    data = mujoco.MjData(model)
    data.qpos[:] = planted[0]
    mujoco.mj_forward(model, data)
    touched = False
    for _ in range(100):
        bounded_pd(model, data, planted[0], np.zeros(model.nv))
        mujoco.mj_step(model, data)
        touched |= data.ncon > 0
    assert touched
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert abs(data.qpos[2] - planted[0, 2]) < .05
    assert np.linalg.norm(data.qpos[:2] - planted[0, :2]) < .05
    np.testing.assert_allclose(data.qfrc_applied, 0)
    np.testing.assert_allclose(data.xfrc_applied, 0)
    # A 0.2 s hold is a controller/contact smoke test, not proof of standing skill.
