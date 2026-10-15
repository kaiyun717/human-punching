"""Physical and anatomical invariants of the scalar SOMA23 derivative."""
import hashlib
import json

import mujoco
import numpy as np
import pytest

from human_punching.scalar_body import VENDOR, build_scalar_human


@pytest.fixture(scope="module")
def scalar(tmp_path_factory):
    metadata = build_scalar_human(tmp_path_factory.mktemp("scalar"))
    model = mujoco.MjModel.from_xml_path(metadata["xml_path"])
    return metadata, model


def test_pinned_source_and_native_scale(scalar):
    meta, model = scalar
    manifest = json.loads((VENDOR / "manifest.json").read_text())
    for name, row in manifest["files"].items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == row["sha256"]
    assert meta["native_scale"]
    assert 1.70 < meta["standing_height_m"] < 1.85
    assert model.body_mass.sum() == pytest.approx(75)
    assert (model.nq, model.nv, model.nu) == (55, 54, 48)


def test_inertia_and_scalar_motor_contract(scalar):
    meta, model = scalar
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    matrix = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, matrix)
    np.linalg.cholesky(matrix)
    moving = model.jnt_bodyid
    assert np.all(model.body_mass[moving] > 0)
    inertia = model.body_inertia[moving]
    assert np.all(inertia > 0)
    assert np.all(2 * inertia.max(axis=1) <= inertia.sum(axis=1) + 1e-12)
    np.testing.assert_array_equal(model.jnt_type[1:], mujoco.mjtJoint.mjJNT_HINGE)
    np.testing.assert_array_equal(model.actuator_trnid[:, 0], np.arange(1, model.njnt))
    np.testing.assert_allclose(model.actuator_gear[:, 0], 1)
    np.testing.assert_allclose(model.actuator_ctrlrange[:, 1], meta["effort"])
    assert np.all(model.actuator_ctrllimited)
    assert np.all(model.actuator_forcelimited)
    assert not np.any(model.jnt_stiffness)
    assert not np.any(model.dof_damping)
    data.ctrl[:] = 10 * np.array(meta["effort"])
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator[:6], 0)
    assert np.all(np.abs(data.qfrc_actuator[6:]) <= np.array(meta["effort"]) + 1e-10)
    np.testing.assert_allclose(data.qfrc_passive, 0)


def test_elbows_flex_forward_and_knees_backward(scalar):
    _, model = scalar
    for side in ("Left", "Right"):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        glove = model.site(f"{side}Hand_center").id
        foot = model.site(f"{side}Foot_sole").id
        initial_glove = data.site_xpos[glove].copy()
        initial_foot = data.site_xpos[foot].copy()
        elbow = model.joint(f"{side}ForeArm_z").id
        data.qpos[model.jnt_qposadr[elbow]] = np.pi / 2 * (1 if side == "Right" else -1)
        knee = model.joint(f"{side}Shin_x").id
        data.qpos[model.jnt_qposadr[knee]] = np.pi / 4
        mujoco.mj_forward(model, data)
        assert data.site_xpos[glove, 0] > initial_glove[0] + .25
        assert data.site_xpos[foot, 0] < initial_foot[0] - .20
        assert model.jnt_range[knee, 0] == 0
        for forbidden in (f"{side}Shin_y", f"{side}Shin_z", f"{side}ForeArm_y"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, forbidden) == -1


def test_sites_glove_orientation_and_selected_collision(scalar):
    meta, model = scalar
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for side in ("Left", "Right"):
        site = model.site(f"{side}Hand_center").id
        geom = model.geom(f"physical/{side}Hand").id
        np.testing.assert_allclose(data.site_xpos[site], data.geom_xpos[geom])
        assert model.geom_size[geom, 0] == pytest.approx(.08)
        before = data.site_xmat[site].copy()
        wrist = model.joint(f"{side}Hand_y").id
        data.qpos[model.jnt_qposadr[wrist]] = .2
        mujoco.mj_forward(model, data)
        assert not np.allclose(before, data.site_xmat[site])
        np.testing.assert_allclose(data.site_xmat[site], data.geom_xmat[geom])
    assert model.npair == len(meta["self_collision_pairs"]) == 13
    floor = model.geom("floor").id
    feet = [model.geom(f"physical/{side}Foot").id for side in ("Left", "Right")]
    assert any(floor in c.geom and any(f in c.geom for f in feet) for c in data.contact)
    assert meta["sole_height_difference_m"] < .001


@pytest.mark.parametrize("timestep", [.002, .005])
def test_short_bounded_physical_step_without_root_forces(tmp_path, timestep):
    meta = build_scalar_human(tmp_path / str(timestep), timestep=timestep)
    model = mujoco.MjModel.from_xml_path(meta["xml_path"])
    data = mujoco.MjData(model)
    reference = data.qpos[7:].copy()
    kp, kd, effort = (np.array(meta[key]) for key in ("kp", "kd", "effort"))
    for _ in range(round(.15 / timestep)):
        data.ctrl[:] = np.clip(kp * (reference - data.qpos[7:]) - kd * data.qvel[6:],
                               -effort, effort)
        mujoco.mj_step(model, data)
        assert np.all(np.isfinite(data.qpos))
        np.testing.assert_allclose(data.qfrc_actuator[:6], 0)
        np.testing.assert_allclose(data.qfrc_applied, 0)
        np.testing.assert_allclose(data.xfrc_applied, 0)
    # This is a compilation/finite integration check, not a standing-balance claim.
    assert data.qpos[2] > .8
