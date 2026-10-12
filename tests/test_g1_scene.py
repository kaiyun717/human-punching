"""Combined-scene contracts checked with native MuJoCo and actual local assets."""

import shutil
from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.g1_scene import create_scene
from human_punching.human import build_human
from human_punching.motion import JOINT_NAMES, load_crop

WORKSPACE = Path(__file__).resolve().parents[2]
DATASET = WORKSPACE / "DuoBox"
MJLAB = WORKSPACE / "mjlab"
pytestmark = pytest.mark.skipif(
    not (DATASET / "fbx_export_joints").is_dir()
    or not (MJLAB / "src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml").is_file(),
    reason="Local DuoBox data and mjlab G1 assets are required",
)


@pytest.fixture
def combined_scene(tmp_path):
    clip = load_crop(DATASET, "jab")
    human = build_human(tmp_path / "human", clip.rest_offsets)
    info = create_scene(Path(human["xml_path"]), tmp_path / "scene", mjlab_root=MJLAB)
    return (
        mujoco.MjModel.from_xml_path(human["xml_path"]),
        mujoco.MjModel.from_xml_path(info["xml_path"]),
        info,
    )


def test_human_coordinates_and_fk_survive_combination(combined_scene):
    human, scene, info = combined_scene
    assert info["human_qpos_indices"] == list(range(human.nq))
    assert info["human_qvel_indices"] == list(range(human.nv))
    assert sorted(info["human_qpos_indices"] + info["g1_qpos_indices"]) == list(range(scene.nq))
    assert sorted(info["human_qvel_indices"] + info["g1_qvel_indices"]) == list(range(scene.nv))
    for joint_id in range(human.njnt):
        name = human.joint(joint_id).name
        combined_id = scene.joint(name).id
        assert combined_id == joint_id
        assert scene.jnt_qposadr[combined_id] == human.jnt_qposadr[joint_id]
        assert scene.jnt_dofadr[combined_id] == human.jnt_dofadr[joint_id]
        assert scene.jnt_type[combined_id] == human.jnt_type[joint_id]
    hdata, sdata = mujoco.MjData(human), mujoco.MjData(scene)
    tangent = np.zeros(human.nv)
    tangent[:6] = [0.2, -0.1, 0.07, 0.1, 0.2, -0.25]
    wrist_adr = int(human.joint("LeftHand_joint").dofadr[0])
    tangent[wrist_adr : wrist_adr + 3] = [0.2, 0.5, -0.1]
    mujoco.mj_integratePos(human, hdata.qpos, tangent, 1.0)
    sdata.qpos[:] = info["qpos_reset"]
    sdata.qpos[info["human_qpos_indices"]] = hdata.qpos
    mujoco.mj_forward(human, hdata)
    mujoco.mj_forward(scene, sdata)
    for name in JOINT_NAMES:
        assert human.body(name).id == scene.body(name).id
        np.testing.assert_allclose(sdata.body(name).xpos, hdata.body(name).xpos, atol=1e-12)
        np.testing.assert_allclose(sdata.body(name).xmat, hdata.body(name).xmat, atol=1e-12)


def test_standing_keyframe_head_proxy_and_disabled_physics(combined_scene):
    _, scene, info = combined_scene
    data = mujoco.MjData(scene)
    mujoco.mj_resetDataKeyframe(scene, data, scene.key("standing").id)
    mujoco.mj_forward(scene, data)
    np.testing.assert_allclose(data.qpos, info["qpos_reset"], atol=1e-11, rtol=0)
    pelvis = data.body("g1/pelvis")
    # Human motion is initially aligned toward +X; G1 faces back toward -X.
    np.testing.assert_allclose(pelvis.xmat.reshape(3, 3)[:, 0], [-1, 0, 0], atol=1e-12)
    # Independent expected values from the accepted asset's configured stance.
    expected = {
        "left_hip_pitch_joint": -0.312,
        "right_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363,
        "right_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "waist_yaw_joint": 0.0,
    }
    for name, value in expected.items():
        address = int(scene.joint("g1/" + name).qposadr[0])
        assert data.qpos[address] == pytest.approx(value, abs=1e-12)
    head = scene.geom("g1/head_collision")
    assert int(head.type[0]) == mujoco.mjtGeom.mjGEOM_SPHERE
    assert head.size[0] == pytest.approx(0.06)
    np.testing.assert_allclose(head.pos, [0, 0, 0.43], atol=1e-12)
    np.testing.assert_allclose(data.geom_xpos[head.id], info["head_center_w"], atol=1e-11)
    assert info["robot_visual_sole_z_m"] == pytest.approx(0, abs=1e-10)
    assert scene.nu == 0
    np.testing.assert_array_equal(scene.opt.gravity, np.zeros(3))
    assert not np.any(scene.geom_contype)
    assert not np.any(scene.geom_conaffinity)


def test_moving_human_leaves_every_robot_body_fixed(combined_scene):
    human, scene, info = combined_scene
    data = mujoco.MjData(scene)
    data.qpos[:] = info["qpos_reset"]
    mujoco.mj_forward(scene, data)
    robot_bodies = info["entities"]["g1"]["body_ids"]
    initial_positions = data.xpos[robot_bodies].copy()
    initial_orientations = data.xmat[robot_bodies].copy()
    initial_robot_qpos = data.qpos[info["g1_qpos_indices"]].copy()
    original_human = data.body("Hips").xpos.copy()
    velocity = np.linspace(-0.35, 0.45, human.nv)
    changed_pose = data.qpos[info["human_qpos_indices"]].copy()
    mujoco.mj_integratePos(human, changed_pose, velocity, 0.5)
    data.qpos[info["human_qpos_indices"]] = changed_pose
    data.qvel[info["human_qvel_indices"]] = velocity
    mujoco.mj_forward(scene, data)
    assert np.linalg.norm(data.body("Hips").xpos - original_human) > 0.1
    np.testing.assert_array_equal(data.qpos[info["g1_qpos_indices"]], initial_robot_qpos)
    np.testing.assert_array_equal(data.xpos[robot_bodies], initial_positions)
    np.testing.assert_array_equal(data.xmat[robot_bodies], initial_orientations)
    np.testing.assert_array_equal(data.qvel[info["g1_qvel_indices"]], 0)


def test_generated_scene_assets_can_be_relocated(combined_scene, tmp_path):
    _, original, info = combined_scene
    source = Path(info["xml_path"]).parent
    relocated = tmp_path / "relocated_scene"
    shutil.move(str(source), relocated)
    model = mujoco.MjModel.from_xml_path(str(relocated / "scene.xml"))
    assert model.nmesh == original.nmesh
    assert model.nq == original.nq
    assert (relocated / "MJLAB-LICENSE").is_file()
