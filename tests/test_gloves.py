"""Check actual rendered glove transforms under wrist motion and DuoBox replay."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from human_punching.human import build_human
from human_punching.motion import CROP_SPECS, load_crop
from human_punching.retarget import retarget_clip, wxyz

DATASET = Path(__file__).resolve().parents[2] / "DuoBox"


def _synthetic_offsets():
    # A T-pose rig suffices: this test must also run without the source dataset.
    offsets = np.zeros((21, 3))
    offsets[1:5] = [[0, 0.082, 0], [0, 0.183, 0], [0, 0.203, 0], [0, 0.155, 0.020]]
    offsets[5:9] = [[0.04, 0.166, -0.002], [0.112, 0, 0], [0.304, 0, 0], [0.251, 0, 0]]
    offsets[9:13] = offsets[5:9] * [-1, 1, 1]
    offsets[13:17] = [[0.1, 0, 0], [0, -0.5, 0], [0, -0.422, 0], [0, -0.065, 0.15]]
    offsets[17:21] = offsets[13:17] * [-1, 1, 1]
    return offsets


def _glove_geoms(model, side):
    # Include cuff and orientation patch as well as the main glove surface.
    geoms = [
        geom
        for geom in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom))
        and name.startswith(f"{side}_")
        and "glove" in name
    ]
    assert len(geoms) >= 2
    assert all(model.geom_bodyid[geom] == model.body(f"{side}Hand").id for geom in geoms)
    return geoms


def _vertices_world(model, data, geom):
    mesh = model.geom_dataid[geom]
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    return vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


@pytest.fixture(scope="module")
def synthetic_model(tmp_path_factory):
    info = build_human(tmp_path_factory.mktemp("glove-roll"), _synthetic_offsets())
    return mujoco.MjModel.from_xml_path(info["xml_path"])


@pytest.mark.parametrize("side", ["Left", "Right"])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_glove_vertices_follow_isolated_wrist_rotation(side, axis, synthetic_model):
    """A fixed forearm must not stop the glove following a moving wrist joint."""
    model = synthetic_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hand = model.body(f"{side}Hand").id
    forearm = model.body(f"{side}ForeArm").id
    geoms = _glove_geoms(model, side)
    hand_pos = data.xpos[hand].copy()
    hand_rot = data.xmat[hand].reshape(3, 3).copy()
    forearm_rot = data.xmat[forearm].copy()
    vertices_before = {geom: _vertices_world(model, data, geom) for geom in geoms}
    wrist_rotation = Rotation.from_euler(axis, 90, degrees=True).as_matrix()
    address = int(model.joint(f"{side}Hand_joint").qposadr[0])
    data.qpos[address : address + 4] = wxyz(wrist_rotation)
    mujoco.mj_forward(model, data)
    world_delta = hand_rot @ wrist_rotation @ hand_rot.T
    np.testing.assert_allclose(data.xpos[hand], hand_pos, atol=1e-12)
    np.testing.assert_allclose(data.xmat[forearm], forearm_rot, atol=1e-12)
    np.testing.assert_allclose(data.xmat[hand].reshape(3, 3), hand_rot @ wrist_rotation, atol=1e-12)
    for geom in geoms:
        expected = (vertices_before[geom] - hand_pos) @ world_delta.T + hand_pos
        np.testing.assert_allclose(_vertices_world(model, data, geom), expected, atol=1e-12)
        assert np.max(np.linalg.norm(expected - vertices_before[geom], axis=-1)) > 0.02


@pytest.fixture(scope="module")
def source_model(tmp_path_factory):
    if not DATASET.is_dir():
        pytest.skip("DuoBox data not installed")
    clip = load_crop(DATASET, "jab")
    info = build_human(tmp_path_factory.mktemp("glove-source"), clip.rest_offsets)
    return mujoco.MjModel.from_xml_path(info["xml_path"]), info


@pytest.mark.parametrize("punch", CROP_SPECS)
def test_gloves_follow_source_hand_orientation_on_every_frame(punch, source_model):
    """Position-only FK tests cannot detect dropping a terminal hand rotation."""
    model, info = source_model
    clip = load_crop(DATASET, punch, padding_s=0.25)
    result = retarget_clip(clip, model, info["scale"])
    align = Rotation.from_euler("z", result["metrics"]["heading_rotation_rad"]).as_matrix()
    data = mujoco.MjData(model)
    hands = []
    for side, index in [("Left", 8), ("Right", 12)]:
        geoms = _glove_geoms(model, side)
        geom_local_rotations = {
            geom: Rotation.from_quat(np.roll(model.geom_quat[geom], -1)).as_matrix()
            for geom in geoms
        }
        hands.append((model.body(f"{side}Hand").id, index, geoms, geom_local_rotations))
    for frame, pose in enumerate(result["trajectory"]["qpos"]):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for body, index, geoms, local_rotations in hands:
            expected = align @ clip.global_rotations[frame, index]
            actual = data.xmat[body].reshape(3, 3)
            error = Rotation.from_matrix(actual @ expected.T).magnitude()
            assert np.rad2deg(error) < 1e-6
            np.testing.assert_allclose(actual, expected, atol=3e-6)
            for geom in geoms:
                np.testing.assert_allclose(
                    data.geom_xmat[geom].reshape(3, 3),
                    expected @ local_rotations[geom],
                    atol=3e-6,
                )
                np.testing.assert_allclose(
                    data.geom_xpos[geom],
                    data.xpos[body] + expected @ model.geom_pos[geom],
                    atol=3e-7,
                )
