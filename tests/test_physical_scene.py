"""The baked standing robot must not alter the human's dynamics or state layout."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.physical_scene import attach_static_g1

PROJECT = Path(__file__).resolve().parents[1]
PHYSICAL = PROJECT / "outputs/physics-pilot/model/human_physical.xml"
STANDING = PROJECT / "outputs/g1-targeting/scene/scene.xml"
pytestmark = pytest.mark.skipif(
    not PHYSICAL.is_file() or not STANDING.is_file(),
    reason="Generated physical-human and reviewed standing-G1 assets are required",
)


@pytest.fixture(scope="module")
def fixed_scene(tmp_path_factory):
    output = tmp_path_factory.mktemp("physical-static-g1")
    metadata = attach_static_g1(PHYSICAL, output, standing_scene_xml=STANDING)
    return metadata, mujoco.MjModel.from_xml_path(metadata["xml_path"])


def test_fixed_robot_matches_standing_body_and_visual_fk(fixed_scene):
    info, combined = fixed_scene
    reference = mujoco.MjModel.from_xml_path(str(STANDING))
    original_data = mujoco.MjData(reference)
    mujoco.mj_resetDataKeyframe(reference, original_data, reference.key("standing").id)
    mujoco.mj_forward(reference, original_data)
    data = mujoco.MjData(combined)
    mujoco.mj_forward(combined, data)
    for name in info["g1_body_names"]:
        body, original = combined.body(name).id, reference.body(name).id
        assert combined.body_jntnum[body] == 0
        np.testing.assert_allclose(data.xpos[body], original_data.xpos[original], atol=1e-10)
        np.testing.assert_allclose(data.xmat[body], original_data.xmat[original], atol=1e-10)
        assert combined.body_geomnum[body] == reference.body_geomnum[original]
        count = int(combined.body_geomnum[body])
        geoms = np.arange(combined.body_geomadr[body], combined.body_geomadr[body] + count)
        old_geoms = np.arange(reference.body_geomadr[original], reference.body_geomadr[original] + count)
        np.testing.assert_allclose(data.geom_xpos[geoms], original_data.geom_xpos[old_geoms], atol=1e-10)
        np.testing.assert_allclose(data.geom_xmat[geoms], original_data.geom_xmat[old_geoms], atol=1e-10)
        assert not np.any(combined.geom_contype[geoms])
        assert not np.any(combined.geom_conaffinity[geoms])
    head = combined.geom("g1/head_collision").id
    np.testing.assert_allclose(data.geom_xpos[head], [.8539635, .2, 1.231490745258], atol=1e-10)
    assert combined.geom_size[head, 0] == pytest.approx(.06)
    assert info["g1_dynamic_dofs"] == 0
    assert not info["g1_contact_response"]


def test_human_mass_actuation_and_freefall_survive_fixed_robot_attachment(fixed_scene):
    info, combined = fixed_scene
    human = mujoco.MjModel.from_xml_path(str(PHYSICAL))
    assert (combined.nq, combined.nv, combined.nu) == (human.nq, human.nv, human.nu) == (87, 66, 60)
    assert info["human_mass_kg"] == pytest.approx(75.)
    np.testing.assert_allclose(combined.body_mass[:human.nbody], human.body_mass)
    np.testing.assert_allclose(combined.body_inertia[:human.nbody], human.body_inertia)
    np.testing.assert_array_equal(combined.jnt_type, human.jnt_type)
    np.testing.assert_array_equal(combined.jnt_qposadr, human.jnt_qposadr)
    np.testing.assert_array_equal(combined.jnt_dofadr, human.jnt_dofadr)
    np.testing.assert_allclose(combined.actuator_gear, human.actuator_gear)
    np.testing.assert_allclose(combined.actuator_forcerange, human.actuator_forcerange)
    np.testing.assert_allclose(combined.opt.gravity, human.opt.gravity)
    np.testing.assert_array_equal(combined.geom_contype[:human.ngeom], human.geom_contype)
    np.testing.assert_array_equal(combined.geom_conaffinity[:human.ngeom], human.geom_conaffinity)
    hdata, cdata = mujoco.MjData(human), mujoco.MjData(combined)
    hdata.qpos[2] += 2.
    cdata.qpos[:] = hdata.qpos
    for model, data in ((human, hdata), (combined, cdata)):
        data.ctrl[:] = np.linspace(-.2, .2, model.nu)
        mujoco.mj_forward(model, data)
    robot_ids = [combined.body(name).id for name in info["g1_body_names"]]
    initial_robot_pos, initial_robot_rot = cdata.xpos[robot_ids].copy(), cdata.xmat[robot_ids].copy()
    initial_height = cdata.qpos[2]
    for _ in range(50):
        mujoco.mj_step(human, hdata)
        mujoco.mj_step(combined, cdata)
    assert cdata.qpos[2] < initial_height - .04
    np.testing.assert_allclose(cdata.qpos, hdata.qpos, atol=1e-11, rtol=0)
    np.testing.assert_allclose(cdata.qvel, hdata.qvel, atol=1e-11, rtol=0)
    np.testing.assert_array_equal(cdata.xpos[robot_ids], initial_robot_pos)
    np.testing.assert_array_equal(cdata.xmat[robot_ids], initial_robot_rot)
    np.testing.assert_allclose(cdata.qfrc_actuator[:6], 0, atol=1e-12)


def test_fixed_scene_is_portable_with_copied_assets_and_license(fixed_scene, tmp_path):
    info, original = fixed_scene
    source = Path(info["xml_path"]).parent
    moved = tmp_path / "relocated"
    shutil.copytree(source, moved)
    # Check the XML does not retain absolute asset paths into either input tree.
    xml = ET.parse(moved / "physical_scene.xml").getroot()
    assert xml.find("compiler").get("meshdir") == "meshes"
    for mesh in xml.findall("asset/mesh"):
        assert not Path(mesh.get("file")).is_absolute()
        assert (moved / "meshes" / mesh.get("file")).is_file()
    assert (moved / "MJLAB-LICENSE").is_file()
    reloaded = mujoco.MjModel.from_xml_path(str(moved / "physical_scene.xml"))
    assert reloaded.nmesh == original.nmesh
    assert reloaded.nq == original.nq
