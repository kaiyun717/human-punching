"""Check the mesh/bone scale contract and the MuJoCo kinematic rig."""

import json

import mujoco
import numpy as np
import pytest

from human_punching.human import JOINT_NAMES, PARENTS, SOURCE_TO_WORLD, build_human


def _offsets():
    offsets = np.zeros((21, 3))
    offsets[1] = [0, 0.082, 0]
    offsets[2] = [0, 0.183, 0]
    offsets[3] = [0, 0.203, 0]
    offsets[4] = [0, 0.155, 0.020]
    offsets[5] = [0.04, 0.166, -0.002]
    offsets[6] = [0.112, 0, 0]
    offsets[7] = [0.304, 0, 0]
    offsets[8] = [0.251, 0, 0]
    offsets[9:13] = offsets[5:9] * [-1, 1, 1]
    offsets[13] = [0.1, 0, 0]
    offsets[14] = [0, -0.5, 0]
    offsets[15] = [0, -0.422, 0]
    offsets[16] = [0, -0.065, 0.15]
    offsets[17:21] = offsets[13:17] * [-1, 1, 1]
    return offsets


def _world_mesh_vertices(model, data):
    result = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
            result.append(vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
    return np.concatenate(result)


def test_mesh_height_and_joint_contract(tmp_path):
    source_offsets = _offsets()
    result = build_human(tmp_path, source_offsets)
    model = mujoco.MjModel.from_xml_path(result["xml_path"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vertices = _world_mesh_vertices(model, data)
    assert model.nq == 87
    assert model.nv == 66
    assert model.nmesh == result["mesh_count"] >= 20
    assert vertices[:, 2].min() == pytest.approx(0, abs=2e-7)
    assert np.ptp(vertices[:, 2]) == pytest.approx(1.8, abs=2e-7)
    assert model.nu == 0
    assert np.all(model.geom_contype == 0)
    assert np.all(model.geom_conaffinity == 0)
    np.testing.assert_allclose(model.opt.gravity, 0)
    np.testing.assert_allclose(result["target_offsets"], source_offsets * result["scale"])
    bind = np.zeros((21, 3))
    for j in range(1, 21):
        bind[j] = bind[PARENTS[j]] + source_offsets[j] * result["scale"]
    for j, name in enumerate(JOINT_NAMES):
        np.testing.assert_allclose(
            data.body(name).xpos,
            SOURCE_TO_WORLD @ bind[j] + [0, 0, result["root_height_m"]],
            atol=1e-9,
        )
        if j:
            assert model.joint(f"{name}_joint").qposadr == 7 + (j - 1) * 4
    assert json.loads((tmp_path / "human_metadata.json").read_text())[
        "standing_height_m"
    ] == pytest.approx(1.8)


def test_rejects_bad_morphology(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        build_human(tmp_path, np.full((21, 3), np.nan))
    with pytest.raises(ValueError, match="nonzero"):
        build_human(tmp_path, np.zeros((21, 3)))
    with pytest.raises(ValueError, match="positive"):
        build_human(tmp_path, _offsets(), height_m=0)
