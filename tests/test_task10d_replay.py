"""CPU contracts for prescribed H2 in the actual combined mjlab Scene.

These are geometry/ownership checks, without dynamic rollouts or GPU execution.
"""
from dataclasses import FrozenInstanceError
import hashlib
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from human_punching.task10c.robot import H2_XML
from human_punching.task10d.replay import (
    APPROVED_JAB_SHA256, CommittedReplay, ENCOUNTER_RADIUS, GLOVE_GEOM_NAMES,
    PHYSICS_DT, Placement, PrescribedH2,
)


@pytest.fixture(scope="module")
def reference():
    return PrescribedH2()


@pytest.fixture(scope="module")
def combined(reference):
    from mjlab.scene import Scene
    from human_punching.task10d.g1 import make_task08_cfg
    cfg = make_task08_cfg(extra_entities={"h2": reference.entity_cfg()})
    model = Scene(cfg.scene, device="cpu").compile()
    cfg.sim.mujoco.apply(model)
    return model, reference.bind(model)


def visible_geom_pairs(reference, model):
    """Map native and flattened geoms by their body-local XML order/name."""
    pairs = []
    for body in ET.parse(H2_XML).iter("body"):
        name = body.attrib["name"]
        native_body = reference.model.body(name).id
        local_ids = np.flatnonzero(reference.model.geom_bodyid == native_body)
        for local_index, geom in enumerate(body.findall("geom")):
            if geom.get("class") != "visual" and geom.get("name") not in GLOVE_GEOM_NAMES:
                continue
            scene_name = "h2/" + geom.get("name", f"{name}_visual_{local_index}")
            pairs.append((int(local_ids[local_index]), model.geom(scene_name).id))
    return np.asarray(pairs)


def test_unchanged_reference_knots_and_immutable_source(reference):
    assert reference.sha256 == APPROVED_JAB_SHA256
    for index in (0, 17, 34, len(reference.times) - 1):
        sample = reference.sample(float(reference.times[index]))
        np.testing.assert_allclose(sample.qpos, reference.qposes[index], atol=1e-15)
        np.testing.assert_allclose(sample.body_pos,
                                   reference.reference.arrays["body_pos_w"][index], atol=1e-14)
        np.testing.assert_allclose(sample.glove_pos,
                                   reference.reference.arrays["glove_pos_w"][index], atol=1e-14)
    with pytest.raises(ValueError, match="read-only"):
        reference.qposes[0, 0] = 100
    with pytest.raises(ValueError, match="read-only"):
        reference.reference.arrays["qpos"][0, 0] = 100
    assert hashlib.sha256(reference.path.read_bytes()).hexdigest() == APPROVED_JAB_SHA256


def test_every_visible_geom_retains_native_fk_and_attachment(reference, combined):
    model, bound = combined
    data = mujoco.MjData(model)
    native_ids, scene_ids = visible_geom_pairs(reference, model).T
    # Nontrivial registration catches root-only transforms and double offsets.
    placement = Placement((0.23, -0.19), 0.37)
    for time in (0.0, 0.005, 0.5098774175291865, 0.68, 1.155, 1.42):
        sample = reference.sample(time, placement)
        bound.write(sample, data)
        mujoco.mj_kinematics(model, data)
        np.testing.assert_allclose(data.xpos[bound.body_ids], sample.body_pos, atol=1e-14)
        np.testing.assert_allclose(data.geom_xpos[bound.glove_geom_ids], sample.glove_pos, atol=1e-14)
        np.testing.assert_allclose(data.geom_xpos[scene_ids],
            placement.transform_points(reference.data.geom_xpos[native_ids]), atol=1e-14)
        expected_rot = placement.rotation * Rotation.from_matrix(
            reference.data.geom_xmat[native_ids].reshape(-1, 3, 3))
        np.testing.assert_allclose(data.geom_xmat[scene_ids].reshape(-1, 3, 3),
                                   expected_rot.as_matrix(), atol=1e-14)
    np.testing.assert_array_equal(model.geom_type[scene_ids], reference.model.geom_type[native_ids])
    np.testing.assert_array_equal(model.geom_size[scene_ids], reference.model.geom_size[native_ids])
    assert len(scene_ids) == 34  # One mesh for each of 32 links plus two gloves.


def test_combined_entity_adds_no_dofs_or_contact_and_is_visible(reference, combined):
    from human_punching.task10d.g1 import compiled_baseline
    model, bound = combined
    baseline = compiled_baseline()
    for key in ("nq", "nv", "nu", "njnt", "neq", "ntendon", "npair", "nexclude"):
        assert getattr(model, key) == getattr(baseline, key)
    assert len(bound.body_ids) == model.nmocap == 32
    assert np.all(model.body_parentid[bound.body_ids] == 0)
    ids = np.flatnonzero(np.isin(model.geom_bodyid, bound.body_ids))
    assert not np.any(model.geom_contype[ids])
    assert not np.any(model.geom_conaffinity[ids])
    data = mujoco.MjData(model)
    sample = reference.sample(0.68)
    bound.write(sample, data)
    mujoco.mj_forward(model, data)
    for contact in data.contact:
        assert not set(contact.geom) & set(ids)
    # MuJoCo ray geometry remains present despite contype/conaffinity=0.
    groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    result = np.array([-1], dtype=np.int32)
    origin = sample.glove_pos[0] + [0, 0, 0.12]
    distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]),
                              groups, True, -1, result)
    assert result[0] == bound.glove_geom_ids[0]
    assert distance == pytest.approx(0.04, abs=1e-12)


def test_mocap_writer_does_not_touch_g1_or_unselected_worlds(reference, combined):
    import torch
    model, bound = combined
    data = mujoco.MjData(model)
    data.qvel[:] = np.arange(model.nv) / 19
    data.ctrl[:] = np.arange(model.nu) / 23
    before = [x.copy() for x in (data.qpos, data.qvel, data.ctrl)]
    sample = reference.sample(0.63)
    bound.write(sample, data)
    for value, original in zip((data.qpos, data.qvel, data.ctrl), before, strict=True):
        np.testing.assert_array_equal(value, original)
    tensor = SimpleNamespace(mocap_pos=torch.full((3, model.nmocap, 3), -9.),
                             mocap_quat=torch.full((3, model.nmocap, 4), -9.))
    bound.write(sample, tensor, env_ids=[2, 0], origins=[[1, 2, 0], [-1, 0, 0]])
    np.testing.assert_array_equal(tensor.mocap_pos[1], -9)
    np.testing.assert_allclose(tensor.mocap_pos[2, bound.mocap_ids],
                               sample.body_pos + [1, 2, 0], atol=2e-7)
    np.testing.assert_allclose(tensor.mocap_pos[0, bound.mocap_ids],
                               sample.body_pos + [-1, 0, 0], atol=2e-7)


def test_frozen_launch_and_ttc_keep_entry_extension_completion_distinct(reference):
    head = np.array(reference.reference.metadata["nominal_target_adaptation"]["target"]["target_world"])
    placement_xy = [0., 0.]
    placement = Placement(placement_xy)
    replay = CommittedReplay(reference, head, placement, 0.7)
    head[:] += 5
    placement_xy[:] = [20., 20.]
    np.testing.assert_allclose(replay.frozen_head, [1.0611747276415282, 0.2603441867822694, 1.2284314477179406])
    assert replay.placement.xy == (0., 0.)
    with pytest.raises(FrozenInstanceError):
        replay.requested_ttc = 0.4
    assert replay.entry.sampled_s == 0.51
    assert replay.entry.interpolated_s == pytest.approx(0.5098774175291865)
    assert replay.source_time(0.7) == pytest.approx(replay.entry.interpolated_s)
    extension = reference.reference.metadata["strike_center_time_s"] / replay.rate
    assert 0.7 < extension < replay.duration_s
    assert not replay.complete(extension)
    assert replay.complete(replay.duration_s)
    np.testing.assert_array_equal(replay.sample(replay.duration_s).qpos,
                                   replay.sample(replay.duration_s + 1).qpos)
    grid = np.arange(0, replay.duration_s + PHYSICS_DT, PHYSICS_DT)
    distances = np.array([np.linalg.norm(replay.sample(t).glove_pos - replay.frozen_head,
                                        axis=1).min() for t in grid])
    sampled_entry = grid[np.flatnonzero(distances <= ENCOUNTER_RADIUS)[0]]
    assert 0.7 - 1e-12 <= sampled_entry <= 0.705 + 1e-12


def test_both_hands_remain_hazards_and_unreachable_target_is_explicit(reference):
    right = reference.sample(0).glove_pos[1]
    overlap = CommittedReplay(reference, right)
    assert overlap.entry.launch_overlap and overlap.entry.hand == "right"
    assert overlap.entry.sampled_s == 0
    with pytest.raises(ValueError, match="future entry"):
        CommittedReplay(reference, right, requested_ttc=.7)
    miss = CommittedReplay(reference, (20, 20, 20))
    assert miss.entry.interpolated_s is None and miss.rate == 1
    with pytest.raises(ValueError, match="future entry"):
        CommittedReplay(reference, (20, 20, 20), requested_ttc=.7)
