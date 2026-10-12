"""Target-reaching checks using the real four motion clips and MuJoCo FK."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.human import build_human
from human_punching.motion import CROP_SPECS, load_crop
from human_punching.retarget import retarget_clip
from human_punching.targeting import aim_clip

DATASET = Path(__file__).resolve().parents[2] / "DuoBox"
pytestmark = pytest.mark.skipif(not DATASET.is_dir(), reason="DuoBox data not installed")
HEAD = np.array([0.8539635, 0.20, 1.231490745])


@pytest.fixture(scope="module")
def target_model(tmp_path_factory):
    clip = load_crop(DATASET, "jab")
    info = build_human(tmp_path_factory.mktemp("aim-human"), clip.rest_offsets)
    return mujoco.MjModel.from_xml_path(info["xml_path"]), info


@pytest.mark.parametrize("punch", CROP_SPECS)
def test_aim_reaches_head_with_consistent_fk_and_preserves_other_joints(punch, target_model):
    model, info = target_model
    source = retarget_clip(load_crop(DATASET, punch, padding_s=0.25), model, info["scale"])[
        "trajectory"
    ]
    source_copy = source["qpos"].copy()
    result = aim_clip(model, source, punch, HEAD)
    aimed, metrics = result["trajectory"], result["metrics"]
    np.testing.assert_array_equal(source["qpos"], source_copy)
    np.testing.assert_array_equal(aimed["qpos"][[0, -1]], source["qpos"][[0, -1]])
    np.testing.assert_array_equal(aimed["time_s"], source["time_s"])
    np.testing.assert_array_equal(aimed["source_frames"], source["source_frames"])
    changed = []
    for name in metrics["modified_joints"]:
        address = int(model.joint(name).qposadr[0])
        changed.extend(range(address, address + 4))
        np.testing.assert_allclose(
            np.linalg.norm(aimed["qpos"][:, address : address + 4], axis=1), 1, atol=1e-12
        )
    unchanged = np.setdiff1d(np.arange(model.nq), changed)
    np.testing.assert_array_equal(aimed["qpos"][:, unchanged], source["qpos"][:, unchanged])
    body_names = list(aimed["body_names"])
    feet = [body_names.index(name) for name in ("LeftFoot", "RightFoot")]
    np.testing.assert_array_equal(aimed["body_pos_w"][:, feet], source["body_pos_w"][:, feet])
    assert metrics["anchor_position_error_m"] < 0.004
    assert metrics["path_position_error_max_m"] < 0.012
    assert 0.010 < metrics["minimum_proxy_surface_gap_m"] < 0.020
    assert metrics["maximum_glove_center_correction_m"] > 0.20
    assert metrics["maximum_glove_orientation_change_deg"] < 5.0
    # Inspect actual compiled mesh centers, not a wrist-origin approximation.
    data = mujoco.MjData(model)
    glove = model.geom(f"{metrics['active_hand'].title()}_boxing_glove").id
    for frame in (0, metrics["anchor_index"], metrics["closest_index"], len(source_copy) - 1):
        data.qpos[:] = aimed["qpos"][frame]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.geom_xpos[glove], aimed["actual_glove_pos_w"][frame])
        hand = model.body(f"{metrics['active_hand'].title()}Hand").id
        assert np.linalg.norm(data.geom_xpos[glove] - data.xpos[hand]) > 0.03
    frame = metrics["anchor_index"]
    velocity = np.empty(model.nv)
    mujoco.mj_differentiatePos(
        model,
        velocity,
        float(aimed["time_s"][frame + 1] - aimed["time_s"][frame - 1]),
        aimed["qpos"][frame - 1],
        aimed["qpos"][frame + 1],
    )
    np.testing.assert_allclose(velocity, aimed["qvel"][frame], atol=1e-12)
    for field in ("qpos", "qvel", "body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        assert np.isfinite(aimed[field]).all()
    if punch == "right_hook":
        # A hook can approach the head before the outward-completion label.
        assert metrics["anchor_source_frame"] < CROP_SPECS[punch]["outward_completion_frame"]


def test_aim_rejects_invalid_head_and_outside_anchor(target_model):
    model, info = target_model
    source = retarget_clip(load_crop(DATASET, "jab"), model, info["scale"])["trajectory"]
    with pytest.raises(ValueError, match="head_center"):
        aim_clip(model, source, "jab", [np.nan, 0.0, 1.0])
    with pytest.raises(ValueError, match="strictly inside"):
        aim_clip(model, source, "jab", HEAD, anchor_frame=0)
    with pytest.raises(ValueError, match="radii"):
        aim_clip(model, source, "jab", HEAD, head_radius_m=0)


@pytest.mark.parametrize("punch", CROP_SPECS)
def test_proxy_contact_mode_reaches_published_envelope(punch, target_model):
    model, info = target_model
    source = retarget_clip(load_crop(DATASET, punch), model, info["scale"])["trajectory"]
    result = aim_clip(model, source, punch, HEAD, target_mode="proxy_contact")
    metrics = result["metrics"]
    assert metrics["target_mode"] == "proxy_contact"
    assert metrics["head_radius_m"] == .06
    assert metrics["glove_proxy_radius_m"] == .08
    assert .131 < metrics["minimum_center_distance_m"] < .14
    assert metrics["anchor_position_error_m"] < .004
    assert metrics["endpoints_unchanged"] and metrics["unselected_qpos_unchanged"]


def test_proxy_contact_rejects_target_outside_envelope(target_model):
    model, info = target_model
    source = retarget_clip(load_crop(DATASET, "jab"), model, info["scale"])["trajectory"]
    with pytest.raises(ValueError, match="strictly inside"):
        aim_clip(model, source, "jab", HEAD, target_mode="proxy_contact",
                 contact_center_distance_m=.15)
