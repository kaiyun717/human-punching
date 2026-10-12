"""Numerical integration checks against actual MuJoCo FK, not array mocks."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from human_punching.human import build_human
from human_punching.motion import CROP_SPECS, JOINT_NAMES, load_crop
from human_punching.retarget import retarget_clip, surface_bounds

DATASET = Path(__file__).resolve().parents[2] / "DuoBox"
pytestmark = pytest.mark.skipif(not DATASET.is_dir(), reason="DuoBox data not installed")


@pytest.fixture(scope="module")
def human_model(tmp_path_factory):
    clip = load_crop(DATASET, "jab")
    info = build_human(tmp_path_factory.mktemp("retarget-human"), clip.rest_offsets)
    return mujoco.MjModel.from_xml_path(info["xml_path"]), info


@pytest.mark.parametrize("punch", CROP_SPECS)
def test_retarget_preserves_motion_and_has_finite_clip_local_derivatives(punch, human_model):
    model, info = human_model
    clip = load_crop(DATASET, punch)
    result = retarget_clip(clip, model, info["scale"])
    t = result["trajectory"]
    m = result["metrics"]
    assert m["joint_fk_max_error_m"] < 1e-5
    assert t["qpos"].shape == (len(clip.time_s), 87)
    assert t["qvel"].shape == (len(clip.time_s), 66)
    np.testing.assert_array_equal(t["source_frames"], clip.source_frames)
    np.testing.assert_array_equal(t["body_names"], JOINT_NAMES)
    # This catches per-frame floor clamping and accidental root locking.
    root_z_delta = t["qpos"][:, 2] - clip.positions[:, 0, 2] * info["scale"]
    np.testing.assert_allclose(root_z_delta, m["floor_shift_m"], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(t["body_quat_w"], axis=-1), 1, atol=1e-12)
    assert np.min((t["body_quat_w"][1:] * t["body_quat_w"][:-1]).sum(axis=-1)) > 0
    for key in ("qvel", "body_lin_vel_w", "body_ang_vel_w"):
        assert np.isfinite(t[key]).all()
    # Independent centered finite difference in MuJoCo's tangent coordinates.
    frame = len(clip.time_s) // 2
    dq = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, dq, 2 / clip.fps, t["qpos"][frame - 1], t["qpos"][frame + 1])
    np.testing.assert_allclose(dq, t["qvel"][frame], atol=1e-10)
    data = mujoco.MjData(model)
    data.qpos[:] = t["qpos"][frame]
    mujoco.mj_forward(model, data)
    assert surface_bounds(model, data)[0][2] >= 0.002 - 1e-6
    assert abs(m["mesh_bounds_min_m"][2] - 0.002) < 1e-6
