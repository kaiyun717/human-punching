from dataclasses import replace
import mujoco
import numpy as np
import pytest

from human_punching.wbc.references import (
    FAMILIES, PROJECT_ROOT, convert_release, kinematic_arrays,
    load_release_reference, project_output, resample_reference, sha256,
)

RELEASE = PROJECT_ROOT / "outputs/four-punch-milestone"


def test_present_release_payloads_and_native_timestamps():
    for family in FAMILIES:
        clip = load_release_reference(RELEASE, family)
        assert clip.qpos.shape[1] == 55
        assert not clip.metadata["qualified_for_training"]
        assert clip.metadata["requested_ttc_s"] is None
        assert np.array_equal(clip.target_w, [.8539635, .2, 1.2314907452582464])
    right = load_release_reference(RELEASE, "right_hook")
    assert not np.isclose(np.diff(right.time_s).mean(), 1/60, atol=1e-6)


def test_real_conversion_has_exact_clock_order_and_fk(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Conversion must never execute physics")
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    root = tmp_path / "converted"
    manifest = convert_release(RELEASE, root)
    assert manifest["qualified_for_training"] is False
    assert len(manifest["clips"]) == 4
    for row in manifest["clips"]:
        with np.load(root / row["master_path"], allow_pickle=False) as master, \
             np.load(root / row["motion_path"], allow_pickle=False) as motion:
            np.testing.assert_allclose(np.diff(master["time_s"]), .005, atol=1e-12)
            assert motion["fps"] == 50
            assert master["body_names"][0] == "Hips"
            assert len(master["body_names"]) == 23
            assert len(master["joint_names"]) == 48
            np.testing.assert_array_equal(master["joint_pos"][::4], motion["joint_pos"])
            np.testing.assert_array_equal(master["body_quat_w"][::4], motion["body_quat_w"])
            np.testing.assert_allclose(np.linalg.norm(master["body_quat_w"], axis=-1), 1, atol=1e-12)
            np.testing.assert_allclose(master["qvel"][-1], 0, atol=1e-10)
            model = mujoco.MjModel.from_xml_path(str(root / manifest["model_path"]))
            data = mujoco.MjData(model)
            i = len(master["qpos"]) // 2
            data.qpos[:] = master["qpos"][i]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.xpos[1:], master["body_pos_w"][i], atol=1e-12)
        assert sha256(root / row["master_path"]) == row["master_sha256"]
        assert sha256(root / row["motion_path"]) == row["motion_sha256"]
    with pytest.raises(FileExistsError):
        convert_release(RELEASE, root)


def test_root_angular_velocity_frame_and_quaternion_sign_invariance():
    clip = load_release_reference(RELEASE, "jab")
    changed = clip.qpos.copy()
    changed[::2, 3:7] *= -1
    equivalent = replace(clip, qpos=changed)
    a, b = resample_reference(clip), resample_reference(equivalent)
    np.testing.assert_allclose(a.qvel, b.qvel, atol=1e-10)
    np.testing.assert_allclose(kinematic_arrays(a)["body_pos_w"],
                               kinematic_arrays(b)["body_pos_w"], atol=1e-10)


def test_conversion_rejects_bad_clock_quaternion_and_protected_paths(tmp_path):
    clip = load_release_reference(RELEASE, "jab")
    bad_times = clip.time_s.copy()
    bad_times[2] = bad_times[1]
    with pytest.raises(ValueError, match="timestamps"):
        replace(clip, time_s=bad_times)
    bad_pose = clip.qpos.copy()
    bad_pose[0, 3:7] = 0
    with pytest.raises(ValueError, match="quaternions"):
        replace(clip, qpos=bad_pose)
    with pytest.raises(ValueError):
        project_output(PROJECT_ROOT.parent / "robust_sav/output.npz")
    with pytest.raises(ValueError):
        project_output(RELEASE / "new_output.npz")
    link = tmp_path / "escaped"
    link.symlink_to(PROJECT_ROOT.parent / "robust_sav", target_is_directory=True)
    with pytest.raises(ValueError):
        project_output(link / "output.npz")
