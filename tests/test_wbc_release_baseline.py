from pathlib import Path
import json

import numpy as np
import pytest
import torch

from human_punching.wbc.provenance import PROJECT_ROOT
from human_punching.wbc.release_baseline import (
    DIAGNOSTIC_SCOPE, convert_release_baseline, frozen_source, sample_source,
    source_runtime_arrays,
)
from human_punching.wbc.runtime.bundle import load_runtime_bundle

RELEASE = PROJECT_ROOT / "outputs/four-punch-milestone"


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    destination = tmp_path_factory.mktemp("task09_baseline") / "converted"
    convert_release_baseline(RELEASE, destination)
    return load_runtime_bundle(destination)


def test_full_source_velocity_feedforward_and_hold_preserved(converted):
    """Regression against replacing saved velocities/FF during generic conversion."""
    for clip in converted.clips:
        module, _, _, _, source, timing, _ = source_runtime_arrays(RELEASE, clip["family"])
        with np.load(converted.path(clip["master_path"]), allow_pickle=False) as master:
            times = torch.tensor(master["time_s"], dtype=torch.float32)
            source_time = torch.tensor(source["t"], dtype=torch.float32)
            for old, new in (("qvel", "qvel"), ("torque", "joint_feedforward")):
                expected = module.interpolate_reference(source_time,
                    torch.tensor(source[old], dtype=torch.float32), times).numpy()
                np.testing.assert_array_equal(master[new], expected)
            expected_pose = module.interpolate_reference(source_time,
                torch.tensor(source["qpos"], dtype=torch.float32), times)
            expected_pose[:, 3:7] /= expected_pose[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-8)
            np.testing.assert_array_equal(master["qpos"], expected_pose.numpy())
            assert np.any(master["joint_feedforward"][0] != 0.)
            assert np.any(master["joint_feedforward"][-1] != 0.)
            np.testing.assert_allclose(master["qpos"][0], source["qpos"][0], atol=1e-7)
            np.testing.assert_allclose(np.linalg.norm(master["qpos"][:, 3:7], axis=1), 1., atol=1e-7)
            assert clip["launch_time_s"] == timing["launch_time"]
            assert clip["source_duration_s"] == source["t"][-1]
            assert clip["legacy_episode_duration_s"] == source["t"][-1] + 1.
            assert master["time_s"][-1] >= source["t"][-1]
            assert master["time_s"][-1] - source["t"][-1] < .02 + 1e-10
        with np.load(converted.path(clip["legacy_runtime_path"]), allow_pickle=False) as legacy:
            for key in ("t", "qpos", "qvel", "torque", "glove_pos"):
                np.testing.assert_array_equal(legacy[key], source[key])


def test_view_identity_and_no_production_or_snapshot_claim(converted):
    assert converted.manifest["diagnostic_scope"] == DIAGNOSTIC_SCOPE
    assert converted.manifest["production_qualified"] is False
    with pytest.raises(ValueError, match="unqualified"):
        converted.require_training_qualification()
    guards = []
    for clip in converted.clips:
        assert clip["g1_snapshot_present"] is False
        assert clip["launch_features"] == []
        assert clip["requested_ttc_s"] is None
        assert clip["guard_source"] == "task09_family_guard"
        with np.load(converted.path(clip["master_path"]), allow_pickle=False) as master, np.load(
                converted.path(clip["motion_path"]), allow_pickle=False) as view:
            for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                np.testing.assert_array_equal(view[name], master[name][::4])
            guards.append(master["qpos"][0])
    assert any(np.max(abs(guard - guards[0])) > .1 for guard in guards[1:])
    assert not list(converted.root.rglob("*.pt"))


def test_normalized_linear_interpolation_and_endpoint_clamping():
    module, _ = frozen_source(RELEASE, "jab")
    poses = np.zeros((2, 55))
    poses[0, 3:7] = [1., 0., 0., 0.]
    poses[1, 3:7] = [0., 0., 0., 1.]
    source = {"t": np.asarray([0., 1.]), "qpos": poses,
              "qvel": np.stack((np.zeros(54), np.ones(54) * 4)),
              "torque": np.stack((np.ones(48) * -3, np.ones(48) * 5)),
              "glove_pos": np.zeros((2, 2, 3))}
    sampled = sample_source(module, source, np.asarray([-.1, .25, 1.2]))
    expected = np.asarray([.75, 0., 0., .25]); expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(sampled["qpos"][1, 3:7], expected, atol=1e-7)
    np.testing.assert_array_equal(sampled["qvel"][:, 0], [0., 1., 4.])
    np.testing.assert_array_equal(sampled["torque"][:, 0], [-3., -1., 5.])


def test_no_overwrite_or_unknown_family(converted):
    with pytest.raises(FileExistsError):
        convert_release_baseline(RELEASE, converted.root)
    with pytest.raises(ValueError, match="Unknown"):
        frozen_source(RELEASE, "uppercut")


def test_changed_imported_controller_helper_is_rejected(tmp_path, monkeypatch):
    from human_punching import scalar_control
    changed = tmp_path / "scalar_control.py"
    changed.write_text(Path(scalar_control.__file__).read_text() + "\n# changed helper\n")
    monkeypatch.setattr(scalar_control, "__file__", str(changed))
    with pytest.raises(ValueError, match="Imported scalar controller helper differs"):
        frozen_source(RELEASE, "jab")
