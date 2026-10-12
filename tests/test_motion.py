"""Source/kinematics regression tests; no GPU, Torch, or legacy imports required."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from human_punching.motion import (
    CROP_SPECS,
    FPS,
    JOINT_NAMES,
    PARENTS,
    compose_global_rotations,
    forward_kinematics,
    load_crop,
)

DATASET = Path(__file__).resolve().parents[2] / "DuoBox"


class KinematicsTests(unittest.TestCase):
    def test_parent_rotation_actually_rotates_child_translation(self):
        local = np.broadcast_to(np.eye(3), (2, 21, 3, 3)).copy()
        rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        local[1, 0] = rz
        offsets = np.zeros((21, 3))
        offsets[1:] = [1.0, 0.0, 0.0]
        root = np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
        global_r = compose_global_rotations(local)
        xyz = forward_kinematics(root, global_r, offsets)
        np.testing.assert_allclose(xyz[0, 2], [4.0, 3.0, 4.0])
        np.testing.assert_allclose(xyz[1, 2], [2.0, 5.0, 4.0])
        # Rotation belongs to a joint's children, not to its own translation.
        local[:, 1] = rz
        xyz = forward_kinematics(root, compose_global_rotations(local), offsets)
        np.testing.assert_allclose(xyz[0, 1], [3.0, 3.0, 4.0])
        np.testing.assert_allclose(xyz[0, 2], [3.0, 4.0, 4.0])

    def test_parent_schema(self):
        self.assertEqual(len(JOINT_NAMES), 21)
        self.assertEqual(JOINT_NAMES[8], "LeftHand")
        self.assertEqual(JOINT_NAMES[12], "RightHand")
        for index, parent in enumerate(PARENTS):
            self.assertLess(parent, index)

    def test_invalid_inputs_rejected_before_loading(self):
        for padding in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                load_crop("missing", "jab", padding)
        with self.assertRaises(ValueError):
            load_crop("missing", "uppercut")

    def test_corrupted_input_is_rejected_before_pickle_loading(self):
        # Keep every temporary write inside this project, including tests.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            raw = Path(directory) / "fbx_export_joints"
            raw.mkdir()
            (raw / CROP_SPECS["jab"]["filename"]).write_bytes(b"untrusted replacement")
            with patch("numpy.load") as loader:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    load_crop(directory, "jab")
                loader.assert_not_called()


@unittest.skipUnless((DATASET / "fbx_export_joints").is_dir(), "Local DuoBox dataset not present")
class SourceTests(unittest.TestCase):
    def test_selected_crops_reconstruct_actual_recordings(self):
        for punch, spec in CROP_SPECS.items():
            with self.subTest(punch=punch):
                clip = load_crop(DATASET, punch)
                self.assertEqual(
                    clip.positions.shape, (spec["last_frame"] - spec["first_frame"] + 1, 21, 3)
                )
                self.assertEqual(
                    clip.source_frames[[0, -1]].tolist(), [spec["first_frame"], spec["last_frame"]]
                )
                self.assertEqual(clip.metadata["source_sha256"], spec["source_sha256"])
                self.assertEqual(clip.active_hand_index, spec["active_hand_index"])
                self.assertEqual(clip.fps, FPS)
                self.assertAlmostEqual(clip.duration_s, (len(clip.positions) - 1) / FPS)
                xyz = forward_kinematics(
                    clip.positions[:, 0], clip.global_rotations, clip.rest_offsets
                )
                np.testing.assert_allclose(xyz, clip.positions, atol=1e-6, rtol=0)
                self.assertLess(clip.metadata["rest_offset_max_deviation_m"], 1e-6)
                # Catches accidental local/global confusion in the source adapter.
                self.assertGreater(
                    np.max(np.abs(clip.local_rotations - clip.global_rotations)), 0.5
                )

    def test_padding_adds_recorded_context_without_altering_selected_motion(self):
        unpadded = load_crop(DATASET, "jab")
        padded = load_crop(DATASET, "jab", padding_s=0.25)
        self.assertEqual(padded.source_frames[0], 340)
        self.assertEqual(padded.source_frames[-1], 548)
        np.testing.assert_array_equal(padded.positions[30:-30], unpadded.positions)
        np.testing.assert_array_equal(padded.local_rotations[30:-30], unpadded.local_rotations)
        all_frames = load_crop(DATASET, "jab", padding_s=100)
        self.assertEqual(all_frames.source_frames[0], 0)
        self.assertEqual(all_frames.source_frames[-1], 878)


if __name__ == "__main__":
    unittest.main()
