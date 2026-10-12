"""Audit saved MuJoCo preview trajectories against DuoBox hand rotations.

Run from human-punching: .venv/bin/python -B scripts/audit_glove_rotation.py
This reads the dataset and preview without modifying either.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from human_punching.motion import CROP_SPECS, load_crop


def angle_degrees(actual, expected):
    return np.rad2deg(Rotation.from_matrix(actual @ np.swapaxes(expected, -1, -2)).magnitude())


def audit(preview, dataset):
    report = []
    for punch in CROP_SPECS:
        folder = preview / punch
        manifest = json.loads((folder / "manifest.json").read_text())
        model = mujoco.MjModel.from_xml_path(str(folder / manifest["model_relative_path"]))
        data = mujoco.MjData(model)
        clip = load_crop(dataset, punch, manifest["source"]["padding_requested_s"])
        align = Rotation.from_euler("z", manifest["heading_rotation_rad"]).as_matrix()
        with np.load(folder / "trajectory.npz") as trajectory:
            qpos = trajectory["qpos"]
            np.testing.assert_array_equal(trajectory["source_frames"], clip.source_frames)
        row = {"punch": punch, "frames": len(qpos), "hands": {}}
        for side, joint_index in [("Left", 8), ("Right", 12)]:
            body = model.body(f"{side}Hand").id
            geoms = [
                geom
                for geom in range(model.ngeom)
                if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom))
                and name.startswith(f"{side}_")
                and "glove" in name
            ]
            if not geoms or any(model.geom_bodyid[geom] != body for geom in geoms):
                raise ValueError(f"{side} glove geoms must belong to its hand")
            main = model.geom(f"{side}_boxing_glove").id
            hand_rotation, glove_rotation, local_rotation, local_position = [], [], [], []
            for pose in qpos:
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                rotation = data.xmat[body].reshape(3, 3)
                hand_rotation.append(rotation.copy())
                glove_rotation.append(data.geom_xmat[main].reshape(3, 3).copy())
                local_rotation.append(
                    [rotation.T @ data.geom_xmat[geom].reshape(3, 3) for geom in geoms]
                )
                local_position.append(
                    [rotation.T @ (data.geom_xpos[geom] - data.xpos[body]) for geom in geoms]
                )
            hand_rotation = np.asarray(hand_rotation)
            glove_rotation = np.asarray(glove_rotation)
            local_rotation = np.asarray(local_rotation)
            local_position = np.asarray(local_position)
            expected = align @ clip.global_rotations[:, joint_index]
            metrics = {
                "geom_names": [model.geom(geom).name for geom in geoms],
                "hand_source_max_angle_error_deg": float(
                    angle_degrees(hand_rotation, expected).max()
                ),
                "hand_max_rotation_from_start_deg": float(
                    angle_degrees(hand_rotation, hand_rotation[0]).max()
                ),
                "glove_max_rotation_from_start_deg": float(
                    angle_degrees(glove_rotation, glove_rotation[0]).max()
                ),
                "glove_hand_relative_rotation_max_drift_deg": float(
                    angle_degrees(local_rotation, local_rotation[0]).max()
                ),
                "glove_hand_relative_position_max_drift_m": float(
                    np.linalg.norm(local_position - local_position[0], axis=-1).max()
                ),
                "source_local_wrist_max_rotation_from_start_deg": float(
                    angle_degrees(
                        clip.local_rotations[:, joint_index], clip.local_rotations[0, joint_index]
                    ).max()
                ),
            }
            if metrics["hand_source_max_angle_error_deg"] > 1e-6:
                raise ValueError(f"{punch} {side}: source hand rotation mismatch")
            if metrics["glove_hand_relative_rotation_max_drift_deg"] > 1e-8:
                raise ValueError(f"{punch} {side}: glove rotation is not fixed to hand")
            if metrics["glove_hand_relative_position_max_drift_m"] > 1e-10:
                raise ValueError(f"{punch} {side}: glove position is not fixed to hand")
            row["hands"][side.lower()] = metrics
        report.append(row)
    return report


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, default=project / "outputs" / "preview")
    parser.add_argument("--dataset", type=Path, default=project.parent / "DuoBox")
    args = parser.parse_args()
    print(json.dumps(audit(args.preview, args.dataset), indent=2))
