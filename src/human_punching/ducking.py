"""DuoBox defender motion -> GMR -> a kinematic 29-DoF G1 trajectory."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import time

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from human_punching.motion import (
    FPS, JOINT_NAMES, PARENTS, MotionClip, compose_global_rotations, forward_kinematics,
)
from human_punching.task10c._gmr.motion_retarget import GeneralMotionRetargeting
from human_punching.task10c.retargeting import BIND_TO_ROBOT, DuoBoxAdapter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT.parent / "DuoBox/fbx_export_joints/006_ma_leadhook_liu_shanbi_joints.npy"
SOURCE_SHA256 = "790fb445dd9dd002756f9902f29730e69a40de8259a8c6add7bf4bd3f48de23b"
DEFAULT_MODEL = ROOT / ".venv-task10c/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"


class G1DuoBoxRetargeter(GeneralMotionRetargeting):
    """Explicit-path constructor; GMR target updates and both IK stages are inherited."""

    def __init__(self, model_path: Path, config: dict):
        self.xml_file = str(model_path.resolve())
        self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        if (self.model.nq, self.model.nv) != (36, 35):
            raise ValueError("Expected the local floating-base 29-DoF G1")
        self.model.opt.timestep = 1 / FPS
        cfg = deepcopy(config)
        for key in ("ik_match_table1", "ik_match_table2", "human_root_name", "robot_root_name",
                    "use_ik_match_table1", "use_ik_match_table2", "human_scale_table"):
            setattr(self, key, cfg[key])
        self.ground = np.zeros(3)
        self.max_iter = 10
        self.solver, self.damping = "proxqp", 0.5
        for key in ("human_body_to_task1", "human_body_to_task2", "pos_offsets1", "rot_offsets1",
                    "pos_offsets2", "rot_offsets2", "task_errors1", "task_errors2"):
            setattr(self, key, {})
        self.ik_limits = [mink.ConfigurationLimit(self.model, min_distance_from_limits=0.001)]
        self.setup_retarget_configuration()
        self.ground_offset = 0.0


def _source_clip(source_path: Path, first: int, last: int) -> tuple[MotionClip, np.ndarray]:
    payload = source_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Unexpected source recording SHA256: {digest}")
    recording = np.load(io.BytesIO(payload), allow_pickle=True).item()
    if not 0 <= first < last < len(recording["liu_position"]):
        raise ValueError("Invalid inclusive source frame bounds")
    positions = np.asarray(recording["liu_position"][first:last + 1], dtype=float).copy()
    local = np.asarray(recording["liu_rotation"][first:last + 1], dtype=float).copy()
    opponent = np.asarray(recording["ma_position"][first:last + 1], dtype=float).copy()
    global_rotations = compose_global_rotations(local)
    offsets = np.zeros_like(positions)
    for j, parent in enumerate(PARENTS[1:], 1):
        offsets[:, j] = np.einsum("tba,tb->ta", global_rotations[:, parent],
                                  positions[:, j] - positions[:, parent])
    rest = np.median(offsets, axis=0)
    errors = np.linalg.norm(forward_kinematics(positions[:, 0], global_rotations, rest) - positions, axis=-1)
    if not np.isfinite(positions).all() or not np.isfinite(local).all() or errors.max() > 1e-4:
        raise ValueError("Source joint data failed its finite/rigid-skeleton check")
    frames = np.arange(first, last + 1)
    metadata = {"actor": "liu", "opponent": "ma", "source_path": str(source_path),
                "source_sha256": digest, "source_rate_hz": FPS,
                "source_frame_range_inclusive": [first, last], "fk_max_error_m": float(errors.max())}
    clip = MotionClip("duck", positions, local, global_rotations, rest,
                      np.arange(len(frames)) / FPS, frames, metadata)
    return clip, opponent


def _g1_config(clip: MotionClip, adapter: DuoBoxAdapter, model: mujoco.MjModel) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot_leg = np.linalg.norm(data.body("left_ankle_roll_link").xpos - data.body("pelvis").xpos)
    human_leg = np.linalg.norm(sum((clip.rest_offsets[i] for i in (13, 14, 15)), np.zeros(3)))
    scale = float(robot_leg / human_leg)
    mapping = {"pelvis": "Hips", "torso_link": "Spine1"}
    for side in ("left", "right"):
        h = side.title()
        mapping.update({side + "_hip_yaw_link": h + "UpLeg", side + "_knee_link": h + "Leg",
                        side + "_ankle_roll_link": h + "FootFrame",
                        side + "_shoulder_yaw_link": h + "Arm", side + "_elbow_link": h + "ForeArm",
                        side + "_wrist_yaw_link": h + "Hand"})
    cfg = {"human_root_name": "Hips", "robot_root_name": "pelvis", "ground_height": 0.0,
           "human_scale_table": {h: scale for h in mapping.values()},
           "use_ik_match_table1": True, "use_ik_match_table2": True,
           "ik_match_table1": {}, "ik_match_table2": {}, "leg_scale": scale}
    for body, human in mapping.items():
        rot = BIND_TO_ROBOT.T.copy()
        if human.endswith("UpLeg"):
            rot = rot @ data.body(body).xmat.reshape(3, 3)
        elif human.endswith("Arm") and not human.endswith("ForeArm"):
            sign = 1 if human.startswith("Left") else -1
            rot = rot @ Rotation.from_euler("x", sign * np.pi / 2).as_matrix()
        elif human.endswith(("ForeArm", "Hand")):
            rot = np.eye(3) if human.startswith("Left") else Rotation.from_euler("z", np.pi).as_matrix()
        elif human.endswith("FootFrame"):
            side = "Left" if human.startswith("Left") else "Right"
            rot = adapter.foot_bind_frames[side].T @ BIND_TO_ROBOT.T
        if human == "Hips":
            weights = ((150, 100), (150, 100))
        elif human == "Spine1":
            weights = ((0, 80), (0, 80))
        elif human.endswith("FootFrame"):
            weights = ((120, 30), (150, 30))
        elif human.endswith("Hand"):
            weights = ((60, 2), (60, 2))
        elif human.endswith("ForeArm"):
            weights = ((40, 1), (40, 1))
        elif human.endswith("Arm"):
            weights = ((0, 3), (0, 3))
        else:
            weights = ((0, 10), (8, 5))
        quat = Rotation.from_matrix(rot).as_quat(scalar_first=True).tolist()
        for stage, (position, orientation) in enumerate(weights, 1):
            cfg[f"ik_match_table{stage}"][body] = [human, position, orientation, [0, 0, 0], quat]
    return cfg


def retarget_duck(output_dir: str | Path, source_path: str | Path | None = None,
                  first_frame: int = 360, last_frame: int = 600,
                  model_path: str | Path | None = None) -> dict:
    """Save recorded defender motion and finite joint-limited GMR kinematics at 120 Hz."""
    began = time.monotonic()
    output = Path(output_dir).resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("All output must stay inside human-punching")
    source_path = Path(source_path or DEFAULT_SOURCE).resolve()
    model_path = Path(model_path or DEFAULT_MODEL).resolve()
    clip, opponent = _source_clip(source_path, first_frame, last_frame)
    adapter = DuoBoxAdapter(clip)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    cfg = _g1_config(clip, adapter, model)
    solver = G1DuoBoxRetargeter(model_path, cfg)
    convergence = []
    for _ in range(20):
        solver.retarget(adapter.frame(0))
        convergence.append([float(solver.error1()), float(solver.error2())])
    trajectory = []
    for i in range(len(clip.time_s)):
        trajectory.append(solver.retarget(adapter.frame(i)))
    qpos = np.asarray(trajectory)
    if not np.isfinite(qpos).all():
        raise ValueError("Nonfinite GMR result")
    quat_error = float(np.max(abs(np.linalg.norm(qpos[:, 3:7], axis=-1) - 1)))
    violation = 0.0
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            v = qpos[:, model.jnt_qposadr[j]]
            violation = max(violation, float(np.max(model.jnt_range[j, 0] - v)),
                            float(np.max(v - model.jnt_range[j, 1])))
    if quat_error > 1e-8 or violation > 1e-5:
        raise ValueError(f"Kinematic check failed: quaternion={quat_error}, joint-limit={violation}")
    scale = cfg["leg_scale"]
    source = (clip.positions - adapter.origin) @ adapter.scene_rotation.T * scale
    opponent = (opponent - adapter.origin) @ adapter.scene_rotation.T * scale
    # A single constant scene translation grounds the lowest rendered foot vertex
    # across the clip; no per-frame root rewrite, balance solve, or physics rollout.
    data = mujoco.MjData(model)
    sole_min = []
    pelvis = []
    torso_tilt = []
    foot_ids = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                and model.body(model.geom_bodyid[g]).name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        z = []
        for gid in foot_ids:
            mid = model.geom_dataid[gid]
            start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            vertices = model.mesh_vert[start:start + count] @ data.geom_xmat[gid].reshape(3, 3).T
            z.append(float(np.min(vertices[:, 2] + data.geom_xpos[gid, 2])))
        sole_min.append(min(z))
        pelvis.append(data.body("pelvis").xpos.copy())
        axis = data.body("torso_link").xmat.reshape(3, 3)[:, 2]
        torso_tilt.append(float(np.rad2deg(np.arctan2(np.linalg.norm(axis[:2]), axis[2]))))
    shift = 0.002 - min(sole_min)
    qpos[:, 2] += shift
    source[:, :, 2] += shift
    opponent[:, :, 2] += shift
    pelvis = np.asarray(pelvis)
    pelvis[:, 2] += shift
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "g1_ducking.npz"
    np.savez_compressed(trajectory_path, qpos=qpos, time_s=clip.time_s, source_frames=clip.source_frames,
                        source_joint_positions=source, opponent_joint_positions=opponent,
                        raw_source_joint_positions=clip.positions,
                        source_joint_rotations=adapter.scene_rotation @ clip.global_rotations,
                        scene_rotation=adapter.scene_rotation, source_ground_origin=adapter.origin,
                        leg_scale=np.asarray(scale), ground_shift_m=np.asarray(shift),
                        joint_names=np.asarray([model.joint(j).name for j in range(1, model.njnt)]),
                        source_joint_names=np.asarray(JOINT_NAMES), parents=PARENTS,
                        robot_pelvis_positions=pelvis, robot_torso_tilt_deg=np.asarray(torso_tilt))
    result = {"trajectory_path": str(trajectory_path), "model_xml_path": str(model_path),
              "model_xml_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "source": clip.metadata, "adapter": adapter.metadata, "config": cfg,
              "solver": "vendored GMR two-stage Mink IK, proxqp; no dynamics",
              "frames": len(qpos), "fps": FPS, "duration_s": clip.duration_s,
              "scene_rotation": adapter.scene_rotation.tolist(), "source_ground_origin": adapter.origin.tolist(),
              "leg_scale": scale, "ground_shift_m": shift,
              "initial_convergence_errors": convergence,
              "measurements": {"source_head_drop_m": float(np.ptp(clip.positions[:, 4, 2])),
                               "source_pelvis_drop_m": float(np.ptp(clip.positions[:, 0, 2])),
                               "robot_pelvis_drop_m": float(np.ptp(pelvis[:, 2])),
                               "robot_torso_max_tilt_deg": max(torso_tilt),
                               "minimum_foot_clearance_m": 0.002,
                               "maximum_lower_foot_clearance_m": float(max(sole_min) + shift),
                               "joint_limit_violation_rad": violation, "quaternion_norm_error": quat_error},
              "wall_seconds": time.monotonic() - began}
    (output / "g1_ducking.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ducking")
    parser.add_argument("--first-frame", type=int, default=360)
    parser.add_argument("--last-frame", type=int, default=600)
    args = parser.parse_args()
    result = retarget_duck(args.output, first_frame=args.first_frame, last_frame=args.last_frame)
    print(json.dumps({k: result[k] for k in ("trajectory_path", "model_xml_path", "measurements", "wall_seconds")}, indent=2))
