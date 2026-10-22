"""DuoBox frame adapter and explicit-path constructor for the existing GMR solver.

PR181 supplies the two-stage H2 matching organization and final-foot mapping;
PR183 cross-checks the 31-joint body names. Their source-specific rotations,
scales and model/controller are deliberately not copied into the DuoBox adapter.
The local solver copy has only four explicit ``limits=`` keyword corrections
for Mink's actual API; see _gmr/SOURCE.json. Its algorithm remains unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from human_punching.motion import JOINT_NAMES as SOURCE_NAMES, MotionClip
from ._gmr.motion_retarget import GeneralMotionRetargeting
from .robot import H2_XML, ROOT, assert_selected_model

CONFIG_PATH = ROOT / "configs/task10c/duobox_to_h2_31.json"
# Neutral source +x left, +y up, +z forward -> robot +x forward,+y left,+z up.
BIND_TO_ROBOT = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
# Small, explicit morphology offsets for the inactive arm of each hook. The
# source torso rotates these targets; its own motion and active punch targets
# are untouched. This is an IK target adjustment, not collision-aware solving.
FAMILY_SOURCE_TARGET_OFFSETS_TORSO_M = {
    "left_hook": {"RightHand": [0.05, -0.025, 0.0], "RightForeArm": [0.04, -0.025, 0.0]},
    "right_hook": {"LeftHand": [0.05, 0.035, 0.0], "LeftForeArm": [0.04, 0.035, 0.0]},
}


class ExplicitH2Retargeter(GeneralMotionRetargeting):
    """Constructor-only adaptation. All target updates/tasks/solves are inherited."""

    def __init__(self, xml_path: str | Path = H2_XML, config_path: str | Path = CONFIG_PATH,
                 ik_dt: float = 1 / 120, solver: str = "proxqp"):
        self.xml_file = str(Path(xml_path).resolve())
        self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        assert_selected_model(self.model)
        self.model.opt.timestep = float(ik_dt)
        cfg = deepcopy(json.loads(Path(config_path).read_text()))
        self.robot_dof_names = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                               self.model.dof_jntid[i]): i
                                for i in range(self.model.nv)}
        self.robot_body_names = {self.model.body(i).name: i for i in range(self.model.nbody)}
        self.robot_motor_names = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        for key in ("ik_match_table1", "ik_match_table2", "human_root_name", "robot_root_name",
                    "use_ik_match_table1", "use_ik_match_table2", "human_scale_table"):
            setattr(self, key, cfg[key])
        self.ground = cfg["ground_height"] * np.array([0, 0, 1])
        self.max_iter = 10
        self.solver, self.damping = solver, 0.5
        for key in ("human_body_to_task1", "human_body_to_task2", "pos_offsets1", "rot_offsets1",
                    "pos_offsets2", "rot_offsets2", "task_errors1", "task_errors2"):
            setattr(self, key, {})
        # A 1 mrad interior margin prevents proxqp's few-microradian tolerance
        # from tripping Mink's strict next-iteration configuration check.
        self.ik_limits = [mink.ConfigurationLimit(self.model, min_distance_from_limits=0.001)]
        self.setup_retarget_configuration()
        self.ground_offset = 0.0


class DuoBoxAdapter:
    """Emit metre/z-up positions and scalar-first world rotations without LAFAN transforms.

    Only a recorded scene yaw/origin is removed. Toe frames use actual recorded
    ankle-to-toe vectors and a bind-derived correction; no fabricated FootMod.
    """

    def __init__(self, clip: MotionClip, source_target_offsets_torso_m: dict | None = None):
        self.clip = clip
        self.source_target_offsets_torso_m = deepcopy(source_target_offsets_torso_m or {})
        for name, value in self.source_target_offsets_torso_m.items():
            if name not in ("LeftHand", "RightHand", "LeftForeArm", "RightForeArm"):
                raise ValueError("Only explicit wrist/elbow target offsets are supported")
            if np.asarray(value).shape != (3,) or not np.isfinite(value).all():
                raise ValueError("Arm target offsets need three finite metres")
        forward = clip.global_rotations[0, 0] @ np.array([0., 0., 1.])
        self.yaw = float(np.arctan2(forward[1], forward[0]))
        self.scene_rotation = Rotation.from_euler("z", -self.yaw).as_matrix()
        self.origin = clip.positions[0, 0].copy()
        self.origin[2] = 0
        self.foot_bind_frames = {}
        for side in ("Left", "Right"):
            toe = SOURCE_NAMES.index(side + "ToeBase")
            x = clip.rest_offsets[toe].copy()
            x /= np.linalg.norm(x)
            y = np.cross(np.array([0., 1., 0.]), x)
            y /= np.linalg.norm(y)
            self.foot_bind_frames[side] = np.column_stack((x, y, np.cross(x, y)))

    def frame(self, index: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        p = (self.clip.positions[index] - self.origin) @ self.scene_rotation.T
        r = self.scene_rotation @ self.clip.global_rotations[index]
        result = {name: (p[i].copy(), Rotation.from_matrix(r[i]).as_quat(scalar_first=True))
                  for i, name in enumerate(SOURCE_NAMES)}
        # Optional small morphology/clearance adjustment of an IK target. This
        # does not modify the recorded skeleton, its rotations, or torso motion.
        # Offsets are in source metres, in torso +x forward/+y left/+z up axes;
        # GMR's documented pelvis-relative scale is applied afterward.
        torso_axes = r[SOURCE_NAMES.index("Spine1")] @ BIND_TO_ROBOT.T
        for name, offset in self.source_target_offsets_torso_m.items():
            pos, quat = result[name]
            result[name] = (pos + torso_axes @ np.asarray(offset), quat)
        for side in ("Left", "Right"):
            foot = SOURCE_NAMES.index(side + "Foot")
            result[side + "FootFrame"] = (
                p[foot].copy(),
                Rotation.from_matrix(r[foot] @ self.foot_bind_frames[side]).as_quat(scalar_first=True),
            )
        return result

    @property
    def metadata(self) -> dict:
        return {"source_world_convention": "z-up metres; no centimetre conversion or LAFAN world rotation",
                "source_to_reference_rotation": self.scene_rotation.tolist(),
                "source_ground_origin": self.origin.tolist(),
                "removed_scene_yaw_rad": self.yaw,
                "foot_frame": "recorded Foot-to-Toe direction with bind-derived local orientation offset",
                "source_target_offsets_torso_m": self.source_target_offsets_torso_m,
                "target_offset_convention": "Optional elbow/wrist IK target adjustment in source metres, torso axes +x forward/+y left/+z up; GMR scaling follows. Original recorded skeleton and all source rotations remain unchanged."}


def create_duobox_config(clip: MotionClip, output: str | Path = CONFIG_PATH) -> dict:
    """Fit constant bind-frame offsets while retaining the PR two-stage organization."""
    adapter = DuoBoxAdapter(clip)
    pr181 = json.loads((ROOT / "configs/task10c/pr181_lafan1_to_h2.json").read_text())
    pr183 = json.loads((ROOT / "configs/task10c/pr183_smplx_to_h2.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Fit root and lower-body scale to the actual hip-to-final-ankle bind lengths.
    robot_leg = np.linalg.norm(data.body("left_ankle_pitch_link").xpos - data.body("pelvis").xpos)
    human_leg = np.linalg.norm(sum((clip.rest_offsets[i] for i in (13, 14, 15)), np.zeros(3)))
    leg_scale = float(robot_leg / human_leg)
    mapping = {"pelvis": "Hips", "torso_link": "Spine1"}
    for side in ("left", "right"):
        h = side.title()
        mapping.update({side + "_hip_yaw_link": h + "UpLeg", side + "_knee_link": h + "Leg",
                        side + "_ankle_pitch_link": h + "FootFrame",
                        side + "_shoulder_yaw_link": h + "Arm",
                        side + "_elbow_link": h + "ForeArm", side + "_wrist_yaw_link": h + "Hand"})
    # Bodies in PR183 sometimes use the upstream preceding shoulder/hip link;
    # all our targets are verified by name against selected 31-joint kinematics.
    for body in mapping:
        model.body(body)
    offsets = {}
    for body, human in mapping.items():
        rot = BIND_TO_ROBOT.T.copy()
        if human.endswith("Arm") and not human.endswith("ForeArm"):
            sign = 1 if human.startswith("Left") else -1
            rot = BIND_TO_ROBOT.T @ Rotation.from_euler("x", sign * np.pi / 2).as_matrix()
        elif human.endswith(("ForeArm", "Hand")):
            # At right T-pose, Rx(-pi/2) @ Ry(pi/2) has the mirrored elbow
            # hinge (world -z); the required source-local offset is Rz(pi).
            rot = np.eye(3) if human.startswith("Left") else Rotation.from_euler("z", np.pi).as_matrix()
        elif human.endswith("FootFrame"):
            rot = adapter.foot_bind_frames[human[:4] if human.startswith("Left") else "Right"].T @ BIND_TO_ROBOT.T
        offsets[human] = Rotation.from_matrix(rot).as_quat(scalar_first=True).tolist()
    cfg = {"robot_root_name": "pelvis", "human_root_name": "Hips", "ground_height": 0.,
           "human_height_assumption": 1.8, "use_ik_match_table1": True, "use_ik_match_table2": True,
           "human_scale_table": {h: leg_scale for h in mapping.values()},
           "ik_match_table1": {}, "ik_match_table2": {},
           "family_source_target_offsets_torso_m": deepcopy(FAMILY_SOURCE_TARGET_OFFSETS_TORSO_M)}
    # GMR uses pelvis-relative scaling. The human elbow frame contains twist
    # that the selected H2 single-hinge elbow cannot realize. Full orientation
    # matching used to displace the elbow/wrist targets, leaving the robot wrist
    # to compensate by roughly 50 degrees in the jab guard. Fit their positions
    # more strongly while retaining a modest hand orientation objective. This
    # changes only task weights, not GMR's solver or the source bind conventions.
    for body, h in mapping.items():
        if h in ("Hips", "Spine1"):
            w1, w2 = ([100, 100], [100, 80]) if h == "Hips" else ([0, 40], [0, 25])
        elif h.endswith("FootFrame"):
            w1, w2 = [60, 25], [80, 25]
        elif h.endswith("Hand"):
            w1, w2 = [150, 3], [150, 3]
        elif h.endswith("ForeArm"):
            w1, w2 = [100, 1], [100, 1]
        elif h.endswith("Arm"):
            w1, w2 = [10, 2], [10, 2]
        else:
            w1, w2 = [0, 10], [10, 5]
        for stage, weights in ((1, w1), (2, w2)):
            cfg[f"ik_match_table{stage}"][body] = [h, *weights, [0, 0, 0], offsets[h]]
    cfg["provenance"] = {"pr181_head": "5d63425d789309f032df83177ced85a272cb5260",
                         "pr183_head": "bb5b73a6871142013b61334e237ce9edd51a90b6",
                         "pr181_match_body_names": list(pr181["ik_match_table1"]),
                         "pr183_match_body_names": list(pr183["ik_match_table1"]),
                         "bind_source_sha256": clip.metadata["source_sha256"],
                         "leg_scale_from_bind_geometry": leg_scale,
                         "head": "selected model coordinates remain explicitly zero; no head task",
                         "offset_convention": "R_world_target = R_source_world @ R_bind_offset",
                         "torso": "source-specific axis correction; no inherited -18 degree offset",
                         "arm_target_revision": "position_fit_v2",
                         "arm_target_rationale": "Source elbow frames contain twist not realizable by H2's single-hinge elbows. Stronger elbow/wrist position objectives prevent full-frame orientation conflict from cranking the wrists; modest hand orientation tracking remains. Source transforms, bind offsets and the inherited GMR solver are unchanged.",
                         "hook_clearance_rationale": "Small inactive wrist/elbow target offsets follow source torso axes to address selected-H2 torso-arm and head-glove penetration during hooks. Recorded source rotations, torso/root and active-arm targets are unchanged; this does not add collision avoidance to GMR."}
    Path(output).write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg
