"""Selected 31-joint H2 proxy and source-derived IdealPd controller.

Adapted from wbc-mjlab-extension-h2 3d9a05d; license and source identity are
retained in assets/task10c/h2. Only foot/hand contacts are locally corrected.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

ROOT = Path(__file__).resolve().parents[3]
H2_XML = ROOT / "assets/task10c/h2/h2.xml"
PHYSICS_DT = 0.005
ACTOR_DT = 0.020
GLOVE_RADIUS = 0.08
HEAD_RADIUS = 0.06
GLOVE_SITE_NAMES = ("left_glove_center", "right_glove_center")
GLOVE_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
FOOT_BODY_NAMES = ("left_ankle_pitch_link", "right_ankle_pitch_link")
GLOVE_OFFSETS = ((0.135, -0.01, 0.0), (0.135, 0.01, 0.0))
_xml = ET.parse(H2_XML)
JOINT_NAMES = H2_JOINT_NAMES = tuple(j.attrib["name"] for j in _xml.iter("joint") if "name" in j.attrib)
BODY_NAMES = tuple(b.attrib["name"] for b in _xml.iter("body"))
assert len(JOINT_NAMES) == 31

# Lazy imports keep the offline geometry/retargeting path independent of Torch.
def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(H2_XML))
    spec.option.timestep = PHYSICS_DT
    return spec


def get_h2_robot_cfg():
    from mjlab.entity import EntityCfg
    from mjlab.utils.spec_config import CollisionCfg
    from .actuators import H2_ARTICULATION

    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, 1.2),
            joint_pos={
                ".*_hip_pitch_joint": -0.1,
                ".*_knee_joint": 0.3,
                ".*_ankle_pitch_joint": -0.2,
                ".*_shoulder_pitch_joint": 0.35,
                ".*_elbow_joint": 0.87,
                "left_shoulder_roll_joint": 0.18,
                "right_shoulder_roll_joint": -0.18,
            },
            joint_vel={".*": 0.0},
        ),
        articulation=H2_ARTICULATION,
        collisions=(
            CollisionCfg(
                geom_names_expr=(".*_collision",), contype=1, conaffinity=1,
                condim={".*_foot_sole_collision": 3, ".*_glove_collision": 3, ".*": 1},
                priority={".*_foot_sole_collision": 1, ".*": 0},
                friction={".*_foot_sole_collision": (0.6, 0.005, 0.0001)},
            ),
        ),
    )


def __getattr__(name):
    if name in ("H2_ACTION_SCALE", "H2_ARTICULATION"):
        from . import actuators
        return getattr(actuators, name)
    raise AttributeError(name)


def joint_addresses(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    """Map all scalar coordinates by name, never actuator insertion order."""
    ids = [model.joint(name).id for name in JOINT_NAMES]
    return ([int(model.jnt_qposadr[i]) for i in ids], [int(model.jnt_dofadr[i]) for i in ids])


def assert_selected_model(model: mujoco.MjModel) -> None:
    if (model.nq, model.nv) != (38, 37):
        raise ValueError(f"Task10C requires selected 31-joint H2, got nq={model.nq}, nv={model.nv}")
    joint_addresses(model)
    for site in GLOVE_SITE_NAMES:
        model.site(site)
