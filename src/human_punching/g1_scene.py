"""Standalone kinematic human/G1 scenes using the local mjlab G1 asset.

Only generated output is written. The mjlab and robust_sav source trees remain
read-only, and neither project is imported or added as a runtime dependency.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .motion import JOINT_NAMES

MJLAB_ROBUST_PIN = "8ee51fbcf806a7419189f706d9e394cbeb7790fa"
G1_RELATIVE_DIRECTORY = Path("src/mjlab/asset_zoo/robots/unitree_g1")
PREFIX = "g1/"
# KNEES_BENT_KEYFRAME in the mjlab G1 asset, also used by robust_sav's
# unitree_g1_flat_env_cfg. This is its configured pose, not a learned checkpoint
# or the unavailable settled qpos_star stance artifact.
STANDING_JOINT_PATTERNS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _prefix_g1(element: ET.Element) -> None:
    references = {
        "name",
        "class",
        "childclass",
        "mesh",
        "material",
        "texture",
        "body",
        "body1",
        "body2",
        "joint",
        "joint1",
        "joint2",
        "site",
        "site1",
        "site2",
        "geom",
        "geom1",
        "geom2",
        "camera",
        "target",
        "objname",
        "refname",
    }
    for node in element.iter():
        for key in references.intersection(node.attrib):
            if node.attrib[key] != "world":
                node.set(key, PREFIX + node.attrib[key])


def _joint_width(joint_type: int) -> tuple[int, int]:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7, 6
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4, 3
    return 1, 1


def _entity_mapping(model: mujoco.MjModel, *, robot: bool) -> dict:
    names, qpos, dof, joints = [], [], [], []
    for index in range(model.njnt):
        joint = model.joint(index)
        if joint.name.startswith(PREFIX) != robot:
            continue
        nq, nv = _joint_width(int(model.jnt_type[index]))
        qa, va = int(model.jnt_qposadr[index]), int(model.jnt_dofadr[index])
        names.append(joint.name)
        qpos.extend(range(qa, qa + nq))
        dof.extend(range(va, va + nv))
        joints.append(
            {
                "name": joint.name,
                "id": index,
                "qpos_adr": qa,
                "qpos_width": nq,
                "dof_adr": va,
                "dof_width": nv,
            }
        )
    body_ids = [
        index
        for index in range(1, model.nbody)
        if model.body(index).name.startswith(PREFIX) == robot
    ]
    return {
        "joint_names": names,
        "joints": joints,
        "qpos_indices": qpos,
        "dof_indices": dof,
        "body_ids": body_ids,
        "body_names": [model.body(index).name for index in body_ids],
    }


def _robot_mesh_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    for geom in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[geom])).name
        if not body.startswith(PREFIX) or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[geom])
        first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[first : first + count]
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        minimum = np.minimum(minimum, world.min(axis=0))
        maximum = np.maximum(maximum, world.max(axis=0))
    if not np.isfinite(minimum).all():
        raise ValueError("The G1 visual asset contains no triangle meshes")
    return minimum, maximum


def create_scene(
    human_xml: Path,
    output_dir: Path,
    robot_position=(0.85, 0.20, 0.0),
    robot_yaw: float = np.pi,
    *,
    mjlab_root: Path | None = None,
) -> dict:
    """Write a self-contained combined scene and return its standing-pose metadata.

    Human joint/body names and qpos addresses are preserved. All robot names
    receive ``g1/`` prefixes. ``robot_position`` gives pelvis X/Y and the desired
    lowest visual sole Z; the pelvis height is calculated from the standing FK.
    The returned ``qpos_reset`` (also XML keyframe ``standing``) must be copied
    before writing human trajectory indices each frame. This is kinematic replay:
    gravity, contacts and actuators are disabled; no standing policy is implied.
    """
    human_xml = Path(human_xml).resolve()
    output_dir = Path(output_dir).resolve()
    mjlab_root = (
        Path(mjlab_root).resolve()
        if mjlab_root is not None
        else Path(__file__).resolve().parents[3] / "mjlab"
    )
    position = np.asarray(robot_position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(robot_yaw):
        raise ValueError("robot_position and robot_yaw must be finite")
    source_dir = mjlab_root / G1_RELATIVE_DIRECTORY
    source_xml = source_dir / "xmls/g1.xml"
    if not source_xml.is_file():
        raise FileNotFoundError(f"Local authoritative mjlab G1 XML not found: {source_xml}")
    root = ET.parse(human_xml).getroot()
    robot = ET.parse(source_xml).getroot()
    if root.find("keyframe") is not None:
        raise ValueError("The input human model must not already have keyframes")
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Human model is missing its compiler element")
    human_mesh_dir = human_xml.parent / compiler.get("meshdir", "")
    compiler.set("meshdir", "assets")
    compiler.attrib.pop("assetdir", None)
    assets = root.find("asset")
    defaults = root.find("default")
    world = root.find("worldbody")
    if assets is None or defaults is None or world is None:
        raise ValueError("Human model must contain asset/default/worldbody sections")
    for mesh in assets.findall("mesh"):
        source = human_mesh_dir / mesh.attrib["file"]
        destination = assets_dir / ("human_" + source.name)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        mesh.set("file", destination.name)
    source_hashes = {"g1.xml": _sha256(source_xml)}
    for item in robot.find("asset"):
        item = copy.deepcopy(item)
        if item.tag == "mesh":
            source = source_dir / "xmls/assets" / item.attrib["file"]
            destination = assets_dir / ("g1_" + source.name)
            shutil.copyfile(source, destination)
            source_hashes["assets/" + source.name] = _sha256(source)
            item.set("file", destination.name)
        _prefix_g1(item)
        assets.append(item)
    for item in robot.find("default"):
        item = copy.deepcopy(item)
        _prefix_g1(item)
        # Disable all G1 contacts explicitly; these classes otherwise override
        # the human model's nonphysical global contact defaults.
        for geom in item.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        defaults.append(item)
    robot_body = copy.deepcopy(robot.find("worldbody/body"))
    _prefix_g1(robot_body)
    # Preview camera/light are scene-owned; omit the model's tracking helpers.
    for child in list(robot_body):
        if child.tag in {"light", "camera"}:
            robot_body.remove(child)
    robot_body.set("pos", _fmt([position[0], position[1], 0.76]))
    robot_body.set("quat", _fmt([np.cos(robot_yaw / 2), 0, 0, np.sin(robot_yaw / 2)]))
    for geom in robot_body.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    world.append(robot_body)
    root.set("model", "HumanPunching_standing_G1_kinematic_preview")
    root.find("option").set("gravity", "0 0 0")
    xml_path = output_dir / "scene.xml"

    def write_xml():
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(xml_path, encoding="unicode", xml_declaration=True)

    write_xml()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    for index in range(model.njnt):
        name = model.joint(index).name
        if not name.startswith(PREFIX) or model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        for pattern, value in STANDING_JOINT_PATTERNS.items():
            if re.fullmatch(pattern, name.removeprefix(PREFIX)):
                data.qpos[model.jnt_qposadr[index]] = value
                break
    mujoco.mj_forward(model, data)
    base_address = int(model.joint(PREFIX + "floating_base_joint").qposadr[0])
    floor_shift = float(position[2] - _robot_mesh_bounds(model, data)[0][2])
    data.qpos[base_address + 2] += floor_shift
    robot_body.set("pos", _fmt(data.qpos[base_address : base_address + 3]))
    mujoco.mj_forward(model, data)
    head_geom = model.geom(PREFIX + "head_collision").id
    head_center = data.geom_xpos[head_geom].copy()
    human_mapping = _entity_mapping(model, robot=False)
    robot_mapping = _entity_mapping(model, robot=True)
    # Guard the key reason for this builder: existing human trajectories can be
    # scattered by index without the robot silently consuming any coordinates.
    source_model = mujoco.MjModel.from_xml_path(str(human_xml))
    if human_mapping["qpos_indices"] != list(range(source_model.nq)):
        raise AssertionError("Combined model changed human qpos indexing")
    if human_mapping["dof_indices"] != list(range(source_model.nv)):
        raise AssertionError("Combined model changed human velocity indexing")
    for name in JOINT_NAMES:
        if model.body(name).id != source_model.body(name).id:
            raise AssertionError(f"Combined model changed human body index: {name}")
    qpos_reset = data.qpos.copy()
    keyframes = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframes, "key", name="standing", qpos=_fmt(qpos_reset))
    write_xml()
    # Reload the exported file rather than only accepting the in-memory version.
    exported = mujoco.MjModel.from_xml_path(str(xml_path))
    check_data = mujoco.MjData(exported)
    mujoco.mj_resetDataKeyframe(exported, check_data, exported.key("standing").id)
    mujoco.mj_forward(exported, check_data)
    if not np.allclose(check_data.geom_xpos[head_geom], head_center, atol=1e-10, rtol=0):
        raise AssertionError("Exported standing keyframe changed the head target")
    license_source = mjlab_root / "LICENSE"
    if license_source.is_file():
        shutil.copyfile(license_source, output_dir / "MJLAB-LICENSE")
    relative = G1_RELATIVE_DIRECTORY.as_posix()
    pinned_tree = _git(mjlab_root, "rev-parse", MJLAB_ROBUST_PIN + ":" + relative)
    current_tree = _git(mjlab_root, "rev-parse", "HEAD:" + relative)
    dirty = _git(mjlab_root, "status", "--porcelain=v1", "--", relative)
    metadata = {
        "xml_path": str(xml_path),
        "human_xml": str(human_xml),
        "qpos_reset": qpos_reset.tolist(),
        "human_qpos_indices": human_mapping["qpos_indices"],
        "human_dof_indices": human_mapping["dof_indices"],
        "human_qvel_indices": human_mapping["dof_indices"],
        "g1_qpos_indices": robot_mapping["qpos_indices"],
        "g1_dof_indices": robot_mapping["dof_indices"],
        "g1_qvel_indices": robot_mapping["dof_indices"],
        "entities": {"human": human_mapping, "g1": robot_mapping},
        "head_geom_name": PREFIX + "head_collision",
        "head_geom_id": head_geom,
        "head_body_name": model.body(int(model.geom_bodyid[head_geom])).name,
        "head_center_w": head_center.tolist(),
        "head_radius_m": float(model.geom_size[head_geom, 0]),
        "head_local_center_m": model.geom_pos[head_geom].tolist(),
        "robot_position_requested": position.tolist(),
        "robot_yaw_rad": float(robot_yaw),
        "robot_pelvis_position_w": data.qpos[base_address : base_address + 3].tolist(),
        "robot_visual_sole_z_m": _robot_mesh_bounds(model, data)[0][2],
        "robot_floor_alignment_shift_m": floor_shift,
        "robot_visual_bounds_min_m": _robot_mesh_bounds(model, data)[0].tolist(),
        "robot_visual_bounds_max_m": _robot_mesh_bounds(model, data)[1].tolist(),
        "robot_standing_joint_patterns": STANDING_JOINT_PATTERNS,
        "robot_pose_definition": "mjlab KNEES_BENT_KEYFRAME with visual soles aligned to requested Z",
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "provenance": {
            "source": str(source_dir),
            "source_url": "https://github.com/mujocolab/mjlab",
            "mjlab_checkout_commit": _git(mjlab_root, "rev-parse", "HEAD"),
            "robust_sav_mjlab_pin": MJLAB_ROBUST_PIN,
            "g1_asset_tree_identical_to_robust_pin": (
                pinned_tree != "unavailable" and pinned_tree == current_tree and dirty == ""
            ),
            "source_sha256": source_hashes,
            "license": "Source repository Apache-2.0; copied MJLAB-LICENSE accompanies the asset",
        },
        "limitations": [
            "G1 is held kinematically in the configured standing pose, not controlled by a learned policy.",
            "This is not the unavailable robust_sav settled stance artifact qpos_star.",
            "All contacts and gravity are disabled; replay does not simulate impact or balance.",
            "The robust_sav head proxy is a 6 cm sphere, not the entire visible head surface.",
        ],
    }
    (output_dir / "scene_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
