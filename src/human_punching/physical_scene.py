"""Attach a baked, immobile G1 visual target to the physical human pilot.

The robot pose comes from the reviewed standing keyframe. Robot joints are
removed rather than controlled or overwritten during simulation. Human physics
and generalized-coordinate addresses remain unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .g1_scene import PREFIX, _fmt
from .motion import JOINT_NAMES

PROJECT = Path(__file__).resolve().parents[2]


def _copy_meshes(asset, source_directory, output_directory):
    for item in asset:
        if item.tag != "mesh" or "file" not in item.attrib:
            continue
        source = source_directory / item.get("file")
        destination = output_directory / source.name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        item.set("file", destination.name)


def attach_static_g1(
    physical_human_xml: Path,
    output: Path,
    *,
    standing_scene_xml: Path | None = None,
) -> dict:
    """Export a portable physics scene with a fixed standing G1 target.

    The default source is the existing reviewed G1 preview, including its asset
    license. An explicit source allows reproduction from another generated
    standing scene. No robot source repository is modified or imported.
    """
    physical_human_xml = Path(physical_human_xml).resolve()
    output = Path(output).resolve()
    standing_scene_xml = Path(
        standing_scene_xml or PROJECT / "outputs/g1-targeting/scene/scene.xml"
    ).resolve()
    human = mujoco.MjModel.from_xml_path(str(physical_human_xml))
    source = mujoco.MjModel.from_xml_path(str(standing_scene_xml))
    source_data = mujoco.MjData(source)
    mujoco.mj_resetDataKeyframe(source, source_data, source.key("standing").id)
    mujoco.mj_forward(source, source_data)
    root = ET.parse(physical_human_xml).getroot()
    original = ET.parse(standing_scene_xml).getroot()
    compiler, asset, defaults, world = (
        root.find(name) for name in ("compiler", "asset", "default", "worldbody")
    )
    if any(section is None for section in (compiler, asset, defaults, world)):
        raise ValueError("Physical human XML requires compiler, asset, default and worldbody")
    if any(body.get("name", "").startswith(PREFIX) for body in world.iter("body")):
        raise ValueError("Physical human XML already contains a G1")
    output.mkdir(parents=True, exist_ok=True)
    mesh_directory = output / "meshes"
    mesh_directory.mkdir(exist_ok=True)
    human_mesh_directory = physical_human_xml.parent / compiler.get("meshdir", "")
    _copy_meshes(asset, human_mesh_directory, mesh_directory)
    compiler.set("meshdir", "meshes")
    compiler.attrib.pop("assetdir", None)
    source_mesh_directory = standing_scene_xml.parent / original.find("compiler").get(
        "meshdir", ""
    )
    robot_assets = [
        copy.deepcopy(item)
        for item in original.find("asset")
        if item.get("name", "").startswith(PREFIX)
    ]
    _copy_meshes(robot_assets, source_mesh_directory, mesh_directory)
    asset.extend(robot_assets)
    for item in original.find("default"):
        if item.tag == "default" and item.get("class", "").startswith(PREFIX):
            defaults.append(copy.deepcopy(item))
    robot_original = original.find("worldbody/body[@name='g1/pelvis']")
    if robot_original is None:
        raise ValueError("Standing source scene has no g1/pelvis body")
    robot = copy.deepcopy(robot_original)
    for body in robot.iter("body"):
        body_id = source.body(body.get("name")).id
        parent_id = int(source.body_parentid[body_id])
        parent_rotation = source_data.xmat[parent_id].reshape(3, 3)
        local_position = parent_rotation.T @ (
            source_data.xpos[body_id] - source_data.xpos[parent_id]
        )
        local_rotation = parent_rotation.T @ source_data.xmat[body_id].reshape(3, 3)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, local_rotation.ravel())
        body.set("pos", _fmt(local_position))
        for attribute in ("euler", "axisangle", "xyaxes", "zaxis", "mocap"):
            body.attrib.pop(attribute, None)
        body.set("quat", _fmt(quaternion))
        for child in list(body):
            if child.tag in {"joint", "freejoint", "inertial"}:
                body.remove(child)
        for geom in body.findall("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            # The static target contributes neither dynamic mass nor inferred
            # inertia, even when collision-class geoms occupy inertia group 3.
            geom.set("mass", "0")
    world.append(robot)
    root.set("model", "HumanPunching_physical_human_static_G1")
    xml_path = output / "physical_scene.xml"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(xml_path, encoding="unicode", xml_declaration=True)
    combined = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(combined)
    mujoco.mj_forward(combined, data)
    if (combined.nq, combined.nv, combined.nu) != (human.nq, human.nv, human.nu):
        raise AssertionError("Attaching fixed G1 changed human state/action dimensions")
    for joint_id in range(human.njnt):
        name = human.joint(joint_id).name
        attached = combined.joint(name).id
        if (
            attached != joint_id
            or combined.jnt_qposadr[attached] != human.jnt_qposadr[joint_id]
            or combined.jnt_dofadr[attached] != human.jnt_dofadr[joint_id]
        ):
            raise AssertionError(f"Attaching fixed G1 changed human joint map: {name}")
    robot_ids = [
        index for index in range(combined.nbody)
        if combined.body(index).name.startswith(PREFIX)
    ]
    for body_id in robot_ids:
        name = combined.body(body_id).name
        source_id = source.body(name).id
        if not (
            np.allclose(data.xpos[body_id], source_data.xpos[source_id], atol=1e-10, rtol=0)
            and np.allclose(data.xmat[body_id], source_data.xmat[source_id], atol=1e-10, rtol=0)
        ):
            raise AssertionError(f"Baked robot FK differs from standing keyframe: {name}")
    license_source = standing_scene_xml.parent / "MJLAB-LICENSE"
    if license_source.is_file():
        shutil.copyfile(license_source, output / "MJLAB-LICENSE")
    head_id = combined.geom("g1/head_collision").id
    metadata = {
        "xml_path": str(xml_path),
        "physical_human_xml": str(physical_human_xml),
        "standing_scene_xml": str(standing_scene_xml),
        "standing_scene_sha256": hashlib.sha256(standing_scene_xml.read_bytes()).hexdigest(),
        "robot_pose": "standing keyframe baked into fixed body transforms",
        "g1_dynamic_dofs": 0,
        "g1_contact_response": False,
        "g1_is_learned_standing_policy": False,
        "nq": combined.nq,
        "nv": combined.nv,
        "nu": combined.nu,
        "human_mass_kg": float(sum(combined.body(name).mass[0] for name in JOINT_NAMES)),
        "g1_body_names": [combined.body(index).name for index in robot_ids],
        "head_geom_name": "g1/head_collision",
        "head_center_w": data.geom_xpos[head_id].tolist(),
        "head_radius_m": float(combined.geom_size[head_id, 0]),
        "human_physics_preserved": True,
        "limitations": [
            "G1 is a fixed target; it has no balance policy or impact response.",
            "Task contact is evaluated analytically using head/glove centers.",
            "A physical human rollout is required; reference playback is not dynamics evidence.",
        ],
    }
    (output / "physical_scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
