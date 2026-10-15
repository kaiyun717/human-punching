"""Constrained scalar-joint SOMA23 derivative for physical imitation pilots.

Derived from the pinned Apache-2.0 ProtoMotions SOMA23 MJCF under assets/vendor.
Native bone lengths are retained. Limits and strengths are engineering choices,
not a calibrated musculoskeletal model. All restoring torque comes from the
explicit bounded controller: the exported joints have no passive spring/damper.
"""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR = PROJECT_ROOT / "assets" / "vendor" / "protomotions"
BLUE = "0.40 0.72 0.93 1"
RED = "0.72 0.10 0.08 1"

# Exact scalar coordinate order per body. Axes below are positive canonical XYZ,
# including left limbs (upstream mirrors some left X axes).
AXES = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}
BODY_MASS = {
    "Hips": 10.0, "Spine1": 4.0, "Spine2": 5.0, "Chest": 12.0,
    "Neck1": 0.5, "Neck2": 0.5, "Head": 4.5,
    "Shoulder": 0.5, "Arm": 2.2, "ForeArm": 1.3, "Hand": 0.85,
    "Leg": 8.0, "Shin": 3.7, "Foot": 1.1, "ToeBase": 0.0,
}


def _kind(name):
    return name.removeprefix("Left").removeprefix("Right")


def _fmt(values):
    return " ".join(f"{x:.12g}" for x in values)


def _joint_spec(body):
    """Return ordered (axis, min deg, max deg), effort Nm, Kp, Kd."""
    kind = _kind(body)
    side = 1 if body.startswith("Right") else -1
    if kind in ("Neck2", "Head", "ToeBase"):
        return [], 0., 0., 0.
    if kind in ("Spine1", "Spine2"):
        return [("x", -20, 25), ("y", -20, 20), ("z", -25, 25)], 100., 220., 18.
    if kind == "Chest":
        return [("x", -25, 35), ("y", -25, 25), ("z", -35, 35)], 120., 240., 20.
    if kind == "Neck1":
        return [("x", -35, 45), ("y", -30, 30), ("z", -65, 65)], 20., 45., 4.
    if kind == "Shoulder":
        return [("x", -25, 25), ("y", -30, 30), ("z", -30, 30)], 35., 80., 7.
    if kind == "Arm":
        # T-pose: Y lowers/raises the arm; X is humeral axial rotation.
        return [("x", -110, 110), ("y", -110, 110), ("z", -135, 135)], 100., 150., 12.
    if kind == "ForeArm":
        # Flexion precedes pronation: the elbow bend axis cannot precess from a
        # proximal axial rotation. Positive Z flexes right arm toward -Y.
        flexion = ("z", 0, 155) if side == 1 else ("z", -155, 0)
        return [flexion, ("x", -80, 80)], 60., 110., 9.
    if kind == "Hand":
        # Wrist roll is small; most axial rotation belongs to forearm pronation.
        return [("x", -25, 25), ("y", -65, 65), ("z", -35, 35)], 15., 40., 3.
    if kind == "Leg":
        return [("x", -120, 35), ("y", -45, 45), ("z", -45, 45)], 200., 350., 25.
    if kind == "Shin":
        return [("x", 0, 155)], 150., 350., 25.
    if kind == "Foot":
        return [("x", -40, 30), ("y", -25, 25), ("z", -15, 15)], 90., 220., 15.
    raise KeyError(body)


def build_scalar_human(output: Path, timestep=.005, mass_kg=75.):
    """Export native-scale SOMA23-derived body and its machine-readable contract.

    The default neutral qpos is a standing T pose, with world +X forward and +Y
    left. The body itself uses SOMA's +Z up / -Y forward coordinates. Output XML
    is self-contained and contains no meshes or external assets.
    """
    if not 0 < timestep <= .005 or mass_kg <= 0:
        raise ValueError("Require positive mass and timestep in (0, .005]")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = VENDOR / "soma23_humanoid.xml"
    provenance = json.loads((VENDOR / "manifest.json").read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != provenance["files"][source.name]["sha256"]:
        raise ValueError("Vendored SOMA23 source hash does not match manifest")
    root = ET.parse(source).getroot()
    root.set("model", "HumanPunching_SOMA23_constrained")
    root.find("compiler").attrib.update(angle="degree", inertiafromgeom="true",
                                        inertiagrouprange="3 3", fusestatic="false")
    ET.SubElement(root, "option", timestep=str(timestep), gravity="0 0 -9.81",
                  integrator="implicitfast", cone="elliptic", iterations="60",
                  tolerance="1e-9")
    ET.SubElement(root, "visual")
    ET.SubElement(root.find("visual"), "global", offwidth="1280", offheight="720")
    defaults = root.find("default")
    defaults.find("joint").attrib.update(damping="0", stiffness="0", armature="0.02")
    defaults.find("geom").attrib.update(contype="2", conaffinity="1", condim="3",
                                        margin="0", rgba=BLUE, friction="0.9 0.02 0.002",
                                        solref="0.015 1", solimp="0.9 0.95 0.001")
    # Keep a plain color: no texture, facial detail, or dependency on mesh assets.
    asset = root.find("asset")
    asset.clear()
    ET.SubElement(asset, "material", name="geom", rgba=BLUE)
    ET.SubElement(asset, "material", name="glove", rgba=RED)
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="6 6 .1",
                  contype="1", conaffinity="2", group="0", mass="0",
                  rgba="0.22 0.25 0.30 1", friction="0.9 0.02 0.002")
    ET.SubElement(world, "camera", name="scalar_observer", pos="3 -4 2.1",
                  xyaxes="0.8 0.6 0 -0.15 0.20 0.968", fovy="45")
    pelvis = world.find("body[@name='Hips']")
    pelvis.set("quat", "0.7071067811865476 0 0 0.7071067811865475")
    # Upstream actuators and passive springs are deliberately replaced.
    motors = root.find("actuator")
    motors.clear()
    root.find("contact").clear()
    contact = root.find("contact")
    joint_rows = []
    bodies = list(world.iter("body"))
    mass_norm = sum(BODY_MASS[_kind(body.get("name"))] for body in bodies)
    body_names = []
    for body in bodies:
        name = body.get("name")
        kind = _kind(name)
        body_names.append(name)
        for joint in body.findall("joint"):
            body.remove(joint)
        for geom in body.findall("geom"):
            body.remove(geom)
        spec, effort, kp, kd = _joint_spec(name) if name != "Hips" else ([], 0, 0, 0)
        for axis, low, high in spec:
            joint_name = f"{name}_{axis}"
            ET.SubElement(body, "joint", name=joint_name, type="hinge", axis=AXES[axis],
                          range=f"{low} {high}", limited="true", damping="0", stiffness="0",
                          armature="0.02", frictionloss="0")
            ET.SubElement(motors, "motor", name=joint_name, joint=joint_name, gear="1",
                          ctrllimited="true", ctrlrange=f"{-effort} {effort}",
                          forcelimited="true", forcerange=f"{-effort} {effort}")
            joint_rows.append(dict(name=joint_name, body=name, axis=axis,
                                   range_deg=[low, high], effort_nm=effort, kp=kp, kd=kd))
        mass = mass_kg * BODY_MASS[kind] / mass_norm
        attrs = dict(name=f"physical/{name}", group="3", mass=str(mass), material="geom",
                     contype="2", conaffinity="1")
        if name == "Hips":
            attrs.update(type="ellipsoid", size=".13 .085 .085", pos="0 -.02 -.015")
        elif name in ("Spine1", "Spine2"):
            attrs.update(type="ellipsoid", size=".12 .085 .075", pos="0 -.025 .035")
        elif name == "Chest":
            attrs.update(type="ellipsoid", size=".15 .095 .13", pos="0 -.03 .12")
        elif kind in ("Neck1", "Neck2"):
            attrs.update(type="capsule", size=".037", fromto="0 0 0 0 -.012 .05")
        elif name == "Head":
            attrs.update(type="ellipsoid", size=".075 .083 .11", pos="0 .015 .07")
        elif kind in ("Shoulder", "Arm", "ForeArm"):
            child = body.find("body")
            end = np.fromstring(child.get("pos"), sep=" ")
            radius = {"Shoulder": .035, "Arm": .040, "ForeArm": .032}[kind]
            attrs.update(type="capsule", size=str(radius), fromto=_fmt(np.r_[.15 * end, .85 * end]))
        elif kind == "Hand":
            sign = 1 if name.startswith("Left") else -1
            center = [sign * .065, 0, 0]
            attrs.update(type="sphere", size=".08", pos=_fmt(center), material="glove")
            ET.SubElement(body, "site", name=f"{name}_center", pos=_fmt(center),
                          size=".004", rgba="0 1 0 0", group="4")
            # A zero-mass wrist band makes full hand/glove orientation visible.
            ET.SubElement(body, "geom", name=f"visual/{name}_cuff", type="box",
                          size=".026 .042 .036", pos=_fmt([sign * .002, 0, 0]),
                          rgba="0.12 0.20 0.30 1", group="0", mass="0",
                          contype="0", conaffinity="0")
        elif kind in ("Leg", "Shin"):
            child = body.find("body")
            end = np.fromstring(child.get("pos"), sep=" ")
            attrs.update(type="capsule", size=".059" if kind == "Leg" else ".043",
                         fromto=_fmt(np.r_[.1 * end, .90 * end]))
        elif kind == "Foot":
            attrs.update(type="box", size=".052 .125 .025", pos="0 -.065 -.025")
            ET.SubElement(body, "site", name=f"{name}_sole", pos="0 -.065 -.05",
                          size=".005", group="4", rgba="0 1 0 0")
            for k, (x, y) in enumerate(((-.048, -.185), (.048, -.185),
                                        (-.048, .055), (.048, .055))):
                ET.SubElement(body, "site", name=f"{name}_sole_{k}", pos=f"{x} {y} -.05",
                              size=".003", group="4", rgba="0 1 0 0")
        elif kind == "ToeBase":
            continue  # Fixed hierarchy landmark; whole-foot rigid sole replaces toes.
        else:
            raise KeyError(name)
        ET.SubElement(body, "geom", **attrs)
    # Deliberate selected self collision, avoiding overlapping adjacent link
    # proxies. Geom masks otherwise enable all human-floor contacts.
    pairs = []
    for side in ("Left", "Right"):
        other = "Right" if side == "Left" else "Left"
        for upper in ("Head", "Chest", "Spine2", "Hips", f"{other}ForeArm"):
            pairs.append((f"{side}Hand", upper))
    pairs.extend([("LeftHand", "RightHand"), ("LeftShin", "RightShin"),
                  ("LeftFoot", "RightFoot")])
    for first, second in pairs:
        ET.SubElement(contact, "pair", geom1=f"physical/{first}", geom2=f"physical/{second}",
                      condim="3", friction="0.8 0.8 .02 .002 .002")
    path = output / "human.xml"
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sole_ids = [model.site(f"{side}Foot_sole").id for side in ("Left", "Right")]
    pelvis_height = float(-min(data.site_xpos[i, 2] for i in sole_ids))
    pelvis.set("pos", f"0 0 {pelvis_height:.12g}")
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    head_geom = model.geom("physical/Head").id
    standing_height = float(data.geom_xpos[head_geom, 2] + model.geom_size[head_geom, 2])
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                   for j in range(1, model.njnt)]
    site_map = {name: model.site(name).id for name in
                [f"{side}{part}" for side in ("Left", "Right")
                 for part in ("Hand_center", "Foot_sole", "Foot_sole_0", "Foot_sole_1",
                              "Foot_sole_2", "Foot_sole_3")]}
    metadata = dict(
        xml_path=str(path), asset="SOMA23-constrained-48", source_revision=provenance["revision"],
        source_xml_sha256=digest, xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        mass_kg=float(model.body_mass.sum()), native_scale=True, standing_height_m=standing_height,
        timestep_s=timestep, nq=model.nq, nv=model.nv, nu=model.nu,
        root_body="Hips", root_joint="Hips", root_actuated=False,
        neutral_qpos=model.qpos0.tolist(), neutral_pose="standing T pose, world forward +X",
        canonical_axes="+Z up, -Y forward, +X left; root neutral quaternion +90deg Z",
        body_map={name: model.body(name).id for name in body_names}, site_map=site_map,
        joint_names=joint_names, joints=joint_rows,
        kp=[row["kp"] for row in joint_rows], kd=[row["kd"] for row in joint_rows],
        effort=[row["effort_nm"] for row in joint_rows],
        joint_limits_rad=model.jnt_range[1:].tolist(),
        qpos_addresses=model.jnt_qposadr[1:].tolist(), dof_addresses=model.jnt_dofadr[1:].tolist(),
        passive_stiffness=0, passive_damping=0, actuator_gear=1,
        self_collision_pairs=pairs, human_floor_contact=True, glove_radius_m=.08,
        sole_height_difference_m=float(abs(data.site_xpos[sole_ids[0], 2] -
                                           data.site_xpos[sole_ids[1], 2])),
        calibrated_human_biomechanics=False,
        limitations=["Engineering joint limits, segment masses and effort caps; not person calibrated",
                     "Selected self-collision pairs, not all body-part pairs",
                     "Rigid fixed feet and fists; neck/head motion reduced to Neck1",
                     "Native skeleton retained but primitive collision/visual envelopes replaced"],
    )
    (output / "scalar_model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/scalar-body")
    parser.add_argument("--timestep", type=float, default=.005)
    args = parser.parse_args()
    result = build_scalar_human(args.output, args.timestep)
    print(json.dumps({key: result[key] for key in
                      ("xml_path", "standing_height_m", "mass_kg", "nq", "nv", "nu")}, indent=2))
