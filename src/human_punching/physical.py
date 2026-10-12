"""Source-compatible rigid-body pilot, constrained references and bounded control.

This 60-motor ball-joint rig is an engineering prototype, not human biomechanics.
The pelvis is always free and never actuated. No state is overwritten after reset.
"""
from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .human import _fmt
from .motion import JOINT_NAMES

# Segment masses are proportions, normalized to the requested total incl. gloves.
MASS_WEIGHTS = [10, 8, 14, 1, 4.5, .3, 2.2, 1.3, .9, .3, 2.2, 1.3, .9,
                7.5, 3.8, .9, .1, 7.5, 3.8, .9, .1]
# torque-vector norm bound [Nm], Kp [Nm/rad], Kd [Nm s/rad], cone [deg].
PARAMETERS = {
    "Spine": (150, 220, 22, 55), "Spine1": (150, 220, 22, 55),
    "Neck": (20, 30, 3, 60), "Head": (20, 30, 3, 60),
    "Shoulder": (35, 55, 5, 90), "Arm": (100, 150, 12, 175),
    "ForeArm": (60, 100, 8, 165), "Hand": (15, 25, 2, 100),
    "UpLeg": (200, 350, 30, 150), "Leg": (150, 350, 30, 165),
    "Foot": (90, 220, 18, 100), "ToeBase": (8, 15, 1, 40),
}


def parameters(name):
    return PARAMETERS[name.removeprefix("Left").removeprefix("Right")]


def build_physical_human(source_xml: Path, output: Path, mass_kg=75., timestep=.002):
    """Copy blue visuals and infer link inertias from assigned primitive masses.

    Floor-only contact is intentional in this first pilot. Human self collision
    and impact response with G1 are disabled and reported in the metadata.
    """
    source_xml, output = Path(source_xml).resolve(), Path(output).resolve()
    if mass_kg <= 0 or not 0 < timestep <= .005:
        raise ValueError("Positive mass and timestep <= 5 ms required")
    output.mkdir(parents=True, exist_ok=True)
    root = ET.parse(source_xml).getroot()
    root.set("model", "HumanPunching_physical_ball_pilot")
    compiler = root.find("compiler")
    mesh_source = source_xml.parent / compiler.get("meshdir", "")
    shutil.copytree(mesh_source, output / "meshes", dirs_exist_ok=True)
    compiler.set("meshdir", "meshes")
    compiler.set("inertiafromgeom", "true")
    compiler.set("inertiagrouprange", "3 3")
    # Both gloves must be red for the existing Task08 HSV detector contract.
    root.find("asset/material[@name='left_glove']").set("rgba", "0.70 0.12 0.10 1")
    root.find("option").attrib.update(timestep=str(timestep), gravity="0 0 -9.81",
                                      integrator="implicitfast", cone="elliptic",
                                      iterations="60", tolerance="1e-9")
    # Do not inherit placeholder inertia or contact-free defaults.
    default_joint = root.find("default/joint")
    default_joint.attrib.update(damping="0.05", armature="0.015", stiffness="0")
    world = root.find("worldbody")
    floor = world.find("geom[@name='floor']")
    floor.attrib.update(pos="0 0 0", contype="1", conaffinity="2", friction="0.8 0.02 0.002")
    actuators = ET.SubElement(root, "actuator")
    rows = []
    weights = np.array(MASS_WEIGHTS) / sum(MASS_WEIGHTS) * mass_kg
    for idx, name in enumerate(JOINT_NAMES):
        body = world.find(f".//body[@name='{name}']")
        for inertia in body.findall("inertial"):
            body.remove(inertia)
        for geom in body.findall("geom"):
            geom.set("mass", "0")
            geom.set("group", "0")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        kind = name.removeprefix("Left").removeprefix("Right")
        attrs = dict(name=f"physical/{name}", group="3", mass=str(weights[idx]),
                     rgba="0.2 0.8 0.6 0.28", contype="2", conaffinity="1",
                     friction="0.8 0.02 0.002", condim="3", solref="0.012 1")
        children = body.findall("body")
        if name == "Hips":
            attrs.update(type="ellipsoid", pos="0 -0.025 0", size="0.14 0.095 0.09")
        elif kind in ("Foot", "ToeBase"):
            if kind == "Foot":
                # Flat sole, in the foot's Y-up source coordinates.
                toe = np.fromstring(children[0].get("pos"), sep=" ")
                center = np.array([0, toe[1] * .65, toe[2] * .48])
                attrs.update(type="box", pos=_fmt(center), size="0.05 0.04 0.115")
                ET.SubElement(body, "site", name=f"{name}_sole",
                              pos=_fmt(center + [0, -.04, 0]), size=".007", group="3")
            else:
                attrs.update(type="sphere", size=".015", contype="0", conaffinity="0")
        elif kind == "Hand":
            # Match the actual attached visual mesh center, not the wrist origin.
            visual_model = mujoco.MjModel.from_xml_path(str(source_xml))
            g = visual_model.geom(f"{name.removesuffix('Hand')}_boxing_glove").id
            center = visual_model.geom_pos[g]
            attrs.update(type="sphere", pos=_fmt(center), size=".08")
            ET.SubElement(body, "site", name=f"{name}_center", pos=_fmt(center),
                          size=".006", group="3")
        elif name == "Head":
            attrs.update(type="ellipsoid", pos="0 .02 0", size=".08 .12 .08")
        else:
            # Anatomical axial sections following the actual bind offsets.
            if name in ("Spine", "Spine1", "Neck"):
                child = next(c for c in children if c.get("name") in ("Spine1", "Neck", "Head"))
            else:
                child = children[0]
            end = np.fromstring(child.get("pos"), sep=" ")
            radius = {"Spine": .105, "Spine1": .12, "Neck": .045,
                      "Shoulder": .04, "Arm": .043, "ForeArm": .032,
                      "UpLeg": .065, "Leg": .045}[kind]
            attrs.update(type="capsule", size=str(radius), fromto=_fmt(np.r_[end * .12, end * .88]))
        ET.SubElement(body, "geom", **attrs)
        if idx:
            effort, kp, kd, angle = parameters(name)
            body.find("joint").attrib.update(limited="true", range=f"0 {np.deg2rad(angle)}")
            for axis in range(3):
                gear = np.zeros(6)
                gear[axis] = 1
                ET.SubElement(actuators, "motor", name=f"{name}_motor_{axis}",
                              joint=f"{name}_joint", gear=_fmt(gear),
                              ctrllimited="true", ctrlrange=f"{-effort} {effort}",
                              forcelimited="true", forcerange=f"{-effort} {effort}")
            rows.append(dict(body=name, effort_norm_nm=effort, kp=kp, kd=kd,
                             rotation_cone_deg=angle, mass_kg=float(weights[idx])))
    # Extra fixed camera also permits Warp RGB/depth/id preflight of this model.
    ET.SubElement(world, "camera", name="physical_observer", pos="3 -4 2.3",
                  xyaxes="0.8 0.6 0 -0.17 0.226 0.959", fovy="45")
    path = output / "human_physical.xml"
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    metadata = dict(xml_path=str(path), mass_kg=float(model.body_mass.sum()),
                    timestep_s=timestep, nq=model.nq, nv=model.nv, nu=model.nu,
                    root="free, unactuated", model_kind="20 ball joints / 60 torque motors",
                    joint_limits="isotropic rotation cones; not anatomical axis limits",
                    inertia="MuJoCo primitive geometry mass integration", controller=rows,
                    self_collision=False, human_floor_contact=True, human_g1_impacts=False,
                    source_xml=str(source_xml), calibrated_human_biomechanics=False)
    (output / "physical_model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def sole_targets(model, initial_pose):
    """Two planted, flat physical soles, retaining the initial stance footprint."""
    data = mujoco.MjData(model)
    data.qpos[:] = initial_pose
    mujoco.mj_forward(model, data)
    targets = []
    for side in ("Left", "Right"):
        site = model.site(f"{side}Foot_sole").id
        body = model.body(f"{side}Foot").id
        rot = data.xmat[body].reshape(3, 3)
        forward = rot[:, 2].copy()
        forward[2] = 0
        forward /= np.linalg.norm(forward)
        up = np.array([0., 0., 1.])
        desired_rotation = np.column_stack((np.cross(up, forward), up, forward))
        pos = data.site_xpos[site].copy()
        pos[2] = .0005
        targets.append((site, body, pos, desired_rotation))
    return targets


def plant_feet(model, source_qpos, *, iterations=45):
    """Whole-clip double-support reference, with COM and fixed sole pose IK.

    The upper-body rotations retain the punch. Root and both legs solve foot
    support, while a soft COM objective brings weight inside the footprint.
    This deliberately creates a planted-stance variant, not recovered mocap feet.
    """
    source_qpos = np.asarray(source_qpos)
    qpos = source_qpos.copy()
    targets = sole_targets(model, qpos[0])
    dofs = np.concatenate([np.arange(6), *[np.arange(model.joint(f"{s}{p}_joint").dofadr[0],
                           model.joint(f"{s}{p}_joint").dofadr[0] + 3)
                           for s in ("Left", "Right") for p in ("UpLeg", "Leg", "Foot")]])
    data = mujoco.MjData(model)
    jp, jr, jc = (np.zeros((3, model.nv)) for _ in range(3))
    delta, prior = np.zeros(model.nv), np.zeros(model.nv)
    center = np.mean([t[2] for t in targets], axis=0)
    errors, com_errors, changes = [], [], []
    for frame, reference in enumerate(source_qpos):
        pose = reference.copy()
        if frame:
            pose[:3] += qpos[frame - 1, :3] - source_qpos[frame - 1, :3]
        for _ in range(iterations):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            matrices, residuals = [], []
            for site, body, target_pos, target_rot in targets:
                mujoco.mj_jacSite(model, data, jp, jr, site)
                matrices.extend((jp[:, dofs], .12 * jr[:, dofs]))
                residuals.extend((target_pos - data.site_xpos[site], .12 * Rotation.from_matrix(
                    target_rot @ data.xmat[body].reshape(3, 3).T).as_rotvec()))
            mujoco.mj_jacSubtreeCom(model, data, jc, model.body("Hips").id)
            matrices.append(.7 * jc[:2, dofs])
            residuals.append(.7 * (center[:2] - data.subtree_com[model.body("Hips").id, :2]))
            mujoco.mj_differentiatePos(model, prior, 1., pose, reference)
            regularizer = np.full(len(dofs), .005)
            regularizer[3:6] = .08
            matrices.append(np.diag(regularizer))
            residuals.append(regularizer * prior[dofs])
            step = np.linalg.lstsq(np.vstack(matrices), np.concatenate(residuals), rcond=None)[0]
            step *= min(1., .10 / max(np.max(np.abs(step)), 1e-10))
            delta[:] = 0
            delta[dofs] = step
            mujoco.mj_integratePos(model, pose, delta, 1.)
            if np.max(np.abs(step)) < 1e-6:
                break
        qpos[frame] = pose
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        errors.append(max(np.linalg.norm(data.site_xpos[t[0]] - t[2]) for t in targets))
        com_errors.append(np.linalg.norm(data.subtree_com[model.body("Hips").id, :2] - center[:2]))
        mujoco.mj_differentiatePos(model, prior, 1, reference, pose)
        changes.append(np.linalg.norm(prior))
    return qpos, dict(foot_position_max_error_m=float(max(errors)),
                      com_horizontal_max_error_m=float(max(com_errors)),
                      change_norm_max=float(max(changes)),
                      sole_target_positions=[t[2].tolist() for t in targets],
                      method="planted double support, root/leg IK and soft COM objective")



def smooth_reference(model, qpos, sigma_frames=4.):
    """Remove source-rate quaternion jitter with sign-continuous local smoothing.

    Preserves exact endpoints; this is an explicitly altered reference, not a
    claim to preserve capture timing/peaks. Physical retargeting follows it.
    """
    from scipy.ndimage import gaussian_filter1d
    result = np.asarray(qpos).copy()
    filtered = gaussian_filter1d(result[:, :3], sigma_frames, axis=0, mode="nearest")
    envelope = np.minimum(np.arange(len(result)), np.arange(len(result))[::-1])
    envelope = np.clip(envelope / (3 * sigma_frames), 0, 1)
    result[:, :3] += envelope[:, None] * (filtered - result[:, :3])
    for address in [3] + [int(model.jnt_qposadr[j]) for j in range(1, model.njnt)]:
        quats = result[:, address:address+4].copy()
        for frame in range(1, len(quats)):
            if quats[frame] @ quats[frame-1] < 0:
                quats[frame] *= -1
        filtered = gaussian_filter1d(quats, sigma_frames, axis=0, mode="nearest")
        quats += envelope[:, None] * (filtered - quats)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        result[:, address:address+4] = quats
    return result

def controller_arrays(model):
    kp, kd, limits = [], [], []
    for name in JOINT_NAMES[1:]:
        effort, p, d, _ = parameters(name)
        kp.extend([p] * 3)
        kd.extend([d] * 3)
        limits.append(effort)
    return np.r_[np.zeros(6), kp], np.r_[np.zeros(6), kd], np.array(limits)


def bounded_pd(model, data, target_qpos, target_qvel, gain_scale=1., stable=True, balance=False):
    """Quaternion-error PD with per-joint norm limits, no root wrench.

    A mass-matrix implicit-derivative approximation stabilizes the PD torque;
    contact forces are left to MuJoCo. This is not inverse-dynamics tracking.
    """
    kp, kd, limits = controller_arrays(model)
    kp *= gain_scale
    kd *= np.sqrt(gain_scale)
    error = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, error, 1., data.qpos, target_qpos)
    dt = model.opt.timestep
    p = kp * (error - dt * data.qvel) if stable else kp * error
    d = kd * (target_qvel - data.qvel)
    torque = p + d
    if stable:
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)
        # Root entries have zero Kp/Kd; acceleration can still include gravity.
        accel = np.linalg.solve(mass + np.diag(kd * dt), p + d - data.qfrc_bias)
        torque -= kd * dt * accel
    if balance:
        # Quasi-static, double-support inverse statics. The solved forces are
        # bookkeeping only: they are NEVER applied to the root or contact data.
        # Actual ground reaction must arise from bounded joint torques + contact.
        jacobians = []
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for side in ("Left", "Right"):
            geom = model.geom(f"physical/{side}Foot").id
            body = int(model.geom_bodyid[geom])
            rot = data.geom_xmat[geom].reshape(3, 3)
            for x in (-.04, .04):
                for z in (-.10, .10):
                    point = data.geom_xpos[geom] + rot @ np.array([x, -.04, z])
                    mujoco.mj_jac(model, data, jp, jr, point, body)
                    jacobians.append(jp.copy())
        jac = np.vstack(jacobians)
        # Root acceleration feedback acts through the contact-aware joint torques.
        root_acc = 12. * error[:6] - 5. * (data.qvel[:6] - target_qvel[:6])
        mass = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)
        required = data.qfrc_bias[:6] + mass[:6, :6] @ root_acc
        forces = np.linalg.lstsq(jac[:, :6].T, required, rcond=None)[0].reshape(-1, 3)
        # Conservative friction-pyramid projection; residual is not disguised as
        # a feasible contact solve and the rollout still has to pass support gates.
        forces[:, 2] = np.maximum(forces[:, 2], 0)
        forces[:, :2] = np.clip(forces[:, :2], -.5*forces[:, 2, None], .5*forces[:, 2, None])
        torque[6:] += data.qfrc_bias[6:] - (jac.T @ forces.ravel())[6:]
    joints = torque[6:].reshape(-1, 3)
    raw_norms = np.linalg.norm(joints, axis=1)
    joints *= np.minimum(1., limits / np.maximum(raw_norms, 1e-10))[:, None]
    data.ctrl[:] = joints.ravel()
    return float(np.mean(raw_norms >= limits)), float(np.max(raw_norms / limits))


def interpolate_pose(model, times, poses, t):
    """Shortest-arc interpolation in generalized coordinates; zero endpoint speed."""
    right = int(np.clip(np.searchsorted(times, t, side="right"), 1, len(times) - 1))
    left = right - 1
    dt = float(times[right] - times[left])
    velocity = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, velocity, dt, poses[left], poses[right])
    pose = poses[left].copy()
    mujoco.mj_integratePos(model, pose, velocity, float(np.clip(t-times[left], 0, dt)))
    if t <= times[0] or t >= times[-1]:
        velocity[:] = 0
    return pose, velocity
