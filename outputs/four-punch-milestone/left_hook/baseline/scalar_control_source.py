"""Bounded native PD actuation for the scalar human at a genuine 5 ms step.

The native actuator has force ``u - kp*q - kd*v`` and receives
``u = kp*q_target + kd*v_target + feedforward``. This is the same continuous
force law as explicit PD, but exposes the damping derivative to implicitfast.
Output force is clamped *after* all terms, with the original motor strength.
No passive damping, extra inertia, root force or physics substep is introduced.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def scalar_actuator_layout(model):
    """Return joint/qpos/dof IDs, rejecting root and duplicate transmissions."""
    ids = model.actuator_trnid[:, 0].astype(int)
    if np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT):
        raise ValueError("Expected only direct joint transmissions")
    if np.any(model.jnt_type[ids] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Expected only scalar hinge actuators")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Expected one actuator per hinge")
    if not np.allclose(model.actuator_gear, [1, 0, 0, 0, 0, 0]):
        raise ValueError("Expected unit transmission gears")
    qadr, vadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    if np.any(qadr < 7) or np.any(vadr < 6):
        raise ValueError("The free root must remain unactuated")
    return ids, qadr, vadr


def actuator_torque_ranges(model):
    """Compiled output torque caps; ctrl is not torque for native PD."""
    scalar_actuator_layout(model)
    ranges = np.zeros((model.nu, 2))
    for i in range(model.nu):
        if model.actuator_forcelimited[i]:
            ranges[i] = model.actuator_forcerange[i]
        elif (model.actuator_ctrllimited[i]
              and model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_FIXED
              and model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE
              and model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_NONE):
            ranges[i] = np.sort(model.actuator_ctrlrange[i] * model.actuator_gainprm[i, 0])
        else:
            raise ValueError("Every joint must have a provable actuator output torque cap")
    if not np.isfinite(ranges).all() or np.any(ranges[:, 0] >= ranges[:, 1]):
        raise ValueError("Invalid actuator torque ranges")
    return ranges


def configure_implicit_pd(model, kp, kd):
    """Convert a loaded motor model in place, before constructing Simulation.

    Returns the unchanged output strength limits. The original source XML is
    unaffected. Save the runtime model/schema for independent torque auditing.
    """
    scalar_actuator_layout(model)
    if np.any(model.actuator_dyntype != mujoco.mjtDyn.mjDYN_NONE):
        raise ValueError("Native scalar PD requires actuators without activation dynamics")
    kp, kd = np.asarray(kp, dtype=float), np.asarray(kd, dtype=float)
    if kp.shape != (model.nu,) or kd.shape != kp.shape:
        raise ValueError("kp and kd must follow actuator order")
    if not np.isfinite(kp).all() or not np.isfinite(kd).all() or np.any(kp <= 0) or np.any(kd < 0):
        raise ValueError("Expected finite kp > 0 and kd >= 0")
    limits = actuator_torque_ranges(model)
    model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_gainprm[:] = 0
    model.actuator_gainprm[:, 0] = 1
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_biasprm[:] = 0
    model.actuator_biasprm[:, 1], model.actuator_biasprm[:, 2] = -kp, -kd
    model.actuator_ctrllimited[:] = False
    model.actuator_forcelimited[:] = True
    model.actuator_forcerange[:] = limits
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return limits


def native_pd_drive(target, velocity_target, kp, kd, feedforward=0):
    """Native actuator input, supports NumPy arrays and Torch tensors."""
    return kp * target + kd * velocity_target + feedforward


def export_implicit_pd_model(source, destination, kp, kd, *, timestep=.005):
    """Write a new standalone runtime XML, preserving source body and geometry."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Refusing to overwrite the source body asset")
    model = mujoco.MjModel.from_xml_path(str(source))
    limits = configure_implicit_pd(model, kp, kd)
    tree = ET.parse(source)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(timestep))
    option.set("integrator", "implicitfast")
    actuator = root.find("actuator")
    if actuator is None or len(actuator) != model.nu:
        raise ValueError("Expected standalone XML with explicit actuator elements")
    for i, node in enumerate(actuator):
        name, joint = node.get("name"), node.get("joint")
        node.clear()
        node.tag = "general"
        node.attrib.update(name=name, joint=joint, gear="1", gaintype="fixed", gainprm="1",
                           biastype="affine", biasprm=f"0 {-float(kp[i]):.17g} {-float(kd[i]):.17g}",
                           dyntype="none", ctrllimited="false", forcelimited="true",
                           forcerange=" ".join(f"{x:.17g}" for x in limits[i]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")
    return destination


def export_runtime_model(source, destination, model):
    """Export the actual scalar actuator/solver contract without body rounding.

    ``mj_saveLastXML`` copies numeric actuator parameters from an edited MjModel
    but does not reliably copy changed enum/boolean actuator fields. In
    particular an affine force law may silently recompile as a plain motor.
    Write those fields explicitly into a copy of the exact source body XML,
    then recompile and verify the runtime contract before accepting the file.
    This helper intentionally supports only this project's scalar, fixed-gain,
    no-activation actuator models; it rejects changes it cannot serialize.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Refusing to overwrite the source body asset")
    original = mujoco.MjModel.from_xml_path(str(source))
    scalar_actuator_layout(model)
    actuator_torque_ranges(model)
    body_fields = ("body_mass", "body_inertia", "body_pos", "body_quat", "jnt_range",
                   "jnt_pos", "jnt_axis", "dof_armature", "dof_damping", "dof_frictionloss",
                   "geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_solref",
                   "geom_solimp", "geom_contype", "geom_conaffinity")
    if (original.nq, original.nv, original.nu) != (model.nq, model.nv, model.nu):
        raise ValueError("Runtime dimensions differ from source body")
    for key in body_fields:
        if not np.array_equal(getattr(original, key), getattr(model, key)):
            raise ValueError(f"Runtime body changed; source-preserving export cannot serialize {key}")
    if np.any(model.actuator_dyntype != mujoco.mjtDyn.mjDYN_NONE):
        raise ValueError("Runtime exporter supports only no-activation actuators")
    if np.any(model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED):
        raise ValueError("Runtime exporter supports only fixed-gain actuators")
    bias_names = {int(mujoco.mjtBias.mjBIAS_NONE): "none",
                  int(mujoco.mjtBias.mjBIAS_AFFINE): "affine"}
    if any(int(value) not in bias_names for value in model.actuator_biastype):
        raise ValueError("Runtime exporter supports only no-bias or affine-bias actuators")
    tree = ET.parse(source)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    numeric_options = ("timestep", "impratio", "tolerance", "ls_tolerance", "iterations",
                       "ls_iterations", "ccd_iterations")
    for key in numeric_options:
        option.set(key, str(getattr(model.opt, key)))
    option.set("gravity", " ".join(f"{x:.17g}" for x in model.opt.gravity))
    enum_options = dict(integrator=("Euler", "RK4", "implicit", "implicitfast"),
                        cone=("pyramidal", "elliptic"), jacobian=("dense", "sparse", "auto"),
                        solver=("PGS", "CG", "Newton"))
    for key, names in enum_options.items():
        value = int(getattr(model.opt, key))
        if value >= len(names):
            raise ValueError(f"Unsupported runtime {key} enum {value}")
        option.set(key, names[value])
    flags = option.find("flag")
    if flags is not None:
        option.remove(flags)
    flag_attributes = {}
    for enum, prefix, bitfield, setting in (
        (mujoco.mjtDisableBit, "mjDSBL_", model.opt.disableflags, "disable"),
        (mujoco.mjtEnableBit, "mjENBL_", model.opt.enableflags, "enable"),
    ):
        for name in dir(enum):
            if name.startswith(prefix) and int(getattr(enum, name)) & int(bitfield):
                flag_attributes[name.removeprefix(prefix).lower()] = setting
    if flag_attributes:
        ET.SubElement(option, "flag", flag_attributes)
    actuator = root.find("actuator")
    if actuator is None or len(actuator) != model.nu:
        raise ValueError("Expected standalone XML with explicit actuator elements")
    for i, node in enumerate(actuator):
        name = model.actuator(i).name
        joint = model.joint(int(model.actuator_trnid[i, 0])).name
        if node.get("name") != name or node.get("joint") != joint:
            raise ValueError("Source actuator order differs from compiled model")
        node.clear()
        node.tag = "general"
        node.attrib.update(name=name, joint=joint, dyntype="none", gaintype="fixed",
                           biastype=bias_names[int(model.actuator_biastype[i])],
                           ctrllimited=str(bool(model.actuator_ctrllimited[i])).lower(),
                           forcelimited=str(bool(model.actuator_forcelimited[i])).lower())
        for key in ("gear", "gainprm", "biasprm", "dynprm", "ctrlrange", "forcerange"):
            node.set(key, " ".join(f"{x:.17g}" for x in getattr(model, f"actuator_{key}")[i]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")
    compiled = mujoco.MjModel.from_xml_path(str(destination))
    actuator_fields = ("actuator_trnid", "actuator_trntype", "actuator_dyntype",
                       "actuator_gaintype", "actuator_biastype", "actuator_ctrllimited",
                       "actuator_forcelimited", "actuator_gear", "actuator_gainprm",
                       "actuator_biasprm", "actuator_dynprm", "actuator_ctrlrange",
                       "actuator_forcerange")
    for key in (*body_fields, *actuator_fields):
        if not np.array_equal(getattr(compiled, key), getattr(model, key)):
            raise ValueError(f"Runtime XML round trip changed {key}")
    for key in (*numeric_options, *enum_options, "gravity", "disableflags", "enableflags"):
        if not np.array_equal(getattr(compiled.opt, key), getattr(model.opt, key)):
            raise ValueError(f"Runtime XML round trip changed option {key}")
    return destination
