"""Offline static-guard support LP: bounded joint feedforward, never base support.

The LP allocates hypothetical floor reactions to satisfy all six free-base
equations. Only its 48 joint torques may enter the native affine actuators. Its
idealized support solution is a diagnostic, not measured contact force or proof
that the guard will remain stable. No simulation integration occurs here.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .provenance import (PROJECT_ROOT, configure_local_caches, runtime_record,
                         sha256_file, writable_path, write_json)
from .runtime.bundle import SOURCE_PINS, load_runtime_bundle

SCHEMA = "human_wbc_static_guard_feedforward_v1"
MODE = "static_guard_support_lp_joint_feedforward"


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def source_identity() -> dict:
    return {name: sha256_file(PROJECT_ROOT/"src/human_punching/wbc"/name) for name in (
        "guard_feedforward.py", "guard_probe.py", "effort_diagnostics.py",
        "runtime/body.py", "runtime/action.py", "runtime/task.py")}


def solver_contract(bundle) -> dict:
    from .runtime.task import build_human_env_cfg
    cfg = build_human_env_cfg(bundle, num_envs=16, evaluation=True)
    return json.loads(json.dumps({"mujoco": asdict(cfg.sim.mujoco), "decimation": cfg.decimation,
                                 "nconmax": cfg.sim.nconmax, "njmax": cfg.sim.njmax}))


def verify_equilibrium(payload: dict) -> dict:
    """Pure algebraic integrity checks; no LP or physics is rerun when loading."""
    required = np.asarray(payload["required_generalized_force"], dtype=float)
    matrix = np.asarray(payload["generalized_contact_edges"], dtype=float)
    loads = np.asarray(payload["hypothetical_contact_edge_loads_n"], dtype=float)
    torque = np.asarray(payload["joint_feedforward_nm"], dtype=float)
    caps = np.asarray(payload["effort_caps_nm"], dtype=float)
    if (required.shape != (54,) or matrix.shape != (54, 32) or loads.shape != (32,)
            or torque.shape != (48,) or caps.shape != (48,)
            or not all(np.isfinite(value).all() for value in (required, matrix, loads, torque, caps))
            or np.any(caps <= 0) or np.any(loads < -1e-8)):
        raise ValueError("Invalid static-guard equilibrium arrays")
    if payload.get("applied_base_wrench") != [0.]*6 or payload.get("external_support_applied") is not False:
        raise ValueError("Static feedforward must never apply a base wrench or external support")
    residual = required-matrix@loads-np.r_[np.zeros(6), torque]
    ratio = float(np.max(abs(torque)/caps))
    if np.max(abs(residual)) > 1e-5 or ratio > 1.+1e-6:
        raise ValueError("Static joint feedforward violates equilibrium or existing effort limits")
    if not np.allclose(payload.get("qvel"), np.zeros(54), atol=0, rtol=0):
        raise ValueError("Static feedforward requires zero reference velocity")
    return {"maximum_equilibrium_residual": float(np.max(abs(residual))),
            "maximum_base_force_residual_n": float(np.max(abs(residual[:3]))),
            "maximum_base_moment_residual_nm": float(np.max(abs(residual[3:6]))),
            "peak_joint_effort_ratio": ratio}


def generate_guard_feedforward(bundle, output: Path) -> dict:
    """Create one explicit, new offline artifact; never alter source references."""
    configure_local_caches()
    import mujoco
    from .effort_diagnostics import contact_edges, support_effort_solution, unconstrained_inverse
    from .guard_probe import common_guard_identity
    checked = load_runtime_bundle(bundle)
    guard = common_guard_identity(checked)
    runtime = runtime_record()
    if not runtime["declared_pins_match"]:
        raise ValueError("Static feedforward generation requires the pinned candidate runtime")
    destination = writable_path(output)
    if destination.exists():
        raise FileExistsError(destination)
    model = mujoco.MjModel.from_xml_path(str(checked.model_path))
    execution = solver_contract(checked)
    if not np.array_equal(model.opt.gravity, execution["mujoco"]["gravity"]):
        raise ValueError("Source inverse-statics gravity differs from the selected execution solver")
    caps = np.asarray(checked.body["effort"], dtype=float)
    if (model.nq != 55 or model.nv != 54 or model.nu != 48
            or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
            or not np.array_equal(model.actuator_trnid[:, 0], np.arange(1, 49))
            or not np.allclose(model.actuator_gear[:, 0], 1.)
            or not np.allclose(model.actuator_forcerange, np.column_stack((-caps, caps)))):
        raise ValueError("Static LP expects the unchanged constrained human actuator mapping/caps")
    original_flags = {"disableflags": int(model.opt.disableflags), "enableflags": int(model.opt.enableflags)}
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    if hasattr(mujoco.mjtEnableBit, "mjENBL_INVDISCRETE"):
        model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
    data = mujoco.MjData(model)
    required = unconstrained_inverse(model, data, np.asarray(guard["qpos"]), np.zeros(54), np.zeros(54))
    matrix, points, friction = contact_edges(model, data)
    solution = support_effort_solution(required, matrix, caps)
    if not solution["support_feasible"] or not solution["effort_within_caps"]:
        raise ValueError("No static support-LP solution within current human effort limits")
    payload = {"schema": SCHEMA, "status": "offline_equilibrium_candidate", "mode": MODE,
        "physics_executed": False, "qualified_for_training": False, "guard_admission_passed": False,
        "reference_manifest_sha256": sha256_file(checked.root/"manifest.json"),
        "model_sha256": sha256_file(checked.model_path), "source_controller_sha256": checked.controller_sha256(),
        "joint_names": list(checked.joint_names), "qpos": guard["qpos"], "qpos_sha256": guard["qpos_sha256"],
        "qvel": [0.]*54, "qacc": [0.]*54, "joint_feedforward_nm": solution["joint_torque"].tolist(),
        "effort_caps_nm": caps.tolist(), "applied_base_wrench": [0.]*6, "external_support_applied": False,
        "required_generalized_force": required.tolist(), "generalized_contact_edges": matrix.tolist(),
        "hypothetical_contact_edge_loads_n": solution["edge_loads"].tolist(),
        "hypothetical_projected_support_points_w": points.tolist(), "friction_mu": friction.tolist(),
        "executed_solver_contract": execution, "source_pins": SOURCE_PINS,
        "source_sha256": source_identity(), "generation_runtime": runtime,
        "diagnostic_original_flags": original_flags,
        "diagnostic_constraint_forces_disabled": True, "diagnostic_discrete_inverse_disabled": True,
        "support_assumption": "Eight projected foot-box corners with conservative |fx|+|fy|<=mu*fz pyramids; hypothetical floor reactions, never applied support forces.",
        "interpretation": "Joint feedforward candidate for a separate native guard test; not a physically qualified guard or punch reference."}
    payload["equilibrium_check"] = verify_equilibrium(payload)
    payload["payload_sha256"] = _digest(payload)
    write_json(destination, payload)
    return {"path": str(destination), "sha256": sha256_file(destination),
            "peak_joint_effort_ratio": payload["equilibrium_check"]["peak_joint_effort_ratio"],
            "physics_executed": False, "qualified_for_training": False}


def load_guard_feedforward(path: Path, bundle, *, actual_solver_contract: dict | None = None) -> dict:
    """Fail closed on changed pose, body, controller, solver, source or torque."""
    from .guard_probe import common_guard_identity
    checked = load_runtime_bundle(bundle)
    payload = json.loads(Path(path).read_text())
    stored = payload.get("payload_sha256")
    if stored != _digest({key: value for key, value in payload.items() if key != "payload_sha256"}):
        raise ValueError("Static feedforward payload digest mismatch")
    guard = common_guard_identity(checked)
    expected = {"schema": SCHEMA, "status": "offline_equilibrium_candidate", "mode": MODE,
        "physics_executed": False, "qualified_for_training": False, "guard_admission_passed": False,
        "model_sha256": sha256_file(checked.model_path), "source_controller_sha256": checked.controller_sha256(),
        "reference_manifest_sha256": sha256_file(checked.root/"manifest.json"),
        "qpos": guard["qpos"], "qpos_sha256": guard["qpos_sha256"],
        "joint_names": list(checked.joint_names), "effort_caps_nm": checked.body["effort"],
        "executed_solver_contract": actual_solver_contract or solver_contract(checked),
        "source_pins": SOURCE_PINS, "source_sha256": source_identity(), "qacc": [0.]*54}
    changed = [key for key, value in expected.items() if _digest(payload.get(key)) != _digest(value)]
    if changed:
        raise ValueError(f"Static feedforward pose/body/controller/solver/source contract mismatch: {changed}")
    verify_equilibrium(payload)
    return payload


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(generate_guard_feedforward(args.bundle, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
