"""Preserve Task 09 references/controllers for a diagnostic WBC baseline.

This converter intentionally does not redesign guards, retime punches, fabricate
G1 snapshots, differentiate interpolated poses, or qualify production training.
It samples the frozen controller's original interpolation and inverse-statics
feedforward. Only CPU kinematics are executed; no physics integration or learning.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
import torch

from .provenance import PROJECT_ROOT, configure_local_caches, write_json
from .references import (FAMILIES, ReferenceClip, kinematic_arrays, project_output,
                         sha256, validate_model_order)
from .runtime.bundle import load_runtime_bundle

DIAGNOSTIC_SCOPE = "task09_reference_controller_migration"
FEEDFORWARD_MODE = "task09_source_inverse_statics"


def frozen_source(release_root: Path, family: str):
    """Read the manifest-verified historical numerical helpers, never its trainer."""
    root = Path(release_root).resolve(strict=True)
    if family not in FAMILIES:
        raise ValueError("Unknown punch family")
    manifest = json.loads((root / "manifest.json").read_text())
    files = {}
    for name in ("reference.npz", "model/human.xml", "body_metadata.json",
                 "controller_schema.json", "reference_manifest.json", "arguments.json",
                 "jab_learning_source.py", "scalar_control_source.py"):
        relative = f"{family}/{name}"
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or sha256(path) != manifest["artifacts"][relative]["sha256"]:
            raise ValueError(f"Frozen release input differs: {relative}")
        files[name] = path
    # The preserved module imports this package helper by name. Bind the file
    # actually imported, not merely an unused archived copy with the same name.
    from human_punching import scalar_control
    imported_helper = Path(scalar_control.__file__).resolve(strict=True)
    if sha256(imported_helper) != sha256(files["scalar_control_source.py"]):
        raise ValueError("Imported scalar controller helper differs from frozen release")
    files["imported_scalar_control.py"] = imported_helper
    spec = importlib.util.spec_from_file_location(f"task09_diagnostic_{family}", files["jab_learning_source.py"])
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load frozen interpolation helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, files


def source_runtime_arrays(release_root: Path, family: str):
    """Rebuild the source-grid reference/FF exactly as the preserved evaluator."""
    module, files = frozen_source(release_root, family)
    model = mujoco.MjModel.from_xml_path(str(files["model/human.xml"]))
    body = json.loads(files["body_metadata.json"].read_text())
    validate_model_order(model, body)
    schema = json.loads(files["controller_schema.json"].read_text())
    if (schema["speed"] != 1. or schema["guard_hold_s"] != 1.
            or schema["feedforward"] != "offline inverse statics joint torques"
            or schema["balance_feedback"] is not False
            or schema["action_scale_rad"] != .25
            or schema["physics_dt"] != .005 or schema["actor_dt"] != .020):
        raise ValueError("Unsupported historical controller semantics")
    source = module.load_reference(model, files["reference.npz"], speed=1.)
    source["torque"] = module.support_feedforward(model, source, np.asarray(body["dof_addresses"]))
    with np.load(files["reference.npz"], allow_pickle=False) as archive:
        timing = {name: float(archive[name]) for name in
                  ("launch_time", "return_start", "guard_hold_start", "active_first_contact_time")}
    return module, model, body, schema, source, timing, files


def sample_source(module, source: dict, query_s: np.ndarray) -> dict[str, np.ndarray]:
    """Use source float32 interpolation, including normalized-linear root quats.

    Saved qvel and source-grid FF are independently interpolated, matching the
    old controller. Recomputing either from a resampled pose changes that law.
    """
    times = torch.as_tensor(source["t"], dtype=torch.float32)
    query = torch.as_tensor(query_s, dtype=torch.float32)
    result = {}
    with torch.no_grad():
        for name in ("qpos", "qvel", "torque", "glove_pos"):
            value = module.interpolate_reference(times,
                torch.as_tensor(source[name], dtype=torch.float32), query)
            if name == "qpos":
                value[:, 3:7] /= value[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-8)
            result[name] = value.numpy().copy()
    return result


def convert_release_baseline(release_root: Path, output_root: Path) -> dict:
    """Create a new diagnostic bundle with original family guards and feedforward."""
    configure_local_caches()
    release = Path(release_root).resolve(strict=True)
    root = project_output(output_root)
    if root.exists():
        raise FileExistsError(root)
    resolved = [source_runtime_arrays(release, family) for family in FAMILIES]
    if len({sha256(item[-1]["model/human.xml"]) for item in resolved}) != 1:
        raise ValueError("All four references must use the same physical human")
    root.mkdir(parents=True)
    assets = root / "assets"
    assets.mkdir()
    for name, destination in (("model/human.xml", "human.xml"),
                              ("body_metadata.json", "body_metadata.json"),
                              ("controller_schema.json", "controller_schema.json")):
        shutil.copyfile(resolved[0][-1][name], assets / destination)
    clips, proofs = [], []
    for family, (module, model, body, schema, source, timing, files) in zip(FAMILIES, resolved, strict=True):
        duration = float(source["t"][-1])
        end_tick = int(np.ceil((duration - 1e-10) / .020)) * 4
        times = np.arange(end_tick + 1, dtype=np.float64) * .005
        sampled = sample_source(module, source, times)
        name = f"task09_{family}_preserved"
        master_path = root / "masters" / f"{name}.npz"
        motion_path = root / "motions/npz" / f"{name}.npz"
        legacy_path = root / "legacy_runtime" / f"{family}.npz"
        for path in (master_path, motion_path, legacy_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        clip = ReferenceClip(family, assets / "human.xml", times,
            sampled["qpos"], sampled["qvel"], source["head_center"],
            timing["launch_time"], timing["guard_hold_start"], sha256(files["reference.npz"]))
        arrays = kinematic_arrays(clip)
        # Keep float32 samples identical to the historical tensor law. Body FK
        # and Jacobians are computed from those poses/velocities, not old skin data.
        arrays.update(qpos=sampled["qpos"], qvel=sampled["qvel"],
            joint_pos=sampled["qpos"][:, 7:], joint_vel=sampled["qvel"][:, 6:],
            joint_feedforward=sampled["torque"], legacy_glove_reference=sampled["glove_pos"])
        np.savez_compressed(master_path, **arrays)
        names = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        np.savez_compressed(motion_path, fps=np.asarray(50.),
            **{key: arrays[key][::4] for key in names},
            joint_names=arrays["joint_names"], body_names=arrays["body_names"])
        np.savez_compressed(legacy_path, **{k: v for k, v in source.items() if v is not None})
        provenance = root / "provenance" / family
        input_hashes = {}
        for label, path in files.items():
            dest = provenance / label
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            if sha256(path) != sha256(dest):
                raise ValueError("Source changed during baseline conversion")
            input_hashes[str(path.relative_to(PROJECT_ROOT))] = sha256(path)
        distance = np.linalg.norm(arrays["glove_pos_w"] - source["head_center"], axis=-1)
        active = int(source["active_hand_index"])
        hits = np.flatnonzero((times >= timing["launch_time"])
                             & (times <= timing["return_start"]) & (distance[:, active] <= .14))
        contact = None if not len(hits) else float(times[hits[0]])
        if contact is None or not np.all(distance[0] >= .16):
            raise ValueError("Historical baseline lost geometric contact/initial clearance")
        caps = np.asarray(body["effort"])
        if np.any(abs(sampled["torque"]) > caps + 1e-4):
            raise ValueError("Source feedforward exceeds unchanged effort caps")
        clips.append({"reference_id": name, "family": family,
            "motion_path": str(motion_path.relative_to(root)), "motion_sha256": sha256(motion_path),
            "master_path": str(master_path.relative_to(root)), "master_sha256": sha256(master_path),
            "legacy_runtime_path": str(legacy_path.relative_to(root)), "legacy_runtime_sha256": sha256(legacy_path),
            "source_sha256": sha256(files["reference.npz"]), "source_inputs": input_hashes,
            "source_duration_s": duration, "legacy_episode_duration_s": duration + schema["guard_hold_s"],
            "duration_s": float(times[-1]), "launch_time_s": timing["launch_time"],
            "outward_end_s": timing["return_start"], "guard_time_s": timing["guard_hold_start"],
            "requested_ttc_s": None, "reference_contact_delay_s": contact - timing["launch_time"],
            "source_contact_reset_time_s": timing["active_first_contact_time"],
            "sampled_contact_reset_time_s": contact, "target_w": source["head_center"].tolist(),
            "guard_id": f"task09_{family}_guard", "guard_source": "task09_family_guard",
            "initial_guard_qpos": sampled["qpos"][0].tolist(), "common_guard_validated": False,
            "feedforward_mode": FEEDFORWARD_MODE, "g1_snapshot_present": False,
            "launch_features": [], "launch_feature_layout": [], "snapshot_fingerprint": None,
            "launch_snapshot": None, "qualified_for_training": False})
        proofs.append({"family": family, "master_frames": len(times), "view_frames": len(times[::4]),
            "original_timestamps_preserved_in_legacy_payload": True,
            "qvel_method": "preserved source velocities, float32 linear interpolation",
            "quaternion_method": "source sign continuity then float32 normalized linear interpolation",
            "feedforward_method": "frozen source inverse statics at original timestamps, then float32 interpolation",
            "terminal_padding_s": float(times[-1] - duration),
            "original_initial_hold_s": timing["launch_time"],
            "sampled_first_contact_since_reset_s": contact,
            "reference_schedule_is_validated_requested_ttc": False,
            "maximum_feedforward_cap_fraction": float(np.max(abs(sampled["torque"]) / caps)),
            "external_force_or_root_support": False})
    report = {"schema": "human_wbc_task09_baseline_conversion_v1", "diagnostic_scope": DIAGNOSTIC_SCOPE,
        "status": "converted_not_physically_qualified", "native_integration_steps": 0,
        "learning_updates": 0, "policy_weights_included": False,
        "clips": proofs, "converter_sha256": sha256(Path(__file__)),
        "qualification": "Known historical references/controller, newly converted WBC views; native migration remains to be tested."}
    write_json(root / "conversion_report.json", report)
    manifest = {"schema": "human_wbc_references_v1", "diagnostic_scope": DIAGNOSTIC_SCOPE,
        "qualified_for_training": False, "production_qualified": False,
        "reason": "Historical fixed target/family guards; no walking snapshot or requested-TTC domain claim",
        "release_root": str(release.relative_to(PROJECT_ROOT)),
        "release_manifest_sha256": sha256(release / "manifest.json"),
        "model_path": "assets/human.xml", "body_metadata_path": "assets/body_metadata.json",
        "controller_schema_path": "assets/controller_schema.json",
        "asset_sha256": {p.name: sha256(p) for p in assets.iterdir()},
        "physics_dt_s": .005, "policy_dt_s": .020, "decimation": 4,
        "quaternion_order": "wxyz", "body_frame_convention": "Task09 original world frame, +Z up; no placement transform",
        "feedforward_mode": FEEDFORWARD_MODE, "launch_feature_layout": [],
        "conversion_report": {"path": "conversion_report.json", "sha256": sha256(root / "conversion_report.json")},
        "clips": clips}
    write_json(root / "manifest.json", manifest)
    load_runtime_bundle(root)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = convert_release_baseline(args.release, args.output)
    print(json.dumps({"output": str(args.output), "clips": len(manifest["clips"]),
                      "qualified_for_training": False, "native_integration_steps": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
