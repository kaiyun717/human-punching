"""Presampled canonical engineering bundles, never physical qualification.

The generic reference writer resamples input trajectories. This adapter instead
preserves every supplied array and supplies missing runtime fields after fresh
kinematic/derivative checks. It submits no jobs and performs no optimization or
integration. Source selection, phase evidence and unresolved effort diagnostics
are explicit inputs, not optimizer decisions made during packaging.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile

import mujoco
import numpy as np

from .contracts import LaunchSnapshot, PunchRequest
from .planning import LAUNCH_FEATURE_LAYOUT, launch_features
from .provenance import PROJECT_ROOT, configure_local_caches
from .reference_screening import screen_reference_geometry
from .references import (FAMILIES, ReferenceClip, kinematic_arrays, project_output,
                         sha256, tangent_velocities, validate_model_order)
from .runtime.bundle import load_runtime_bundle
from .snapshots import SnapshotBank, head_point_state, load_verified_g1_model

SELECTION_SCHEMA = "human_wbc_canonical_selection_v1"


def _dump(path: Path, value: dict) -> None:
    path = project_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _copy(source: Path, destination: Path) -> None:
    destination = project_output(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise ValueError("Conflicting canonical provenance destination")
        return
    shutil.copyfile(source, destination)
    if sha256(destination) != sha256(source):
        raise ValueError("Copied canonical provenance differs")


def _source(record: dict) -> Path:
    path = (PROJECT_ROOT/record["path"]).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT) or sha256(path) != record.get("sha256"):
        raise ValueError("Canonical source path or hash differs")
    return path


def _verify_phase_evidence(row: dict) -> Path:
    """Bind exact final phase values to the selected master, never infer them."""
    path = _source(row["phase_evidence"])
    evidence = json.loads(path.read_text())
    records = evidence.get("clips", evidence.get("families", []))
    matches = [record for record in records if record.get("family") == row["family"]
               and record.get("master_sha256") == row["source_master"]["sha256"]]
    if len(matches) != 1:
        raise ValueError("Phase evidence does not identify the exact selected master")
    phases = matches[0].get("retiming", matches[0])
    if any(phases.get(key) != row[key] for key in ("outward_end_s", "guard_time_s")):
        raise ValueError("Canonical phase endpoints differ from retained source evidence")
    return path


def write_presampled_reference(source_master: Path, parent_bundle, parent_reference_id: str, *,
                              output_root: Path, reference_id: str,
                              outward_end_s: float, guard_time_s: float,
                              expected_source_sha256: str) -> tuple[dict, dict]:
    """Write an exact 200 Hz source and its 50 Hz view; never retime or interpolate.

    Optional source arrays are retained. Missing joint/body mappings, body-point
    velocities, qacc and zero feedforward are derived explicitly. Existing fields
    must agree with the physical model and finite-difference convention before
    preservation. Output files must not already exist.
    """
    bundle = load_runtime_bundle(parent_bundle)
    source_master = Path(source_master).resolve(strict=True)
    if sha256(source_master) != expected_source_sha256:
        raise ValueError("Presampled source master hash differs")
    if not reference_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                               for c in reference_id):
        raise ValueError("Reference id must be a simple filename component")
    old = next((c for c in bundle.clips if c["reference_id"] == parent_reference_id), None)
    if old is None:
        raise ValueError("Parent reference identity is absent")
    if old["feedforward_mode"] != "disabled_zero" or bundle.manifest["feedforward_mode"] != "disabled_zero":
        raise ValueError("Presampled canonical conversion currently supports explicit zero feedforward only")
    with np.load(bundle.path(old["master_path"]), allow_pickle=False) as data:
        if np.any(data["joint_feedforward"]):
            raise ValueError("Parent feedforward arrays contradict its zero mode")
    with np.load(source_master, allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in archive.files}
    for name in ("time_s", "qpos", "qvel"):
        if name not in source:
            raise ValueError(f"Missing presampled source field: {name}")
    times, poses, velocity = (source[name] for name in ("time_s", "qpos", "qvel"))
    if (times.ndim != 1 or len(times) < 5 or (len(times)-1) % 4
            or not np.array_equal(times, np.arange(len(times))*.005)
            or not np.isfinite(outward_end_s) or not np.isfinite(guard_time_s)
            or not 0 < outward_end_s < guard_time_s <= times[-1]):
        raise ValueError("Presampled clock requires exact 5 ms samples, a 20 ms endpoint and ordered phases")
    for name, values in source.items():
        if values.dtype.kind in "fc" and not np.isfinite(values).all():
            raise ValueError(f"Nonfinite presampled field: {name}")
    clip = ReferenceClip(old["family"], bundle.model_path, times, poses, velocity,
                         old["target_w"], 0., guard_time_s, expected_source_sha256)
    model = mujoco.MjModel.from_xml_path(str(bundle.model_path))
    validate_model_order(model, bundle.body)
    fresh_velocity = tangent_velocities(model, times, poses)
    if not np.array_equal(velocity, fresh_velocity):
        raise ValueError("Presampled qvel differs from fresh per-clip tangent derivatives")
    fresh = kinematic_arrays(clip)
    fresh["qacc"] = np.gradient(velocity, times, axis=0, edge_order=2)
    fresh["cycle_phase_id"] = np.where(times < outward_end_s, 0,
        np.where(times < guard_time_s, 1, 2)).astype(np.uint8)
    fresh["outward_end_s"], fresh["guard_time_s"] = np.asarray(outward_end_s), np.asarray(guard_time_s)
    for name in fresh.keys() & source.keys():
        if not np.array_equal(fresh[name], source[name]):
            raise ValueError(f"Presampled field differs from fresh model/phase/derivative data: {name}")
    arrays = {**fresh, **source}
    root = project_output(output_root)
    master_path = root/"masters"/f"{reference_id}.npz"
    motion_path = root/"motions/npz"/f"{reference_id}.npz"
    if master_path.exists() or motion_path.exists():
        raise FileExistsError("Presampled outputs already exist")
    for path in (master_path, motion_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(master_path, **arrays)
    fields = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    np.savez_compressed(motion_path, fps=np.asarray(50.),
        **{key: arrays[key][::4] for key in fields},
        joint_names=arrays["joint_names"], body_names=arrays["body_names"])
    active = 0 if clip.family in ("jab", "left_hook") else 1
    distances = np.linalg.norm(arrays["glove_pos_w"]-clip.target_w, axis=-1)
    hits = np.flatnonzero((times > 0) & (times <= outward_end_s) & (distances[:, active] <= .14))
    contact = None if not len(hits) else float(times[hits[0]])
    screen = screen_reference_geometry(bundle.model_path, master_path, guard_time_s=guard_time_s)
    basic = {"initial_clearance": bool(np.all(distances[0] >= .16)),
        "contact_present": contact is not None,
        "sampled_ttc_within_one_tick": contact is not None and abs(contact-old["requested_ttc_s"]) <= .005+1e-10,
        "guard_return": bool(np.max(abs(poses[-1]-poses[0])) <= 1e-10)}
    row = deepcopy(old)
    # A new trajectory has no reusable parent timing map. Final phases are
    # explicit; previous maps stay in copied source provenance only.
    for key in ("retiming_source_knots_s", "retiming_final_knots_s"):
        row.pop(key, None)
    row.update(reference_id=reference_id, source_reference_id=parent_reference_id,
        master_path=master_path.relative_to(root).as_posix(), master_sha256=sha256(master_path),
        motion_path=motion_path.relative_to(root).as_posix(), motion_sha256=sha256(motion_path),
        source_sha256=expected_source_sha256, launch_time_s=0., guard_time_s=guard_time_s,
        outward_end_s=outward_end_s, duration_s=float(times[-1]), reference_contact_delay_s=contact,
        qualified_for_training=False, common_guard_validated=False,
        geometry_screen_passed=screen["passed"], kinematic_screen_passed=screen["passed"] and all(basic.values()),
        feedforward_mode="disabled_zero", initial_guard_qpos=poses[0].tolist(),
        cycle_phase_labels=["outward", "recovery", "guard"])
    proof = {"reference_id": reference_id, "family": clip.family,
        "source_arrays_preserved": all(np.array_equal(arrays[k], v) for k, v in source.items()),
        "preserved_array_names": list(source), "derived_array_names": sorted(fresh.keys()-source.keys()),
        "source_qvel_matches_fresh_derivatives": True, "geometry_screen": screen,
        "command_screen": basic, "frames_200hz": len(times), "frames_50hz": len(times[::4]),
        "reference_contact_delay_s": contact, "qualified_for_training": False}
    return row, proof


def assemble_canonical_bundle(selection_manifest: Path, *, output_root: Path,
                              bank_root: Path, g1_asset_root: Path,
                              nominal_plant_path: Path) -> dict:
    """Build one canonical clip per family from an explicit hashed selection.

    Selection paths are human-project-relative. Every row binds a parent bundle
    manifest, original reference ID, selected master and exact phase endpoints.
    Additional provenance files are copied verbatim. Failed kinematic screens
    remain visible; this API can never set training qualification to true.
    """
    configure_local_caches()
    selected = json.loads(Path(selection_manifest).read_text())
    rows = selected.get("references", [])
    if (selected.get("schema") != SELECTION_SCHEMA or len(rows) != 4
            or {row.get("family") for row in rows} != set(FAMILIES)
            or len({row.get("reference_id") for row in rows}) != 4):
        raise ValueError("Canonical selection requires exactly one identified reference per family")
    destination = project_output(output_root)
    if destination.exists():
        raise FileExistsError(destination)
    bank = SnapshotBank(bank_root)
    g1, geometry = load_verified_g1_model(g1_asset_root, bank, nominal_plant_path)
    resolved = []
    for row in rows:
        master = _source(row["source_master"])
        _verify_phase_evidence(row)
        parent_manifest = _source(row["parent_manifest"])
        parent = load_runtime_bundle(parent_manifest)
        clip = next((c for c in parent.clips if c["reference_id"] == row["parent_reference_id"]), None)
        if clip is None or clip["family"] != row["family"]:
            raise ValueError("Selected family differs from its parent reference")
        snapshot = LaunchSnapshot(**clip["launch_snapshot"])
        state = bank.state(int(snapshot.snapshot_id))
        head, head_velocity = head_point_state(g1, np.asarray(state.qpos), np.asarray(state.qvel))
        request = PunchRequest(clip["family"], clip["requested_ttc_s"], snapshot, clip["guard_id"])
        if (not np.array_equal(state.qpos, snapshot.g1_qpos) or not np.array_equal(state.qvel, snapshot.g1_qvel)
                or not np.array_equal(head, snapshot.head_position_w)
                or not np.array_equal(head_velocity, snapshot.head_velocity_w)
                or not np.array_equal(clip["target_w"], head)
                or snapshot.fingerprint != clip["snapshot_fingerprint"]
                or request.fingerprint != clip["request_fingerprint"]
                or not np.array_equal(launch_features(request), clip["launch_features"])):
            raise ValueError("Actual snapshot payload, frozen target or launch conditioning differs")
        resolved.append((row, master, parent, clip, state))
    baseline = resolved[0][2]
    for _, _, parent, _, _ in resolved:
        if parent.controller_sha256() != baseline.controller_sha256():
            raise ValueError("Canonical families require one body/controller recipe")
        for field in ("model_path", "body_metadata_path", "controller_schema_path"):
            if sha256(parent.path(parent.manifest[field])) != sha256(baseline.path(baseline.manifest[field])):
                raise ValueError("Canonical families require identical body assets")
    cache = project_output(PROJECT_ROOT/".cache/task10/canonical_assembly")
    cache.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="presampled_", dir=cache))
    _copy(Path(selection_manifest), root/"provenance/selection.json")
    _copy(Path(__file__), root/"provenance/canonical.py")
    for field in ("model_path", "body_metadata_path", "controller_schema_path"):
        source = baseline.path(baseline.manifest[field])
        _copy(source, root/"assets"/source.name)
    clips, proofs, snapshots = [], [], {}
    for row, master, parent, old, state in resolved:
        clip, proof = write_presampled_reference(master, parent, row["parent_reference_id"],
            output_root=root, reference_id=row["reference_id"], outward_end_s=row["outward_end_s"],
            guard_time_s=row["guard_time_s"], expected_source_sha256=row["source_master"]["sha256"])
        prefix = root/"provenance"/row["reference_id"]
        _copy(master, prefix/"source_master.npz")
        _copy(parent.root/"manifest.json", prefix/"parent_manifest.json")
        _copy(_source(row["phase_evidence"]), prefix/"phase_evidence.json")
        for index, record in enumerate(row.get("provenance", [])):
            _copy(_source(record), prefix/f"evidence_{index:02d}{Path(record['path']).suffix}")
        proof["exact_path_diagnostics"] = {"available": False}
        if row.get("effort_diagnostics"):
            effort_path = _source(row["effort_diagnostics"])
            with np.load(effort_path, allow_pickle=False) as effort, np.load(master, allow_pickle=False) as source:
                if not np.array_equal(effort["time_s"], source["time_s"]):
                    raise ValueError("Effort diagnostics use a different selected sample clock")
                feasible, ratio = effort["support_feasible"], effort["minimum_peak_effort_ratio"]
                if (feasible.dtype != np.bool_ or feasible.shape != effort["time_s"].shape
                        or ratio.shape != feasible.shape or not np.isfinite(ratio[feasible]).all()):
                    raise ValueError("Malformed exact-path effort diagnostics")
                proof["exact_path_diagnostics"] = {"available": True,
                    "source_sha256": sha256(effort_path), "frames": len(feasible),
                    "support_infeasible_frames": int(np.sum(~feasible)),
                    "over_cap_frames": int(np.sum(feasible & (ratio > 1+1e-6)))}
            _copy(effort_path, prefix/"exact_path_effort.npz")
        proof["diagnostic_scope"] = "Prescribed-path diagnostic only; not proof all nearby motions are infeasible."
        clips.append(clip); proofs.append(proof)
        if state.slot_id not in snapshots:
            ordinal = int(bank.locations["shard_ordinal"][state.slot_id])
            entry = bank.index["shards"][ordinal]
            snapshot_prefix = root/"provenance/snapshots"/str(state.slot_id)
            _copy(bank.directory/entry["payload"], snapshot_prefix/"payload.npz")
            _copy(bank.directory/entry["manifest"], snapshot_prefix/"manifest.json")
            snapshots[state.slot_id] = {"state": asdict(state), "original_shard_entry": entry,
                "row_in_shard": int(bank.locations["row_in_shard"][state.slot_id])}
    poses = np.asarray([row["initial_guard_qpos"] for row in clips])
    posture_error = float(np.max(abs(poses[:, 3:]-poses[0, 3:])))
    if posture_error > 1e-10 or float(np.ptp(poses[:, 2])) > 1e-10:
        raise ValueError("Canonical references do not share the initial guard posture/root height")
    report = {"schema": "human_wbc_canonical_presampled_assembly_v1", "qualified_for_training": False,
        "physics_executed": False, "optimization_executed": False, "policy_initialized": False,
        "selection_sha256": sha256(Path(selection_manifest)), "references": proofs,
        "snapshot_geometry": geometry, "actual_snapshot_payloads": list(snapshots.values()),
        "common_guard_posture_max_error": posture_error,
        "identical_world_initial_qpos": bool(np.max(abs(poses-poses[0])) <= 1e-10),
        "guard_stability_qualified": False, "domain_qualified": False,
        "guard_semantics": "Shared posture with explicit per-clip world placement; no physical readiness claim.",
        "feedforward_mode": "disabled_zero", "guard_learner_weights_used": False,
        "limitations": ["Four canonical requests are not a feasible target/TTC domain or held-out split.",
            "No physical zero-residual or learned success follows from assembly.",
            "Runtime must hold the final reference; acceptance still requires actual quiet guard dwell."]}
    _dump(root/"assembly_report.json", report)
    manifest = {"schema": "human_wbc_references_v1", "qualified_for_training": False,
        "reason": "Canonical engineering data; explicit learning admission and physical domain qualification are separate.",
        "model_path": "assets/human.xml", "body_metadata_path": "assets/body_metadata.json",
        "controller_schema_path": "assets/controller_schema.json",
        "asset_sha256": {p.name: sha256(p) for p in (root/"assets").iterdir()},
        "physics_dt_s": .005, "policy_dt_s": .020, "decimation": 4,
        "quaternion_order": "wxyz", "feedforward_mode": "disabled_zero",
        "launch_feature_layout": list(LAUNCH_FEATURE_LAYOUT), "complete_family_library": True,
        "assembly_report": {"path": "assembly_report.json", "sha256": sha256(root/"assembly_report.json")},
        "clips": clips}
    _dump(root/"manifest.json", manifest)
    bundle = load_runtime_bundle(root)
    from .native_cycle_probe import load_phase_endpoints
    load_phase_endpoints(bundle)
    _dump(root/"phases.json", {"schema": "human_wbc_reference_phases_v1",
        "bundle_manifest_sha256": sha256(root/"manifest.json"),
        "references": [{k: c[k] for k in ("reference_id", "master_sha256", "outward_end_s", "guard_time_s")}
                       for c in clips]})
    load_phase_endpoints(bundle, root/"phases.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    root.rename(destination)
    load_runtime_bundle(destination)
    return {"manifest_path": str(destination/"manifest.json"), "manifest_sha256": sha256(destination/"manifest.json"),
            "families": 4, "qualified_for_training": False,
            "all_kinematic_screens_passed": all(c["kinematic_screen_passed"] for c in clips)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection-manifest", "output", "bank-root", "g1-asset-root", "nominal-plant"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = assemble_canonical_bundle(args.selection_manifest, output_root=args.output,
        bank_root=args.bank_root, g1_asset_root=args.g1_asset_root, nominal_plant_path=args.nominal_plant)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
