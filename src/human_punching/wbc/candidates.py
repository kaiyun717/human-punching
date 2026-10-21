"""Small reproducible offline candidate bundles from real G1 walking payloads.

These are reference-generation experiments, not a feasible-domain certificate.
The API never steps physics, trains a policy, modifies a source, or substitutes a
failed target/TTC. Its report retains every requested recipe and its disposition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Iterable

import mujoco
import numpy as np

from .contracts import LaunchSnapshot, PunchFamily, PunchRequest, SearchBounds, quaternion_matrix
from .planning import (LAUNCH_FEATURE_LAYOUT, GuardCandidate, common_guard_candidate,
                       retarget_candidate)
from .references import (FAMILIES, PHYSICS_DT, POLICY_DT,
                         load_release_reference, project_output, sha256, write_reference)
from .reference_screening import screen_reference_geometry
from .snapshots import (SnapshotBank, SnapshotState, head_point_state,
                        load_verified_g1_model)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class CandidateSpec:
    """One explicit command; parameters are search coordinates, not admissions."""
    snapshot_slot_id: int
    family: PunchFamily
    forward_m: float
    lateral_m: float
    ttc_s: float
    stance_width_m: float = .50
    left_lead_m: float = .20

    def __post_init__(self):
        object.__setattr__(self, "family", PunchFamily.coerce(self.family))
        if type(self.snapshot_slot_id) is not int or self.snapshot_slot_id < 0:
            raise ValueError("An original nonnegative integer snapshot slot is required")
        for name in ("forward_m", "lateral_m", "ttc_s", "stance_width_m", "left_lead_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.ttc_s <= 0:
            raise ValueError("Requested TTC must be positive")
        if not .45 <= self.stance_width_m <= .55 or not .15 <= self.left_lead_m <= .25:
            raise ValueError("Guard search dimensions are outside the declared search box")

    @property
    def recipe_id(self) -> str:
        digest = hashlib.sha256(_canonical(asdict(self))).hexdigest()[:20]
        return f"{self.family.value}_{self.snapshot_slot_id}_{digest}"


def frozen_snapshot(state: SnapshotState, model: mujoco.MjModel, spec: CandidateSpec) -> LaunchSnapshot:
    """Place human from pelvis heading, freeze true head position and velocity.

    Human +X faces opposite G1's launch pelvis yaw. Head articulation is retained:
    neither the human's placement nor its heading is fitted to the head target.
    """
    if state.slot_id != spec.snapshot_slot_id or not state.eligible:
        raise ValueError("Recipe requires the matching originally eligible snapshot")
    rotation = quaternion_matrix(state.qpos[3:7])
    projected_forward = rotation[:2, 0]
    if np.linalg.norm(projected_forward) < 1e-8:
        raise ValueError("G1 pelvis heading is undefined at this launch pose")
    yaw = math.atan2(projected_forward[1], projected_forward[0])
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.])
    origin = np.array([state.qpos[0], state.qpos[1], 0.])
    origin += spec.forward_m * forward + spec.lateral_m * left
    heading = yaw + math.pi
    quat = (math.cos(heading / 2), 0., 0., math.sin(heading / 2))
    head, velocity = head_point_state(model, np.asarray(state.qpos), np.asarray(state.qvel))
    return LaunchSnapshot(str(state.slot_id), state.qpos, state.qvel, tuple(head),
                          tuple(velocity), tuple(origin), quat, state.source_group_id)


def _guard_id(spec: CandidateSpec, guard: GuardCandidate) -> str:
    # World pose is encoded in each launch snapshot; ID also binds this exact
    # deterministic pose rather than claiming a whole interval is admissible.
    value = {"kind": "common_orthodox_candidate", "stance_width_m": spec.stance_width_m,
             "left_lead_m": spec.left_lead_m, "qpos_w": guard.qpos_w}
    return "common_" + hashlib.sha256(_canonical(value)).hexdigest()[:20]


def _snapshot_record(state: SnapshotState) -> dict:
    return {key: value for key, value in asdict(state).items() if key not in ("qpos", "qvel")}


def build_candidate_bundle(specs: Iterable[CandidateSpec], *, release_root: Path,
                           bank_root: Path, g1_asset_root: Path, nominal_plant_path: Path,
                           output_root: Path, ik_iterations: int = 35,
                           guard_iterations: int = 120,
                           backend: str = "original") -> dict:
    """Build an explicitly requested small diagnostic library, including failures.

    All four families must be requested. A complete generated bundle can be read
    by the WBC evaluation adapter, but remains unqualified for learning. This API
    does not assign train/validation/test sets: original grouping is preserved and
    the report marks split assignment pending. A separate frozen split plan must
    precede production expansion. No timestamps or random sampling affect IDs.
    """
    specs = tuple(specs)
    if not specs or any(not isinstance(spec, CandidateSpec) for spec in specs):
        raise ValueError("Provide explicit CandidateSpec records")
    if len({spec.recipe_id for spec in specs}) != len(specs):
        raise ValueError("Duplicate candidate recipe")
    if {spec.family.value for spec in specs} != set(FAMILIES):
        raise ValueError("Request all four punch families for a shared library")
    if type(ik_iterations) is not int or ik_iterations < 1:
        raise ValueError("IK iterations must be a positive integer")
    if type(guard_iterations) is not int or guard_iterations < 1:
        raise ValueError("Guard iterations must be a positive integer")
    if backend not in ("original", "refined_v1", "refined_v2", "refined_v3"):
        raise ValueError("Unknown candidate planning backend")
    planner = retarget_candidate
    generator_sources = [Path(__file__), Path(__file__).with_name("planning.py"),
                         Path(__file__).with_name("references.py"), Path(__file__).with_name("snapshots.py"),
                         Path(__file__).with_name("reference_screening.py"),
                         Path(__file__).with_name("contracts.py"),
                         Path(__file__).parents[1] / "scalar_reference.py"]
    if backend in ("refined_v1", "refined_v2", "refined_v3"):
        from .planning_refinement import RefinementSettings, refined_retarget_candidate
        from functools import partial
        planner = refined_retarget_candidate
        if backend in ("refined_v2", "refined_v3"):
            planner = partial(refined_retarget_candidate, settings=RefinementSettings(
                recovery_corridor=True, project_support_after_smoothing=True,
                unbiased_support_projection=backend=="refined_v3",
                quiet_boundary_s=.04 if backend=="refined_v3" else 0.))
        generator_sources.append(Path(__file__).with_name("planning_refinement.py"))
    output = project_output(output_root)
    if output.exists():
        raise FileExistsError("Choose a new candidate output directory")
    release_root = Path(release_root).resolve()
    bank = SnapshotBank(bank_root)
    model, g1_provenance = load_verified_g1_model(g1_asset_root, bank, nominal_plant_path)
    sources = {family: load_release_reference(release_root, family) for family in FAMILIES}
    if len({sha256(source.model_path) for source in sources.values()}) != 1:
        raise ValueError("Every family must use the identical constrained human body")
    # Verify actual rows before writing anything. Bad provenance is an input
    # failure, not a planner rejection; never continue on unverified payloads.
    states = {slot: bank.state(slot) for slot in sorted({s.snapshot_slot_id for s in specs})}
    snapshots = {spec.recipe_id: frozen_snapshot(states[spec.snapshot_slot_id], model, spec)
                 for spec in specs}
    assets = output / "assets"
    assets.mkdir(parents=True)
    for source, name in ((sources["jab"].model_path, "human.xml"),
                         (release_root / "jab/body_metadata.json", "body_metadata.json"),
                         (release_root / "jab/controller_schema.json", "controller_schema.json")):
        shutil.copyfile(source, assets / name)
    rows, dispositions, guard_records = [], [], {}
    guards = {}
    for spec in specs:
        snapshot = snapshots[spec.recipe_id]
        disposition = {"recipe_id": spec.recipe_id, "spec": asdict(spec),
                       "snapshot_provenance": _snapshot_record(states[spec.snapshot_slot_id]),
                       "launch_snapshot": asdict(snapshot),
                       "snapshot_fingerprint": snapshot.fingerprint,
                       "qualified_for_training": False}
        # Out-of-search requests are retained as rejections, never clamped.
        screening_request = PunchRequest(spec.family, spec.ttc_s, snapshot, "pending_common_guard")
        if not SearchBounds().contains(screening_request):
            disposition.update(status="rejected", reason="outside_search_bounds")
            dispositions.append(disposition)
            continue
        key = (snapshot.human_origin_w, snapshot.human_quat_wxyz,
               spec.stance_width_m, spec.left_lead_m)
        try:
            if key not in guards:
                guards[key] = common_guard_candidate(sources["jab"],
                    human_origin_w=snapshot.human_origin_w,
                    human_quat_wxyz=snapshot.human_quat_wxyz,
                    lateral_width_m=spec.stance_width_m, left_lead_m=spec.left_lead_m,
                    iterations=guard_iterations)
            guard = guards[key]
            guard_id = _guard_id(spec, guard)
            guard_records[guard_id] = asdict(guard)
            request = PunchRequest(spec.family, spec.ttc_s, snapshot, guard_id)
            candidate = planner(sources[spec.family.value], request,
                                          guard_qpos_w=np.asarray(guard.qpos_w),
                                          ik_iterations=ik_iterations)
        except (ValueError, np.linalg.LinAlgError) as error:
            disposition.update(status="planner_failed", reason=str(error),
                               proves_physical_infeasibility=False)
            dispositions.append(disposition)
            continue
        row = write_reference(candidate.clip, output, spec.recipe_id)
        if "outward_end_s" in candidate.clip.metadata:
            outward_end=float(candidate.clip.metadata["outward_end_s"])
            master_path=output / row["master_path"]
            with np.load(master_path,allow_pickle=False) as saved:
                contents={name:saved[name].copy() for name in saved.files}
            contents["cycle_phase_id"]=np.where(contents["time_s"] < outward_end,0,
                np.where(contents["time_s"] < row["guard_time_s"],1,2)).astype(np.uint8)
            contents["outward_end_s"]=np.array(outward_end)
            np.savez_compressed(master_path,**contents)
            row.update(outward_end_s=outward_end,master_sha256=sha256(master_path),
                       cycle_phase_labels=["outward","recovery","guard"],
                       retiming_source_knots_s=candidate.clip.metadata["retiming_source_knots_s"],
                       retiming_final_knots_s=candidate.clip.metadata["retiming_final_knots_s"])
        row.update(recipe_id=spec.recipe_id, snapshot_provenance=disposition["snapshot_provenance"],
                   request_fingerprint=request.fingerprint)
        rows.append(row)
        # Screen the exact serialized master, not only a sparse optimizer path.
        with np.load(output / row["master_path"], allow_pickle=False) as master:
            gloves = master["glove_pos_w"]
            distances = np.linalg.norm(gloves[0]-snapshot.head_position_w, axis=-1)
            max_return = float(np.max(np.abs(master["qpos"][-1]-master["qpos"][0])))
            velocity = master["qvel"]
            acceleration = np.diff(velocity, axis=0) / PHYSICS_DT
            derivatives = {
                "peak_root_linear_speed_m_s": float(np.linalg.norm(velocity[:, :3], axis=-1).max()),
                "peak_root_angular_speed_rad_s": float(np.linalg.norm(velocity[:, 3:6], axis=-1).max()),
                "peak_joint_speed_rad_s": float(np.max(np.abs(velocity[:, 6:]))),
                "peak_joint_acceleration_rad_s2": float(np.max(np.abs(acceleration[:, 6:]))),
                "derivative_feasibility": "not_qualified; engineering limits require physical screening",
            }
        delay = row["reference_contact_delay_s"]
        screen = {"launch_time_zero": row["launch_time_s"] == 0.,
                  "initial_glove_clearance": bool(np.all(distances >= .16)),
                  "analytic_contact_present": delay is not None,
                  "sampled_ttc_within_one_physics_tick": delay is not None and abs(delay-spec.ttc_s) <= .005+1e-10,
                  "common_guard_return": max_return <= 1e-10,
                  "target_frozen": np.array_equal(candidate.clip.target_w, snapshot.head_position_w),
                  "launch_context_77d": len(row["launch_features"]) == 77}
        # np.bool_ cannot go into strict JSON; preserve normal Boolean evidence.
        screen = {key: bool(value) for key, value in screen.items()}
        basic_screen = dict(screen)
        geometry_screen = screen_reference_geometry(assets / "human.xml", output / row["master_path"],
                                                   guard_time_s=row["guard_time_s"])
        screen.update(geometry_screen["checks"])
        row["kinematic_screen_passed"] = all(screen.values())
        row["geometry_screen_passed"] = geometry_screen["passed"]
        disposition.update(status="generated", reason="kinematic_candidate_only",
            request_fingerprint=request.fingerprint, guard_id=guard_id,
            diagnostics=candidate.diagnostics, kinematic_screen=screen,
            basic_command_screen=basic_screen, geometry_screen=geometry_screen,
            derivative_diagnostics=derivatives,
            kinematic_screen_passed=all(screen.values()),
            initial_glove_distances_m=distances.tolist(), sampled_contact_delay_s=delay,
            sampled_ttc_error_s=None if delay is None else delay-spec.ttc_s,
            master_path=row["master_path"], master_sha256=row["master_sha256"])
        dispositions.append(disposition)
    all_families = {row["family"] for row in rows} == set(FAMILIES)
    report = {"schema": "human_wbc_candidate_generation_v1", "qualified_for_training": False,
              "physics_executed": False, "status": "generated" if len(rows) == len(specs) else "partial",
              "requested_count": len(specs), "generated_count": len(rows),
              "all_families_generated": all_families,
              "all_basic_command_screens_passed": bool(rows) and len(rows) == len(specs) and
                  all(all(item.get("basic_command_screen", {"missing": False}).values()) for item in dispositions),
              "all_kinematic_screens_passed": bool(rows) and len(rows) == len(specs) and
                  all(item.get("kinematic_screen_passed", False) for item in dispositions),
              "dynamic_feasibility": "unqualified", "guard_balance": "unqualified",
              "split_assignment": "pending; diagnostic candidates only",
              "g1_geometry": g1_provenance,
              "snapshot_bank_manifest_sha256": sha256(Path(bank_root) / "manifest.json"),
              "generator_source_sha256": {p.name: sha256(p) for p in generator_sources},
              "parameters": {"ik_iterations": ik_iterations, "guard_iterations": guard_iterations,
                             "backend": backend},
              "guards": guard_records, "requests": dispositions}
    report_path = output / "candidate_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    manifest = {"schema": "human_wbc_references_v1", "qualified_for_training": False,
                "reason": "Offline common-guard target/TTC candidates; physics, guard, domain and splits unqualified",
                "model_path": "assets/human.xml", "body_metadata_path": "assets/body_metadata.json",
                "controller_schema_path": "assets/controller_schema.json",
                "asset_sha256": {p.name: sha256(p) for p in sorted(assets.iterdir())},
                "release_manifest_sha256": sha256(release_root / "manifest.json"),
                "physics_dt_s": PHYSICS_DT, "policy_dt_s": POLICY_DT, "decimation": 4,
                "body_frame_convention": "world +Z up; canonical human +X forward,+Y left",
                "quaternion_order": "wxyz", "feedforward_mode": "disabled_zero",
                "launch_feature_layout": list(LAUNCH_FEATURE_LAYOUT),
                "candidate_report": {"path": report_path.name, "sha256": sha256(report_path)},
                "complete_family_library": all_families, "clips": rows}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return {"manifest_path": str(output / "manifest.json"), "report_path": str(report_path),
            "requested_count": len(specs), "generated_count": len(rows),
            "all_families_generated": all_families, "qualified_for_training": False}
