"""Focused common-ACTUAL-pelvis diagnostic; no change to the 3C procedure.

Candidates are existing +25 degree top-four poses, expressed as actual initial
H2 pelvis XY/yaw. Each of the eight unchanged motions is aligned to each shared
pose with one rigid world-up transform. Recipient equivalent yaw/residuals are
checked against the existing continuous numerical envelope; off-grid values
are reported and never snapped. This diagnostic does not introduce shared
articulation, guards, velocities, transitions, or a live switching mechanism.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .common_start_phase3 import accepted_to_common, common_start_transform
from .g1 import PROJECT, transform_snapshot
from .phase1 import dump
from .placement_phase3 import (
    PRIMITIVE_IDS, SearchBounds, _placement_metrics, _rotate_xy,
    cache_library, mount_optical_rotation, project_metrics,
)

SNAPSHOT_IDS = (53874, 67549, 28655, 22353, 39178)
FAMILIES = ("jab", "cross", "left_hook", "right_hook")


def wrap_radians(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def common_pose_candidates(paths, search_dir: Path, snapshot_id: int):
    """Deterministic union of the existing top-four actual pelvis poses."""
    poses, lookup = [], {}
    for path in paths:
        identity = path.primitive.primitive_id
        with np.load(search_dir / f"{identity}.npz", allow_pickle=False) as record:
            row = int(np.flatnonzero(record["snapshot_id"] == snapshot_id)[0])
            pitch = int(np.flatnonzero(record["pitches_up_deg"] == 25)[0])
            for rank in range(record["top_available"].shape[-1]):
                if not record["top_available"][pitch, row, rank]:
                    continue
                native_yaw = float(record["top_yaw_rad"][pitch, row, rank])
                native_xy = record["top_xy"][pitch, row, rank]
                delta = native_yaw - path.primitive.placement.yaw
                shift = native_xy - _rotate_xy(path.primitive.placement.xy, delta)
                xy = _rotate_xy(path.initial_pelvis[:2], delta) + shift
                yaw = float(wrap_radians(path.initial_heading_rad + delta))
                key = tuple(np.round([*xy, yaw], 10))
                if key not in lookup:
                    index = len(poses)
                    lookup[key] = index
                    poses.append(dict(common_pose_id=f"s{snapshot_id}_pose{index:02d}",
                        actual_initial_pelvis_xy=xy.tolist(), actual_initial_pelvis_yaw_rad=yaw,
                        donors=[]))
                poses[lookup[key]]["donors"].append(dict(primitive_id=identity, geometric_rank=rank))
    return poses


def evaluate_common_pose(path, state, pose, *, nominal_heading_rad, bounds):
    """Independent reach/FOV/clearance/bounds metrics for one exact shared pose."""
    xy, yaw = pose["actual_initial_pelvis_xy"], pose["actual_initial_pelvis_yaw_rad"]
    placement = common_start_transform(path, pelvis_xy=xy, heading_rad=yaw)
    delta = float(wrap_radians(yaw - path.initial_heading_rad))
    shift = np.asarray(xy) - _rotate_xy(path.initial_pelvis[:2], delta)
    head = state["head_world"]
    query = head.copy()
    query[:2] = _rotate_xy(head[:2] - shift, -delta)
    outward_min = float(path.tree.query(query)[0])
    distances = np.linalg.norm(path.gloves - query, axis=-1)
    hits = np.flatnonzero(path.outward & (distances[:, path.active_index] <= .14))
    first = int(hits[0]) if len(hits) else None
    other_hits = np.flatnonzero(distances[:first, 1-path.active_index] <= .14) if first is not None else []
    initial_gloves = accepted_to_common(path, path.gloves[0], pelvis_xy=xy, heading_rad=yaw)
    clearance = np.linalg.norm(initial_gloves-head, axis=-1)
    rotation = state["torso_rotation"] @ mount_optical_rotation(25.)
    fov, margin, sphere_margin, depth, image_separation = project_metrics(
        initial_gloves[None, None], state["camera_position_world"][None], rotation[None])
    heading = float(state["pelvis_heading_rad"])
    relative_yaw_deg = float(np.rad2deg(wrap_radians(delta - (heading-nominal_heading_rad))))
    residual = _rotate_xy(shift + _rotate_xy(path.target[:2], delta) - head[:2], -heading)
    actual_pelvis = np.array([*xy, path.initial_pelvis[2]])
    batch = {key: np.asarray(value)[None] for key, value in state.items()}
    plausible, bearing, separation, facing, heading_xy = _placement_metrics(
        actual_pelvis[None, None], np.array([[yaw]]), batch, bounds)
    # Only compare inclusive numerical envelope bounds, not donor grid points.
    # A 1e-10 tolerance handles rigid-transform roundoff at exactly +/- bounds;
    # the .14 encounter and .16 clearance thresholds have no added tolerance.
    yaw_allowed = min(bounds.yaw_degrees)-1e-10 <= relative_yaw_deg <= max(bounds.yaw_degrees)+1e-10
    residual_allowed = bool(np.all(residual >= min(bounds.residual_xy_m)-1e-10)
                            and np.all(residual <= max(bounds.residual_xy_m)+1e-10))
    on_yaw_grid = bool(np.isclose(relative_yaw_deg, bounds.yaw_degrees, atol=1e-8, rtol=0).any())
    on_residual_grid = bool(np.isclose(residual[:, None], bounds.residual_xy_m,
                                      atol=1e-8, rtol=0).any(-1).all())
    allowed = bool(plausible[0, 0] and yaw_allowed and residual_allowed)
    reach, frustum, safe = outward_min <= .14, bool(fov[0, 0]), bool((clearance >= .16).all())
    reasons = []
    for passes, name in ((reach, "no_sampled_active_outward_reach"),
        (frustum, "initial_center_frustum"), (safe, "initial_glove_clearance"),
        (bool(plausible[0,0]), "frontal_region"), (yaw_allowed, "relative_yaw_envelope"),
        (residual_allowed, "residual_xy_envelope")):
        if not passes:
            reasons.append(name)
    return dict(primitive_id=path.primitive.primitive_id, family=path.primitive.family,
        active_hand=path.primitive.active_hand, placement=asdict(placement),
        actual_initial_pelvis_z_m=float(path.initial_pelvis[2]),
        equivalent_relative_yaw_deg=relative_yaw_deg,
        equivalent_residual_xy_heading_m=residual.tolist(),
        on_original_yaw_grid=on_yaw_grid, on_original_residual_grid=on_residual_grid,
        allowed_numeric_envelope=allowed, initial_frontal_region=bool(plausible[0,0]),
        initial_bearing_deg=float(bearing[0,0]), initial_pelvis_separation_m=float(separation[0,0]),
        initial_facing_relative_deg=float(facing[0,0]), initial_pelvis_heading_xy_m=heading_xy[0,0].tolist(),
        reach_only=reach, initial_both_centers_in_frustum=frustum,
        reach_within_allowed_envelope=reach and allowed,
        initial_both_gloves_clear=safe, initial_glove_distances_m=clearance.tolist(),
        initial_center_margins=margin[0,0].tolist(), initial_sphere_margins_m=sphere_margin[0,0].tolist(),
        initial_optical_depth_m=depth[0,0].tolist(), normalized_glove_separation=float(image_separation[0,0]),
        outward_min_distance_m=outward_min,
        first_active_outward_sample_s=None if first is None else float(path.time_s[first]),
        previous_source_sample_s=None if first is None else float(path.time_s[max(first-1,0)]),
        earlier_inactive_encounter_sample_s=None if not len(other_hits) else float(path.time_s[other_hits[0]]),
        reach_frustum_clearance_allowed=reach and frustum and safe and allowed,
        rejection_reasons=reasons, gross_intersection_status="unverified",
        clean_observation_status="unverified")


def summarize_pose(evaluations):
    def selected(key):
        rows = [row for row in evaluations if row[key]]
        return dict(primitive_count=len(rows), family_count=len({r["family"] for r in rows}),
                    primitives=[r["primitive_id"] for r in rows],
                    families=[f for f in FAMILIES if any(r["family"] == f for r in rows)])
    result = {key: selected(key) for key in ("reach_only", "reach_within_allowed_envelope", "reach_frustum_clearance_allowed",
                                           "reach_frustum_initial_scene_valid")}
    available = [r for r in evaluations if r["reach_frustum_initial_scene_valid"]]
    result["worst_available_center_margin"] = min(
        (min(r["initial_center_margins"]) for r in available), default=-1.)
    return result


def pose_rank(pose):
    result = pose["summary"]["reach_frustum_initial_scene_valid"]
    return (-result["family_count"], -result["primitive_count"],
            -pose["summary"]["worst_available_center_margin"], pose["common_pose_id"])


def native_checker(reference, output):
    """Compile the accepted combined model once for static native geometry only."""
    from .g1 import _imports, make_task08_cfg
    _imports()
    from mjlab.scene import Scene
    from .camera import add_camera_only, mount_config
    from .phase3_render import InitialIntersections
    cfg = make_task08_cfg(count=1, seed=102403, output_dir=output,
        extra_entities={"h2": reference.entity_cfg()},
        scene_spec_fn=lambda spec: add_camera_only(spec, camera_config=mount_config(25.)))
    model = Scene(cfg.scene, device="cpu").compile()
    return InitialIntersections(model, reference.bind(model))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=PROJECT/"outputs/task10d/phase3/snapshots/geometry.npz")
    parser.add_argument("--search-dir", type=Path, default=PROJECT/"outputs/task10d/phase3/search01")
    parser.add_argument("--bounds-config", type=Path, default=PROJECT/"configs/task10d/phase3_review_bounds.json")
    parser.add_argument("--snapshot-ids", nargs="+", type=int, default=list(SNAPSHOT_IDS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to((PROJECT/"outputs/task10d/phase3/3b_ii").resolve()):
        raise ValueError("Only a new Phase3/3b_ii output is allowed")
    output.mkdir(parents=True, exist_ok=False)
    geometry = dict(np.load(args.geometry, allow_pickle=False))
    locations = {int(value): i for i, value in enumerate(geometry["snapshot_id"])}
    nominal_heading = float(geometry["pelvis_heading_rad"][locations[53874]])
    bounds = SearchBounds(**json.loads(args.bounds_config.read_text()))
    paths = cache_library()
    checker = native_checker(paths[0].primitive.reference, output)
    states = []
    for sid in args.snapshot_ids:
        i = locations[sid]
        state = {key: value[i] for key,value in geometry.items()
                 if value.shape[:1] == geometry["snapshot_id"].shape}
        q, v = transform_snapshot(state["sim/data/qpos"], state["sim/data/qvel"], state["world_from_collection"])
        candidates = common_pose_candidates(paths, args.search_dir, sid)
        print(f"SNAPSHOT {sid} shared_actual_poses={len(candidates)}", flush=True)
        for pose in candidates:
            evaluations = []
            for path in paths:
                row = evaluate_common_pose(path, state, pose,
                    nominal_heading_rad=nominal_heading, bounds=bounds)
                placement = common_start_transform(path, pelvis_xy=pose["actual_initial_pelvis_xy"],
                    heading_rad=pose["actual_initial_pelvis_yaw_rad"])
                sample = path.primitive.reference.sample(0., placement)
                np.testing.assert_allclose(sample.body_pos[0,:2], pose["actual_initial_pelvis_xy"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(sample.body_pos[:,2],
                    path.primitive.reference.sample(0., path.primitive.placement).body_pos[:,2], atol=1e-12, rtol=0)
                overlap = checker.measure(q,v,sample)
                row["native_initial_intersection"] = overlap
                row["gross_intersection_status"] = "failed" if overlap["gross_intersection"] else "passed"
                row["reach_frustum_initial_scene_valid"] = bool(
                    row["reach_frustum_clearance_allowed"] and not overlap["gross_intersection"])
                if overlap["gross_intersection"]:
                    row["rejection_reasons"].append("gross_initial_intercharacter_intersection")
                evaluations.append(row)
            pose["evaluations"] = evaluations
            pose["summary"] = summarize_pose(evaluations)
        ordered = sorted(candidates, key=pose_rank)
        state_record = dict(snapshot_id=sid, head_world=state["head_world"].tolist(),
            candidate_count=len(candidates), best_pose_id=ordered[0]["common_pose_id"],
            best_summary=ordered[0]["summary"], candidates=candidates)
        states.append(state_record)
        print(f"BEST {sid} {ordered[0]['common_pose_id']} {ordered[0]['summary']}", flush=True)
    # Nominal plus two extremes chosen for distinct posture/height. Keep all
    # eight motions at each selected actual pose, including honest failures.
    requested_states = [sid for sid in (53874,67549,28655) if sid in args.snapshot_ids]
    requests = []
    for state in states:
        if state["snapshot_id"] not in requested_states:
            continue
        pose = next(p for p in state["candidates"] if p["common_pose_id"] == state["best_pose_id"])
        for row in pose["evaluations"]:
            requests.append(dict(snapshot_id=state["snapshot_id"], primitive_id=row["primitive_id"],
                pitch_up_deg=25, rank=0, kind="3b_ii_common_actual_pelvis",
                placement=row["placement"], label=pose["common_pose_id"]+"_"+row["primitive_id"],
                common_pose_id=pose["common_pose_id"],
                common_initial_pelvis_xy=pose["actual_initial_pelvis_xy"],
                common_initial_pelvis_yaw_rad=pose["actual_initial_pelvis_yaw_rad"],
                expected_initial_pelvis_z_m=row["actual_initial_pelvis_z_m"],
                geometric_joint_available=row["reach_frustum_clearance_allowed"],
                native_scene_joint_available=row["reach_frustum_initial_scene_valid"],
                source_reach_only=row["reach_only"], save_panel=True))
    report = dict(schema="task10d_3b_ii_common_actual_pelvis_v1", pitch_up_deg=25,
        snapshot_ids=args.snapshot_ids, primitive_ids=list(PRIMITIVE_IDS), bounds=asdict(bounds),
        candidate_procedure="Union of existing +25 top4 actual pelvis XY/yaw, deduplicated at 1e-10",
        recipient_placement="Exactly one whole-motion planar transform; preserve original Z/articulation/root and foot motion",
        off_grid_semantics="Continuous existing numerical envelope checked without snapping; this does not change 3C's discrete generator",
        reach_only_semantics="Raw sampled active-outward reach regardless of recipient placement-envelope membership; admissible reach is separately reach_within_allowed_envelope",
        rank_rule="descending available family count, primitive count, worst admitted initial-center margin; then pose ID",
        head_radius_plus_glove_radius_m=.14, initial_clearance_m=.16,
        native_gross_penetration_threshold_m=-.03,
        native_qualification="Existing native convex cross-character distance check; no contact/dynamics enabled; H2 self-overlaps excluded",
        clean_observation_status="unknown until actual rendering; never inferred from projected centers or native scene validity",
        states=states, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [args.geometry,args.bounds_config,*[args.search_dir/f"{p}.npz" for p in PRIMITIVE_IDS]]})
    dump(output/"common_placements.json", report)
    dump(output/"render_requests.json", dict(requests=requests,
        selection="best shared pose by declared geometric/native family-first rank for nominal, low and high head",
        clean_observation_status="pending actual render"))
    concise = dict(snapshot_count=len(states), candidate_pose_count=sum(s["candidate_count"] for s in states),
        evaluated_motion_placements=sum(s["candidate_count"]*8 for s in states),
        states=[{k:s[k] for k in ("snapshot_id","head_world","candidate_count","best_pose_id","best_summary")} for s in states],
        render_requests=len(requests), no_3c_run_or_procedure_change=True)
    dump(output/"summary.json",concise)


if __name__ == "__main__":
    main()
