"""Bounded, frozen-target Phase 3 placement geometry; no dynamics or rendering.

All paths use production native FK at 5 ms. Geometry is an initial candidate
screen: clean observation and gross inter-character intersection remain UNKNOWN
until the actual initial scene is checked. No recovery-only entry is credited.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .g1 import PROJECT
from .phase2 import Primitive, load_library
from .replay import Placement

PRIMITIVE_IDS = (
    "jab_nominal", "cross_nominal", "left_hook_nominal", "right_hook_nominal",
    "jab_004_01107", "cross_012_01441", "left_hook_013_03243", "right_hook_014_02074",
)
INDEX = PROJECT / "constants/kinematic_punches/v1_20261023/library_index.json"
INDEX_SHA256 = "f405c875e2f1b41dfa0798e50f588d8f5dad32e3e377a4d92ef650e4190966f7"
PITCHES_UP_DEG = tuple(range(-15, 36, 5))
TAN_V = np.tan(np.deg2rad(54. / 2))
TAN_H = 672. / 376. * TAN_V
DT = .005
ENCOUNTER = .14
SAFE_CLEARANCE = .16


@dataclass(frozen=True)
class SearchBounds:
    """Initial engineering domain, explicitly proposed for the review gate."""

    yaw_degrees: tuple[float, ...] = tuple(range(-60, 61, 10))
    residual_xy_m: tuple[float, ...] = (-.15, -.075, 0., .075, .15)
    max_abs_bearing_deg: float = 75.
    min_forward_m: float = .15
    min_pelvis_separation_m: float = .35
    max_pelvis_separation_m: float = 1.75
    max_abs_lateral_m: float = 1.35

    def __post_init__(self):
        values = [*self.yaw_degrees, *self.residual_xy_m, self.max_abs_bearing_deg,
                  self.min_forward_m, self.min_pelvis_separation_m,
                  self.max_pelvis_separation_m, self.max_abs_lateral_m]
        if not np.isfinite(values).all() or not self.yaw_degrees or not self.residual_xy_m:
            raise ValueError("Finite, nonempty placement bounds required")
        if not (0 < self.max_abs_bearing_deg < 90 and self.min_forward_m > 0
                and 0 < self.min_pelvis_separation_m < self.max_pelvis_separation_m
                and self.max_abs_lateral_m > 0):
            raise ValueError("Placement domain must be bounded and frontal")


@dataclass(frozen=True)
class CachedPath:
    primitive: Primitive
    time_s: np.ndarray
    gloves: np.ndarray
    outward: np.ndarray
    initial_pelvis: np.ndarray
    initial_heading_rad: float
    target: np.ndarray
    tree: cKDTree

    @property
    def active_index(self):
        return ("left", "right").index(self.primitive.active_hand)


def cache_library(index: Path = INDEX) -> list[CachedPath]:
    """Select ALL EIGHT explicitly; sample once before any state/pitch search."""
    if hashlib.sha256(index.read_bytes()).hexdigest() != INDEX_SHA256:
        raise ValueError("Expected the accepted eight-reference constants index")
    result = []
    for primitive in load_library(index, list(PRIMITIVE_IDS)):
        end = primitive.reference.end_time_s
        times = np.minimum(np.arange(int(np.ceil(end / DT)) + 1) * DT, end)
        samples = [primitive.reference.sample(float(t), primitive.placement) for t in times]
        gloves = np.stack([sample.glove_pos for sample in samples])
        annotations = primitive.record["annotations"]
        # This is preparation/approach/outward through the annotated withdrawal
        # boundary. The historical source outward landmark is provenance, not
        # the adapted reference's withdrawal boundary.
        outward_end = float(annotations["withdrawal_start_s"])
        outward = times <= outward_end + 1e-12
        initial = samples[0]
        forward = Rotation.from_quat(initial.body_quat[0], scalar_first=True).as_matrix()[:, 0]
        active = ("left", "right").index(primitive.active_hand)
        result.append(CachedPath(primitive, times, gloves, outward,
            initial.body_pos[0].copy(), float(np.arctan2(forward[1], forward[0])),
            np.asarray(primitive.record["target"]["head_world"], float),
            cKDTree(gloves[outward, active])))
    return result


def mount_optical_rotation(pitch_up_deg: float) -> np.ndarray:
    """Torso <- right/down/forward; positive pitch is UP in Phase 3 reports.

    Existing CameraConfig uses pitch_down_deg, hence its value is -pitch_up_deg.
    MuJoCo right/up/back columns become optical by negating columns 1 and 2.
    """
    level = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    return Rotation.from_rotvec([0., -np.deg2rad(pitch_up_deg), 0.]).as_matrix() @ level


def _rotate_xy(xy, angles):
    xy = np.asarray(xy)
    c, s = np.cos(angles), np.sin(angles)
    return np.stack((c * xy[..., 0] - s * xy[..., 1],
                     s * xy[..., 0] + c * xy[..., 1]), -1)


def target_centered_placement(accepted: Placement, nominal_target, head_world,
                              heading_rad: float, nominal_heading_rad: float,
                              relative_yaw_deg: float = 0., residual_xy=(0., 0.)) -> Placement:
    """Compose an entire-motion planar transform about the nominal target line."""
    delta = heading_rad - nominal_heading_rad + np.deg2rad(relative_yaw_deg)
    shift = (np.asarray(head_world)[:2] + _rotate_xy(residual_xy, heading_rad)
             - _rotate_xy(np.asarray(nominal_target)[:2], delta))
    xy = shift + _rotate_xy(accepted.xy, delta)
    return Placement(tuple(xy), accepted.yaw + delta)


def project_metrics(gloves, camera_position, optical_rotation):
    """Actual angular center admission, separate sphere/depth diagnostics."""
    optical = np.einsum("ncgi,nij->ncgj", gloves - camera_position[:, None, None], optical_rotation)
    r, d, f = np.moveaxis(optical, -1, 0)
    inside = ((f > 0) & (np.abs(r) <= f * TAN_H) & (np.abs(d) <= f * TAN_V))
    safe_f = np.maximum(f, 1e-12)
    margin = np.minimum(1 - np.abs(r) / (safe_f * TAN_H),
                        1 - np.abs(d) / (safe_f * TAN_V))
    sphere_margin = np.minimum((f * TAN_H - np.abs(r)) / np.sqrt(1 + TAN_H**2),
                               (f * TAN_V - np.abs(d)) / np.sqrt(1 + TAN_V**2)) - .08
    uv = optical[..., :2] / safe_f[..., None] / np.array([TAN_H, TAN_V])
    separation = np.linalg.norm(uv[..., 0, :] - uv[..., 1, :], axis=-1)
    return inside.all(-1), margin, sphere_margin, f, separation


def _placement_metrics(pelvis, headings, states, bounds):
    relative = pelvis[..., :2] - states["pelvis_world"][:, None, :2]
    local = _rotate_xy(relative, -states["pelvis_heading_rad"][:, None])
    distance = np.linalg.norm(local, axis=-1)
    bearing = np.rad2deg(np.arctan2(local[..., 1], local[..., 0]))
    facing = np.rad2deg(np.arctan2(
        np.sin(headings - states["pelvis_heading_rad"][:, None]),
        np.cos(headings - states["pelvis_heading_rad"][:, None])))
    valid = ((local[..., 0] >= bounds.min_forward_m)
             & (np.abs(local[..., 1]) <= bounds.max_abs_lateral_m)
             & (np.abs(bearing) <= bounds.max_abs_bearing_deg)
             & (distance >= bounds.min_pelvis_separation_m)
             & (distance <= bounds.max_pelvis_separation_m))
    # Facing is reported, deliberately not constrained to opposite pelvis axes.
    return valid, bearing, distance, facing, local


def search_primitive(states: Mapping[str, np.ndarray], path: CachedPath, *,
                     nominal_heading_rad: float, pitches=PITCHES_UP_DEG,
                     bounds=SearchBounds(), top_k: int = 4) -> dict[str, np.ndarray]:
    """Return compact candidates for ONE primitive; rendering is still required.

    Output leading axes: [pitch, state, rank] for top_*; [pitch,state] for
    available/baseline_joint; [state] for reach/rejection counts. Runtime xy/yaw
    are composed native-reference Placement arguments, not residual corrections.
    """
    if top_k < 1:
        raise ValueError("At least one retained alternative is required")
    head = np.asarray(states["head_world"], float)
    n = len(head)
    if n == 0:
        raise ValueError("An empty state domain is not a coverage study")
    for key, shape in (("head_world", (n, 3)), ("pelvis_world", (n, 3)),
                       ("pelvis_heading_rad", (n,)), ("torso_rotation", (n, 3, 3)),
                       ("camera_position_world", (n, 3))):
        if np.shape(states[key]) != shape or not np.isfinite(states[key]).all():
            raise ValueError(f"Invalid snapshot geometry: {key}")
    grid = np.array([(yaw, x, y) for yaw in bounds.yaw_degrees
                     for x in bounds.residual_xy_m for y in bounds.residual_xy_m], float)
    count = len(grid)
    k = min(top_k, count)
    heading = states["pelvis_heading_rad"][:, None]
    delta = heading - nominal_heading_rad + np.deg2rad(grid[None, :, 0])
    residual = _rotate_xy(grid[None, :, 1:], heading)
    shift = head[:, None, :2] + residual - _rotate_xy(path.target[:2], delta)
    # Inverse-transform frozen heads to one stored accepted-world path. The
    # nearest sampled outward point is independent of camera mounting pitch.
    queries = np.empty((n, count, 3))
    queries[..., :2] = path.target[:2] - _rotate_xy(residual, -delta)
    queries[..., 2] = head[:, None, 2]
    outward_min = path.tree.query(queries.reshape(-1, 3), workers=1)[0].reshape(n, count)
    initial_gloves = np.broadcast_to(path.gloves[0], (n, count, 2, 3)).copy()
    initial_gloves[..., :2] = _rotate_xy(path.gloves[0, :, :2], delta[..., None]) + shift[..., None, :]
    clearance = np.linalg.norm(initial_gloves - head[:, None, None], axis=-1)
    pelvis = np.broadcast_to(path.initial_pelvis, (n, count, 3)).copy()
    pelvis[..., :2] = _rotate_xy(path.initial_pelvis[:2], delta) + shift
    plausible, bearing, separation, facing, pelvis_heading_xy = _placement_metrics(
        pelvis, delta + path.initial_heading_rad, states, bounds)
    reached = outward_min <= ENCOUNTER
    safe = (clearance >= SAFE_CLEARANCE).all(-1)
    eligible = reached & safe & plausible
    runtime_xy = shift + _rotate_xy(path.primitive.placement.xy, delta)
    runtime_yaw = delta + path.primitive.placement.yaw
    # Report accepted placement literally, without target relocation or yaw.
    baseline_gloves = np.broadcast_to(path.gloves[0], (n, 1, 2, 3))
    baseline_min = path.tree.query(head, workers=1)[0]
    baseline_clearance = np.linalg.norm(path.gloves[0] - head[:, None], axis=-1)
    baseline_pelvis = np.broadcast_to(path.initial_pelvis, (n, 1, 3))
    baseline_plausible = _placement_metrics(baseline_pelvis,
        np.full((n, 1), path.initial_heading_rad), states, bounds)[0][:, 0]
    result = dict(snapshot_id=np.asarray(states["snapshot_id"]),
        pitches_up_deg=np.asarray(pitches),
        reachable=reached.any(-1), reachable_plausible=(reached & plausible).any(-1),
        reachable_safe_plausible=eligible.any(-1),
        reachable_candidate_count=reached.sum(-1),
        unsafe_reachable_candidate_count=(reached & ~safe).sum(-1),
        implausible_reachable_candidate_count=(reached & ~plausible).sum(-1),
        baseline_reachable=baseline_min <= ENCOUNTER,
        baseline_outward_min_m=baseline_min, baseline_initial_clearance_m=baseline_clearance,
        baseline_plausible=baseline_plausible)
    gathered: dict[str, list] = {}
    ranks = np.arange(n)[:, None]
    adjustment = (np.abs(grid[:, 0]) / 60 + np.linalg.norm(grid[:, 1:], axis=-1) / .15)
    for pitch in pitches:
        rotation = states["torso_rotation"] @ mount_optical_rotation(float(pitch))
        frustum, margin, sphere_margin, depth, image_separation = project_metrics(
            initial_gloves, states["camera_position_world"], rotation)
        joint = eligible & frustum
        # Exact reproducible ranking: normalized worst-hand boundary margin,
        # then a modest glove-separation reward and adjustment penalty. Ties
        # keep yaw/x/y grid order. Sphere containment is diagnostic, not a gate.
        score = (margin.min(-1) + .05 * np.minimum(image_separation, 1.)
                 - .025 * adjustment[None])
        order = np.argsort(-np.where(joint, score, -np.inf), axis=-1, kind="stable")[:, :k]
        keep = joint[ranks, order]
        values = dict(available=joint.any(-1), joint_candidate_count=joint.sum(-1),
            top_available=keep, top_candidate_index=order,
            top_xy=runtime_xy[ranks, order], top_yaw_rad=runtime_yaw[ranks, order],
            top_relative_yaw_deg=grid[order, 0], top_residual_xy_heading_m=grid[order, 1:],
            top_outward_min_m=outward_min[ranks, order],
            top_initial_clearance_m=clearance[ranks, order],
            top_center_margin=margin[ranks, order], top_sphere_margin_m=sphere_margin[ranks, order],
            top_optical_depth_m=depth[ranks, order], top_image_separation=image_separation[ranks, order],
            top_bearing_deg=bearing[ranks, order], top_pelvis_separation_m=separation[ranks, order],
            top_facing_relative_deg=facing[ranks, order],
            top_pelvis_heading_xy_m=pelvis_heading_xy[ranks, order], top_rank_score=score[ranks, order])
        baseline_fov = project_metrics(baseline_gloves, states["camera_position_world"], rotation)
        values.update(baseline_frustum=baseline_fov[0][:, 0],
            baseline_center_margin=baseline_fov[1][:, 0],
            baseline_joint=(baseline_min <= ENCOUNTER)
                & (baseline_clearance >= SAFE_CLEARANCE).all(-1) & baseline_fov[0][:, 0],
            baseline_joint_plausible=(baseline_min <= ENCOUNTER)
                & (baseline_clearance >= SAFE_CLEARANCE).all(-1)
                & baseline_fov[0][:, 0] & baseline_plausible)
        # Entry diagnostics refer to source-grid samples. No linear segment or
        # continuous crossing is substituted for a production FK event.
        sample_entry = np.full((n, k), np.nan)
        previous = np.full_like(sample_entry, np.nan)
        other_entry = np.full_like(sample_entry, np.nan)
        prep_fraction = np.full_like(sample_entry, np.nan)
        for i, j in zip(*np.nonzero(keep)):
            c = order[i, j]
            distances = np.linalg.norm(path.gloves - queries[i, c], axis=-1)
            active_hits = np.flatnonzero(path.outward & (distances[:, path.active_index] <= ENCOUNTER))
            first = active_hits[0]
            sample_entry[i, j] = path.time_s[first]
            previous[i, j] = path.time_s[max(first - 1, 0)]
            other_hits = np.flatnonzero(distances[:first, 1 - path.active_index] <= ENCOUNTER)
            if len(other_hits):
                other_entry[i, j] = path.time_s[other_hits[0]]
            # Short preparation/outward prefix only; terminal hold cannot
            # improve this trajectory-visibility diagnostic.
            prefix = path.gloves[:first + 1].copy()
            prefix[..., :2] = _rotate_xy(prefix[..., :2], delta[i, c]) + shift[i, c]
            optical = (prefix - states["camera_position_world"][i]) @ rotation[i]
            r, d, f = np.moveaxis(optical, -1, 0)
            visible = (f > 0) & (np.abs(r) <= f * TAN_H) & (np.abs(d) <= f * TAN_V)
            prep_fraction[i, j] = visible.all(-1).mean()
        values.update(top_first_entry_sample_s=sample_entry, top_previous_sample_s=previous,
            top_earlier_inactive_entry_s=other_entry,
            top_pre_entry_both_center_visible_fraction=prep_fraction)
        for name, value in values.items():
            gathered.setdefault(name, []).append(value)
    result.update({name: np.stack(value) for name, value in gathered.items()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--bounds-config", type=Path,
                        help="JSON with every explicit SearchBounds field; omitted uses initial study bounds")
    args = parser.parse_args(argv)
    bounds = SearchBounds()
    if args.bounds_config is not None:
        payload = json.loads(args.bounds_config.read_text())
        if set(payload) != {field.name for field in fields(SearchBounds)}:
            raise ValueError("Bounds config must specify every SearchBounds field, without extra keys")
        bounds = SearchBounds(**payload)
    output = args.output.resolve()
    if not output.is_relative_to((PROJECT / "outputs/task10d/phase3").resolve()):
        raise ValueError("Phase 3 study outputs must remain in outputs/task10d/phase3")
    output.mkdir(parents=True, exist_ok=False)
    with np.load(args.geometry) as source:
        geometry = {key: source[key].copy() for key in source.files}
    subset = json.loads(args.subset.read_text())
    ids = np.asarray(subset["snapshot_ids"] if isinstance(subset, dict) else subset, dtype=np.int64)
    locations = {int(value): i for i, value in enumerate(geometry["snapshot_id"])}
    indices = np.array([locations[int(value)] for value in ids])
    if len(set(ids)) != len(ids) or 53874 not in ids:
        raise ValueError("Training subset must retain distinct identities and nominal 53874")
    if "split_id" in geometry and not np.all(geometry["split_id"][indices] == 0):
        raise ValueError("Camera/placement selection is restricted to the TRAIN subset")
    states = {key: value[indices] for key, value in geometry.items() if value.shape[:1] == geometry["snapshot_id"].shape}
    nominal_index = int(np.flatnonzero(ids == 53874)[0])
    nominal_heading = float(states["pelvis_heading_rad"][nominal_index])
    paths = cache_library()
    records, files = [], []
    for path in paths:
        print(f"SEARCH {path.primitive.primitive_id} states={len(ids)}", flush=True)
        result = search_primitive(states, path, nominal_heading_rad=nominal_heading,
                                  top_k=args.top_k, bounds=bounds)
        destination = output / (path.primitive.primitive_id + ".npz")
        np.savez_compressed(destination, **result)
        files.append(destination.name)
        records.append(result)
        print(f"AVAILABLE {path.primitive.primitive_id} {result['available'].sum(-1).tolist()}", flush=True)
    availability = np.stack([r["available"] for r in records], -1)
    baseline = np.stack([r["baseline_joint"] for r in records], -1)
    families = ("jab", "cross", "left_hook", "right_hook")
    summary = dict(schema="task10d_phase3_geometric_search_v1", state_count=len(ids),
        primitive_ids=list(PRIMITIVE_IDS), candidate_files=files,
        geometry_path=str(args.geometry.resolve()), subset_path=str(args.subset.resolve()),
        geometry_sha256=hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        subset_sha256=hashlib.sha256(args.subset.read_bytes()).hexdigest(),
        index_sha256=INDEX_SHA256, nominal_snapshot_id=53874,
        nominal_heading_rad=nominal_heading, bounds=asdict(bounds),
        bounds_config_path=None if args.bounds_config is None else str(args.bounds_config.resolve()),
        bounds_config_sha256=None if args.bounds_config is None else hashlib.sha256(args.bounds_config.read_bytes()).hexdigest(),
        source_path=str(Path(__file__).resolve()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        command_argv=list(sys.argv[1:] if argv is None else argv),
        path_dt_s=DT, encounter_m=ENCOUNTER, initial_clearance_m=SAFE_CLEARANCE,
        outward_interval="source 0 through selected annotations.withdrawal_start_s, inclusive",
        ranking="worst-hand normalized boundary margin + .05*min(normalized separation,1) - .025*(abs(yaw_deg)/60 + norm(residual_xy)/.15); stable grid-order tie break",
        placement_frame="pelvis heading: +x forward, +y left; entire motion rotated about nominal target then shifted horizontally",
        baseline="literal accepted runtime Placement, no target-centering or heading change",
        pitch_convention="positive UP; existing mount_pitch_down_deg = -pitch_up_deg",
        observation_status="render-unverified; gross initial inter-character intersection unverified",
        counts_are="sampled angular geometry, NOT verified clean observation coverage",
        pitches=[])
    for p, pitch in enumerate(PITCHES_UP_DEG):
        family_available = np.stack([availability[p, :, [i for i, path in enumerate(paths)
            if path.primitive.family == family]].any(axis=0) for family in families], -1)
        summary["pitches"].append(dict(pitch_up_deg=pitch,
            states_with_attack=int(availability[p].any(-1).sum()),
            baseline_states_with_attack=int(baseline[p].any(-1).sum()),
            per_primitive_available=dict(zip(PRIMITIVE_IDS, availability[p].sum(0).tolist())),
            per_primitive_baseline=dict(zip(PRIMITIVE_IDS, baseline[p].sum(0).tolist())),
            nominal_available=dict(zip(PRIMITIVE_IDS, availability[p, nominal_index].tolist())),
            nominal_baseline=dict(zip(PRIMITIVE_IDS, baseline[p, nominal_index].tolist())),
            per_primitive_rejection_counts={identity: {
                "no_sampled_outward_reach": int((~record["reachable"]).sum()),
                "reach_but_outside_frontal_bounds": int((record["reachable"]
                    & ~record["reachable_plausible"]).sum()),
                "reach_and_frontal_but_unsafe_initial_gloves": int((record["reachable_plausible"]
                    & ~record["reachable_safe_plausible"]).sum()),
                "reach_fov_conflict": int((record["reachable_safe_plausible"]
                    & ~record["available"][p]).sum()),
                "render_unverified": int(record["available"][p].sum()),
            } for identity, record in zip(PRIMITIVE_IDS, records)},
            per_family_available=dict(zip(families, family_available.sum(0).tolist())),
            family_count_histogram=np.bincount(family_available.sum(-1), minlength=5).tolist(),
            primitive_count_histogram=np.bincount(availability[p].sum(-1), minlength=9).tolist()))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output / "cached_paths.npz", **{
        f"{p.primitive.primitive_id}__{key}": value for p in paths
        for key, value in (("time_s", p.time_s), ("gloves", p.gloves), ("outward", p.outward))})
    print(json.dumps(summary["pitches"], indent=2), flush=True)


if __name__ == "__main__":
    main()
