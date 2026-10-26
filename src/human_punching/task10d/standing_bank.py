"""Discrete actual-pelvis placement/action bank for one standing launch fixture.

Axes are [primitive, XY, yaw], with yaw samples on the circle.  Validity and
active outward aiming are separate.  A missing native/render check is unknown,
never an implicit pass.  This module performs no simulation or reference edits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from .common_start_phase3 import common_start_transform
from .placement_phase3 import (
    CachedPath, DT, ENCOUNTER, PRIMITIVE_IDS, SAFE_CLEARANCE,
    _rotate_xy, mount_optical_rotation, project_metrics,
)


FRAME_CONVENTION = (
    "G origin is the ground projection of the initial G1 pelvis; +X is its "
    "projected initial heading, +Y is left, +Z is world up. xy_G is the ACTUAL "
    "initial H2 pelvis position. yaw_deg is its projected heading relative to G; "
    "world heading = G1 launch heading + yaw. One whole-motion planar transform "
    "places and rotates H2 about its initial pelvis, preserving world Z."
)
COMPONENT_RULE = (
    "Circular runs of tested valid yaw samples; retain the longest run. Ties "
    "choose the smallest component-start index in the ordered [-180,175] grid. "
    "An all-valid circle starts at index zero. No untested angle is certified. "
    "Intersect with outward hit only AFTER selecting the valid component."
)
SAMPLING_RULE = (
    "Uniform over supported primary XY; then uniform over locally available "
    "families, uniform over that family's available reference variants, then "
    "uniform over the selected primitive's retained aimed tested yaw samples."
)
CALIBRATION_RULE = (
    "Among primary retained clean-valid aimed samples, prefer XY with >=0.05 m "
    "clearance from each primary envelope boundary, then yaw with both tested "
    "neighbors valid, then XY closest to (1.05,0) m, then greatest worst-hand "
    "camera center margin, then deepest sampled outward entry, then XY/yaw "
    "index. Preferences do not add admission thresholds. No dynamic outcome "
    "is used."
)


def candidate_grid() -> tuple[np.ndarray, np.ndarray]:
    """506 distinct XY and 72 headings, without the duplicate +180 endpoint."""
    x = np.arange(45, 151, 5, dtype=np.int16) / 100.
    y = np.arange(-55, 56, 5, dtype=np.int16) / 100.
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack((xx.ravel(), yy.ravel()), -1), np.arange(-180., 180., 5.)


def primary_envelope(xy_G: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy_G, float)
    r = np.linalg.norm(xy, axis=-1)
    # Grid values are exact decimal specifications; tolerate roundoff only at
    # the envelope boundaries, never at glove encounter/clearance thresholds.
    eps = 1e-12
    return ((r >= .70-eps) & (r <= 1.50+eps) & (xy[..., 0] >= .65-eps)
            & (np.abs(xy[..., 1]) <= .55+eps)
            & (np.abs(np.arctan2(xy[..., 1], xy[..., 0])) <= np.deg2rad(35)+eps))


def envelope_margin_m(xy_G: np.ndarray) -> np.ndarray:
    """Signed Euclidean distance to the nearest radial/line envelope boundary."""
    xy = np.asarray(xy_G, float)
    x, y = xy[..., 0], xy[..., 1]
    r = np.linalg.norm(xy, axis=-1)
    side = x*np.sin(np.deg2rad(35)) - np.abs(y)*np.cos(np.deg2rad(35))
    return np.minimum.reduce((r-.70, 1.50-r, x-.65, .55-np.abs(y), side))


def candidate_placement(path: CachedPath, state: Mapping[str, Any],
                        xy_G: Sequence[float], yaw_deg: float):
    heading = float(state["pelvis_heading_rad"])
    world_xy = np.asarray(state["pelvis_world"], float)[:2] + _rotate_xy(xy_G, heading)
    return common_start_transform(path, pelvis_xy=world_xy,
                                  heading_rad=heading+np.deg2rad(yaw_deg))


def _sampled_path_metrics(path: CachedPath, query: np.ndarray) -> dict[str, np.ndarray]:
    """Inverse-transform frozen heads, then query cached unchanged source paths."""
    shape = query.shape[:-1]
    flat = query.reshape(-1, 3)
    active, inactive = path.active_index, 1-path.active_index
    outward_min = path.tree.query(flat, workers=1)[0]
    recovery = ~path.outward
    recovery_min = (cKDTree(path.gloves[recovery, active]).query(flat, workers=1)[0]
                    if recovery.any() else np.full(len(flat), np.inf))
    inactive_min = cKDTree(path.gloves[:, inactive]).query(flat, workers=1)[0]
    fields = dict(outward_min_distance_m=outward_min,
                  active_recovery_min_distance_m=recovery_min,
                  inactive_cycle_min_distance_m=inactive_min,
                  outward_hit=outward_min <= ENCOUNTER,
                  active_recovery_hit=recovery_min <= ENCOUNTER,
                  inactive_cycle_hit=inactive_min <= ENCOUNTER)
    for name, points, times, selected in (
        ("outward", path.gloves[path.outward, active], path.time_s[path.outward],
         np.flatnonzero(fields["outward_hit"])),
        ("active_recovery", path.gloves[recovery, active], path.time_s[recovery],
         np.flatnonzero(fields["active_recovery_hit"])),
        ("inactive_cycle", path.gloves[:, inactive], path.time_s,
         np.flatnonzero(fields["inactive_cycle_hit"])),
    ):
        first = np.full(len(flat), np.nan)
        previous = np.full(len(flat), np.nan)
        for start in range(0, len(selected), 512):
            ids = selected[start:start+512]
            distances = np.linalg.norm(flat[ids, None, :] - points[None, :, :], axis=-1)
            hits = distances <= ENCOUNTER
            if not hits.any(-1).all():
                raise AssertionError("Nearest-sample reach and direct distances disagree")
            entry = hits.argmax(-1)
            first[ids] = times[entry]
            source_index = np.searchsorted(path.time_s, times[entry])
            previous[ids] = np.where(source_index > 0,
                path.time_s[np.maximum(source_index-1, 0)], np.nan)
        fields[f"first_{name}_sample_s"] = first
        fields[f"previous_{name}_sample_s"] = previous
    fields["active_recovery_only_hit"] = fields["active_recovery_hit"] & ~fields["outward_hit"]
    return {name: value.reshape(shape) for name, value in fields.items()}


def build_geometry(paths: Sequence[CachedPath], state: Mapping[str, Any], *,
                   xy_G: np.ndarray | None = None,
                   yaw_deg: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Batched static geometry and 5 ms reach for all eight accepted references.

    ``state`` must be measured after restoration/FK without integration. The
    function advances no simulation time. Native intersection/render checks are
    intentionally left to the combined scene pipeline.
    """
    if tuple(p.primitive.primitive_id for p in paths) != PRIMITIVE_IDS:
        raise ValueError("Explicitly supply all eight accepted primitives in index order")
    default_xy, default_yaw = candidate_grid()
    xy = default_xy if xy_G is None else np.asarray(xy_G, float)
    yaw = default_yaw if yaw_deg is None else np.asarray(yaw_deg, float)
    if xy.ndim != 2 or xy.shape[-1] != 2 or yaw.ndim != 1 or not len(xy) or not len(yaw):
        raise ValueError("Expected a nonempty [XY,2] placement grid and [yaw] headings")
    if not np.isfinite(xy).all() or not np.isfinite(yaw).all():
        raise ValueError("Finite placement samples are required")
    if len(np.unique(np.round(np.mod(yaw, 360), 10))) != len(yaw):
        raise ValueError("Yaw samples must not duplicate a circular endpoint")
    required = {"head_world": (3,), "pelvis_world": (3,),
                "pelvis_heading_rad": (), "camera_position_world": (3,),
                "torso_rotation": (3, 3)}
    measured = {key: np.asarray(state[key], float) for key in required}
    for key, shape in required.items():
        if measured[key].shape != shape or not np.isfinite(measured[key]).all():
            raise ValueError(f"Expected measured finite {key} of shape {shape}")
    head, pelvis = measured["head_world"], measured["pelvis_world"]
    heading = float(measured["pelvis_heading_rad"])
    world_xy = pelvis[:2] + _rotate_xy(xy, heading)
    world_yaw = heading + np.deg2rad(yaw)
    optical = measured["torso_rotation"] @ mount_optical_rotation(25.)
    if "camera_rotation" in state:
        actual_optical = np.asarray(state["camera_rotation"], float)
        if actual_optical.shape != (3, 3) or not np.isfinite(actual_optical).all():
            raise ValueError("Expected measured finite camera_rotation of shape (3,3)")
        np.testing.assert_allclose(actual_optical, optical, atol=2e-6, rtol=0,
                                   err_msg="Fixture must retain accepted +25 degree optical mount")
        optical = actual_optical
    measured["camera_rotation"] = optical
    rows = []
    for path in paths:
        delta = world_yaw-path.initial_heading_rad
        shift = world_xy[:, None, :] - _rotate_xy(path.initial_pelvis[:2], delta)[None]
        query = np.broadcast_to(head, (len(xy), len(yaw), 3)).copy()
        query[..., :2] = _rotate_xy(head[:2]-shift, -delta[None])
        row = _sampled_path_metrics(path, query)
        gloves = np.broadcast_to(path.gloves[0], (len(xy), len(yaw), 2, 3)).copy()
        gloves[..., :2] = (_rotate_xy(path.gloves[0, :, :2][None, :, :], delta[:, None])[None]
                           + shift[:, :, None, :])
        distances = np.linalg.norm(gloves-head, axis=-1)
        frustum, margin, sphere_margin, depth, separation = project_metrics(
            gloves.reshape(1, -1, 2, 3), measured["camera_position_world"][None], optical[None])
        row.update(initial_glove_distances_m=distances,
                   initial_clearance=(distances >= SAFE_CLEARANCE).all(-1),
                   initial_frustum=frustum.reshape(len(xy), len(yaw)),
                   initial_center_margins=margin.reshape(len(xy), len(yaw), 2),
                   initial_sphere_margins_m=sphere_margin.reshape(len(xy), len(yaw), 2),
                   initial_optical_depth_m=depth.reshape(len(xy), len(yaw), 2),
                   initial_image_separation=separation.reshape(len(xy), len(yaw)))
        row["geometric_initial_valid"] = row["initial_clearance"] & row["initial_frustum"]
        rows.append(row)
    bank = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    bank.update(xy_G=xy, yaw_deg=yaw, primary_xy=primary_envelope(xy),
                envelope_margin_m=envelope_margin_m(xy), actual_pelvis_world_xy=world_xy,
                actual_pelvis_world_yaw_rad=world_yaw,
                primitive_ids=np.array([p.primitive.primitive_id for p in paths]),
                families=np.array([p.primitive.family for p in paths]),
                active_hand=np.array([p.primitive.active_hand for p in paths]),
                reference_sha256=np.array([p.primitive.reference.sha256 for p in paths]),
                source_dt_s=np.array(DT), **measured)
    return bank


def circular_components(mask: np.ndarray) -> list[np.ndarray]:
    """Each component contains ordered tested indices, including wraparound."""
    values = np.asarray(mask, bool)
    if values.ndim != 1 or not len(values):
        raise ValueError("A nonempty one-dimensional circular mask is required")
    if values.all():
        return [np.arange(len(values), dtype=np.int16)]
    starts = np.flatnonzero(values & ~np.roll(values, 1))
    result = []
    for start in starts:
        length = 1
        while values[(start+length) % len(values)]:
            length += 1
        result.append((start+np.arange(length, dtype=np.int16)) % len(values))
    return result


def retain_largest_components(valid: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.asarray(valid, bool)
    if valid.ndim < 1 or not valid.shape[-1]:
        raise ValueError("Last axis must contain tested circular yaw samples")
    retained = np.zeros_like(valid)
    shape = valid.shape[:-1]
    count = np.zeros(shape, np.int16)
    start = np.full(shape, -1, np.int16)
    length = np.zeros(shape, np.int16)
    for loc in np.ndindex(shape):
        components = circular_components(valid[loc])
        count[loc] = len(components)
        if components:
            winner = min(components, key=lambda x: (-len(x), int(x[0])))
            retained[loc][winner] = True
            start[loc], length[loc] = winner[0], len(winner)
    return dict(retained_valid=retained, valid_component_count=count,
                retained_component_start=start, retained_component_length=length)


def finalize_bank(geometry: Mapping[str, np.ndarray], native_checked: np.ndarray,
                  native_valid: np.ndarray, clean_checked: np.ndarray,
                  clean_valid: np.ndarray) -> dict[str, np.ndarray]:
    """Join independent checks, retain valid run first, then intersect aiming."""
    bank = dict(geometry)
    shape = bank["outward_hit"].shape
    for key, values in (("native_checked", native_checked), ("native_valid", native_valid),
                        ("clean_checked", clean_checked), ("clean_valid", clean_valid)):
        values = np.asarray(values, bool)
        if values.shape != shape:
            raise ValueError(f"{key} shape must be {shape}")
        bank[key] = values.copy()
    if np.any(bank["native_valid"] & ~bank["native_checked"]):
        raise ValueError("Unchecked native scenes cannot pass")
    if np.any(bank["clean_valid"] & ~bank["clean_checked"]):
        raise ValueError("Unrendered scenes cannot pass clean observation")
    bank["initial_scene_valid"] = (bank["geometric_initial_valid"]
        & bank["native_checked"] & bank["native_valid"])
    bank["raw_valid"] = (bank["initial_scene_valid"] & bank["clean_checked"] & bank["clean_valid"])
    bank.update(retain_largest_components(bank["raw_valid"]))
    bank["raw_aimed"] = bank["raw_valid"] & bank["outward_hit"]
    bank["retained_aimed"] = bank["retained_valid"] & bank["outward_hit"]
    bank["aimed_lost_by_retention"] = bank["raw_aimed"] & ~bank["retained_aimed"]
    bank["raw_supported_xy"] = bank["raw_aimed"].any(axis=(0, 2))
    bank["retained_supported_xy"] = bank["retained_aimed"].any(axis=(0, 2))
    bank["primary_sampler_xy"] = bank["primary_xy"] & bank["retained_supported_xy"]
    bank["exploratory_supported_xy"] = ~bank["primary_xy"] & bank["retained_supported_xy"]
    bank["xy_lost_by_retention"] = bank["raw_supported_xy"] & ~bank["retained_supported_xy"]
    bank["primitive_xy_lost_by_retention"] = bank["raw_aimed"].any(-1) & ~bank["retained_aimed"].any(-1)
    return bank


def summarize_bank(bank: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Compact raw/retained, primary/exploratory and unknown-count ledger."""
    metrics = ("geometric_initial_valid", "native_checked", "initial_scene_valid",
               "clean_checked", "raw_valid", "outward_hit", "raw_aimed",
               "retained_valid", "retained_aimed", "aimed_lost_by_retention",
               "active_recovery_only_hit", "inactive_cycle_hit")
    result = dict(frame_convention=FRAME_CONVENTION, component_rule=COMPONENT_RULE,
        sampling_rule=SAMPLING_RULE, tested_yaws_only=True,
        candidate_xy_count=int(len(bank["xy_G"])), yaw_count=int(len(bank["yaw_deg"])),
        primitive_count=int(len(bank["primitive_ids"])),
        candidate_count=int(bank["outward_hit"].size), domains={})
    for domain, selected in (("primary", bank["primary_xy"]),
                             ("exploratory", ~bank["primary_xy"])):
        counts = {key: int(bank[key][:, selected].sum()) for key in metrics}
        counts.update(xy_count=int(selected.sum()),
            raw_supported_xy=int(np.count_nonzero(bank["raw_supported_xy"] & selected)),
            retained_supported_xy=int(np.count_nonzero(bank["retained_supported_xy"] & selected)),
            xy_lost_by_retention=int(np.count_nonzero(bank["xy_lost_by_retention"] & selected)),
            primitive_xy_lost_by_retention=int(bank["primitive_xy_lost_by_retention"][:, selected].sum()),
            geometrically_valid_native_unverified=int(np.count_nonzero(
                bank["geometric_initial_valid"][:, selected] & ~bank["native_checked"][:, selected])),
            initial_scene_valid_clean_unverified=int(np.count_nonzero(
                bank["initial_scene_valid"][:, selected] & ~bank["clean_checked"][:, selected])),
            by_primitive={})
        for p, identity in enumerate(bank["primitive_ids"]):
            row = {key: int(bank[key][p, selected].sum()) for key in metrics}
            row.update(raw_supported_xy=int(bank["raw_aimed"][p, selected].any(-1).sum()),
                       retained_supported_xy=int(bank["retained_aimed"][p, selected].any(-1).sum()),
                       multiple_valid_component_xy=int((bank["valid_component_count"][p, selected] > 1).sum()))
            counts["by_primitive"][str(identity)] = row
        result["domains"][domain] = counts
    return result


def choice_record(bank: Mapping[str, np.ndarray], primitive_index: int,
                  xy_index: int, yaw_index: int) -> dict[str, Any]:
    p, x, y = primitive_index, xy_index, yaw_index
    return dict(primitive_index=int(p), xy_index=int(x), yaw_index=int(y),
        primitive_id=str(bank["primitive_ids"][p]), family=str(bank["families"][p]),
        xy_G_m=bank["xy_G"][x].tolist(), yaw_deg=float(bank["yaw_deg"][y]),
        primary=bool(bank["primary_xy"][x]), retained_valid=bool(bank["retained_valid"][p,x,y]),
        frozen_head_outward_hit=bool(bank["outward_hit"][p,x,y]),
        first_outward_sample_s=float(bank["first_outward_sample_s"][p,x,y]),
        outward_min_distance_m=float(bank["outward_min_distance_m"][p,x,y]))


def sample_placement_first(bank: Mapping[str, np.ndarray], rng: np.random.Generator,
                           count: int = 1) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("Count must be nonnegative")
    supported = np.flatnonzero(bank["primary_sampler_xy"])
    if not len(supported):
        raise ValueError("No primary XY has a retained valid aimed choice")
    records = []
    for _ in range(count):
        xy = int(rng.choice(supported))
        primitives = np.flatnonzero(bank["retained_aimed"][:, xy].any(-1))
        available_families = np.unique(bank["families"][primitives])
        family = str(rng.choice(available_families))
        variants = primitives[bank["families"][primitives] == family]
        primitive = int(rng.choice(variants))
        yaw = int(rng.choice(np.flatnonzero(bank["retained_aimed"][primitive, xy])))
        records.append(choice_record(bank, primitive, xy, yaw))
    return records


def expected_sampler_exposure(bank: Mapping[str, np.ndarray]) -> dict[str, Any]:
    supported = np.flatnonzero(bank["primary_sampler_xy"])
    probability = np.zeros(len(bank["primitive_ids"]))
    if len(supported):
        for xy in supported:
            primitives = np.flatnonzero(bank["retained_aimed"][:, xy].any(-1))
            families = np.unique(bank["families"][primitives])
            for family in families:
                variants = primitives[bank["families"][primitives] == family]
                probability[variants] += 1. / len(supported) / len(families) / len(variants)
    return dict(rule=SAMPLING_RULE, supported_xy_count=int(len(supported)),
        primitive_probability={str(name):float(p) for name,p in zip(bank["primitive_ids"],probability)},
        family_probability={str(family):float(probability[bank["families"] == family].sum())
                            for family in np.unique(bank["families"])})


def calibrate_landmarks(paths: Sequence[CachedPath], bank: Mapping[str, np.ndarray], *,
                        fixture_id: str) -> dict[str, Any]:
    """One fixed active-outward source landmark per supported primitive.

    Uses the Phase 1 native-FK/brentq calculation within the first sampled entry
    bracket, restricted to the chosen active hand and accepted outward interval.
    This is not a moving-head measurement or an exhaustive continuous crossing
    proof. Per-candidate sampled entries remain untouched in the bank.
    """
    rows = []
    for p, path in enumerate(paths):
        selected = np.argwhere(bank["retained_aimed"][p] & bank["primary_xy"][:,None])
        base = dict(primitive_id=path.primitive.primitive_id, family=path.primitive.family,
                    reference_sha256=path.primitive.reference.sha256,
                    reference_npz=str(path.primitive.reference.path),
                    reference_json_sha256=path.primitive.metadata_sha256,
                    fixture_id=fixture_id, frozen_head_world=bank["head_world"].tolist(),
                    active_hand=path.primitive.active_hand,
                    accepted_outward_end_s=float(path.time_s[path.outward][-1]))
        if not len(selected):
            rows.append(dict(**base, supported=False, s_j_star=None,
                reason="No primary retained clean-valid sampled outward encounter on tested grid"))
            continue
        def rank(candidate):
            x, y = map(int, candidate)
            interior = bank["envelope_margin_m"][x] >= .05-1e-12
            neighbor = bool(bank["retained_valid"][p,x,(y-1)%len(bank["yaw_deg"])]
                            and bank["retained_valid"][p,x,(y+1)%len(bank["yaw_deg"])])
            return (not interior, not neighbor,
                    float(np.sum((bank["xy_G"][x]-[1.05,0.])**2)),
                    -float(bank["initial_center_margins"][p,x,y].min()),
                    float(bank["outward_min_distance_m"][p,x,y]), x, y)
        x,y = map(int, min(selected, key=rank))
        placement = candidate_placement(path, bank, bank["xy_G"][x], bank["yaw_deg"][y])
        first = float(bank["first_outward_sample_s"][p,x,y])
        previous = float(bank["previous_outward_sample_s"][p,x,y])
        if not np.isfinite(previous) or not 0 <= previous < first:
            raise ValueError("An admissible initial scene needs a strictly future outward entry bracket")
        def signed_distance(t):
            return float(np.linalg.norm(path.primitive.reference.sample(t, placement).glove_pos[
                path.active_index]-bank["head_world"])-ENCOUNTER)
        crossing = float(brentq(signed_distance, previous, first, xtol=1e-12))
        if not 0 < crossing <= path.time_s[path.outward][-1]+1e-12:
            raise ValueError("Calibration crossing must be positive and in the accepted outward interval")
        rows.append(dict(**base, supported=True, s_j_star=crossing,
            calibration_choice=choice_record(bank,p,x,y), native_placement=asdict(placement),
            first_sampled_entry_s=first, previous_sample_s=previous,
            crossing_bracket_s=[previous, first],
            canonical_timestamp_convention="Native production FK brentq root in first 5 ms sampled active-outward entry bracket, xtol=1e-12; not an exhaustive subgrid crossing proof",
            crossing_center_distance_m=signed_distance(crossing)+ENCOUNTER))
    return dict(schema="task10d_standing_fixed_landmarks_v1", fixture_id=fixture_id,
                calibration_rule=CALIBRATION_RULE, landmarks=rows,
                downstream_clock="alpha_j=s_j_star/T_strike; source_time=min(alpha_j*elapsed_time,reference_end); terminal hold",
                qualification="Fixed nominal calibration landmarks, not commanded TTC; no 0.5-0.8 s timing qualification")


@dataclass(frozen=True)
class FixedStrikeClock:
    """Clock using stored fixed calibration; never remeasures an episode target."""
    s_j_star: float
    T_strike: float
    reference_end_s: float

    def __post_init__(self):
        if not np.isfinite((self.s_j_star, self.T_strike, self.reference_end_s)).all():
            raise ValueError("Clock values must be finite")
        if not 0 < self.s_j_star <= self.reference_end_s or self.T_strike <= 0:
            raise ValueError("Positive fixed landmark, command and reference end are required")

    @property
    def alpha(self) -> float:
        return self.s_j_star / self.T_strike

    def source_time(self, elapsed_time_s: float) -> float:
        if not np.isfinite(elapsed_time_s) or elapsed_time_s < 0:
            raise ValueError("Elapsed time must be finite and nonnegative")
        return min(self.alpha*elapsed_time_s, self.reference_end_s)
