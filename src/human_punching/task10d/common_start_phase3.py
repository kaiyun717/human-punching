"""Fixed whole-motion registration and sample-ball reach maps for Phase 3B-II.

The frame uses H2's actual initial pelvis XY and projected heading. It preserves
world Z and every subsequent body/foot displacement. Reach is the union of 0.14 m
balls around active-hand 5 ms production samples in the accepted outward interval.
It is specific to these reference motions, not the humanoid's full workspace.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

from .placement_phase3 import CachedPath, ENCOUNTER
from .replay import Placement


def transform_points_yaw(points: np.ndarray, yaw: float, xy=(0., 0.)) -> np.ndarray:
    """One rigid world-up yaw/XY map, with arbitrary leading point dimensions."""
    points = np.asarray(points, dtype=float)
    if points.shape[-1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Expected finite XYZ points")
    if np.shape(xy) != (2,) or not np.isfinite((*xy, yaw)).all():
        raise ValueError("Expected finite planar yaw/translation")
    result = points.copy()
    c, s = np.cos(yaw), np.sin(yaw)
    result[..., 0] = c * points[..., 0] - s * points[..., 1] + xy[0]
    result[..., 1] = s * points[..., 0] + c * points[..., 1] + xy[1]
    return result


def common_start_transform(path: CachedPath, pelvis_xy=(0., 0.), heading_rad=0.) -> Placement:
    """Native runtime placement with the requested ACTUAL initial pelvis pose.

    ``path`` already includes its accepted runtime placement. Compose a single
    additional transform into a native Placement, without applying baked source
    translations again. Default actual initial pelvis XY=0 and heading=0.
    """
    delta = float(heading_rad) - path.initial_heading_rad
    rotated_initial = transform_points_yaw(path.initial_pelvis, delta)[:2]
    added_xy = np.asarray(pelvis_xy, dtype=float) - rotated_initial
    accepted = path.primitive.placement
    rotated_accepted = transform_points_yaw(np.array([*accepted.xy, 0.]), delta)[:2]
    return Placement(tuple(added_xy + rotated_accepted), delta + accepted.yaw)


def accepted_to_common(path: CachedPath, points: np.ndarray, pelvis_xy=(0., 0.), heading_rad=0.) -> np.ndarray:
    """Transform already accepted-world points; do not reapply native placement."""
    delta = float(heading_rad) - path.initial_heading_rad
    shift = np.asarray(pelvis_xy) - transform_points_yaw(path.initial_pelvis, delta)[:2]
    return transform_points_yaw(points, delta, shift)


def sample_ball_membership(samples: np.ndarray, query_xyz: np.ndarray,
                           radius: float = ENCOUNTER) -> np.ndarray:
    """Exact nearest retained-sample test; preserves gaps between sample balls."""
    samples, query_xyz = np.asarray(samples, float), np.asarray(query_xyz, float)
    if samples.ndim != 2 or samples.shape[1] != 3 or not len(samples):
        raise ValueError("Expected a nonempty sampled path [T,3]")
    if query_xyz.shape[-1:] != (3,) or not np.isfinite(query_xyz).all() or not np.isfinite(samples).all():
        raise ValueError("Sample-ball membership requires finite XYZ")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Sample-ball radius must be positive")
    distance = cKDTree(samples).query(query_xyz.reshape(-1, 3), workers=1)[0]
    return (distance <= radius).reshape(query_xyz.shape[:-1])


def overlap_counts(primitive_masks: np.ndarray, families: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Count each primitive once and each family once, including both variants."""
    masks = np.asarray(primitive_masks, dtype=bool)
    if masks.ndim < 2 or masks.shape[0] != len(families):
        raise ValueError("Expected primitive-first masks and one family per primitive")
    ordered = tuple(dict.fromkeys(families))
    family_masks = np.stack([masks[np.array([name == family for name in families])].any(0)
                             for family in ordered])
    return masks.sum(0, dtype=np.uint8), family_masks.sum(0, dtype=np.uint8)
