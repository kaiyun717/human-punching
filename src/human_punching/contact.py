"""Task 08 head/glove proxy geometry, independent of rendering and impact forces.

All centers are world-coordinate meters. Both gloves count at every phase.
These labels cover glove clearance only, not the other safe-arrival conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HEAD_RADIUS_M = 0.06
GLOVE_RADIUS_M = 0.08
GLOVE_RADII_M = (GLOVE_RADIUS_M, GLOVE_RADIUS_M)
SAFETY_MARGIN_M = 0.02
# Explicit decimal thresholds preserve the published equality convention.
CONTACT_RADIUS_M = 0.14
SAFE_CLEARANCE_RADIUS_M = 0.16
GEOMETRY_VERSION = "task08_head06_gloves08_margin02_v1"


@dataclass(frozen=True)
class Clearance:
    """Per-glove margins and their reductions over the two-glove axis."""

    distances_m: np.ndarray
    contact_clearance_m: np.ndarray
    safety_clearance_m: np.ndarray
    any_contact: np.ndarray
    fist_clearance_safe: np.ndarray
    min_safety_clearance_m: np.ndarray


def _finite_vectors(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite 3-vectors")
    return result


def evaluate_clearance(glove_centers, head_center) -> Clearance:
    """Evaluate both gloves against the *current* head center.

    ``glove_centers`` has shape ``(..., 2, 3)`` and ``head_center`` has
    broadcast-compatible shape ``(..., 3)``. Contact includes 0.14 m equality;
    safe glove clearance includes 0.16 m equality. Between these boundaries a
    pose violates clearance without physical proxy contact. Nonfinite input is
    rejected rather than silently classified as safe.
    """
    gloves = _finite_vectors(glove_centers, "glove_centers")
    head = _finite_vectors(head_center, "head_center")
    if gloves.ndim < 2 or gloves.shape[-2] != 2:
        raise ValueError("glove_centers must have shape (..., 2, 3)")
    distance = np.linalg.norm(gloves - head[..., None, :], axis=-1)
    contact = distance - CONTACT_RADIUS_M
    safety = distance - SAFE_CLEARANCE_RADIUS_M
    return Clearance(
        distances_m=distance,
        contact_clearance_m=contact,
        safety_clearance_m=safety,
        any_contact=np.any(distance <= CONTACT_RADIUS_M, axis=-1),
        fist_clearance_safe=np.all(distance >= SAFE_CLEARANCE_RADIUS_M, axis=-1),
        min_safety_clearance_m=np.min(safety, axis=-1),
    )


def world_attached_centers(body_positions, body_rotations, local_offsets) -> np.ndarray:
    """Apply ``p_center = p_body + R_body @ local_center`` with broadcasting.

    Use the attachment center in the compiled hand frame, not the wrist origin.
    Rotations have shape ``(..., 3, 3)`` and map body vectors into world space.
    In MuJoCo, a hand-attached site's ``data.site_xpos`` or a spherical proxy's
    ``data.geom_xpos`` already gives this result after forward kinematics.
    """
    positions = _finite_vectors(body_positions, "body_positions")
    offsets = _finite_vectors(local_offsets, "local_offsets")
    rotations = np.asarray(body_rotations, dtype=np.float64)
    if rotations.ndim < 2 or rotations.shape[-2:] != (3, 3):
        raise ValueError("body_rotations must have shape (..., 3, 3)")
    if not np.isfinite(rotations).all():
        raise ValueError("body_rotations must be finite")
    return positions + np.matmul(rotations, offsets[..., None])[..., 0]


def first_contact_time(time_s, glove_centers, head_center) -> float:
    """First two-glove contact time along a piecewise-linear center trajectory.

    Return an absolute time in the supplied clock, or infinity if there is no
    contact. ``glove_centers`` is ``(T, 2, 3)``; ``head_center`` is either ``(3,)``
    (a head frozen at launch for requested/reference TTC) or ``(T, 3)`` (actual
    moving head for observed contact). Subtract launch time to obtain TTC.

    This catches sphere crossings between saved frames. It interpolates centers,
    not articulated poses, and is an offline diagnostic rather than a replacement
    for evaluating actual physics substeps and latching their first event.
    """
    times = np.asarray(time_s, dtype=np.float64)
    gloves = _finite_vectors(glove_centers, "glove_centers")
    head = _finite_vectors(head_center, "head_center")
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("time_s must be a nonempty finite vector")
    if np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be strictly increasing")
    if gloves.shape != (len(times), 2, 3):
        raise ValueError("glove_centers must have shape (len(time_s), 2, 3)")
    if head.shape not in ((3,), (len(times), 3)):
        raise ValueError("head_center must have shape (3,) or (len(time_s), 3)")
    relative = gloves - (head if head.ndim == 1 else head[:, None, :])
    distances = np.linalg.norm(relative, axis=-1)
    if np.any(distances[0] <= CONTACT_RADIUS_M):
        return float(times[0])
    for index in range(len(times) - 1):
        earliest = np.inf
        for hand in range(2):
            start = relative[index, hand]
            delta = relative[index + 1, hand] - start
            a = float(delta @ delta)
            b = float(2 * (start @ delta))
            c = float(start @ start - CONTACT_RADIUS_M**2)
            discriminant = b * b - 4 * a * c
            if a > 0 and discriminant >= 0:
                entry = (-b - np.sqrt(discriminant)) / (2 * a)
                if 0 <= entry <= 1:
                    earliest = min(earliest, entry)
            # Preserve exact sampled equality despite root roundoff.
            if distances[index + 1, hand] <= CONTACT_RADIUS_M:
                earliest = min(earliest, 1.0)
        if np.isfinite(earliest):
            return float(times[index] + earliest * (times[index + 1] - times[index]))
    return float("inf")
