"""Fail-closed request admission against an explicitly qualified coupled mask.

Being inside search bounds never admits a request. Qualification manifests are
evidence produced by a later physical experiment, not by this module or by IK.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .contracts import CLEARANCE_DISTANCE_M, PunchFamily, PunchRequest, SearchBounds


class RejectionReason(str, Enum):
    OUTSIDE_SEARCH_BOUNDS = "outside_search_bounds"
    DOMAIN_UNQUALIFIED = "domain_unqualified"
    REQUEST_NOT_QUALIFIED = "request_not_qualified"
    GUARD_EVIDENCE_MISSING = "guard_evidence_missing"
    GUARD_UNREADY = "guard_unready"
    INITIAL_GLOVE_CLEARANCE = "initial_glove_clearance"
    INITIAL_BODY_COLLISION = "initial_body_collision"


def _interval(value, name):
    value = tuple(float(item) for item in value)
    if len(value) != 2 or not all(map(math.isfinite, value)) or value[0] > value[1]:
        raise ValueError(f"{name} requires an ordered finite interval")
    return value


@dataclass(frozen=True)
class QualificationCell:
    cell_id: str
    family: PunchFamily
    forward_m: tuple[float, float]
    lateral_m: tuple[float, float]
    ttc_s: tuple[float, float]
    head_height_m: tuple[float, float]
    guard_ids: tuple[str, ...]
    snapshot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", PunchFamily.coerce(self.family))
        for name in ("forward_m", "lateral_m", "ttc_s", "head_height_m"):
            object.__setattr__(self, name, _interval(getattr(self, name), name))
        for name in ("guard_ids", "snapshot_ids"):
            value = tuple(getattr(self, name))
            if not value or any(not isinstance(item, str) or not item for item in value):
                raise ValueError(f"{name} must explicitly identify supported initial conditions")
            object.__setattr__(self, name, value)
        if not self.cell_id:
            raise ValueError("Qualification cells require stable identities")

    def contains(self, request: PunchRequest) -> bool:
        forward, lateral = request.snapshot.relative_placement()
        return (request.family == self.family and request.guard_id in self.guard_ids
                and request.snapshot.snapshot_id in self.snapshot_ids
                and all(low <= value <= high for value, (low, high) in (
                    (forward, self.forward_m), (lateral, self.lateral_m),
                    (request.ttc_s, self.ttc_s),
                    (request.snapshot.head_position_w[2], self.head_height_m))))


@dataclass(frozen=True)
class FeasibleDomain:
    version: str
    cells: tuple[QualificationCell, ...] = ()
    qualification_passed: bool = False
    report_sha256: str = ""
    model_sha256: str = ""
    controller_sha256: str = ""
    reference_generator_sha256: str = ""
    runtime_identity: str = ""
    physical_trial_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", tuple(self.cells))
        if not self.version or len({cell.cell_id for cell in self.cells}) != len(self.cells):
            raise ValueError("Domain version and unique cell identities are required")
        if type(self.qualification_passed) is not bool:
            raise ValueError("qualification_passed must be Boolean")
        if self.qualification_passed:
            hashes = (self.report_sha256, self.model_sha256, self.controller_sha256,
                      self.reference_generator_sha256)
            if (not self.cells or not self.runtime_identity or self.physical_trial_count <= 0
                    or any(len(item) != 64 or any(c not in "0123456789abcdef" for c in item)
                           for item in hashes)):
                raise ValueError("Qualified domains require physical evidence and pinned components")


@dataclass(frozen=True)
class GuardEvidence:
    stable_guard: bool
    admissible_joint_state: bool
    in_place_stance: bool
    no_external_support: bool
    body_collision_free: bool
    glove_distances_m: tuple[float, float]

    def __post_init__(self) -> None:
        for name in ("stable_guard", "admissible_joint_state", "in_place_stance",
                     "no_external_support", "body_collision_free"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} requires explicit Boolean evidence")
        distances = tuple(float(item) for item in self.glove_distances_m)
        if len(distances) != 2 or not all(math.isfinite(x) and x >= 0 for x in distances):
            raise ValueError("Both finite glove/head distances are required")
        object.__setattr__(self, "glove_distances_m", distances)


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reasons: tuple[RejectionReason, ...]
    cell_id: str | None = None
    domain_version: str | None = None


def admit_request(request: PunchRequest, domain: FeasibleDomain | None,
                  guard: GuardEvidence | None, *, bounds: SearchBounds | None = None
                  ) -> AdmissionDecision:
    """Reject rather than clamp, re-aim, substitute a guard or alter requested TTC."""
    reasons = []
    if not (bounds or SearchBounds()).contains(request):
        reasons.append(RejectionReason.OUTSIDE_SEARCH_BOUNDS)
    cell = None
    if domain is None or not domain.qualification_passed:
        reasons.append(RejectionReason.DOMAIN_UNQUALIFIED)
    else:
        cell = next((item for item in domain.cells if item.contains(request)), None)
        if cell is None:
            reasons.append(RejectionReason.REQUEST_NOT_QUALIFIED)
    if guard is None:
        reasons.append(RejectionReason.GUARD_EVIDENCE_MISSING)
    else:
        if not all((guard.stable_guard, guard.admissible_joint_state,
                    guard.in_place_stance, guard.no_external_support)):
            reasons.append(RejectionReason.GUARD_UNREADY)
        if not guard.body_collision_free:
            reasons.append(RejectionReason.INITIAL_BODY_COLLISION)
        if min(guard.glove_distances_m) < CLEARANCE_DISTANCE_M:
            reasons.append(RejectionReason.INITIAL_GLOVE_CLEARANCE)
    return AdmissionDecision(not reasons, tuple(reasons),
                             cell.cell_id if cell is not None and not reasons else None,
                             domain.version if domain is not None else None)
