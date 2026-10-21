"""Independent 200 Hz recorded-evidence evaluator; never advances a simulator.

The recorder supplies actual forces, support-patch state and reset events.
Missing evidence fails closed. Geometric contact is not a physical-impact model.
Synthetic traces may exercise the metrics but never count as acceptance evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import CLEARANCE_DISTANCE_M, CONTACT_DISTANCE_M, PunchRequest, TimingContract


@dataclass(frozen=True)
class EvaluationThresholds:
    timing_error_s: float = .020
    outward_glove_rms_m: float = .020
    minimum_root_height_fraction: float = .75
    maximum_root_tilt_rad: float = .60
    maximum_support_anchor_drift_m: float = .030
    maximum_joint_limit_violation_rad: float = .020
    maximum_effort_limit_violation: float = 1e-4
    maximum_external_force: float = 1e-9
    guard_root_position_m: float = .030
    guard_root_orientation_rad: float = .10
    guard_joint_position_rad: float = .12
    guard_root_linear_speed_m_s: float = .05
    guard_root_angular_speed_rad_s: float = .12
    guard_joint_speed_rad_s: float = .50
    guard_glove_relative_position_m: float = .030
    guard_glove_relative_orientation_rad: float = math.pi / 12
    guard_glove_relative_speed_m_s: float = .15
    guard_dwell_s: float = 1.0
    maximum_guard_entry_after_ttc_s: float = 3.0

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError("Evaluation thresholds must be finite and positive")


@dataclass(frozen=True)
class EpisodeResult:
    family: str
    training_seed: int
    recipe_id: str
    spatial_cell_id: str
    ttc_cell_id: str
    source_group_id: str
    complete_cycle_success: bool
    acceptance_eligible: bool
    failure_categories: tuple[str, ...]
    first_event: str | None = None
    first_event_time_s: float | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    request_sha256: str = ""


_SHAPES = {
    "glove_position_w": (2, 3), "reference_glove_position_w": (2, 3),
    "root_position_w": (3,), "root_quat_wxyz": (4,),
    "root_linear_velocity_w": (3,), "root_angular_velocity_w": (3,),
    "glove_position_guard_frame": (2, 3), "glove_quat_guard_frame": (2, 4),
    "glove_velocity_guard_frame": (2, 3),
    "support_anchor_position_w": (2, 3), "support_patch_contact": (2,),
    "support_contact": (2,), "root_actuator_force": (6,),
    "external_support": (), "reset_event": (), "forbidden_collision": (),
    "human_g1_collision": (), "numerical_failure": (), "contact_overflow": (),
}
_BOOL_FIELDS = {"support_patch_contact", "support_contact", "external_support", "reset_event",
                "forbidden_collision", "human_g1_collision", "numerical_failure", "contact_overflow"}
_SCALARS = {"physics_dt_s", "policy_dt_s", "decimation", "integration_substeps",
            "physics_step_count", "outward_end_s", "inter_character_collision_enabled",
            "evidence_kind", "admission_passed", "domain_version"}


def _quat_angle(quaternion: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return 2 * np.arccos(np.clip(np.abs(np.sum(quaternion * reference, axis=-1)), 0, 1))


def evaluate_episode(request: PunchRequest, arrays: Mapping[str, Any], *, training_seed: int,
                     recipe_id: str, spatial_cell_id: str, ttc_cell_id: str,
                     thresholds: EvaluationThresholds | None = None) -> EpisodeResult:
    """TTC starts at request.launch_time_s; first sampled distance <= .14 is contact.

    Arrays contain a launch sample followed by every real 5 ms state. Reset is
    allowed only at sample zero. external_wrench is (samples, bodies, 6) evidence
    for *all* externally applied body forces, without cancellation by summation.
    Support anchors identify persistent planted patches, so heel pivot motion is
    not mistaken for slip. Changing anchors after slipping is prohibited.
    """
    arrays = {key: value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value
              for key, value in arrays.items()}
    base = dict(family=request.family.value, training_seed=training_seed, recipe_id=recipe_id,
                spatial_cell_id=spatial_cell_id, ttc_cell_id=ttc_cell_id,
                source_group_id=request.snapshot.source_group_id or request.snapshot.snapshot_id)
    required = set(_SHAPES) | _SCALARS | {
        "time_s", "joint_position", "joint_velocity", "joint_limits", "effort", "effort_limits",
        "external_wrench"}
    missing = sorted(required - set(arrays))
    if missing:
        return EpisodeResult(**base, complete_cycle_success=False, acceptance_eligible=False,
                             failure_categories=("missing_evidence",), metrics={"missing": missing})
    try:
        return _evaluate(request, arrays, thresholds or EvaluationThresholds(), base)
    except (ValueError, TypeError, IndexError, OverflowError) as error:
        return EpisodeResult(**base, complete_cycle_success=False, acceptance_eligible=False,
                             failure_categories=("malformed_evidence",),
                             metrics={"error": str(error)})


def _evaluate(request, arrays, thresholds, base):
    times = np.asarray(arrays["time_s"], dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("At least two finite physics timestamps are required")
    n = len(times)
    data = {}
    external_wrench = np.asarray(arrays["external_wrench"], dtype=float)
    if (external_wrench.ndim != 3 or external_wrench.shape[0] != n
            or external_wrench.shape[1] < 1 or external_wrench.shape[2] != 6
            or not np.isfinite(external_wrench).all()):
        raise ValueError("external_wrench must contain every body's finite applied wrench")
    for name, tail in _SHAPES.items():
        value = np.asarray(arrays[name])
        if value.shape != (n, *tail):
            raise ValueError(f"{name} must have shape {(n, *tail)}")
        if name in _BOOL_FIELDS:
            if value.dtype.kind != "b":
                raise ValueError(f"{name} requires Boolean recorded evidence")
        else:
            value = value.astype(float)
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite {name}")
        data[name] = value
    for name in ("joint_position", "joint_velocity", "effort"):
        value = np.asarray(arrays[name], dtype=float)
        if value.ndim != 2 or value.shape[0] != n or value.shape[1] == 0:
            raise ValueError(f"{name} requires a nonempty (samples, joints) array")
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {name}")
        data[name] = value
    q, qd, effort = (data[name] for name in ("joint_position", "joint_velocity", "effort"))
    limits = np.asarray(arrays["joint_limits"], dtype=float)
    effort_limits = np.asarray(arrays["effort_limits"], dtype=float)
    if (q.shape != qd.shape or q.shape != effort.shape or limits.shape != (q.shape[1], 2)
            or effort_limits.shape != (q.shape[1],) or not np.isfinite(limits).all()
            or not np.isfinite(effort_limits).all() or np.any(limits[:, 0] >= limits[:, 1])
            or np.any(effort_limits <= 0)):
        raise ValueError("Joint and effort arrays/limits disagree")
    for name in ("root_quat_wxyz", "glove_quat_guard_frame"):
        if np.any(abs(np.linalg.norm(data[name], axis=-1) - 1) > 1e-6):
            raise ValueError(f"{name} contains non-unit quaternions")
    for name in ("admission_passed", "inter_character_collision_enabled"):
        if not isinstance(arrays[name], (bool, np.bool_)):
            raise ValueError(f"{name} must be Boolean")
    TimingContract(float(arrays["physics_dt_s"]), float(arrays["policy_dt_s"]),
                   arrays["decimation"], arrays["integration_substeps"])
    if (not np.allclose(np.diff(times), .005, atol=1e-9, rtol=0)
            or abs(times[0] - request.launch_time_s) > 1e-9
            or arrays["physics_step_count"] != n - 1):
        raise ValueError("Trace must record each actual 5 ms step from the launch boundary")
    elapsed = times - request.launch_time_s
    outward_end = float(arrays["outward_end_s"])
    if not math.isfinite(outward_end) or not 0 < outward_end <= elapsed[-1]:
        raise ValueError("outward_end_s must be recorded relative to launch within the trace")
    failures, events = [], []

    def fail(name, bad):
        bad = np.asarray(bad, dtype=bool)
        indices = np.flatnonzero(bad)
        if len(indices):
            failures.append(name)
            events.append((float(elapsed[indices[0]]), name))

    distances = np.linalg.norm(data["glove_position_w"]
                               - request.snapshot.head_position_w, axis=-1)
    active = request.family.active_glove_index
    contact = np.flatnonzero(distances[:, active] <= CONTACT_DISTANCE_M)
    contact_time = float(elapsed[contact[0]]) if len(contact) else None
    if contact_time is not None:
        events.append((contact_time, "active_analytic_contact"))
    else:
        failures.append("active_contact_missing")
    timing_error = None if contact_time is None else contact_time - request.ttc_s
    if timing_error is not None and abs(timing_error) > thresholds.timing_error_s + 1e-12:
        failures.append("contact_timing")
    if contact_time is not None and contact_time > outward_end:
        failures.append("contact_outside_outward")
    fail("inactive_glove_contact", distances[:, 1 - active] <= CONTACT_DISTANCE_M)
    fail("initial_glove_clearance", np.array([np.any(distances[0] < CLEARANCE_DISTANCE_M)]))
    root = data["root_position_w"]
    if root[0, 2] <= 0:
        raise ValueError("Positive initial root height is required")
    quat = data["root_quat_wxyz"]
    tilt = np.arccos(np.clip(1 - 2*(quat[:, 1]**2 + quat[:, 2]**2), -1, 1))
    fail("balance", (root[:, 2] < thresholds.minimum_root_height_fraction * root[0, 2])
         | (tilt > thresholds.maximum_root_tilt_rad) | ~data["support_contact"].any(axis=1))
    joint_violation = np.maximum(limits[:, 0] - q, q - limits[:, 1]).max(axis=1)
    fail("joint_limit", joint_violation > thresholds.maximum_joint_limit_violation_rad)
    effort_violation = (np.abs(effort) - effort_limits).max(axis=1)
    fail("effort_limit", effort_violation > thresholds.maximum_effort_limit_violation)
    fail("external_force", np.max(np.abs(external_wrench), axis=(1, 2))
         > thresholds.maximum_external_force)
    fail("actuated_root", np.max(np.abs(data["root_actuator_force"]), axis=1)
         > thresholds.maximum_external_force)
    for name in ("external_support", "forbidden_collision", "numerical_failure", "contact_overflow"):
        fail(name, data[name])
    resets = data["reset_event"].copy()
    resets[0] = False
    fail("mid_cycle_reset", resets)
    anchors = data["support_anchor_position_w"]
    drift = np.linalg.norm(anchors[:, :, :2] - anchors[0, :, :2], axis=-1)
    fail("foot_slip", ((drift > thresholds.maximum_support_anchor_drift_m)
                       & data["support_patch_contact"]).any(axis=1))
    fail("foot_reposition", ((drift > thresholds.maximum_support_anchor_drift_m)
                            & ~data["support_patch_contact"]).any(axis=1))
    if arrays["inter_character_collision_enabled"]:
        # Required diagnostic probes are not counted as analytic-task acceptance.
        impact_diagnostic = True
    else:
        impact_diagnostic = False
        fail("unexpected_human_g1_collision", data["human_g1_collision"])
    glove = data["glove_position_guard_frame"]
    guard = (
        (np.linalg.norm(root - root[0], axis=-1) <= thresholds.guard_root_position_m)
        & (_quat_angle(quat, quat[0]) <= thresholds.guard_root_orientation_rad)
        & (np.max(np.abs(q - q[0]), axis=-1) <= thresholds.guard_joint_position_rad)
        & (np.linalg.norm(data["root_linear_velocity_w"], axis=-1)
           <= thresholds.guard_root_linear_speed_m_s)
        & (np.linalg.norm(data["root_angular_velocity_w"], axis=-1)
           <= thresholds.guard_root_angular_speed_rad_s)
        & (np.max(np.abs(qd), axis=-1) <= thresholds.guard_joint_speed_rad_s)
        & (np.linalg.norm(glove - glove[0], axis=-1).max(axis=1)
           <= thresholds.guard_glove_relative_position_m)
        & (_quat_angle(data["glove_quat_guard_frame"], data["glove_quat_guard_frame"][0])
           .max(axis=1) <= thresholds.guard_glove_relative_orientation_rad)
        & (np.linalg.norm(data["glove_velocity_guard_frame"], axis=-1).max(axis=1)
           <= thresholds.guard_glove_relative_speed_m_s)
        & data["support_contact"].all(axis=1)
        & (distances >= CLEARANCE_DISTANCE_M).all(axis=1)
    )
    if not guard[0]:
        failures.append("initial_guard_unready")
    guard &= elapsed >= outward_end
    if contact_time is not None:
        guard &= elapsed > contact_time
    false = np.flatnonzero(~guard)
    start = int(false[-1] + 1) if len(false) else 0
    dwell = float(elapsed[-1] - elapsed[start]) if start < n else 0.0
    guard_entry = float(elapsed[start]) if start < n else None
    if (dwell + 1e-12 < thresholds.guard_dwell_s or guard_entry is None
            or guard_entry > request.ttc_s + thresholds.maximum_guard_entry_after_ttc_s + 1e-12):
        failures.append("guard_recovery")
    if elapsed[-1] > request.ttc_s + thresholds.maximum_guard_entry_after_ttc_s + thresholds.guard_dwell_s + 1e-9:
        failures.append("cycle_deadline")
    outward = elapsed <= outward_end
    error = data["glove_position_w"][:, active] - data["reference_glove_position_w"][:, active]
    rms = float(np.sqrt(np.mean(np.sum(error[outward]**2, axis=1))))
    if rms > thresholds.outward_glove_rms_m:
        failures.append("outward_tracking")
    # Mechanical failure wins ties, preserving a substep failure at contact.
    events.sort(key=lambda event: (event[0], event[1] == "active_analytic_contact"))
    physical = str(arrays["evidence_kind"]) == "physics_rollout"
    eligible = (physical and bool(arrays["admission_passed"]) and bool(arrays["domain_version"])
                and bool(request.snapshot.source_group_id) and not impact_diagnostic)
    metrics = dict(first_active_contact_s=contact_time, timing_error_s=timing_error,
                   active_min_distance_m=float(distances[:, active].min()),
                   inactive_min_distance_m=float(distances[:, 1-active].min()),
                   outward_glove_rms_m=rms, guard_entry_s=guard_entry, guard_dwell_s=dwell,
                   maximum_support_anchor_drift_m=float(drift[data["support_patch_contact"]].max())
                   if data["support_patch_contact"].any() else None,
                   human_g1_contact_samples=int(data["human_g1_collision"].sum()),
                   impact_diagnostic=impact_diagnostic, sampling_quantization_s=.005,
                   maximum_root_tilt_rad=float(tilt.max()),
                   maximum_joint_limit_violation_rad=float(max(0, joint_violation.max())),
                   maximum_effort_limit_violation=float(max(0, effort_violation.max())))
    return EpisodeResult(**base, complete_cycle_success=not failures, acceptance_eligible=eligible,
                         failure_categories=tuple(dict.fromkeys(failures)),
                         first_event=events[0][1] if events else None,
                         first_event_time_s=events[0][0] if events else None, metrics=metrics,
                         request_sha256=request.fingerprint)


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if count == 0:
        return None
    if not 0 <= successes <= count or count < 0:
        raise ValueError("Invalid binomial counts")
    p = successes / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return max(0.0, center-radius), min(1.0, center+radius)


def summarize_results(results: Sequence[EpisodeResult], *, expected_seeds: Sequence[int],
                      expected_trials_per_family_seed: int = 1000,
                      expected_cells_per_family_seed: int = 100,
                      bootstrap_samples: int = 2000, bootstrap_seed: int = 10) -> dict[str, Any]:
    """Report each family/seed independently; point-rate gates are not CI gates.

    Cluster bootstrap resamples entire source command-trajectory groups. Pairing
    is checked using the same recipe IDs across seeds; missing/duplicate trials
    never disappear from the sample-completeness gate.
    """
    from .contracts import PunchFamily

    seeds = tuple(expected_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or bootstrap_samples < 1:
        raise ValueError("Unique expected seeds and a positive bootstrap budget are required")
    if expected_trials_per_family_seed < 1 or expected_cells_per_family_seed < 1:
        raise ValueError("Expected trial/cell counts must be positive")
    expected_families = {family.value for family in PunchFamily}
    if any(row.training_seed not in seeds or row.family not in expected_families for row in results):
        raise ValueError("Results contain an undeclared seed or punch family")
    rng = np.random.default_rng(bootstrap_seed)
    groups = {}
    paired = True
    for family in PunchFamily:
        recipe_sets = []
        for seed in seeds:
            rows = [row for row in results if row.family == family.value and row.training_seed == seed]
            ids = [row.recipe_id for row in rows]
            recipe_sets.append({row.recipe_id: row.request_sha256 for row in rows})
            success = np.array([row.complete_cycle_success and row.acceptance_eligible for row in rows])
            cell_counts, failures, clusters = {}, {}, {}
            for row, passed in zip(rows, success, strict=True):
                key = (row.spatial_cell_id, row.ttc_cell_id)
                cell_counts.setdefault(key, []).append(bool(passed))
                clusters.setdefault(row.source_group_id, []).append(bool(passed))
                for reason in row.failure_categories:
                    failures[reason] = failures.get(reason, 0) + 1
                if not row.acceptance_eligible:
                    failures["ineligible_evidence"] = failures.get("ineligible_evidence", 0) + 1
            cluster_interval = None
            if len(clusters) >= 2:
                counts = np.array([[sum(values), len(values)] for values in clusters.values()])
                samples = rng.integers(len(counts), size=(bootstrap_samples, len(counts)))
                sums = counts[samples].sum(axis=1)
                cluster_interval = tuple(np.quantile(sums[:, 0]/sums[:, 1], [.025, .975]).tolist())
            rate = float(success.mean()) if len(rows) else None
            expected_per_cell = expected_trials_per_family_seed / expected_cells_per_family_seed
            complete = (len(rows) == expected_trials_per_family_seed and len(ids) == len(set(ids))
                        and all(len(row.request_sha256) == 64 for row in rows)
                        and len(cell_counts) == expected_cells_per_family_seed
                        and all(len(values) == expected_per_cell for values in cell_counts.values()))
            groups[f"{family.value}/{seed}"] = dict(
                trials=len(rows), successes=int(success.sum()), success_rate=rate,
                wilson_95=wilson_interval(int(success.sum()), len(rows)),
                trajectory_cluster_95=cluster_interval, trajectory_groups=len(clusters),
                counts_complete=complete, failure_categories=failures,
                cells={f"{key[0]}/{key[1]}": {"count": len(values), "rate": float(np.mean(values))}
                       for key, values in cell_counts.items()},
                passed=bool(complete and rate is not None and rate >= .95
                            and all(np.mean(values) >= .90 for values in cell_counts.values())))
        paired &= all(ids == recipe_sets[0] for ids in recipe_sets[1:])
    return {"family_seed": groups, "paired_recipes_across_seeds": paired,
            "all_gates_passed": paired and len(seeds) >= 3 and all(row["passed"] for row in groups.values()),
            "rate_gate": "observed rate >= .95 per family/seed; >= .90 per spatial/TTC cell",
            "confidence": "Wilson and trajectory-cluster intervals reported, not substituted for point gates"}
