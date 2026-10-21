"""Task 10B Phase 2 geometry-only domain declarations and fail-closed commands.

This module performs no G1 policy inference or physics integration. A selected
snapshot or a constructed reference is never a qualified physical command.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .wbc.splits import assign_trajectory_groups

FAMILIES = ("jab", "cross", "left_hook", "right_hook")
SPLITS = ("train", "validation", "test")
COUNTS = {"train": 32, "validation": 8, "test": 16}
SEEDS = (21021, 21022, 21023)
CANDIDATE_TIMINGS = {"jab": (.60, .75, .90), "cross": (.60, .75, .90),
                     "left_hook": (.95, 1.10, 1.25), "right_hook": (.80, 1., 1.20)}
# Final timing triplets are provisional until the learned route is frozen. The
# middle value is never a training bin. No final evaluation is authorized by data.
FINAL_TIMINGS = {"jab": (.60, .675, .75), "cross": (.60, .675, .75),
                 "left_hook": (.95, 1.025, 1.10), "right_hook": (.80, .90, 1.)}
DISPOSITIONS = {0: "original_bank_ineligible", 1: "original_replica_split_conflict",
                2: "standing_calibration_only", 3: "eligible_not_selected",
                4: "selected_domain_snapshot"}


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_dispositions(command_ids, replicas, eligible):
    """Every original slot receives a reason; no conflicting row is resampled."""
    commands, replicas, eligible = np.asarray(command_ids), np.asarray(replicas), np.asarray(eligible, bool)
    if commands.shape != replicas.shape or commands.shape != eligible.shape or commands.ndim != 1:
        raise ValueError("Matched one-dimensional slot arrays required")
    if np.any(commands < 0) or np.any((replicas < 0) | (replicas > 2)):
        raise ValueError("Invalid original command or replica identity")
    assignment = assign_trajectory_groups([int(x) for x in np.unique(commands)])
    assigned = np.asarray([SPLITS.index(assignment[int(c)]) for c in commands], np.int8)
    codes = np.zeros(len(commands), np.int8)
    codes[eligible] = 3
    codes[eligible & (assigned != replicas)] = 1
    codes[eligible & (commands == 3307)] = 2
    return codes, assigned


def measured_domain(velocity) -> np.ndarray:
    """Keep original body vx/vy and native body-z gyro eligibility convention."""
    v = np.asarray(velocity, float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("Expected measured body vx, vy, native body-z gyro")
    eligible = np.isfinite(v).all(1) & (abs(v) <= .5).all(1)
    near = (np.linalg.norm(v[:, :2], axis=1) <= .1) & (abs(v[:, 2]) <= .1)
    return np.where(eligible, np.where(near, "D1", "D2"), "excluded")


def _maximin(rows, features, count):
    """Deterministic balanced coverage in predeclared normalized coordinates."""
    ids, x = np.asarray(rows, np.int64), np.asarray(features, float)
    if len(ids) < count or x.shape[0] != len(ids) or not np.isfinite(x).all():
        raise ValueError("Insufficient finite geometry candidates")
    order = np.argsort(ids, kind="stable")
    ids, x = ids[order], x[order]
    # First point is closest to the coordinate median; ties use original slot ID.
    first = int(np.argmin(np.sum((x - np.median(x, axis=0))**2, axis=1)))
    selected, distances = [], np.full(len(ids), np.inf)
    for _ in range(count):
        i = first if not selected else int(np.argmax(distances))
        selected.append(i)
        distances = np.minimum(distances, np.sum((x-x[i])**2, axis=1))
        distances[selected] = -np.inf
    return ids[selected].tolist()


def _representatives(commands, rows, velocity, root_height):
    # One medoid per command and D stratum, so sixteen adjacent times cannot
    # masquerade as sixteen independent groups.
    representatives = []
    for command in np.unique(commands[rows]):
        members = rows[commands[rows] == command]
        x = np.column_stack((velocity[members] / .5, root_height[members] / .1))
        representatives.append(int(members[np.argmin(np.sum((x-np.median(x, axis=0))**2, axis=1))]))
    return np.asarray(representatives, np.int64)


def start_perturbations(seed: int, *, starts: int = 4) -> dict:
    """Nominal plus three local joint perturbations; root/support remain fixed.

    These are deltas, not pose replacements inside a trial. Joint limits must be
    applied once to an admitted saved start before uninterrupted execution.
    """
    if starts != 4 or type(seed) is not int:
        raise ValueError("Phase 2 declares exactly four starts and an integer seed")
    rng = np.random.default_rng(seed)
    q = np.zeros((starts, 48)); v = np.zeros((starts, 54))
    q[1:] = np.clip(rng.normal(0, .005, (starts-1, 48)), -.015, .015)
    v[1:, 6:] = np.clip(rng.normal(0, .02, (starts-1, 48)), -.06, .06)
    return {"schema": "Task10BPhase2Starts/v1", "seed": seed,
            "joint_position_delta_rad": q.tolist(), "generalized_velocity_delta": v.tolist(),
            "root_position_orientation_and_velocity_delta": "zero; root/support expansion remains unqualified",
            "previous_raw_action_and_controller_history": "restore from identical saved state for family comparisons",
            "joint_limit_rule": "clip nominal plus delta to original limits before trial only"}


@dataclass(frozen=True)
class LaunchCommand:
    family: str
    requested_ttc_s: float
    target_w: tuple[float, float, float]
    start_id: str
    reference_sha256: str
    checkpoint_sha256: str
    predicate_version: str = "task10b_functional_guard_v1"
    style: str = "original"

    def __post_init__(self):
        object.__setattr__(self, "target_w", tuple(float(x) for x in self.target_w))
        if (self.family not in FAMILIES or not math.isfinite(self.requested_ttc_s)
                or self.requested_ttc_s < .5 or len(self.target_w) != 3
                or not all(map(math.isfinite, self.target_w)) or not self.start_id
                or len(self.reference_sha256) != 64 or len(self.checkpoint_sha256) != 64):
            raise ValueError("A complete finite immutable launch request is required")

    @property
    def identity(self):
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class EventClock:
    request_s: float
    preparation_complete_s: float
    readiness_s: float
    final_planning_s: float
    launch_s: float
    outward_complete_s: float
    first_contact_s: float | None = None
    recovery_s: float | None = None

    def __post_init__(self):
        times = [self.request_s, self.preparation_complete_s, self.readiness_s,
                 self.final_planning_s, self.launch_s, self.outward_complete_s]
        if not all(map(math.isfinite, times)) or any(b < a for a, b in zip(times, times[1:])):
            raise ValueError("Preparation, commitment, launch, and outward clocks must be ordered")
        if self.launch_s < 0 or abs(self.launch_s/.020-round(self.launch_s/.020)) > 1.e-8:
            raise ValueError("Actual launch requires a 20 ms actor boundary")
        if self.first_contact_s is not None and (not math.isfinite(self.first_contact_s) or self.first_contact_s < self.request_s):
            raise ValueError("First contact precedes execution")
        if self.recovery_s is not None and (not math.isfinite(self.recovery_s) or self.recovery_s < self.outward_complete_s):
            raise ValueError("Recovery must follow outward completion")

    def timing_valid(self, requested_ttc_s: float):
        if not math.isfinite(requested_ttc_s) or requested_ttc_s < .5 or self.first_contact_s is None:
            return False
        delay = self.first_contact_s-self.launch_s
        return (delay >= max(.5, requested_ttc_s-.020)-1.e-10
                and delay <= requested_ttc_s+.020+1.e-10
                and self.first_contact_s <= self.outward_complete_s+1.e-10)


def feasibility_catalog(requests: Iterable[LaunchCommand], evidence: Mapping[str, dict]) -> dict:
    """Exact identity lookup, no nearest target/TTC/start substitution.

    A truthy screening flag cannot admit a request: physical qualification and
    an independent audit receipt are mandatory evidence on the exact command.
    """
    rows, seen, scenarios = [], set(), {}
    for request in requests:
        if request.identity in seen:
            raise ValueError("Duplicate immutable request")
        seen.add(request.identity)
        proof = evidence.get(request.identity, {})
        legal = (proof.get("qualified") is True and proof.get("independent_audit_pass") is True
                 and proof.get("request_identity") == request.identity
                 and bool(proof.get("qualification_receipt_sha256")))
        scenario = fingerprint({"target_w": request.target_w, "start_id": request.start_id,
                                "ttc_s": request.requested_ttc_s, "style": request.style})
        scenarios.setdefault(scenario, set())
        if legal:
            scenarios[scenario].add(request.family)
        rows.append({"request_identity": request.identity, "scenario": scenario,
                     "request": asdict(request), "legal": legal,
                     "reason": "qualified_exact_command" if legal else proof.get("rejection_reason", "physical_qualification_missing")})
    coverage = Counter(len(v) for v in scenarios.values())
    return {"schema": "Task10BPhase2FeasibilityMask/v1", "requests": rows,
            "requested_scenarios": len(scenarios),
            "coverage_zero_to_four": {str(i): coverage[i] for i in range(5)},
            "multi_choice_scenarios": sorted(k for k,v in scenarios.items() if len(v) >= 2),
            "four_family_scenarios": sorted(k for k,v in scenarios.items() if len(v) == 4),
            "nearest_reference_fallback": False}


def build_domain(output: Path, *, project_root: Path, bank_root: Path,
                 g1_asset_root: Path, nominal_plant_path: Path) -> dict:
    """Freeze geometry-only candidate group IDs before native timing trials."""
    from .wbc.snapshots import SnapshotBank, head_point_state, load_verified_g1_model
    from .wbc.contracts import quaternion_matrix

    root, output = Path(project_root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("Fresh human-punching-owned output required")
    output.mkdir(parents=True, exist_ok=False)
    bank = SnapshotBank(bank_root)
    coverage = bank.index["coverage_index"]
    coverage_path = (bank.directory/coverage["payload"]).resolve()
    if not coverage_path.is_relative_to(bank.directory) or file_hash(coverage_path) != coverage["sha256"]:
        raise ValueError("Coverage payload provenance mismatch")
    with np.load(coverage_path, allow_pickle=False) as source:
        stored = source["stored_slot_ids"].copy()
        velocity = source["stored_actual_velocity"].copy()
        height = source["stored_root_height"].copy()
        eligible_ids = source["eligible_slot_ids"].copy()
    slots = bank.locations["slot_id"]
    if not np.array_equal(stored, slots):
        raise ValueError("Explicit slot addressing required")
    eligible = np.zeros(len(slots), bool); eligible[eligible_ids] = True
    commands, replicas = bank.locations["command_id"], bank.locations["replica_id"]
    codes, assignment = snapshot_dispositions(commands, replicas, eligible)
    domain = measured_domain(velocity)
    if np.any(eligible & (domain == "excluded")):
        raise ValueError("Original eligibility disagrees with measured velocity")
    g1, geometry = load_verified_g1_model(g1_asset_root, bank, nominal_plant_path)
    # Geometry-only medoid/maximin shortlist: four times each domain quota.
    # Head coordinates are subsequently measured for every shortlisted state.
    shortlists, pool_counts = {}, {}
    for split, total in COUNTS.items():
        for stage in ("D1", "D2"):
            available = slots[(codes == 3) & (assignment == SPLITS.index(split)) & (domain == stage)]
            reps = _representatives(commands, available, velocity, height)
            x = np.column_stack((velocity[reps]/.5, height[reps]/.1))
            shortlists[split, stage] = _maximin(reps, x, min(len(reps), 4*total//2))
            pool_counts[split+"_"+stage] = {"eligible_compatible_rows": len(available), "command_groups": len(reps)}
    derived = {}
    all_shortlisted = sorted(set(x for rows in shortlists.values() for x in rows),
                             key=lambda x: (int(bank.locations["shard_ordinal"][x]), x))
    for slot in all_shortlisted:
        state = bank.state(slot)
        head, head_velocity = head_point_state(g1, state.qpos, state.qvel)
        rotation = quaternion_matrix(state.qpos[3:7])
        measured = np.r_[rotation.T @ np.asarray(state.qvel[:3]), state.qvel[5]]
        if not np.allclose(measured[[0,1,3]], velocity[slot], atol=2.e-6, rtol=0):
            raise ValueError("Saved root velocity disagrees with original native measurement")
        derived[slot] = {"state": state, "head_position_w": head.tolist(),
                         "head_velocity_w": head_velocity.tolist(),
                         "head_position_root_frame": (rotation.T @ (head-np.asarray(state.qpos[:3]))).tolist()}
    chosen, selected_groups = [], set()
    for split, total in COUNTS.items():
        for stage in ("D1", "D2"):
            rows = [s for s in shortlists[split,stage] if int(commands[s]) not in selected_groups]
            features = np.array([list(velocity[s]/.5)+list(np.asarray(derived[s]["head_position_root_frame"])/.1) for s in rows])
            selected = _maximin(rows, features, total//2)
            for slot in selected:
                d = derived[slot]; state = d.pop("state")
                selected_groups.add(state.command_id); codes[slot] = 4
                chosen.append({"split": split, "domain_stage": stage,
                    "measured_velocity_body_vx_vy_native_gyro_z": velocity[slot].tolist(),
                    "snapshot": asdict(state), **d})
    if len(selected_groups) != sum(COUNTS.values()):
        raise ValueError("Command group independence failed")
    ledger = output/"all_slot_dispositions.npz"
    np.savez_compressed(ledger, slot_id=slots, command_id=commands, replica_id=replicas,
                        assigned_split_id=assignment, domain_stage=domain, disposition_code=codes)
    report = {"schema": "Task10BPhase2Domain/v1", "geometry_only": True,
        "g1_actor_executed": False, "physics_integrated": False,
        "physical_domain_qualified": False, "selected_command_groups": chosen,
        "counts": COUNTS, "stage_quota_per_split": "half D1, half D2; distinct command groups",
        "selection": "Per command/stage medoid in measured body velocity/root height; 4x quota maximin shortlist; final maximin in measured velocity and actual head center relative to pelvis",
        "selection_feature_scales": {"velocity": .5, "root_height_and_head_position": .1},
        "split_assignment": "task10-command-split-v1 SHA256 seed 10, 70/15/15; original replica split preserved",
        "calibration_command": 3307, "pool_counts": pool_counts,
        "velocity_convention": "body-frame root vx/vy; native body-z gyro qvel[5], not Euler yaw derivative; head point velocity is Jp(q) qvel",
        "candidate_timings_s": CANDIDATE_TIMINGS, "provisional_final_timings_s": FINAL_TIMINGS,
        "final_timing_semantics": "two trained endpoints plus untrained interior middle; frozen learned-route receipt required before final tests",
        "fine_tuning_seeds": SEEDS,
        "starts": {split: start_perturbations(21020+i) for i,split in enumerate(SPLITS)},
        "placement_candidates": {"longitudinal_delta_m": [-.05, 0., .05], "lateral_delta_m": [-.05, 0., .05], "yaw_delta_deg": [-5., 0., 5.], "qualified": False},
        "validation_attempts_per_family": 96, "final_attempts_per_family_seed": 192,
        "final_evaluation_executed": False,
        "all_slot_ledger": {"path": ledger.name, "sha256": file_hash(ledger), "codes": DISPOSITIONS,
                            "counts": {DISPOSITIONS[k]: int(np.sum(codes == k)) for k in DISPOSITIONS}},
        "snapshot_manifest_sha256": file_hash(bank.directory/"manifest.json"),
        "coverage_sha256": coverage["sha256"], "g1_geometry": geometry,
        "qualified_command_catalog": [], "requested_scenario_coverage": "not yet instantiated; no physical qualification credit",
        "limitations": ["Target library is predeclared geometry only; timing is expanded first.",
            "No G1 dynamics, moving-head interaction or SAV outcome.",
            "Four-start deltas require physically admitted states; shared-start connectors remain unqualified.",
            "Calibration trajectories are not independent train/validation/test groups."]}
    destination = output/"domain.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return {"path": str(destination), "sha256": file_hash(destination), "groups": len(chosen),
            "physical_domain_qualified": False, "counts": report["all_slot_ledger"]["counts"]}
