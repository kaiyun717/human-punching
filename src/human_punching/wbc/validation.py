"""Validation orchestration around inherited PPO, without a physical runner.

The callback protocol deliberately requires hashed native per-episode traces.
No callback is supplied here: production must fail closed until the independent
trained-actor whole-cycle evaluator exists and is qualified. CPU fixtures can
exercise decisions, but cannot masquerade as native validation panels.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .budget import plateau
from .checkpointing import capture_rng, restore_rng
from .contracts import LaunchSnapshot, PunchRequest
from .evaluation import evaluate_episode
from .provenance import PROJECT_ROOT, sha256_file

INTERVAL = 250
FAMILIES = ("jab", "cross", "left_hook", "right_hook")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tensor_digest(values: dict) -> str:
    import torch
    result = hashlib.sha256()
    for key, value in sorted(values.items()):
        tensor = value.detach().contiguous().cpu()
        result.update(key.encode()+str(tensor.dtype).encode()+str(tuple(tensor.shape)).encode())
        result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def iteration_chunks(completed: int, requested: int) -> tuple[int, ...]:
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (completed, requested)) or completed < 0 or requested < 1:
        raise ValueError("Completed/requested counts must be nonnegative/positive integers")
    chunks = []
    while requested:
        size = min(INTERVAL-completed % INTERVAL, requested)
        chunks.append(size)
        completed += size
        requested -= size
    return tuple(chunks)


@dataclass(frozen=True)
class PanelExpectation:
    completed_iterations: int
    stage: str
    training_seed: int
    policy_sha256: str
    normalization_sha256: str
    checkpoint_contract_sha256: str
    reference_manifest_sha256: str
    recipe_manifest_sha256: str
    domain_version: str
    episodes_per_family: int = 256

    def __post_init__(self):
        for name in ("completed_iterations", "training_seed", "episodes_per_family"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "training_seed" else 1):
                raise ValueError(f"{name} must be an exact nonnegative seed or positive count")
        if self.stage not in ("canonical", "domain", "guard_start"):
            raise ValueError("Validation requires a completed learner boundary and declared stage")
        if not isinstance(self.domain_version, str) or not self.domain_version:
            raise ValueError("A qualified domain and positive per-family sample count are required")
        for name, value in vars(self).items():
            if name.endswith("sha256") and (not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError(f"Invalid {name}")


class PhysicalValidationCallback(Protocol):
    """Caller owns a separate environment; returns a native evidence report path.

    It must neither reset nor advance the training environment, run PPO, update
    normalization, nor read later G1 state into committed punch plans. Every
    recipe needs a complete 5 ms trace including failures and rejected starts.
    """
    def __call__(self, actor, expectation: PanelExpectation) -> Path: ...


@contextmanager
def isolated_validation_actor(runner):
    """Copy actor+normalizer, preserve live learner mode/state and rank-local RNG."""
    live = runner.alg.get_policy()
    state_before = tensor_digest(live.state_dict())
    mode_before = {name: module.training for name, module in live.named_modules()}
    rng = capture_rng(runner.device)
    try:
        actor = live.as_jit().to(runner.device).eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        state = actor.state_dict()
        identity = {"policy_sha256": tensor_digest(state),
                    "normalization_sha256": tensor_digest({key: value for key, value in state.items()
                                                           if key.startswith("obs_normalizer.")})}
        if not any(key.startswith("obs_normalizer.") for key in state):
            raise ValueError("Validation requires the copied running observation normalizer")
        yield actor, identity
        if tensor_digest(actor.state_dict()) != identity["policy_sha256"]:
            raise RuntimeError("Validation modified its frozen actor/normalizer")
    finally:
        restore_rng(rng, runner.device)
        if (tensor_digest(live.state_dict()) != state_before
                or {name: module.training for name, module in live.named_modules()} != mode_before):
            raise RuntimeError("Validation modified live learner parameters, normalization or modes")


def _member(root: Path, relative: str, expected_hash: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve(strict=True)):
        raise ValueError("Validation evidence path escapes its report directory")
    if sha256_file(path) != expected_hash:
        raise ValueError("Validation evidence hash mismatch")
    return path


def validate_placed_request(row: dict, recipe: dict) -> PunchRequest:
    """Preserve recipe identity across parallel-world translation and actual t0."""
    source = dict(row["source_request"])
    source["snapshot"] = LaunchSnapshot(**source["snapshot"])
    source = PunchRequest(**source)
    actual = dict(row["request"])
    actual["snapshot"] = LaunchSnapshot(**actual["snapshot"])
    actual = PunchRequest(**actual)
    origin = np.asarray(row["scene_origin_w"], float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Finite declared parallel-world origin required")
    if source.fingerprint != recipe["request_sha256"] or source.family.value != recipe["family"]:
        raise ValueError("Source request differs from frozen validation recipe")
    qpos = np.asarray(source.snapshot.g1_qpos).copy()
    qpos[:3] += origin
    shifted = replace(source.snapshot, g1_qpos=qpos,
                      head_position_w=np.asarray(source.snapshot.head_position_w)+origin,
                      human_origin_w=np.asarray(source.snapshot.human_origin_w)+origin)
    expected = replace(source, snapshot=shifted, launch_time_s=actual.launch_time_s)
    if expected.fingerprint != actual.fingerprint:
        raise ValueError("Executed request changes more than scene origin and measured launch timestamp")
    return actual


def validate_trace_identity(arrays: dict, request: PunchRequest, expected: PanelExpectation) -> None:
    """Bind every trace to the actual evaluated actor, references and controller."""
    wanted = {name: getattr(expected, name) for name in (
        "policy_sha256", "normalization_sha256", "checkpoint_contract_sha256", "reference_manifest_sha256",
        "domain_version")}
    wanted["request_sha256"] = request.fingerprint
    for name, value in wanted.items():
        actual = np.asarray(arrays.get(name, ""))
        if actual.shape != () or str(actual.item()) != value:
            raise ValueError(f"Trace {name} differs from the validation contract")


def verify_physical_panel(report_path: Path, recipes_path: Path, expected: PanelExpectation) -> dict:
    """Re-evaluate all hashed traces; never trust claimed success in a report."""
    report_path, recipes_path = Path(report_path).resolve(strict=True), Path(recipes_path).resolve(strict=True)
    if not report_path.is_relative_to(PROJECT_ROOT) or not recipes_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Validation evidence must be confined to the human project")
    if sha256_file(recipes_path) != expected.recipe_manifest_sha256:
        raise ValueError("Validation recipe manifest changed")
    report = json.loads(report_path.read_text())
    recipes = json.loads(recipes_path.read_text())
    if (report.get("schema") != "human_wbc_native_validation_panel_v1"
            or report.get("native_execution") is not True or not report.get("slurm_job_id")
            or report.get("expectation") != asdict(expected)
            or report.get("training_environment_mutated") is not False
            or report.get("normalization_updated") is not False):
        raise ValueError("Missing bound native physical validation evidence")
    if (recipes.get("schema") != "human_wbc_validation_recipes_v1" or recipes.get("split") != "validation"
            or recipes.get("domain_version") != expected.domain_version):
        raise ValueError("Only frozen validation recipes from the declared domain may select checkpoints")
    requested = recipes.get("recipes", [])
    wanted = {row["recipe_id"]: row for row in requested}
    records = report.get("episodes", [])
    actual = {row["recipe_id"]: row for row in records}
    if len(wanted) != len(requested) or len(actual) != len(records) or set(actual) != set(wanted):
        raise ValueError("Duplicate, missing or unexpected validation episodes")
    results, metrics = [], []
    counts = {family: 0 for family in FAMILIES}
    for recipe_id, recipe in wanted.items():
        row = actual[recipe_id]
        request = validate_placed_request(row, recipe)
        path = _member(report_path.parent, row["trace_path"], row["trace_sha256"])
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        validate_trace_identity(arrays, request, expected)
        from .native_cycle_probe import deadline_prefix
        result = evaluate_episode(request, deadline_prefix(arrays, request), training_seed=expected.training_seed,
            recipe_id=recipe_id, spatial_cell_id=recipe["spatial_cell_id"], ttc_cell_id=recipe["ttc_cell_id"])
        if not result.acceptance_eligible:
            raise ValueError("Every validation recipe requires complete physical evidence and prior domain admission")
        counts[request.family.value] += 1
        results.append(asdict(result))
        # Failure-inclusive finite errors; a miss never disappears from averages.
        item = result.metrics
        metrics.append({"targeting_error": max(0., item["active_min_distance_m"]-.14),
                        "timing_error": abs(item["timing_error_s"]) if item["timing_error_s"] is not None else 4.,
                        "recovery_time": item["guard_entry_s"] if item["guard_entry_s"] is not None else request.ttc_s+4.})
    if any(value != expected.episodes_per_family for value in counts.values()):
        raise ValueError("Per-family validation sample count differs from its frozen plan")
    rates = {family: sum(r["complete_cycle_success"] for r in results if r["family"] == family)/counts[family]
             for family in FAMILIES}
    cells = {}
    for row in results:
        key = (row["family"], row["spatial_cell_id"], row["ttc_cell_id"])
        cells.setdefault(key, []).append(row["complete_cycle_success"])
    return {"schema": "human_wbc_verified_validation_panel_v1", "native_trace_evidence_verified": True,
            "report_sha256": sha256_file(report_path), "expectation": asdict(expected),
            "family_counts": counts, "family_success": rates,
            "worst_cell_success": min(sum(values)/len(values) for values in cells.values()),
            "acceptance_thresholds_all_episodes_evaluated": True,
            "worst_family_success": min(rates.values()),
            **{key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]},
            "failure_metric_policy": "miss timing penalty4s; absent recovery penalty requestedTTC+4s; all recipes retained",
            "episodes": results}


def validation_decision(panels: list[dict], stage: str) -> dict:
    if stage not in ("canonical", "domain", "guard_start") or not panels:
        raise ValueError("A declared stage and verified validation panel are required")
    if any(p.get("native_trace_evidence_verified") is not True or p.get("expectation", {}).get("stage") != stage for p in panels):
        raise ValueError("Synthetic, unverified or mixed-stage panels cannot decide training")
    identity_fields = ("stage", "training_seed", "checkpoint_contract_sha256", "reference_manifest_sha256",
                       "recipe_manifest_sha256", "domain_version", "episodes_per_family")
    identities = [{name: p["expectation"][name] for name in identity_fields} for p in panels]
    if any(digest(identity) != digest(identities[0]) for identity in identities):
        raise ValueError("Validation stopping requires one seed and unchanged recipes, domain and learner contract")
    steps = [p["expectation"]["completed_iterations"] for p in panels]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Validation panels must have strictly increasing completed iterations")
    threshold = .90 if stage == "canonical" else .95
    latest = panels[-1]
    good = lambda p: (all(p["family_success"].get(f, -1) >= threshold for f in FAMILIES)
                      and (stage == "canonical" or p["worst_cell_success"] >= .90))
    separated_three = len(steps) >= 3 and all(b-a >= INTERVAL for a, b in zip(steps[-3:], steps[-2:]))
    separated_four = len(steps) >= 4 and all(b-a >= INTERVAL for a, b in zip(steps[-4:], steps[-3:]))
    stage_ready = good(latest) if stage != "guard_start" else separated_three and all(good(p) for p in panels[-3:])
    stopped = stage_ready or (not stage_ready and separated_four and plateau(panels))
    return {"stop_learning": stopped, "reason": "validation_stage_ready" if stage_ready else
            "validation_plateau" if stopped else "continue", "stage_ready_for_review": stage_ready,
            "automatic_stage_transition": False, "selected_completed_iteration": max(panels,
                key=lambda p: (p["worst_family_success"], -p["timing_error"], -p["targeting_error"], -p["recovery_time"]))["expectation"]["completed_iterations"]}


def collective_agreement(runner, value: dict) -> dict:
    """Every rank calls at the same completed-PPO boundary; never inside save()."""
    if not runner.is_distributed:
        return value
    import torch
    gathered = [None]*runner.gpu_world_size
    torch.distributed.all_gather_object(gathered, value)
    if any(digest(item) != digest(gathered[0]) for item in gathered):
        raise RuntimeError("Ranks disagree on the validation decision")
    return gathered[0]


def run_validated_chunks(runner, *, requested_iterations: int, completed_iterations: int,
                         inherited_learn: Callable, validate_boundary: Callable,
                         checkpoint_boundary: Callable) -> dict:
    """Reuse inherited learning in bounded chunks; callback must verify real traces.

    This is orchestration only, not an environment/evaluator implementation.
    ``checkpoint_boundary`` is collective and runs AFTER each completed panel.
    No training environment reset is performed between chunks.
    """
    if validate_boundary is None or checkpoint_boundary is None:
        raise ValueError("Production requires a physical validation callback and collective checkpoint callback")
    completed, reports = completed_iterations, []
    for count in iteration_chunks(completed, requested_iterations):
        runner.current_learning_iteration = completed
        inherited_learn(num_learning_iterations=count, init_at_random_ep_len=False)
        if runner.current_learning_iteration != completed+count-1:
            raise RuntimeError("Inherited runner completed-iteration index disagrees with chunk accounting")
        completed += count
        writer = runner.logger.writer
        if writer is not None:
            if hasattr(writer, "flush"):
                writer.flush()
            if hasattr(writer, "close"):
                writer.close()
        # Also validate a partial final chunk; never finish with an unexamined actor.
        try:
            decision = validate_boundary(completed)
            if not isinstance(decision.get("stop_learning"), bool) or not decision.get("verified_panel_sha256"):
                raise ValueError("Boundary callback lacks a verified physical validation panel")
            outcome = {"ok": True, "decision": decision}
        except Exception as error:
            outcome = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        if runner.is_distributed:
            import torch
            outcomes = [None]*runner.gpu_world_size
            torch.distributed.all_gather_object(outcomes, outcome)
        else:
            outcomes = [outcome]
        if any(not item["ok"] for item in outcomes):
            raise RuntimeError("Physical validation failed on at least one rank: " +
                               "; ".join(item["error"] for item in outcomes if not item["ok"]))
        decision = collective_agreement(runner, outcome["decision"])
        checkpoint_boundary(completed, decision)
        reports.append(decision)
        if decision["stop_learning"]:
            break
    return {"completed_iterations": completed, "new_iterations": completed-completed_iterations,
            "boundaries": reports, "stopped_early": completed-completed_iterations < requested_iterations}


def require_bounded_smoke_without_validation(settings: dict, iterations: int) -> None:
    """Temporary fail-closed launch gate until the physical callback is implemented.

    A named scratch canonical smoke run is explicitly not production and cannot
    resume or chain around validation boundaries. It never advances a stage.
    """
    if (settings.get("validation_mode") != "bounded_smoke_without_validation"
            or settings.get("stage") != "canonical" or settings.get("resume")
            or isinstance(iterations, bool) or not isinstance(iterations, int)
            or not 1 <= iterations < INTERVAL):
        raise RuntimeError("Production training requires the independent physical whole-cycle validation runner; "
                           "only explicit scratch canonical smoke sessions below 250 iterations are currently supported")
