"""Recorded-recipe, guard-start evaluation for the Phase 2 jab specialist.

Development uses only original validation snapshots; final evaluation uses only
original test snapshots. Recipes choose actual cached heads, without modifying
their positions. First-encounter times come exclusively from the physics latch.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .targets import PROJECT_ROOT, TargetPool, sha256, target_subset_indices

RECIPE_SCHEMA = "task10c_jab_full_cycle_recipes_v1"
REFERENCE_EXPERIMENT_SCHEMA = "task10c_jab_reference_experiment_v1"
NOMINAL_CONTROL_SCHEMA = "task10c_jab_nominal_imitation_control_v1"
LEGACY_REFERENCE_SCHEMA = "task10c_jab_legacy_reference_v1"
LEGACY_JAB_SHA256 = "f4c710d2d6839d8b86486d141f68d307a268a999aed1ecdbe36b49713878f8f6"
CONTACT_JAB_SHA256 = "50887274783da7364d8b7ce39ee7f956c583174b0339ca602d65bfd71ff600e8"
CONTACT_JAB_METADATA_SHA256 = "96bf91e8a60f66847adfaa9f56a0319410424a522db5c53fba64b3e5ec36a74d"
LEGACY_SELECTION_RULE = ("mean(hit + 10*hit*exp(-0.5*(signed_error_s/0.10)^2) "
                  "+ 2*settled_terminal_guard + complete_horizon_without_fall - 2*fall); "
                  "engineering retention score, not an acceptance criterion")
SELECTION_RULE = ("mean(hit + 10*hit*exp(-0.5*(signed_error_s/0.10)^2) "
                  "+ 4*terminal_whole_body_score + 2*terminal_hold_support_quality "
                  "+ raw_component_totals.command_excess + complete_horizon_without_fall - 2*fall); "
                  "whole_body_score uses imitation weights .5/.5/1/1/1/1 divided by 5; "
                  "engineering retention score, not acceptance; inspect videos and actual support loads")


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def _destination(path: str | Path) -> Path:
    path = Path(path).resolve()
    path.relative_to(PROJECT_ROOT / "outputs/task10c")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _reference_path(value: str) -> Path:
    path = (PROJECT_ROOT / value).resolve()
    path.relative_to(PROJECT_ROOT.resolve())
    return path


def verify_reference_contract(contract: dict) -> dict:
    """Verify the selected files and the explicitly approved jab clock contract."""
    contract = deepcopy(contract)
    path = _reference_path(contract["reference_path"])
    metadata_path = path.with_suffix(".json")
    if (sha256(path) != contract["reference_sha256"]
            or sha256(metadata_path) != contract["reference_metadata_sha256"]):
        raise ValueError("Reference NPZ/metadata hash differs from the experiment contract")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("family") != "jab":
        raise ValueError("The Phase 2 reference contract is jab-only")
    if contract.get("schema") in (REFERENCE_EXPERIMENT_SCHEMA, NOMINAL_CONTROL_SCHEMA):
        expected = {
            "reference_sha256": CONTACT_JAB_SHA256,
            "reference_metadata_sha256": CONTACT_JAB_METADATA_SHA256,
            "clock_mode": "smooth_entry", "ttc_anchor_time": .510,
            "strike_center_time": .680, "post_entry_duration": .910,
            "easy_ttc_range": [.555, .6783333333333333], "ttc_range": [.5, .9],
            "easy_fraction": .5, "target_fraction": 1.0,
            "target_height_curriculum": False, "terminal_hold_s": .4,
        }
        if contract["schema"] == NOMINAL_CONTROL_SCHEMA:
            frame = contract.get("imitation_frame")
            if frame not in ("relative", "launch_registered"):
                raise ValueError("Nominal control requires relative or launch_registered imitation")
            expected.update(ttc_range=[.7, .7], easy_ttc_range=[.7, .7],
                            fixed_target_snapshot_id=53874, task_rewards_enabled=False,
                            imitation_frame=frame)
        if set(contract) != {"schema", "reference_path", *expected}:
            raise ValueError("Unexpected or missing reference experiment configuration fields")
        for key, value in expected.items():
            if contract.get(key) != value:
                raise ValueError(f"Reference experiment differs from the approved configuration: {key}")
        for key, value in (("nominal_first_entry_s", contract["ttc_anchor_time"]),
                           ("strike_time_s", contract["strike_center_time"]),
                           ("terminal_guard_hold_s", contract["terminal_hold_s"])):
            if not math.isclose(float(metadata[key]), value, rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"Reference metadata does not match its clock contract: {key}")
        if not math.isclose(float(metadata["end_time_s"]) - contract["ttc_anchor_time"],
                            contract["post_entry_duration"], rel_tol=0, abs_tol=1e-8):
            raise ValueError("Post-entry duration must preserve the annotated withdrawal interval")
    elif contract.get("schema") == LEGACY_REFERENCE_SCHEMA:
        if (contract["reference_sha256"] != LEGACY_JAB_SHA256
                or contract.get("clock_mode") != "legacy"
                or contract.get("ttc_anchor_time") is not None
                or contract.get("post_entry_duration") is not None
                or contract.get("easy_ttc_range") is not None):
            raise ValueError("Default jab selection must retain the pinned legacy reference and clock")
    else:
        raise ValueError("Unknown reference experiment schema")
    contract["reference_path"] = str(path.relative_to(PROJECT_ROOT.resolve()))
    return contract


def validate_reference_resume(infos: dict, contract: dict) -> None:
    """Never reinterpret an existing checkpoint under a different reference clock."""
    previous = infos.get("reference_contract")
    if previous is None:
        if contract["schema"] != LEGACY_REFERENCE_SCHEMA:
            raise ValueError("The contact-reference experiment requires a fresh lineage")
    elif verify_reference_contract(previous) != contract:
        raise ValueError("Checkpoint reference/timing/sampler contract changed")
    if (infos.get("reference_metadata_sha256") is not None
            and infos["reference_metadata_sha256"] != contract["reference_metadata_sha256"]):
        raise ValueError("Checkpoint reference metadata identity changed")


def resolve_reference_contract(experiment_config: Path | None = None,
                               checkpoint_infos: dict | None = None) -> dict:
    """Explicit experiment or embedded checkpoint contract; otherwise the old pin."""
    if experiment_config is not None:
        contract = json.loads(Path(experiment_config).read_text())
        if contract.get("schema") not in (REFERENCE_EXPERIMENT_SCHEMA, NOMINAL_CONTROL_SCHEMA):
            raise ValueError("--experiment-config requires the explicit jab reference experiment schema")
    elif checkpoint_infos and checkpoint_infos.get("reference_contract") is not None:
        contract = checkpoint_infos["reference_contract"]
    else:
        selection = json.loads((PROJECT_ROOT / "configs/task10c/references.json").read_text())
        path = _reference_path(selection["jab"])
        metadata = json.loads(path.with_suffix(".json").read_text())
        contract = {"schema": LEGACY_REFERENCE_SCHEMA,
                    "reference_path": str(path.relative_to(PROJECT_ROOT.resolve())),
                    "reference_sha256": LEGACY_JAB_SHA256,
                    "reference_metadata_sha256": sha256(path.with_suffix(".json")),
                    "clock_mode": "legacy", "ttc_anchor_time": None,
                    "strike_center_time": float(metadata["strike_time_s"]),
                    "post_entry_duration": None, "easy_ttc_range": None,
                    "ttc_range": [.5, .9], "easy_fraction": .5, "target_fraction": 1.,
                    "target_height_curriculum": False, "terminal_hold_s": .4}
    contract = verify_reference_contract(contract)
    if checkpoint_infos:
        validate_reference_resume(checkpoint_infos, contract)
    return contract


def reference_env_kwargs(contract: dict) -> dict:
    """The same resolved reference and clock feed collection and all evaluation."""
    return {"reference_path": _reference_path(contract["reference_path"]),
            "clock_mode": contract["clock_mode"], "ttc_anchor_time": contract["ttc_anchor_time"],
            "post_entry_duration": contract["post_entry_duration"],
            "easy_ttc_range": (tuple(contract["easy_ttc_range"])
                               if contract["easy_ttc_range"] is not None else None),
            "ttc_range": tuple(contract["ttc_range"]),
            "easy_fraction": contract["easy_fraction"], "target_fraction": contract["target_fraction"],
            "imitation_frame": contract.get("imitation_frame", "relative"),
            "task_rewards_enabled": contract.get("task_rewards_enabled", True),
            "fixed_target_snapshot_id": contract.get("fixed_target_snapshot_id")}


def create_nominal_control_recipes(pool_path, count: int, seed: int) -> dict:
    """Repeated full-guard nominal command, explicitly not a held-out test panel."""
    if count < 1:
        raise ValueError("Positive nominal diagnostic attempt count required")
    path = Path(pool_path).resolve()
    pool = TargetPool.load(path)
    ids = pool.snapshot_ids("train")
    matches = np.flatnonzero(ids == 53874)
    if len(matches) != 1:
        raise ValueError("Approved nominal snapshot 53874 is absent or ambiguous")
    index = int(matches[0])
    point = pool.points("train", "jab")[index].tolist()
    result = {
        "schema": RECIPE_SCHEMA, "family": "jab", "split": "train", "count": count,
        "seed": seed, "target_pool": str(path), "target_pool_sha256": sha256(path),
        "target_pool_metadata_sha256": sha256(path.with_suffix(".json")),
        "ttc_range": [.7, .7], "target_fraction": 1., "candidate_target_count": 1,
        "reset_kind": "full_guard_only", "diagnostic_fixed_snapshot_id": 53874,
        "sampling": "same real snapshot53874 and TTC0.700; repeated deterministic diagnostic, not independent trials or held-out testing",
        "rows": [{"recipe_id": i, "target_index": index, "snapshot_id": 53874,
                  "requested_ttc": .7, "target_world": point.copy()} for i in range(count)],
    }
    result["sha256"] = _digest(result)
    return result


def create_recipes(pool_path, split: str, count: int, seed: int,
                   ttc_range: tuple[float, float], target_fraction: float = 1.0) -> dict:
    """Create immutable commands, preserving original validation/test splits.

    Target IDs are shuffled without replacement in each pass through the subset;
    repeated targets in larger panels have independently stratified TTC commands.
    """
    if split not in ("validation", "test"):
        raise ValueError("Evaluation recipes require original validation or test snapshots")
    low, high = map(float, ttc_range)
    if count < 1 or not (math.isfinite(low) and math.isfinite(high) and 0.5 <= low <= high):
        raise ValueError("Positive attempt count and finite 0.5 <= TTC low <= high are required")
    path = Path(pool_path).resolve()
    pool = TargetPool.load(path)
    points = pool.points(split, "jab")
    candidates = target_subset_indices(points, target_fraction)
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.permutation(candidates)
                              for _ in range(math.ceil(count / len(candidates)))])[:count]
    times = low + (high - low) * (np.arange(count) + rng.random(count)) / count
    rng.shuffle(times)
    snapshot_ids = pool.snapshot_ids(split)
    rows = [{"recipe_id": i, "target_index": int(index),
             "snapshot_id": int(snapshot_ids[index]), "requested_ttc": float(ttc),
             "target_world": points[index].tolist()}
            for i, (index, ttc) in enumerate(zip(indices, times, strict=True))]
    result = {
        "schema": RECIPE_SCHEMA, "family": "jab", "split": split, "count": count,
        "seed": seed, "target_pool": str(path), "target_pool_sha256": sha256(path),
        "target_pool_metadata_sha256": sha256(path.with_suffix(".json")),
        "ttc_range": [low, high], "target_fraction": float(target_fraction),
        "candidate_target_count": len(candidates), "reset_kind": "full_guard_only",
        "sampling": "central XY rank subset, original head XYZ; shuffled IDs and stratified TTC",
        "rows": rows,
    }
    result["sha256"] = _digest(result)
    return result


def save_recipes(path: str | Path, recipes: dict) -> None:
    """Never replace an existing recipe panel with a different command set."""
    path = Path(path).resolve()
    _destination(path.parent)
    if path.exists():
        if json.loads(path.read_text()) != recipes:
            raise ValueError(f"Refusing to replace a different frozen recipe panel: {path}")
    else:
        _write(path, recipes)


def validate_recipes(recipes: dict, pool_path=None) -> None:
    payload = {k: v for k, v in recipes.items() if k != "sha256"}
    if recipes.get("schema") != RECIPE_SCHEMA or recipes.get("sha256") != _digest(payload):
        raise ValueError("Recipe schema/digest mismatch")
    nominal = recipes.get("diagnostic_fixed_snapshot_id") == 53874
    if nominal and (recipes["split"] != "train" or recipes["ttc_range"] != [.7, .7]
                    or recipes["candidate_target_count"] != 1):
        raise ValueError("Nominal diagnostic must use the approved single training command")
    if (recipes["split"] not in (("train",) if nominal else ("validation", "test"))
            or recipes["family"] != "jab"):
        raise ValueError("Phase 2 evaluation is jab-only on validation/test snapshots")
    path = Path(pool_path or recipes["target_pool"]).resolve()
    if sha256(path) != recipes["target_pool_sha256"] or sha256(path.with_suffix(".json")) != recipes["target_pool_metadata_sha256"]:
        raise ValueError("Frozen target-pool identity changed")
    pool = TargetPool.load(path)
    points, ids = pool.points(recipes["split"], "jab"), pool.snapshot_ids(recipes["split"])
    eligible = set(target_subset_indices(points, recipes["target_fraction"]).tolist())
    if len(recipes["rows"]) != recipes["count"]:
        raise ValueError("Recipe row count differs from declared attempt count")
    recipe_ids = set()
    for row in recipes["rows"]:
        index = row["target_index"]
        if (index not in eligible or int(ids[index]) != row["snapshot_id"]
                or (nominal and row["snapshot_id"] != 53874)
                or not np.allclose(points[index], row["target_world"], rtol=0, atol=1e-12)
                or not recipes["ttc_range"][0] <= row["requested_ttc"] <= recipes["ttc_range"][1]
                or row["recipe_id"] in recipe_ids):
            raise ValueError("Recipe target, timing, split or unique ID mismatch")
        recipe_ids.add(row["recipe_id"])


def validate_checkpoint_identity(infos: dict, reference_hash: str, target_hash: str) -> None:
    """Reject setup/other-family checkpoints before any evaluation state loads."""
    if infos.get("phase") != "task10c_phase2_jab" or infos.get("family") != "jab":
        raise ValueError("Evaluation requires a Phase 2 jab checkpoint, never setup or another family")
    if not infos.get("lineage_id"):
        raise ValueError("Phase 2 checkpoint is missing its fresh-jab lineage identity")
    if infos.get("reference_sha256") != reference_hash:
        raise ValueError("Checkpoint reference differs from the selected jab reference")
    if infos.get("target_pool_sha256") != target_hash:
        raise ValueError("Checkpoint target pool differs from the requested pool")


def make_evaluation_env(train_cfg, split="validation", num_envs=64, output=None):
    """Build one reusable small environment, independent of PPO collection."""
    from .environment import PunchEnv
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from .whole_body_metrics import solver_observer
    cfg = deepcopy(train_cfg)
    cfg.scene.num_envs = num_envs
    cfg.auto_reset = False
    cfg.commands["punch"].recovery_fraction = 0.0
    cfg.commands["punch"].target_split = split
    if cfg.commands["punch"].family != "jab":
        raise ValueError("Phase 2 evaluation only supports jab")
    if output is not None:
        cfg.sim.nan_guard.output_dir = str(_destination(output) / "numerical_diagnostics")
    # Appending preserves the encounter callback's earlier timer update. Forces
    # must be captured before mjlab's final forward refresh, not after step().
    cfg.metrics["whole_body_solver_observer"] = MetricsTermCfg(
        func=solver_observer, per_substep=True, reduce="last")
    return PunchEnv(cfg, device="cuda:0")


def _finite(value):
    result = float(value)
    return result if math.isfinite(result) else None


def _distribution(values) -> dict:
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None,
                "p05": None, "p95": None}
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max()),
            "p05": float(np.quantile(values, .05)), "p95": float(np.quantile(values, .95))}


def summarize_attempts(rows: list[dict], recipes: dict) -> dict:
    """Keep explicit denominators; misses never receive an invented TTC/error."""
    attempted = [r for r in rows if r["attempted"]]
    complete = [r for r in attempted if r["disposition"] == "completed"]
    hits = [r for r in complete if r["hit"]]
    misses = [r for r in complete if not r["hit"]]
    n = len(complete)
    whole_body_rows = [r for r in complete if r.get("reward_mode") == "whole_body"]
    def retention_score(row):
        recovery_score = (4 * row["terminal_whole_body_score"] + 2 * row["terminal_hold_support_quality"]
                          + row["raw_component_totals"].get("command_excess", 0.)) if row.get("reward_mode") == "whole_body" else 2 * row["settled_terminal_guard"]
        return (float(row["hit"]) + 10 * (math.exp(-.5 * (row["signed_timing_error_s"] / .1) ** 2)
                                          if row["hit"] else 0.) + recovery_score
                + (row["complete_horizon"] and not row["fall"]) - 2 * row["fall"])
    score = sum(retention_score(r) for r in complete) / n if n else None
    names = sorted({key for row in complete for key in row["raw_component_totals"]})
    targets = np.asarray([r["target_world"] for r in recipes["rows"]])
    recovery = {}
    for name, subset in (("all", complete), ("hits", hits), ("misses", misses)):
        recovery[name] = {"denominator": len(subset),
            "ever_settled": sum(r["settled_guard_elapsed_s"] is not None for r in subset),
            "settled_at_terminal": sum(r["settled_terminal_guard"] for r in subset),
            "complete_horizon_without_fall": sum(r["complete_horizon"] and not r["fall"] for r in subset),
            "falls": sum(r["fall"] for r in subset),
            "terminal_guard_error_m": _distribution(r["guard_glove_error_m"] for r in subset),
            "terminal_guard_speed_m_s": _distribution(r["guard_glove_speed_m_s"] for r in subset),
            "first_sustained_settling_elapsed_s": _distribution(r["settled_guard_elapsed_s"] for r in subset)}
    return {
        "requested_attempts": recipes["count"], "attempted": len(attempted), "completed": n,
        "unattempted": recipes["count"] - len(attempted), "incomplete_or_failed": len(attempted) - n,
        "hit_count": len(hits), "miss_count": len(misses), "hit_denominator": n,
        "hit_rate": len(hits) / n if n else None,
        "fall_count": sum(r["fall"] for r in complete), "fall_denominator": n,
        "completed_horizon_count": sum(r["complete_horizon"] for r in complete),
        "incidental_other_glove_encounters": sum(r["inactive_encounter"] for r in complete),
        "launch_overlap_count": sum(r["launch_overlap"] for r in complete),
        "early_encounters_before_requested": sum(r["actual_ttc_s"] < r["requested_ttc_s"] for r in hits),
        "early_encounters_below_0_5_s": sum(r["actual_ttc_s"] < .5 for r in hits),
        "early_encounter_denominator": len(hits),
        "timing_within_0_10_s_count": sum(abs(r["signed_timing_error_s"]) <= .1 for r in hits),
        "timing_error_denominator": len(hits),
        "actual_first_encounter_ttc_s": _distribution(r["actual_ttc_s"] for r in hits),
        "signed_timing_error_s": _distribution(r["signed_timing_error_s"] for r in hits),
        "absolute_timing_error_s": _distribution(abs(r["signed_timing_error_s"]) for r in hits),
        "requested_ttc_s": _distribution(r["requested_ttc"] for r in recipes["rows"]),
        "target_world_min": targets.min(0).tolist(), "target_world_max": targets.max(0).tolist(),
        "unique_snapshot_count": len({r["snapshot_id"] for r in recipes["rows"]}),
        "recovery": recovery,
        "raw_component_totals": {key: _distribution(r["raw_component_totals"].get(key, 0.) for r in complete)
                                 for key in names},
        "selection_score": score, "development_score": score,
        "selection_score_rule": SELECTION_RULE if whole_body_rows else LEGACY_SELECTION_RULE,
        "whole_body": {
            "denominator": len([r for r in complete if "terminal_whole_body_score" in r]),
            "terminal_tracking_score": _distribution(r.get("terminal_whole_body_score") for r in complete),
            "terminal_hold_support_quality": _distribution(r.get("terminal_hold_support_quality") for r in complete),
            "maximum_command_excess_rad": _distribution(r.get("maximum_command_excess_rad") for r in complete),
            "terminal_hold_foot_vertical_force_mean_N": {
                side: _distribution(r["terminal_hold_foot_force_world_mean_N"][index][2] for r in complete
                                    if r.get("terminal_hold_foot_force_world_mean_N") is not None)
                for index, side in enumerate(("left", "right"))},
            "terminal_hold_foot_positive_vertical_load_duration_s": {
                side: _distribution(r["terminal_hold_foot_positive_vertical_load_duration_s"][index] for r in complete
                                    if "terminal_hold_foot_positive_vertical_load_duration_s" in r)
                for index, side in enumerate(("left", "right"))},
            "terminal_body_tilt_degrees": {name: _distribution(r.get("terminal_body_tilt_degrees", {}).get(name) for r in complete)
                                           for name in sorted({name for r in complete for name in r.get("terminal_body_tilt_degrees", {})})},
        },
        "recovery_measure": "legacy hand/pelvis diagnostic only: glove error <=.12 m, speed <=.30 m/s, pelvis upright >=.8 sustained .20 s; whole-body tracking and measured native foot loads reported separately",
        "event_measure": "first active glove-center distance <=.14 m, including launch overlap, latched at .005 s physics resolution",
    }


def _terminal_row(env, index, recipe, min_distance, disposition="completed"):
    from . import mdp
    c = env.command_manager.get_term("punch")
    error, speed = mdp.guard_errors(env)
    actual = _finite(c.state.first_encounter_elapsed[index])
    settled = float(c.settled_duration[index]) >= .2 - 1e-6
    row = {
        "recipe_id": recipe["recipe_id"], "attempted": True, "disposition": disposition,
        "family": "jab", "reset_kind": "full", "snapshot_id": recipe["snapshot_id"],
        "target_world": recipe["target_world"], "requested_ttc_s": float(c.state.requested_ttc[index]),
        "elapsed_start_s": float(c.state.start_elapsed[index]), "elapsed_end_s": float(c.state.elapsed[index]),
        "horizon_s": float(c.horizon[index]), "actor_steps": int(c.steps_seen[index]),
        "source_phase_end_s": float(c.source_time[index]), "hit": bool(c.state.encountered[index]),
        "actual_ttc_s": actual, "signed_timing_error_s": None if actual is None else actual - float(c.state.requested_ttc[index]),
        "launch_local_sim_time_s": float(c.state.launch_time[index]),
        "first_encounter_local_sim_time_s": None if actual is None else actual + float(c.state.launch_time[index]),
        "launch_overlap": bool(c.state.launch_overlap[index]),
        "inactive_encounter": bool(c.state.inactive_encountered[index]),
        "inactive_first_encounter_ttc_s": _finite(c.state.first_inactive_elapsed[index]),
        "fall": bool(mdp.fallen(env)[index]),
        "complete_horizon": bool(c.state.elapsed[index] >= c.horizon[index] - 1e-6),
        "guard_glove_error_m": float(error[index]), "guard_glove_speed_m_s": float(speed[index]),
        "settled_guard_elapsed_s": _finite(c.first_settled_elapsed[index]),
        "settled_terminal_guard": bool(settled and not mdp.fallen(env)[index]),
        "terminal_upright_cosine": float(-c.robot.data.projected_gravity_b[index, 2]),
        "terminal_root_height_m": float(c.robot.data.root_link_pos_w[index, 2]),
        "minimum_actor_sample_active_glove_distance_m": float(min_distance),
        "minimum_distance_note": "20 ms actor samples, diagnostic only; does not define encounter or TTC",
        "raw_component_totals": {name: float(value[index]) for name, value in c.component_totals.items()},
    }
    metrics = getattr(env, "whole_body_metrics", None)
    if metrics is not None:
        row.update(metrics.terminal(index))
    return row


def evaluate_policy(env, policy, recipes: dict, output, video_count=0) -> dict:
    """Execute all frozen commands once, with zero optimizer/normalizer updates.

    ``policy`` must be a deterministic inference callable taking a TensorDict.
    The caller preserves learner mode/RNG; this function never touches a learner.
    """
    import torch
    from tensordict import TensorDict
    from .whole_body_metrics import WholeBodyMetrics
    c = env.command_manager.get_term("punch")
    if env.cfg.auto_reset or c.cfg.target_split != recipes["split"] or c.cfg.recovery_fraction != 0:
        raise ValueError("Evaluation requires manual-reset, matching split, 100% guard-start environment")
    validate_recipes(recipes, c.cfg.target_pool_path)
    if not isinstance(policy, torch.nn.Module) or policy.training:
        raise ValueError("Evaluation requires the learner's frozen inference policy module")
    before_policy = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    out = _destination(output)
    save_recipes(out / "recipes.json", recipes)
    if (out / "attempts.jsonl").exists():
        raise ValueError(f"Evaluation attempt ledger already exists: {out}")
    rows, traces, started_ids = [], {}, set()
    started = time.monotonic()
    failure = None
    physical_transitions = 0
    metrics = WholeBodyMetrics(env)
    env.whole_body_metrics = metrics
    try:
        with (out / "attempts.jsonl").open("x") as ledger, torch.inference_mode():
            for offset in range(0, recipes["count"], env.num_envs):
                batch = recipes["rows"][offset:offset + env.num_envs]
                c.set_evaluation_recipes({"rows": batch})
                obs, _ = env.reset()
                # This dedicated environment's canonical evidence is the compact
                # evaluation ledger below, not duplicate auto-reset diagnostics.
                c.episode_records.clear()
                started_ids.update(row["recipe_id"] for row in batch)
                n = len(batch)
                metrics.begin_batch(n)
                target = c.state.target_world[:n] - env.scene.env_origins[:n]
                if (not np.allclose(target.cpu().numpy(), [r["target_world"] for r in batch], atol=2e-5, rtol=0)
                        or not bool(c.state.history_known[:n].all())
                        or bool((c.state.elapsed[:n] != 0).any())
                        or not np.allclose(c.state.requested_ttc[:n].cpu().numpy(),
                                           [r["requested_ttc"] for r in batch], atol=1e-7, rtol=0)):
                    raise AssertionError("Frozen recipe was not installed at the actual command launch")
                active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                active[:n] = True
                minimum = torch.linalg.vector_norm(c.robot.data.site_pos_w[:, c.site_ids[c.active_index]]
                                                   - c.state.target_world, dim=-1)
                if video_count:
                    initial = metrics.trace_arrays(n)
                    for i, recipe in enumerate(batch):
                        traces[recipe["recipe_id"]] = {name: [value[i]] for name, value in initial.items()}
                max_steps = math.ceil(float(c.horizon[:n].max()) / env.step_dt) + 2
                for _ in range(max_steps):
                    obs, reward, terminated, truncated, _ = env.step(policy(TensorDict(obs, batch_size=[env.num_envs])))
                    physical_transitions += env.num_envs
                    metrics.capture_actor()
                    distance = torch.linalg.vector_norm(c.robot.data.site_pos_w[:, c.site_ids[c.active_index]]
                                                        - c.state.target_world, dim=-1)
                    minimum = torch.where(active, torch.minimum(minimum, distance), minimum)
                    if video_count:
                        selected = active.nonzero().flatten().tolist()
                        frame = metrics.trace_arrays(n)
                        for i in selected:
                            trace = traces[batch[i]["recipe_id"]]
                            for name, value in frame.items():
                                trace[name].append(value[i])
                    done = terminated | truncated
                    for i in (active & done).nonzero().flatten().tolist():
                        row = _terminal_row(env, i, batch[i], minimum[i])
                        rows.append(row)
                        ledger.write(json.dumps(row, allow_nan=False) + "\n")
                        ledger.flush()
                    active &= ~done
                    metrics.active.copy_(active)
                    if not bool(active.any()):
                        break
                    # mjlab requires all terminal environments reset before the
                    # next vector step. These additional cycles are ignored.
                    if bool(done.any()):
                        obs, _ = env.reset(env_ids=done.nonzero().flatten())
                        c.episode_records.clear()
                if bool(active.any()):
                    for i in active.nonzero().flatten().tolist():
                        row = _terminal_row(env, i, batch[i], minimum[i], "step_budget_incomplete")
                        rows.append(row)
                        ledger.write(json.dumps(row, allow_nan=False) + "\n")
                    raise RuntimeError("Full attempt exceeded its recorded horizon without termination")
    except Exception as error:
        failure = error
    finally:
        metrics.active.zero_()
    completed_ids = {r["recipe_id"] for r in rows}
    for recipe in recipes["rows"]:
        if recipe["recipe_id"] not in completed_ids:
            attempted = recipe["recipe_id"] in started_ids
            rows.append({"recipe_id": recipe["recipe_id"], "attempted": attempted,
                         "disposition": "failed_before_terminal_record" if attempted else "unattempted",
                         "snapshot_id": recipe["snapshot_id"], "requested_ttc_s": recipe["requested_ttc"],
                         "target_world": recipe["target_world"], "failure": repr(failure)})
    rows.sort(key=lambda row: row["recipe_id"])
    _write(out / "attempts.json", rows)
    after_policy = policy.state_dict()
    policy_unchanged = (before_policy.keys() == after_policy.keys()
                        and all(torch.equal(value, after_policy[name]) for name, value in before_policy.items()))
    if not policy_unchanged and failure is None:
        failure = RuntimeError("Actor or normalizer state changed during deterministic evaluation")
    summary = summarize_attempts(rows, recipes)
    summary.update({"status": "failed" if failure else "completed", "split": recipes["split"],
                    "recipe_sha256": recipes["sha256"], "elapsed_seconds": time.monotonic() - started,
                    "physical_transitions_including_ignored_restarts": physical_transitions,
                    "counted_full_attempt_transitions": sum(r.get("actor_steps", 0) for r in rows),
                    "failure": repr(failure) if failure else None, "optimizer_updates": 0,
                    "normalizer_updates": 0 if policy_unchanged else None,
                    "actor_and_normalizer_state_unchanged": policy_unchanged,
                    "policy_state_tensor_count_checked": len(before_policy), "target_is_non_solid": True})
    summary["whole_body_telemetry"] = metrics.metadata()
    _write(out / "summary.json", summary)
    if failure:
        raise failure
    if video_count:
        summary["videos"] = render_representatives(env, rows, traces, out / "videos", video_count)
        summary["elapsed_seconds"] = time.monotonic() - started
        _write(out / "summary.json", summary)
    return summary


def select_representatives(rows: list[dict], count: int) -> list[tuple[dict, str]]:
    """Show the first recipe and outcome examples; never call this a sample rate."""
    complete = [r for r in rows if r["disposition"] == "completed"]
    survivors = [r for r in complete if r["hit"] and r["complete_horizon"] and not r["fall"]]
    recovered = [r for r in survivors if r["settled_terminal_guard"]]
    recovery_examples = sorted(recovered or survivors, key=lambda r: r["guard_glove_error_m"])
    early = [r for r in complete if r["hit"] and r["actual_ttc_s"] < .5]
    groups = [([complete[0]] if complete else [], "first_recipe_unselected_by_outcome"),
              (recovery_examples, "lowest_guard_error_settled_hit" if recovered else "lowest_guard_error_hit_with_full_horizon_without_fall"),
              (sorted((r for r in complete if r["hit"]), key=lambda r: abs(r["signed_timing_error_s"])), "smallest_timing_error_hit"),
              ([r for r in complete if r["fall"]], "first_fall"),
              ([r for r in complete if not r["hit"]], "first_miss"),
              (early or sorted((r for r in complete if r["hit"]), key=lambda r: -abs(r["signed_timing_error_s"])),
               "first_encounter_below_0_5_s" if early else "largest_timing_error_hit")]
    if any(r.get("reward_mode") == "whole_body" for r in complete):
        groups = [([complete[0]] if complete else [], "first_recipe_unselected_by_outcome"),
                  (sorted(survivors, key=lambda r: -r["terminal_whole_body_score"]), "highest_whole_body_score_full_horizon_hit"),
                  (sorted(complete, key=lambda r: r["terminal_hold_support_quality"]), "lowest_terminal_hold_support_quality"),
                  (sorted(complete, key=lambda r: -r["maximum_command_excess_rad"]), "largest_commanded_target_excess"),
                  ([r for r in complete if r["fall"]], "first_fall"),
                  ([r for r in complete if not r["hit"]], "first_miss"),
                  (sorted((r for r in complete if r["hit"]), key=lambda r: abs(r["signed_timing_error_s"])), "smallest_timing_error_hit"),
                  (complete, "next_recipe_unselected_by_outcome")]
    selected, used = [], set()
    for group, reason in groups:
        candidate = next((r for r in group if r["recipe_id"] not in used), None)
        if candidate is not None:
            selected.append((candidate, reason))
            used.add(candidate["recipe_id"])
        if len(selected) >= count:
            break
    return selected


def render_representatives(env, rows, traces, output, count=4):
    """Render the actual evaluated trajectories from two views at native 50 Hz."""
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    out = _destination(output)
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    cameras = []
    for azimuth in (135, 90):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.1, 0, .9]
        camera.distance, camera.azimuth, camera.elevation = 3.2, azimuth, -12
        cameras.append(camera)
    result = []
    try:
        for row, reason in select_representatives(rows, count):
            recipe_id = row["recipe_id"]
            trace = traces[recipe_id]
            path = out / f"recipe_{recipe_id:04d}.mp4"
            with imageio.get_writer(path, fps=50, macro_block_size=16) as writer:
                for qpos, elapsed in zip(trace["qpos"], trace["elapsed_s"], strict=True):
                    data.qpos[:] = qpos
                    mujoco.mj_forward(model, data)
                    frames = []
                    for camera in cameras:
                        renderer.update_scene(data, camera)
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.full(3, .06), np.asarray(row["target_world"]), np.eye(3).ravel(),
                            np.array([1., .2, .2, .4], dtype=np.float32))
                        renderer.scene.ngeom += 1
                        frames.append(renderer.render().copy())
                    frame = Image.fromarray(np.concatenate(frames, axis=1))
                    actual = row["actual_ttc_s"]
                    event = "none yet" if actual is None or elapsed < actual else f"{actual:.3f}s"
                    label = (f"Learned JAB | recipe {recipe_id} | elapsed {elapsed:.2f}s | "
                             f"requested {row['requested_ttc_s']:.3f}s | first encounter {event}\n"
                             f"Frozen NON-SOLID head | {reason} | {'FALL' if row['fall'] else 'full horizon'}")
                    ImageDraw.Draw(frame).text((10, 10), label, fill="white", stroke_width=1, stroke_fill="black")
                    writer.append_data(np.asarray(frame))
            arrays = {name: np.asarray(value) for name, value in trace.items()}
            arrays['target_world'] = np.asarray(row["target_world"])
            if getattr(env, "whole_body_metrics", None) is not None:
                metadata = env.whole_body_metrics.metadata()
                arrays['joint_names'] = np.asarray(metadata['joint_names'])
                arrays['body_names'] = np.asarray(metadata['body_names'])
                arrays['hard_joint_limits_rad'] = np.asarray(metadata['hard_joint_limits_rad'])
            np.savez_compressed(path.with_suffix(".npz"), **arrays)
            result.append({"recipe_id": recipe_id, "selection_reason": reason, "video": str(path),
                           "frames": len(trace["qpos"]), "last_elapsed_s": float(trace["elapsed_s"][-1]),
                           "actual_evaluation_trajectory": True, "terminal_frame_included": True})
    finally:
        renderer.close()
    _write(out / "index.json", {"selection": "illustrative outcomes, not a performance sample", "videos": result})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path,
                        help="Optional explicit contract check; normally restored from checkpoint metadata")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--attempts", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=103222)
    parser.add_argument("--target-pool", type=Path, default=PROJECT_ROOT / "outputs/task10c/setup/target_pool.npz")
    parser.add_argument("--ttc-range", type=float, nargs=2)
    parser.add_argument("--target-fraction", type=float)
    parser.add_argument("--recipes", type=Path)
    parser.add_argument("--video-count", type=int, default=6)
    args = parser.parse_args()
    import torch
    import warp as wp
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from .env_cfg import make_env_cfg
    from .rl_cfg import runner_cfg
    if not torch.cuda.is_available() or "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Native evaluation/rendering requires an allocated CUDA GPU")
    # Match the training/development inference backend in the standalone process.
    configure_torch_backends()
    torch.manual_seed(args.seed)
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    infos = checkpoint.get("infos", {})
    reference_contract = resolve_reference_contract(args.experiment_config, infos)
    validate_checkpoint_identity(infos, reference_contract["reference_sha256"], sha256(args.target_pool))
    curriculum = infos.get("curriculum", {})
    ttc_range = args.ttc_range or curriculum.get("ttc_range")
    fraction = args.target_fraction if args.target_fraction is not None else curriculum.get("target_fraction", 1.)
    if ttc_range is None and args.recipes is None:
        raise ValueError("Supply --ttc-range or a Phase 2 checkpoint with its attained curriculum range")
    nominal_control = reference_contract["schema"] == NOMINAL_CONTROL_SCHEMA
    recipes = (json.loads(args.recipes.read_text()) if args.recipes else
               create_nominal_control_recipes(args.target_pool, args.attempts, args.seed) if nominal_control else
               create_recipes(args.target_pool, args.split, args.attempts, args.seed, tuple(ttc_range), fraction))
    if nominal_control and recipes.get("diagnostic_fixed_snapshot_id") != 53874:
        raise ValueError("Nominal control evaluation must retain the fixed command")
    reference_kwargs = reference_env_kwargs(reference_contract)
    for key in ("ttc_range", "target_fraction", "easy_fraction"):
        if key in curriculum:
            reference_kwargs[key] = curriculum[key]
    cfg = make_env_cfg(target_pool_path=args.target_pool, num_envs=args.num_envs, seed=args.seed,
                       reward_mode=infos.get("reward_mode", "legacy"),
                       imitation_budget=infos.get("imitation_budget", 7.),
                       recovery_budget=infos.get("recovery_budget", 6.),
                       command_excess_weight=infos.get("command_excess_weight", 1.),
                       **reference_kwargs)
    cfg.commands["punch"].reward_scales = infos.get("reward_scales", {"balance": 1., "guard": 1., "style": 1.})
    cfg.commands["punch"].guard_speed_profile = infos.get("guard_speed_profile", "gaussian")
    env = make_evaluation_env(cfg, split=recipes["split"], num_envs=args.num_envs, output=args.output)
    agent_cfg = deepcopy(infos.get("agent_config", asdict(runner_cfg(args.seed))))
    runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env), agent_cfg, None, "cuda:0")
    runner.load(str(args.checkpoint))
    try:
        result = evaluate_policy(env, runner.get_inference_policy(), recipes, args.output, args.video_count)
        result["checkpoint"] = str(args.checkpoint.resolve())
        result["checkpoint_sha256"] = sha256(args.checkpoint)
        result["checkpoint_infos"] = infos
        result["reference_contract"] = reference_contract
        result["reward_scales"] = dict(cfg.commands["punch"].reward_scales)
        result["guard_speed_profile"] = cfg.commands["punch"].guard_speed_profile
        result["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
        result["command"] = os.sys.argv
        result["job_id"] = os.environ.get("SLURM_JOB_ID")
        _write(Path(args.output).resolve() / "summary.json", result)
        print(json.dumps({key: result[key] for key in (
            "status", "split", "attempted", "completed", "hit_count", "miss_count", "fall_count",
            "signed_timing_error_s", "actor_and_normalizer_state_unchanged", "checkpoint_sha256")}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
