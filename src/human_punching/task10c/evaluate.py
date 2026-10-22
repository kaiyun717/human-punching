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
SELECTION_RULE = ("mean(hit + 10*hit*exp(-0.5*(signed_error_s/0.10)^2) "
                  "+ 2*settled_terminal_guard + complete_horizon_without_fall - 2*fall); "
                  "engineering retention score, not an acceptance criterion")


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
    if recipes["split"] not in ("validation", "test") or recipes["family"] != "jab":
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
    cfg = deepcopy(train_cfg)
    cfg.scene.num_envs = num_envs
    cfg.auto_reset = False
    cfg.commands["punch"].recovery_fraction = 0.0
    cfg.commands["punch"].target_split = split
    if cfg.commands["punch"].family != "jab":
        raise ValueError("Phase 2 evaluation only supports jab")
    if output is not None:
        cfg.sim.nan_guard.output_dir = str(_destination(output) / "numerical_diagnostics")
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
    score = sum(float(r["hit"]) + 10 * (math.exp(-.5 * (r["signed_timing_error_s"] / .1) ** 2)
                                       if r["hit"] else 0)
                + 2 * r["settled_terminal_guard"] + (r["complete_horizon"] and not r["fall"])
                - 2 * r["fall"] for r in complete) / n if n else None
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
        "selection_score": score, "selection_score_rule": SELECTION_RULE,
        "recovery_measure": "existing diagnostic: glove error <=.12 m, speed <=.30 m/s, upright >=.8 sustained .20 s; no exact root/joint/foot return",
        "event_measure": "first active glove-center distance <=.14 m, including launch overlap, latched at .005 s physics resolution",
    }


def _terminal_row(env, index, recipe, min_distance, disposition="completed"):
    from . import mdp
    c = env.command_manager.get_term("punch")
    error, speed = mdp.guard_errors(env)
    actual = _finite(c.state.first_encounter_elapsed[index])
    settled = float(c.settled_duration[index]) >= .2 - 1e-6
    return {
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


def evaluate_policy(env, policy, recipes: dict, output, video_count=0) -> dict:
    """Execute all frozen commands once, with zero optimizer/normalizer updates.

    ``policy`` must be a deterministic inference callable taking a TensorDict.
    The caller preserves learner mode/RNG; this function never touches a learner.
    """
    import torch
    from tensordict import TensorDict
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
                    initial = env.sim.data.qpos[:n].cpu().numpy().copy()
                    initial[:, :3] -= env.scene.env_origins[:n].cpu().numpy()
                    for i, recipe in enumerate(batch):
                        traces[recipe["recipe_id"]] = {"qpos": [initial[i]], "elapsed_s": [0.]}
                max_steps = math.ceil(float(c.horizon[:n].max()) / env.step_dt) + 2
                for _ in range(max_steps):
                    obs, reward, terminated, truncated, _ = env.step(policy(TensorDict(obs, batch_size=[env.num_envs])))
                    physical_transitions += env.num_envs
                    distance = torch.linalg.vector_norm(c.robot.data.site_pos_w[:, c.site_ids[c.active_index]]
                                                        - c.state.target_world, dim=-1)
                    minimum = torch.where(active, torch.minimum(minimum, distance), minimum)
                    if video_count:
                        selected = active.nonzero().flatten().tolist()
                        qpos = env.sim.data.qpos[:n].cpu().numpy().copy()
                        qpos[:, :3] -= env.scene.env_origins[:n].cpu().numpy()
                        for i in selected:
                            trace = traces[batch[i]["recipe_id"]]
                            trace["qpos"].append(qpos[i])
                            trace["elapsed_s"].append(float(c.state.elapsed[i]))
                    done = terminated | truncated
                    for i in (active & done).nonzero().flatten().tolist():
                        row = _terminal_row(env, i, batch[i], minimum[i])
                        rows.append(row)
                        ledger.write(json.dumps(row, allow_nan=False) + "\n")
                        ledger.flush()
                    active &= ~done
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
            np.savez_compressed(path.with_suffix(".npz"), qpos=np.asarray(trace["qpos"]),
                                elapsed_s=np.asarray(trace["elapsed_s"]), target_world=np.asarray(row["target_world"]))
            result.append({"recipe_id": recipe_id, "selection_reason": reason, "video": str(path),
                           "frames": len(trace["qpos"]), "last_elapsed_s": trace["elapsed_s"][-1],
                           "actual_evaluation_trajectory": True, "terminal_frame_included": True})
    finally:
        renderer.close()
    _write(out / "index.json", {"selection": "illustrative outcomes, not a performance sample", "videos": result})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
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
    from .env_cfg import make_env_cfg, selected_reference_path
    from .rl_cfg import runner_cfg
    if not torch.cuda.is_available() or "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("Native evaluation/rendering requires an allocated CUDA GPU")
    # Match the training/development inference backend in the standalone process.
    configure_torch_backends()
    torch.manual_seed(args.seed)
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    infos = checkpoint.get("infos", {})
    validate_checkpoint_identity(infos, sha256(selected_reference_path("jab")), sha256(args.target_pool))
    curriculum = infos.get("curriculum", {})
    ttc_range = args.ttc_range or curriculum.get("ttc_range")
    fraction = args.target_fraction if args.target_fraction is not None else curriculum.get("target_fraction", 1.)
    if ttc_range is None and args.recipes is None:
        raise ValueError("Supply --ttc-range or a Phase 2 checkpoint with its attained curriculum range")
    recipes = (json.loads(args.recipes.read_text()) if args.recipes else
               create_recipes(args.target_pool, args.split, args.attempts, args.seed, tuple(ttc_range), fraction))
    cfg = make_env_cfg(target_pool_path=args.target_pool, num_envs=args.num_envs, seed=args.seed)
    cfg.commands["punch"].reward_scales = infos.get("reward_scales", {"balance": 1., "guard": 1., "style": 1.})
    cfg.commands["punch"].guard_speed_profile = infos.get("guard_speed_profile", "gaussian")
    env = make_evaluation_env(cfg, split=recipes["split"], num_envs=args.num_envs, output=args.output)
    runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env), asdict(runner_cfg(args.seed)), None, "cuda:0")
    runner.load(str(args.checkpoint))
    try:
        result = evaluate_policy(env, runner.get_inference_policy(), recipes, args.output, args.video_count)
        result["checkpoint"] = str(args.checkpoint.resolve())
        result["checkpoint_sha256"] = sha256(args.checkpoint)
        result["checkpoint_infos"] = infos
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
